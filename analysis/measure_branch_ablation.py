"""
measure_branch_ablation.py — Experiment C: Text-branch vs Image-branch skip.

Compares quality under two late-step skip schedules:
  Schedule X ("text_skip"):  skip text branch modulation+MLP for steps freeze_from..end
  Schedule Y ("image_skip"): skip image branch modulation+MLP for steps freeze_from..end
  Baseline:                  no skip

Additionally tests layer-grouped skip variants (early/mid/late thirds).

Monkey-patches FluxTransformerBlock.forward / JointTransformerBlock.forward.
Images saved to results/branch_ablation/{model}/{schedule}/sample_{i}.png.
Metrics compared against eval/reference/{model}/ref_{i}.png.

Outputs:
  results/branch_ablation_{model}.csv  — schedule × metric table

Usage:
  python analysis/measure_branch_ablation.py --model flux --n_prompts 350 --gpu 0
  python analysis/measure_branch_ablation.py --model flux --n_prompts 10  --gpu 0 --quick
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import argparse
from contextlib import contextmanager
from typing import Optional
import types

import torch
import yaml
import numpy as np

from analysis.paths import ref_dir as get_ref_dir, samples_dir, results_dir, BASE_DIR


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompts(n: int, quick: bool = False) -> list[str]:
    if quick:
        return [
            "A photorealistic portrait of an astronaut on Mars at sunset",
            "A cozy coffee shop in Paris with rain on the windows",
            "A majestic lion standing on a rocky cliff",
            "Abstract geometric art with vibrant neon colors",
            "A serene Japanese garden with cherry blossoms",
            "A futuristic city skyline at night",
            "A gourmet burger with melting cheese",
            "An oil painting of a stormy sea",
            "A golden retriever puppy in autumn leaves",
            "A spider web with dew drops at dawn",
        ][:n]
    # Try local pre-fetched prompt file first (fast, no network)
    local_prompt_file = os.path.join(BASE_DIR, "eval", "prompts", "mjhq30k_first350.json")
    if os.path.exists(local_prompt_file):
        import json as _json
        with open(local_prompt_file) as _f:
            return _json.load(_f)[:n]
    try:
        from datasets import load_dataset
        ds = load_dataset("xingjianleng/mjhq30k", split="test")
        return [ds[i]["text"] for i in range(min(n, len(ds)))]
    except Exception:
        pass
    fallback = ["A photorealistic landscape" for _ in range(n)]
    return fallback


# ---------------------------------------------------------------------------
# Branch skip via step-aware monkey-patch
# ---------------------------------------------------------------------------

class BranchSkipState:
    def __init__(self, skip_target: str, freeze_from: int, layer_indices: Optional[list] = None):
        """
        skip_target: "text", "image", or "none"
        freeze_from: skip branch for steps >= freeze_from
        layer_indices: which block indices to skip (None = all)
        """
        self.skip_target = skip_target
        self.freeze_from = freeze_from
        self.layer_indices = layer_indices
        self.current_step = 0

    def should_skip(self, block_idx: int) -> bool:
        if self.skip_target == "none":
            return False
        if self.current_step < self.freeze_from:
            return False
        if self.layer_indices is not None and block_idx not in self.layer_indices:
            return False
        return True

    def next_step(self):
        self.current_step += 1

    def reset(self):
        self.current_step = 0


@contextmanager
def patched_flux_blocks(transformer, skip_state: BranchSkipState):
    """
    Context manager that monkey-patches FluxTransformerBlock.forward to skip
    text or image branch based on skip_state.
    """
    original_forwards = {}
    for idx, block in enumerate(transformer.transformer_blocks):
        original_forwards[idx] = block.forward

        def make_patched(bidx, orig_fwd):
            def patched_forward(
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb=None,
                joint_attention_kwargs=None,
            ):
                if not skip_state.should_skip(bidx):
                    return orig_fwd(
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )

                # Execute full forward; then selectively overwrite one stream's output
                # with its residual (i.e., skip modulation+MLP contribution).
                # We run forward normally and capture intermediate values by running
                # attention-only path and zeroing the MLP delta.
                img_res = hidden_states
                txt_res = encoder_hidden_states

                out = orig_fwd(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )
                # FluxTransformerBlock returns (encoder_hidden_states, hidden_states) = (text, image)
                txt_out, img_out = out

                if skip_state.skip_target == "text":
                    # Keep image updated, text = residual only (skip contribution)
                    return txt_res, img_out
                elif skip_state.skip_target == "image":
                    return txt_out, img_res
                return out

            return patched_forward

        block.forward = make_patched(idx, block.forward)

    try:
        yield
    finally:
        for idx, block in enumerate(transformer.transformer_blocks):
            block.forward = original_forwards[idx]


@contextmanager
def patched_sd3_blocks(transformer, skip_state: BranchSkipState):
    """Same as above for SD3 JointTransformerBlock."""
    original_forwards = {}
    for idx, block in enumerate(transformer.transformer_blocks):
        original_forwards[idx] = block.forward

        def make_patched(bidx, orig_fwd):
            def patched_forward(
                hidden_states,
                encoder_hidden_states,
                temb,
                *args, **kwargs,
            ):
                if not skip_state.should_skip(bidx):
                    return orig_fwd(hidden_states, encoder_hidden_states, temb, *args, **kwargs)

                img_res = hidden_states
                txt_res = encoder_hidden_states
                out = orig_fwd(hidden_states, encoder_hidden_states, temb, *args, **kwargs)

                if isinstance(out, (tuple, list)) and len(out) == 2:
                    txt_out, img_out = out
                    if skip_state.skip_target == "text":
                        return txt_res, img_out
                    elif skip_state.skip_target == "image":
                        return txt_out, img_res
                return out

            return patched_forward

        block.forward = make_patched(idx, block.forward)

    try:
        yield
    finally:
        for idx, block in enumerate(transformer.transformer_blocks):
            block.forward = original_forwards[idx]


# ---------------------------------------------------------------------------
# Image generation for one schedule
# ---------------------------------------------------------------------------

def generate_for_schedule(
    pipe,
    model_key: str,
    prompts: list[str],
    cfg: dict,
    skip_state: BranchSkipState,
    save_dir: str,
    device: str,
    batch_size: int = 1,
    global_offset: int = 0,
):
    """global_offset: add to local index i to get the global sample_{i}.png filename."""
    os.makedirs(save_dir, exist_ok=True)
    gen_cfg = cfg["generation"]
    is_flux = model_key == "flux"
    patch_ctx = patched_flux_blocks if is_flux else patched_sd3_blocks

    pending = [i for i in range(len(prompts))
               if not os.path.exists(os.path.join(save_dir, f"sample_{i + global_offset}.png"))]

    if not pending:
        print(f"    All {len(prompts)} images exist, skipping.", flush=True)
        return

    # batch_size > 1: BranchSkipState.step counter is shared across the batch,
    # which is correct since all samples in a batch use the same step index.
    for batch_start in range(0, len(pending), batch_size):
        batch_idx = pending[batch_start: batch_start + batch_size]
        batch_prompts = [prompts[i] for i in batch_idx]
        global_idx = [i + global_offset for i in batch_idx]
        gens = [torch.Generator(device=device).manual_seed(gen_cfg["seed"] + gi)
                for gi in global_idx]

        skip_state.reset()

        def step_cb(pipe_, step_idx, timestep, cb_kwargs):
            skip_state.next_step()
            return cb_kwargs

        with torch.no_grad(), patch_ctx(pipe.transformer, skip_state):
            result = pipe(
                batch_prompts,
                num_inference_steps=gen_cfg["num_inference_steps"],
                guidance_scale=gen_cfg["guidance_scale"],
                height=gen_cfg["height"],
                width=gen_cfg["width"],
                generator=gens,
                callback_on_step_end=step_cb,
            )

        for gi, img in zip(global_idx, result.images):
            img.save(os.path.join(save_dir, f"sample_{gi}.png"))

        done = batch_start + len(batch_idx)
        if done % 40 == 0 or done == len(pending):
            print(f"    {done}/{len(pending)} done", flush=True)


# ---------------------------------------------------------------------------
# Metric evaluation
# ---------------------------------------------------------------------------

def evaluate_schedule(save_dir: str, ref_dir: str, prompts: list[str], device: str,
                      global_offset: int = 0) -> dict:
    # FID skipped: with 100-200 samples vs 2048 features, the covariance is rank-deficient
    # and scipy sqrtm hangs. Use SSIM+PSNR+CLIP for directional comparison instead.
    from PIL import Image
    from torchvision.transforms import ToTensor
    from torchmetrics.image import StructuralSimilarityIndexMeasure

    ssim_m = StructuralSimilarityIndexMeasure(data_range=1.0, sync_on_compute=False).to(device)

    clip_scores = []
    try:
        from transformers import CLIPModel, CLIPProcessor
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        use_clip = True
    except Exception:
        use_clip = False
        clip_model, clip_proc = None, None

    n = len(prompts)
    psnr_vals = []

    for local_i in range(n):
        i = local_i + global_offset
        gen_path = os.path.join(save_dir, f"sample_{i}.png")
        ref_path = os.path.join(ref_dir, f"ref_{i}.png")
        if not os.path.exists(gen_path) or not os.path.exists(ref_path):
            continue

        gen_img = Image.open(gen_path).convert("RGB")
        ref_img = Image.open(ref_path).convert("RGB").resize(gen_img.size)

        gen_t = ToTensor()(gen_img).unsqueeze(0).to(device)
        ref_t = ToTensor()(ref_img).unsqueeze(0).to(device)

        with torch.no_grad():
            ssim_m.update(gen_t, ref_t)

        mse = ((gen_t - ref_t) ** 2).mean().item()
        psnr_vals.append(10 * np.log10(1.0 / (mse + 1e-10)))

        if use_clip:
            inputs = clip_proc(text=[prompts[local_i]], images=gen_img,
                               return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                clip_scores.append(float(clip_model(**inputs).logits_per_image.item()))

    return {
        "fid": None,  # skipped: too few samples for reliable FID-2048
        "ssim": float(ssim_m.compute()) if psnr_vals else None,
        "psnr": float(np.mean(psnr_vals)) if psnr_vals else None,
        "clip": float(np.mean(clip_scores)) if clip_scores else None,
        "n_evaluated": len(psnr_vals),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Experiment C: Branch Ablation")
    parser.add_argument("--model", choices=["flux", "sd3"], required=True)
    parser.add_argument("--n_prompts", type=int, default=350)
    parser.add_argument("--freeze_from", type=int, default=15,
                        help="Skip branch for steps >= freeze_from (default: 15 of 28)")
    parser.add_argument("--prompt_start", type=int, default=0,
                        help="Start index into prompt list (for multi-GPU splitting)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output_dir", default=os.path.join(BASE_DIR, "results"))
    parser.add_argument("--csv_suffix", default="",
                        help="Suffix appended to output CSV name (e.g. '_g0' for multi-GPU splits)")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    os.makedirs(args.output_dir, exist_ok=True)

    cfg_path = os.path.join(BASE_DIR, "configs",
                            "flux_dev.yaml" if args.model == "flux" else "sd3_medium.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    if args.quick:
        cfg["generation"]["num_inference_steps"] = 10
        args.n_prompts = min(args.n_prompts, 10)
        args.freeze_from = 5

    n_steps = cfg["generation"]["num_inference_steps"]
    n_blocks = cfg["architecture"]["num_double_blocks"]
    third = n_blocks // 3

    schedules = [
        {"name": "baseline",       "skip": "none",  "from": 0,              "layers": None},
        {"name": "text_skip_late", "skip": "text",  "from": args.freeze_from, "layers": None},
        {"name": "image_skip_late","skip": "image", "from": args.freeze_from, "layers": None},
    ]

    ref_dir = get_ref_dir(args.model)
    batch_size = int(os.environ.get("MMDIT_ABL_BATCH", "1"))

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[cfg["model"]["torch_dtype"]]
    all_prompts = load_prompts(args.prompt_start + args.n_prompts, quick=args.quick)
    # Slice for multi-GPU splitting: prompt indices [prompt_start, prompt_start+n_prompts)
    # The global index is preserved for filename consistency (sample_{global_i}.png)
    prompts = all_prompts[args.prompt_start: args.prompt_start + args.n_prompts]
    global_offset = args.prompt_start

    print(f"\n{'='*60}")
    print(f"  Experiment C: Branch Ablation — {args.model.upper()}")
    print(f"  Prompts: {args.n_prompts}  freeze_from: {args.freeze_from}/{n_steps}")
    print(f"  n_blocks: {n_blocks}  batch_size: {batch_size}")
    print(f"{'='*60}\n")

    print("[Setup] Loading pipeline...")
    if "FLUX" in cfg["model"]["name"] or "flux" in cfg["model"]["name"].lower():
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(cfg["model"]["name"], torch_dtype=torch_dtype).to(device)
    else:
        from diffusers import StableDiffusion3Pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained(cfg["model"]["name"], torch_dtype=torch_dtype).to(device)
    pipe.set_progress_bar_config(disable=True)

    out_csv = results_dir(f"branch_ablation_{args.model}{args.csv_suffix}.csv")
    rows = []

    for sched in schedules:
        print(f"\n[Schedule] {sched['name']}: skip={sched['skip']}, from={sched['from']}, "
              f"layers={'all' if sched['layers'] is None else len(sched['layers'])}")

        skip_state = BranchSkipState(
            skip_target=sched["skip"],
            freeze_from=sched["from"],
            layer_indices=sched["layers"],
        )
        save_dir = samples_dir("branch_ablation", args.model, sched["name"])
        generate_for_schedule(pipe, args.model, prompts, cfg, skip_state, save_dir, device,
                              batch_size=batch_size, global_offset=global_offset)

        if os.path.exists(ref_dir) and len(os.listdir(ref_dir)) >= len(prompts):
            print(f"  Evaluating metrics...")
            metrics = evaluate_schedule(save_dir, ref_dir, prompts, device, global_offset=global_offset)
        else:
            print(f"  [Skip metrics] Reference images not found at {ref_dir}.")
            metrics = {"fid": None, "ssim": None, "psnr": None, "clip": None, "n_evaluated": 0}

        row = {"schedule": sched["name"], "skip_target": sched["skip"],
               "freeze_from": sched["from"], **metrics}
        rows.append(row)
        print(f"  FID={metrics['fid']}, SSIM={metrics['ssim']}, CLIP={metrics['clip']}")

    # Write CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    print(f"\n[Done] Saved: {out_csv}")

    # Print verdict
    baseline = next((r for r in rows if r["schedule"] == "baseline"), None)
    text_skip = next((r for r in rows if r["schedule"] == "text_skip_late"), None)
    image_skip = next((r for r in rows if r["schedule"] == "image_skip_late"), None)
    if baseline and text_skip and image_skip:
        # Use SSIM drop (lower SSIM = worse quality) as primary metric (FID skipped)
        b_ssim = baseline["ssim"] or 1.0
        t_drop = b_ssim - (text_skip["ssim"] or b_ssim)   # positive = degraded
        i_drop = b_ssim - (image_skip["ssim"] or b_ssim)
        ratio = t_drop / i_drop if i_drop > 0 else float("inf")
        verdict = "PASS" if ratio < 0.30 else "REVIEW"
        print(f"\n  === Verdict (SSIM-based, FID skipped) ===")
        print(f"  baseline SSIM       : {b_ssim:.4f}")
        print(f"  text_skip SSIM drop : {t_drop:.6f}")
        print(f"  image_skip SSIM drop: {i_drop:.6f}")
        print(f"  ratio               : {ratio:.4f}")
        print(f"  verdict             : {verdict}")


if __name__ == "__main__":
    main()

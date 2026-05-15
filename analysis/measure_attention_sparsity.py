"""
measure_attention_sparsity.py — Experiment B: Attention block sparsity.

Decomposes the joint attention map A into 4 quadrants:
  A_TT (text×text), A_TI (text×image), A_IT (image×text), A_II (image×image)

For each quadrant × layer × captured-step, measures:
  - Shannon entropy  (lower → sparser / more focused)
  - Top-5% key coverage  (higher → more concentrated)
  - Sparsity ratio at threshold 0.01

Uses AttnProcessor sub-classing to intercept post-softmax attention weights.
Captures only step_sample_indices steps (not all 28) to bound memory.

Outputs:
  results/attn_sparsity_{model}.json

Usage:
  python analysis/measure_attention_sparsity.py --model flux --n_prompts 30 --gpu 0
  python analysis/measure_attention_sparsity.py --model flux --n_prompts 5  --gpu 0 --quick
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
from collections import defaultdict

import torch
import yaml
import numpy as np

from analysis.hook_utils import (
    AttentionMapCapture,
    _split_attention_map,
    make_step_callback,
)
from analysis.paths import BASE_DIR, results_dir

QUADRANTS = ["TT", "TI", "IT", "II"]


# ---------------------------------------------------------------------------
# Custom AttnProcessor that captures attention weights
# ---------------------------------------------------------------------------

class CapturingAttnProcessor:
    """Wraps a FluxAttnProcessor/JointAttnProcessor to capture attention maps."""

    def __init__(self, original_processor, capture: AttentionMapCapture, block_idx: int,
                 n_txt: int):
        self.orig = original_processor
        self.capture = capture
        self.block_idx = block_idx
        self.n_txt = n_txt

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, **kwargs):
        # Run original processor
        out = self.orig(attn, hidden_states, encoder_hidden_states=encoder_hidden_states, **kwargs)

        step = self.capture.current_step
        if step not in self.capture.target_steps:
            return out

        # Re-compute attention weights (read-only, no grad)
        # We use the stored Q/K from the last forward pass indirectly:
        # For efficiency, we hook at a lower level via the qkv projection.
        # This is handled by _attach_qkv_hook below.
        return out


def _attach_qkv_hooks(
    transformer,
    capture: AttentionMapCapture,
    n_txt: int,
    is_flux: bool,
) -> list:
    """
    Capture post-softmax attention maps by:
    1. Setting a per-block context via pre/post hooks on each attn module.
    2. Monkey-patching F.scaled_dot_product_attention globally so that when
       SDPA is called inside block[bidx].attn, we intercept (q, k) which are
       already [B, H, N_txt+N_img, head_dim] with RoPE applied.
    """
    import torch.nn.functional as F
    import math

    handles = []
    blocks = transformer.transformer_blocks
    orig_sdpa = F.scaled_dot_product_attention
    block_ctx = {"idx": -1}  # which block is currently running

    def patched_sdpa(*args, **kwargs):
        out = orig_sdpa(*args, **kwargs)
        q = args[0] if len(args) > 0 else kwargs.get("query")
        k = args[1] if len(args) > 1 else kwargs.get("key")
        if q is None or k is None:
            return out
        bidx = block_ctx["idx"]
        step = capture.current_step
        if bidx < 0 or step not in capture.target_steps:
            return out
        if q.dim() != 4:
            return out
        B, H, N, D = q.shape
        if N < n_txt + 2:
            return out
        with torch.no_grad():
            scale = math.sqrt(D)
            scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / scale
            attn_w = torch.softmax(scores, dim=-1)   # [B, H, N_total, N_total]
            # Compute metrics per quadrant ON GPU — never transfer full tensor to CPU
            # FLUX: concat = [txt, img] → split at n_txt
            # SD3:  concat = [img, txt] → split at n_img=(N-n_txt), then remap labels
            if is_flux:
                quads = _split_attention_map(attn_w, n_txt)
            else:
                n_img = N - n_txt
                raw = _split_attention_map(attn_w, n_img)
                # raw uses n_img as split → raw["TT"]=II, raw["TI"]=IT, raw["IT"]=TI, raw["II"]=TT
                quads = {"II": raw["TT"], "IT": raw["TI"], "TI": raw["IT"], "TT": raw["II"]}
            metrics = {}
            for quad, blk in quads.items():
                if blk.numel() == 0:
                    metrics[quad] = {"entropy": 0.0, "top5_cov": 0.0, "sparsity": 0.0}
                    continue
                p = blk.clamp(min=1e-10)
                ent = float(-(p * p.log()).sum(dim=-1).mean().item())
                Kq = blk.shape[-1]
                k_top = max(1, int(Kq * 0.05))
                top5 = float(blk.topk(k_top, dim=-1).values.sum(dim=-1).mean().item())
                spar = float((blk < 0.01).float().mean().item())
                metrics[quad] = {"entropy": ent, "top5_cov": top5, "sparsity": spar}
            capture.maps[bidx][step] = metrics   # only scalars stored
            del attn_w, quads
        return out

    F.scaled_dot_product_attention = patched_sdpa

    # Store orig so remove_hooks can restore it
    capture._orig_sdpa = orig_sdpa
    orig_remove = capture.remove_hooks.__func__ if hasattr(capture.remove_hooks, '__func__') else None

    def patched_remove(self=capture):
        if hasattr(self, '_orig_sdpa'):
            F.scaled_dot_product_attention = self._orig_sdpa
            del self._orig_sdpa
        for h in self._handles:
            h.remove()
        self._handles.clear()

    capture.remove_hooks = patched_remove

    for block_idx, block in enumerate(blocks):
        attn_module = getattr(block, "attn", None)
        if attn_module is None:
            continue

        def make_hooks(bidx):
            def pre_hook(module, args):
                block_ctx["idx"] = bidx
            def post_hook(module, args, output):
                block_ctx["idx"] = -1
            return pre_hook, post_hook

        pre, post = make_hooks(block_idx)
        h_pre  = attn_module.register_forward_pre_hook(pre)
        h_post = attn_module.register_forward_hook(post)
        handles.extend([h_pre, h_post])
        capture._handles.extend([h_pre, h_post])

    return handles


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompts(n: int, quick: bool = False) -> list[str]:
    if quick:
        return [
            "A photorealistic portrait of an astronaut on Mars at sunset",
            "A cozy coffee shop in Paris with rain on the windows",
            "A majestic lion standing on a rocky cliff, golden hour",
            "Abstract geometric art with vibrant neon colors",
            "A serene Japanese garden with cherry blossoms",
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
    fallback = [
        "A photorealistic portrait of an astronaut on Mars at sunset",
        "A cozy coffee shop in Paris with rain on the windows",
        "A majestic lion standing on a rocky cliff, golden hour",
        "Abstract geometric art with vibrant neon colors",
        "A serene Japanese garden with cherry blossoms reflected in a pond",
    ]
    return (fallback * (n // len(fallback) + 1))[:n]


# ---------------------------------------------------------------------------
# Main measurement
# ---------------------------------------------------------------------------

def run_sparsity_measurement(pipe, model_key, prompts, cfg, device):
    gen_cfg = cfg["generation"]
    n_steps = gen_cfg["num_inference_steps"]
    is_flux = model_key == "flux"
    n_txt = cfg["architecture"]["num_text_tokens"]
    target_steps = cfg["analysis"]["step_sample_indices"]
    target_steps = [s for s in target_steps if s < n_steps]

    # Aggregate: block → step → quad → [metric values]
    agg = defaultdict(lambda: defaultdict(lambda: {q: {"entropy": [], "top5_cov": [], "sparsity": []}
                                                    for q in QUADRANTS}))

    for p_idx, prompt in enumerate(prompts):
        print(f"  prompt {p_idx+1}/{len(prompts)}", flush=True)

        capture = AttentionMapCapture(target_steps=target_steps)
        capture.current_step = 0

        _attach_qkv_hooks(pipe.transformer, capture, n_txt, is_flux)

        def step_cb(pipe_, step_idx, timestep, cb_kwargs):
            capture.next_step()
            return cb_kwargs

        gen = torch.Generator(device=device).manual_seed(gen_cfg["seed"] + p_idx)
        with torch.no_grad():
            pipe(
                prompt,
                num_inference_steps=n_steps,
                guidance_scale=gen_cfg["guidance_scale"],
                height=gen_cfg["height"],
                width=gen_cfg["width"],
                generator=gen,
                output_type="latent",
                callback_on_step_end=step_cb,
            )

        # Accumulate metrics — maps now stores scalars, not tensors
        for block_idx, step_dict in capture.maps.items():
            for step, quad_dict in step_dict.items():
                for quad, m in quad_dict.items():
                    if m["entropy"] == 0.0 and m["top5_cov"] == 0.0:
                        continue
                    agg[block_idx][step][quad]["entropy"].append(m["entropy"])
                    agg[block_idx][step][quad]["top5_cov"].append(m["top5_cov"])
                    agg[block_idx][step][quad]["sparsity"].append(m["sparsity"])

        capture.remove_hooks()
        del capture

        # Free GPU memory
        torch.cuda.empty_cache()

    # Summarize
    result = {}
    for block_idx in sorted(agg.keys()):
        result[block_idx] = {}
        for step in sorted(agg[block_idx].keys()):
            result[block_idx][step] = {}
            for quad in QUADRANTS:
                vals = agg[block_idx][step][quad]
                result[block_idx][step][quad] = {
                    "entropy_mean": float(np.mean(vals["entropy"])) if vals["entropy"] else None,
                    "top5_coverage_mean": float(np.mean(vals["top5_cov"])) if vals["top5_cov"] else None,
                    "sparsity_mean": float(np.mean(vals["sparsity"])) if vals["sparsity"] else None,
                }
    return result


def summarize_sparsity(result: dict) -> dict:
    """Compare II entropy vs TI entropy to check asymmetry."""
    ii_ent, ti_ent = [], []
    for block_dict in result.values():
        for step_dict in block_dict.values():
            if result.get(list(result.keys())[0]):
                if step_dict.get("II", {}).get("entropy_mean") is not None:
                    ii_ent.append(step_dict["II"]["entropy_mean"])
                if step_dict.get("TI", {}).get("entropy_mean") is not None:
                    ti_ent.append(step_dict["TI"]["entropy_mean"])

    ii_mean = float(np.mean(ii_ent)) if ii_ent else 0.0
    ti_mean = float(np.mean(ti_ent)) if ti_ent else 1.0
    ratio = ii_mean / ti_mean if ti_mean > 0 else 1.0
    verdict = "PASS" if ratio < 0.50 else "REVIEW"
    return {
        "II_entropy_mean": ii_mean,
        "TI_entropy_mean": ti_mean,
        "II_TI_entropy_ratio": ratio,
        "verdict": verdict,
        "criterion": "II_entropy < 50% of TI_entropy → asymmetric sparse mask justified",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Experiment B: Attention Sparsity")
    parser.add_argument("--model", choices=["flux", "sd3"], required=True)
    parser.add_argument("--n_prompts", type=int, default=30)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output_dir", default=results_dir())
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    os.makedirs(args.output_dir, exist_ok=True)

    cfg_name = "flux_schnell.yaml" if (args.quick and args.model == "flux") else \
               ("flux_dev.yaml" if args.model == "flux" else "sd3_medium.yaml")
    cfg_path = os.path.join(BASE_DIR, "configs", cfg_name)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    if args.quick:
        cfg["generation"]["num_inference_steps"] = 10
        cfg["analysis"]["step_sample_indices"] = [0, 4, 9]
        args.n_prompts = min(args.n_prompts, 5)

    out_path = os.path.join(args.output_dir, f"attn_sparsity_{args.model}.json")

    print(f"\n{'='*60}")
    print(f"  Experiment B: Attention Sparsity — {args.model.upper()}")
    print(f"  Captured steps: {cfg['analysis']['step_sample_indices']}")
    print(f"  Prompts: {args.n_prompts}")
    print(f"{'='*60}\n")

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[cfg["model"]["torch_dtype"]]
    prompts = load_prompts(args.n_prompts, quick=args.quick)

    print("[Setup] Loading pipeline...")
    if "FLUX" in cfg["model"]["name"] or "flux" in cfg["model"]["name"].lower():
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(cfg["model"]["name"], torch_dtype=torch_dtype).to(device)
    else:
        from diffusers import StableDiffusion3Pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained(cfg["model"]["name"], torch_dtype=torch_dtype).to(device)
    pipe.set_progress_bar_config(disable=True)

    print("[Run] Measuring attention sparsity...")
    result = run_sparsity_measurement(pipe, args.model, prompts, cfg, device)
    summary = summarize_sparsity(result)

    output = {
        "config": {
            "model": args.model,
            "n_prompts": args.n_prompts,
            "step_sample_indices": cfg["analysis"]["step_sample_indices"],
        },
        "summary": summary,
        "data": result,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[Done] Saved: {out_path}")
    print(f"\n  === Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

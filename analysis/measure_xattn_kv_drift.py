"""
measure_xattn_kv_drift.py — Experiment D: Cross-attention (text-side) KV drift.

In MM-DiT, joint self-attention uses [text; image] concatenation. The text-side
portion of K and V tensors effectively acts as cross-attention keys/values for
image tokens. This script measures how much these text-side K/V change across steps.

If they are nearly static, caching them from step 0 is safe and yields large
memory bandwidth savings (one of the biggest wins in idea2).

Additionally runs a "freeze_kv" ablation: forces text-side K,V to their step-0
values for all subsequent steps and measures quality impact.

Outputs:
  results/xattn_kv_drift_{model}.json
  results/freeze_text_kv_quality.csv

Usage:
  python analysis/measure_xattn_kv_drift.py --model flux --n_prompts 50 --gpu 0
  python analysis/measure_xattn_kv_drift.py --model flux --n_prompts 5  --gpu 0 --quick
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import json
import argparse
from collections import defaultdict

import torch
import yaml
import numpy as np

from analysis.paths import ref_dir as get_ref_dir, samples_dir, results_dir, BASE_DIR


# ---------------------------------------------------------------------------
# KV capture hook
# ---------------------------------------------------------------------------

class KVCaptureState:
    """
    GPU-efficient KV drift state: computes relative L2 drift on-GPU per step,
    stores only scalar floats. No GPU→CPU tensor transfers during inference.
    """

    def __init__(self, n_txt: int, target_steps: list[int] | None = None):
        self.n_txt = n_txt
        self.target_steps = set(target_steps) if target_steps else None
        self.step = 0
        # data[block_idx][step] = {"K_drift": float, "V_drift": float}
        self.data: dict[int, dict[int, dict[str, float]]] = defaultdict(dict)
        # step-0 reference tensors kept on GPU
        self._k_ref: dict[int, torch.Tensor] = {}
        self._v_ref: dict[int, torch.Tensor] = {}
        # intra-step buffer: K hook fires before V hook
        self._k_cur: dict[int, torch.Tensor] = {}
        self._handles: list = []

    def next_step(self):
        self.step += 1

    def reset(self):
        self.step = 0
        self._k_ref.clear()
        self._v_ref.clear()
        self._k_cur.clear()

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._k_ref.clear()
        self._v_ref.clear()
        self._k_cur.clear()


def _kv_relative_l2(cur: torch.Tensor, ref: torch.Tensor) -> float:
    """Global Frobenius norm: ||cur - ref||_F / ||ref||_F, computed on GPU."""
    cur_f = cur.float().squeeze(0)
    ref_f = ref.float().squeeze(0)
    return float((cur_f - ref_f).norm() / ref_f.norm().clamp(min=1e-8))


def _make_k_capture_hook(state: KVCaptureState, block_idx: int, k_buf: list):
    """Post-hook on attn.add_k_proj: buffer current K on GPU for pairing with V."""
    def hook(module, args, output):
        if state.target_steps is not None and state.step not in state.target_steps:
            return
        state._k_cur[block_idx] = output.detach()
    return hook


def _make_v_capture_hook(state: KVCaptureState, block_idx: int, k_buf: list):
    """Post-hook on attn.add_v_proj: compute drift scalar on GPU, store float only."""
    def hook(module, args, output):
        step = state.step
        if state.target_steps is not None and step not in state.target_steps:
            return
        k_cur = state._k_cur.pop(block_idx, None)
        if k_cur is None:
            return
        v_cur = output.detach()
        if block_idx not in state._k_ref:
            # Step 0: store reference on GPU
            state._k_ref[block_idx] = k_cur.clone()
            state._v_ref[block_idx] = v_cur.clone()
            state.data[block_idx][step] = {"K_drift": 0.0, "V_drift": 0.0}
        else:
            state.data[block_idx][step] = {
                "K_drift": _kv_relative_l2(k_cur, state._k_ref[block_idx]),
                "V_drift": _kv_relative_l2(v_cur, state._v_ref[block_idx]),
            }
    return hook


def register_kv_hooks(transformer, state: KVCaptureState, is_flux: bool):
    # add_k_proj / add_v_proj project encoder (text) hidden states.
    # to_k / to_v project image hidden states — those are NOT what we want here.
    for idx, block in enumerate(transformer.transformer_blocks):
        attn = getattr(block, "attn", None)
        if attn is None:
            continue
        add_k = getattr(attn, "add_k_proj", None)
        add_v = getattr(attn, "add_v_proj", None)
        if add_k is None or add_v is None:
            continue
        h_k = add_k.register_forward_hook(_make_k_capture_hook(state, idx, None))
        h_v = add_v.register_forward_hook(_make_v_capture_hook(state, idx, None))
        state._handles.extend([h_k, h_v])


# ---------------------------------------------------------------------------
# Relative drift: ||K^t - K^0|| / ||K^0||
# ---------------------------------------------------------------------------

def compute_kv_drift(state: KVCaptureState) -> dict[int, dict[int, dict[str, float]]]:
    """Returns {block: {step: {K_drift: float, V_drift: float}}}."""
    result = {}
    for block_idx, step_dict in state.data.items():
        steps = sorted(step_dict.keys())
        if not steps:
            continue
        k0 = step_dict[steps[0]]["K"].float()
        v0 = step_dict[steps[0]]["V"].float()
        result[block_idx] = {}
        for step in steps:
            kt = step_dict[step]["K"].float()
            vt = step_dict[step]["V"].float()
            k_drift = float((kt - k0).norm() / k0.norm().clamp(min=1e-8))
            v_drift = float((vt - v0).norm() / v0.norm().clamp(min=1e-8))
            result[block_idx][step] = {"K_drift": k_drift, "V_drift": v_drift}
    return result


# ---------------------------------------------------------------------------
# Freeze-KV ablation: monkey-patch attention to use step-0 KV for text
# ---------------------------------------------------------------------------

class FreezeKVState:
    def __init__(self, n_txt: int, freeze_from: int = 1):
        self.n_txt = n_txt
        self.freeze_from = freeze_from
        self.step = 0
        # stored[block_idx] = {"K": ..., "V": ...}  filled at step 0
        self.stored: dict[int, dict[str, torch.Tensor]] = {}

    def next_step(self):
        self.step += 1

    def reset(self):
        self.step = 0
        self.stored.clear()


def _make_proj_freeze_hook(fstate: FreezeKVState, block_idx: int, kv_key: str):
    """
    Post-hook on attn.add_k_proj or attn.add_v_proj.
    - step < freeze_from : capture output as the cached reference (keep updating).
    - step >= freeze_from: return cached step-(freeze_from-1) output instead.
    kv_key: "K" or "V"
    """
    def hook(module, args, output):
        stored_block = fstate.stored.setdefault(block_idx, {})
        if fstate.step < fstate.freeze_from:
            stored_block[kv_key] = output.detach().clone()
            return  # pass through unchanged
        cached = stored_block.get(kv_key)
        if cached is None:
            return
        return cached.to(device=output.device, dtype=output.dtype)
    return hook


# ---------------------------------------------------------------------------
# Prompt + pipeline loading
# ---------------------------------------------------------------------------

def load_prompts(n, quick=False):
    if quick:
        return [
            "A photorealistic portrait of an astronaut on Mars at sunset",
            "A cozy coffee shop in Paris with rain on the windows",
            "A majestic lion standing on a rocky cliff",
            "Abstract geometric art",
            "A serene Japanese garden",
        ][:n]
    try:
        from datasets import load_dataset
        ds = load_dataset("xingjianleng/mjhq30k", split="test", streaming=True)
        return [entry["text"] for i, entry in enumerate(ds) if i < n]
    except Exception:
        return ["A beautiful landscape" for _ in range(n)]


def load_pipe(cfg, device):
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[cfg["model"]["torch_dtype"]]
    name = cfg["model"]["name"]
    if "FLUX" in name or "flux" in name.lower():
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(name, torch_dtype=torch_dtype).to(device)
    else:
        from diffusers import StableDiffusion3Pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained(name, torch_dtype=torch_dtype).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Experiment D: Cross-Attn KV Drift")
    parser.add_argument("--model", choices=["flux", "sd3"], required=True)
    parser.add_argument("--n_prompts", type=int, default=50)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output_dir", default=os.path.join(BASE_DIR, "results"))
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    os.makedirs(args.output_dir, exist_ok=True)

    cfg_path = os.path.join(BASE_DIR, "configs",
                            "flux_dev.yaml" if args.model == "flux" else "sd3_medium.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    if args.quick:
        cfg["generation"]["num_inference_steps"] = 10
        args.n_prompts = min(args.n_prompts, 5)

    n_steps = cfg["generation"]["num_inference_steps"]
    n_txt = cfg["architecture"]["num_text_tokens"]
    is_flux = args.model == "flux"

    print(f"\n{'='*60}")
    print(f"  Experiment D: Cross-Attn KV Drift — {args.model.upper()}")
    print(f"  n_txt: {n_txt}  Steps: {n_steps}  Prompts: {args.n_prompts}")
    print(f"{'='*60}\n")

    prompts = load_prompts(args.n_prompts, args.quick)
    pipe = load_pipe(cfg, device)

    # ── Part 1: Drift measurement ────────────────────────────────────────────
    print("[Part 1] Measuring KV drift across steps...")
    agg_drift: dict[int, dict[int, dict[str, list]]] = defaultdict(
        lambda: defaultdict(lambda: {"K_drift": [], "V_drift": []}))

    for p_idx, prompt in enumerate(prompts):
        print(f"  prompt {p_idx+1}/{len(prompts)}", flush=True)

        kv_state = KVCaptureState(n_txt=n_txt)
        register_kv_hooks(pipe.transformer, kv_state, is_flux)

        def step_cb(pipe_, step_idx, timestep, cb_kwargs):
            kv_state.next_step()
            return cb_kwargs

        gen = torch.Generator(device=device).manual_seed(cfg["generation"]["seed"] + p_idx)
        with torch.no_grad():
            pipe(
                prompt,
                num_inference_steps=n_steps,
                guidance_scale=cfg["generation"]["guidance_scale"],
                height=cfg["generation"]["height"],
                width=cfg["generation"]["width"],
                generator=gen,
                output_type="latent",
                callback_on_step_end=step_cb,
            )

        for block_idx, step_dict in kv_state.data.items():
            for step, metrics in step_dict.items():
                agg_drift[block_idx][step]["K_drift"].append(metrics["K_drift"])
                agg_drift[block_idx][step]["V_drift"].append(metrics["V_drift"])

        kv_state.remove_hooks()
        del kv_state
        torch.cuda.empty_cache()

    # Summarize drift
    drift_result = {}
    for block_idx in sorted(agg_drift.keys()):
        drift_result[block_idx] = {}
        for step in sorted(agg_drift[block_idx].keys()):
            d = agg_drift[block_idx][step]
            drift_result[block_idx][step] = {
                "K_drift_mean": float(np.mean(d["K_drift"])),
                "V_drift_mean": float(np.mean(d["V_drift"])),
            }

    # Compute final-step drift average (step T-1)
    final_k_drifts, final_v_drifts = [], []
    for block_idx, step_dict in drift_result.items():
        if step_dict:
            last_step = max(int(s) for s in step_dict)
            final_k_drifts.append(step_dict[last_step]["K_drift_mean"])
            final_v_drifts.append(step_dict[last_step]["V_drift_mean"])

    avg_final_k = float(np.mean(final_k_drifts)) if final_k_drifts else 0.0
    avg_final_v = float(np.mean(final_v_drifts)) if final_v_drifts else 0.0
    drift_verdict = "PASS" if avg_final_k < 0.10 and avg_final_v < 0.10 else "REVIEW"

    drift_summary = {
        "final_step_K_drift_mean": avg_final_k,
        "final_step_V_drift_mean": avg_final_v,
        "verdict": drift_verdict,
        "criterion": "K_drift < 10% and V_drift < 10% at final step → KV reuse justified",
    }

    # Save drift JSON
    drift_out = results_dir(f"xattn_kv_drift_{args.model}.json")
    with open(drift_out, "w") as f:
        json.dump({"config": {"model": args.model, "n_prompts": args.n_prompts},
                   "summary": drift_summary, "data": drift_result}, f, indent=2)
    print(f"[Done] Saved: {drift_out}")
    print(f"  Final K drift: {avg_final_k:.4f}  V drift: {avg_final_v:.4f}  → {drift_verdict}")

    # ── Part 2: Freeze-KV ablation (quality impact) ──────────────────────────
    print("\n[Part 2] Freeze text-side KV from step 0 → quality impact...")
    ref_dir = get_ref_dir(args.model)

    schedules = [
        ("no_freeze", 9999),    # effectively never freeze (baseline)
        ("freeze_from_1", 1),   # freeze KV after step 0
        ("freeze_from_5", 5),   # freeze KV after step 5
    ]

    csv_rows = []
    for sched_name, freeze_from in schedules:
        save_dir = samples_dir("freeze_kv", args.model, sched_name)
        os.makedirs(save_dir, exist_ok=True)

        fstate = FreezeKVState(n_txt=n_txt, freeze_from=freeze_from)
        handles = []

        for idx, block in enumerate(pipe.transformer.transformer_blocks):
            attn = getattr(block, "attn", None)
            if attn is None:
                continue
            add_k = getattr(attn, "add_k_proj", None)
            add_v = getattr(attn, "add_v_proj", None)
            if add_k is None or add_v is None:
                continue
            h_k = add_k.register_forward_hook(_make_proj_freeze_hook(fstate, idx, "K"))
            h_v = add_v.register_forward_hook(_make_proj_freeze_hook(fstate, idx, "V"))
            handles.extend([h_k, h_v])

        for i, prompt in enumerate(prompts[:min(50, args.n_prompts)]):
            out_path = os.path.join(save_dir, f"sample_{i}.png")
            if os.path.exists(out_path):
                continue
            fstate.reset()

            def step_cb_freeze(pipe_, step_idx, timestep, cb_kwargs):
                fstate.next_step()
                return cb_kwargs

            gen = torch.Generator(device=device).manual_seed(cfg["generation"]["seed"] + i)
            with torch.no_grad():
                result = pipe(
                    prompt,
                    num_inference_steps=n_steps,
                    guidance_scale=cfg["generation"]["guidance_scale"],
                    height=cfg["generation"]["height"],
                    width=cfg["generation"]["width"],
                    generator=gen,
                    callback_on_step_end=step_cb_freeze,
                )
            result.images[0].save(out_path)

        for h in handles:
            h.remove()

        # Quick PSNR vs reference if available
        n_eval = min(50, args.n_prompts)
        psnr_vals = []
        if os.path.exists(ref_dir):
            from PIL import Image
            from torchvision.transforms import ToTensor
            for i in range(n_eval):
                gen_p = os.path.join(save_dir, f"sample_{i}.png")
                ref_p = os.path.join(ref_dir, f"ref_{i}.png")
                if not os.path.exists(gen_p) or not os.path.exists(ref_p):
                    continue
                g = ToTensor()(Image.open(gen_p).convert("RGB")).unsqueeze(0)
                r = ToTensor()(Image.open(ref_p).convert("RGB").resize(
                    Image.open(gen_p).size)).unsqueeze(0)
                mse = ((g - r) ** 2).mean().item()
                psnr_vals.append(10 * np.log10(1.0 / (mse + 1e-10)))

        psnr_mean = float(np.mean(psnr_vals)) if psnr_vals else None
        csv_rows.append({"schedule": sched_name, "freeze_from": freeze_from,
                         "psnr": psnr_mean, "n_eval": n_eval})
        print(f"  {sched_name}: PSNR={psnr_mean}")

    csv_out = results_dir("freeze_text_kv_quality.csv")
    if csv_rows:
        with open(csv_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
    print(f"[Done] Saved: {csv_out}")


if __name__ == "__main__":
    main()

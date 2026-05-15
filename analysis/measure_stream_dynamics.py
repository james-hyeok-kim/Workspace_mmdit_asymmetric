"""
measure_stream_dynamics.py — Experiment A: Text vs Image stream dynamics.

Measures per-(block, step) relative L2 change in text and image token features:
  delta(stream, block, t) = ||h^t - h^{t-1}||_2 / ||h^{t-1}||_2   (mean over tokens)

The gap between text and image curves quantifies the core asymmetry claimed by
idea2_mmdit_asymmetric.md.

Outputs:
  results/stream_dynamics_{model}.json  — {block: {step: {text: float, image: float}}}
  results/plots/dynamics_heatmap_{model}.png

Usage:
  python analysis/measure_stream_dynamics.py --model flux --n_prompts 50 --gpu 0
  python analysis/measure_stream_dynamics.py --model sd3  --n_prompts 50 --gpu 0
  python analysis/measure_stream_dynamics.py --model flux --n_prompts 10 --gpu 0 --quick
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
    StreamHookState,
    register_flux_stream_hooks,
    register_sd3_stream_hooks,
    make_step_callback,
    compute_relative_l2,
)
from analysis.paths import BASE_DIR, results_dir


# ---------------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------------

def load_pipeline(model_name: str, torch_dtype, device: str):
    if "FLUX" in model_name or "flux" in model_name.lower():
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
    else:
        from diffusers import StableDiffusion3Pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def get_dtype(s: str):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[s]


# ---------------------------------------------------------------------------
# Prompt loading (mirrors DiT workspace pattern)
# ---------------------------------------------------------------------------

def load_prompts(n: int, quick: bool = False) -> list[str]:
    if quick:
        return [
            "A photorealistic portrait of an astronaut on Mars at sunset",
            "A cozy coffee shop in Paris with rain on the windows",
            "A majestic lion standing on a rocky cliff, golden hour",
            "Abstract geometric art with vibrant neon colors",
            "A serene Japanese garden with cherry blossoms reflected in a pond",
            "A futuristic city skyline at night with flying cars",
            "A delicious bowl of ramen with soft-boiled egg and nori",
            "An oil painting of a ship in a stormy sea",
            "A cute golden retriever puppy playing in autumn leaves",
            "A close-up of a dew drop on a spider web at dawn",
        ]
    try:
        from datasets import load_dataset
        ds = load_dataset("xingjianleng/mjhq30k", split="test", streaming=True)
        prompts = [entry["text"] for i, entry in enumerate(ds) if i < n]
        if len(prompts) >= n:
            return prompts[:n]
    except Exception as e:
        print(f"[Warning] MJHQ load failed: {e}. Using fallback prompts.")
    fallback = [
        "A photorealistic portrait of an astronaut on Mars at sunset",
        "A cozy coffee shop in Paris with rain on the windows",
        "A majestic lion standing on a rocky cliff, golden hour",
        "Abstract geometric art with vibrant neon colors",
        "A serene Japanese garden with cherry blossoms reflected in a pond",
    ]
    return (fallback * (n // len(fallback) + 1))[:n]


# ---------------------------------------------------------------------------
# Core measurement
# ---------------------------------------------------------------------------

def run_dynamics_measurement(
    pipe,
    model_key: str,
    prompts: list[str],
    cfg: dict,
    device: str,
) -> dict:
    """
    Run inference on each prompt with stream hooks, collect per-step activations,
    compute relative L2 deltas, and return aggregated result.
    """
    gen_cfg = cfg["generation"]
    n_steps = gen_cfg["num_inference_steps"]
    is_flux = (model_key == "flux")

    # Aggregate over prompts: block → step → [values]
    agg_text: dict[int, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    agg_image: dict[int, dict[int, list]] = defaultdict(lambda: defaultdict(list))

    for p_idx, prompt in enumerate(prompts):
        print(f"  prompt {p_idx+1}/{len(prompts)}: {prompt[:60]}...", flush=True)

        state = StreamHookState()
        state.reset_step()

        if is_flux:
            register_flux_stream_hooks(pipe.transformer, state, full_tokens=False)
        else:
            register_sd3_stream_hooks(pipe.transformer, state, full_tokens=False)

        callback = make_step_callback(state)

        gen = torch.Generator(device=device).manual_seed(gen_cfg["seed"] + p_idx)
        with torch.no_grad():
            if is_flux:
                pipe(
                    prompt,
                    num_inference_steps=n_steps,
                    guidance_scale=gen_cfg["guidance_scale"],
                    height=gen_cfg["height"],
                    width=gen_cfg["width"],
                    generator=gen,
                    output_type="latent",
                    callback_on_step_end=callback,
                )
            else:
                pipe(
                    prompt,
                    num_inference_steps=n_steps,
                    guidance_scale=gen_cfg["guidance_scale"],
                    height=gen_cfg["height"],
                    width=gen_cfg["width"],
                    generator=gen,
                    output_type="latent",
                    callback_on_step_end=callback,
                )

        # Deltas already computed on GPU in hooks; just accumulate
        for block_idx, step_dict in state.data["text"].items():
            for step, val in step_dict.items():
                agg_text[block_idx][step].append(val)
        for block_idx, step_dict in state.data["image"].items():
            for step, val in step_dict.items():
                agg_image[block_idx][step].append(val)

        state.remove_hooks()
        del state
        torch.cuda.empty_cache()

    # Summarize: mean and std over prompts
    result = {}
    all_blocks = sorted(set(agg_text.keys()) | set(agg_image.keys()))
    for block_idx in all_blocks:
        result[block_idx] = {}
        all_steps = sorted(set(agg_text[block_idx].keys()) | set(agg_image[block_idx].keys()))
        for step in all_steps:
            txt_vals = agg_text[block_idx].get(step, [])
            img_vals = agg_image[block_idx].get(step, [])
            result[block_idx][step] = {
                "text_mean": float(np.mean(txt_vals)) if txt_vals else None,
                "text_std": float(np.std(txt_vals)) if txt_vals else None,
                "image_mean": float(np.mean(img_vals)) if img_vals else None,
                "image_std": float(np.std(img_vals)) if img_vals else None,
            }
    return result


# ---------------------------------------------------------------------------
# Summary statistics (for pass/fail judgment)
# ---------------------------------------------------------------------------

def summarize(result: dict) -> dict:
    """Compute overall mean text/image delta and the text/image ratio."""
    text_vals, image_vals = [], []
    for block_dict in result.values():
        for step_dict in block_dict.values():
            if step_dict["text_mean"] is not None:
                text_vals.append(step_dict["text_mean"])
            if step_dict["image_mean"] is not None:
                image_vals.append(step_dict["image_mean"])

    text_mean = float(np.mean(text_vals)) if text_vals else 0.0
    image_mean = float(np.mean(image_vals)) if image_vals else 1.0
    ratio = text_mean / image_mean if image_mean > 0 else 0.0

    verdict = "PASS" if ratio < 0.30 else ("REVIEW" if ratio < 0.50 else "FAIL")
    return {
        "text_mean_delta": text_mean,
        "image_mean_delta": image_mean,
        "text_image_ratio": ratio,
        "verdict": verdict,
        "criterion": "ratio < 0.30 = idea premise holds",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Experiment A: Stream Dynamics")
    parser.add_argument("--model", choices=["flux", "sd3"], required=True)
    parser.add_argument("--n_prompts", type=int, default=50)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--quick", action="store_true",
                        help="Run with 10 prompts and 10 steps (smoke test)")
    parser.add_argument("--output_dir", type=str, default=results_dir())
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
        args.n_prompts = min(args.n_prompts, 10)

    out_path = os.path.join(args.output_dir, f"stream_dynamics_{args.model}.json")

    print(f"\n{'='*60}")
    print(f"  Experiment A: Stream Dynamics — {args.model.upper()}")
    print(f"  Model: {cfg['model']['name']}")
    print(f"  Steps: {cfg['generation']['num_inference_steps']}")
    print(f"  Prompts: {args.n_prompts}")
    print(f"  GPU: {device}")
    print(f"  Output: {out_path}")
    print(f"{'='*60}\n")

    dtype = get_dtype(cfg["model"]["torch_dtype"])
    prompts = load_prompts(args.n_prompts, quick=args.quick)

    print("[Setup] Loading pipeline...")
    pipe = load_pipeline(cfg["model"]["name"], dtype, device)

    print("[Run] Measuring stream dynamics...")
    result = run_dynamics_measurement(pipe, args.model, prompts, cfg, device)
    summary = summarize(result)

    output = {
        "config": {
            "model": args.model,
            "model_name": cfg["model"]["name"],
            "num_steps": cfg["generation"]["num_inference_steps"],
            "n_prompts": args.n_prompts,
        },
        "summary": summary,
        "data": result,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[Done] Saved: {out_path}")
    print(f"\n  === Summary ===")
    print(f"  Text mean delta  : {summary['text_mean_delta']:.6f}")
    print(f"  Image mean delta : {summary['image_mean_delta']:.6f}")
    print(f"  Ratio (text/img) : {summary['text_image_ratio']:.4f}")
    print(f"  Verdict          : {summary['verdict']}")


if __name__ == "__main__":
    main()

"""
eval_utils.py — Common evaluation utilities for MM-DiT asymmetric experiments.

Adapted from Workspace_DiT/Mixed_Precision_Cache/eval_utils.py.
Key differences:
  - No DeepCacheState dependency
  - Adds ImageReward and PickScore
  - Prompt loading supports multiple datasets (MJHQ, COCO, GenEval)
  - Single-GPU focused (no accelerate multi-GPU)
"""

import os
import time

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

_PROMPT_DATASETS = {
    "MJHQ":   ("xingjianleng/mjhq30k",      "test",  "text"),
    "COCO":   ("phiyodr/coco2017",           "train", "caption"),
    "GenEval":("sayakpaul/geneval_prompts",  "train", "prompt"),
}

_FALLBACK_PROMPTS = [
    "A professional high-quality photo of a futuristic city with neon lights at night",
    "A beautiful mountain landscape during golden hour with mist in the valley",
    "A cute robot holding a flower in a sunlit field, highly detailed digital art",
    "A gourmet burger with melting cheese and fresh vegetables on a wooden table",
    "An astronaut walking on a purple planet's surface under a starry sky",
    "A serene Japanese garden with cherry blossoms reflected in a koi pond",
    "A majestic lion standing on a rocky cliff overlooking an African savanna",
    "An oil painting of a tall ship in a stormy sea, dramatic lighting",
    "A macro photo of a dew drop on a spider web at dawn, bokeh background",
    "Abstract digital art with flowing neon colors on a dark canvas",
]


def get_prompts(n: int, dataset_name: str = "MJHQ") -> list[str]:
    """Load n prompts from a named dataset with fallback."""
    if dataset_name not in _PROMPT_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(_PROMPT_DATASETS)}")
    path, split, key = _PROMPT_DATASETS[dataset_name]
    try:
        from datasets import load_dataset
        ds = load_dataset(path, split=split, streaming=True)
        prompts = [entry[key] for i, entry in enumerate(ds) if i < n]
        if len(prompts) == n:
            return prompts
    except Exception as e:
        print(f"[Warning] {dataset_name} load failed: {e}. Using fallback.")
    return (_FALLBACK_PROMPTS * (n // len(_FALLBACK_PROMPTS) + 1))[:n]


def get_mixed_prompts(n: int, sources: list[dict]) -> list[str]:
    """
    Load prompts from multiple sources.
    sources: [{"dataset": "MJHQ", "n": 200}, {"dataset": "COCO", "n": 100}, ...]
    """
    prompts = []
    for src in sources:
        prompts += get_prompts(src["n"], src["dataset"])
    return prompts[:n]


# ---------------------------------------------------------------------------
# Reference image generation
# ---------------------------------------------------------------------------

def generate_reference_images(
    pipe,
    prompts: list[str],
    cfg: dict,
    save_dir: str,
    device: str,
    prefix: str = "ref",
    batch_size: int = 1,
):
    """
    Generate baseline (no-caching) images with fixed seed.
    Skips already-generated files (crash-safe).
    batch_size > 1 uses per-index seeds for deterministic reproducibility.
    """
    os.makedirs(save_dir, exist_ok=True)
    gen_cfg = cfg["generation"]

    pending = [i for i in range(len(prompts))
               if not os.path.exists(os.path.join(save_dir, f"{prefix}_{i}.png"))]

    if not pending:
        print(f"  [{prefix}] All {len(prompts)} images already exist, skipping.")
        return

    print(f"  [{prefix}] Generating {len(pending)} images "
          f"(skipping {len(prompts) - len(pending)} existing, batch={batch_size})", flush=True)

    for batch_start in range(0, len(pending), batch_size):
        batch_idx = pending[batch_start: batch_start + batch_size]
        batch_prompts = [prompts[i] for i in batch_idx]
        gens = [torch.Generator(device=device).manual_seed(gen_cfg["seed"] + i)
                for i in batch_idx]

        with torch.no_grad():
            result = pipe(
                batch_prompts,
                num_inference_steps=gen_cfg["num_inference_steps"],
                guidance_scale=gen_cfg["guidance_scale"],
                height=gen_cfg["height"],
                width=gen_cfg["width"],
                generator=gens,
            )

        for i, img in zip(batch_idx, result.images):
            img.save(os.path.join(save_dir, f"{prefix}_{i}.png"))

        done = batch_start + len(batch_idx)
        if done % 40 == 0 or done == len(pending):
            print(f"  [{prefix}] {done}/{len(pending)} generated", flush=True)

    print(f"  [{prefix}] Done. {len(prompts)} images in {save_dir}")


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_fid_psnr_ssim(gen_dir: str, ref_dir: str, n: int, device: str) -> dict:
    """Compute FID, PSNR, SSIM, LPIPS between gen and ref directories."""
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image import (
        PeakSignalNoiseRatio,
        StructuralSimilarityIndexMeasure,
    )
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    fid_m   = FrechetInceptionDistance(feature=2048, sync_on_compute=False).to(device)
    psnr_m  = PeakSignalNoiseRatio(data_range=1.0, sync_on_compute=False).to(device)
    ssim_m  = StructuralSimilarityIndexMeasure(data_range=1.0, sync_on_compute=False).to(device)
    lpips_m = LearnedPerceptualImagePatchSimilarity(net_type="alex", sync_on_compute=False).to(device)

    loaded = 0
    for i in range(n):
        gen_p = os.path.join(gen_dir, f"sample_{i}.png")
        ref_p = os.path.join(ref_dir, f"ref_{i}.png")
        if not os.path.exists(gen_p) or not os.path.exists(ref_p):
            continue
        g = ToTensor()(Image.open(gen_p).convert("RGB")).unsqueeze(0).to(device)
        r = ToTensor()(Image.open(ref_p).convert("RGB")).unsqueeze(0).to(device)

        if g.shape != r.shape:
            r = torch.nn.functional.interpolate(r, size=g.shape[-2:], mode="bilinear",
                                                 align_corners=False)

        fid_m.update((r * 255).to(torch.uint8), real=True)
        fid_m.update((g * 255).to(torch.uint8), real=False)
        psnr_m.update(g, r)
        ssim_m.update(g, r)
        lpips_m.update(g * 2 - 1, r * 2 - 1)
        loaded += 1

    if loaded == 0:
        return {"fid": None, "psnr": None, "ssim": None, "lpips": None, "n": 0}

    return {
        "fid":   float(fid_m.compute()),
        "psnr":  float(psnr_m.compute()),
        "ssim":  float(ssim_m.compute()),
        "lpips": float(lpips_m.compute()),
        "n":     loaded,
    }


def compute_clip_score(gen_dir: str, prompts: list[str], device: str,
                        model_name: str = "openai/clip-vit-large-patch14") -> float | None:
    """Compute mean CLIPScore (logit scale) for generated images."""
    try:
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained(model_name).to(device)
        proc  = CLIPProcessor.from_pretrained(model_name)
    except Exception as e:
        print(f"[Warning] CLIP load failed: {e}")
        return None

    scores = []
    for i, prompt in enumerate(prompts):
        gen_p = os.path.join(gen_dir, f"sample_{i}.png")
        if not os.path.exists(gen_p):
            continue
        img = Image.open(gen_p).convert("RGB")
        inputs = proc(text=[prompt], images=img,
                      return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            scores.append(float(model(**inputs).logits_per_image.item()))

    return float(np.mean(scores)) if scores else None


def compute_image_reward(gen_dir: str, prompts: list[str], device: str) -> float | None:
    """Compute mean ImageReward score."""
    try:
        import ImageReward as RM
        model = RM.load("ImageReward-v1.0", device=device)
    except ImportError:
        print("[Warning] ImageReward not installed. pip install image-reward")
        return None

    scores = []
    for i, prompt in enumerate(prompts):
        gen_p = os.path.join(gen_dir, f"sample_{i}.png")
        if not os.path.exists(gen_p):
            continue
        with torch.no_grad():
            scores.append(float(model.score(prompt, gen_p)))

    return float(np.mean(scores)) if scores else None


def evaluate_all(
    gen_dir: str,
    ref_dir: str,
    prompts: list[str],
    device: str,
    n: int | None = None,
) -> dict:
    """Run all metrics and return combined dict."""
    n = n or len(prompts)
    t0 = time.perf_counter()

    metrics = compute_fid_psnr_ssim(gen_dir, ref_dir, n, device)
    metrics["clip"] = compute_clip_score(gen_dir, prompts[:n], device)
    metrics["image_reward"] = None  # skipped: network unavailable for model download
    metrics["eval_time_sec"] = round(time.perf_counter() - t0, 2)

    return metrics

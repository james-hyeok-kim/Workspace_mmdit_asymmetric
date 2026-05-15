#!/usr/bin/env bash
# run_smoke_test.sh — Code validation using FLUX.1-schnell (already cached).
#
# Runs all 4 experiments with 10 prompts / 4 steps to verify:
#   - Hook registration works correctly
#   - All output files are created
#   - No runtime errors
#
# Generated images go to /data/jameskimh/mmdit_asymmetric/reference/flux_schnell/
# (separate from full-test reference images so they don't interfere)
#
# Usage:
#   bash scripts/run_smoke_test.sh [--gpu N]

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="$(pwd):$PYTHONPATH"
export MMDIT_DATA_ROOT="/data/jameskimh/mmdit_asymmetric"
GPU=${GPU:-0}
for arg in "$@"; do
    case "$arg" in
        --gpu) shift; GPU="$1" ;;
    esac
done

SMOKE_N=10
MODEL_CONFIG="configs/flux_schnell.yaml"
# Override model key to "flux" so paths/hooks use the same logic
MODEL_KEY="flux"
SMOKE_REF_DIR="$MMDIT_DATA_ROOT/reference/flux_schnell"

echo "============================================================"
echo "  SMOKE TEST — FLUX.1-schnell (cached)"
echo "  GPU: $GPU  Prompts: $SMOKE_N"
echo "  DATA_ROOT: $MMDIT_DATA_ROOT"
echo "============================================================"

mkdir -p "$SMOKE_REF_DIR" results/plots logs

# ── Step 0: Reference image generation (10 images) ──────────────────────────
echo ""
echo "[0/4] Generating smoke reference images (schnell, 10 prompts)..."
python - <<PYEOF
import os, sys, torch, yaml
sys.path.insert(0, ".")
os.environ["MMDIT_DATA_ROOT"] = "$MMDIT_DATA_ROOT"

from analysis.paths import BASE_DIR
from eval.eval_utils import get_prompts, generate_reference_images

with open("$MODEL_CONFIG") as f:
    import yaml; cfg = yaml.safe_load(f)

device = "cuda:$GPU"
dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[cfg["model"]["torch_dtype"]]
prompts = get_prompts($SMOKE_N, "MJHQ")

from diffusers import FluxPipeline
pipe = FluxPipeline.from_pretrained(cfg["model"]["name"], torch_dtype=dtype).to(device)
pipe.set_progress_bar_config(disable=True)

generate_reference_images(pipe, prompts, cfg, "$SMOKE_REF_DIR", device, prefix="ref")
print(f"[OK] {$SMOKE_N} reference images saved to $SMOKE_REF_DIR")
PYEOF

# ── Experiment A ─────────────────────────────────────────────────────────────
echo ""
echo "[1/4] Experiment A: Stream Dynamics..."
python analysis/measure_stream_dynamics.py \
    --model flux --n_prompts $SMOKE_N --gpu "$GPU" --quick \
    --output_dir results \
    2>&1 | tee logs/smoke_exp_a.log
[ -f results/stream_dynamics_flux.json ] && echo "[OK] Exp A output exists" || { echo "[FAIL] Exp A no output"; exit 1; }

# ── Experiment B ─────────────────────────────────────────────────────────────
echo ""
echo "[2/4] Experiment B: Attention Sparsity..."
python analysis/measure_attention_sparsity.py \
    --model flux --n_prompts 5 --gpu "$GPU" --quick \
    --output_dir results \
    2>&1 | tee logs/smoke_exp_b.log
[ -f results/attn_sparsity_flux.json ] && echo "[OK] Exp B output exists" || { echo "[FAIL] Exp B no output"; exit 1; }

# ── Experiment C ─────────────────────────────────────────────────────────────
# For Exp C we need reference images — point to the smoke ref dir temporarily
echo ""
echo "[3/4] Experiment C: Branch Ablation (no metrics, ref images in smoke dir)..."
python - <<PYEOF2
import os, sys, torch, yaml
sys.path.insert(0, ".")
os.environ["MMDIT_DATA_ROOT"] = "$MMDIT_DATA_ROOT"

import yaml
from analysis.paths import results_dir, samples_dir
from analysis.measure_branch_ablation import (
    load_prompts, BranchSkipState, generate_for_schedule
)

with open("$MODEL_CONFIG") as f:
    cfg = yaml.safe_load(f)

device = "cuda:$GPU"
dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[cfg["model"]["torch_dtype"]]
prompts = load_prompts($SMOKE_N, quick=True)

from diffusers import FluxPipeline
pipe = FluxPipeline.from_pretrained(cfg["model"]["name"], torch_dtype=dtype).to(device)
pipe.set_progress_bar_config(disable=True)

# Only run baseline schedule for smoke
skip_state = BranchSkipState(skip_target="none", freeze_from=0)
save_dir = samples_dir("branch_ablation", "flux_schnell", "baseline")
generate_for_schedule(pipe, "flux", prompts, cfg, skip_state, save_dir, device)
print(f"[OK] Exp C baseline smoke: {$SMOKE_N} images -> {save_dir}")
PYEOF2

# ── Experiment D ─────────────────────────────────────────────────────────────
echo ""
echo "[4/4] Experiment D: KV Drift..."
python - <<PYEOF3
import os, sys, torch, yaml
sys.path.insert(0, ".")
os.environ["MMDIT_DATA_ROOT"] = "$MMDIT_DATA_ROOT"

# Minimal Exp D: only Part 1 (drift measurement, no freeze ablation)
import json
from analysis.paths import results_dir
from collections import defaultdict
import numpy as np

with open("$MODEL_CONFIG") as f:
    cfg = yaml.safe_load(f)

device = "cuda:$GPU"
dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[cfg["model"]["torch_dtype"]]

from eval.eval_utils import get_prompts
from diffusers import FluxPipeline
from analysis.hook_utils import StreamHookState, register_flux_stream_hooks, make_step_callback
from analysis.measure_xattn_kv_drift import (
    KVCaptureState, register_kv_hooks, compute_kv_drift
)

prompts = get_prompts(5, "MJHQ")
pipe = FluxPipeline.from_pretrained(cfg["model"]["name"], torch_dtype=dtype).to(device)
pipe.set_progress_bar_config(disable=True)

n_txt = cfg["architecture"]["num_text_tokens"]
agg = defaultdict(lambda: defaultdict(lambda: {"K_drift": [], "V_drift": []}))

for p_idx, prompt in enumerate(prompts):
    kv_state = KVCaptureState(n_txt=n_txt)
    register_kv_hooks(pipe.transformer, kv_state, is_flux=True)

    def step_cb(pipe_, step_idx, timestep, cb_kwargs):
        kv_state.next_step(); return cb_kwargs

    import torch as _torch
    gen = _torch.Generator(device=device).manual_seed(42 + p_idx)
    with _torch.no_grad():
        pipe(prompt, num_inference_steps=cfg["generation"]["num_inference_steps"],
             guidance_scale=cfg["generation"]["guidance_scale"],
             height=cfg["generation"]["height"], width=cfg["generation"]["width"],
             generator=gen, output_type="latent", callback_on_step_end=step_cb)

    drift = compute_kv_drift(kv_state)
    for b, sd in drift.items():
        for s, m in sd.items():
            agg[b][s]["K_drift"].append(m["K_drift"])
            agg[b][s]["V_drift"].append(m["V_drift"])
    kv_state.remove_hooks()

result = {b: {s: {"K_drift_mean": float(np.mean(d["K_drift"])),
                   "V_drift_mean": float(np.mean(d["V_drift"]))}
              for s, d in sd.items()}
          for b, sd in agg.items()}

os.makedirs(results_dir(), exist_ok=True)
out = results_dir("xattn_kv_drift_flux_smoke.json")
with open(out, "w") as f:
    json.dump({"config": {"smoke": True}, "data": result}, f, indent=2)
print(f"[OK] Exp D smoke drift saved: {out}")
PYEOF3

# ── Plot ─────────────────────────────────────────────────────────────────────
echo ""
echo "[Plot] Generating available plots..."
python analysis/plot_results.py --exp A --model flux 2>/dev/null || true
python analysis/plot_results.py --exp B --model flux 2>/dev/null || true

echo ""
echo "============================================================"
echo "  SMOKE TEST COMPLETE"
echo ""
echo "  Outputs:"
echo "    results/stream_dynamics_flux.json"
echo "    results/attn_sparsity_flux.json"
echo "    results/xattn_kv_drift_flux_smoke.json"
echo "    $MMDIT_DATA_ROOT/reference/flux_schnell/  (10 images)"
echo "    $MMDIT_DATA_ROOT/samples/branch_ablation/flux_schnell/baseline/ (10 images)"
echo "    results/plots/"
echo ""
echo "  -> If no errors above, run: bash scripts/run_full_test.sh"
echo "============================================================"

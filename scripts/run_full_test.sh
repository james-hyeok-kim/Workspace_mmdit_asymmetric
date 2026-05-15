#!/usr/bin/env bash
# run_full_test.sh — Full Step 1 analysis: FLUX.1-dev + SD3-medium, 350 prompts each.
#
# Prerequisites:
#   bash scripts/run_smoke_test.sh   (must pass cleanly)
#
# Image storage (all under DATA_ROOT = /data/jameskimh/mmdit_asymmetric/):
#   reference/{flux,sd3}/ref_{i}.png        — baseline images (reused if exist)
#   samples/branch_ablation/{model}/{sched}/sample_{i}.png
#   samples/freeze_kv/{model}/{sched}/sample_{i}.png
#
# The script skips already-generated images, so:
#   - If smoke test generated 10 FLUX schnell refs, those are separate.
#   - Full test generates up to 350 FLUX dev + 350 SD3 refs (skips existing).
#   - First run: 350+350=700 images total (no prior refs in flux/sd3 dirs).
#
# Usage:
#   bash scripts/run_full_test.sh [--gpu N] [--skip-ref] [--exp A|B|C|D|all]

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="$(pwd):$PYTHONPATH"
export MMDIT_DATA_ROOT="/data/jameskimh/mmdit_asymmetric"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

GPU=${GPU:-0}
SKIP_REF=0
EXP="all"

for arg in "$@"; do
    case "$arg" in
        --gpu)     shift; GPU="$1" ;;
        --skip-ref) SKIP_REF=1 ;;
        --exp)     shift; EXP="$1" ;;
    esac
done

N_PROMPTS=350
MODELS=("flux" "sd3")
FLUX_MODEL="black-forest-labs/FLUX.1-dev"
SD3_MODEL="stabilityai/stable-diffusion-3-medium-diffusers"
FLUX_CACHE="/data/jameskimh/flux_pretrained/FLUX.1-dev"
SD3_CACHE="/data/jameskimh/flux_pretrained/SD3-medium"

echo "============================================================"
echo "  FULL TEST — FLUX.1-dev + SD3-medium"
echo "  GPU: $GPU  Prompts: $N_PROMPTS each"
echo "  DATA_ROOT: $MMDIT_DATA_ROOT"
echo "  Experiments: $EXP"
echo "============================================================"

mkdir -p results/plots logs "$FLUX_CACHE" "$SD3_CACHE"

# ── Model download (cache to /data/jameskimh/flux_pretrained/) ───────────────
download_if_needed() {
    local model_id="$1"
    local local_dir="$2"
    if [ -d "$local_dir/transformer" ] || [ -d "$local_dir/snapshots" ]; then
        echo "  [Cache hit] $model_id already at $local_dir"
        return 0
    fi
    echo "  [Download] $model_id -> $local_dir"
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='$model_id',
    local_dir='$local_dir',
    ignore_patterns=['*.gguf', '*.safetensors.index.json'],
)
print('  [Done] Downloaded $model_id')
"
}

echo ""
echo "[Setup] Checking model caches..."
download_if_needed "$FLUX_MODEL" "$FLUX_CACHE"
download_if_needed "$SD3_MODEL" "$SD3_CACHE"

# Update configs to use local paths if downloaded
python - <<PYEOF
import yaml, os
for cfg_file, local_dir in [
    ("configs/flux_dev.yaml", "$FLUX_CACHE"),
    ("configs/sd3_medium.yaml", "$SD3_CACHE"),
]:
    with open(cfg_file) as f:
        cfg = yaml.safe_load(f)
    # Use local path if it exists and has the transformer directory
    if os.path.isdir(os.path.join(local_dir, "transformer")):
        cfg["model"]["name"] = local_dir
        with open(cfg_file, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        print(f"  Updated {cfg_file} -> {local_dir}")
    else:
        print(f"  Using HF hub for {cfg_file}: {cfg['model']['name']}")
PYEOF

# ── Reference image generation ───────────────────────────────────────────────
if [ "$SKIP_REF" -eq 0 ]; then
    echo ""
    echo "[REF] Generating reference images (skips existing)..."

    for model in "${MODELS[@]}"; do
        cfg_file="configs/${model}_dev.yaml"
        [ "$model" = "sd3" ] && cfg_file="configs/sd3_medium.yaml"
        ref_path="$MMDIT_DATA_ROOT/reference/$model"
        existing=$(ls "$ref_path"/ref_*.png 2>/dev/null | wc -l)
        need=$(( $N_PROMPTS - existing ))
        echo "  [$model] existing=$existing need=$need"
        if [ "$need" -le 0 ]; then
            echo "  [$model] All $N_PROMPTS refs already exist, skipping."
            continue
        fi

        python - <<PYEOF2
import os, sys, torch, yaml
sys.path.insert(0, ".")
os.environ["MMDIT_DATA_ROOT"] = "$MMDIT_DATA_ROOT"

from eval.eval_utils import get_mixed_prompts, generate_reference_images
from analysis.paths import ref_dir

with open("$cfg_file") as f:
    cfg = yaml.safe_load(f)

device = "cuda:$GPU"
dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[cfg["model"]["torch_dtype"]]
prompts = get_mixed_prompts($N_PROMPTS, cfg["eval"]["prompt_sources"])

model_name = cfg["model"]["name"]
if "FLUX" in model_name or "flux" in model_name.lower() or "FLUX" in model_name.upper():
    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained(model_name, torch_dtype=dtype).to(device)
else:
    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(model_name, torch_dtype=dtype).to(device)
pipe.set_progress_bar_config(disable=True)

save_dir = ref_dir("$model")
print(f"Generating reference images -> {save_dir}")
generate_reference_images(pipe, prompts, cfg, save_dir, device, prefix="ref")
print(f"[OK] $model reference images complete")
PYEOF2
    done
fi

# ── Analysis experiments ──────────────────────────────────────────────────────

run_exp() {
    local exp_id="$1"
    local model="$2"
    local cfg_file="configs/${model}_dev.yaml"
    [ "$model" = "sd3" ] && cfg_file="configs/sd3_medium.yaml"

    echo ""
    echo "──────────────────────────────────────────────────────────"
    echo "  Experiment $exp_id — $model"
    echo "──────────────────────────────────────────────────────────"

    case "$exp_id" in
        A)
            python analysis/measure_stream_dynamics.py \
                --model "$model" --n_prompts 50 --gpu "$GPU" \
                --output_dir results \
                2>&1 | tee logs/full_exp_a_${model}.log
            ;;
        B)
            python analysis/measure_attention_sparsity.py \
                --model "$model" --n_prompts 30 --gpu "$GPU" \
                --output_dir results \
                2>&1 | tee logs/full_exp_b_${model}.log
            ;;
        C)
            python analysis/measure_branch_ablation.py \
                --model "$model" --n_prompts $N_PROMPTS --gpu "$GPU" \
                2>&1 | tee logs/full_exp_c_${model}.log
            ;;
        D)
            python analysis/measure_xattn_kv_drift.py \
                --model "$model" --n_prompts 50 --gpu "$GPU" \
                2>&1 | tee logs/full_exp_d_${model}.log
            ;;
    esac
}

EXPS=()
if [ "$EXP" = "all" ]; then EXPS=("A" "B" "C" "D")
else EXPS=("$EXP"); fi

for model in "${MODELS[@]}"; do
    for exp in "${EXPS[@]}"; do
        run_exp "$exp" "$model"
    done
done

# ── Plots ────────────────────────────────────────────────────────────────────
echo ""
echo "[Plot] Generating all plots..."
python analysis/plot_results.py --exp all --model both 2>&1 | tee logs/full_plots.log

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  FULL TEST COMPLETE"
echo ""
echo "  Results:"
for model in "${MODELS[@]}"; do
    for json_file in results/stream_dynamics_${model}.json results/attn_sparsity_${model}.json results/xattn_kv_drift_${model}.json; do
        if [ -f "$json_file" ]; then
            verdict=$(python -c "import json; d=json.load(open('$json_file')); print(d.get('summary',{}).get('verdict','N/A'))" 2>/dev/null || echo "N/A")
            printf "    %-45s  %s\n" "$(basename $json_file)" "$verdict"
        fi
    done
done
echo ""
echo "  Image artifacts: $MMDIT_DATA_ROOT/"
echo "  JSON/CSV:        results/"
echo "  Plots:           results/plots/"
echo "============================================================"

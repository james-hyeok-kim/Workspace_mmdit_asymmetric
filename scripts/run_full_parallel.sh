#!/usr/bin/env bash
# run_full_parallel.sh — Full Step 1 analysis with maximum GPU utilization.
#
# Strategy:
#   GPU 1 : FLUX.1-dev  (reference gen + Exp A/B/C/D)
#   GPU 2 : SD3-medium  (reference gen + Exp A/B/C/D)
#   GPU 1&2 run simultaneously via background subprocesses.
#   GPU 0 is left for other workloads (Lerobot etc).
#
# Batch sizes:
#   Reference gen  : batch_size=8  (no hooks, fits B200 183GB easily)
#   Exp A (dynamics): batch_size=1 (hook state is per-prompt)
#   Exp B (sparsity): batch_size=1 (attention map capture is per-prompt)
#   Exp C (ablation): batch_size=4 (no hooks during generation)
#   Exp D (KV drift): batch_size=1 (KV capture is per-prompt)
#
# Image reuse: skips already-generated files (smoke test images are separate dirs).
#
# Usage:
#   bash scripts/run_full_parallel.sh
#   bash scripts/run_full_parallel.sh --exp A        # only Exp A
#   bash scripts/run_full_parallel.sh --skip-ref     # skip reference gen
#   bash scripts/run_full_parallel.sh --gpu1 1 --gpu2 2

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="$(pwd):$PYTHONPATH"
export MMDIT_DATA_ROOT="/data/jameskimh/mmdit_asymmetric"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

GPU1=1   # FLUX primary
GPU2=2   # SD3
GPU3=3   # FLUX secondary (Exp C split)
SKIP_REF=0
EXP="all"
N=350
REF_BATCH=8   # batch size for reference generation
ABL_BATCH=4   # batch size for branch ablation (Exp C)

for arg in "$@"; do
    case "$arg" in
        --gpu1)    shift; GPU1="$1" ;;
        --gpu2)    shift; GPU2="$1" ;;
        --skip-ref) SKIP_REF=1 ;;
        --exp)     shift; EXP="$1" ;;
        --n)       shift; N="$1" ;;
    esac
done

FLUX_LOCAL="/data/jameskimh/flux_pretrained/FLUX.1-dev"
SD3_LOCAL="/data/jameskimh/flux_pretrained/SD3-medium"

mkdir -p results/plots logs \
    "$MMDIT_DATA_ROOT/reference/flux" \
    "$MMDIT_DATA_ROOT/reference/sd3" \
    "$MMDIT_DATA_ROOT/samples/branch_ablation/flux" \
    "$MMDIT_DATA_ROOT/samples/branch_ablation/sd3" \
    "$MMDIT_DATA_ROOT/samples/freeze_kv/flux" \
    "$MMDIT_DATA_ROOT/samples/freeze_kv/sd3"

echo "============================================================"
echo "  FULL TEST (Parallel Multi-GPU)"
echo "  GPU1=$GPU1 (FLUX.1-dev)   GPU2=$GPU2 (SD3-medium)"
echo "  N=$N prompts  ref_batch=$REF_BATCH  abl_batch=$ABL_BATCH"
echo "  DATA_ROOT=$MMDIT_DATA_ROOT"
echo "  Experiments: $EXP"
echo "============================================================"

# ── Model download ───────────────────────────────────────────────────────────
download_model() {
    local repo="$1"; local dest="$2"
    if [ -d "$dest/transformer" ] || [ -f "$dest/model_index.json" ]; then
        echo "  [cache] $repo"
        return
    fi
    echo "  [download] $repo -> $dest"
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$repo', local_dir='$dest',
                  ignore_patterns=['*.gguf','flax_model*','tf_model*'])
print('  [done] $repo')
"
}

echo ""
echo "[Setup] Downloading models (if needed)..."
download_model "black-forest-labs/FLUX.1-dev" "$FLUX_LOCAL" &
DPID1=$!
download_model "stabilityai/stable-diffusion-3-medium-diffusers" "$SD3_LOCAL" &
DPID2=$!
wait $DPID1 $DPID2
echo "  Models ready."

# Point configs to local paths
python - <<PYEOF
import yaml, os
pairs = [("configs/flux_dev.yaml", "$FLUX_LOCAL"),
         ("configs/sd3_medium.yaml", "$SD3_LOCAL")]
for cfg_file, local in pairs:
    with open(cfg_file) as f:
        cfg = yaml.safe_load(f)
    name = local if (os.path.isdir(os.path.join(local,"transformer")) or
                     os.path.isfile(os.path.join(local,"model_index.json"))) \
           else cfg["model"]["name"]
    cfg["model"]["name"] = name
    with open(cfg_file, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    print(f"  {cfg_file}: {name}")
PYEOF

# ── Reference generation (parallel) ─────────────────────────────────────────
gen_ref() {
    local model="$1"; local gpu="$2"
    local cfg="configs/${model}_dev.yaml"
    [ "$model" = "sd3" ] && cfg="configs/sd3_medium.yaml"
    local ref_path="$MMDIT_DATA_ROOT/reference/$model"
    local existing; existing=$(ls "$ref_path"/ref_*.png 2>/dev/null | wc -l)
    if [ "$existing" -ge "$N" ]; then
        echo "  [$model] All $N refs exist, skipping ref gen."
        return
    fi
    echo "  [$model] Generating refs on GPU $gpu (existing=$existing, batch=$REF_BATCH)..."
    python - <<PYEOF2 2>&1 | tee logs/full_ref_${model}.log
import os, sys, torch, yaml
sys.path.insert(0, ".")
os.environ["MMDIT_DATA_ROOT"] = "$MMDIT_DATA_ROOT"
from eval.eval_utils import get_mixed_prompts, generate_reference_images
from analysis.paths import ref_dir

with open("$cfg") as f:
    cfg = yaml.safe_load(f)

device = "cuda:$gpu"
dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[cfg["model"]["torch_dtype"]]
prompts = get_mixed_prompts($N, cfg["eval"]["prompt_sources"])

name = cfg["model"]["name"]
if "FLUX" in name or "flux" in name.lower():
    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained(name, torch_dtype=dtype).to(device)
else:
    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(name, torch_dtype=dtype).to(device)
pipe.set_progress_bar_config(disable=True)

generate_reference_images(pipe, prompts, cfg, ref_dir("$model"), device,
                          prefix="ref", batch_size=$REF_BATCH)
PYEOF2
}

if [ "$SKIP_REF" -eq 0 ]; then
    echo ""
    echo "[REF] Generating reference images in parallel..."
    gen_ref flux "$GPU1" &
    RPID1=$!
    gen_ref sd3  "$GPU2" &
    RPID2=$!
    wait $RPID1 $RPID2
    echo "  [REF] Both models done."
fi

# ── Experiment runner (one model per GPU) ────────────────────────────────────

run_exp_model() {
    local exp="$1"; local model="$2"; local gpu="$3"
    local cfg="configs/${model}_dev.yaml"
    [ "$model" = "sd3" ] && cfg="configs/sd3_medium.yaml"

    echo "  [Exp $exp | $model | GPU $gpu] starting..." >&2

    case "$exp" in
        A)
            python analysis/measure_stream_dynamics.py \
                --model "$model" --n_prompts 50 --gpu "$gpu" \
                --output_dir results \
                2>&1 | tee logs/full_a_${model}.log
            ;;
        B)
            python analysis/measure_attention_sparsity.py \
                --model "$model" --n_prompts 30 --gpu "$gpu" \
                --output_dir results \
                2>&1 | tee logs/full_b_${model}.log
            ;;
        C)
            if [ "$model" = "flux" ] && [ "$GPU3" != "" ]; then
                # Split FLUX Exp C across GPU1 and GPU3 (first/second half)
                HALF=$(( N / 2 ))
                MMDIT_ABL_BATCH=$ABL_BATCH python analysis/measure_branch_ablation.py \
                    --model flux --n_prompts $HALF --prompt_start 0 --gpu "$gpu" \
                    2>&1 | tee logs/full_c_flux_gpu${gpu}.log &
                CPID1=$!
                MMDIT_ABL_BATCH=$ABL_BATCH python analysis/measure_branch_ablation.py \
                    --model flux --n_prompts $(( N - HALF )) --prompt_start $HALF --gpu "$GPU3" \
                    2>&1 | tee logs/full_c_flux_gpu${GPU3}.log &
                CPID2=$!
                wait $CPID1 $CPID2
            else
                MMDIT_ABL_BATCH=$ABL_BATCH python analysis/measure_branch_ablation.py \
                    --model "$model" --n_prompts "$N" --gpu "$gpu" \
                    2>&1 | tee logs/full_c_${model}.log
            fi
            ;;
        D)
            python analysis/measure_xattn_kv_drift.py \
                --model "$model" --n_prompts 50 --gpu "$gpu" \
                2>&1 | tee logs/full_d_${model}.log
            ;;
    esac
    echo "  [Exp $exp | $model] DONE" >&2
}

EXPS=()
if [ "$EXP" = "all" ]; then EXPS=("A" "B" "C" "D")
else EXPS=("$EXP"); fi

for exp in "${EXPS[@]}"; do
    echo ""
    echo "──────────────────────────────────────────────────────────"
    echo "  Experiment $exp  [FLUX=GPU$GPU1 || SD3=GPU$GPU2]"
    echo "──────────────────────────────────────────────────────────"
    run_exp_model "$exp" flux "$GPU1" &
    PID1=$!
    run_exp_model "$exp" sd3  "$GPU2" &
    PID2=$!
    wait $PID1 $PID2
    echo "  Experiment $exp complete (both models)."
done

# ── Plots ────────────────────────────────────────────────────────────────────
echo ""
echo "[Plot] Generating all result plots..."
python analysis/plot_results.py --exp all --model both 2>&1 | tee logs/full_plots.log

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  FULL TEST COMPLETE"
echo ""
printf "  %-48s  %s\n" "File" "Verdict"
printf "  %-48s  %s\n" "----" "-------"
for model in flux sd3; do
    for f in results/stream_dynamics_${model}.json \
              results/attn_sparsity_${model}.json \
              results/xattn_kv_drift_${model}.json; do
        [ -f "$f" ] || continue
        v=$(python -c "import json; d=json.load(open('$f')); \
            print(d.get('summary',{}).get('verdict','N/A'))" 2>/dev/null || echo "N/A")
        printf "  %-48s  %s\n" "$(basename $f)" "$v"
    done
    [ -f "results/branch_ablation_${model}.csv" ] && \
        printf "  %-48s  %s\n" "branch_ablation_${model}.csv" "(see CSV)"
done
echo ""
echo "  Images : $MMDIT_DATA_ROOT/"
echo "  Results: results/  Plots: results/plots/"
echo "============================================================"

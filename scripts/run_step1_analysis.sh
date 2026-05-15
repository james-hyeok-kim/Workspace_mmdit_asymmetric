#!/usr/bin/env bash
# run_step1_analysis.sh — Run Experiments A–D for both FLUX and SD3.
#
# Prerequisites:
#   - Reference images generated (run_reference_gen.sh)
#   - Models downloaded (HuggingFace cache or local paths in configs/)
#
# Usage:
#   bash scripts/run_step1_analysis.sh               # full run (1-2 days)
#   bash scripts/run_step1_analysis.sh --quick       # smoke test (~5 min)
#   bash scripts/run_step1_analysis.sh --exp A       # run only Experiment A
#   bash scripts/run_step1_analysis.sh --model flux  # FLUX only

set -euo pipefail
cd "$(dirname "$0")/.."

GPU=${GPU:-0}
QUICK=""
EXP="all"
MODEL="both"

for arg in "$@"; do
    case "$arg" in
        --quick)  QUICK="--quick" ;;
        --exp)    shift; EXP="$1" ;;
        --model)  shift; MODEL="$1" ;;
    esac
done

N_DYN=50;  N_SPAR=30;  N_ABL=350;  N_KV=50
if [ -n "$QUICK" ]; then
    N_DYN=10; N_SPAR=5; N_ABL=10; N_KV=5
    echo "[Quick mode] Using reduced prompt counts."
fi

run_exp() {
    local exp_id="$1"
    local model="$2"
    echo ""
    echo "──────────────────────────────────────────────────────────"
    echo "  Experiment $exp_id — $model"
    echo "──────────────────────────────────────────────────────────"

    case "$exp_id" in
        A)
            python analysis/measure_stream_dynamics.py \
                --model "$model" --n_prompts "$N_DYN" --gpu "$GPU" $QUICK
            ;;
        B)
            python analysis/measure_attention_sparsity.py \
                --model "$model" --n_prompts "$N_SPAR" --gpu "$GPU" $QUICK
            ;;
        C)
            python analysis/measure_branch_ablation.py \
                --model "$model" --n_prompts "$N_ABL" --gpu "$GPU" $QUICK
            ;;
        D)
            python analysis/measure_xattn_kv_drift.py \
                --model "$model" --n_prompts "$N_KV" --gpu "$GPU" $QUICK
            ;;
    esac
}

MODELS=()
if [ "$MODEL" = "both" ]; then MODELS=("flux" "sd3")
else MODELS=("$MODEL"); fi

EXPS=()
if [ "$EXP" = "all" ]; then EXPS=("A" "B" "C" "D")
else EXPS=("$EXP"); fi

echo "============================================================"
echo "  Step 1 Analysis — Asymmetric MM-DiT"
echo "  Experiments: ${EXPS[*]}"
echo "  Models:      ${MODELS[*]}"
echo "  GPU:         $GPU"
echo "============================================================"

for m in "${MODELS[@]}"; do
    for e in "${EXPS[@]}"; do
        run_exp "$e" "$m"
    done
done

echo ""
echo "──────────────────────────────────────────────────────────"
echo "  Generating plots..."
echo "──────────────────────────────────────────────────────────"
python analysis/plot_results.py --exp all --model "${MODEL}"

echo ""
echo "============================================================"
echo "  All done. Results saved in results/"
echo "  Plots saved in results/plots/"
echo "============================================================"

# Print summary verdicts
echo ""
echo "  === Verdict Summary ==="
for m in "${MODELS[@]}"; do
    for json_file in results/stream_dynamics_${m}.json results/attn_sparsity_${m}.json results/xattn_kv_drift_${m}.json; do
        if [ -f "$json_file" ]; then
            verdict=$(python -c "import json; d=json.load(open('$json_file')); print(d.get('summary',{}).get('verdict','N/A'))" 2>/dev/null || echo "N/A")
            echo "  $(basename $json_file .json): $verdict"
        fi
    done
    if [ -f "results/branch_ablation_${m}.csv" ]; then
        echo "  branch_ablation_${m}: see results/branch_ablation_${m}.csv"
    fi
done

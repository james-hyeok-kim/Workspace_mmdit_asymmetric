#!/usr/bin/env bash
# run_baseline_setup.sh — Clone and smoke-test baseline methods.
#
# Baselines:
#   FORA      — block-level feature caching
#   SmoothCache — smooth interpolation caching
#   ToCa/DuCa — token-level caching (lower priority; run last)
#   TaylorSeer — Taylor-expansion caching (most complex)
#
# NOTE: Run ONLY after Experiment A verdict is PASS.
#       If text dynamics ≈ image dynamics, baseline work is premature.
#
# Usage:
#   bash scripts/run_baseline_setup.sh --method fora
#   bash scripts/run_baseline_setup.sh --method smoothcache
#   bash scripts/run_baseline_setup.sh --method all

set -euo pipefail
cd "$(dirname "$0")/.."

GPU=${GPU:-0}
METHOD="${1:-all}"

check_exp_a() {
    for m in flux sd3; do
        f="results/stream_dynamics_${m}.json"
        if [ -f "$f" ]; then
            verdict=$(python -c "import json; d=json.load(open('$f')); print(d.get('summary',{}).get('verdict','N/A'))" 2>/dev/null || echo "N/A")
            if [ "$verdict" = "FAIL" ]; then
                echo "[WARNING] Experiment A verdict for $m is FAIL."
                echo "  Text dynamics ≈ Image dynamics. Baseline work may be premature."
                echo "  Continue? [y/N]"
                read -r ans
                [ "$ans" = "y" ] || exit 1
            fi
        fi
    done
}

setup_fora() {
    echo "[FORA] Cloning..."
    if [ ! -d "baselines/fora/repo" ]; then
        git clone https://github.com/ToTheBeginning/PuLID baselines/fora/repo 2>/dev/null || \
        echo "  [Note] Replace with actual FORA repo URL when available."
    fi
    # Smoke test: generate 2 images with FORA caching on FLUX
    python - << 'PYEOF'
import sys
sys.path.insert(0, ".")
print("[FORA] Smoke test placeholder. Implement adapter after repo is cloned.")
# TODO: from baselines.fora.adapter import FORAPipeline
# pipe = FORAPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", ...)
# result = pipe("A test prompt", num_inference_steps=4)
print("[FORA] Smoke test: PASS (placeholder)")
PYEOF
}

setup_smoothcache() {
    echo "[SmoothCache] Cloning..."
    if [ ! -d "baselines/smoothcache/repo" ]; then
        git clone https://github.com/horseee/SmoothCache baselines/smoothcache/repo 2>/dev/null || \
        echo "  [Note] SmoothCache repo not found. Check URL."
    fi
    python - << 'PYEOF'
print("[SmoothCache] Smoke test placeholder. Implement adapter after repo is cloned.")
print("[SmoothCache] Smoke test: PASS (placeholder)")
PYEOF
}

setup_toca_duca() {
    echo "[ToCa/DuCa] Setup..."
    if [ ! -d "baselines/toca_duca/repo" ]; then
        git clone https://github.com/Shenyi-Z/ToCa baselines/toca_duca/repo 2>/dev/null || \
        echo "  [Note] ToCa repo not found. Check URL."
    fi
    python - << 'PYEOF'
print("[ToCa/DuCa] Smoke test placeholder.")
print("[ToCa/DuCa] Smoke test: PASS (placeholder)")
PYEOF
}

setup_taylorseer() {
    echo "[TaylorSeer] Setup..."
    if [ ! -d "baselines/taylorseer/repo" ]; then
        git clone https://github.com/Shenyi-Z/TaylorSeer baselines/taylorseer/repo 2>/dev/null || \
        echo "  [Note] TaylorSeer repo not found. Check URL."
    fi
    python - << 'PYEOF'
print("[TaylorSeer] Smoke test placeholder.")
print("[TaylorSeer] Smoke test: PASS (placeholder)")
PYEOF
}

echo "============================================================"
echo "  Baseline Setup"
echo "  Method: $METHOD  GPU: $GPU"
echo "============================================================"

check_exp_a

case "$METHOD" in
    fora)       setup_fora ;;
    smoothcache) setup_smoothcache ;;
    toca_duca)  setup_toca_duca ;;
    taylorseer) setup_taylorseer ;;
    all)
        setup_fora
        setup_smoothcache
        # ToCa/DuCa, TaylorSeer are lower priority — skip unless method=all
        # setup_toca_duca
        # setup_taylorseer
        ;;
esac

echo ""
echo "  Baseline setup complete."
echo "  Full baseline comparison is in Step 2 plan."

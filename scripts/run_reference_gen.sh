#!/usr/bin/env bash
# run_reference_gen.sh — Generate reference (no-caching) images for FLUX and SD3.
# Must be run before branch ablation (Exp C) and KV freeze ablation (Exp D).
#
# Usage:
#   bash scripts/run_reference_gen.sh           # full 350 prompts, both models
#   bash scripts/run_reference_gen.sh --quick   # 10 prompts, FLUX only (smoke test)

set -euo pipefail
cd "$(dirname "$0")/.."

GPU=${GPU:-0}
QUICK=${1:-""}

echo "============================================================"
echo "  Reference Image Generation"
echo "  GPU: $GPU"
echo "============================================================"

cat > /tmp/gen_reference.py << 'EOF'
import os, sys, json, argparse, torch, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["flux", "sd3"])
parser.add_argument("--n_prompts", type=int, default=350)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--quick", action="store_true")
args = parser.parse_args()

BASE_DIR = "/home/jovyan/workspace/Workspace_mmdit_asymmetric"
sys.path.insert(0, BASE_DIR)

cfg_path = os.path.join(BASE_DIR, "configs",
                        "flux_dev.yaml" if args.model == "flux" else "sd3_medium.yaml")
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

if args.quick:
    cfg["generation"]["num_inference_steps"] = 10
    args.n_prompts = min(args.n_prompts, 10)

device = f"cuda:{args.gpu}"
torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[cfg["model"]["torch_dtype"]]

from eval.eval_utils import get_mixed_prompts, generate_reference_images

prompts = get_mixed_prompts(args.n_prompts, cfg["eval"]["prompt_sources"])

name = cfg["model"]["name"]
if "FLUX" in name or "flux" in name.lower():
    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained(name, torch_dtype=torch_dtype).to(device)
else:
    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(name, torch_dtype=torch_dtype).to(device)
pipe.set_progress_bar_config(disable=True)

save_dir = os.path.join(BASE_DIR, "eval", "reference", args.model)
print(f"Generating {len(prompts)} reference images → {save_dir}")
generate_reference_images(pipe, prompts, cfg, save_dir, device, prefix="ref")
print("Done.")
EOF

if [ "$QUICK" = "--quick" ]; then
    echo "[Quick mode] FLUX only, 10 prompts"
    python /tmp/gen_reference.py --model flux --n_prompts 10 --gpu "$GPU" --quick
else
    echo "[Full mode] FLUX + SD3, 350 prompts each"
    python /tmp/gen_reference.py --model flux --n_prompts 350 --gpu "$GPU"
    python /tmp/gen_reference.py --model sd3  --n_prompts 350 --gpu "$GPU"
fi

echo "============================================================"
echo "  Reference generation complete."
echo "============================================================"

"""
plot_results.py — Generate analysis plots for all 4 experiments.

Reads results/*.json and results/*.csv, saves heatmaps/lineplots to results/plots/.

Usage:
  python analysis/plot_results.py --all
  python analysis/plot_results.py --exp A --model flux
  python analysis/plot_results.py --exp B --model flux
  python analysis/plot_results.py --exp C --model flux
  python analysis/plot_results.py --exp D --model flux
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import csv
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")


# ---------------------------------------------------------------------------
# Experiment A: Stream Dynamics Heatmap
# ---------------------------------------------------------------------------

def plot_dynamics(model: str):
    path = os.path.join(RESULTS_DIR, f"stream_dynamics_{model}.json")
    if not os.path.exists(path):
        print(f"[Skip] {path} not found")
        return

    with open(path) as f:
        data = json.load(f)

    block_data = data["data"]
    blocks = sorted(int(k) for k in block_data.keys())
    steps = sorted(int(s) for s in block_data[str(blocks[0])].keys())

    text_mat  = np.zeros((len(blocks), len(steps)))
    image_mat = np.zeros((len(blocks), len(steps)))

    for bi, block in enumerate(blocks):
        for si, step in enumerate(steps):
            entry = block_data[str(block)].get(str(step), {})
            text_mat[bi, si]  = entry.get("text_mean") or 0.0
            image_mat[bi, si] = entry.get("image_mean") or 0.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    summary = data.get("summary", {})
    fig.suptitle(
        f"Experiment A: Stream Dynamics — {model.upper()}\n"
        f"Ratio (text/img): {summary.get('text_image_ratio', '?'):.4f}  "
        f"Verdict: {summary.get('verdict', '?')}",
        fontsize=12,
    )

    vmax = max(text_mat.max(), image_mat.max())
    norm = mcolors.LogNorm(vmin=1e-5, vmax=vmax + 1e-8)

    for ax, mat, title in zip(axes, [text_mat, image_mat], ["Text stream", "Image stream"]):
        im = ax.imshow(mat, aspect="auto", norm=norm, cmap="viridis", origin="lower")
        ax.set_xlabel("Denoising step t")
        ax.set_ylabel("Block index")
        ax.set_title(title)
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels(steps, fontsize=7, rotation=45)
        ax.set_yticks(range(0, len(blocks), max(1, len(blocks) // 10)))
        plt.colorbar(im, ax=ax, label="Relative L2 change")

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f"dynamics_heatmap_{model}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[Saved] {out}")


# ---------------------------------------------------------------------------
# Experiment B: Attention Sparsity Heatmap
# ---------------------------------------------------------------------------

def plot_sparsity(model: str):
    path = os.path.join(RESULTS_DIR, f"attn_sparsity_{model}.json")
    if not os.path.exists(path):
        print(f"[Skip] {path} not found")
        return

    with open(path) as f:
        data = json.load(f)

    block_data = data["data"]
    blocks = sorted(int(k) for k in block_data.keys())
    quadrants = ["TT", "TI", "IT", "II"]

    # For each quadrant, compute mean entropy over all (block, step) combos
    quad_entropy: dict[str, list] = {q: [] for q in quadrants}
    for block in blocks:
        for step_dict in block_data[str(block)].values():
            for q in quadrants:
                val = step_dict.get(q, {}).get("entropy_mean")
                if val is not None:
                    quad_entropy[q].append(val)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    summary = data.get("summary", {})
    fig.suptitle(
        f"Experiment B: Attention Sparsity — {model.upper()}\n"
        f"II/TI entropy ratio: {summary.get('II_TI_entropy_ratio', '?'):.4f}  "
        f"Verdict: {summary.get('verdict', '?')}",
        fontsize=12,
    )

    # Left: bar chart of mean entropy per quadrant
    means = [np.mean(quad_entropy[q]) if quad_entropy[q] else 0.0 for q in quadrants]
    stds  = [np.std(quad_entropy[q])  if quad_entropy[q] else 0.0 for q in quadrants]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    axes[0].bar(quadrants, means, yerr=stds, color=colors, capsize=5)
    axes[0].set_title("Mean Shannon entropy per attention block")
    axes[0].set_ylabel("Entropy (nats)")
    axes[0].set_xlabel("Attention quadrant")

    # Right: II entropy vs step for each layer (lines, using available blocks)
    step_keys_all = set()
    for bd in block_data.values():
        step_keys_all.update(bd.keys())
    steps = sorted(int(s) for s in step_keys_all)

    ii_by_step: dict[int, list] = {s: [] for s in steps}
    ti_by_step: dict[int, list] = {s: [] for s in steps}
    for bd in block_data.values():
        for step in steps:
            sd = bd.get(str(step), {})
            if sd.get("II", {}).get("entropy_mean") is not None:
                ii_by_step[step].append(sd["II"]["entropy_mean"])
            if sd.get("TI", {}).get("entropy_mean") is not None:
                ti_by_step[step].append(sd["TI"]["entropy_mean"])

    ii_means = [np.mean(ii_by_step[s]) if ii_by_step[s] else 0.0 for s in steps]
    ti_means = [np.mean(ti_by_step[s]) if ti_by_step[s] else 0.0 for s in steps]
    axes[1].plot(steps, ii_means, "o-", color="#C44E52", label="A_II (image-image)")
    axes[1].plot(steps, ti_means, "s-", color="#DD8452", label="A_TI (text-image)")
    axes[1].set_title("II vs TI entropy across denoising steps")
    axes[1].set_xlabel("Denoising step t")
    axes[1].set_ylabel("Mean entropy")
    axes[1].legend()

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f"attn_sparsity_{model}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[Saved] {out}")


# ---------------------------------------------------------------------------
# Experiment C: Branch Ablation Bar Chart
# ---------------------------------------------------------------------------

def plot_branch_ablation(model: str):
    path = os.path.join(RESULTS_DIR, f"branch_ablation_{model}.csv")
    if not os.path.exists(path):
        print(f"[Skip] {path} not found")
        return

    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if not rows:
        return

    schedules = [r["schedule"] for r in rows]
    ssim_vals = [float(r["ssim"]) if r.get("ssim") and r["ssim"] not in ("", "None") else 0.0 for r in rows]
    psnr_vals = [float(r["psnr"]) if r.get("psnr") and r["psnr"] not in ("", "None") else 0.0 for r in rows]
    clip_vals = [float(r["clip"]) if r.get("clip") and r["clip"] not in ("", "None") else 0.0 for r in rows]

    # Compute ratio and verdict
    baseline_row = next((r for r in rows if r["schedule"] == "baseline"), None)
    text_row     = next((r for r in rows if "text_skip" in r["schedule"]), None)
    image_row    = next((r for r in rows if "image_skip" in r["schedule"]), None)
    ratio, verdict = None, ""
    if baseline_row and text_row and image_row:
        b_ssim = float(baseline_row["ssim"])
        t_drop = b_ssim - float(text_row["ssim"])
        i_drop = b_ssim - float(image_row["ssim"])
        ratio  = t_drop / i_drop if i_drop > 0 else float("inf")
        verdict = "PASS" if ratio < 0.30 else "FAIL"

    x = np.arange(len(schedules))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    title = f"Experiment C: Branch Ablation — {model.upper()}"
    if ratio is not None:
        title += f"\nSSIM ratio (text_drop/img_drop): {ratio:.4f}  Verdict: {verdict}"
    fig.suptitle(title, fontsize=12)

    colors = ["#4C72B0" if "baseline" in s else
              "#C44E52" if "text" in s else "#55A868" for s in schedules]

    labels = ["baseline", "text skip\n(step 15–27)", "image skip\n(step 15–27)"]

    axes[0].bar(x, ssim_vals, color=colors)
    axes[0].set_title("SSIM (higher is better)")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel("SSIM"); axes[0].set_ylim(0, 1.05)
    for xi, v in zip(x, ssim_vals):
        axes[0].text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)

    axes[1].bar(x, psnr_vals, color=colors)
    axes[1].set_title("PSNR dB (higher is better)")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel("PSNR (dB)")
    for xi, v in zip(x, psnr_vals):
        axes[1].text(xi, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)

    axes[2].bar(x, clip_vals, color=colors)
    axes[2].set_title("CLIPScore (higher is better)")
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels, fontsize=9)
    axes[2].set_ylabel("CLIPScore")
    for xi, v in zip(x, clip_vals):
        axes[2].text(xi, v + 0.3, f"{v:.1f}", ha="center", fontsize=8)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4C72B0", label="baseline"),
        Patch(facecolor="#C44E52", label="text-branch skip"),
        Patch(facecolor="#55A868", label="image-branch skip"),
    ]
    axes[0].legend(handles=legend_elements, fontsize=8)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f"branch_ablation_{model}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[Saved] {out}")


# ---------------------------------------------------------------------------
# Experiment D: KV Drift Line Plot
# ---------------------------------------------------------------------------

def plot_kv_drift(model: str):
    path = os.path.join(RESULTS_DIR, f"xattn_kv_drift_{model}.json")
    if not os.path.exists(path):
        print(f"[Skip] {path} not found")
        return

    with open(path) as f:
        data = json.load(f)

    block_data = data["data"]
    blocks = sorted(int(b) for b in block_data.keys())
    all_steps: set = set()
    for bd in block_data.values():
        all_steps.update(int(s) for s in bd.keys())
    steps = sorted(all_steps)

    # Build block×step matrices
    k_mat = np.zeros((len(blocks), len(steps)))
    v_mat = np.zeros((len(blocks), len(steps)))
    for bi, block in enumerate(blocks):
        bd = block_data[str(block)]
        for si, step in enumerate(steps):
            sd = bd.get(str(step), {})
            k_mat[bi, si] = sd.get("K_drift_mean") or 0.0
            v_mat[bi, si] = sd.get("V_drift_mean") or 0.0

    # Block-averaged drift per step
    k_means = k_mat.mean(axis=0)
    v_means = v_mat.mean(axis=0)

    summary = data.get("summary", {})

    # Freeze ablation data
    freeze_path = os.path.join(RESULTS_DIR, "freeze_text_kv_quality.csv")
    freeze_rows = []
    if os.path.exists(freeze_path):
        with open(freeze_path) as f:
            freeze_rows = list(csv.DictReader(f))

    # Layout: 2 heatmaps + 1 line plot, then optionally freeze ablation bar
    has_freeze = bool(freeze_rows)
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(
        f"Experiment D: Text-side KV Drift — {model.upper()}\n"
        f"Final K_drift: {summary.get('final_step_K_drift_mean', 0)*100:.1f}%  "
        f"V_drift: {summary.get('final_step_V_drift_mean', 0)*100:.1f}%  "
        f"Verdict: {summary.get('verdict', '?')}",
        fontsize=13, fontweight="bold",
        color="darkred" if summary.get("verdict") != "PASS" else "darkgreen",
    )

    ncols = 4 if has_freeze else 3
    gs = fig.add_gridspec(1, ncols, wspace=0.35)

    # Heatmap shared colorbar range
    vmax = max(k_mat.max(), v_mat.max(), 0.01)

    ax_k = fig.add_subplot(gs[0, 0])
    im_k = ax_k.imshow(k_mat, aspect="auto", origin="upper",
                        vmin=0, vmax=vmax, cmap="hot_r",
                        extent=[steps[0]-0.5, steps[-1]+0.5, blocks[-1]+0.5, blocks[0]-0.5])
    ax_k.set_title("K_text drift vs step 0")
    ax_k.set_xlabel("Denoising step")
    ax_k.set_ylabel("Block index")
    plt.colorbar(im_k, ax=ax_k, label="Relative L2")

    ax_v = fig.add_subplot(gs[0, 1])
    im_v = ax_v.imshow(v_mat, aspect="auto", origin="upper",
                        vmin=0, vmax=vmax, cmap="hot_r",
                        extent=[steps[0]-0.5, steps[-1]+0.5, blocks[-1]+0.5, blocks[0]-0.5])
    ax_v.set_title("V_text drift vs step 0")
    ax_v.set_xlabel("Denoising step")
    ax_v.set_ylabel("Block index")
    plt.colorbar(im_v, ax=ax_v, label="Relative L2")

    ax_line = fig.add_subplot(gs[0, 2])
    ax_line.plot(steps, k_means, "o-", color="#4C72B0", label="K_text (mean over blocks)")
    ax_line.plot(steps, v_means, "s-", color="#DD8452", label="V_text (mean over blocks)")
    ax_line.axhline(0.10, color="green", linestyle="--", alpha=0.7, label="10% threshold")
    ax_line.fill_between(steps, 0, 0.10, alpha=0.1, color="green")
    ax_line.set_title("Mean drift per denoising step")
    ax_line.set_xlabel("Denoising step t")
    ax_line.set_ylabel("Relative L2 drift from step 0")
    ax_line.legend(fontsize=8)
    ax_line.set_ylim(bottom=0)

    if has_freeze:
        ax_frz = fig.add_subplot(gs[0, 3])
        psnr_vals = [float(r["psnr"]) if r.get("psnr") else 0.0 for r in freeze_rows]
        xlabels = [r["schedule"].replace("_", "\n") for r in freeze_rows]
        bar_colors = ["#4C72B0", "#C44E52", "#DD8452"][:len(psnr_vals)]
        ax_frz.bar(range(len(psnr_vals)), psnr_vals, color=bar_colors)
        ax_frz.set_title("Freeze KV Ablation: PSNR")
        ax_frz.set_xticks(range(len(psnr_vals)))
        ax_frz.set_xticklabels(["no freeze\n(baseline)", "freeze\nfrom step 1", "freeze\nfrom step 5"],
                                fontsize=8)
        ax_frz.set_ylabel("PSNR (dB)")
        ax_frz.axhline(psnr_vals[0], color="gray", linestyle="--", alpha=0.5)
        for xi, v in enumerate(psnr_vals):
            ax_frz.text(xi, v + 0.3, f"{v:.1f} dB", ha="center", fontsize=9, fontweight="bold")

    out = os.path.join(PLOTS_DIR, f"xattn_kv_drift_{model}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot analysis results")
    parser.add_argument("--exp", choices=["A", "B", "C", "D", "all"], default="all")
    parser.add_argument("--model", choices=["flux", "sd3", "both"], default="both")
    args = parser.parse_args()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    models = ["flux", "sd3"] if args.model == "both" else [args.model]
    exps   = ["A", "B", "C", "D"] if args.exp == "all" else [args.exp]

    dispatch = {"A": plot_dynamics, "B": plot_sparsity,
                "C": plot_branch_ablation, "D": plot_kv_drift}

    for model in models:
        for exp in exps:
            print(f"[Plot] Experiment {exp} — {model}")
            dispatch[exp](model)


if __name__ == "__main__":
    main()

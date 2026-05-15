"""
paths.py — Centralized path management for MM-DiT asymmetric experiments.

DATA_ROOT is read from env var MMDIT_DATA_ROOT, falling back to /data/jameskimh/mmdit_asymmetric.
All large image artifacts (reference images, generated samples) live under DATA_ROOT
so they persist across sessions and are reusable.
Small artifacts (JSON, CSV, PNG plots) stay in the repo under results/.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_ROOT = os.environ.get(
    "MMDIT_DATA_ROOT",
    "/data/jameskimh/mmdit_asymmetric",
)


def ref_dir(model: str) -> str:
    """Directory for reference (no-caching baseline) images."""
    return os.path.join(DATA_ROOT, "reference", model)


def samples_dir(*parts: str) -> str:
    """Directory for generated sample images from ablation experiments."""
    return os.path.join(DATA_ROOT, "samples", *parts)


def results_dir(*parts: str) -> str:
    """Directory for JSON/CSV result files (stays in repo)."""
    return os.path.join(BASE_DIR, "results", *parts)


def plots_dir() -> str:
    return results_dir("plots")

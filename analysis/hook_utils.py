"""
hook_utils.py — Forward hook utilities for MM-DiT stream activation capture.

FLUX double-stream block structure (FluxTransformerBlock):
  - Inputs: hidden_states (image), encoder_hidden_states (text)
  - After attention: both streams updated separately then MLP'd
  - Output: (updated_image_hidden, updated_text_hidden)

FLUX single-stream block (FluxSingleTransformerBlock):
  - Input: hidden_states already contains [text; image] concatenated
  - No separate text/image outputs

SD3 block (JointTransformerBlock):
  - Same as FLUX double-stream pattern (separate streams)
"""

from __future__ import annotations
import torch
import numpy as np
from collections import defaultdict
from typing import Callable


# ---------------------------------------------------------------------------
# Core: step-indexed hook state
# ---------------------------------------------------------------------------

class StreamHookState:
    """
    Accumulates per-(block, step) relative L2 deltas for text and image streams.

    Instead of storing full [N_tokens, D] tensors (expensive: 50MB/entry for FLUX),
    we keep only the previous step's activation on GPU and compute the scalar delta
    immediately in the hook. This reduces memory from ~25GB/prompt to ~500MB GPU.
    """

    def __init__(self):
        self.step: int = 0
        # deltas[stream][block_idx][step] = scalar float (mean relative L2 vs prev step)
        self.data: dict[str, dict[int, dict[int, float]]] = {
            "text": defaultdict(dict),
            "image": defaultdict(dict),
        }
        # prev[stream][block_idx] = tensor kept on GPU for delta computation
        self._prev: dict[str, dict[int, torch.Tensor]] = {
            "text": {},
            "image": {},
        }
        self._handles: list = []

    def reset_step(self):
        self.step = 0
        self._prev["text"].clear()
        self._prev["image"].clear()

    def next_step(self):
        self.step += 1

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._prev["text"].clear()
        self._prev["image"].clear()


# ---------------------------------------------------------------------------
# Relative L2 change between consecutive steps: ||h^t - h^{t-1}|| / ||h^{t-1}||
# ---------------------------------------------------------------------------

def compute_relative_l2(state: StreamHookState, stream: str) -> dict[int, dict[int, float]]:
    """Returns {block_idx: {step: relative_l2}} — already computed in hooks, just return."""
    return {block_idx: dict(step_dict)
            for block_idx, step_dict in state.data[stream].items()}


# ---------------------------------------------------------------------------
# FLUX double-stream hook registration
# ---------------------------------------------------------------------------

def _relative_l2_gpu(h_cur: torch.Tensor, h_prev: torch.Tensor) -> float:
    """Mean per-token relative L2: mean_tokens(||h_t - h_{t-1}|| / ||h_{t-1}||). On GPU."""
    h_cur = h_cur.float().squeeze(0)   # [N, D]
    h_prev = h_prev.float().squeeze(0)
    delta = (h_cur - h_prev).norm(dim=-1)          # [N]
    denom = h_prev.norm(dim=-1).clamp(min=1e-8)    # [N]
    return float((delta / denom).mean().item())


def _make_flux_double_hook(state: StreamHookState, block_idx: int,
                            capture_kv: bool = False) -> Callable:
    """
    Hook for FluxTransformerBlock.
    Computes relative L2 delta on GPU immediately; stores only scalar.
    Module output: (image_hidden_states, encoder_hidden_states) [diffusers convention]
    """
    def hook(module, input, output):
        step = state.step
        if not (isinstance(output, (tuple, list)) and len(output) >= 2):
            return
        img_h, txt_h = output[0], output[1]

        prev_img = state._prev["image"].get(block_idx)
        prev_txt = state._prev["text"].get(block_idx)
        if prev_img is not None:
            state.data["image"][block_idx][step] = _relative_l2_gpu(img_h, prev_img)
        if prev_txt is not None:
            state.data["text"][block_idx][step] = _relative_l2_gpu(txt_h, prev_txt)

        state._prev["image"][block_idx] = img_h.detach()
        state._prev["text"][block_idx] = txt_h.detach()
    return hook


def _make_flux_double_hook_full(state: StreamHookState, block_idx: int) -> Callable:
    """Alias — full-token mode now same as default (delta computed on GPU)."""
    return _make_flux_double_hook(state, block_idx)


def register_flux_stream_hooks(
    transformer,
    state: StreamHookState,
    full_tokens: bool = False,
) -> StreamHookState:
    """
    Register output hooks on all FLUX double-stream blocks.
    Single-stream blocks are skipped (text/image are merged there).

    Args:
        transformer: FluxTransformer2DModel
        state: StreamHookState to fill
        full_tokens: if True, store full [B, N, D] tensors (high memory); else mean over tokens
    """
    hook_fn = _make_flux_double_hook_full if full_tokens else _make_flux_double_hook

    for idx, block in enumerate(transformer.transformer_blocks):
        h = block.register_forward_hook(hook_fn(state, idx))
        state._handles.append(h)

    return state


# ---------------------------------------------------------------------------
# FLUX step counter hook (wraps the denoising call_module)
# ---------------------------------------------------------------------------

def install_step_counter(transformer, state: StreamHookState):
    """
    Insert a pre-hook on the transformer forward to auto-increment step counter.
    Call state.reset_step() before the full inference loop.
    """
    def pre_hook(module, input):
        pass  # step incremented externally in the sampling loop via callback

    # Diffusers pipelines expose a callback_on_step_end; use that instead.
    # This is a no-op here — caller uses pipe(callback_on_step_end=...) pattern.
    pass


def make_step_callback(state: StreamHookState):
    """
    Returns a callback compatible with diffusers pipe(callback_on_step_end=...).
    Increments state.step after each denoising step.
    """
    def callback(pipe, step_index, timestep, callback_kwargs):
        state.next_step()
        return callback_kwargs
    return callback


# ---------------------------------------------------------------------------
# SD3 double-stream hook registration
# ---------------------------------------------------------------------------

def _make_sd3_joint_hook(state: StreamHookState, block_idx: int,
                          full_tokens: bool = False) -> Callable:
    """
    Hook for JointTransformerBlock (SD3).
    Output: (encoder_hidden_states, hidden_states) OR (hidden_states,) for is_last.
    Computes relative L2 delta on GPU; stores only scalar.
    """
    def hook(module, input, output):
        step = state.step
        if isinstance(output, (tuple, list)):
            if len(output) == 2:
                txt_h, img_h = output[0], output[1]
            else:
                img_h, txt_h = output[0], None
        else:
            img_h, txt_h = output, None

        prev_img = state._prev["image"].get(block_idx)
        if prev_img is not None:
            state.data["image"][block_idx][step] = _relative_l2_gpu(img_h, prev_img)
        state._prev["image"][block_idx] = img_h.detach()

        if txt_h is not None:
            prev_txt = state._prev["text"].get(block_idx)
            if prev_txt is not None:
                state.data["text"][block_idx][step] = _relative_l2_gpu(txt_h, prev_txt)
            state._prev["text"][block_idx] = txt_h.detach()
    return hook


def register_sd3_stream_hooks(
    transformer,
    state: StreamHookState,
    full_tokens: bool = False,
) -> StreamHookState:
    """Register output hooks on all SD3 JointTransformerBlocks."""
    for idx, block in enumerate(transformer.transformer_blocks):
        h = block.register_forward_hook(
            _make_sd3_joint_hook(state, idx, full_tokens=full_tokens)
        )
        state._handles.append(h)
    return state


# ---------------------------------------------------------------------------
# Attention map capture (for sparsity analysis, Experiment B)
# ---------------------------------------------------------------------------

class AttentionMapCapture:
    """Stores post-softmax attention maps for a set of steps."""

    def __init__(self, target_steps: list[int]):
        self.target_steps = set(target_steps)
        self.current_step: int = 0
        # maps[block_idx][step] = {"TT": ..., "TI": ..., "IT": ..., "II": ...}
        self.maps: dict[int, dict[int, dict[str, torch.Tensor]]] = defaultdict(dict)
        self._handles: list = []

    def next_step(self):
        self.current_step += 1

    def reset(self):
        self.current_step = 0
        self.maps.clear()

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def _split_attention_map(
    attn_map: torch.Tensor,
    n_txt: int,
) -> dict[str, torch.Tensor]:
    """
    Split joint attention map into 4 quadrants.

    attn_map: [B, heads, N_total, N_total] where N_total = N_txt + N_img
              (assumes text tokens come first, image tokens after — FLUX/SD3 convention)
    """
    n_tot = attn_map.shape[-1]
    n_img = n_tot - n_txt
    return {
        "TT": attn_map[..., :n_txt, :n_txt],            # [B, H, Nt, Nt]
        "TI": attn_map[..., :n_txt, n_txt:],             # [B, H, Nt, Ni]
        "IT": attn_map[..., n_txt:, :n_txt],             # [B, H, Ni, Nt]
        "II": attn_map[..., n_txt:, n_txt:],             # [B, H, Ni, Ni]
    }


def compute_block_entropy(block: torch.Tensor) -> float:
    """Mean Shannon entropy of a 4D attention block [B, H, Q, K] (already softmax'd)."""
    p = block.float().clamp(min=1e-10)
    ent = -(p * p.log()).sum(dim=-1)  # [B, H, Q]
    return float(ent.mean().item())


def compute_top_k_coverage(block: torch.Tensor, k_frac: float = 0.05) -> float:
    """Fraction of attention weight captured by top-k_frac of keys."""
    B, H, Q, K = block.shape
    k = max(1, int(K * k_frac))
    top_vals, _ = block.float().topk(k, dim=-1)
    return float(top_vals.sum(dim=-1).mean().item())


def compute_sparsity_ratio(block: torch.Tensor, threshold: float = 0.01) -> float:
    """Fraction of attention weights below threshold (effectively zero)."""
    return float((block < threshold).float().mean().item())

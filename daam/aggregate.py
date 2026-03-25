"""Aggregate upsampled attention maps into per-token heatmaps."""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def aggregate_maps(
    upsampled: Dict[Tuple[int, int], torch.Tensor],
    normalize: str = "token",
) -> torch.Tensor:
    """Collapse per-(step, layer) maps into one heatmap per caption token.

    Averages across all ODE steps, transformer layers, and attention heads
    to produce a single ``(B, C, n_mels, T_spec)`` tensor.  Optionally
    normalises so that heatmaps are comparable across tokens.

    Parameters
    ----------
    upsampled : dict
        Output of ``upsample_attn()`` — maps ``(step, layer_idx)`` to
        tensors of shape ``(B, H, C, n_mels, T_spec)``.
    normalize : str
        ``"token"`` — min-max each token's heatmap to [0, 1] independently.
        ``"global"`` — min-max across all tokens per batch item to [0, 1].
        ``"none"`` — return raw mean values.

    Returns
    -------
    Tensor  (B, C, n_mels, T_spec)
        One 2-D heatmap per caption token per batch item.
    """
    if not upsampled:
        raise ValueError("upsampled dict is empty")

    stacked = torch.stack(list(upsampled.values()), dim=0)
    # stacked: (S*L, B, H, C, n_mels, T_spec)

    agg = stacked.mean(dim=0).mean(dim=1)
    # mean over steps+layers (dim 0), then mean over heads (dim 1)
    # -> (B, C, n_mels, T_spec)

    if normalize == "none":
        return agg

    if normalize == "token":
        B, C, M, T = agg.shape
        flat = agg.view(B, C, -1)               # (B, C, M*T)
        lo = flat.min(dim=-1, keepdim=True).values
        hi = flat.max(dim=-1, keepdim=True).values
        denom = (hi - lo).clamp(min=1e-8)
        flat = (flat - lo) / denom
        return flat.view(B, C, M, T)

    if normalize == "global":
        B, C, M, T = agg.shape
        flat = agg.view(B, -1)                   # (B, C*M*T)
        lo = flat.min(dim=-1, keepdim=True).values
        hi = flat.max(dim=-1, keepdim=True).values
        denom = (hi - lo).clamp(min=1e-8)
        flat = (flat - lo) / denom
        return flat.view(B, C, M, T)

    raise ValueError(f"Unknown normalize mode: {normalize!r}")

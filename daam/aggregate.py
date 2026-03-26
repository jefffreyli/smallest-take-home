"""Aggregate upsampled attention maps into per-token heatmaps."""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def aggregate_maps(
    upsampled: Dict[Tuple[int, int], torch.Tensor],
) -> torch.Tensor:
    """Collapse per-(step, layer) maps into one heatmap per caption token.

    Averages across all ODE steps, transformer layers, and attention heads
    to produce a single ``(B, C, n_mels, T_spec)`` tensor, then applies
    per-token min-max normalization to [0, 1].

    Parameters
    ----------
    upsampled : dict
        Output of ``upsample_attn()`` — maps ``(step, layer_idx)`` to
        tensors of shape ``(B, H, C, n_mels, T_spec)``.

    Returns
    -------
    Tensor  (B, C, n_mels, T_spec)
        One 2-D heatmap per caption token per batch item.
    """
    if not upsampled:
        raise ValueError("upsampled dict is empty")

    stacked = torch.stack(list(upsampled.values()), dim=0)
    agg = stacked.mean(dim=0).mean(dim=1)

    B, C, M, T = agg.shape
    flat = agg.view(B, C, -1)
    lo = flat.min(dim=-1, keepdim=True).values
    hi = flat.max(dim=-1, keepdim=True).values
    denom = (hi - lo).clamp(min=1e-8)
    flat = (flat - lo) / denom
    return flat.view(B, C, M, T)

"""Build per-transcript-word heatmaps from caption-token heatmaps."""

from __future__ import annotations

from typing import List, Tuple

import torch


def build_word_heatmaps(
    heatmaps: torch.Tensor,
    word_regions: List[Tuple[str, int, int]],
    normalize: str = "token",
) -> torch.Tensor:
    """Create per-word heatmaps from caption-token heatmaps.

    Sums attention energy across all caption tokens to get a total
    attention profile, then masks it to each word's time region.

    Parameters
    ----------
    heatmaps : Tensor  (B, C, n_mels, T_spec)
        Per-caption-token heatmaps from ``aggregate_mean_attn``.
    word_regions : list of (word, t_start, t_end)
        Output of ``words_to_time_regions``.
    normalize : str
        ``"token"`` | ``"global"`` | ``"none"``.

    Returns
    -------
    Tensor  (B, W, n_mels, T_spec)
        One heatmap per transcript word.
    """
    B, _C, n_mels, T_spec = heatmaps.shape
    W = len(word_regions)

    total_attn = heatmaps.sum(dim=1)  # (B, n_mels, T_spec)

    word_maps = torch.zeros(B, W, n_mels, T_spec)
    for w_idx, (_word, t_start, t_end) in enumerate(word_regions):
        word_maps[:, w_idx, :, t_start:t_end] = total_attn[:, :, t_start:t_end]

    if normalize == "token":
        cmin = word_maps.flatten(2).min(dim=2).values[:, :, None, None]
        cmax = word_maps.flatten(2).max(dim=2).values[:, :, None, None]
        span = (cmax - cmin).clamp(min=1e-8)
        word_maps = (word_maps - cmin) / span
    elif normalize == "global":
        for b in range(B):
            gmin = word_maps[b].min()
            gmax = word_maps[b].max()
            word_maps[b] = (word_maps[b] - gmin) / max(gmax - gmin, 1e-8)

    return word_maps

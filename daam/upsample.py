"""Upsample raw cross-attention maps to mel spectrogram dimensions."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def upsample_map(
    attn: torch.Tensor,
    n_mels: int,
    T_spec: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Interpolate a single attention tensor to ``(n_mels, T_spec)``.

    The raw attention tensor has shape ``(B, H, audio_frames, C)`` where
    ``C = caption_tokens``.  Each per-token time curve is treated as a
    ``1 x audio_frames`` "row" and resized to ``(n_mels, T_spec)`` via
    ``F.interpolate``, broadcasting the same temporal profile across all
    mel frequency bins (since cross-attention carries no per-frequency
    information).

    Parameters
    ----------
    attn : Tensor  (B, H, audio_frames, C)
        Raw attention weights from one ``(step, layer)`` entry.
    n_mels : int
        Target frequency bins (height of the mel spectrogram).
    T_spec : int
        Target time frames (width of the mel spectrogram).
    mode : str
        Interpolation mode passed to ``F.interpolate`` (default ``"bilinear"``).

    Returns
    -------
    Tensor  (B, H, C, n_mels, T_spec)
        Per-token 2-D heatmaps aligned to the mel spectrogram grid.
    """
    B, H, T_in, C = attn.shape

    # move caption dim before time dim
    x = attn.permute(0, 1, 3, 2)           # (B, H, C, T_in)
    # flatten B, H, C into one batch dimension
    x = x.reshape(B * H * C, 1, 1, T_in) 

    # interpolate to (n_mels, T_spec)
    x = F.interpolate(
        x,
        size=(n_mels, T_spec),
        mode=mode,
        align_corners=False,
    )                                        # (N, 1, n_mels, T_spec)

    x = x.squeeze(1)                        # (N, n_mels, T_spec)
    x = x.reshape(B, H, C, n_mels, T_spec)
    x = x.clamp(min=0.0)

    return x

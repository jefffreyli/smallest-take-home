"""Visualization helpers for DAAM-style token heatmap overlays."""

from __future__ import annotations

import math
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_token_heatmaps(
    mel: torch.Tensor,
    heatmaps: torch.Tensor,
    token_labels: List[str],
    save_path: Optional[str] = None,
    max_tokens: Optional[int] = 10,
) -> matplotlib.figure.Figure:
    """Create a figure with per-token attention overlays on a mel spectrogram.

    Parameters
    ----------
    mel : Tensor  (n_mels, T_spec)
        Log-mel spectrogram to use as the base image.
    heatmaps : Tensor  (C, n_mels, T_spec)
        One heatmap per caption token (output of ``aggregate_attn`` for a
        single batch item).
    token_labels : list[str]
        Human-readable label for each of the *C* caption tokens.
    save_path : str or None
        If provided, saves the figure to this path.
    max_tokens : int or None
        Maximum number of token panels to show.  When the caption has
        more tokens than this limit, the tokens with the highest total
        attention energy are selected.  ``None`` means show all.

    Returns
    -------
    matplotlib.figure.Figure
    """
    mel_np = mel.detach().float().cpu().numpy()
    heatmaps_np = heatmaps.detach().float().cpu().numpy()
    C = heatmaps_np.shape[0]

    if max_tokens is not None and C > max_tokens:
        energies = heatmaps_np.reshape(C, -1).sum(axis=1)
        top_idx = np.argsort(energies)[::-1][:max_tokens]
        top_idx = np.sort(top_idx)
    else:
        top_idx = np.arange(C)

    n_panels = 1 + len(top_idx)
    cols = min(n_panels, 5)
    rows = math.ceil(n_panels / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes:
        ax.set_axis_off()

    axes[0].imshow(mel_np, origin="lower", aspect="auto", cmap="magma")
    axes[0].set_title("Mel spectrogram", fontsize=9)
    axes[0].set_axis_on()
    axes[0].set_yticks([])
    axes[0].set_xlabel("Time frame", fontsize=7)

    for panel_i, token_i in enumerate(top_idx, start=1):
        ax = axes[panel_i]
        ax.imshow(mel_np, origin="lower", aspect="auto", cmap="gray_r")
        ax.imshow(
            heatmaps_np[token_i],
            origin="lower",
            aspect="auto",
            cmap="hot",
            alpha=0.55,
        )
        label = token_labels[token_i] if token_i < len(token_labels) else f"tok {token_i}"
        ax.set_title(label, fontsize=8)
        ax.set_axis_on()
        ax.set_yticks([])
        ax.set_xticks([])

    fig.tight_layout(pad=0.5)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig

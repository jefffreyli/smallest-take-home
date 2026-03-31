"""Accumulator for raw cross-attention weight tensors (online running mean)."""

from __future__ import annotations

from typing import Optional

import torch


class AttentionStore:
    """Accumulates cross-attention weights via a running sum.

    Instead of storing every ``(step, layer)`` tensor in a dict (which
    can consume tens of GB for a full CapSpeech run), this class keeps a
    **single accumulator** of shape ``(B, H, audio_frames, C)`` and adds
    each incoming tensor to it.  Call :meth:`get_mean` at the end to
    retrieve the average over all steps and layers.

    Tensors are detached and moved to CPU before accumulation to avoid
    GPU memory buildup.
    """

    def __init__(self):
        self.sum: Optional[torch.Tensor] = None # (B, H, audio_frames, C)
        self.count: int = 0
        self.cur_step: int = 0
        self.num_steps: int = 0
        self.layer_indices: set = set() # used to report how many unique layers were captured

    def update(self, layer_idx: int, attn_weights: torch.Tensor) -> None:
        """Add one attention tensor to the running sum."""
        t = attn_weights.detach().cpu()
        if self.sum is None:
            self.sum = torch.zeros_like(t)
        self.sum += t
        self.count += 1
        self.layer_indices.add(layer_idx)

    def step(self) -> None:
        self.cur_step += 1
        self.num_steps = self.cur_step

    def reset(self) -> None:
        self.sum = None
        self.count = 0
        self.cur_step = 0
        self.num_steps = 0
        self.layer_indices.clear()

    def get_mean(self) -> torch.Tensor:
        """Return the mean attention over all accumulated updates.

        Returns
        -------
        Tensor  (B, H, audio_frames, caption_tokens)
        """
        if self.sum is None or self.count == 0:
            raise RuntimeError("No attention maps have been accumulated")
        return self.sum / self.count

    @property
    def num_layers(self) -> int:
        return len(self.layer_indices)

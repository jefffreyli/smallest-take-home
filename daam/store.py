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
        self._sum: Optional[torch.Tensor] = None
        self._count: int = 0
        self.cur_step: int = 0
        self._num_steps: int = 0
        self._layer_indices: set = set()

    def update(self, layer_idx: int, attn_weights: torch.Tensor) -> None:
        """Add one attention tensor to the running sum."""
        t = attn_weights.detach().cpu()
        if self._sum is None:
            self._sum = torch.zeros_like(t)
        self._sum += t
        self._count += 1
        self._layer_indices.add(layer_idx)

    def step(self) -> None:
        self.cur_step += 1
        self._num_steps = self.cur_step

    def reset(self) -> None:
        self._sum = None
        self._count = 0
        self.cur_step = 0
        self._num_steps = 0
        self._layer_indices.clear()

    def get_mean(self) -> torch.Tensor:
        """Return the mean attention over all accumulated updates.

        Returns
        -------
        Tensor  (B, H, audio_frames, caption_tokens)
        """
        if self._sum is None or self._count == 0:
            raise RuntimeError("No attention maps have been accumulated")
        return self._sum / self._count

    @property
    def num_steps(self) -> int:
        return self._num_steps

    @property
    def num_layers(self) -> int:
        return len(self._layer_indices)

    @property
    def count(self) -> int:
        return self._count

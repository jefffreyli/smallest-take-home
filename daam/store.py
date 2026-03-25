"""Accumulator for raw cross-attention weight tensors."""

from __future__ import annotations

from typing import Dict, Tuple

import torch


class AttentionStore:
    """Collects cross-attention weight tensors keyed by (ode_step, layer_idx).

    Each stored tensor has shape (batch, heads, audio_frames, caption_tokens).
    Tensors are detached and moved to CPU to avoid GPU memory buildup.
    """

    def __init__(self):
        self._maps: Dict[Tuple[int, int], torch.Tensor] = {}
        self.cur_step: int = 0

    def update(self, layer_idx: int, attn_weights: torch.Tensor) -> None:
        key = (self.cur_step, layer_idx)
        self._maps[key] = attn_weights.detach().cpu()

    def step(self) -> None:
        self.cur_step += 1

    def reset(self) -> None:
        self._maps.clear()
        self.cur_step = 0

    def get_all(self) -> Dict[Tuple[int, int], torch.Tensor]:
        return dict(self._maps)

    @property
    def num_steps(self) -> int:
        if not self._maps:
            return 0
        return max(s for s, _ in self._maps) + 1

    @property
    def num_layers(self) -> int:
        if not self._maps:
            return 0
        return max(l for _, l in self._maps) + 1

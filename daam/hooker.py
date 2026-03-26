"""Context manager that instruments CrossDiT cross-attention layers."""

from __future__ import annotations

from typing import List, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "CapSpeech"))

from capspeech.nar.model.modules import Attention, AttnProcessor
from capspeech.nar.network.crossdit import CrossDiT

from daam.store import AttentionStore
from daam.processor import CrossAttnCaptureProcessor


class CapSpeechAttentionHooker:
    """Instruments a ``CrossDiT`` model by replacing the ``AttnProcessor``
    inside every ``CrossDiTBlock.cross_attn`` with a
    ``CrossAttnCaptureProcessor``.

    Example usage:
        hooker = CapSpeechAttentionHooker(model)
        with hooker:
            # run inference — attention maps accumulate in hooker.store
            ...
        mean = hooker.store.get_mean()
    """

    def __init__(self, model: CrossDiT):
        self.model = model
        self.store = AttentionStore()
        self._processors: List[CrossAttnCaptureProcessor] = []
        self._originals: List[Tuple[Attention, AttnProcessor]] = []

    @staticmethod
    def _find_cross_attn_modules(model: CrossDiT) -> List[Attention]:
        """Return every ``cross_attn`` Attention module in block order
        (in_blocks -> mid_block -> out_blocks)."""
        modules: List[Attention] = []
        for block in model.in_blocks:
            modules.append(block.cross_attn)
        modules.append(model.mid_block.cross_attn)
        for block in model.out_blocks:
            modules.append(block.cross_attn)
        return modules

    def hook(self) -> None:
        """
        Hook the model by replacing the AttnProcessor inside every CrossDiTBlock.cross_attn with a CrossAttnCaptureProcessor.
        """
        cross_attns = self._find_cross_attn_modules(self.model)
        for layer_idx, attn_module in enumerate(cross_attns):
            original = attn_module.processor
            capture = CrossAttnCaptureProcessor(
                store=self.store,
                layer_idx=layer_idx,
                original_processor=original,
            )
            self._originals.append((attn_module, original))
            self._processors.append(capture)
            attn_module.processor = capture

    def unhook(self) -> None:
        for attn_module, original in self._originals:
            attn_module.processor = original
        self._originals.clear()
        self._processors.clear()

    def set_capture(self, enabled: bool) -> None:
        """Set the capture enabled state for all processors."""
        for p in self._processors:
            p.capture_enabled = enabled

    def __enter__(self):
        self.hook()
        return self

    def __exit__(self, *exc):
        self.unhook()
        return False

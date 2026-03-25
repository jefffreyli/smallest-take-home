"""DAAM internals for CapSpeech cross-attention extraction."""

from daam.store import AttentionStore
from daam.processor import CrossAttnCaptureProcessor
from daam.hooker import CapSpeechAttentionHooker
from daam.upsample import upsample_map
from daam.aggregate import aggregate_maps
from daam.visualize import plot_token_heatmaps
from daam.utils import pick_inference_device

__all__ = [
    "AttentionStore",
    "CrossAttnCaptureProcessor",
    "CapSpeechAttentionHooker",
    "upsample_map",
    "aggregate_maps",
    "plot_token_heatmaps",
    "pick_inference_device",
]

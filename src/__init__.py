"""DAAM internals for CapSpeech cross-attention extraction."""

from src.store import AttentionStore
from src.processor import CrossAttnCaptureProcessor
from src.hooker import CapSpeechAttentionHooker
from src.upsample import upsample_map
from src.visualize import plot_token_heatmaps
from src.utils import pick_inference_device

__all__ = [
    "AttentionStore",
    "CrossAttnCaptureProcessor",
    "CapSpeechAttentionHooker",
    "upsample_map",
    "plot_token_heatmaps",
    "pick_inference_device",
]

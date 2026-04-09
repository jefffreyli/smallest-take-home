"""Small helpers shared by the DAAM pipeline."""

from __future__ import annotations

import torch


def pick_inference_device() -> str:
    """Return ``cuda:0`` when a CUDA GPU is available, otherwise ``cpu``."""
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"

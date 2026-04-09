from __future__ import annotations
from src.utils import pick_inference_device
from src.visualize import plot_token_heatmaps
from src.upsample import upsample_map
from src import CapSpeechAttentionHooker
from capspeech.nar.network.crossdit import CrossDiT

from typing import Dict, List, Optional, Tuple

import torch
from einops import rearrange
from torchdiffeq import odeint

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "CapSpeech"))


# Task 1 - Attention Extraction

@torch.no_grad()
def extract_attn(
    model: CrossDiT,
    vocoder, # converts the mel spectrogram to a waveform
    x: torch.Tensor,
    cond: Optional[torch.Tensor],
    text: torch.Tensor,
    prompt: torch.Tensor,
    clap: torch.Tensor,
    prompt_mask: torch.Tensor,
    *,
    steps: int = 25,
    cfg: float = 2.0, # cfg = w, guidance scalar
    sway_sampling_coef: float = -1.0,
    device: str = "cuda",
) -> dict:
    """Run CapSpeech inference while capturing cross-attention maps.

    This is a modified version of ``capspeech.nar.inference.sample`` that
    hooks into every ``CrossDiTBlock.cross_attn`` layer and records the
    attention weight matrix at each ODE step (conditional pass only).

    Parameters
    ----------
    model : CrossDiT
        The CapSpeech NAR backbone (already loaded & on *device*).
    vocoder : nn.Module
        BigVGAN vocoder to convert the final mel to a waveform.
    x : Tensor  (B, N, mel_dim)
        Zero-filled mel placeholder whose length encodes the target duration.
    cond : Tensor | None
        Conditioning audio (or ``None``).
    text : LongTensor  (B, T_text)
        Phoneme token IDs.
    prompt : Tensor  (B, T_cap, t5_dim)
        T5-encoded caption embeddings.
    clap : Tensor  (B, clap_dim)
        CLAP tag embedding.
    prompt_mask : BoolTensor  (B, T_cap)
        Padding mask for the caption.
    steps : int
        Number of Euler ODE time-points (default 25).
    cfg : float
        Classifier-free guidance scale (default 2.0).
    sway_sampling_coef : float
        Sway sampling coefficient (default -1.0).
    device : str
        Device string.

    Returns
    -------
    dict with keys:
        ``"attention_mean"``
            Mean attention tensor of shape
            ``(B, heads, audio_frames, caption_tokens)`` averaged over
            all ODE steps and transformer layers.
        ``"mel"``
            Generated mel spectrogram as a Tensor of shape
            ``(B, mel_dim, T_spec)``.
        ``"wav"``
            Generated waveform as a numpy array.
        ``"metadata"``
            Dict with ``num_steps`` (number of ODE fn evaluations,
            which is ``steps - 1`` for Euler), ``num_layers``,
            ``num_heads``, ``audio_frames``, ``caption_tokens``.
    """

    model.eval()
    vocoder.eval()

    hooker = CapSpeechAttentionHooker(model)

    # ODE starts from Gaussian noise
    y0 = torch.randn_like(x)

    # CFG setup: Use negative/null inputs for unconditional branch of CFG
    neg_text = torch.ones_like(text) * -1  # negative text tokens
    neg_clap = torch.zeros_like(clap)
    neg_prompt = torch.zeros_like(prompt)
    neg_prompt_mask = torch.zeros_like(prompt_mask)
    neg_prompt_mask[:, 0] = 1

    # CFG
    with hooker:
        def fn(t, x_t):
            # conditional pass: capture attention
            hooker.set_capture(True)
            pred = model(
                x=x_t, cond=cond, text=text, time=t,
                prompt=prompt, clap=clap,
                mask=None,
                prompt_mask=prompt_mask,
            )

            # unconditional pass: disable attention capture
            hooker.set_capture(False)
            null_pred = model(
                x=x_t, cond=cond, text=neg_text, time=t,
                prompt=neg_prompt, clap=neg_clap,
                mask=None,
                prompt_mask=neg_prompt_mask,
            )

            hooker.store.step()  # increment step counter

            return pred + (pred - null_pred) * cfg # velocity field

        # sway sampling: warps the time curve to concetrate more steps near t=0 to improve sample quality without increasing the number of steps
        t_start = 0
        t = torch.linspace(t_start, 1, steps, device=device)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (
                torch.cos(torch.pi / 2 * t) - 1 + t
            )

        # ODE simulation: solve the ODE using Euler steps to get the trajectory
        trajectory = odeint(fn, y0, t, method="euler")

    out = trajectory[-1] # final output of the ODE simulation (mel tensor)
    out = rearrange(out, "b n d -> b d n")
    mel_out = out.cpu()

    with torch.inference_mode():
        wav_gen = vocoder(out) # convert the mel spectrogram to a waveform
    wav_gen_float = wav_gen.squeeze().cpu().numpy()

    # get the mean attention weight matrix over all steps and layers
    attention_mean = hooker.store.get_mean()

    metadata = {
        "num_steps": hooker.store.num_steps,
        "num_layers": hooker.store.num_layers,
        "num_heads": attention_mean.shape[1],
        "audio_frames": attention_mean.shape[2],
        "caption_tokens": attention_mean.shape[3],
    }

    return {
        "attention_mean": attention_mean,
        "mel": mel_out,
        "wav": wav_gen_float,
        "metadata": metadata,
    }

# Task 2 - Mapping to Speech

def upsample_attn(
    attention_maps: Dict[Tuple[int, int], torch.Tensor],
    n_mels: int = 100,
    T_spec: Optional[int] = None,
) -> Dict[Tuple[int, int], torch.Tensor]:
    """
    TL;DR: (B, H, audio_frames, caption_tokens) -> (B, H, caption_tokens, n_mels, T_spec)

    Upsample raw attention maps to align with a mel spectrogram grid.

    Each input tensor has shape ``(B, H, audio_frames, caption_tokens)``.
    The time curve for every caption token is expanded into a 2-D
    ``(n_mels, T_spec)`` heatmap suitable for overlaying on a mel
    spectrogram.  Because cross-attention carries no per-frequency
    information, the temporal profile is broadcast uniformly across all
    mel bins.

    Parameters:
    attention_maps : dict
        Output of ``extract_attn()["attention_maps"]`` — maps
        ``(step, layer_idx)`` to tensors of shape
        ``(B, H, audio_frames, caption_tokens)``.
    n_mels : int
        Number of mel frequency bins (default 100, matching CapSpeech).
    T_spec : int or None
        Target number of time frames.  If ``None``, defaults to the
        existing ``audio_frames`` dimension (no time resampling).

    Returns:
    dict
        Same keys as *attention_maps*; values are tensors of shape
        ``(B, H, caption_tokens, n_mels, T_spec)``.
    """
    result: Dict[Tuple[int, int], torch.Tensor] = {}

    for key, attn in attention_maps.items():
        t_spec = T_spec if T_spec is not None else attn.shape[2]
        result[key] = upsample_map(attn, n_mels=n_mels, T_spec=t_spec)

    return result


# Task 3 - Aggregation

def aggregate_mean_attn(
    upsampled: torch.Tensor,
) -> torch.Tensor:
    """Aggregate an upsampled attention map into per-token heatmaps.

    Accepts the output of ``upsample_map`` and collapses the head
    dimension, then applies per-token min-max normalization.

    Goal: (B, H, C, n_mels, T_spec) -> (B, C, n_mels, T_spec)

    Parameters
    ----------
    upsampled : Tensor  (B, H, C, n_mels, T_spec)
        Output of ``upsample_map`` applied to ``extract_attn()["attention_mean"]``.

    Returns
    -------
    Tensor  (B, C, n_mels, T_spec)
    """
    # average across heads
    agg = upsampled.mean(dim=1)

    # per token min-max normalization of each token's heatmap to the range [0, 1]
    cmin = agg.flatten(2).min(dim=2).values[:, :, None, None]
    cmax = agg.flatten(2).max(dim=2).values[:, :, None, None]
    span = (cmax - cmin).clamp(min=1e-8)
    agg = (agg - cmin) / span

    return agg


# Task 4 - Visualization

def visualize_maps(
    heatmaps: torch.Tensor,
    mel: torch.Tensor,
    token_labels: List[str],
    save_path: Optional[str] = None,
    max_tokens: Optional[int] = 10,
    batch_idx: int = 0,
):
    """Overlay per-token attention heatmaps on a mel spectrogram.

    Parameters
    ----------
    heatmaps : Tensor  (B, C, n_mels, T_spec)
        Output of ``aggregate_attn()``.
    mel : Tensor  (B, n_mels, T_spec) or (n_mels, T_spec)
        Generated mel spectrogram from ``extract_attn()["mel"]``.
    token_labels : list[str]
        Human-readable label for each caption token.
    save_path : str or None
        If provided, saves the figure to this path.
    max_tokens : int or None
        Maximum number of token overlay panels to show.
    batch_idx : int
        Which batch element to visualise.

    Returns
    -------
    matplotlib.figure.Figure
    """
    mel_2d = mel[batch_idx] if mel.ndim == 3 else mel
    return plot_token_heatmaps(
        mel=mel_2d,
        heatmaps=heatmaps[batch_idx],
        token_labels=token_labels,
        save_path=save_path,
        max_tokens=max_tokens,
    )

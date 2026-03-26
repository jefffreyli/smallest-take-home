from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from einops import rearrange
from torchdiffeq import odeint

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "CapSpeech"))

from capspeech.nar.network.crossdit import CrossDiT
from daam import CapSpeechAttentionHooker
from daam.upsample import upsample_map
from daam.aggregate import aggregate_maps
from daam.visualize import plot_token_heatmaps
from daam.utils import pick_inference_device

import soundfile as sf
from capspeech.nar.generate import load_model, encode, get_duration
from capspeech.nar.utils import make_pad_mask

import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from capspeech.nar.generate import seed_everything
import config

# Task 1 - Attention Extraction

@torch.no_grad()
def extract_attn(
    model: CrossDiT,
    vocoder,
    x: torch.Tensor,
    cond: Optional[torch.Tensor],
    text: torch.Tensor,
    prompt: torch.Tensor,
    clap: torch.Tensor,
    prompt_mask: torch.Tensor,
    *,
    steps: int = 25,
    cfg: float = 2.0,
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

    y0 = torch.randn_like(x)

    neg_text = torch.ones_like(text) * -1
    neg_clap = torch.zeros_like(clap)
    neg_prompt = torch.zeros_like(prompt)
    neg_prompt_mask = torch.zeros_like(prompt_mask)
    neg_prompt_mask[:, 0] = 1

    with hooker:

        def fn(t, x_t):
            hooker.set_capture(True)
            pred = model(
                x=x_t, cond=cond, text=text, time=t,
                prompt=prompt, clap=clap,
                mask=None,
                prompt_mask=prompt_mask,
            )

            hooker.set_capture(False)
            null_pred = model(
                x=x_t, cond=cond, text=neg_text, time=t,
                prompt=neg_prompt, clap=neg_clap,
                mask=None,
                prompt_mask=neg_prompt_mask,
            )

            hooker.store.step()

            return pred + (pred - null_pred) * cfg

        t_start = 0
        t = torch.linspace(t_start, 1, steps, device=device)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (
                torch.cos(torch.pi / 2 * t) - 1 + t
            )

        trajectory = odeint(fn, y0, t, method="euler")

    out = trajectory[-1]
    out = rearrange(out, "b n d -> b d n")
    mel_out = out.cpu()

    with torch.inference_mode():
        wav_gen = vocoder(out)
    wav_gen_float = wav_gen.squeeze().cpu().numpy()

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
    """Upsample raw attention maps to align with a mel spectrogram grid.

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
    out: Dict[Tuple[int, int], torch.Tensor] = {}

    for key, attn in attention_maps.items():
        t_spec = T_spec if T_spec is not None else attn.shape[2]
        out[key] = upsample_map(attn, n_mels=n_mels, T_spec=t_spec)

    return out


# Task 3 - Aggregation

def aggregate_attn(
    upsampled_maps: Dict[Tuple[int, int], torch.Tensor],
) -> torch.Tensor:
    """Aggregate upsampled attention maps into per-token heatmaps.

    Averages across all ODE steps, transformer layers, and attention
    heads, then min-max normalizes each token independently to [0, 1].

    Parameters
    ----------
    upsampled_maps : dict
        Output of ``upsample_attn()`` — maps ``(step, layer_idx)`` to
        tensors of shape ``(B, H, C, n_mels, T_spec)``.

    Returns
    -------
    Tensor  (B, C, n_mels, T_spec)
        One 2-D heatmap per caption token per batch item.
    """
    return aggregate_maps(upsampled_maps)

def aggregate_mean_attn(
    mean_attn: torch.Tensor,
    n_mels: int = 100,
    T_spec: Optional[int] = None,
) -> torch.Tensor:
    """Upsample + head-average + normalize from a single mean-attention tensor.

    This is the memory-efficient fast path that replaces the dict-based
    ``upsample_attn`` -> ``aggregate_attn`` chain.  It operates on a
    single pre-averaged tensor, avoiding the need to materialise hundreds
    of intermediate tensors.  Each token is min-max normalized
    independently to [0, 1].

    Parameters
    ----------
    mean_attn : Tensor  (B, H, audio_frames, C)
        Mean attention over all ODE steps and layers, as returned by
        ``extract_attn()["attention_mean"]``.
    n_mels : int
        Number of mel frequency bins (default 100).
    T_spec : int or None
        Target time frames.  Defaults to ``audio_frames``.

    Returns
    -------
    Tensor  (B, C, n_mels, T_spec)
    """
    if T_spec is None:
        T_spec = mean_attn.shape[2]

    upsampled = upsample_map(mean_attn, n_mels=n_mels, T_spec=T_spec)
    agg = upsampled.mean(dim=1)

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

# Main
EXAMPLES = [
    {
        "transcript": "Hello world.",
        "caption": "A calm male voice.",
    },
    {
        "transcript": "The weather is so nice today because of the bright yellow sun.",
        "caption": "A cheerful female voice.",
    },
    {
        "transcript": "I love music and I love dance.",
        "caption": "A warm, expressive voice.",
    },
    {
        "transcript": "Good morning everyone, I am happy to be here today.",
        "caption": "A deep, slow male voice.",
    },
    {
        "transcript": "Thank you very much for your time and your attention.",
        "caption": "A soft, gentle female voice.",
    },
]


if __name__ == "__main__":
    task = config.TASK
    output_dir = config.OUTPUT_DIR
    device = config.DEVICE
    steps = config.STEPS
    cfg = config.CFG
    max_tokens = config.MAX_TOKENS
    seed = config.SEED

    if device == "auto":
        device = pick_inference_device()

    seed_everything(seed)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Device: {device}")
    model_list = load_model(device, task)
    (model, vocoder, phn2num, text_tokenizer, clap_model,
     duration_tokenizer, duration_model, caption_tokenizer, caption_encoder) = model_list

    for idx, example in enumerate(EXAMPLES, start=1):
        transcript = example["transcript"]
        caption = example["caption"]
        print(f"\n{'='*60}")
        print(f"Example {idx}: {transcript[:60]}...")

        tag = "none"
        phn = encode(transcript, text_tokenizer)
        text_tokens = [phn2num[p] for p in phn]
        text = torch.LongTensor(text_tokens).unsqueeze(0).to(device)

        with torch.no_grad():
            batch_enc = caption_tokenizer(caption, return_tensors="pt")
            ori_token_ids = batch_enc["input_ids"].to(device)
            prompt = caption_encoder(input_ids=ori_token_ids).last_hidden_state.squeeze().unsqueeze(0).to(device)

            tag_embed = clap_model.get_text_embedding([tag], use_tensor=True)
            clap = tag_embed.squeeze().unsqueeze(0).to(device)

            duration_inputs = caption + " <NEW_SEP> " + transcript
            duration_inputs = duration_tokenizer(
                duration_inputs, return_tensors="pt",
                padding="max_length", truncation=True, max_length=400,
            )
            predicted_dur = duration_model(**duration_inputs).logits.squeeze().item()
            duration = get_duration(transcript, predicted_dur)

        audio_clips = torch.zeros([1, math.ceil(duration * 24000 / 256), 100]).to(device)
        seq_len_prompt = prompt.shape[1]
        prompt_lens = torch.Tensor([seq_len_prompt])
        prompt_mask = make_pad_mask(prompt_lens, seq_len_prompt).to(prompt.device)

        result = extract_attn(
            model, vocoder, audio_clips, None, text, prompt, clap, prompt_mask,
            steps=steps, cfg=cfg, sway_sampling_coef=-1.0, device=device,
        )

        mel_spec = result["mel"]
        n_mels = mel_spec.shape[1]
        T_spec = mel_spec.shape[2]

        heatmaps = aggregate_mean_attn(
            result["attention_mean"], n_mels=n_mels, T_spec=T_spec,
        )

        token_ids = ori_token_ids.squeeze().tolist()
        token_labels = caption_tokenizer.convert_ids_to_tokens(token_ids)

        fig_path = os.path.join(output_dir, f"example_{idx}.png")
        wav_path = os.path.join(output_dir, f"example_{idx}.wav")

        visualize_maps(
            heatmaps, mel_spec, token_labels,
            save_path=fig_path, max_tokens=max_tokens,
        )
        
        plt.close("all")

        sf.write(wav_path, result["wav"], 24000)
        print(f"  Saved: {fig_path}, {wav_path}")

    print(f"\nAll 5 examples saved to {output_dir}/")
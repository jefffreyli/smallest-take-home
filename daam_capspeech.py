"""
DAAM (Diffusion Attentive Attribution Maps) adapted for CapSpeech.

Public API with the four deliverable functions:
  - extract_attn()    — Task 1: capture cross-attention maps during inference
  - upsample_attn()   — Task 2: map attention to spectrogram time axis
  - aggregate_attn()   — Task 3: aggregate across layers/steps into per-token heatmaps
  - visualize_maps()   — Task 4: overlay token heatmaps on spectrogram plots

Reference:
  - Tang et al., "What the DAAM: Interpreting Stable Diffusion Using
    Cross Attention" (ACL / arXiv)
  - Wang et al., "CapSpeech: Enabling Downstream Applications in
    Style-Captioned Text-to-Speech" (arXiv)
"""

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

# Task 1 — Attention Extraction

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

# Task 2 — Mapping to Speech

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


# ---------------------------------------------------------------------------
# Task 3 — Aggregation
# ---------------------------------------------------------------------------

def aggregate_attn(
    upsampled_maps: Dict[Tuple[int, int], torch.Tensor],
    normalize: str = "token",
) -> torch.Tensor:
    """Aggregate upsampled attention maps into per-token heatmaps.

    Averages across all ODE steps, transformer layers, and attention
    heads, then normalises so heatmaps are comparable across tokens.

    Parameters
    ----------
    upsampled_maps : dict
        Output of ``upsample_attn()`` — maps ``(step, layer_idx)`` to
        tensors of shape ``(B, H, C, n_mels, T_spec)``.
    normalize : str
        ``"token"``  — min-max each token independently to [0, 1].
        ``"global"`` — min-max across all tokens per batch item.
        ``"none"``   — return raw mean values.

    Returns
    -------
    Tensor  (B, C, n_mels, T_spec)
        One 2-D heatmap per caption token per batch item.
    """
    return aggregate_maps(upsampled_maps, normalize=normalize)


# ---------------------------------------------------------------------------
# Optimized single-tensor aggregation (used by __main__ fast path)
# ---------------------------------------------------------------------------

def aggregate_mean_attn(
    mean_attn: torch.Tensor,
    n_mels: int = 100,
    T_spec: Optional[int] = None,
    normalize: str = "token",
) -> torch.Tensor:
    """Upsample + head-average + normalize from a single mean-attention tensor.

    This is the memory-efficient fast path that replaces the dict-based
    ``upsample_attn`` -> ``aggregate_attn`` chain.  It operates on a
    single pre-averaged tensor, avoiding the need to materialise hundreds
    of intermediate tensors.

    Parameters
    ----------
    mean_attn : Tensor  (B, H, audio_frames, C)
        Mean attention over all ODE steps and layers, as returned by
        ``extract_attn()["attention_mean"]``.
    n_mels : int
        Number of mel frequency bins (default 100).
    T_spec : int or None
        Target time frames.  Defaults to ``audio_frames``.
    normalize : str
        ``"token"`` | ``"global"`` | ``"none"``.

    Returns
    -------
    Tensor  (B, C, n_mels, T_spec)
    """
    if T_spec is None:
        T_spec = mean_attn.shape[2]

    upsampled = upsample_map(mean_attn, n_mels=n_mels, T_spec=T_spec)
    # upsampled: (B, H, C, n_mels, T_spec)
    agg = upsampled.mean(dim=1)  # average over heads -> (B, C, n_mels, T_spec)

    if normalize == "token":
        cmin = agg.flatten(2).min(dim=2).values[:, :, None, None]
        cmax = agg.flatten(2).max(dim=2).values[:, :, None, None]
        span = (cmax - cmin).clamp(min=1e-8)
        agg = (agg - cmin) / span
    elif normalize == "global":
        for b in range(agg.shape[0]):
            gmin = agg[b].min()
            gmax = agg[b].max()
            agg[b] = (agg[b] - gmin) / max(gmax - gmin, 1e-8)

    return agg


# ---------------------------------------------------------------------------
# Task 4 — Visualization
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# __main__ — run 5 example visualizations end-to-end
# ---------------------------------------------------------------------------

EXAMPLES = [
    {
        "transcript": "From these genres and from these spaces, you know, and the feelings of what these games can bring.",
        "caption": "An elderly woman, with a low-pitched voice, delivers her speech in a slow, yet expressive and animated manner. Her words flow like a captivating story, each sentence filled with emotion and wisdom, resonating deeply with her audience.",
    },
    {
        "transcript": "Use hir or ze as gender neutral pronouns?",
        "caption": "A male speaker delivers his words in a measured pace, exhibiting a high-pitched, happy, and animated tone in a clean environment.",
    },
    {
        "transcript": "in germany they generally hock the kaiser",
        "caption": "Her voice, a combination of feminine allure and intellectual brilliance, resonates with a sense of calm and elegance, making her every word a testament to her cool sophistication.",
    },
    {
        "transcript": "i want to see what he was when he was bright and young before the world had hardened him",
        "caption": "A mature male voice, rough and husky, ideal for public speaking engagements.",
    },
    {
        "transcript": "If only I had pursued my passion for dance earlier, I could have become a professional dancer.",
        "caption": "A voice tinged with regret, conveying a sense of longing for what could have been.",
    },
]


if __name__ == "__main__":
    import argparse
    import math
    import matplotlib
    matplotlib.use("Agg")

    import soundfile as sf
    from capspeech.nar.generate import load_model, encode, get_duration
    from capspeech.nar.utils import make_pad_mask

    parser = argparse.ArgumentParser(description="Generate DAAM visualizations for CapSpeech")
    parser.add_argument("--task", type=str, default="CapTTS",
                        choices=["PT", "CapTTS", "EmoCapTTS", "AccCapTTS", "AgentTTS"])
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device string, or 'auto' to use GPU (cuda:0) when available else CPU.",
    )
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--max_tokens", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.device == "auto":
        args.device = pick_inference_device()

    from capspeech.nar.generate import seed_everything
    seed_everything(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Device: {args.device}")
    model_list = load_model(args.device, args.task)
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
        text = torch.LongTensor(text_tokens).unsqueeze(0).to(args.device)

        with torch.no_grad():
            batch_enc = caption_tokenizer(caption, return_tensors="pt")
            ori_token_ids = batch_enc["input_ids"].to(args.device)
            prompt = caption_encoder(input_ids=ori_token_ids).last_hidden_state.squeeze().unsqueeze(0).to(args.device)

            tag_embed = clap_model.get_text_embedding([tag], use_tensor=True)
            clap = tag_embed.squeeze().unsqueeze(0).to(args.device)

            duration_inputs = caption + " <NEW_SEP> " + transcript
            duration_inputs = duration_tokenizer(
                duration_inputs, return_tensors="pt",
                padding="max_length", truncation=True, max_length=400,
            )
            predicted_dur = duration_model(**duration_inputs).logits.squeeze().item()
            duration = get_duration(transcript, predicted_dur)

        audio_clips = torch.zeros([1, math.ceil(duration * 24000 / 256), 100]).to(args.device)
        seq_len_prompt = prompt.shape[1]
        prompt_lens = torch.Tensor([seq_len_prompt])
        prompt_mask = make_pad_mask(prompt_lens, seq_len_prompt).to(prompt.device)

        result = extract_attn(
            model, vocoder, audio_clips, None, text, prompt, clap, prompt_mask,
            steps=args.steps, cfg=args.cfg, sway_sampling_coef=-1.0, device=args.device,
        )

        mel_spec = result["mel"]
        n_mels = mel_spec.shape[1]
        T_spec = mel_spec.shape[2]

        heatmaps = aggregate_mean_attn(
            result["attention_mean"], n_mels=n_mels, T_spec=T_spec, normalize="token",
        )

        token_ids = ori_token_ids.squeeze().tolist()
        token_labels = caption_tokenizer.convert_ids_to_tokens(token_ids)

        fig_path = os.path.join(args.output_dir, f"example_{idx}.png")
        wav_path = os.path.join(args.output_dir, f"example_{idx}.wav")

        visualize_maps(
            heatmaps, mel_spec, token_labels,
            save_path=fig_path, max_tokens=args.max_tokens,
        )
        import matplotlib.pyplot as plt
        plt.close("all")

        sf.write(wav_path, result["wav"], 24000)
        print(f"  Saved: {fig_path}, {wav_path}")

    print(f"\nAll 5 examples saved to {args.output_dir}/")
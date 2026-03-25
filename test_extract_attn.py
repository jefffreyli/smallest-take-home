"""
Verification script for daam_capspeech.py.

Creates a small CrossDiT model with random weights and verifies that:
1. CrossAttnCaptureProcessor produces identical outputs to AttnProcessor
2. CapSpeechAttentionHooker correctly locates all cross-attention layers
3. AttentionStore accumulates maps with expected shapes
4. extract_attn() runs end-to-end and returns well-formed results
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "CapSpeech"))

import torch
import torch.nn as nn
import numpy as np

from capspeech.nar.model.modules import Attention, AttnProcessor, CrossDiTBlock, create_mask
from capspeech.nar.network.crossdit import CrossDiT

from daam import AttentionStore, CrossAttnCaptureProcessor, CapSpeechAttentionHooker
from daam_capspeech import extract_attn, upsample_attn, aggregate_attn, visualize_maps

DEVICE = "cpu"
torch.manual_seed(42)


def test_processor_equivalence():
    """Verify the capture processor gives identical output to AttnProcessor."""
    print("=== Test 1: Processor numerical equivalence ===")

    dim, heads, dim_head = 64, 4, 16
    batch, audio_len, ctx_len = 1, 20, 10

    original_proc = AttnProcessor()
    attn_module = Attention(
        processor=original_proc, dim=dim, heads=heads,
        dim_head=dim_head, dropout=0.0, qk_norm=True,
    ).to(DEVICE).eval()

    x = torch.randn(batch, audio_len, dim, device=DEVICE)
    c = torch.randn(batch, ctx_len, dim, device=DEVICE)

    with torch.no_grad():
        out_original = attn_module(x=x, c=c, mask=None, rope=None)

    store = AttentionStore()
    capture_proc = CrossAttnCaptureProcessor(
        store=store, layer_idx=0, original_processor=original_proc,
    )
    attn_module.processor = capture_proc

    with torch.no_grad():
        out_capture = attn_module(x=x, c=c, mask=None, rope=None)

    attn_module.processor = original_proc

    max_diff = (out_original - out_capture).abs().max().item()
    print(f"  Max absolute difference: {max_diff:.2e}")
    assert max_diff < 1e-5, f"Outputs differ by {max_diff}"

    maps = store.get_all()
    assert len(maps) == 1, f"Expected 1 map, got {len(maps)}"
    key = (0, 0)
    assert key in maps, f"Expected key {key} in maps"
    attn_tensor = maps[key]
    assert attn_tensor.shape == (batch, heads, audio_len, ctx_len), (
        f"Expected shape {(batch, heads, audio_len, ctx_len)}, got {attn_tensor.shape}"
    )

    row_sums = attn_tensor.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
        "Attention weights don't sum to 1 along key dimension"
    )

    print("  PASSED\n")


def test_hooker_discovery():
    """Verify the hooker finds the correct number of cross-attn layers."""
    print("=== Test 2: Hooker layer discovery ===")

    depth = 4
    model = CrossDiT(
        dim=64, depth=depth, heads=4, dim_head=16,
        mel_dim=8, t5_dim=32, clap_dim=16,
        text_num_embeds=30, text_dim=8, conv_layers=0,
        skip=False, use_checkpoint=False, qk_norm=True,
    ).to(DEVICE).eval()

    expected_layers = depth // 2 + 1 + depth // 2  # in + mid + out
    found = CapSpeechAttentionHooker._find_cross_attn_modules(model)
    assert len(found) == expected_layers, (
        f"Expected {expected_layers} cross-attn modules, found {len(found)}"
    )
    print(f"  Found {len(found)} cross-attention layers (depth={depth})")

    hooker = CapSpeechAttentionHooker(model)
    hooker.hook()
    for blk in list(model.in_blocks) + [model.mid_block] + list(model.out_blocks):
        assert isinstance(blk.cross_attn.processor, CrossAttnCaptureProcessor)
    hooker.unhook()
    for blk in list(model.in_blocks) + [model.mid_block] + list(model.out_blocks):
        assert isinstance(blk.cross_attn.processor, AttnProcessor)

    print("  PASSED\n")


def test_extract_attn_e2e():
    """End-to-end test with a tiny model and a dummy vocoder."""
    print("=== Test 3: extract_attn end-to-end ===")

    depth = 4
    mel_dim = 8
    model = CrossDiT(
        dim=64, depth=depth, heads=4, dim_head=16,
        mel_dim=mel_dim, t5_dim=32, clap_dim=16,
        text_num_embeds=30, text_dim=8, conv_layers=0,
        skip=False, use_checkpoint=False, qk_norm=True,
    ).to(DEVICE).eval()

    class DummyVocoder(nn.Module):
        def forward(self, mel):
            b, d, n = mel.shape
            return torch.randn(b, 1, n * 256)

    vocoder = DummyVocoder().eval()

    batch = 1
    audio_len = 30
    caption_tokens = 12
    text_len = 15
    ode_steps = 3

    x = torch.zeros(batch, audio_len, mel_dim, device=DEVICE)
    text = torch.randint(0, 29, (batch, text_len), device=DEVICE)
    prompt = torch.randn(batch, caption_tokens, 32, device=DEVICE)
    clap = torch.randn(batch, 16, device=DEVICE)
    prompt_mask = torch.ones(batch, caption_tokens, dtype=torch.bool, device=DEVICE)

    result = extract_attn(
        model, vocoder, x, None, text, prompt, clap, prompt_mask,
        steps=ode_steps, cfg=2.0, sway_sampling_coef=-1.0, device=DEVICE,
    )

    assert "attention_maps" in result
    assert "wav" in result
    assert "metadata" in result

    maps = result["attention_maps"]
    meta = result["metadata"]
    expected_layers = depth // 2 + 1 + depth // 2

    print(f"  metadata: {meta}")
    # Euler with N time points evaluates fn N-1 times
    expected_fn_evals = ode_steps - 1
    assert meta["num_steps"] == expected_fn_evals, (
        f"Expected {expected_fn_evals} fn evaluations, got {meta['num_steps']}"
    )
    assert meta["num_layers"] == expected_layers, (
        f"Expected {expected_layers} layers, got {meta['num_layers']}"
    )

    expected_entries = expected_fn_evals * expected_layers
    assert len(maps) == expected_entries, (
        f"Expected {expected_entries} map entries, got {len(maps)}"
    )

    for (step, layer), tensor in maps.items():
        assert 0 <= step < expected_fn_evals
        assert 0 <= layer < expected_layers
        assert tensor.shape[0] == batch
        assert tensor.shape[1] == 4  # heads
        assert tensor.shape[2] == audio_len  # audio frames
        assert tensor.shape[3] == caption_tokens

    assert isinstance(result["wav"], np.ndarray)
    print(f"  wav shape: {result['wav'].shape}")

    for blk in list(model.in_blocks) + [model.mid_block] + list(model.out_blocks):
        assert isinstance(blk.cross_attn.processor, AttnProcessor), (
            "Processor was not restored after extract_attn"
        )

    print("  PASSED\n")


def test_upsample_attn():
    """Verify upsample_attn produces correct shapes and properties."""
    print("=== Test 4: upsample_attn ===")

    batch, heads, audio_frames, caption_tokens = 1, 4, 30, 12
    n_mels = 16

    raw_maps = {
        (0, 0): torch.rand(batch, heads, audio_frames, caption_tokens),
        (0, 1): torch.rand(batch, heads, audio_frames, caption_tokens),
        (1, 0): torch.rand(batch, heads, audio_frames, caption_tokens),
        (1, 1): torch.rand(batch, heads, audio_frames, caption_tokens),
    }

    # --- Key preservation ---
    up = upsample_attn(raw_maps, n_mels=n_mels)
    assert set(up.keys()) == set(raw_maps.keys()), "Keys differ after upsample"

    # --- Shape correctness (T_spec defaults to audio_frames) ---
    for key, tensor in up.items():
        assert tensor.shape == (batch, heads, caption_tokens, n_mels, audio_frames), (
            f"Expected {(batch, heads, caption_tokens, n_mels, audio_frames)}, "
            f"got {tensor.shape}"
        )

    # --- Non-negativity ---
    for tensor in up.values():
        assert (tensor >= 0).all(), "Found negative values after upsample"

    # --- Frequency uniformity: all mel rows should be identical ---
    for tensor in up.values():
        first_row = tensor[:, :, :, 0, :]
        for m in range(1, n_mels):
            assert torch.allclose(tensor[:, :, :, m, :], first_row, atol=1e-5), (
                f"Mel row {m} differs from row 0 — frequency should be uniform"
            )

    # --- Identity case: T_spec == audio_frames, check time values preserved ---
    single = torch.rand(batch, heads, audio_frames, caption_tokens)
    single_maps = {(0, 0): single}
    up_single = upsample_attn(single_maps, n_mels=1, T_spec=audio_frames)
    recovered = up_single[(0, 0)]
    assert recovered.shape == (batch, heads, caption_tokens, 1, audio_frames)
    original_per_token = single.permute(0, 1, 3, 2)  # (B, H, C, F)
    recovered_squeezed = recovered.squeeze(3)         # (B, H, C, F)
    assert torch.allclose(original_per_token, recovered_squeezed, atol=1e-5), (
        "Time content changed when T_spec == audio_frames and n_mels == 1"
    )

    # --- Custom T_spec (time resampling) ---
    T_new = 60
    up_resized = upsample_attn(raw_maps, n_mels=n_mels, T_spec=T_new)
    for tensor in up_resized.values():
        assert tensor.shape == (batch, heads, caption_tokens, n_mels, T_new), (
            f"Expected T_spec={T_new}, got shape {tensor.shape}"
        )

    print("  PASSED\n")


def test_aggregate_attn():
    """Verify aggregate_attn produces correct shapes, normalization, and edge cases."""
    print("=== Test 5: aggregate_attn ===")

    batch, heads, caption_tokens, n_mels, T_spec = 2, 4, 6, 8, 20

    upsampled = {
        (s, l): torch.rand(batch, heads, caption_tokens, n_mels, T_spec) + 0.1
        for s in range(3) for l in range(2)
    }

    # --- 1. Shape ---
    out = aggregate_attn(upsampled, normalize="none")
    assert out.shape == (batch, caption_tokens, n_mels, T_spec), (
        f"Expected {(batch, caption_tokens, n_mels, T_spec)}, got {out.shape}"
    )
    print("  Shape check passed")

    # --- 2. Token normalization ---
    out_tok = aggregate_attn(upsampled, normalize="token")
    assert out_tok.shape == (batch, caption_tokens, n_mels, T_spec)
    for b in range(batch):
        for c in range(caption_tokens):
            heatmap = out_tok[b, c]
            assert abs(heatmap.min().item()) < 1e-6, (
                f"Token ({b},{c}) min should be ~0, got {heatmap.min().item()}"
            )
            assert abs(heatmap.max().item() - 1.0) < 1e-6, (
                f"Token ({b},{c}) max should be ~1, got {heatmap.max().item()}"
            )
    print("  Token normalization check passed")

    # --- 3. Global normalization ---
    out_glob = aggregate_attn(upsampled, normalize="global")
    assert out_glob.shape == (batch, caption_tokens, n_mels, T_spec)
    for b in range(batch):
        all_tokens = out_glob[b]  # (C, n_mels, T_spec)
        assert abs(all_tokens.min().item()) < 1e-6, (
            f"Batch {b} global min should be ~0, got {all_tokens.min().item()}"
        )
        assert abs(all_tokens.max().item() - 1.0) < 1e-6, (
            f"Batch {b} global max should be ~1, got {all_tokens.max().item()}"
        )
    print("  Global normalization check passed")

    # --- 4. No normalization returns raw means ---
    out_none = aggregate_attn(upsampled, normalize="none")
    stacked = torch.stack(list(upsampled.values()), dim=0)
    expected = stacked.mean(dim=0).mean(dim=1)
    assert torch.allclose(out_none, expected, atol=1e-6), (
        "normalize='none' should return raw mean values"
    )
    print("  No normalization check passed")

    # --- 5. Single entry edge case ---
    single = {(0, 0): torch.rand(batch, heads, caption_tokens, n_mels, T_spec)}
    out_single = aggregate_attn(single, normalize="token")
    assert out_single.shape == (batch, caption_tokens, n_mels, T_spec)
    for b in range(batch):
        for c in range(caption_tokens):
            heatmap = out_single[b, c]
            assert abs(heatmap.min().item()) < 1e-6
            assert abs(heatmap.max().item() - 1.0) < 1e-6
    print("  Single entry edge case passed")

    print("  PASSED\n")


def test_visualize_maps():
    """Verify visualize_maps produces a figure and saves to disk."""
    print("=== Test 6: visualize_maps ===")
    import tempfile
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.figure

    batch, caption_tokens, n_mels, T_spec = 1, 8, 16, 40

    heatmaps = torch.rand(batch, caption_tokens, n_mels, T_spec)
    mel = torch.randn(batch, n_mels, T_spec)
    labels = [f"tok_{i}" for i in range(caption_tokens)]

    fig = visualize_maps(heatmaps, mel, labels, max_tokens=None)
    assert isinstance(fig, matplotlib.figure.Figure), "Expected a matplotlib Figure"
    drawn = [ax for ax in fig.get_axes() if ax.images]
    assert len(drawn) == 1 + caption_tokens, (
        f"Expected {1 + caption_tokens} drawn panels, got {len(drawn)}"
    )
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("  Figure creation passed")

    # --- Save to temp file ---
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
    fig2 = visualize_maps(heatmaps, mel, labels, save_path=tmp_path, max_tokens=5)
    assert os.path.isfile(tmp_path), f"File not created at {tmp_path}"
    assert os.path.getsize(tmp_path) > 0, "Saved file is empty"
    drawn2 = [ax for ax in fig2.get_axes() if ax.images]
    assert len(drawn2) == 1 + 5, (
        f"Expected {1 + 5} drawn panels with max_tokens=5, got {len(drawn2)}"
    )
    plt.close(fig2)
    os.unlink(tmp_path)
    print("  Save and max_tokens limiting passed")

    # --- 2D mel input (no batch dim) ---
    mel_2d = torch.randn(n_mels, T_spec)
    fig3 = visualize_maps(heatmaps, mel_2d, labels, max_tokens=3, batch_idx=0)
    assert isinstance(fig3, matplotlib.figure.Figure)
    plt.close(fig3)
    print("  2D mel input passed")

    print("  PASSED\n")


if __name__ == "__main__":
    test_processor_equivalence()
    test_hooker_discovery()
    test_extract_attn_e2e()
    test_upsample_attn()
    test_aggregate_attn()
    test_visualize_maps()
    print("All tests passed!")

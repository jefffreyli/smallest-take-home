"""Drop-in attention processor that captures cross-attention weights."""

from __future__ import annotations

import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "CapSpeech"))

from capspeech.nar.model.modules import (
    Attention,
    AttnProcessor,
    create_mask,
)

from daam.store import AttentionStore


class CrossAttnCaptureProcessor:
    """Replaces ``AttnProcessor`` on a single ``Attention`` module to
    manually compute softmax(QK^T / sqrt(d)) and store the resulting
    attention-weight matrix before multiplying by V.

    The processor faithfully replicates the original ``AttnProcessor``
    logic (Q/K/V projection, QK-norm, optional RoPE, masking, output
    projection) so the model's numerical output is unchanged.
    """

    def __init__(
        self,
        store: AttentionStore,
        layer_idx: int,
        original_processor: AttnProcessor,
    ):
        self.store = store
        self.layer_idx = layer_idx
        self.original_processor = original_processor
        self.capture_enabled: bool = True

    def __call__(
        self,
        attn: Attention,
        x,
        mask=None,
        rope=None,
        c=None,
        c_rope=None,
    ) -> torch.FloatTensor:

        batch_size = x.shape[0]

        if c is None:
            c = x

        query = attn.to_q(x)
        key = attn.to_k(c)
        value = attn.to_v(c)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.q_norm is not None:
            query = attn.q_norm(query)
        if attn.k_norm is not None:
            key = attn.k_norm(key)

        if rope is not None:
            from capspeech.nar.model.modules import apply_rotary_pos_emb
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (
                (xpos_scale, xpos_scale ** -1.0) if xpos_scale is not None else (1.0, 1.0)
            )
            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        if mask is not None:
            attn_mask = create_mask(x.shape, c.shape, x.device, None, mask)
        else:
            attn_mask = None

        # Manual attention weight computation (replaces F.scaled_dot_product_attention)
        scale = head_dim ** -0.5
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale

        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(~attn_mask, float("-inf"))

        attn_weights = torch.softmax(attn_weights, dim=-1)

        if self.capture_enabled:
            self.store.update(self.layer_idx, attn_weights)

        out = torch.matmul(attn_weights, value)
        out = out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        out = out.to(query.dtype)

        out = attn.to_out[0](out)
        out = attn.to_out[1](out)

        return out

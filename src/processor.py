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

from src.store import AttentionStore
from capspeech.nar.model.modules import apply_rotary_pos_emb


class CrossAttnCaptureProcessor:
    """Replaces ``AttnProcessor`` on a single ``Attention`` module to
    manually compute softmax(QK^T / sqrt(d)) and store the resulting
    attention-weight matrix before multiplying by V.
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
        self.capture_enabled: bool = True # CFG control. Only the conditional forward pass contributes to the attention store.

    # use __call__ dunder method to make the class callable to mimic the original AttnProcessor module
    def __call__(self, attn: Attention, x, mask=None, rope=None, c=None, c_rope=None) -> torch.FloatTensor:
        """Call the processor to compute the attention weights.

        Args:
            attn (Attention): The attention module.
            x (torch.FloatTensor): The audio sequence. (batch, audio_frames, audio_dim)
            mask (torch.BoolTensor, optional): The padding mask. (batch, audio_frames)
            rope (tuple, optional): The rotary position embedding. (freqs, xpos_scale)
            c (torch.FloatTensor, optional): The context. (batch, caption_tokens, caption_dim)
            c_rope (tuple, optional): The context rotary position embedding. (freqs, xpos_scale)

        Returns:
            torch.FloatTensor: The attention score calculated by the attention module. (batch, audio_frames, audio_dim)
        """
        batch_size = x.shape[0]

        if c is None:
            c = x

        # linear projections
        query = attn.to_q(x) # maps hidden vector of size dim to size inner_dim (heads * head_dim)
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

        # apply rotary position embeddings
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (
                (xpos_scale, xpos_scale ** -1.0) if xpos_scale is not None else (1.0, 1.0)
            )
            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        # create attention mask
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

        # capture the attention weights if capture is enabled
        if self.capture_enabled:
            self.store.update(self.layer_idx, attn_weights)

        # get weighted value matrix
        out = torch.matmul(attn_weights, value)
        out = out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        out = out.to(query.dtype)

        # linear projection and dropout
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)

        return out

"""Perceiver IO fusion: learnable latent array with cross-attention.

Based on DeepMind's Perceiver IO (arXiv 2107.14795).
Key architectural elements:
- Latent array queries input via cross-attention
- FFN after cross-attention (widening_factor=1)
- Pre-norm (LayerNorm on both Q and KV before attention)
- Self-attention blocks between cross-attention layers
"""
from __future__ import annotations

import torch
import torch.nn as nn

from mac.fusion.base import BaseFusionModule


class CrossAttentionBlock(nn.Module):
    """Cross-attention with FFN, matching the original Perceiver pattern.

    Pattern: LN(Q) + LN(KV) -> MultiheadAttention -> dropout -> residual -> LN -> FFN -> residual
    """

    def __init__(
        self,
        d_latent: int,
        d_input: int,
        n_heads: int,
        dropout: float = 0.1,
        ffn_widening: int = 1,
    ) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_latent)
        self.norm_kv = nn.LayerNorm(d_input)
        self.attn = nn.MultiheadAttention(
            d_latent, n_heads, dropout=dropout, batch_first=True,
            kdim=d_input, vdim=d_input,
        )
        self.attn_dropout = nn.Dropout(dropout)

        # FFN after cross-attention (widening_factor=1 in original)
        self.norm_ffn = nn.LayerNorm(d_latent)
        ffn_dim = d_latent * ffn_widening
        self.ffn = nn.Sequential(
            nn.Linear(d_latent, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_latent),
            nn.Dropout(dropout),
        )

    def forward(
        self, latents: torch.Tensor, kv: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Cross-attention with pre-norm on both Q and KV
        residual = latents
        q = self.norm_q(latents)
        kv_normed = self.norm_kv(kv)
        attended, _ = self.attn(q, kv_normed, kv_normed, key_padding_mask=key_padding_mask)
        latents = residual + self.attn_dropout(attended)

        # FFN with pre-norm
        residual = latents
        latents = residual + self.ffn(self.norm_ffn(latents))
        return latents


class PerceiverIOFusion(BaseFusionModule):
    supports_sequence_input = True

    """Perceiver IO: latent array cross-attends to all modality inputs.

    Architecture per layer:
    1. CrossAttentionBlock: latents query all modality tokens (with FFN)
    2. Self-attention block: latents refine via self-attention (with FFN)
    """

    def __init__(
        self,
        d_common: int = 256,
        n_latents: int = 32,
        d_latent: int | None = None,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        d_latent = d_latent or d_common
        super().__init__(d_common=d_common, d_out=d_latent)

        self.latents = nn.Parameter(torch.randn(1, n_latents, d_latent) * 0.02)
        self.modality_embed = nn.Embedding(16, d_common)

        self.cross_attn_blocks = nn.ModuleList()
        self.self_attn_blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.cross_attn_blocks.append(
                CrossAttentionBlock(d_latent, d_common, n_heads, dropout, ffn_widening=1)
            )
            self.self_attn_blocks.append(
                nn.TransformerEncoderLayer(
                    d_latent, n_heads, dim_feedforward=d_latent * 4,
                    dropout=dropout, batch_first=True, norm_first=True,
                )
            )

        self.output_norm = nn.LayerNorm(d_latent)
        self.output_proj = nn.Linear(d_latent, d_latent)

    def forward(
        self,
        embeddings: list[torch.Tensor],
        modality_ids: list[str],
        masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        B = embeddings[0].shape[0]
        device = embeddings[0].device

        # Prepare input: add modality embeddings, concatenate
        processed = []
        mask_parts = []
        any_mask = masks is not None and any(m is not None for m in masks)

        for i, emb in enumerate(embeddings):
            if emb.dim() == 2:
                emb = emb.unsqueeze(1)
            T_i = emb.shape[1]
            emb = emb + self.modality_embed.weight[i].unsqueeze(0).unsqueeze(0)
            processed.append(emb)

            if any_mask:
                if masks[i] is not None:
                    mask_parts.append(masks[i])
                else:
                    # All tokens valid for this modality
                    mask_parts.append(torch.ones(B, T_i, dtype=torch.bool, device=device))

        kv = torch.cat(processed, dim=1)
        # key_padding_mask: True = IGNORE (PyTorch convention)
        key_padding_mask = ~torch.cat(mask_parts, dim=1) if any_mask else None

        # Expand latents for batch
        latents = self.latents.expand(B, -1, -1)

        # Iterative cross-attention + self-attention
        for cross_block, self_block in zip(self.cross_attn_blocks, self.self_attn_blocks):
            latents = cross_block(latents, kv, key_padding_mask=key_padding_mask)
            latents = self_block(latents)

        # Output: normalize, mean pool, project
        latents = self.output_norm(latents)
        out = latents.mean(dim=1)
        return self.output_proj(out)

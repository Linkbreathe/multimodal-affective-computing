"""Perceiver IO fusion: learnable latent array with cross-attention."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.fusion.base import BaseFusionModule


class PerceiverIOFusion(BaseFusionModule):
    """Perceiver IO: latent array cross-attends to all modality inputs."""

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
        self.n_latents = n_latents

        self.latents = nn.Parameter(torch.randn(1, n_latents, d_latent) * 0.02)
        self.modality_embed = nn.Embedding(16, d_common)

        self.cross_attn_layers = nn.ModuleList()
        self.self_attn_layers = nn.ModuleList()
        self.cross_norms = nn.ModuleList()
        for _ in range(n_layers):
            self.cross_attn_layers.append(
                nn.MultiheadAttention(
                    d_latent,
                    n_heads,
                    dropout=dropout,
                    batch_first=True,
                    kdim=d_common,
                    vdim=d_common,
                )
            )
            self.self_attn_layers.append(
                nn.TransformerEncoderLayer(
                    d_latent,
                    n_heads,
                    dim_feedforward=d_latent * 4,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                )
            )
            self.cross_norms.append(nn.LayerNorm(d_latent))

        self.output_proj = nn.Linear(d_latent, d_latent)

    def forward(
        self,
        embeddings: list[torch.Tensor],
        modality_ids: list[str],
        masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        B = embeddings[0].shape[0]

        processed = []
        all_masks = []
        for i, emb in enumerate(embeddings):
            if emb.dim() == 2:
                emb = emb.unsqueeze(1)
            emb = emb + self.modality_embed.weight[i].unsqueeze(0).unsqueeze(0)
            processed.append(emb)
            if masks is not None:
                m = masks[i]
                if m.dim() == 1:
                    m = m.unsqueeze(1)
                all_masks.append(m)

        kv = torch.cat(processed, dim=1)
        if all_masks:
            kv_mask = torch.cat(all_masks, dim=1)
            key_padding_mask = ~kv_mask
        else:
            key_padding_mask = None

        latents = self.latents.expand(B, -1, -1)

        for cross_attn, self_attn, norm in zip(
            self.cross_attn_layers, self.self_attn_layers, self.cross_norms
        ):
            residual = latents
            latents_normed = norm(latents)
            attended, _ = cross_attn(
                latents_normed, kv, kv, key_padding_mask=key_padding_mask
            )
            latents = residual + attended
            latents = self_attn(latents)

        out = latents.mean(dim=1)
        return self.output_proj(out)

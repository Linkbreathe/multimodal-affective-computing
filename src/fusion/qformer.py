"""Q-Former fusion: learnable query tokens with cross-attention to modalities."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.fusion.base import BaseFusionModule


class QFormerFusion(BaseFusionModule):
    """Q-Former: learnable queries extract task-relevant info via cross-attention."""

    def __init__(
        self,
        d_common: int = 256,
        n_queries: int = 16,
        d_query: int | None = None,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        d_query = d_query or d_common
        super().__init__(d_common=d_common, d_out=d_query)

        self.queries = nn.Parameter(torch.randn(1, n_queries, d_query) * 0.02)
        self.modality_embed = nn.Embedding(16, d_common)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "self_attn": nn.TransformerEncoderLayer(
                    d_query, n_heads, dim_feedforward=d_query * 4,
                    dropout=dropout, batch_first=True, norm_first=True,
                ),
                "cross_attn": nn.MultiheadAttention(
                    d_query, n_heads, dropout=dropout, batch_first=True,
                    kdim=d_common, vdim=d_common,
                ),
                "cross_norm": nn.LayerNorm(d_query),
            }))

        self.output_proj = nn.Linear(d_query, d_query)

    def forward(
        self,
        embeddings: list[torch.Tensor],
        modality_ids: list[str],
        masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        B = embeddings[0].shape[0]

        processed = []
        for i, emb in enumerate(embeddings):
            if emb.dim() == 2:
                emb = emb.unsqueeze(1)
            emb = emb + self.modality_embed.weight[i]
            processed.append(emb)

        kv = torch.cat(processed, dim=1)
        queries = self.queries.expand(B, -1, -1)

        for layer in self.layers:
            queries = layer["self_attn"](queries)
            residual = queries
            q_normed = layer["cross_norm"](queries)
            attended, _ = layer["cross_attn"](q_normed, kv, kv)
            queries = residual + attended

        out = queries.mean(dim=1)
        return self.output_proj(out)

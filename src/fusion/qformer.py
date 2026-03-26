"""Q-Former fusion: learnable query tokens with cross-attention to modalities.

Based on BLIP-2's Q-Former (arXiv 2301.12597, Salesforce).
Key architectural elements from the original:
- Learnable query tokens that cross-attend to multimodal inputs
- Cross-attention interleaved (every other layer by default)
- Post-norm residual connections (BERT-style: LayerNorm(x + sublayer(x)))
- Dedicated FFN after cross-attention
- GELU activation in FFN
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.fusion.base import BaseFusionModule


class QFormerLayer(nn.Module):
    """Single Q-Former layer with optional cross-attention.

    Uses post-norm (BERT-style) residual connections.
    """

    def __init__(
        self,
        d_query: int,
        d_kv: int,
        n_heads: int,
        has_cross_attention: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Self-attention (always present)
        self.self_attn = nn.MultiheadAttention(
            d_query, n_heads, dropout=dropout, batch_first=True,
        )
        self.self_attn_norm = nn.LayerNorm(d_query)
        self.self_attn_dropout = nn.Dropout(dropout)

        # Self-attention FFN
        self.self_ffn = nn.Sequential(
            nn.Linear(d_query, d_query * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_query * 4, d_query),
            nn.Dropout(dropout),
        )
        self.self_ffn_norm = nn.LayerNorm(d_query)

        # Cross-attention (only in some layers)
        self.has_cross_attention = has_cross_attention
        if has_cross_attention:
            self.cross_attn = nn.MultiheadAttention(
                d_query, n_heads, dropout=dropout, batch_first=True,
                kdim=d_kv, vdim=d_kv,
            )
            self.cross_attn_norm = nn.LayerNorm(d_query)
            self.cross_attn_dropout = nn.Dropout(dropout)

            # Dedicated query FFN after cross-attention
            self.cross_ffn = nn.Sequential(
                nn.Linear(d_query, d_query * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_query * 4, d_query),
                nn.Dropout(dropout),
            )
            self.cross_ffn_norm = nn.LayerNorm(d_query)

    def forward(
        self,
        queries: torch.Tensor,
        kv: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Self-attention (post-norm: LayerNorm(x + sublayer(x)))
        attn_out, _ = self.self_attn(queries, queries, queries)
        queries = self.self_attn_norm(queries + self.self_attn_dropout(attn_out))

        # Self-attention FFN (post-norm)
        ffn_out = self.self_ffn(queries)
        queries = self.self_ffn_norm(queries + ffn_out)

        # Cross-attention + cross-FFN (if this layer has it)
        if self.has_cross_attention and kv is not None:
            cross_out, _ = self.cross_attn(
                queries, kv, kv, key_padding_mask=key_padding_mask,
            )
            queries = self.cross_attn_norm(queries + self.cross_attn_dropout(cross_out))

            cross_ffn_out = self.cross_ffn(queries)
            queries = self.cross_ffn_norm(queries + cross_ffn_out)

        return queries


class QFormerFusion(BaseFusionModule):
    supports_sequence_input = True

    """Q-Former: learnable queries extract task-relevant info via cross-attention.

    Architecture:
    - Learnable query tokens
    - N layers, cross-attention every `cross_attn_freq` layers
    - Post-norm residual connections (BERT-style)
    - Dedicated FFN after both self-attention and cross-attention
    """

    def __init__(
        self,
        d_common: int = 256,
        n_queries: int = 16,
        d_query: int | None = None,
        n_layers: int = 6,
        n_heads: int = 4,
        cross_attn_freq: int = 2,
        dropout: float = 0.1,
    ) -> None:
        d_query = d_query or d_common
        super().__init__(d_common=d_common, d_out=d_query)

        self.queries = nn.Parameter(torch.zeros(1, n_queries, d_query))
        nn.init.normal_(self.queries, std=0.02)

        self.modality_embed = nn.Embedding(16, d_common)

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            has_cross = (i % cross_attn_freq == 0)
            self.layers.append(
                QFormerLayer(d_query, d_common, n_heads, has_cross, dropout)
            )

        self.output_norm = nn.LayerNorm(d_query)
        self.output_proj = nn.Linear(d_query, d_query)

    def forward(
        self,
        embeddings: list[torch.Tensor],
        modality_ids: list[str],
        masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        B = embeddings[0].shape[0]
        device = embeddings[0].device

        # Prepare KV from all modalities
        processed = []
        mask_parts = []
        any_mask = masks is not None and any(m is not None for m in masks)

        for i, emb in enumerate(embeddings):
            if emb.dim() == 2:
                emb = emb.unsqueeze(1)
            T_i = emb.shape[1]
            emb = emb + self.modality_embed.weight[i]
            processed.append(emb)

            if any_mask:
                if masks[i] is not None:
                    mask_parts.append(masks[i])
                else:
                    mask_parts.append(torch.ones(B, T_i, dtype=torch.bool, device=device))

        kv = torch.cat(processed, dim=1)
        # key_padding_mask: True = IGNORE (PyTorch convention)
        key_padding_mask = ~torch.cat(mask_parts, dim=1) if any_mask else None
        queries = self.queries.expand(B, -1, -1)

        for layer in self.layers:
            queries = layer(queries, kv, key_padding_mask=key_padding_mask)

        # Output: normalize, pool, project
        queries = self.output_norm(queries)
        out = queries.mean(dim=1)
        return self.output_proj(out)

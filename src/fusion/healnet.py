"""HEALNet: Hybrid Early-fusion Attention Learning Network."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.fusion.base import BaseFusionModule


class HEALNetFusion(BaseFusionModule):
    """Iterative early fusion with shared memory updated by each modality."""

    def __init__(
        self,
        d_common: int = 256,
        memory_size: int = 16,
        n_layers: int = 2,
        n_heads: int = 4,
        num_modalities: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(d_common=d_common, d_out=d_common)
        self.memory_size = memory_size
        self.init_memory = nn.Parameter(torch.randn(1, memory_size, d_common) * 0.02)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "cross_attn": nn.MultiheadAttention(
                    d_common, n_heads, dropout=dropout, batch_first=True,
                ),
                "norm_q": nn.LayerNorm(d_common),
                "norm_kv": nn.LayerNorm(d_common),
                "ffn": nn.Sequential(
                    nn.Linear(d_common, d_common * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_common * 4, d_common),
                    nn.Dropout(dropout),
                ),
                "norm_ffn": nn.LayerNorm(d_common),
            }))

        self.output_proj = nn.Linear(d_common, d_common)

    def forward(
        self,
        embeddings: list[torch.Tensor],
        modality_ids: list[str],
        masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        B = embeddings[0].shape[0]
        memory = self.init_memory.expand(B, -1, -1)

        processed = []
        for emb in embeddings:
            if emb.dim() == 2:
                emb = emb.unsqueeze(1)
            processed.append(emb)

        for layer in self.layers:
            for emb in processed:
                residual = memory
                q = layer["norm_q"](memory)
                kv = layer["norm_kv"](emb)
                attended, _ = layer["cross_attn"](q, kv, kv)
                memory = residual + attended
            residual = memory
            memory = residual + layer["ffn"](layer["norm_ffn"](memory))

        out = memory.mean(dim=1)
        return self.output_proj(out)

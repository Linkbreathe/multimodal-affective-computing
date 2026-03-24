"""Multimodal Lego: composable atomic fusion blocks."""
from __future__ import annotations

from itertools import combinations

import torch
import torch.nn as nn

from src.fusion.base import BaseFusionModule


class CrossAttnBlock(nn.Module):
    """Cross-attention between two modalities."""
    def __init__(self, d: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        if query.dim() == 2:
            query = query.unsqueeze(1)
        if key_value.dim() == 2:
            key_value = key_value.unsqueeze(1)
        residual = query
        q = self.norm(query)
        out, _ = self.attn(q, key_value, key_value)
        return (residual + out).squeeze(1)


class GatedFusionBlock(nn.Module):
    """Sigmoid-gated element-wise fusion."""
    def __init__(self, d: int, n_inputs: int):
        super().__init__()
        self.gate = nn.Linear(d * n_inputs, n_inputs)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        concat = torch.cat(inputs, dim=-1)
        gates = torch.sigmoid(self.gate(concat))
        stacked = torch.stack(inputs, dim=-1)
        gated = (stacked * gates.unsqueeze(1)).sum(dim=-1)
        return gated


class MultimodalLegoFusion(BaseFusionModule):
    """Composable fusion with configurable topology."""

    def __init__(
        self,
        d_common: int = 256,
        modality_ids: list[str] | None = None,
        topology: str = "pairwise",
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(d_common=d_common, d_out=d_common)
        modality_ids = modality_ids or ["video", "eye_tracking", "ppg"]
        self.topology = topology
        self.mod_ids = modality_ids
        n_mod = len(modality_ids)

        if topology == "pairwise":
            self.pair_blocks = nn.ModuleDict()
            for a, b in combinations(range(n_mod), 2):
                self.pair_blocks[f"{a}_{b}"] = CrossAttnBlock(d_common, n_heads, dropout)
                self.pair_blocks[f"{b}_{a}"] = CrossAttnBlock(d_common, n_heads, dropout)
            n_outputs = n_mod * (n_mod - 1)
            self.output_mlp = nn.Sequential(
                nn.Linear(d_common * n_outputs, d_common),
                nn.ReLU(),
                nn.Linear(d_common, d_common),
            )

        elif topology == "hierarchical":
            self.stage1 = CrossAttnBlock(d_common, n_heads, dropout)
            self.stage2 = CrossAttnBlock(d_common, n_heads, dropout)
            self.output_mlp = nn.Sequential(
                nn.Linear(d_common * 2, d_common),
                nn.ReLU(),
                nn.Linear(d_common, d_common),
            )

        elif topology == "gated":
            self.self_attns = nn.ModuleList([
                nn.TransformerEncoderLayer(d_common, n_heads, d_common * 4, dropout, batch_first=True, norm_first=True)
                for _ in range(n_mod)
            ])
            self.cross_blocks = nn.ModuleDict()
            for a, b in combinations(range(n_mod), 2):
                self.cross_blocks[f"{a}_{b}"] = CrossAttnBlock(d_common, n_heads, dropout)
            self.gate = GatedFusionBlock(d_common, n_mod)
            self.output_mlp = nn.Linear(d_common, d_common)

    def forward(
        self,
        embeddings: list[torch.Tensor],
        modality_ids: list[str],
        masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        embs = [e.squeeze(1) if e.dim() == 3 and e.shape[1] == 1 else e for e in embeddings]
        if embs[0].dim() == 3:
            embs = [e.mean(dim=1) for e in embs]

        if self.topology == "pairwise":
            outputs = []
            for (a, b) in combinations(range(len(embs)), 2):
                outputs.append(self.pair_blocks[f"{a}_{b}"](embs[a], embs[b]))
                outputs.append(self.pair_blocks[f"{b}_{a}"](embs[b], embs[a]))
            return self.output_mlp(torch.cat(outputs, dim=-1))

        elif self.topology == "hierarchical":
            physio_fused = self.stage1(embs[1], embs[2])
            combined = self.stage2(embs[0], physio_fused)
            return self.output_mlp(torch.cat([combined, physio_fused], dim=-1))

        elif self.topology == "gated":
            refined = [sa(e.unsqueeze(1)).squeeze(1) for sa, e in zip(self.self_attns, embs)]
            for (a, b) in combinations(range(len(refined)), 2):
                crossed = self.cross_blocks[f"{a}_{b}"](refined[a], refined[b])
                refined[a] = refined[a] + crossed
            gated = self.gate(refined)
            return self.output_mlp(gated)

        raise ValueError(f"Unknown topology: {self.topology}")

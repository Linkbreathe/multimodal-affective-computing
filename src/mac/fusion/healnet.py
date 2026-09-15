"""HEALNet: Hybrid Early-fusion Attention Learning Network.

Based on the original HEALNet (arXiv 2311.09115).
Key architectural elements:
- Shared latent bottleneck updated iteratively by each modality
- Per-modality cross-attention weights (each modality has its own attention)
- Cross-FFN immediately after each modality's cross-attention
- Self-attention on latent after each modality update
- Handles missing modalities by skipping their update step
"""
from __future__ import annotations

import torch
import torch.nn as nn

from mac.fusion.base import BaseFusionModule


class ModalityCrossAttention(nn.Module):
    """Per-modality cross-attention: latent queries attend to one modality.

    Pattern per modality:
    1. cross_attn(latent, modality) + residual
    2. cross_ffn(latent) + residual
    3. self_attn(latent) + residual
    4. self_ffn(latent) + residual
    """

    def __init__(self, d_latent: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        # Cross-attention
        self.norm_q = nn.LayerNorm(d_latent)
        self.norm_kv = nn.LayerNorm(d_latent)
        self.cross_attn = nn.MultiheadAttention(
            d_latent, n_heads, dropout=dropout, batch_first=True,
        )

        # Cross-FFN (immediately after cross-attention)
        self.norm_cross_ffn = nn.LayerNorm(d_latent)
        self.cross_ffn = nn.Sequential(
            nn.Linear(d_latent, d_latent * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_latent * 4, d_latent),
            nn.Dropout(dropout),
        )

        # Self-attention on latent
        self.norm_self = nn.LayerNorm(d_latent)
        self.self_attn = nn.MultiheadAttention(
            d_latent, n_heads, dropout=dropout, batch_first=True,
        )

        # Self-FFN
        self.norm_self_ffn = nn.LayerNorm(d_latent)
        self.self_ffn = nn.Sequential(
            nn.Linear(d_latent, d_latent * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_latent * 4, d_latent),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        latent: torch.Tensor,
        modality: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Cross-attention: latent attends to modality
        residual = latent
        q = self.norm_q(latent)
        kv = self.norm_kv(modality)
        attended, _ = self.cross_attn(q, kv, kv, key_padding_mask=key_padding_mask)
        latent = residual + attended

        # Cross-FFN
        residual = latent
        latent = residual + self.cross_ffn(self.norm_cross_ffn(latent))

        # Self-attention on latent
        residual = latent
        ln = self.norm_self(latent)
        self_attended, _ = self.self_attn(ln, ln, ln)
        latent = residual + self_attended

        # Self-FFN
        residual = latent
        latent = residual + self.self_ffn(self.norm_self_ffn(latent))

        return latent


class HEALNetFusion(BaseFusionModule):
    supports_sequence_input = True

    """Iterative early fusion with per-modality cross-attention on shared latent.

    For each layer, for each modality:
    1. Cross-attention: latent queries the modality
    2. Cross-FFN: transform latent
    3. Self-attention: latent refines internally
    4. Self-FFN: transform latent

    Missing modalities are skipped (latent not updated for that modality).
    """

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
        self.num_modalities = num_modalities
        self.init_memory = nn.Parameter(torch.randn(1, memory_size, d_common))

        # Per-modality cross-attention blocks, per layer
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            modality_blocks = nn.ModuleList([
                ModalityCrossAttention(d_common, n_heads, dropout)
                for _ in range(num_modalities)
            ])
            self.layers.append(modality_blocks)

        self.output_norm = nn.LayerNorm(d_common)
        self.output_proj = nn.Linear(d_common, d_common)

    def forward(
        self,
        embeddings: list[torch.Tensor],
        modality_ids: list[str],
        masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        B = embeddings[0].shape[0]
        memory = self.init_memory.expand(B, -1, -1).clone()

        # Prepare modality inputs (ensure 3D)
        processed = []
        for emb in embeddings:
            if emb.dim() == 2:
                emb = emb.unsqueeze(1)
            processed.append(emb)

        for modality_blocks in self.layers:
            for i, block in enumerate(modality_blocks):
                if i < len(processed):
                    # Per-modality mask: True=IGNORE (PyTorch convention)
                    kpm = None
                    if masks is not None and masks[i] is not None:
                        kpm = ~masks[i]
                    memory = block(memory, processed[i], key_padding_mask=kpm)
                # else: modality missing — skip (memory unchanged)

        memory = self.output_norm(memory)
        out = memory.mean(dim=1)
        return self.output_proj(out)

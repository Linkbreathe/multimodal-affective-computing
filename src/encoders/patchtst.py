"""PatchTST encoder for eye tracking (gaze + pupils).

Architecture: channel-independent patching -> Transformer encoder.
Pre-training: self-supervised masked patch prediction.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.encoders.base import BaseEncoder


class PatchTSTEncoder(BaseEncoder):
    """PatchTST for multivariate time series.

    Input: [B, T, C] (batch, time, channels)
    Output: [B, num_patches, d_model]
    """

    def __init__(
        self,
        num_channels: int = 4,
        patch_len: int = 45,
        stride: int = 22,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        seq_len: int = 900,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(embed_dim=d_model)
        self.patch_len = patch_len
        self.stride = stride
        self.num_channels = num_channels
        self.num_patches = (seq_len - patch_len) // stride + 1

        self.patch_proj = nn.Linear(patch_len, d_model)
        self.channel_embed = nn.Embedding(num_channels, d_model)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, d_model)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.norm = nn.LayerNorm(d_model)

    def _create_patches(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
        """[B, T, C] -> [B*C, num_patches, patch_len], B, C, num_patches"""
        B, T, C = x.shape
        patches = x.unfold(1, self.patch_len, self.stride)  # [B, num_patches, C, patch_len]
        num_patches = patches.shape[1]
        patches = patches.permute(0, 2, 1, 3)  # [B, C, num_patches, patch_len]
        patches = patches.reshape(B * C, num_patches, self.patch_len)
        return patches, B, C, num_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches, B, C, num_patches = self._create_patches(x)
        h = self.patch_proj(patches)  # [B*C, num_patches, d_model]

        # Add positional embedding
        pos = self.pos_embed[:, :num_patches, :]
        h = h + pos

        # Add channel embedding so transformer can distinguish gaze_x/y, pupil_L/R
        # channel_ids: [B*C] where each group of C has ids [0, 1, ..., C-1]
        channel_ids = torch.arange(C, device=x.device).repeat(B)  # [B*C]
        h = h + self.channel_embed(channel_ids).unsqueeze(1)  # broadcast [B*C, 1, d_model]

        h = self.transformer(h)
        h = self.norm(h)
        h = h.view(B, C, num_patches, -1)
        h = h.mean(dim=1)  # [B, num_patches, d_model]
        return h


class PatchTSTPreTrainer:
    """Self-supervised masked patch prediction for PatchTST pre-training."""

    def __init__(
        self,
        encoder: PatchTSTEncoder,
        mask_ratio: float = 0.4,
        lr: float = 1e-3,
        device: str = "cuda",
    ) -> None:
        self.encoder = encoder.to(device)
        self.mask_ratio = mask_ratio
        self.device = device
        # Learnable mask token
        self.mask_token = torch.randn(1, 1, encoder.embed_dim, device=device) * 0.02
        self.mask_token.requires_grad_(True)
        # Prediction head
        self.pred_head = nn.Linear(encoder.embed_dim, encoder.patch_len).to(device)
        params = list(encoder.parameters()) + list(self.pred_head.parameters()) + [self.mask_token]
        self.optimizer = torch.optim.AdamW(params, lr=lr)

    def train_step(self, x: torch.Tensor) -> float:
        """One pre-training step. x: [B, T, C]."""
        self.encoder.train()
        x = x.to(self.device)

        patches, B, C, num_patches = self.encoder._create_patches(x)
        num_mask = int(num_patches * self.mask_ratio)
        mask_indices = torch.rand(B * C, num_patches, device=self.device).argsort(dim=1)[:, :num_mask]

        targets = torch.gather(
            patches, 1,
            mask_indices.unsqueeze(-1).expand(-1, -1, self.encoder.patch_len)
        )

        # Replace masked patches with mask token
        h = self.encoder.patch_proj(patches)
        mask_expand = mask_indices.unsqueeze(-1).expand(-1, -1, self.encoder.embed_dim)
        mask_token = self.mask_token.expand(B * C, num_mask, -1)
        h.scatter_(1, mask_expand, mask_token)

        pos = self.encoder.pos_embed[:, :num_patches, :]
        h = h + pos
        h = self.encoder.transformer(h)
        h = self.encoder.norm(h)

        masked_h = torch.gather(
            h, 1,
            mask_indices.unsqueeze(-1).expand(-1, -1, self.encoder.embed_dim)
        )
        pred = self.pred_head(masked_h)

        loss = nn.functional.mse_loss(pred, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return loss.item()

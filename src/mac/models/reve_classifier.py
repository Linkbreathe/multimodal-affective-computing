"""ReveClassifier — port of the official REVE downstream classifier head.

Source: reve_eeg/src/models/classifier.py (brain-bzh/reve official repo).

Adapts the official `ReveClassifier` to use our `ReveEncoder` (which already
loads the pretrained REVE backbone via safetensors and manages position
embeddings internally).

Forward: [B, C, T] EEG -> per-patch features -> attention pooling with a
learnable cls_query_token -> RMSNorm -> Dropout -> Linear -> [B, n_classes].

This matches the official `pooling="last"` mode used for most downstream tasks.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from mac.encoders.reve import ReveEncoder


class RMSNorm(nn.Module):
    """RMSNorm matching the official REVE implementation."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight


class ReveClassifier(nn.Module):
    """Two-stage trainable classifier wrapping a REVE encoder.

    Args:
        encoder: ReveEncoder instance (with pretrained weights loaded).
        n_classes: Number of output classes.
        dropout: Dropout rate before the linear head.
        pooling: One of {"last", "last_avg"}.
            - "last": attention pooling with a learnable cls query token (default).
            - "last_avg": mean pooling over patch features.
    """

    def __init__(
        self,
        encoder: ReveEncoder,
        n_classes: int,
        dropout: float = 0.1,
        pooling: str = "last",
    ) -> None:
        super().__init__()
        if pooling not in ("last", "last_avg"):
            raise ValueError(f"Unsupported pooling '{pooling}', expected 'last' or 'last_avg'")

        self.encoder = encoder
        self.embed_dim = encoder.embed_dim
        self.pooling = pooling

        # Learnable query token for attention pooling (mirrors official ReveClassifier)
        self.cls_query_token = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)

        self.linear_head = nn.Sequential(
            RMSNorm(self.embed_dim),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] EEG input (60 channels, 2000 samples at 200Hz).
        Returns:
            [B, n_classes] logits.
        """
        # Per-patch features: [B, C*num_patches, embed_dim]
        feats = self.encoder.forward_features(x)

        if self.pooling == "last_avg":
            pooled = feats.mean(dim=1)  # [B, embed_dim]
        else:  # "last": attention pooling with cls query token
            b = feats.shape[0]
            query = self.cls_query_token.expand(b, -1, -1)  # [B, 1, D]
            scores = torch.matmul(query, feats.transpose(-1, -2)) / (self.embed_dim ** 0.5)
            attn = torch.softmax(scores, dim=-1)              # [B, 1, N]
            pooled = torch.matmul(attn, feats).squeeze(1)     # [B, D]

        return self.linear_head(pooled)


def freeze_for_linear_probe(model: ReveClassifier) -> None:
    """Stage-1 freeze: only linear_head + cls_query_token are trainable."""
    for param in model.parameters():
        param.requires_grad = False
    for param in model.linear_head.parameters():
        param.requires_grad = True
    model.cls_query_token.requires_grad = True


def unfreeze_all(model: ReveClassifier) -> None:
    """Unfreeze every parameter (used before injecting LoRA in stage 2)."""
    for param in model.parameters():
        param.requires_grad = True

"""Classifier-Guided Gradient Modulation (CGGM).

Attaches lightweight per-modality classifiers to projected embeddings and
modulates gradients so that weaker modalities receive amplified gradients
while the dominant modality is dampened.  This prevents a single strong
modality (e.g. video) from suppressing learning for the others.

Reference: NeurIPS 2024 — Classifier-Guided Gradient Modulation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityClassifier(nn.Module):
    """Lightweight 2-layer MLP classifier for a single modality."""

    def __init__(self, d_input: int, num_classes: int = 9, d_hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If 3D (B, T, D), mean-pool to (B, D) first
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.net(x)


class CGGMModule(nn.Module):
    """Container for per-modality classifiers and gradient scaling logic.

    Usage during training:
        1. Call ``compute_scales()`` with projected embeddings and labels.
           This runs each per-modality classifier, computes their CE losses,
           and returns the gradient scale per modality.
        2. Call ``register_hooks()`` on the projected embedding tensors.
           This attaches backward hooks that multiply gradients by the scales.
        3. The per-modality classifier loss is returned so the trainer can
           add it (weighted by ``lambda_mod``) to the optimiser step.

    The classifiers receive **detached** copies of the projected embeddings
    so their gradients do not flow back through the projectors.
    """

    def __init__(
        self,
        modality_ids: list[str],
        d_common: int = 256,
        num_classes: int = 9,
        d_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.modality_ids = modality_ids
        self.classifiers = nn.ModuleDict(
            {mod: ModalityClassifier(d_common, num_classes, d_hidden) for mod in modality_ids}
        )

    def compute_scales(
        self,
        projected_embeddings: list[torch.Tensor],
        modality_ids: list[str],
        emotion_labels: torch.Tensor,
    ) -> tuple[dict[str, float], torch.Tensor]:
        """Compute gradient scales and aggregate per-modality classifier loss.

        Args:
            projected_embeddings: list of (B, D) or (B, T, D) projected tensors.
            modality_ids: corresponding modality names.
            emotion_labels: (B,) integer class labels.

        Returns:
            scales: dict mapping modality_id -> gradient scale factor.
            mod_loss: scalar tensor, sum of per-modality CE losses (for
                      training the classifiers — gradients detached from
                      the projector).
        """
        M = len(modality_ids)
        mod_losses: dict[str, torch.Tensor] = {}

        for emb, mod_id in zip(projected_embeddings, modality_ids):
            logits = self.classifiers[mod_id](emb.detach())
            mod_losses[mod_id] = F.cross_entropy(logits, emotion_labels)

        total = sum(mod_losses.values())
        # Prevent division by zero when all losses are 0 (e.g. first batch)
        total_safe = total.clamp(min=1e-8)

        scales: dict[str, float] = {}
        for mod_id in modality_ids:
            # alpha_m = L_m / sum(L)  — higher alpha = weaker modality
            # scale_m = M * alpha_m   — amplifies weaker, dampens stronger
            alpha = mod_losses[mod_id] / total_safe
            scales[mod_id] = (M * alpha).item()

        return scales, total

    @staticmethod
    def register_hooks(
        projected_embeddings: list[torch.Tensor],
        modality_ids: list[str],
        scales: dict[str, float],
    ) -> list[torch.utils.hooks.RemovableHook]:
        """Attach gradient scaling hooks to projected embeddings.

        Must be called BEFORE the forward pass through the fusion module
        so the hooks fire during backward.

        Returns the hook handles so the caller can remove them after the step.
        """
        handles = []
        for emb, mod_id in zip(projected_embeddings, modality_ids):
            if emb.requires_grad:
                s = scales[mod_id]
                handle = emb.register_hook(lambda grad, _s=s: grad * _s)
                handles.append(handle)
        return handles

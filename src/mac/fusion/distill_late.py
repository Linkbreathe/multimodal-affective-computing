"""Late fusion with soft-target cross-modal distillation.

Uses the video modality's projection as a TEACHER to train physio modality
projections (PPG, eye tracking) as STUDENTS via soft-target KL distillation.

The distillation enriches physio projections so they carry more
emotion-discriminative information, without unfreezing encoders.

Architecture:
- Enriched projections (2-layer MLP + LayerNorm) for student modalities
- Simple linear projection for the teacher (video)
- Per-modality auxiliary classifiers for distillation
- Late fusion with learnable softmax weights (same as LateFusion)

Distillation math:
    p_teacher = softmax(classifier_video(proj_video.detach()) / tau)
    p_student = log_softmax(classifier_m(proj_m) / tau)
    L_distill = tau^2 * KL(p_teacher || p_student)

The distillation loss is stored as self.last_distill_loss for the trainer.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mac.fusion.base import BaseFusionModule


class EnrichedModalityProjector(nn.Module):
    """Projector with enriched (2-layer MLP) projections for student modalities.

    Student modalities (e.g., PPG, eye_tracking) get a deeper projection to
    give them more capacity to learn from the distillation signal. The teacher
    modality (video) keeps a simple linear projection.
    """

    def __init__(
        self,
        embed_dims: dict[str, int],
        d_common: int,
        enriched_modalities: list[str],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.projectors = nn.ModuleDict()
        for mod, dim in embed_dims.items():
            if mod in enriched_modalities:
                layers = [
                    nn.Linear(dim, d_common),
                    nn.LayerNorm(d_common),
                    nn.ReLU(),
                ]
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                layers.extend([
                    nn.Linear(d_common, d_common),
                    nn.LayerNorm(d_common),
                    nn.ReLU(),
                ])
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                self.projectors[mod] = nn.Sequential(*layers)
            else:
                self.projectors[mod] = nn.Linear(dim, d_common)

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {mod: self.projectors[mod](x) for mod, x in inputs.items()}


class DistillLateFusion(BaseFusionModule):
    """Late fusion + soft-target distillation from video to physio modalities.

    After forward(), the following attributes are available:
        last_distill_loss:   Tensor or None — distillation loss (only during training)
        last_cosine_sims:    dict[str, float] — cosine similarity between each
                             student's projection and the teacher's projection
    """

    supports_sequence_input = False

    def __init__(
        self,
        d_common: int = 256,
        num_modalities: int = 3,
        num_classes: int = 9,
        tau: float = 3.0,
        enriched_modalities: list[str] | None = None,
        teacher_modality: str = "video",
        dropout: float = 0.1,
        d_out: int | None = None,
    ) -> None:
        super().__init__(d_common=d_common, d_out=d_out or d_common)
        self.num_classes = num_classes
        self.tau = tau
        self.teacher_modality = teacher_modality
        self.enriched_modalities = enriched_modalities or []

        # Late fusion branches (same as LateFusion weighted)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_common, d_common),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_common, self.d_out),
            )
            for _ in range(num_modalities)
        ])
        self.weights = nn.Parameter(torch.ones(num_modalities))

        # Per-modality auxiliary classifiers for distillation
        self.classifiers = nn.ModuleDict()

        # Inspection buffers
        self.last_distill_loss: torch.Tensor | None = None
        self.last_cosine_sims: dict[str, float] = {}

    def register_modality_classifiers(self, modality_ids: list[str]) -> None:
        """Create auxiliary classifiers after modality_ids are known.

        Called once during model construction in build_fusion_model.
        """
        for mod_id in modality_ids:
            self.classifiers[mod_id] = nn.Linear(self.d_common, self.num_classes)

    def forward(
        self,
        embeddings: list[torch.Tensor],
        modality_ids: list[str],
        masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        # --- Late fusion (weighted average of per-modality branches) ---
        branch_outs = [branch(emb) for branch, emb in zip(self.branches, embeddings)]
        stacked = torch.stack(branch_outs, dim=0)
        w = torch.softmax(self.weights, dim=0).view(-1, 1, 1)
        fused = (stacked * w).sum(dim=0)

        # --- Distillation loss (training only) ---
        self.last_distill_loss = None
        self.last_cosine_sims = {}

        if self.training and self.teacher_modality in modality_ids and self.classifiers:
            teacher_idx = modality_ids.index(self.teacher_modality)
            teacher_proj = embeddings[teacher_idx]

            # Teacher logits (stop gradient — video is the teacher)
            teacher_logits = self.classifiers[self.teacher_modality](teacher_proj.detach())
            p_teacher = F.softmax(teacher_logits / self.tau, dim=-1)

            distill_loss = torch.tensor(0.0, device=fused.device)
            n_students = 0

            for i, mod_id in enumerate(modality_ids):
                if mod_id == self.teacher_modality:
                    continue
                if mod_id not in self.enriched_modalities:
                    continue

                student_logits = self.classifiers[mod_id](embeddings[i])
                p_student = F.log_softmax(student_logits / self.tau, dim=-1)

                # KL(p_teacher || p_student), scaled by tau^2
                kl = F.kl_div(p_student, p_teacher, reduction="batchmean")
                distill_loss = distill_loss + self.tau ** 2 * kl
                n_students += 1

            if n_students > 0:
                self.last_distill_loss = distill_loss / n_students

        # --- Cosine similarity logging (always, for inspection) ---
        if self.teacher_modality in modality_ids:
            teacher_idx = modality_ids.index(self.teacher_modality)
            teacher_proj = embeddings[teacher_idx].detach()
            teacher_norm = F.normalize(teacher_proj, dim=-1)

            for i, mod_id in enumerate(modality_ids):
                if mod_id == self.teacher_modality:
                    continue
                student_norm = F.normalize(embeddings[i].detach(), dim=-1)
                cos_sim = (teacher_norm * student_norm).sum(dim=-1).mean().item()
                self.last_cosine_sims[mod_id] = cos_sim

        return fused

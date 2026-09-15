"""Trusted Multi-View Classification (TMC) via Evidential Deep Learning.

Each modality produces Dirichlet evidence independently. Evidence is converted
to Dirichlet parameters (alpha), from which per-modality belief masses and
uncertainty are derived.  Beliefs from all modalities are combined using
Dempster's rule of combination, yielding a fused belief vector.

Reference: Han et al., "Trusted Multi-View Classification" (ICLR 2021).

Math overview
-------------
For modality m with K classes:
    evidence_m = Softplus(MLP_m(x_m))          # >= 0
    alpha_m    = evidence_m + 1                 # Dirichlet concentration
    S_m        = sum_k(alpha_m_k)               # Dirichlet strength
    b_m_k      = evidence_m_k / S_m             # belief mass for class k
    u_m        = K / S_m                        # uncertainty (vacuity)

Dempster combination of two belief sets (b_1, u_1) and (b_2, u_2):
    C = sum_{i != j} b_1_i * b_2_j             # conflict
    combined_b_k = (b_1_k * b_2_k + b_1_k * u_2 + u_1 * b_2_k) / (1 - C)
    combined_u   = (u_1 * u_2) / (1 - C)

Applied iteratively across M modalities.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.fusion.base import BaseFusionModule


class EvidenceBranch(nn.Module):
    """Per-modality MLP that maps projected embedding to non-negative evidence."""

    def __init__(self, d_in: int, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_in, num_classes),
            nn.Softplus(),  # evidence must be >= 0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def dempster_combine(
    belief_a: torch.Tensor,
    uncert_a: torch.Tensor,
    belief_b: torch.Tensor,
    uncert_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine two belief-uncertainty pairs via Dempster's rule.

    Args:
        belief_a: (B, K) belief masses from source A.
        uncert_a: (B, 1) uncertainty from source A.
        belief_b: (B, K) belief masses from source B.
        uncert_b: (B, 1) uncertainty from source B.

    Returns:
        combined_belief: (B, K)
        combined_uncert: (B, 1)
    """
    # Compute conflict: C = sum_{i != j} b_a_i * b_b_j
    # Equivalent to: C = (sum b_a) * (sum b_b) - sum(b_a * b_b)
    bb = (belief_a * belief_b).sum(dim=-1, keepdim=True)
    conflict = (
        belief_a.sum(dim=-1, keepdim=True) * belief_b.sum(dim=-1, keepdim=True) - bb
    )
    conflict = conflict.clamp(max=1.0 - 1e-7)  # prevent division by zero

    denom = 1.0 - conflict

    combined_belief = (
        belief_a * belief_b + belief_a * uncert_b + uncert_a * belief_b
    ) / denom
    combined_uncert = (uncert_a * uncert_b) / denom

    return combined_belief, combined_uncert


class TMCFusion(BaseFusionModule):
    """Evidential fusion: per-modality evidence -> Dempster combination.

    Output is a (B, num_classes) combined belief vector. This is NOT a
    d_common-dimensional embedding --- the task head must be adapted.

    After forward(), the following attributes are available for inspection:
        last_evidences:     dict[str, Tensor]  per-modality evidence (B, K)
        last_alphas:        dict[str, Tensor]  per-modality alpha (B, K)
        last_uncertainties: dict[str, Tensor]  per-modality uncertainty (B, 1)
        last_beliefs:       dict[str, Tensor]  per-modality belief (B, K)
        last_combined_uncertainty: Tensor       fused uncertainty (B, 1)
    """

    supports_sequence_input = False

    def __init__(
        self,
        d_common: int = 256,
        num_classes: int = 9,
        num_modalities: int = 3,
        modality_ids: list[str] | None = None,
        dropout: float = 0.1,
    ) -> None:
        # d_out = num_classes because TMC outputs a belief vector
        super().__init__(d_common=d_common, d_out=num_classes)
        self.num_classes = num_classes
        self.modality_ids = list(modality_ids) if modality_ids is not None else None
        self.num_modalities = len(self.modality_ids) if self.modality_ids is not None else num_modalities

        # Per-modality evidence branches. Branch identity is resolved by modality name.
        self.evidence_branches = nn.ModuleList([
            EvidenceBranch(d_common, num_classes, dropout)
            for _ in range(self.num_modalities)
        ])

        # Inspection buffers (set during forward, read during eval)
        self.last_evidences: dict[str, torch.Tensor] = {}
        self.last_alphas: dict[str, torch.Tensor] = {}
        self.last_uncertainties: dict[str, torch.Tensor] = {}
        self.last_beliefs: dict[str, torch.Tensor] = {}
        self.last_combined_uncertainty: torch.Tensor | None = None

    def forward(
        self,
        embeddings: list[torch.Tensor],
        modality_ids: list[str],
        masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        K = self.num_classes
        beliefs: list[torch.Tensor] = []
        uncerts: list[torch.Tensor] = []

        # Clear inspection buffers
        self.last_evidences = {}
        self.last_alphas = {}
        self.last_uncertainties = {}
        self.last_beliefs = {}

        branch_indices = self._resolve_branch_indices(modality_ids, len(embeddings))
        for emb, mod_id, branch_idx in zip(embeddings, modality_ids, branch_indices):
            evidence = self.evidence_branches[branch_idx](emb)  # (B, K)
            alpha = evidence + 1.0                               # (B, K)
            S = alpha.sum(dim=-1, keepdim=True)                  # (B, 1)
            belief = evidence / S                                # (B, K)
            uncertainty = K / S                                  # (B, 1)

            beliefs.append(belief)
            uncerts.append(uncertainty)

            # Store for inspection / loss computation
            self.last_evidences[mod_id] = evidence
            self.last_alphas[mod_id] = alpha
            self.last_uncertainties[mod_id] = uncertainty
            self.last_beliefs[mod_id] = belief

        # Dempster combination across modalities
        combined_b = beliefs[0]
        combined_u = uncerts[0]
        for j in range(1, len(beliefs)):
            combined_b, combined_u = dempster_combine(
                combined_b, combined_u, beliefs[j], uncerts[j]
            )

        self.last_combined_uncertainty = combined_u

        return combined_b  # (B, K) belief vector

    def _resolve_branch_indices(
        self,
        modality_ids: list[str],
        n_embeddings: int,
    ) -> list[int]:
        if len(modality_ids) != n_embeddings:
            raise ValueError("modality_ids and embeddings must have the same length")
        if not modality_ids:
            raise ValueError("TMCFusion requires at least one modality")
        if self.modality_ids is None:
            if len(modality_ids) != len(self.evidence_branches):
                raise ValueError(
                    "TMCFusion without constructor modality_ids must see all "
                    "modalities on the first forward pass"
                )
            self.modality_ids = list(modality_ids)

        index_by_modality = {mod: i for i, mod in enumerate(self.modality_ids)}
        missing = [mod for mod in modality_ids if mod not in index_by_modality]
        if missing:
            raise ValueError(f"Unknown modalities for TMCFusion: {missing}")
        return [index_by_modality[mod] for mod in modality_ids]

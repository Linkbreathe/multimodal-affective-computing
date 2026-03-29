"""Multi-task loss functions: weighted CE, KL divergence, CCC loss."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCELoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, targets)


class SoftLabelKLLoss(nn.Module):
    def forward(self, logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        return F.kl_div(log_probs, soft_targets, reduction="batchmean")


class CCCLoss(nn.Module):
    """1 - CCC as a loss (minimized when CCC = 1)."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mean_pred = pred.mean(dim=0)
        mean_target = target.mean(dim=0)
        var_pred = pred.var(dim=0, correction=0)
        var_target = target.var(dim=0, correction=0)
        covar = ((pred - mean_pred) * (target - mean_target)).mean(dim=0)
        denom = var_pred + var_target + (mean_pred - mean_target) ** 2
        ccc = 2 * covar / (denom + 1e-8)
        return 1.0 - ccc.mean()


class DirichletKLLoss(nn.Module):
    """KL divergence between predicted Dirichlet and a target Dirichlet.

    For TMC: the target Dirichlet concentrates mass on the true class.
    alpha_target_k = 1 for all k, except alpha_target_y = 1 (i.e. we
    remove the evidence from non-target classes to get a uniform-ish target).

    Specifically, we compute:
        alpha_tilde = y_onehot + (1 - y_onehot) * alpha_pred
    so the predicted alpha for the correct class is kept, and wrong-class
    alphas are shrunk toward 1 (uniform).

    The KL is: KL(Dir(alpha_tilde) || Dir(1, ..., 1))
    """

    def forward(
        self, alpha: torch.Tensor, targets: torch.Tensor, num_classes: int = 9,
    ) -> torch.Tensor:
        """Compute Dirichlet KL for one modality's alpha.

        Args:
            alpha: (B, K) Dirichlet parameters (evidence + 1).
            targets: (B,) integer class labels.
            num_classes: K.
        """
        y_onehot = F.one_hot(targets, num_classes).float().to(alpha.device)
        # Remove misleading evidence: keep correct-class alpha, shrink others to 1
        alpha_tilde = y_onehot + (1.0 - y_onehot) * alpha

        S_tilde = alpha_tilde.sum(dim=-1, keepdim=True)
        ones = torch.ones_like(alpha_tilde)
        S_ones = ones.sum(dim=-1, keepdim=True)  # = K

        # KL(Dir(alpha_tilde) || Dir(1,...,1))
        kl = (
            torch.lgamma(S_tilde) - torch.lgamma(S_ones)
            - (torch.lgamma(alpha_tilde) - torch.lgamma(ones)).sum(dim=-1, keepdim=True)
            + ((alpha_tilde - ones) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde))).sum(
                dim=-1, keepdim=True
            )
        )
        return kl.mean()


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        ce_weight: torch.Tensor | None = None,
        lambda_ce: float = 1.0,
        lambda_kl: float = 1.0,
        lambda_vad: float = 1.0,
    ) -> None:
        super().__init__()
        self.ce_loss = WeightedCELoss(weight=ce_weight)
        self.kl_loss = SoftLabelKLLoss()
        self.ccc_loss = CCCLoss()
        self.lambda_ce = lambda_ce
        self.lambda_kl = lambda_kl
        self.lambda_vad = lambda_vad

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        device = outputs["emotion_logits"].device
        total = torch.tensor(0.0, device=device)
        breakdown: dict[str, float] = {}

        if self.lambda_ce > 0:
            ce = self.ce_loss(outputs["emotion_logits"], targets["emotion_label"])
            total = total + self.lambda_ce * ce
            breakdown["ce"] = ce.item()
        else:
            breakdown["ce"] = 0.0

        if self.lambda_kl > 0:
            kl = self.kl_loss(outputs["soft_logits"], targets["soft_label"])
            total = total + self.lambda_kl * kl
            breakdown["kl"] = kl.item()
        else:
            breakdown["kl"] = 0.0

        if self.lambda_vad > 0:
            vad = self.ccc_loss(outputs["vad_pred"], targets["vad"])
            total = total + self.lambda_vad * vad
            breakdown["vad"] = vad.item()
        else:
            breakdown["vad"] = 0.0

        return total, breakdown

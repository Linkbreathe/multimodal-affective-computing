import pytest
import torch
from mac.tasks.losses import WeightedCELoss, SoftLabelKLLoss, CCCLoss, MultiTaskLoss


def test_weighted_ce_loss():
    weights = torch.tensor([1.0, 2.0, 1.5])
    loss_fn = WeightedCELoss(weight=weights)
    logits = torch.randn(4, 3)
    targets = torch.tensor([0, 1, 2, 0])
    loss = loss_fn(logits, targets)
    assert loss.shape == ()
    assert loss.item() > 0


def test_kl_loss():
    loss_fn = SoftLabelKLLoss()
    logits = torch.randn(4, 9)
    soft_targets = torch.softmax(torch.randn(4, 9), dim=-1)
    loss = loss_fn(logits, soft_targets)
    assert loss.shape == ()
    assert loss.item() >= 0


def test_ccc_loss():
    loss_fn = CCCLoss()
    pred = torch.randn(4, 3)
    target = torch.randn(4, 3)
    loss = loss_fn(pred, target)
    assert loss.shape == ()


def test_ccc_loss_perfect():
    loss_fn = CCCLoss()
    # Need batch_size > 1 for meaningful variance
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    loss = loss_fn(x, x)
    assert loss.item() < 0.01  # 1 - CCC ≈ 0 when pred == target


def test_multitask_loss():
    weights = torch.ones(9)
    mt_loss = MultiTaskLoss(
        ce_weight=weights,
        lambda_ce=1.0,
        lambda_kl=1.0,
        lambda_vad=1.0,
    )
    outputs = {
        "emotion_logits": torch.randn(4, 9),
        "soft_logits": torch.randn(4, 9),
        "vad_pred": torch.randn(4, 3),
    }
    targets = {
        "emotion_label": torch.tensor([0, 1, 2, 3]),
        "soft_label": torch.softmax(torch.randn(4, 9), dim=-1),
        "vad": torch.randn(4, 3),
    }
    loss, breakdown = mt_loss(outputs, targets)
    assert loss.shape == ()
    assert "ce" in breakdown
    assert "kl" in breakdown
    assert "vad" in breakdown


from mac.tasks.heads import MultiTaskHead

def test_multitask_head_shapes():
    head = MultiTaskHead(d_fused=256, num_emotions=9, num_vad=3)
    x = torch.randn(4, 256)
    outputs = head(x)
    assert outputs["emotion_logits"].shape == (4, 9)
    assert outputs["soft_logits"].shape == (4, 9)
    assert outputs["vad_pred"].shape == (4, 3)


def test_multitask_loss_zero_lambda_nan_safe():
    """0-weight auxiliary losses must not produce NaN in total loss."""
    mt_loss = MultiTaskLoss(
        ce_weight=torch.ones(9),
        lambda_ce=1.0,
        lambda_kl=0.0,
        lambda_vad=0.0,
    )
    outputs = {
        "emotion_logits": torch.randn(4, 9),
        "soft_logits": torch.randn(4, 9),
        "vad_pred": torch.randn(4, 3),
    }
    # Soft labels with exact zeros → kl_div would produce NaN
    soft_label = torch.zeros(4, 9)
    soft_label[:, 0] = 1.0  # one-hot, rest are 0
    targets = {
        "emotion_label": torch.tensor([0, 1, 2, 3]),
        "soft_label": soft_label,
        "vad": torch.zeros(4, 3),  # all-zero VAD
    }
    loss, breakdown = mt_loss(outputs, targets)

    assert loss.shape == (), f"Expected scalar, got {loss.shape}"
    assert torch.isfinite(loss), f"Loss is {loss.item()}, expected finite"
    assert loss.item() > 0, "CE loss should be positive"
    assert breakdown["kl"] == 0.0, "Skipped KL should be 0.0"
    assert breakdown["vad"] == 0.0, "Skipped VAD should be 0.0"

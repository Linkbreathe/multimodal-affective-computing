from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from mac.experiments.dynamic_texture_five import (
    MODALITIES,
    audit_video_gate,
    make_five_branch_model,
    parameter_signature,
    state_dict_sha256,
)


torch = pytest.importorskip("torch")


def _inputs(batch: int = 3) -> dict[str, np.ndarray]:
    counts = {"eeg": 4, "ecg": 3, "eye": 2, "head": 2, "video": 13}
    rng = np.random.default_rng(7)
    return {
        modality: rng.normal(size=(batch, counts[modality], 8)).astype(np.float32)
        for modality in MODALITIES
    }


def test_five_branch_pair_has_identical_capacity_and_initialization() -> None:
    counts = {"eeg": 4, "ecg": 3, "eye": 2, "head": 2, "video": 13}
    torch.manual_seed(41)
    full = make_five_branch_model(counts, dropout=0.0)
    state = deepcopy(full.state_dict())
    no_video = make_five_branch_model(counts, dropout=0.0)
    no_video.load_state_dict(state)

    assert parameter_signature(full) == parameter_signature(no_video)
    assert sum(parameter.numel() for parameter in full.parameters()) == sum(
        parameter.numel() for parameter in no_video.parameters()
    )
    assert full.fusion_input_dim == no_video.fusion_input_dim
    assert state_dict_sha256(full.state_dict()) == state_dict_sha256(no_video.state_dict())


def test_post_encoder_gate_is_exact_and_parameter_independent() -> None:
    counts = {"eeg": 4, "ecg": 3, "eye": 2, "head": 2, "video": 13}
    model = make_five_branch_model(counts, dropout=0.0)
    inputs = _inputs()
    context = np.zeros((3, 2), dtype=np.float32)

    record = audit_video_gate(model, inputs, context, device=torch.device("cpu"))

    assert record["embedding_max_abs"] == 0.0
    assert record["embedding_with_nonzero_bias_max_abs"] == 0.0
    assert record["prediction_parameter_invariant_exact"] is True
    assert record["video_encoder_gradient_max_abs"] == 0.0


def test_full_video_branch_receives_gradient_and_rgb_is_rejected() -> None:
    counts = {"eeg": 4, "ecg": 3, "eye": 2, "head": 2, "video": 13}
    model = make_five_branch_model(counts, dropout=0.0)
    numpy_inputs = _inputs()
    inputs = {
        name: torch.as_tensor(value, dtype=torch.float32) for name, value in numpy_inputs.items()
    }
    context = torch.zeros((3, 2), dtype=torch.float32)
    gate = torch.ones(3, dtype=torch.float32)
    model(inputs, context, gate).sum().backward()
    gradients = [
        parameter.grad
        for parameter in model.video_encoder.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert max(float(value.abs().max()) for value in gradients) > 0.0

    invalid = dict(inputs)
    invalid["video"] = torch.zeros((3, 3, 16, 112, 112), dtype=torch.float32)
    with pytest.raises(ValueError, match="RGB tensors are rejected"):
        model(invalid, context, gate)

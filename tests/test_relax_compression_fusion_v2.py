from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_relax_compression_fusion_v2 as runner
from scripts.run_relax_foundation_probe import Fold


def _toy() -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    Fold,
]:
    rng = np.random.default_rng(1807)
    participants = tuple(f"P{index:03d}" for index in range(1, 10))
    conditions = tuple(f"C{index}" for index in range(1, 10))
    dims = {"eeg": 16, "ecg": 14, "eye": 10, "head": 8, "video": 12}
    blocks = {name: rng.normal(size=(81, 3, width)) for name, width in dims.items()}
    masks = {name: np.ones((81, 3), dtype=bool) for name in runner.FULL_MODALITIES}
    for name in runner.FULL_MODALITIES:
        masks[name][14] = False
    availability = {name: masks[name].any(axis=1) for name in runner.FULL_MODALITIES}
    presence = masks["eeg"].any(axis=1)
    quality = masks["eeg"].sum(axis=1) / 3.0
    records = []
    for participant_index, participant in enumerate(participants):
        for condition_index, condition in enumerate(conditions):
            row = participant_index * 9 + condition_index
            records.append(
                {
                    "participant_id": participant,
                    "condition": condition,
                    "presentation_position": condition_index + 1,
                    "relaxation": float(np.clip(0.2 + 0.05 * condition_index + 0.02 * blocks["eeg"][row, :, 0].mean(), 0, 1)),
                    "discomfort": float(np.clip(0.7 - 0.05 * condition_index + 0.02 * blocks["video"][row, :, 0].mean(), 0, 1)),
                }
            )
    fold = Fold(1, participants[:7], participants[7], participants[8])
    return blocks, masks, availability, presence, quality, pd.DataFrame(records), fold


def test_exact_full_method_panel_and_ablation_scope():
    assert runner.METHODS == (
        "joint_block_balanced_pca12",
        "modality_rank_alloc_pca12",
        "targetwise_modality_pls1",
        "linear_gcca_shared_private",
        "modality_expert_simplex5",
    )
    assert runner.variant_modalities("full") == runner.FULL_MODALITIES
    assert runner.variant_modalities("no_eeg") == ("ecg", "eye", "head", "video")


def test_expert_simplex_enforces_every_active_modality_floor():
    rng = np.random.default_rng(71)
    predictions = rng.normal(size=(63, 5))
    weights = runner._solve_expert_weights(predictions, predictions[:, 0], 0.02)
    assert np.isclose(weights.sum(), 1.0, atol=1e-10)
    assert np.all(weights >= 0.02 - 1e-10)


def test_missing_experts_are_renormalized_and_all_missing_is_zero():
    corrections = np.asarray([[1.0, 2.0], [1.0, 99.0], [99.0, 99.0]])
    available = np.asarray([[True, True], [True, False], [False, False]])
    observed = runner._weighted_experts(corrections, available, np.asarray([0.25, 0.75]))
    np.testing.assert_allclose(observed, [1.75, 1.0, 0.0])


def test_modalitywise_fold_uses_all_five_and_positive_shared_gamma():
    blocks, masks, availability, presence, quality, frame, fold = _toy()
    result = runner._fit_standard_fold(
        "modality_rank_alloc_pca12",
        blocks,
        masks,
        availability,
        presence,
        quality,
        frame,
        fold,
        {"ridge_alphas": [10.0, 100.0], "positive_gammas": [0.25, 0.5], "residual_cap": 0.2},
    )
    assert result["modalities"] == runner.FULL_MODALITIES
    assert sum(result["modality_components"].values()) == 12
    assert result["gamma"] > 0
    evidence = runner._eligibility("modality_rank_alloc_pca12", "full", result, presence)
    assert evidence["both_targets_learned"]
    assert evidence["condition_fallback_used"] is False
    assert evidence["eligible"]


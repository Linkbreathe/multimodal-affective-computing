from __future__ import annotations

# ruff: noqa: E402

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_relax_condition_anchor_probe import (
    CANDIDATES,
    Candidate,
    apply_condition_centering,
    condition_feature_means,
    cross_fitted_condition_anchors,
    fit_fold,
    pooled_features,
)
from scripts.evaluate_relax_condition_anchor import _exact_sign_flip, _holm, _paired_statistics
from scripts.run_relax_foundation_probe import Fold


def _toy_frame() -> tuple[pd.DataFrame, np.ndarray, Fold]:
    rng = np.random.default_rng(23)
    rows = []
    for participant_index in range(9):
        participant = f"P{participant_index}"
        for condition_index in range(9):
            rows.append(
                {
                    "participant_id": participant,
                    "condition": f"C{condition_index + 1}",
                    "presentation_position": condition_index + 1,
                    "relaxation": 0.25 + 0.04 * condition_index + 0.01 * participant_index,
                    "discomfort": 0.01 * condition_index + 0.005 * participant_index,
                }
            )
    frame = pd.DataFrame(rows)
    features = rng.normal(size=(len(frame), 12))
    fold = Fold(
        fold_index=1,
        train_participants=tuple(f"P{index}" for index in range(7)),
        validation_participant="P7",
        test_participant="P8",
    )
    return frame, features, fold


def _toy_candidate(*, gammas=(0.0, 0.5, 1.0), learned_targets=("relaxation", "discomfort")) -> Candidate:
    return Candidate(
        name="toy",
        modalities=("eye",),
        learned_targets=learned_targets,
        representation="raw",
        quality_features="presence",
        pca_fit_components=4,
        pca_prefixes=(2, 4),
        alphas=(0.1, 10.0),
        gammas=gammas,
        residual_cap=0.2,
        selection_tie_policy="validation_mae_then_gamma_components_stronger_alpha",
        family="test",
    )


def test_formal_candidate_family_is_focused_and_declared():
    assert set(CANDIDATES) == {
        "ecg_raw_dual",
        "ecg_centered_dual",
        "ecg_eye_raw_dual",
        "ecg_eye_centered_dual",
        "no_eeg_raw_dual",
        "no_eeg_centered_dual",
        "eye_discomfort_compact",
        "eye_discomfort_wide",
    }


def test_masked_pooling_keeps_zero_window_observation_finite_and_explicit():
    dataset = SimpleNamespace(
        modalities=("eye",),
        embeddings={
            "eye": torch.tensor(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[9.0, 9.0], [8.0, 8.0]],
                ]
            )
        },
        masks={"eye": torch.tensor([[True, False], [False, False]])},
    )

    features, metadata = pooled_features(dataset, "count_and_missing")

    np.testing.assert_array_equal(features[0], [1.0, 2.0, 0.5, 0.0])
    np.testing.assert_array_equal(features[1], [0.0, 0.0, 0.0, 1.0])
    assert np.isfinite(features).all()
    assert metadata["modalities"]["eye"]["zero_window_observations"] == 1


def test_cross_fitted_anchor_excludes_own_participant_target():
    rows = []
    for participant_index in range(7):
        rows.append(
            {
                "participant_id": f"P{participant_index}",
                "condition": "C1",
                "relaxation": float(participant_index),
            }
        )
    frame = pd.DataFrame(rows)

    anchors = cross_fitted_condition_anchors(frame, np.arange(7), "relaxation")

    assert anchors[0] == np.mean([1, 2, 3, 4, 5, 6])
    assert anchors[6] == np.mean([0, 1, 2, 3, 4, 5])


def test_condition_feature_centroids_fit_only_training_rows():
    features = np.array([[1.0], [3.0], [1000.0]])
    conditions = np.array(["C1", "C1", "C1"])
    means = condition_feature_means(features, conditions, np.array([0, 1]))

    centered = apply_condition_centering(features, conditions, means)

    np.testing.assert_array_equal(means["C1"], [2.0])
    np.testing.assert_array_equal(centered[:, 0], [-1.0, 1.0, 998.0])


def test_mutating_outer_test_targets_cannot_change_fit_selection_or_predictions():
    frame, features, fold = _toy_frame()
    first = fit_fold(_toy_candidate(), features, frame, fold, seed=20260705)
    changed = frame.copy()
    test_mask = changed["participant_id"] == fold.test_participant
    changed.loc[test_mask, "relaxation"] = 1.0 - changed.loc[test_mask, "relaxation"]
    changed.loc[test_mask, "discomfort"] = 1.0 - changed.loc[test_mask, "discomfort"]

    second = fit_fold(_toy_candidate(), features, changed, fold, seed=20260705)

    np.testing.assert_array_equal(first["scaler"].mean_, second["scaler"].mean_)
    np.testing.assert_array_equal(first["pca"].components_, second["pca"].components_)
    for target in ("relaxation", "discomfort"):
        assert first["targets"][target]["selection"] == second["targets"][target]["selection"]
        np.testing.assert_array_equal(
            first["targets"][target]["prediction"], second["targets"][target]["prediction"]
        )


def test_extreme_test_features_do_not_change_scaler_or_pca_fit():
    frame, features, fold = _toy_frame()
    first = fit_fold(_toy_candidate(), features, frame, fold, seed=20260705)
    changed = features.copy()
    changed[frame["participant_id"].to_numpy() == fold.test_participant] = 1e9

    second = fit_fold(_toy_candidate(), changed, frame, fold, seed=20260705)

    np.testing.assert_array_equal(first["scaler"].mean_, second["scaler"].mean_)
    np.testing.assert_array_equal(first["scaler"].scale_, second["scaler"].scale_)
    np.testing.assert_array_equal(first["pca"].components_, second["pca"].components_)


def test_gamma_zero_is_bitwise_equal_to_condition_anchor():
    frame, features, fold = _toy_frame()
    result = fit_fold(_toy_candidate(gammas=(0.0,)), features, frame, fold, seed=20260705)

    for target in ("relaxation", "discomfort"):
        assert result["targets"][target]["selection"]["gamma"] == 0.0
        np.testing.assert_array_equal(
            result["targets"][target]["prediction"], result["targets"][target]["test_anchor"]
        )


def test_target_specific_fallback_never_fits_or_changes_relaxation():
    frame, features, fold = _toy_frame()
    result = fit_fold(
        _toy_candidate(learned_targets=("discomfort",)),
        features,
        frame,
        fold,
        seed=20260705,
    )

    assert result["models"]["relaxation"] is None
    np.testing.assert_array_equal(
        result["targets"]["relaxation"]["prediction"],
        result["targets"]["relaxation"]["test_anchor"],
    )
    assert result["targets"]["relaxation"]["selection"]["fallback_reason"] == "target_specific_condition_only"


def test_repeated_seed_is_deterministic_and_predictions_are_bounded():
    frame, features, fold = _toy_frame()
    first = fit_fold(_toy_candidate(), features, frame, fold, seed=20260707)
    second = fit_fold(_toy_candidate(), features, frame, fold, seed=20260707)

    for target in ("relaxation", "discomfort"):
        np.testing.assert_array_equal(
            first["targets"][target]["prediction"], second["targets"][target]["prediction"]
        )
        assert np.all((first["targets"][target]["prediction"] >= 0.0))
        assert np.all((first["targets"][target]["prediction"] <= 1.0))


def test_exact_sign_flip_enumerates_every_assignment():
    pvalue, assignments = _exact_sign_flip(np.array([1.0, 1.0]))

    assert assignments == 4
    assert pvalue == 0.5


def test_holm_adjustment_is_monotone_in_sorted_pvalues():
    adjusted = _holm(pd.Series([0.01, 0.04, 0.03]))

    np.testing.assert_allclose(adjusted.to_numpy(), [0.03, 0.06, 0.06])


def test_paired_statistics_clusters_after_seed_averaging():
    rows = []
    for seed in (20260705, 20260706, 20260707):
        for participant_index in range(9):
            for outcome in ("relaxation", "discomfort", "macro"):
                rows.append(
                    {
                        "candidate": "toy",
                        "seed": seed,
                        "participant_id": f"P{participant_index}",
                        "outcome": outcome,
                        "model_mae": 0.1,
                        "baseline_mae": 0.2,
                        "delta": -0.01 * (participant_index + 1),
                    }
                )

    result = _paired_statistics(pd.DataFrame(rows))

    assert len(result) == 3
    assert set(result["participant_count"]) == {9}
    assert set(result["bootstrap_resamples"]) == {10_000}
    assert set(result["sign_flip_assignments"]) == {512}
    assert set(result["bootstrap_cluster"]) == {"participant_after_seed_average"}

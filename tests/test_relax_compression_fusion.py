from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from mac.fusion.frozen_compression import (
    FORMAL_METHODS,
    fit_modality_reducers,
    make_compressor,
)
from scripts import run_relax_compression_fusion as runner
from scripts.run_relax_foundation_probe import Fold, file_sha256
from scripts.evaluate_relax_compression_fusion import (
    FLOAT_ATOL,
    _interpret_head_ablation_delta,
)


MODALITIES = ("ecg", "eye", "head", "video")


def _toy() -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rng = np.random.default_rng(431)
    dimensions = {"ecg": 20, "eye": 8, "head": 5, "video": 14}
    blocks = {m: rng.normal(size=(81, d)) for m, d in dimensions.items()}
    counts = {m: np.full(81, 7, dtype=int) for m in MODALITIES}
    for modality in MODALITIES:
        counts[modality][14] = 0
        blocks[modality][14] = 0.0
    train = np.arange(63)
    validation = np.arange(63, 72)
    test = np.arange(72, 81)
    targets = rng.normal(size=(81, 2))
    return blocks, counts, train, validation, test, targets


def test_exact_four_shared_compressor_families_are_registered():
    assert FORMAL_METHODS == (
        "joint_block_balanced_pca8",
        "modality_pca2_additive",
        "supervised_pls2",
        "linear_shared_private",
    )


def test_preregistered_compressors_have_expected_dimensions_and_finite_outputs():
    blocks, counts, train, validation, test, targets = _toy()
    expected = {
        "joint_block_balanced_pca8": 9,
        "modality_pca2_additive": 9,
        "supervised_pls2": 3,
        "linear_shared_private": 9,
    }
    for method, dimension in expected.items():
        compressor = make_compressor(method, MODALITIES).fit(
            blocks, counts, train, targets=targets
        )
        transformed = compressor.transform(blocks, counts, np.concatenate([validation, test]))
        assert transformed.shape == (18, dimension)
        assert np.isfinite(transformed).all()
        assert compressor.provenance()["fit_indexes"] == train.tolist()


def test_outer_test_features_cannot_change_any_fitted_compressor_state():
    blocks, counts, train, _validation, test, targets = _toy()
    changed = {key: value.copy() for key, value in blocks.items()}
    for modality in MODALITIES:
        changed[modality][test] = 1e8
    for method in FORMAL_METHODS:
        first = make_compressor(method, MODALITIES).fit(blocks, counts, train, targets=targets)
        second = make_compressor(method, MODALITIES).fit(changed, counts, train, targets=targets)
        assert first.provenance() == second.provenance()
        np.testing.assert_array_equal(
            first.transform(blocks, counts, train), second.transform(changed, counts, train)
        )


def test_outer_test_targets_cannot_change_supervised_pls_projection():
    blocks, counts, train, validation, test, targets = _toy()
    changed = targets.copy()
    changed[test] = 1000.0
    first = make_compressor("supervised_pls2", MODALITIES).fit(
        blocks, counts, train, targets=targets
    )
    second = make_compressor("supervised_pls2", MODALITIES).fit(
        blocks, counts, train, targets=changed
    )
    assert first.provenance() == second.provenance()
    np.testing.assert_array_equal(
        first.transform(blocks, counts, validation),
        second.transform(blocks, counts, validation),
    )


def test_single_modality_expert_reducers_are_train_only_and_two_dimensional():
    blocks, counts, train, validation, _test, _targets = _toy()
    reducers = fit_modality_reducers(blocks, counts, train, MODALITIES)
    assert set(reducers) == set(MODALITIES)
    for modality, reducer in reducers.items():
        assert reducer.provenance()["fit_indexes"] == train.tolist()
        assert reducer.transform(blocks, counts, validation).shape == (9, 2)
        assert reducer.provenance()["modality"] == modality


def test_every_method_is_deterministic_under_repeated_fit():
    blocks, counts, train, validation, _test, targets = _toy()
    for method in FORMAL_METHODS:
        first = make_compressor(method, MODALITIES).fit(blocks, counts, train, targets=targets)
        second = make_compressor(method, MODALITIES).fit(blocks, counts, train, targets=targets)
        np.testing.assert_array_equal(
            first.transform(blocks, counts, validation),
            second.transform(blocks, counts, validation),
        )


def test_runner_registers_exact_preregistered_candidate_panel():
    assert runner.METHODS == (
        "joint_block_balanced_pca8",
        "modality_pca2_additive",
        "supervised_pls2",
        "linear_shared_private",
        "modality_expert_simplex",
        "modality_expert_simplex_foundation_only",
    )
    assert runner.EXPERT_METHODS == (
        "modality_expert_simplex",
        "modality_expert_simplex_foundation_only",
    )


def test_expert_weights_obey_simplex_and_foundation_floors():
    rng = np.random.default_rng(706)
    predictions = rng.normal(size=(63, len(runner.FORMAL_MODALITIES)))
    truths = predictions[:, 0]
    weights = runner._solve_expert_weights(
        predictions,
        truths,
        runner.FORMAL_MODALITIES,
        minimum_foundation_weight=0.05,
    )

    assert weights.shape == (len(runner.FORMAL_MODALITIES),)
    assert np.isfinite(weights).all()
    assert np.isclose(weights.sum(), 1.0, atol=1e-10)
    assert np.all(weights >= -1e-12)
    for index, modality in enumerate(runner.FORMAL_MODALITIES):
        if modality in runner.FOUNDATION_MODALITIES:
            assert weights[index] >= 0.05 - 1e-10


def test_missing_experts_are_renormalized_and_all_missing_rows_are_zero():
    corrections = np.asarray(
        [
            [10.0, 20.0, 30.0],
            [10.0, 999.0, 30.0],
            [999.0, 999.0, 30.0],
            [999.0, 999.0, 999.0],
        ]
    )
    available = np.asarray(
        [
            [True, True, True],
            [True, False, True],
            [False, False, True],
            [False, False, False],
        ]
    )
    weights = np.asarray([0.2, 0.3, 0.5])

    observed = runner._weighted_experts(corrections, available, weights)
    expected = np.asarray([23.0, 17.0 / 0.7, 30.0, 0.0])
    np.testing.assert_allclose(observed, expected, atol=1e-12, rtol=0.0)


def _synthetic_fold_inputs() -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    np.ndarray,
    pd.DataFrame,
    Fold,
]:
    rng = np.random.default_rng(1707)
    participants = tuple(f"P{index:03d}" for index in range(1, 10))
    conditions = tuple(f"C{index}" for index in range(1, 10))
    dimensions = {"ecg": 20, "eye": 8, "head": 5, "video": 14}
    blocks = {
        modality: rng.normal(size=(81, dimension))
        for modality, dimension in dimensions.items()
    }
    availability = {
        modality: np.ones(81, dtype=bool) for modality in runner.FORMAL_MODALITIES
    }
    records: list[dict[str, float | str]] = []
    for participant_index, participant in enumerate(participants):
        for condition_index, condition in enumerate(conditions):
            row_index = participant_index * len(conditions) + condition_index
            records.append(
                {
                    "participant_id": participant,
                    "condition": condition,
                    "presentation_position": float(condition_index + 1),
                    "relaxation": float(
                        np.clip(
                            0.25
                            + 0.035 * condition_index
                            + 0.025 * np.tanh(blocks["ecg"][row_index, 0]),
                            0.0,
                            1.0,
                        )
                    ),
                    "discomfort": float(
                        np.clip(
                            0.65
                            - 0.03 * condition_index
                            + 0.025 * np.tanh(blocks["video"][row_index, 0]),
                            0.0,
                            1.0,
                        )
                    ),
                }
            )
    fold = Fold(
        fold_index=1,
        train_participants=participants[:7],
        validation_participant=participants[7],
        test_participant=participants[8],
    )
    return blocks, availability, np.ones(81, dtype=bool), pd.DataFrame(records), fold


def test_standard_fold_uses_only_positive_gamma_and_no_condition_only_fallback():
    blocks, availability, shared_presence, frame, fold = _synthetic_fold_inputs()
    rules = {
        "ridge_alphas": [10.0, 100.0],
        "positive_gammas": [0.25, 0.5],
        "residual_cap": 0.2,
    }
    result = runner._fit_standard_fold(
        "modality_pca2_additive",
        blocks,
        availability,
        shared_presence,
        frame,
        fold,
        seed=20260705,
        rules=rules,
    )
    evidence = runner._eligibility(
        "modality_pca2_additive", result, shared_presence
    )

    assert result["gamma"] in rules["positive_gammas"]
    assert result["gamma"] > 0.0
    assert evidence["condition_fallback_used"] is False
    assert evidence["both_targets_learned"] is True
    assert evidence["nonzero_test_correction_for_both_targets"] is True
    assert evidence["eligible"] is True
    for target in runner.TARGETS:
        assert not np.array_equal(
            result["test_predictions"][target], result["test_anchor"][target]
        )


def test_standard_fold_rejects_a_zero_gamma_grid():
    blocks, availability, shared_presence, frame, fold = _synthetic_fold_inputs()
    with pytest.raises(ValueError, match="strictly positive"):
        runner._fit_standard_fold(
            "modality_pca2_additive",
            blocks,
            availability,
            shared_presence,
            frame,
            fold,
            seed=20260705,
            rules={
                "ridge_alphas": [10.0],
                "positive_gammas": [0.0],
                "residual_cap": 0.2,
            },
        )


def _valid_preregistration_payload(paths: dict[str, Path]) -> dict[str, object]:
    method_records = [
        {"name": name, "family": "test", "compression_dimension": 9}
        for name in runner.METHODS
        if name != "modality_expert_simplex_foundation_only"
    ]
    return {
        "schema_version": "relax_foundation_compression_preregistration_v1",
        "frozen_before_formal_candidate_outcomes": True,
        "protocol": {
            "participants": list(runner.EXPECTED_PARTICIPANTS),
            "seeds": list(runner.ALLOWED_SEEDS),
        },
        "common_model_rules": {
            "positive_gammas": [0.25, 0.5, 0.75, 1.0],
            "gamma_zero_allowed": False,
        },
        "input_contract": {
            "formal_modalities": list(runner.FORMAL_MODALITIES),
            "embedding_cache_sha256": file_sha256(paths["embedding_cache"]),
            "labels_sha256": file_sha256(paths["labels"]),
            "split_manifest_sha256": file_sha256(paths["split_manifest"]),
            "common_mask_sha256": file_sha256(paths["mask_manifest"]),
        },
        "methods": method_records,
        "foundation_only_sensitivity": {
            "name": "modality_expert_simplex_foundation_only",
            "method_family": "modality_expert_simplex",
            "modalities": list(runner.FOUNDATION_MODALITIES),
            "purpose": "test sensitivity",
            "multiplicity_status": "secondary",
        },
    }


def test_preregistration_hash_provenance_and_positive_gamma_are_enforced(tmp_path):
    paths = {
        name: tmp_path / f"{name}.bin"
        for name in ("embedding_cache", "labels", "split_manifest", "mask_manifest")
    }
    for name, path in paths.items():
        path.write_bytes(name.encode("utf-8"))
    payload = _valid_preregistration_payload(paths)
    args = SimpleNamespace(
        seed=20260705,
        method="joint_block_balanced_pca8",
        **paths,
    )

    method = runner._validate_preregistration(args, payload)
    assert method["name"] == "joint_block_balanced_pca8"

    paths["labels"].write_bytes(b"changed after preregistration")
    with pytest.raises(ValueError, match="input hash mismatch for labels"):
        runner._validate_preregistration(args, payload)

    payload["common_model_rules"]["positive_gammas"] = [0.0, 0.25]
    with pytest.raises(ValueError, match="forbid gamma=0"):
        runner._validate_preregistration(args, payload)


def test_expert_eligibility_rejects_a_foundation_weight_below_floor():
    shared_presence = np.ones(2, dtype=bool)
    result = {
        "test_indexes": np.asarray([0, 1]),
        "models": {target: object() for target in runner.TARGETS},
        "test_corrections": {
            target: (np.asarray([0.1, -0.1]), np.asarray([0.1, -0.1]))
            for target in runner.TARGETS
        },
        "modalities": runner.FORMAL_MODALITIES,
        "weights": {
            target: np.asarray([0.05, 0.05, 0.0, 0.90])
            for target in runner.TARGETS
        },
        "gamma": 0.25,
    }
    assert runner._eligibility(
        "modality_expert_simplex", result, shared_presence
    )["eligible"]

    # FORMAL_MODALITIES is (ecg, eye, head, video); put video below its floor.
    result["weights"]["discomfort"] = np.asarray([0.05, 0.05, 0.86, 0.04])
    evidence = runner._eligibility(
        "modality_expert_simplex", result, shared_presence
    )
    assert evidence["foundation_weight_floor_satisfied"] is False
    assert evidence["eligible"] is False


def test_head_ablation_interpretation_treats_zero_and_near_zero_as_neutral():
    neutral = "near_zero_delta_is_neutral_for_incremental_head_feature_value"
    assert _interpret_head_ablation_delta(0.0) == neutral
    assert _interpret_head_ablation_delta(6.040892869263459e-09) == neutral
    assert _interpret_head_ablation_delta(FLOAT_ATOL / 2) == neutral
    assert _interpret_head_ablation_delta(-FLOAT_ATOL / 2) == neutral
    assert _interpret_head_ablation_delta(2 * FLOAT_ATOL).startswith("positive_delta")
    assert _interpret_head_ablation_delta(-2 * FLOAT_ATOL).startswith("negative_delta")

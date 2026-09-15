from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from mac.evaluation.alignment import validate_alignment_contract
from mac.training.condition_train import _variant_columns as classical_variant_columns
from mac.models.dcnn import _variant_columns as dcnn_variant_columns


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = (
    ROOT / "analysis" / "supplementary" / "evaluate_project_a_eeg_eligible_ablation.py"
)
SPEC = importlib.util.spec_from_file_location("evaluate_project_a_eeg_eligible_ablation", EVALUATOR_PATH)
EVALUATOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(EVALUATOR)


def test_nine_participant_contract_is_supported_by_shared_validator():
    contract_dir = (
        ROOT
        / "artifacts"
        / "cross_project_alignment_2026-07-16"
        / "eeg_eligible_ablation"
        / "contract"
    )
    contract = validate_alignment_contract(contract_dir)

    assert contract["participant_count"] == 9
    assert contract["observation_count"] == 81
    assert contract["fold_contract"] == {
        "folds": 9,
        "train_participants": 7,
        "validation_participants": 1,
        "test_participants": 1,
    }


def test_project_a_ablation_variants_remove_the_named_modality_and_qc_family():
    columns = [
        "eeg_alpha__mean",
        "qc_eeg_usable__mean",
        "mask_eeg_valid__mean",
        "ecg_hr_bpm__mean",
        "qc_ecg_usable__mean",
        "qc_physio_complete__mean",
        "eye_fixation_fraction_ivt__mean",
        "qc_eye_usable__mean",
        "head_speed_mean__mean",
        "qc_head_usable__mean",
        "intensity",
    ]

    no_eeg = classical_variant_columns(columns, "no_eeg")
    no_ecg = classical_variant_columns(columns, "no_ecg")
    no_eye = classical_variant_columns(columns, "no_eye")
    no_head = classical_variant_columns(columns, "no_head")

    assert not any(name.startswith(("eeg_", "qc_eeg_", "mask_eeg_")) for name in no_eeg)
    assert "qc_physio_complete__mean" not in no_eeg
    assert not any(name.startswith(("ecg_", "qc_ecg_", "mask_ecg_")) for name in no_ecg)
    assert "qc_physio_complete__mean" not in no_ecg
    assert not any(name.startswith(("eye_", "qc_eye_", "mask_eye_")) for name in no_eye)
    assert not any(name.startswith(("head_", "qc_head_", "mask_head_")) for name in no_head)
    assert "intensity" in no_eeg and "intensity" in no_ecg


def test_dcnn_supports_all_four_single_modality_removals():
    columns = ["eeg_alpha", "ecg_hr_bpm", "eye_fixation", "head_speed", "video_embedding"]

    assert dcnn_variant_columns(columns, "full") == columns[:4]
    assert dcnn_variant_columns(columns, "no_eeg") == columns[1:4]
    assert dcnn_variant_columns(columns, "no_ecg") == [columns[0], *columns[2:4]]
    assert dcnn_variant_columns(columns, "no_eye") == [columns[0], columns[1], columns[3]]
    assert dcnn_variant_columns(columns, "no_head") == columns[:3]


def test_independent_baselines_cover_all_81_contract_labels():
    contract_dir = (
        ROOT
        / "artifacts"
        / "cross_project_alignment_2026-07-16"
        / "eeg_eligible_ablation"
        / "contract"
    )
    labels = pd.read_csv(contract_dir / "condition_labels.csv")
    _, _, roles = EVALUATOR._fold_context(contract_dir)

    baseline = EVALUATOR._recompute_baselines(labels, roles, 20260705)

    assert len(baseline) == 81
    assert not baseline.duplicated(["participant_id", "condition"]).any()
    assert set(baseline["fold_index"]) == set(range(1, 10))
    for prefix in ("condition_only", "history"):
        for target in EVALUATOR.TARGETS:
            assert np.isfinite(baseline[f"{prefix}_{target}"]).all()


def test_project_a_inference_uses_two_separate_eight_test_holm_families():
    rows = []
    for model_index, model_family in enumerate(EVALUATOR.MODELS):
        for variant_index, variant in enumerate(EVALUATOR.VARIANTS):
            for seed in EVALUATOR.SEEDS:
                for participant_index, participant in enumerate(EVALUATOR.PARTICIPANTS):
                    for target_index, target in enumerate(EVALUATOR.TARGETS):
                        rows.append(
                            {
                                "model": f"{model_family}_{variant}",
                                "seed": seed,
                                "participant_id": participant,
                                "target": target,
                                "mae": (
                                    0.1
                                    + model_index * 0.001
                                    + variant_index * 0.01
                                    + participant_index * 0.0001
                                    + target_index * 0.00001
                                ),
                            }
                        )
    tests = EVALUATOR._ablation_tests(
        pd.DataFrame(rows), bootstrap=1000, bootstrap_seed=20260717
    )

    assert len(tests) == 16
    assert all(len(group) == 8 for _, group in tests.groupby("holm_family"))
    assert set(tests["n_participants"]) == {9}

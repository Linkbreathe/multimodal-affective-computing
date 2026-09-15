from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from analysis.supplementary.build_alignment_contract import apply_masks_to_window_features
from real_time_ml.config import ProjectConfig, load_config
from real_time_ml.evaluation.alignment import indexes_for_fold, load_split_manifest
from real_time_ml.modeling.condition_models import ModelSpec
from real_time_ml.modeling.condition_train import _rank_regression_on_validation


def _write_manifest(path, participants: list[str]) -> None:
    rows = []
    for fold_index, test in enumerate(participants, start=1):
        validation = participants[fold_index % len(participants)]
        for participant in participants:
            role = "test" if participant == test else "validation" if participant == validation else "train"
            rows.append(
                {
                    "fold_index": fold_index,
                    "test_participant": test,
                    "validation_participant": validation,
                    "participant_id": participant,
                    "role": role,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_split_manifest_enforces_one_test_and_validation_per_fold(tmp_path):
    participants = ["P002", "P003", "P004", "P005"]
    path = tmp_path / "splits.csv"
    _write_manifest(path, participants)

    folds = load_split_manifest(
        path,
        expected_participants=participants,
        expected_train_count=2,
    )

    assert len(folds) == 4
    assert folds[0].train_participants == ("P004", "P005")
    assert folds[0].validation_participant == "P003"
    assert folds[0].test_participant == "P002"
    groups = np.asarray(["P005", "P002", "P004", "P003", "P002"])
    train, validation, test = indexes_for_fold(groups, folds[0])
    assert set(groups[train]) == {"P004", "P005"}
    assert set(groups[validation]) == {"P003"}
    assert set(groups[test]) == {"P002"}


def test_split_manifest_rejects_role_header_disagreement(tmp_path):
    participants = ["P002", "P003", "P004", "P005"]
    path = tmp_path / "splits.csv"
    _write_manifest(path, participants)
    frame = pd.read_csv(path)
    frame.loc[(frame["fold_index"] == 1) & (frame["participant_id"] == "P003"), "role"] = "train"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="exactly one validation"):
        load_split_manifest(path, expected_participants=participants, expected_train_count=2)


def test_shared_masks_remove_only_the_named_modality_values():
    features = pd.DataFrame(
        [
            {
                "participant_id": "P002",
                "condition": "C1",
                "condition_window_index": 0,
                "eeg_alpha": 1.0,
                "ecg_hr_bpm": 60.0,
                "eye_valid_fraction": 0.8,
                "head_speed_mean": 0.2,
                "qc_eeg_usable": False,
            }
        ]
    )
    masks = pd.DataFrame(
        [
            {
                "participant_id": "P002",
                "condition": "C1",
                "condition_window_index": 0,
                "eeg_valid": False,
                "ecg_valid": True,
                "eye_valid": False,
                "head_valid": True,
                "video_valid": False,
            }
        ]
    )

    masked = apply_masks_to_window_features(features, masks).iloc[0]

    assert np.isnan(masked["eeg_alpha"])
    assert np.isnan(masked["eye_valid_fraction"])
    assert masked["ecg_hr_bpm"] == 60.0
    assert masked["head_speed_mean"] == 0.2
    assert masked["qc_eeg_usable"] is False or not bool(masked["qc_eeg_usable"])
    assert not bool(masked["mask_eeg_valid"])
    assert bool(masked["mask_ecg_valid"])


def test_validation_ranking_uses_dedicated_validation_rows(tmp_path):
    data = deepcopy(load_config().data)
    data["modeling"]["condition_level"]["min_non_missing_fraction"] = 0.0
    config = ProjectConfig(source=tmp_path / "config.yaml", data=data)
    train = pd.DataFrame(
        {
            "participant_id": ["P002", "P002", "P003", "P003"],
            "condition": ["C1", "C2", "C1", "C2"],
            "presentation_position": [1, 2, 1, 2],
            "relaxation": [0.2, 0.8, 0.3, 0.7],
            "discomfort": [0.8, 0.2, 0.7, 0.3],
            "feature": [0.0, 1.0, 0.1, 0.9],
        }
    )
    validation = pd.DataFrame(
        {
            "participant_id": ["P004", "P004"],
            "condition": ["C1", "C2"],
            "presentation_position": [1, 2],
            "relaxation": [0.25, 0.75],
            "discomfort": [0.75, 0.25],
            "feature": [0.05, 0.95],
        }
    )

    ranked = _rank_regression_on_validation(
        train,
        validation,
        ["feature"],
        "relaxation",
        [ModelSpec("ridge", 1)],
        config,
        pd,
    )

    assert len(ranked) == 1
    assert np.isfinite(ranked[0]["mae"])
    assert ranked[0]["spec"] == ModelSpec("ridge", 1)

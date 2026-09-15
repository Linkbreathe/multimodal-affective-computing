from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data.relax_foundation import (
    CONDITIONS,
    EEG_DISABLED_PARTICIPANTS,
    RelaxHardFailure,
    assert_relax_modalities,
    build_cohorts,
    build_qc_manifest,
    select_high_quality_pilot,
    validate_phase0_inputs,
)
from src.tasks.relaxation import random_9_condition_baseline


def _write_relax_tables(root, participants=("P003", "P004", "P015")):
    pre = root / "preprocessed"
    feat = root / "features"
    rep = root / "reports"
    pre.mkdir(parents=True)
    feat.mkdir()
    rep.mkdir()

    labels = []
    windows = []
    features = []
    for p_idx, participant in enumerate(participants):
        for c_idx, condition in enumerate(CONDITIONS):
            labels.append(
                {
                    "participant_id": participant,
                    "condition": condition,
                    "relaxation": (c_idx + 1) / 10,
                    "discomfort": (9 - c_idx) / 10,
                }
            )
            for w_idx in range(2):
                row = {
                    "participant_id": participant,
                    "condition": condition,
                    "condition_window_index": w_idx,
                    "condition_window_count": 2,
                    "relaxation": (c_idx + 1) / 10,
                    "discomfort": (9 - c_idx) / 10,
                }
                windows.append(row)
                features.append(
                    {
                        **row,
                        "qc_ecg_usable": True,
                        "qc_eeg_usable": participant not in EEG_DISABLED_PARTICIPANTS,
                        "qc_eeg_strict_coverage": 0.85,
                        "qc_eye_usable": True,
                        "qc_head_usable": True,
                        "qc_video_usable": True,
                        "qc_video_coverage": 0.95,
                        "qc_head_coverage": 0.98,
                        "qc_eye_coverage": 0.91,
                    }
                )
    pd.DataFrame(labels).to_csv(pre / "condition_labels.csv", index=False)
    pd.DataFrame(windows).to_csv(pre / "windows.csv", index=False)
    pd.DataFrame(features).to_csv(feat / "window_features.csv", index=False)
    qc = {
        "participants": [
            {
                "participant_id": p,
                "status": "ok",
                "condition_count": 9,
                "window_count": 18,
                "median_abs_residual_ms": 2.0 + i,
                "max_abs_residual_ms": 8.0 + i,
                "outliers": [],
            }
            for i, p in enumerate(participants)
        ]
    }
    (rep / "data_qc.json").write_text(json.dumps(qc), encoding="utf-8")


def test_validate_phase0_fails_without_reve_weights(tmp_path):
    _write_relax_tables(tmp_path)

    with pytest.raises(RelaxHardFailure, match="REVE weights"):
        validate_phase0_inputs(
            tmp_path,
            eeg_weights=tmp_path / "missing_reve.safetensors",
            eeg_pos_bank=tmp_path / "positions.safetensors",
            labels_root=None,
            require_reve=True,
        )


def test_qc_manifest_and_high_quality_pilot_are_pre_training_and_deterministic(tmp_path):
    _write_relax_tables(tmp_path)
    manifest = build_qc_manifest(tmp_path, participants=["P003", "P004", "P015"])

    pilot = select_high_quality_pilot(
        manifest,
        min_count=2,
        max_count=2,
    )

    assert pilot == ["P003", "P004"]
    cohorts = build_cohorts(manifest, high_quality_min_count=2, high_quality_max_count=2)
    assert cohorts["high_quality_pilot"]["participants"] == ["P003", "P004"]
    assert set(cohorts["eeg_eligible"]["participants"]) == {"P003", "P004", "P015"}


def test_random_9_condition_baseline_uses_only_training_condition_means():
    train = pd.DataFrame(
        {
            "participant_id": ["P003"] * 9,
            "condition": CONDITIONS,
            "relaxation": np.arange(9, dtype=float),
            "discomfort": np.arange(9, dtype=float) + 10,
        }
    )
    test = pd.DataFrame(
        {
            "participant_id": ["P004"] * 3,
            "condition": ["C1", "C2", "C3"],
            "relaxation": [100.0, 100.0, 100.0],
            "discomfort": [200.0, 200.0, 200.0],
        }
    )

    out = random_9_condition_baseline(train, test, seeds=[0, 1, 2])

    assert set(out["predictions"]["chosen_condition"]).issubset(set(CONDITIONS))
    assert "relaxation_mae_mean" in out["summary"]
    assert out["summary"]["num_seeds"] == 3


def test_random_9_condition_baseline_hard_fails_when_training_condition_missing():
    train = pd.DataFrame(
        {
            "participant_id": ["P003"] * 8,
            "condition": CONDITIONS[:8],
            "relaxation": np.arange(8, dtype=float),
            "discomfort": np.arange(8, dtype=float),
        }
    )
    test = pd.DataFrame({"participant_id": ["P004"], "condition": ["C1"], "relaxation": [0.0], "discomfort": [0.0]})

    with pytest.raises(RelaxHardFailure, match="missing Condition"):
        random_9_condition_baseline(train, test, seeds=[0])


def test_relax_modalities_reject_ppg_and_papagei():
    with pytest.raises(RelaxHardFailure, match="PPG/Papagei"):
        assert_relax_modalities(["ecg", "ppg"])
    with pytest.raises(RelaxHardFailure, match="PPG/Papagei"):
        assert_relax_modalities(["papagei_ppg"])

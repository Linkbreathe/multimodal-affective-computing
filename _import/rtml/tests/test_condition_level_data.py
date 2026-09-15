from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from real_time_ml.config import load_config
from real_time_ml.modeling.condition_data import aggregate_window_frame, build_condition_dataset
from real_time_ml.modeling.condition_train import _feature_columns
from real_time_ml.modeling.groups import columns_for_group
from real_time_ml.modeling.safety import deployment_guard


def test_window_features_become_one_condition_label_with_full_summary_statistics():
    windows = pd.DataFrame([
        {"participant_id": "P002", "condition": "C1", "presentation_position": 1, "condition_window_index": 0, "condition_window_count": 3, "intensity": 0.08, "frequency": 0.12, "relaxation": 0.7, "discomfort": 0.2, "head_speed_mean": 1.0},
        {"participant_id": "P002", "condition": "C1", "presentation_position": 1, "condition_window_index": 1, "condition_window_count": 3, "intensity": 0.08, "frequency": 0.12, "relaxation": 0.7, "discomfort": 0.2, "head_speed_mean": np.nan},
        {"participant_id": "P002", "condition": "C1", "presentation_position": 1, "condition_window_index": 2, "condition_window_count": 3, "intensity": 0.08, "frequency": 0.12, "relaxation": 0.7, "discomfort": 0.2, "head_speed_mean": 5.0},
    ])
    aggregated = aggregate_window_frame(windows)
    assert len(aggregated) == 1
    row = aggregated.iloc[0]
    assert row["relaxation"] == 0.7
    assert row["discomfort"] == 0.2
    assert row["window_count"] == 3
    assert row["head_speed_mean__mean"] == 3.0
    assert row["head_speed_mean__min"] == 1.0
    assert row["head_speed_mean__max"] == 5.0
    assert row["head_speed_mean__first"] == 1.0
    assert row["head_speed_mean__last"] == 5.0
    assert abs(row["head_speed_mean__missing_ratio"] - 1 / 3) < 1e-12
    # range (=max-min) and delta (=last-first) are no longer emitted; keep the removal enforced.
    assert "head_speed_mean__range" not in aggregated.columns
    assert "head_speed_mean__delta" not in aggregated.columns


def test_deployment_guard_for_condition_level_requires_ranking_and_risk_recall():
    metrics = {
        "unit_of_analysis": "participant_condition",
        "targets": {
            "relaxation": {"mae": 0.1, "condition_only_baseline_mae": 0.2, "history_baseline_mae": 0.2, "spearman": 0.3},
            "discomfort": {"mae": 0.1, "condition_only_baseline_mae": 0.2, "history_baseline_mae": 0.2, "risk_at_fold_tuned_threshold": {"per_row_threshold_recall": 0.25}},
        },
    }
    deployable, reasons = deployment_guard(metrics)
    assert not deployable
    assert reasons == ["condition_level_discomfort_gate_failed"]


def test_untrusted_ecg_hrv_features_are_excluded_from_model_columns():
    columns = [
        "ecg_hr_bpm",
        "ecg_rr_median_ms",
        "ecg_hrv_30s_rmssd_ms",
        "ecg_hrv_30s_rmssd_ms__mean",
        "ecg_hrv_300s_sdnn_ms",
        "ecg_rr_std_ms_audit_only",
        "ecg_rr_std_ms_audit_only__mean",
        "head_speed_mean",
        "eeg_tp9_alpha_relative",
    ]
    assert columns_for_group(columns, "no_eeg") == [
        "ecg_hr_bpm",
        "ecg_rr_median_ms",
        "head_speed_mean",
    ]

    frame = pd.DataFrame([{name: 1.0 for name in columns}])
    selected = _feature_columns(frame)
    assert "ecg_hr_bpm" in selected
    assert "ecg_rr_median_ms" in selected
    assert "ecg_hrv_30s_rmssd_ms" not in selected
    assert "ecg_hrv_30s_rmssd_ms__mean" not in selected
    assert "ecg_hrv_300s_sdnn_ms" not in selected
    assert "ecg_rr_std_ms_audit_only" not in selected
    assert "ecg_rr_std_ms_audit_only__mean" not in selected


@pytest.mark.integration
def test_real_window_feature_table_aggregates_to_exactly_135_condition_labels():
    config = load_config()
    source = config.path("features") / "window_features.csv"
    if not source.exists():
        pytest.skip("window features have not been extracted")
    frame = build_condition_dataset(source)
    assert len(frame) == 135
    assert not frame[["participant_id", "condition"]].duplicated().any()
    assert 7 in set(frame["window_count"].unique())
    assert any(column.endswith("__slope") for column in frame.columns)
    assert any(column.endswith("__missing_ratio") for column in frame.columns)
    assert not any(column.startswith(("relaxation_raw__", "discomfort_raw__", "arousal_raw__")) for column in frame.columns)


@pytest.mark.integration
def test_trained_condition_model_has_135_label_evidence_and_no_questionnaire_feature_leakage():
    import joblib
    import json

    config = load_config()
    model_path = config.path("models") / "state_model.joblib"
    metrics_path = config.path("reports") / "condition_level_lopo_metrics.json"
    if not model_path.exists() or not metrics_path.exists():
        pytest.skip("condition-level model has not been trained")
    bundle = joblib.load(model_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))["metrics"]
    assert bundle["model_kind"] == "condition_residual_ensemble_v1"
    assert bundle["targets"] == ["relaxation", "discomfort"]
    assert metrics["unit_of_analysis"] == "participant_condition"
    assert metrics["n_labels"] == 135
    assert not any("_raw" in column or column in {"relaxation", "discomfort"} for column in bundle["feature_columns"])

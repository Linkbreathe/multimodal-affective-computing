from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from real_time_ml.cli import build_parser
from real_time_ml.config import ProjectConfig, load_config
from real_time_ml.modeling import minimal_fusion_dcnn, minimal_fusion_dcnn_hp
from real_time_ml.modeling.latest_multimodal_report import write_latest_multimodal_report


torch = pytest.importorskip("torch")


H_FEATURES = [
    *(f"head_speed_{name}" for name in ("mean", "std", "median", "iqr", "range")),
    *(f"head_angular_speed_deg_s_{name}" for name in ("mean", "std", "median", "iqr", "range")),
    *(f"head_jerk_{name}" for name in ("mean", "std", "median", "iqr", "range")),
    "head_stationary_fraction",
    "head_position_range",
    "head_motion_spectral_entropy",
]
P_FEATURES = [
    "ecg_hr_bpm",
    "ecg_hrv_30s_rmssd_ms",
    "ecg_rr_std_ms_audit_only",
    "eeg_tp9_alpha_relative",
    "eeg_tp9_alpha_power",
    "eeg_tp9_hjorth_activity",
    "eeg_alpha_beta_ratio",
]


def _config(tmp_path: Path) -> ProjectConfig:
    data = deepcopy(load_config().data)
    data["paths"] = {key: str(tmp_path / key) for key in data["paths"]}
    data["modeling"]["dcnn"].update(
        {"device": "cpu", "max_epochs": 1, "early_stopping_patience": 1, "batch_size": 256}
    )
    config = ProjectConfig(source=tmp_path / "project.yaml", data=data)
    config.ensure_artifact_dirs()
    return config


def _window_frame() -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for participant_number in range(2, 17):
        for condition_number in range(1, 10):
            relaxation = (condition_number + (participant_number % 4) + 2) / 15.0
            discomfort = ((10 - condition_number) + (participant_number % 3)) / 15.0
            for window_index in range(4):
                row: dict[str, float | int | str] = {
                    "participant_id": f"P{participant_number:03d}",
                    "condition": f"C{condition_number}",
                    "presentation_position": ((condition_number + participant_number - 3) % 9) + 1,
                    "condition_window_index": window_index,
                    "relaxation": relaxation,
                    "discomfort": discomfort,
                }
                for index, name in enumerate(H_FEATURES):
                    row[name] = relaxation + 0.002 * index + 0.001 * window_index
                for index, name in enumerate(P_FEATURES):
                    row[name] = discomfort + 0.003 * index + 0.001 * window_index
                for prefix, value in (("eye_", discomfort), ("video_", relaxation)):
                    for index in range(3):
                        row[f"{prefix}feature_{index:02d}"] = value + 0.01 * index
                rows.append(row)
    return pd.DataFrame(rows)


def _zero_residual_trainer(*args, **kwargs):
    feature_indexes = np.asarray(args[3], dtype=int)
    return object(), {"feature_median": np.zeros(len(feature_indexes)), "feature_scale": np.ones(len(feature_indexes))}, 0.0


def _zero_residual_predict(model, sequences, indexes, feature_indexes, scaler, device):
    return np.zeros(len(indexes), dtype=float)


def test_hp_mapping_is_complete_and_audit_is_fold_local(tmp_path, monkeypatch):
    config = _config(tmp_path)
    source = config.path("features") / "video_ml" / "window_features.csv"
    source.parent.mkdir(parents=True, exist_ok=True)
    _window_frame().to_csv(source, index=False)
    monkeypatch.setattr(minimal_fusion_dcnn, "_train_residual_model", _zero_residual_trainer)
    monkeypatch.setattr(minimal_fusion_dcnn, "_predict_residual_model", _zero_residual_predict)
    minimal_fusion_dcnn.benchmark_minimal_fusion_dcnn(config, random_simulations=3)
    reference = config.path("artifacts") / "fusion_minimal_dcnn" / "oof_predictions.csv"
    original_reference = reference.read_bytes()

    monkeypatch.setattr(minimal_fusion_dcnn_hp, "_train_residual_model", _zero_residual_trainer)
    monkeypatch.setattr(minimal_fusion_dcnn_hp, "_predict_residual_model", _zero_residual_predict)
    result = minimal_fusion_dcnn_hp.analyze_minimal_fusion_dcnn_hp(config)

    audit = result["selection_audit"]
    assert len(audit) == 15 * 2 * (len(H_FEATURES) + len(P_FEATURES))
    assert audit.groupby(["held_out_participant", "target", "feature_name"]).size().eq(1).all()
    assert audit.loc[audit["modality"].eq("H"), "feature_cap_binding"].eq(False).all()
    assert audit.loc[audit["feature_name"].eq("ecg_rr_std_ms_audit_only"), "family"].eq(
        "ecg_rr_std_ms_audit_only"
    ).all()
    assert set(audit.loc[audit["modality"].eq("H"), "family"]) == set(
        minimal_fusion_dcnn_hp.H_FAMILIES
    )
    assert set(audit.loc[audit["modality"].eq("P"), "family"]) == set(
        minimal_fusion_dcnn_hp.P_FAMILIES
    )
    assert reference.read_bytes() == original_reference
    assert not list(config.path("models").glob("*.pt"))

    metrics = result["family_ablation_metrics"]
    assert len(metrics) == 2 + len(minimal_fusion_dcnn_hp.H_FAMILIES) + len(
        minimal_fusion_dcnn_hp.P_FAMILIES
    )
    assert metrics.loc[metrics["variant"].eq("remove_family"), "feature_removal_stage"].eq(
        "before_fold_selection_imputation_scaling_training"
    ).all()
    subgroup = result["subgroup_metrics"]
    reference_subgroups = subgroup.loc[subgroup["variant"].eq("full_reference")]
    assert set(reference_subgroups["subgroup"]) == {"eeg_disabled", "eeg_available"}
    assert set(reference_subgroups.groupby("subgroup")["n_labels"].unique().explode()) == {54, 81}

    output = config.path("artifacts") / "fusion_minimal_dcnn_hp"
    assert {path.name for path in output.iterdir()} == {
        "selection_audit.csv",
        "selection_stability.csv",
        "feature_family_ablation_metrics.csv",
        "subgroup_metrics.csv",
    }
    assert "yaw/pitch/roll 方向变化特征" in Path(result["report"]).read_text(encoding="utf-8")
    assert build_parser().parse_args(["analyze-minimal-fusion-dcnn-hp"]).command == "analyze-minimal-fusion-dcnn-hp"


def test_latest_report_indexes_sources_and_keeps_research_boundary(tmp_path):
    config = _config(tmp_path)
    reports = config.path("reports")
    reports.joinpath("data_qc.json").write_text(
        '{"participants": [{"available": true, "window_count": 9, "median_abs_residual_ms": 1.0}]}',
        encoding="utf-8",
    )
    reports.joinpath("window_level_supervision_retired_zh.md").write_text("历史", encoding="utf-8")
    metrics_dir = config.path("artifacts") / "fusion_minimal_dcnn_hp"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{
            "modality": "H", "variant": "full_reference", "removed_family": "",
            "relaxation_mae": 0.1, "discomfort_mae": 0.2, "discomfort_high_recall": 0.3,
            "feature_count_after_removal": 18, "reference_source": "fusion_minimal_dcnn/oof_predictions.csv",
        }]
    ).to_csv(metrics_dir / "feature_family_ablation_metrics.csv", index=False)
    result = write_latest_multimodal_report(config)
    report = Path(result["report"]).read_text(encoding="utf-8")
    sources = pd.read_csv(result["sources"])
    assert "H/P 可解释性审计" in report
    assert "不改变运行时、自动推荐资格或 Shadow/hold 策略" in report
    assert "research_only" in set(sources["status"])
    assert "historical" in set(sources["status"])
    assert build_parser().parse_args(["report-latest-multimodal"]).command == "report-latest-multimodal"

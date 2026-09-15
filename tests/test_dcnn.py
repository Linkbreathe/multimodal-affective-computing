from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from real_time_ml.config import ProjectConfig, load_config
from real_time_ml.modeling.dcnn import (
    _checkpoint_from_model,
    _fit_scaler,
    _make_model,
    _transform,
    build_condition_sequences,
    load_dcnn_state_model,
    predict_dcnn_state,
)
from real_time_ml.modeling import dcnn
from real_time_ml.realtime.engine import InferenceEngine


torch = pytest.importorskip("torch")


def _config(tmp_path) -> ProjectConfig:
    data = deepcopy(load_config().data)
    data["paths"] = {key: str(tmp_path / key) for key in data["paths"]}
    data["modeling"]["runtime_backend"] = "dcnn"
    data["modeling"]["dcnn"]["device"] = "cpu"
    config = ProjectConfig(source=tmp_path / "project.yaml", data=data)
    config.ensure_artifact_dirs()
    return config


def _window_table(path) -> None:
    rows = []
    for participant, offset in (("P002", 0.0), ("P003", 1.0)):
        for condition, position, label in (("C1", 1, (0.2, 0.6)), ("C2", 2, (0.8, 0.2))):
            for index in range(7):
                rows.append(
                    {
                        "participant_id": participant,
                        "condition": condition,
                        "presentation_position": position,
                        "condition_window_index": index,
                        "relaxation": label[0],
                        "discomfort": label[1],
                        "intensity": 0.08 if condition == "C1" else 0.16,
                        "frequency": 0.12 if condition == "C1" else 0.26,
                        "eeg_alpha": offset + index,
                        "ecg_hr_bpm": 60.0 + offset + index,
                        "head_speed_mean": 0.1 * index,
                        "eye_fixation_fraction_ivt": 0.5 + 0.01 * index,
                        "video_not_realtime": index,
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False)


def _metrics() -> dict:
    return {
        "unit_of_analysis": "participant_condition",
        "targets": {
            "relaxation": {"mae": 0.1, "condition_only_baseline_mae": 0.2, "history_baseline_mae": 0.2, "spearman": 0.1},
            "discomfort": {
                "mae": 0.1,
                "condition_only_baseline_mae": 0.2,
                "history_baseline_mae": 0.2,
                "risk_at_fold_tuned_threshold": {"per_row_threshold_recall": 0.8},
            },
        },
        "deployable": True,
        "deployment_block_reasons": [],
    }


def _save_checkpoint(config: ProjectConfig, sequences) -> None:
    indexes = np.arange(len(sequences.targets))
    scaler = _fit_scaler(sequences, indexes, 0.4)
    architecture = {
        "sequence_length": 8,
        "conv_channels": (16, 32),
        "kernel_sizes": (3, 3),
        "pool_sizes": (2, 2),
        "mlp_hidden": 64,
        "dropout": 0.3,
    }
    model = _make_model(len(scaler["feature_indexes"]), architecture)
    values, context = _transform(sequences, indexes, scaler)
    with torch.no_grad():
        model(torch.as_tensor(values), torch.as_tensor(context))
    checkpoint = _checkpoint_from_model(
        model,
        scaler=scaler,
        sequences=sequences,
        architecture=architecture,
        variant="full",
        metrics=_metrics(),
        interval_by_history={str(index): {"relaxation": 0.2, "discomfort": 0.2} for index in range(1, 9)},
        config=config,
    )
    torch.save(checkpoint, config.path("models") / "dcnn_state_full.pt")


def test_dcnn_condition_sequences_are_feature_by_window_and_exclude_video(tmp_path):
    config = _config(tmp_path)
    source = tmp_path / "window_features.csv"
    _window_table(source)
    full = build_condition_sequences(source, config, "full")
    no_eeg = build_condition_sequences(source, config, "no_eeg")
    behavior = build_condition_sequences(source, config, "behavior_only")

    assert full.values.shape == (4, 4, 8)
    assert full.feature_columns == ("eeg_alpha", "ecg_hr_bpm", "head_speed_mean", "eye_fixation_fraction_ivt")
    assert no_eeg.values.shape == (4, 3, 8)
    assert behavior.feature_columns == ("head_speed_mean", "eye_fixation_fraction_ivt")
    assert np.allclose(full.values[0, 0, :7], np.arange(7))
    assert np.isnan(full.values[0, 0, 7])


def test_dcnn_checkpoint_round_trip_and_causal_history(tmp_path):
    config = _config(tmp_path)
    source = tmp_path / "window_features.csv"
    _window_table(source)
    sequences = build_condition_sequences(source, config, "full")
    _save_checkpoint(config, sequences)
    bundle = load_dcnn_state_model(config.path("models") / "dcnn_state_full.pt", "cpu")

    prediction, widths, history = predict_dcnn_state(
        bundle,
        [{"eeg_alpha": 0.0, "ecg_hr_bpm": 60.0, "head_speed_mean": 0.0, "eye_fixation_fraction_ivt": 0.5}],
        "C1",
        config,
    )
    assert history == 1
    assert set(prediction) == {"relaxation", "discomfort"}
    assert all(0.0 <= value <= 1.0 for value in prediction.values())
    assert widths == {"relaxation": 0.2, "discomfort": 0.2}


def test_dcnn_engine_resets_history_and_holds_until_three_windows(tmp_path):
    config = _config(tmp_path)
    source = tmp_path / "window_features.csv"
    _window_table(source)
    _save_checkpoint(config, build_condition_sequences(source, config, "full"))
    engine = InferenceEngine(config)
    features = {"eeg_alpha": 0.0, "ecg_hr_bpm": 60.0, "head_speed_mean": 0.0, "eye_fixation_fraction_ivt": 0.5}
    coverage = {"eeg": 1.0, "ecg": 1.0, "head": 1.0, "eye": 1.0}

    state, recommendation = engine.infer(
        participant_id="P002", condition="C1", cycle_index=0, start_ms=0, end_ms=10_000,
        features=features, qc={}, coverage=coverage,
    )
    assert state.qc["dcnn_history_windows"] == 1
    assert "dcnn_insufficient_history" in recommendation.reasons
    state, _ = engine.infer(
        participant_id="P002", condition="C1", cycle_index=1, start_ms=10_000, end_ms=20_000,
        features=features, qc={}, coverage=coverage,
    )
    assert state.qc["dcnn_history_windows"] == 2
    state, _ = engine.infer(
        participant_id="P002", condition="C2", cycle_index=0, start_ms=20_000, end_ms=30_000,
        features=features, qc={}, coverage=coverage,
    )
    assert state.qc["dcnn_history_windows"] == 1


@pytest.mark.slow
def test_dcnn_training_writes_lopo_checkpoint_and_report(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config.data["modeling"]["dcnn"].update({"max_epochs": 1, "early_stopping_patience": 1, "batch_size": 256})
    rows = []
    for participant_number in range(2, 17):
        for condition_number in range(1, 10):
            relaxation = min(1.0, 0.1 + 0.08 * condition_number + 0.002 * participant_number)
            discomfort = max(0.0, 0.9 - 0.07 * condition_number + 0.001 * participant_number)
            for index in range(7):
                rows.append(
                    {
                        "participant_id": f"P{participant_number:03d}",
                        "condition": f"C{condition_number}",
                        "presentation_position": condition_number,
                        "condition_window_index": index,
                        "relaxation": relaxation,
                        "discomfort": discomfort,
                        "intensity": (0.08, 0.16, 0.25)[(condition_number - 1) // 3],
                        "frequency": (0.12, 0.26, 0.41)[(condition_number - 1) % 3],
                        "eeg_alpha": participant_number + condition_number + index,
                        "ecg_hr_bpm": 60.0 + participant_number + condition_number + index,
                        "head_speed_mean": 0.01 * (condition_number + index),
                        "eye_fixation_fraction_ivt": 0.2 + 0.01 * participant_number + 0.02 * index,
                    }
                )
    pd.DataFrame(rows).to_csv(config.path("features") / "window_features.csv", index=False)
    monkeypatch.setattr(dcnn, "VARIANTS", ("full",))

    result = dcnn.train_dcnn_state(config)

    assert set(result["variants"]) == {"full"}
    assert (config.path("models") / "dcnn_state_full.pt").exists()
    assert (config.path("reports") / "dcnn_condition_lopo_predictions.csv").exists()
    assert "不得自动切换到 DCNN" in (config.path("reports") / "dcnn_condition_comparison_zh.md").read_text(encoding="utf-8")

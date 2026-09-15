from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from real_time_ml.cli import build_parser
from real_time_ml.config import ProjectConfig, load_config
from real_time_ml.modeling import minimal_fusion_dcnn
from real_time_ml.modeling.minimal_fusion import COMBINATIONS, FEATURES_PER_MODALITY, MODALITY_ORDER


torch = pytest.importorskip("torch")


def _config(tmp_path) -> ProjectConfig:
    data = deepcopy(load_config().data)
    data["paths"] = {key: str(tmp_path / key) for key in data["paths"]}
    data["modeling"]["dcnn"].update(
        {"device": "cpu", "max_epochs": 1, "early_stopping_patience": 1, "batch_size": 256}
    )
    config = ProjectConfig(source=tmp_path / "project.yaml", data=data)
    config.ensure_artifact_dirs()
    return config


def _window_frame(participants=range(2, 17)) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for participant_number in participants:
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
                    "condition_context_leak": relaxation,
                    "qc_eeg_usable": 1.0,
                }
                for prefix, value in (
                    ("eeg_", relaxation),
                    ("ecg_", discomfort),
                    ("head_", relaxation),
                    ("eye_", discomfort),
                    ("video_", relaxation),
                ):
                    for feature_index in range(3):
                        row[f"{prefix}feature_{feature_index:02d}"] = (
                            value + 0.01 * feature_index + 0.001 * window_index
                        )
                rows.append(row)
    return pd.DataFrame(rows)


def _zero_residual_trainer(*args, **kwargs):
    feature_indexes = np.asarray(args[3], dtype=int)
    return object(), {"feature_median": np.zeros(len(feature_indexes)), "feature_scale": np.ones(len(feature_indexes))}, 0.0


def _zero_residual_predict(model, sequences, indexes, feature_indexes, scaler, device):
    return np.zeros(len(indexes), dtype=float)


def test_minimal_fusion_dcnn_protocol_lopo_feature_caps_and_holdout_labels(tmp_path, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.setattr(minimal_fusion_dcnn, "_train_residual_model", _zero_residual_trainer)
    monkeypatch.setattr(minimal_fusion_dcnn, "_predict_residual_model", _zero_residual_predict)
    source = _window_frame()
    result = minimal_fusion_dcnn.evaluate_minimal_fusion_dcnn_frame(source, config, random_simulations=7)

    metrics = result["metrics"]
    combinations = metrics.loc[metrics["record_type"].eq("combination")]
    oof = result["oof_predictions"]
    assert set(combinations["combination"]) == set(COMBINATIONS)
    assert len(oof) == len(COMBINATIONS) * 135 * 2
    assert oof.groupby(["combination", "target"]).size().eq(135).all()
    assert len(metrics.loc[metrics["record_type"].eq("ablation")]) == 28
    assert set(result["sequences"].feature_columns).isdisjoint({"condition_context_leak", "qc_eeg_usable"})
    for modality in MODALITY_ORDER:
        for target in ("relaxation", "discomfort"):
            assert combinations[f"selected_{modality}_{target}_max"].le(FEATURES_PER_MODALITY).all()

    altered = source.copy()
    held_out = altered["participant_id"].eq("P002")
    altered.loc[held_out, "relaxation"] = 1.0 - altered.loc[held_out, "relaxation"]
    altered.loc[held_out, "discomfort"] = 1.0 - altered.loc[held_out, "discomfort"]
    altered_result = minimal_fusion_dcnn.evaluate_minimal_fusion_dcnn_frame(
        altered, config, random_simulations=7
    )
    original_p002 = oof.query("combination == 'P' and participant_id == 'P002'")
    altered_p002 = altered_result["oof_predictions"].query(
        "combination == 'P' and participant_id == 'P002'"
    )
    assert np.allclose(
        original_p002["condition_only_prediction"], altered_p002["condition_only_prediction"]
    )
    assert np.allclose(original_p002["prediction"], altered_p002["prediction"])
    history = original_p002.query("target == 'relaxation'").sort_values("presentation_position")
    fallback = source.loc[~source["participant_id"].eq("P002"), "relaxation"].mean()
    assert np.isclose(history.iloc[0]["history_prediction"], fallback)
    assert np.allclose(history.iloc[1:]["history_prediction"], history.iloc[:-1]["truth"])


def test_minimal_fusion_dcnn_cpu_one_epoch_head_and_seed(tmp_path):
    config = _config(tmp_path)
    sequences = minimal_fusion_dcnn.build_minimal_fusion_sequences(
        _window_frame(range(2, 4)), sequence_length=8, expected_labels=None
    )
    indexes = np.arange(len(sequences.targets))
    feature_indexes = np.asarray([0, 1], dtype=int)
    residual = np.zeros(len(indexes), dtype=float)
    model, scaler, _ = minimal_fusion_dcnn._train_residual_model(
        sequences,
        indexes,
        indexes,
        feature_indexes,
        residual,
        minimal_fusion_dcnn._architecture(config),
        config,
        minimal_fusion_dcnn._fold_seed(7, "relaxation", "P", 1),
        torch.device("cpu"),
    )
    values, context = minimal_fusion_dcnn._transform_sequences(
        sequences, indexes[:1], feature_indexes, scaler
    )
    with torch.no_grad():
        output = model(torch.as_tensor(values), torch.as_tensor(context)).numpy()
    assert context.shape == (1, 0)
    assert output.shape == (1, 1)
    assert -1.0 <= output[0, 0] <= 1.0
    assert minimal_fusion_dcnn._fold_seed(20260621, "discomfort", "PHEV", 15) == minimal_fusion_dcnn._fold_seed(
        20260621, "discomfort", "PHEV", 15
    )


def test_minimal_fusion_dcnn_writes_only_three_requested_outputs(tmp_path, monkeypatch):
    config = _config(tmp_path)
    source = config.path("features") / "video_ml" / "window_features.csv"
    source.parent.mkdir(parents=True, exist_ok=True)
    _window_frame().to_csv(source, index=False)
    monkeypatch.setattr(minimal_fusion_dcnn, "_train_residual_model", _zero_residual_trainer)
    monkeypatch.setattr(minimal_fusion_dcnn, "_predict_residual_model", _zero_residual_predict)
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file()}

    result = minimal_fusion_dcnn.benchmark_minimal_fusion_dcnn(config, random_simulations=5)
    after = {path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file()}
    created = after - before

    assert created == {
        (config.path("artifacts") / "fusion_minimal_dcnn" / "metrics.csv").relative_to(tmp_path),
        (config.path("artifacts") / "fusion_minimal_dcnn" / "oof_predictions.csv").relative_to(tmp_path),
        (config.path("reports") / "minimal_multimodal_fusion_dcnn_zh.md").relative_to(tmp_path),
    }
    assert result["n_labels"] == 135
    assert result["n_combinations"] == 15
    assert result["n_ablations"] == 28
    assert build_parser().parse_args(["benchmark-minimal-fusion-dcnn"]).command == "benchmark-minimal-fusion-dcnn"

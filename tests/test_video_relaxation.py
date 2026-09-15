from __future__ import annotations

from copy import deepcopy

import joblib
import numpy as np
import pandas as pd
import pytest

from real_time_ml.config import ProjectConfig, load_config
from real_time_ml.modeling.video_dcnn import (
    RELAXATION_VIDEO_MODEL_KIND,
    VIDEO_ENCODER_ABLATION_DUAL_KIND,
    VIDEO_ENCODER_MASKED_MEAN_MLP,
    load_video_dcnn_model,
    load_video_relaxation_dcnn_model,
    predict_video_dcnn_state,
    train_video_dcnn_model,
)
from real_time_ml.modeling.video_ridge import (
    RELAXATION_VIDEO_RIDGE_KIND,
    VIDEO_RIDGE_KIND,
    train_visual_ridge,
)
from real_time_ml.realtime.engine import InferenceEngine


def _config(tmp_path) -> ProjectConfig:
    data = deepcopy(load_config().data)
    data["paths"] = {key: str(tmp_path / key) for key in data["paths"]}
    data["modeling"]["dcnn"].update({
        "device": "cpu", "max_epochs": 1, "early_stopping_patience": 1, "batch_size": 64,
    })
    data["features"]["video"]["videomae2"]["pca_components"] = 2
    config = ProjectConfig(source=tmp_path / "project.yaml", data=data)
    config.ensure_artifact_dirs()
    return config


def _tables(tmp_path):
    rows, embeddings = [], []
    for participant_number, participant_offset in ((2, 0.0), (3, 10.0)):
        for condition_number in (1, 2):
            relaxation = 0.15 + 0.30 * condition_number + 0.01 * participant_number
            discomfort = 0.85 - 0.25 * condition_number + 0.01 * participant_number
            for window in range(4):
                rows.append({
                    "participant_id": f"P{participant_number:03d}",
                    "condition": f"C{condition_number}",
                    "presentation_position": condition_number,
                    "condition_window_index": window,
                    "condition_window_count": 4,
                    "relaxation": relaxation,
                    "discomfort": discomfort,
                    "relaxation_raw": relaxation * 10,
                    "intensity": 0.08 * condition_number,
                    "frequency": 0.12 * condition_number,
                    "eeg_alpha": participant_offset + condition_number + window,
                    "ecg_hr_bpm": 60.0 + participant_offset + window,
                    "head_speed_mean": 0.1 * condition_number + 0.01 * window,
                    "eye_fixation_fraction_ivt": 0.4 + 0.01 * window,
                    "video_brightness_mean": 0.2 * condition_number + 0.01 * window,
                })
                embeddings.append({
                    "participant_id": f"P{participant_number:03d}",
                    "condition": f"C{condition_number}",
                    "condition_window_index": window,
                    "video_available": 1.0,
                    "video_embedding_000": participant_offset + window,
                    "video_embedding_001": participant_offset + window + 1,
                    "video_embedding_002": participant_offset + window + 2,
                })
    window_source = tmp_path / "window_features.csv"
    embedding_source = tmp_path / "window_embeddings.csv"
    pd.DataFrame(rows).to_csv(window_source, index=False)
    pd.DataFrame(embeddings).to_csv(embedding_source, index=False)
    return rows, embeddings, window_source, embedding_source


def test_relaxation_ridge_matches_dual_target_relaxation_predictions_and_is_runtime_rejected(tmp_path):
    config = _config(tmp_path)
    _, _, source, _ = _tables(tmp_path)
    dual_models, dual_reports = tmp_path / "dual_models", tmp_path / "dual_reports"
    single_models, single_reports = tmp_path / "single_models", tmp_path / "single_reports"
    train_visual_ridge(
        config, source=source, condition_output=tmp_path / "dual_conditions.csv",
        models_dir=dual_models, reports_dir=dual_reports, include_video=True, expected_labels=4,
    )
    train_visual_ridge(
        config, source=source, condition_output=tmp_path / "single_conditions.csv",
        models_dir=single_models, reports_dir=single_reports, include_video=True, expected_labels=4,
        targets=("relaxation",), model_kind=RELAXATION_VIDEO_RIDGE_KIND, research_only=True,
    )
    dual = joblib.load(dual_models / "state_model_full.joblib")
    single = joblib.load(single_models / "state_model_full.joblib")
    dual_predictions = pd.read_csv(dual_reports / "condition_level_lopo_predictions.csv")
    single_predictions = pd.read_csv(single_reports / "condition_level_lopo_predictions.csv")

    assert dual["model_kind"] == VIDEO_RIDGE_KIND
    assert dual["targets"] == ["relaxation", "discomfort"]
    assert np.allclose(dual_predictions["pred_relaxation"], single_predictions["pred_relaxation"])
    assert single["model_kind"] == RELAXATION_VIDEO_RIDGE_KIND
    assert single["targets"] == ["relaxation"]
    assert single["research_only"] is True
    assert single["deployable"] is False
    assert "discomfort" not in single["metrics"]["targets"]
    assert not any("_raw" in column for column in single["feature_columns"])

    joblib.dump(single, config.path("models") / "state_model.joblib")
    with pytest.raises(ValueError, match="research-only"):
        InferenceEngine(config)


def test_relaxation_dcnn_checkpoint_has_one_output_and_runtime_loader_rejects_it(tmp_path):
    pytest.importorskip("torch")
    config = _config(tmp_path)
    rows, embeddings, window_source, embedding_source = _tables(tmp_path)
    model_path = tmp_path / "relaxation_dcnn.pt"
    train_video_dcnn_model(
        config, window_source=window_source, embedding_source=embedding_source, model_path=model_path,
        reports_dir=tmp_path / "reports", include_video=True, expected_labels=4,
        targets=("relaxation",), model_kind=RELAXATION_VIDEO_MODEL_KIND, research_only=True,
    )
    bundle = load_video_relaxation_dcnn_model(model_path, "cpu")
    prediction, widths, history = predict_video_dcnn_state(
        bundle, [{**rows[index], **embeddings[index]} for index in range(3)], "C1", config
    )

    assert bundle["targets"] == ["relaxation"]
    assert bundle["research_only"] is True
    assert bundle["deployable"] is False
    assert set(prediction) == {"relaxation"}
    assert set(widths) == {"relaxation"}
    assert history == 3
    assert "discomfort" not in bundle["metrics"]["targets"]
    with pytest.raises(ValueError, match="Not a videomae2_fusion_dcnn_condition_regressor_v1 checkpoint"):
        load_video_dcnn_model(model_path, "cpu")


def test_direct_video_mlp_ablation_keeps_dual_safety_metrics_but_is_research_only(tmp_path):
    torch = pytest.importorskip("torch")
    config = _config(tmp_path)
    _, _, window_source, embedding_source = _tables(tmp_path)
    model_path = tmp_path / "video_direct_mlp.pt"
    result = train_video_dcnn_model(
        config, window_source=window_source, embedding_source=embedding_source, model_path=model_path,
        reports_dir=tmp_path / "reports", include_video=True, expected_labels=4,
        model_kind=VIDEO_ENCODER_ABLATION_DUAL_KIND, research_only=True,
        video_encoder_mode=VIDEO_ENCODER_MASKED_MEAN_MLP, model_variant="video_direct_mlp",
    )
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    predictions = pd.read_csv(result["lopo_predictions_path"])

    assert checkpoint["video_encoder_mode"] == VIDEO_ENCODER_MASKED_MEAN_MLP
    assert checkpoint["research_only"] is True
    assert checkpoint["deployable"] is False
    assert checkpoint["parameter_count"] == result["parameter_count"]
    assert "risk_at_fold_tuned_threshold" in checkpoint["metrics"]["targets"]["discomfort"]
    assert set(predictions) >= {"participant_id", "condition", "pred_relaxation", "pred_discomfort"}
    assert len(predictions) == 4
    with pytest.raises(ValueError, match="Not a videomae2_fusion_dcnn_condition_regressor_v1 checkpoint"):
        load_video_dcnn_model(model_path, "cpu")

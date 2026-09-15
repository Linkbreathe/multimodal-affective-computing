from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from scripts.relax_foundation.analyze_relax_attention_video_results import _decision, _render_report, assert_prediction_alignment
from scripts.relax_foundation.run_relax_attention_video_experiments import build_attention_video_jobs
from src.data.relax_attention_video import (
    AttentionVideoConfig,
    RelaxAttentionVideoExtractor,
    apply_gaze_heatmap,
    project_gaze_to_frame,
)
from src.data.relax_foundation import DERIVED_MODALITIES, MODALITIES, RelaxHardFailure, assert_relax_modalities


def test_attention_video_is_allowed_but_not_default_relax_modality():
    assert "attention_video" not in MODALITIES
    assert "attention_video" in DERIVED_MODALITIES
    assert assert_relax_modalities(["ecg", "attention_video"]) == ["ecg", "attention_video"]

    with pytest.raises(RelaxHardFailure, match="PPG/Papagei"):
        assert_relax_modalities(["attention_video", "papagei_ppg"])


def test_project_gaze_to_frame_maps_center_and_rightward_gaze():
    config = AttentionVideoConfig(hfov_deg=90.0)
    center = project_gaze_to_frame(
        gaze_direction=np.array([0.0, 0.0, 1.0]),
        camera_forward=np.array([0.0, 0.0, 1.0]),
        camera_up=np.array([0.0, 1.0, 0.0]),
        width=1280,
        height=720,
        config=config,
    )
    right = project_gaze_to_frame(
        gaze_direction=np.array([0.2, 0.0, 1.0]),
        camera_forward=np.array([0.0, 0.0, 1.0]),
        camera_up=np.array([0.0, 1.0, 0.0]),
        width=1280,
        height=720,
        config=config,
    )

    assert center is not None
    assert right is not None
    assert abs(center[0] - 640.0) < 1.0
    assert abs(center[1] - 360.0) < 1.0
    assert right[0] > center[0]
    assert abs(right[1] - center[1]) < 1.0


def test_apply_gaze_heatmap_preserves_shape_and_brightens_gaze_center():
    frame = np.full((80, 120, 3), 200, dtype=np.uint8)

    out = apply_gaze_heatmap(frame, center_xy=(60.0, 40.0), config=AttentionVideoConfig())

    assert out.shape == frame.shape
    assert out.dtype == np.uint8
    assert out[40, 60].mean() > out[0, 0].mean()


def test_attention_video_extractor_builds_videomae_clip_from_gaze_and_frames(tmp_path):
    frame_paths = []
    frame_rows = []
    eye_rows = []
    for idx in range(16):
        path = tmp_path / f"frame_{idx:03d}.jpg"
        image = np.full((180, 320, 3), 120, dtype=np.uint8)
        cv2.circle(image, (160, 90), 12, (255, 255, 255), thickness=-1)
        cv2.imwrite(str(path), image)
        timestamp = 1_000_000 + idx * 100
        frame_paths.append(path)
        frame_rows.append(
            {
                "unix_time_ms": timestamp,
                "path": path,
                "width": 320,
                "height": 180,
                "camera_forward_x": 0.0,
                "camera_forward_y": 0.0,
                "camera_forward_z": 1.0,
                "camera_up_x": 0.0,
                "camera_up_y": 1.0,
                "camera_up_z": 0.0,
            }
        )
        eye_rows.append(
            {
                "unix_time_ms": timestamp + 20,
                "gaze_available": True,
                "gaze_direction_x": 0.0,
                "gaze_direction_y": 0.0,
                "gaze_direction_z": 1.0,
            }
        )

    clip, valid, metadata = RelaxAttentionVideoExtractor(AttentionVideoConfig()).build_clip(
        frame_rows=frame_rows,
        eye_rows=pd.DataFrame(eye_rows),
    )

    assert valid is True
    assert clip.shape == (3, 16, 224, 224)
    assert metadata["valid_projected_frames"] == 16


def test_attention_video_extractor_rejects_too_few_projected_frames(tmp_path):
    frame_rows = []
    for idx in range(16):
        path = tmp_path / f"frame_{idx:03d}.jpg"
        cv2.imwrite(str(path), np.full((64, 64, 3), 80, dtype=np.uint8))
        frame_rows.append(
            {
                "unix_time_ms": 1_000_000 + idx * 100,
                "path": path,
                "width": 64,
                "height": 64,
                "camera_forward_x": 0.0,
                "camera_forward_y": 0.0,
                "camera_forward_z": 1.0,
                "camera_up_x": 0.0,
                "camera_up_y": 1.0,
                "camera_up_z": 0.0,
            }
        )

    clip, valid, metadata = RelaxAttentionVideoExtractor(AttentionVideoConfig()).build_clip(
        frame_rows=frame_rows,
        eye_rows=pd.DataFrame(
            [
                {
                    "unix_time_ms": 1_000_000,
                    "gaze_available": False,
                    "gaze_direction_x": np.nan,
                    "gaze_direction_y": np.nan,
                    "gaze_direction_z": np.nan,
                }
            ]
        ),
    )

    assert valid is False
    assert clip.shape == (3, 16, 224, 224)
    assert metadata["reason"] == "insufficient_projected_gaze_frames"


def test_attention_video_job_matrix_has_required_variants_and_unique_outputs(tmp_path):
    args = Namespace(
        embedding_cache="artifacts/relax/condition_embeddings_attention_video.pt",
        cohorts="artifacts/relax/cohorts.json",
        handcrafted_features="artifacts/relax_model/runs/relax-foundation-reve-large-20260705/features/condition_features.csv",
        output_dir=str(tmp_path),
        seeds=[20260705, 20260706, 20260707],
        fusions=["early", "healnet", "mm_lego"],
        device="cpu",
        max_epochs=3,
        patience=1,
        batch_size=4,
        d_common=16,
        fusion_depth=1,
        fusion_heads=2,
        dim_head=8,
        latent_dim=8,
        latent_channels=4,
        smoke=True,
    )

    jobs = build_attention_video_jobs(args)
    output_dirs = [job.output_dir for job in jobs]
    variants = {job.variant for job in jobs if job.baseline == "none"}

    assert len(jobs) == 96
    assert len(output_dirs) == len(set(output_dirs))
    assert variants == {"full_eye_video", "attention_replacement", "no_visual", "visual_pair_only", "attention_only"}
    assert any(job.baseline == "random_9_condition" for job in jobs)
    assert any(job.baseline == "relax_handcrafted_ridge_cv" for job in jobs)


def test_attention_video_analyzer_rejects_mismatched_prediction_rows():
    left = pd.DataFrame(
        {
            "participant_id": ["P001"],
            "condition": ["C1"],
            "relaxation_true": [0.1],
            "discomfort_true": [0.2],
            "relaxation_pred": [0.1],
            "discomfort_pred": [0.2],
        }
    )
    right = left.copy()
    right.loc[0, "participant_id"] = "P002"

    with pytest.raises(RelaxHardFailure, match="Prediction rows do not align"):
        assert_prediction_alignment(left, right)


def test_attention_video_report_handles_metric_only_aggregate(tmp_path):
    tables = {
        "attention_video_variant_metrics": pd.DataFrame(
            [
                {
                    "cohort": "eeg_eligible",
                    "fusion": "early",
                    "seed": 20260705,
                    "variant": "full_eye_video",
                    "macro_mae": 0.5,
                    "relaxation_mae": 0.4,
                    "discomfort_mae": 0.6,
                }
            ]
        ),
        "attention_video_replacement_pairs": pd.DataFrame(),
        "attention_video_aggregate": pd.DataFrame(
            [
                {
                    "summary_type": "metric",
                    "cohort": "eeg_eligible",
                    "fusion": "early",
                    "comparison": "full_eye_video",
                    "macro_mae_mean": 0.5,
                    "n_seeds": 1,
                }
            ]
        ),
    }

    report = _render_report(tables, tmp_path, decision="smoke_only")

    assert "Relax Attention-Video Replacement Report" in report
    assert "full_eye_video" in report


def test_attention_video_decision_distinguishes_eeg_eligible_only_support():
    aggregate = pd.DataFrame(
        [
            {
                "summary_type": "replacement_pair",
                "cohort": "all_135",
                "fusion": "mm_lego",
                "comparison": "attention_replacement_minus_full_eye_video",
                "mean_delta": 0.011,
                "min_delta": 0.004,
                "max_delta": 0.015,
                "n_seeds": 3,
            },
            {
                "summary_type": "replacement_pair",
                "cohort": "all_135",
                "fusion": "mm_lego",
                "comparison": "attention_replacement_minus_no_visual",
                "mean_delta": 0.0004,
                "min_delta": -0.008,
                "max_delta": 0.011,
                "n_seeds": 3,
            },
            {
                "summary_type": "replacement_pair",
                "cohort": "eeg_eligible",
                "fusion": "early",
                "comparison": "attention_replacement_minus_full_eye_video",
                "mean_delta": -0.004,
                "min_delta": -0.009,
                "max_delta": -0.001,
                "n_seeds": 3,
            },
            {
                "summary_type": "replacement_pair",
                "cohort": "eeg_eligible",
                "fusion": "early",
                "comparison": "attention_replacement_minus_no_visual",
                "mean_delta": -0.006,
                "min_delta": -0.007,
                "max_delta": -0.003,
                "n_seeds": 3,
            },
        ]
    )

    assert _decision(aggregate) == "partially_supported_on_eeg_eligible_not_all135"

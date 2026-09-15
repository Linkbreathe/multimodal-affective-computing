from __future__ import annotations

from argparse import Namespace

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.relax_foundation.run_relax_foundation_probe import _run_handcrafted_baseline_cv
from scripts.relax_foundation.run_relax_claim_validation import build_claim_validation_jobs
from scripts.relax_foundation.analyze_relax_claim_validation import (
    assert_prediction_alignment,
    participant_cluster_stats,
)
from mac.data.relax_foundation_dataset import RelaxConditionEmbeddingDataset, relax_condition_collate
from mac.data.relax_foundation import RelaxHardFailure
from mac.tasks.relax_condition_control import FoldConditionResidualizer


def _write_embedding_cache(path):
    torch.save(
        {
            "samples": [
                {
                    "participant_id": "P001",
                    "condition": "C1",
                    "condition_index": 0,
                    "labels": {"relaxation": 0.1, "discomfort": 0.2},
                    "embeddings": {"video": torch.tensor([[10.0, 0.0], [12.0, 0.0]])},
                    "masks": {"video": torch.tensor([True, True])},
                },
                {
                    "participant_id": "P002",
                    "condition": "C2",
                    "condition_index": 1,
                    "labels": {"relaxation": 0.3, "discomfort": 0.4},
                    "embeddings": {"video": torch.tensor([[20.0, 5.0], [22.0, 7.0]])},
                    "masks": {"video": torch.tensor([True, True])},
                },
                {
                    "participant_id": "P003",
                    "condition": "C1",
                    "condition_index": 0,
                    "labels": {"relaxation": 0.5, "discomfort": 0.6},
                    "embeddings": {"video": torch.tensor([[30.0, 2.0], [999.0, 999.0]])},
                    "masks": {"video": torch.tensor([True, False])},
                },
            ],
            "embedding_dims": {"video": 2},
        },
        path,
    )


def test_video_condition_residualizer_uses_train_fold_means_only(tmp_path):
    cache = tmp_path / "condition_embeddings.pt"
    _write_embedding_cache(cache)
    dataset = RelaxConditionEmbeddingDataset(cache, modalities=["video"])

    residualizer = FoldConditionResidualizer.fit(dataset, train_indices=[0, 1], modality="video")
    batch = relax_condition_collate([dataset[2]])
    transformed = residualizer.transform_batch(batch)

    assert transformed["video_mask"].tolist() == [[True, False]]
    assert torch.allclose(transformed["video_emb"][0, 0], torch.tensor([19.0, 2.0]))
    assert torch.allclose(transformed["video_emb"][0, 1], torch.zeros(2))


def test_video_condition_residualizer_rejects_unseen_condition(tmp_path):
    cache = tmp_path / "condition_embeddings.pt"
    _write_embedding_cache(cache)
    dataset = RelaxConditionEmbeddingDataset(cache, modalities=["video"])

    residualizer = FoldConditionResidualizer.fit(dataset, train_indices=[0], modality="video")
    batch = relax_condition_collate([dataset[1]])

    with pytest.raises(RelaxHardFailure, match="no train-fold mean"):
        residualizer.transform_batch(batch)


def test_handcrafted_ridge_cv_returns_finite_predictions_and_alpha(tmp_path):
    features = tmp_path / "condition_features.csv"
    pd.DataFrame(
        {
            "participant_id": ["P001", "P002", "P003", "P004"],
            "condition": ["C1", "C1", "C1", "C1"],
            "relaxation": [0.1, 0.2, 0.3, 0.4],
            "discomfort": [0.2, 0.3, 0.4, 0.5],
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "feature_b": [np.nan, 2.0, np.nan, 4.0],
        }
    ).to_csv(features, index=False)
    split = {
        "train_participants": ["P001", "P002"],
        "val_participant": "P003",
        "test_participant": "P004",
    }

    pred, pred_frame, truth, metadata = _run_handcrafted_baseline_cv(
        features_path=features,
        split=split,
        alphas=(0.1, 1.0, 10.0),
    )

    assert pred.shape == truth.shape == (1, 2)
    assert pred_frame[["participant_id", "condition"]].to_dict("records") == [
        {"participant_id": "P004", "condition": "C1"}
    ]
    assert np.isfinite(pred).all()
    assert metadata["selected_alpha"] in {0.1, 1.0, 10.0}


def test_assert_prediction_alignment_rejects_mismatched_condition_rows():
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
    right.loc[0, "condition"] = "C2"

    with pytest.raises(RelaxHardFailure, match="Prediction rows do not align"):
        assert_prediction_alignment(left, right)


def test_participant_cluster_stats_preserves_direction_for_negative_deltas():
    stats = participant_cluster_stats(
        pd.DataFrame(
            {
                "participant_id": ["P001", "P002", "P003", "P004"],
                "delta": [-1.0, -2.0, -3.0, -4.0],
            }
        ),
        value_col="delta",
        rng_seed=7,
        bootstrap_samples=2000,
    )

    assert stats["mean_delta"] < 0
    assert stats["ci_high"] < 0
    assert 0 <= stats["p_signflip_two_sided"] <= 1


def test_claim_validation_job_matrix_has_required_pairs_and_unique_outputs(tmp_path):
    args = Namespace(
        embedding_cache="artifacts/relax/condition_embeddings.pt",
        cohorts="artifacts/relax/cohorts.json",
        handcrafted_features="artifacts/relax_model/runs/20260704T040215Z/features/condition_features.csv",
        output_dir=str(tmp_path),
        seeds=[20260705, 20260706, 20260707],
        fusions=["early", "mid", "late", "qformer", "healnet", "mm_lego"],
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

    jobs = build_claim_validation_jobs(args)
    output_dirs = [job.output_dir for job in jobs]

    assert len(output_dirs) == len(set(output_dirs))
    assert any(job.phase == "seed_stability" and job.fusion == "healnet" for job in jobs)
    assert any(job.phase == "eeg_paired" and "eeg" not in job.modalities for job in jobs)
    assert any(job.phase == "video_condition_control" and job.condition_control == "video_condition_residualized" for job in jobs)
    assert any(job.baseline == "relax_handcrafted_ridge_cv" for job in jobs)

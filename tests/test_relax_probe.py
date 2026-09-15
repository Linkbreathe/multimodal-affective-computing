from __future__ import annotations

import torch

from mac.data.relax_foundation_dataset import RelaxConditionEmbeddingDataset, relax_condition_collate
from mac.models.head_motion_1dcnn import HeadMotion1DCNN
from mac.tasks.relaxation import RelaxFusionRegressor, create_relax_fusion


def test_relax_condition_collate_pads_embedding_and_raw_head_sequences():
    samples = [
        {
            "participant_id": "P003",
            "condition": "C1",
            "condition_index": 0,
            "labels": {"relaxation": 0.1, "discomfort": 0.2},
            "embeddings": {
                "ecg": torch.ones(2, 4),
                "head": torch.ones(2, 10, 20),
            },
            "masks": {
                "ecg": torch.tensor([True, True]),
                "head": torch.tensor([True, False]),
            },
        },
        {
            "participant_id": "P004",
            "condition": "C2",
            "condition_index": 1,
            "labels": {"relaxation": 0.3, "discomfort": 0.4},
            "embeddings": {
                "ecg": torch.ones(1, 4) * 2,
                "head": torch.ones(1, 10, 20) * 2,
            },
            "masks": {
                "ecg": torch.tensor([True]),
                "head": torch.tensor([True]),
            },
        },
    ]

    batch = relax_condition_collate(samples)

    assert batch["ecg_emb"].shape == (2, 2, 4)
    assert batch["head_emb"].shape == (2, 2, 10, 20)
    assert batch["ecg_mask"].tolist() == [[True, True], [True, False]]
    assert batch["head_mask"].tolist() == [[True, False], [True, False]]
    assert torch.allclose(batch["targets"], torch.tensor([[0.1, 0.2], [0.3, 0.4]]))


def test_dataset_filters_by_locked_cohort(tmp_path):
    path = tmp_path / "condition_embeddings.pt"
    torch.save(
        {
            "samples": [
                {
                    "participant_id": "P003",
                    "condition": "C1",
                    "condition_index": 0,
                    "labels": {"relaxation": 0.1, "discomfort": 0.2},
                    "embeddings": {"ecg": torch.ones(1, 4)},
                    "masks": {"ecg": torch.ones(1, dtype=torch.bool)},
                },
                {
                    "participant_id": "P004",
                    "condition": "C1",
                    "condition_index": 0,
                    "labels": {"relaxation": 0.3, "discomfort": 0.4},
                    "embeddings": {"ecg": torch.ones(1, 4)},
                    "masks": {"ecg": torch.ones(1, dtype=torch.bool)},
                },
            ],
            "embedding_dims": {"ecg": 4},
        },
        path,
    )

    dataset = RelaxConditionEmbeddingDataset(path, modalities=["ecg"], participants=["P004"])

    assert len(dataset) == 1
    assert dataset[0]["participant_id"] == "P004"


def test_head_motion_1dcnn_encodes_raw_window():
    model = HeadMotion1DCNN(input_channels=10, embedding_dim=16)
    x = torch.randn(3, 10, 50)

    out = model(x)

    assert out.shape == (3, 16)


def test_create_relax_fusion_supports_mm_lego_with_five_modalities_and_masks():
    modalities = ["ecg", "eeg", "eye", "head", "video"]
    fusion = create_relax_fusion(
        "mm_lego",
        modalities=modalities,
        d_common=16,
        latent_dim=8,
        latent_channels=4,
        depth=1,
        heads=2,
        dim_head=4,
    )
    embeddings = [torch.randn(2, 3 + i, 16) for i in range(5)]
    masks = [torch.ones(2, 3 + i, dtype=torch.bool) for i in range(5)]
    masks[1][0] = False

    out = fusion(embeddings, modalities, masks=masks)

    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()


def test_relax_fusion_regressor_encodes_head_raw_and_missing_modalities_without_nan():
    modalities = ["ecg", "head", "eeg"]
    model = RelaxFusionRegressor(
        embed_dims={"ecg": 4, "head": (10, 20), "eeg": 6},
        modalities=modalities,
        fusion_name="mm_lego",
        d_common=16,
        fusion_kwargs={"latent_dim": 8, "latent_channels": 4, "depth": 1, "heads": 2, "dim_head": 4},
    )
    batch = {
        "ecg_emb": torch.randn(2, 2, 4),
        "ecg_mask": torch.tensor([[True, True], [True, False]]),
        "head_emb": torch.randn(2, 2, 10, 20),
        "head_mask": torch.tensor([[True, False], [True, True]]),
        "eeg_emb": torch.randn(2, 1, 6),
        "eeg_mask": torch.tensor([[False], [True]]),
    }

    pred = model(batch)

    assert pred.shape == (2, 2)
    assert torch.isfinite(pred).all()

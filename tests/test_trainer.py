import pytest
import torch
import torch.nn as nn
from scripts.run_experiment import ProjectedFusion
from mac.training.early_stopping import EarlyStopping
from mac.training.fusion_trainer import FusionTrainer
from mac.fusion.base import BaseFusionModule
from mac.fusion.projector import ModalityProjector
from mac.tasks.heads import MultiTaskHead

def test_early_stopping_improves():
    es = EarlyStopping(patience=3, mode="max")
    assert not es.step(0.5)
    assert not es.step(0.6)
    assert not es.step(0.7)
    assert es.best_score == 0.7

def test_early_stopping_triggers():
    es = EarlyStopping(patience=3, mode="max")
    es.step(0.5)
    es.step(0.4)
    assert not es.step(0.3)
    assert es.step(0.3)

def test_early_stopping_min_mode():
    es = EarlyStopping(patience=3, mode="min")
    es.step(0.5)
    es.step(0.4)
    es.step(0.5)
    assert not es.step(0.6)
    assert es.step(0.7)


class DummyFusion(BaseFusionModule):
    def __init__(self):
        super().__init__(d_common=32, d_out=32)
        self.fc = nn.Linear(32, 32)

    def forward(self, embeddings, modality_ids, masks=None):
        stacked = torch.stack(embeddings, dim=0).mean(dim=0)
        return self.fc(stacked)


class RecordingSequenceFusion(BaseFusionModule):
    supports_sequence_input = True

    def __init__(self):
        super().__init__(d_common=8, d_out=8)
        self.seen_shapes = None
        self.seen_masks = None

    def forward(self, embeddings, modality_ids, masks=None):
        self.seen_shapes = [tuple(emb.shape) for emb in embeddings]
        self.seen_masks = [mask.clone() if mask is not None else None for mask in masks]
        return embeddings[0].mean(dim=1)


def test_trainer_single_fold():
    fusion = DummyFusion()
    head = MultiTaskHead(d_fused=32, num_emotions=9)
    trainer = FusionTrainer(
        fusion_model=fusion,
        task_head=head,
        config={
            "training": {
                "batch_size": 4,
                "lr": 1e-3,
                "max_epochs": 2,
                "patience": 5,
                "num_workers": 0,
            },
            "loss_weights": {"ce": 1.0, "kl": 1.0, "vad": 1.0},
            "seed": 42,
        },
    )

    train_data = [
        {
            "embeddings": [torch.randn(32), torch.randn(32)],
            "modality_ids": ["video", "ppg"],
            "labels": {
                "emotion_label": torch.tensor(i % 9),
                "soft_label": torch.softmax(torch.randn(9), dim=0),
                "vad": torch.randn(3),
            },
        }
        for i in range(16)
    ]
    val_data = train_data[:4]

    metrics = trainer.train_fold(train_data, val_data, fold_name="test_fold")
    assert "weighted_f1" in metrics
    assert "ccc" in metrics
    assert "loss" in metrics
    assert 0.0 <= metrics["weighted_f1"] <= 1.0


def test_variable_length_batches_reach_sequence_aware_fusion_with_masks():
    fusion = RecordingSequenceFusion()
    trainer = FusionTrainer(
        fusion_model=fusion,
        task_head=MultiTaskHead(d_fused=8, num_emotions=9),
        config={
            "training": {
                "batch_size": 2,
                "lr": 1e-3,
                "max_epochs": 1,
                "patience": 1,
                "num_workers": 0,
            },
            "loss_weights": {"ce": 1.0, "kl": 1.0, "vad": 1.0},
            "seed": 42,
        },
        device="cpu",
    )
    projector = ModalityProjector(
        embed_dims={"video": 5, "eye_tracking": 7},
        d_common=8,
    )
    wrapped = ProjectedFusion(projector, fusion, ["video", "eye_tracking"])

    data = [
        {
            "embeddings": [torch.randn(3, 5), torch.randn(2, 7)],
            "modality_ids": ["video", "eye_tracking"],
            "labels": {
                "emotion_label": torch.tensor(0),
                "soft_label": torch.softmax(torch.randn(9), dim=0),
                "vad": torch.randn(3),
            },
        },
        {
            "embeddings": [torch.randn(1, 5), torch.randn(4, 7)],
            "modality_ids": ["video", "eye_tracking"],
            "labels": {
                "emotion_label": torch.tensor(1),
                "soft_label": torch.softmax(torch.randn(9), dim=0),
                "vad": torch.randn(3),
            },
        },
    ]

    loader = trainer._make_loader(data, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    embeddings, modality_ids, _, masks = trainer._unpack_batch(batch)
    out = wrapped(embeddings, modality_ids, masks)

    assert out.shape == (2, 8)
    assert fusion.seen_shapes == [(2, 3, 8), (2, 4, 8)]
    assert torch.equal(
        fusion.seen_masks[0],
        torch.tensor([[True, True, True], [True, False, False]]),
    )
    assert torch.equal(
        fusion.seen_masks[1],
        torch.tensor([[True, True, False, False], [True, True, True, True]]),
    )


def _sample(label: int = 0) -> dict:
    return {
        "embeddings": [torch.randn(32), torch.randn(32)],
        "modality_ids": ["video", "ppg"],
        "labels": {
            "emotion_label": torch.tensor(label),
            "soft_label": torch.softmax(torch.randn(9), dim=0),
            "vad": torch.randn(3),
        },
    }


def test_run_loso_rejects_duplicate_subject_ids():
    trainer = FusionTrainer(
        fusion_model=DummyFusion(),
        task_head=MultiTaskHead(d_fused=32, num_emotions=9),
        config={
            "training": {"batch_size": 2, "lr": 1e-3, "max_epochs": 1, "patience": 1},
            "loss_weights": {"ce": 1.0, "kl": 1.0, "vad": 1.0},
            "seed": 42,
        },
        device="cpu",
    )

    all_data = {"001": [_sample()], "002": [_sample()]}
    with pytest.raises(ValueError, match="Duplicate subject_ids"):
        trainer.run_loso(all_data, ["001", "001", "002"])


def test_run_loso_records_disjoint_fold_metadata(monkeypatch):
    trainer = FusionTrainer(
        fusion_model=DummyFusion(),
        task_head=MultiTaskHead(d_fused=32, num_emotions=9),
        config={
            "training": {"batch_size": 2, "lr": 1e-3, "max_epochs": 1, "patience": 1},
            "loss_weights": {"ce": 1.0, "kl": 1.0, "vad": 1.0},
            "seed": 42,
        },
        device="cpu",
    )
    all_data = {
        "001": [_sample(0)],
        "002": [_sample(1)],
        "003": [_sample(2), _sample(3)],
        "004": [_sample(4)],
    }

    def fake_train_fold(train_data, val_data, fold_name="fold", tb_logger=None):
        trainer._last_fusion = trainer.fusion_model
        trainer._last_head = trainer.task_head
        return {"weighted_f1": 0.0, "ccc": 0.0, "loss": 0.0}

    def fake_evaluate(fusion_model, task_head, loader, loss_fn):
        return {"weighted_f1": 1.0, "ccc": 0.5, "loss": 0.25}

    monkeypatch.setattr(trainer, "train_fold", fake_train_fold)
    monkeypatch.setattr(trainer, "_evaluate", fake_evaluate)

    results = trainer.run_loso(all_data, ["001", "002", "003", "004"])

    assert results[0]["test_subject"] == "001"
    assert results[0]["val_subject"] == "002"
    assert results[0]["train_subjects"] == ["003", "004"]
    assert results[0]["n_train"] == 3
    assert results[0]["n_val"] == 1
    assert results[0]["n_test"] == 1
    assert "001" not in results[0]["train_subjects"]
    assert "002" not in results[0]["train_subjects"]

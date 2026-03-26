import pytest
import torch
import torch.nn as nn
from scripts.run_experiment import ProjectedFusion
from src.trainer.early_stopping import EarlyStopping
from src.trainer.fusion_trainer import FusionTrainer
from src.fusion.base import BaseFusionModule
from src.fusion.projector import ModalityProjector
from src.tasks.heads import MultiTaskHead

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

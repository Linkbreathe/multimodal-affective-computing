import pytest
import torch
import torch.nn as nn
from src.trainer.early_stopping import EarlyStopping
from src.trainer.fusion_trainer import FusionTrainer
from src.fusion.base import BaseFusionModule
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

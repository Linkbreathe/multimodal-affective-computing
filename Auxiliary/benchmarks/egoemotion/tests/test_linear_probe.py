import numpy as np
import torch
import pytest

from Auxiliary.benchmarks.egoemotion.scripts.run_linear_probe_10s import (
    train_probe_fold,
    evaluate_probe,
    load_video_embeddings,
)
import pandas as pd


def test_train_probe_fold_returns_linear():
    """train_probe_fold returns nn.Linear(768, 9) that produces finite outputs."""
    N_train, N_val = 200, 30
    train_embs = torch.randn(N_train, 768)
    train_labels = torch.randint(0, 9, (N_train,))
    val_embs = torch.randn(N_val, 768)
    val_labels = torch.randint(0, 9, (N_val,))

    probe = train_probe_fold(
        train_embs, train_labels, val_embs, val_labels,
        device="cpu", seed=42, max_epochs=5, patience=3,
    )

    assert isinstance(probe, torch.nn.Linear)
    assert probe.in_features == 768
    assert probe.out_features == 9

    with torch.no_grad():
        out = probe(torch.randn(4, 768))
    assert out.shape == (4, 9)
    assert torch.isfinite(out).all()


def test_evaluate_probe_returns_metrics():
    """evaluate_probe returns macro_f1 and weighted_f1 in [0, 1]."""
    probe = torch.nn.Linear(768, 9)
    embs = torch.randn(50, 768)
    labels = torch.randint(0, 9, (50,))

    metrics = evaluate_probe(probe, embs, labels, device="cpu")

    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert 0.0 <= metrics["weighted_f1"] <= 1.0


def test_load_video_embeddings_pools_clips(tmp_path):
    """load_video_embeddings mean-pools [T, 768] → [768]."""
    emb_dir = tmp_path / "video_mae_v2" / "005"
    emb_dir.mkdir(parents=True)

    raw = torch.randn(6, 768)
    torch.save({"embedding": raw, "label": 4, "segment_idx": 5, "subject": "005"}, emb_dir / "segment_0005.pt")

    manifest = pd.DataFrame([
        {"subject": 5, "segment_path": "005/ppg_segments/5.p", "label": 4},
    ])

    data = load_video_embeddings(str(tmp_path / "video_mae_v2"), manifest)

    assert "005" in data
    assert len(data["005"]) == 1
    sample = data["005"][0]
    assert sample["embedding"].shape == (768,)
    assert sample["label"] == 4
    assert torch.allclose(sample["embedding"], raw.mean(dim=0))

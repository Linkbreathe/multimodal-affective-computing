import pytest
import torch
from pathlib import Path
from mac.data.dataset import EmbeddingDataset

@pytest.fixture
def mock_embeddings(tmp_path):
    modalities = {"video_mae_v2": 768, "patchtst_eye": 128, "papagei_ppg": 768}
    for mod, dim in modalities.items():
        for subj in ["005", "006"]:
            for seg_idx in range(3):
                path = tmp_path / mod / subj / f"segment_{seg_idx:04d}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                n_tokens = 5 if mod != "papagei_ppg" else 1
                torch.save({
                    "embedding": torch.randn(n_tokens, dim),
                    "metadata": {"subject_id": subj, "task_name": f"task_{seg_idx}",
                                 "start_idx": seg_idx * 1000, "end_idx": (seg_idx + 1) * 1000},
                }, path)
    return tmp_path

def test_dataset_loads(mock_embeddings):
    ds = EmbeddingDataset(
        embeddings_dir=str(mock_embeddings),
        modalities=["video_mae_v2", "patchtst_eye", "papagei_ppg"],
        subject_ids=["005", "006"],
    )
    assert len(ds) == 6

def test_dataset_getitem(mock_embeddings):
    ds = EmbeddingDataset(
        embeddings_dir=str(mock_embeddings),
        modalities=["video_mae_v2", "patchtst_eye", "papagei_ppg"],
        subject_ids=["005"],
    )
    sample = ds[0]
    assert "embeddings" in sample
    assert "subject_id" in sample
    assert len(sample["embeddings"]) == 3

def test_dataset_with_labels(mock_embeddings):
    labels = {
        "005_0000": {
            "emotion_label": torch.tensor(2, dtype=torch.long),
            "soft_label": torch.softmax(torch.randn(9), dim=0),
            "vad": torch.randn(3),
        },
        "005_0001": {
            "emotion_label": torch.tensor(5, dtype=torch.long),
            "soft_label": torch.softmax(torch.randn(9), dim=0),
            "vad": torch.randn(3),
        },
    }
    ds = EmbeddingDataset(
        embeddings_dir=str(mock_embeddings),
        modalities=["video_mae_v2", "patchtst_eye", "papagei_ppg"],
        subject_ids=["005"],
        labels=labels,
    )
    sample = ds[0]
    assert "labels" in sample
    assert sample["labels"]["emotion_label"].item() == 2
    assert sample["labels"]["soft_label"].shape == (9,)
    assert sample["labels"]["vad"].shape == (3,)

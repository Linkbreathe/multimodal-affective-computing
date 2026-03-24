import pytest
import torch
from src.data.collate import collate_embeddings

def test_collate_pooled():
    batch = [
        {"embeddings": {"vid": torch.randn(5, 768), "ppg": torch.randn(1, 768)},
         "subject_id": "005", "segment_idx": 0},
        {"embeddings": {"vid": torch.randn(3, 768), "ppg": torch.randn(1, 768)},
         "subject_id": "006", "segment_idx": 1},
    ]
    out = collate_embeddings(batch, pool=True)
    assert out["vid_emb"].shape == (2, 768)
    assert out["ppg_emb"].shape == (2, 768)

def test_collate_padded():
    batch = [
        {"embeddings": {"vid": torch.randn(5, 768), "ppg": torch.randn(1, 768)},
         "subject_id": "005", "segment_idx": 0},
        {"embeddings": {"vid": torch.randn(3, 768), "ppg": torch.randn(1, 768)},
         "subject_id": "006", "segment_idx": 1},
    ]
    out = collate_embeddings(batch, pool=False)
    assert out["vid_emb"].shape == (2, 5, 768)
    assert out["vid_mask"].shape == (2, 5)
    assert out["vid_mask"][0].all()
    assert out["vid_mask"][1, :3].all()
    assert not out["vid_mask"][1, 3:].any()

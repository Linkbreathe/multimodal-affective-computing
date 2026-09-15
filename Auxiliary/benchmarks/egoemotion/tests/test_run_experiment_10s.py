from pathlib import Path

import pandas as pd
import torch

from Auxiliary.benchmarks.egoemotion.scripts.run_experiment_10s import load_10s_data_by_subject


def _manifest(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "subject": "005",
        "global_seq": 5,
        "task_name": "video_neutral",
        "chunk_idx_in_task": 0,
        "emotion_label": 4,
        "emotion_name": "Neutral",
        "soft_label": "0,0,0,0,1,0,0,0,0",
        "valence": 0.0,
        "arousal": 0.0,
        "dominance": 0.0,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def _save_embedding(root: Path, enc: str, seq: int, emb: torch.Tensor) -> None:
    path = root / enc / "005" / f"segment_{seq:04d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "embedding": emb,
        "subject": "005",
        "segment_idx": seq,
        "label": 4,
        "emotion": "Neutral",
    }, path)


def test_load_10s_data_preserves_singleton_sequences(tmp_path):
    embeddings_dir = tmp_path / "embeddings_10s"

    for enc, emb in {
        "video_mae_v2": torch.randn(1, 768),
        "patchtst_eye": torch.randn(1, 39, 128),
        "papagei_ppg": torch.randn(1, 512),
    }.items():
        _save_embedding(embeddings_dir, enc, 5, emb)

    manifest = _manifest([{"global_seq": 5}])

    loaded = load_10s_data_by_subject(
        str(embeddings_dir),
        ["video_mae_v2", "patchtst_eye", "papagei_ppg"],
        ["video", "eye_tracking", "ppg"],
        manifest,
    )

    sample = loaded["005"][0]
    assert [tuple(emb.shape) for emb in sample["embeddings"]] == [
        (1, 768),
        (39, 128),
        (512,),
    ]


def test_pool_clips_collapses_video_only(tmp_path):
    """--pool-clips mean-pools video [T, 768] -> [768], leaves eye_tracking untouched."""
    embeddings_dir = tmp_path / "embeddings_10s"

    num_clips = 6
    for enc, emb in {
        "video_mae_v2": torch.randn(num_clips, 768),   # [6, 768] sequence
        "patchtst_eye": torch.randn(39, 128),           # [39, 128] sequence
        "papagei_ppg": torch.randn(1, 512),             # [1, 512] pooled
    }.items():
        _save_embedding(embeddings_dir, enc, 5, emb)

    manifest = _manifest([{"global_seq": 5}])

    loaded = load_10s_data_by_subject(
        str(embeddings_dir),
        ["video_mae_v2", "patchtst_eye", "papagei_ppg"],
        ["video", "eye_tracking", "ppg"],
        manifest,
        pool_clips=True,
    )

    sample = loaded["005"][0]
    shapes = [tuple(emb.shape) for emb in sample["embeddings"]]

    # Video pooled: [6, 768] -> [768]
    assert shapes[0] == (768,), f"Video should be [768], got {shapes[0]}"
    # Eye tracking unchanged: [39, 128]
    assert shapes[1] == (39, 128), f"Eye should be [39, 128], got {shapes[1]}"
    # PPG unchanged: [512]
    assert shapes[2] == (512,), f"PPG should be [512], got {shapes[2]}"


def test_pooled_video_batches_without_masks(tmp_path):
    """Pooled video enters as [B, 768] with no sequence padding or masks."""
    from mac.fusion.base import BaseFusionModule

    class RecordingFusion(BaseFusionModule):
        """Records inputs for inspection."""
        supports_sequence_input = False

        def __init__(self):
            super().__init__(d_common=32)
            self.last_shapes = None
            self.last_masks = None
            self.fc = torch.nn.Linear(32, 32)

        def forward(self, embeddings, modality_ids, masks=None):
            self.last_shapes = [e.shape for e in embeddings]
            self.last_masks = masks
            return self.fc(embeddings[0])

    embeddings_dir = tmp_path / "embeddings_10s"

    # Create 3 samples with DIFFERENT clip counts (would trigger padding if 2D)
    for seg_idx, num_clips in [(0, 6), (1, 10), (2, 4)]:
        _save_embedding(embeddings_dir, "video_mae_v2", seg_idx, torch.randn(num_clips, 768))

    manifest = _manifest([
        {"global_seq": i, "chunk_idx_in_task": i}
        for i in range(3)
    ])

    loaded = load_10s_data_by_subject(
        str(embeddings_dir),
        ["video_mae_v2"],
        ["video"],
        manifest,
        pool_clips=True,
    )

    samples = loaded["005"]
    # All pooled to 1D
    for s in samples:
        assert s["embeddings"][0].dim() == 1, "Pooled video should be 1D"
        assert s["embeddings"][0].shape == (768,)

    # Batch them via the trainer's collation
    from mac.fusion.projector import ModalityProjector
    projector = ModalityProjector({"video": 768}, d_common=32)
    fusion = RecordingFusion()

    from Auxiliary.benchmarks.egoemotion.scripts.run_experiment import ProjectedFusion
    model = ProjectedFusion(projector, fusion, ["video"])

    # Simulate a batch by stacking
    batch_embs = torch.stack([s["embeddings"][0] for s in samples])  # [3, 768]
    assert batch_embs.dim() == 2, f"Expected [B, 768], got {batch_embs.shape}"
    assert batch_embs.shape == (3, 768)

    # Forward through model
    model([batch_embs], ["video"], [None])
    assert fusion.last_shapes[0] == (3, 32), f"Expected [B, 32], got {fusion.last_shapes[0]}"

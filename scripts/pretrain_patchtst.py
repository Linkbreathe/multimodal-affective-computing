"""Self-supervised PatchTST pre-training on all eye tracking data."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.encoders.patchtst import PatchTSTEncoder, PatchTSTPreTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


class EyeTrackingDataset(Dataset):
    def __init__(self, data_dir: str, chunk_len: int = 900) -> None:
        self.chunk_len = chunk_len
        self.segments: list[np.ndarray] = []
        data_path = Path(data_dir)
        for subj_dir in sorted(data_path.iterdir()):
            if not subj_dir.is_dir() or not subj_dir.name.isdigit():
                continue
            gaze_path = subj_dir / "gaze_90fps.npy"
            pupil_path = subj_dir / "pupils_90fps.npy"
            if not gaze_path.exists() or not pupil_path.exists():
                continue
            gaze = np.load(gaze_path)
            pupils = np.load(pupil_path)
            min_len = min(len(gaze), len(pupils))
            combined = np.concatenate([gaze[:min_len], pupils[:min_len]], axis=1)
            for start in range(0, min_len - chunk_len, chunk_len // 2):
                self.segments.append(combined[start : start + chunk_len])
        log.info(f"Loaded {len(self.segments)} chunks from {data_dir}")

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.segments[idx], dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/egoemotion_raw")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_path", default="checkpoints/patchtst_pretrained.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = EyeTrackingDataset(args.data_dir, chunk_len=900)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    encoder = PatchTSTEncoder(
        num_channels=4, patch_len=45, stride=22,
        d_model=128, n_heads=4, n_layers=3, seq_len=900,
    )
    trainer = PatchTSTPreTrainer(encoder, mask_ratio=0.4, lr=args.lr, device=device)

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch in loader:
            epoch_loss += trainer.train_step(batch)
        avg_loss = epoch_loss / len(loader)
        log.info(f"Epoch {epoch+1}/{args.epochs} — loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(encoder.state_dict(), args.save_path)
            log.info(f"Saved best model (loss={best_loss:.4f})")

    log.info(f"Pre-training complete. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()

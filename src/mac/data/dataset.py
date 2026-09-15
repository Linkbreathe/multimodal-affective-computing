"""PyTorch Dataset for cached encoder embeddings."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """Loads pre-extracted embeddings from disk."""

    def __init__(
        self,
        embeddings_dir: str,
        modalities: list[str],
        subject_ids: list[str],
        labels: dict[str, dict] | None = None,
    ) -> None:
        self.embeddings_dir = Path(embeddings_dir)
        self.modalities = modalities
        self.labels = labels or {}

        self.samples: list[tuple[str, int]] = []
        first_mod = modalities[0]
        for subj in subject_ids:
            mod_dir = self.embeddings_dir / first_mod / subj
            if not mod_dir.exists():
                continue
            seg_files = sorted(mod_dir.glob("segment_*.pt"))
            for f in seg_files:
                idx = int(f.stem.split("_")[1])
                all_exist = all(
                    (self.embeddings_dir / m / subj / f.name).exists()
                    for m in modalities
                )
                if all_exist:
                    self.samples.append((subj, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        subject_id, seg_idx = self.samples[index]
        embeddings = {}
        metadata = {}
        for mod in self.modalities:
            path = self.embeddings_dir / mod / subject_id / f"segment_{seg_idx:04d}.pt"
            data = torch.load(path, weights_only=False)
            embeddings[mod] = data["embedding"]
            metadata = data.get("metadata", metadata)

        sample = {
            "embeddings": embeddings,
            "subject_id": subject_id,
            "segment_idx": seg_idx,
            "metadata": metadata,
        }

        key = f"{subject_id}_{seg_idx:04d}"
        if key in self.labels:
            sample["labels"] = self.labels[key]

        return sample

"""Condition-level dataset for Relax foundation probe embeddings."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.data.relax_foundation import CONDITIONS, RelaxHardFailure


TARGET_COLUMNS: tuple[str, str] = ("relaxation", "discomfort")


class RelaxConditionEmbeddingDataset(Dataset):
    """Load one sample per participant-Condition from a strict embedding cache.

    Expected cache format:
        torch.save({
            "samples": [
                {
                    "participant_id": "P003",
                    "condition": "C1",
                    "condition_index": 0,
                    "labels": {"relaxation": 0.1, "discomfort": 0.2},
                    "embeddings": {"ecg": Tensor[W,D], "head": Tensor[W,C,T]},
                    "masks": {"ecg": BoolTensor[W], "head": BoolTensor[W]},
                },
            ],
            "embedding_dims": {"ecg": 1024, "head": [10, 500]},
        }, path)
    """

    def __init__(
        self,
        cache_path: str | Path,
        *,
        modalities: list[str],
        participants: list[str] | None = None,
        conditions: list[str] | None = None,
    ) -> None:
        self.cache_path = Path(cache_path)
        self.modalities = list(modalities)
        if not self.modalities:
            raise RelaxHardFailure("At least one modality is required")
        if not self.cache_path.exists():
            raise RelaxHardFailure(f"Relax embedding cache is missing: {self.cache_path}")

        payload = torch.load(self.cache_path, map_location="cpu", weights_only=False)
        samples = payload.get("samples")
        if not isinstance(samples, list):
            raise RelaxHardFailure("Relax embedding cache must contain a list under 'samples'")

        participant_set = set(participants) if participants is not None else None
        condition_set = set(conditions) if conditions is not None else None
        self.samples: list[dict[str, Any]] = []
        for sample in samples:
            participant = str(sample.get("participant_id"))
            condition = str(sample.get("condition"))
            if participant_set is not None and participant not in participant_set:
                continue
            if condition_set is not None and condition not in condition_set:
                continue
            self._validate_sample(sample)
            self.samples.append(sample)

        self.embedding_dims = payload.get("embedding_dims", {})

    def _validate_sample(self, sample: dict[str, Any]) -> None:
        participant = sample.get("participant_id")
        condition = sample.get("condition")
        if condition not in CONDITIONS:
            raise RelaxHardFailure(f"{participant}: unexpected Relax Condition {condition!r}")
        labels = sample.get("labels", {})
        missing_targets = [name for name in TARGET_COLUMNS if name not in labels]
        if missing_targets:
            raise RelaxHardFailure(f"{participant}/{condition}: missing targets {missing_targets}")
        embeddings = sample.get("embeddings", {})
        masks = sample.get("masks", {})
        for modality in self.modalities:
            if modality not in embeddings:
                raise RelaxHardFailure(f"{participant}/{condition}: missing {modality} embedding")
            emb = torch.as_tensor(embeddings[modality])
            if emb.ndim not in (2, 3):
                raise RelaxHardFailure(
                    f"{participant}/{condition}/{modality}: expected [W,D] or [W,C,T], got {tuple(emb.shape)}"
                )
            if modality not in masks:
                raise RelaxHardFailure(f"{participant}/{condition}: missing {modality} window mask")
            mask = torch.as_tensor(masks[modality], dtype=torch.bool)
            if mask.ndim != 1 or mask.shape[0] != emb.shape[0]:
                raise RelaxHardFailure(
                    f"{participant}/{condition}/{modality}: mask shape {tuple(mask.shape)} "
                    f"does not match window count {emb.shape[0]}"
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        out = {
            "participant_id": str(sample["participant_id"]),
            "condition": str(sample["condition"]),
            "condition_index": int(sample.get("condition_index", CONDITIONS.index(str(sample["condition"])))),
            "labels": {
                "relaxation": float(sample["labels"]["relaxation"]),
                "discomfort": float(sample["labels"]["discomfort"]),
            },
            "embeddings": {},
            "masks": {},
        }
        for modality in self.modalities:
            out["embeddings"][modality] = torch.as_tensor(sample["embeddings"][modality]).float()
            out["masks"][modality] = torch.as_tensor(sample["masks"][modality], dtype=torch.bool)
        return out


def relax_condition_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad within-Condition window sequences without changing labels."""
    if not batch:
        raise RelaxHardFailure("Cannot collate an empty Relax batch")
    modalities = list(batch[0]["embeddings"].keys())
    result: dict[str, Any] = {
        "participant_ids": [sample["participant_id"] for sample in batch],
        "conditions": [sample["condition"] for sample in batch],
        "condition_indices": torch.tensor([sample["condition_index"] for sample in batch], dtype=torch.long),
        "targets": torch.tensor(
            [[sample["labels"]["relaxation"], sample["labels"]["discomfort"]] for sample in batch],
            dtype=torch.float32,
        ),
    }

    for modality in modalities:
        seqs = [sample["embeddings"][modality].float() for sample in batch]
        masks = [sample["masks"][modality].bool() for sample in batch]
        result[f"{modality}_emb"] = pad_sequence(seqs, batch_first=True)
        result[f"{modality}_mask"] = pad_sequence(masks, batch_first=True, padding_value=False)

    return result

"""Condition-level dataset for the shared Relax alignment protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
from torch.utils.data import Dataset


KEYS = ("participant_id", "condition", "condition_window_index")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    text = str(value).strip().lower()
    if text in {"true", "1", "1.0", "yes"}:
        return True
    if text in {"false", "0", "0.0", "no", ""}:
        return False
    raise ValueError(f"Unrecognized boolean mask value: {value!r}")


class RelaxConditionEmbeddingDataset(Dataset):
    """Load one multimodal window sequence per participant-condition label.

    The cache contains an internal availability mask produced during extraction.
    When an external shared mask is supplied, the effective mask is the logical
    AND of both sources. This prevents either project from silently recovering
    windows excluded by the alignment contract.
    """

    def __init__(
        self,
        cache_path: str | Path,
        *,
        modalities: Iterable[str],
        participants: Iterable[str] | None = None,
        mask_manifest: str | Path | None = None,
        strict: bool = True,
    ) -> None:
        self.cache_path = Path(cache_path)
        if not self.cache_path.is_file():
            raise FileNotFoundError(f"Relax embedding cache not found: {self.cache_path}")
        cache = torch.load(self.cache_path, map_location="cpu", weights_only=False)
        required = {"participant_ids", "conditions", "presentation_positions", "targets", "embeddings", "masks"}
        missing = required - set(cache)
        if missing:
            raise ValueError(f"Relax embedding cache lacks fields: {sorted(missing)}")
        self.metadata = dict(cache.get("metadata", {}))
        self.modalities = tuple(str(modality) for modality in modalities)
        unknown = sorted(set(self.modalities) - set(cache["embeddings"]))
        if unknown:
            raise ValueError(f"Cache lacks requested modalities: {unknown}")

        all_participants = [str(value) for value in cache["participant_ids"]]
        selected = set(str(value) for value in participants) if participants is not None else set(all_participants)
        self.indices = [index for index, participant in enumerate(all_participants) if participant in selected]
        if not self.indices:
            raise ValueError("No cache samples remain after participant filtering")
        self.participant_ids = [all_participants[index] for index in self.indices]
        self.conditions = [str(cache["conditions"][index]) for index in self.indices]
        self.presentation_positions = torch.as_tensor(cache["presentation_positions"], dtype=torch.float32)[self.indices]
        self.targets = torch.as_tensor(cache["targets"], dtype=torch.float32)[self.indices]
        self.embeddings = {
            modality: torch.as_tensor(cache["embeddings"][modality], dtype=torch.float32)[self.indices]
            for modality in self.modalities
        }
        self.masks = {
            modality: torch.as_tensor(cache["masks"][modality], dtype=torch.bool)[self.indices].clone()
            for modality in self.modalities
        }
        n = len(self.indices)
        if self.targets.shape != (n, 2):
            raise ValueError(f"Expected two regression targets, got {tuple(self.targets.shape)}")
        for modality in self.modalities:
            if self.embeddings[modality].dim() != 3:
                raise ValueError(f"{modality} embeddings must be [condition, window, dimension]")
            if self.masks[modality].shape != self.embeddings[modality].shape[:2]:
                raise ValueError(f"{modality} mask shape does not match its embeddings")

        self.external_mask_path = Path(mask_manifest) if mask_manifest else None
        if self.external_mask_path is not None:
            self._apply_external_masks(self.external_mask_path, strict=strict)

    def _apply_external_masks(self, path: Path, *, strict: bool) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Shared modality mask not found: {path}")
        frame = pd.read_csv(path)
        missing = set(KEYS) - set(frame.columns)
        if missing:
            raise ValueError(f"Shared mask lacks keys: {sorted(missing)}")
        if frame.duplicated(list(KEYS)).any():
            raise ValueError("Shared modality mask contains duplicate window keys")
        lookup = {
            (str(row.participant_id), str(row.condition), int(row.condition_window_index)): row
            for row in frame.itertuples(index=False)
        }
        used: set[tuple[str, str, int]] = set()
        for sample_index, (participant, condition) in enumerate(zip(self.participant_ids, self.conditions, strict=True)):
            sequence_length = next(iter(self.masks.values())).shape[1]
            for window_index in range(sequence_length):
                key = (participant, condition, window_index)
                row = lookup.get(key)
                if row is None:
                    for modality in self.modalities:
                        self.masks[modality][sample_index, window_index] = False
                    continue
                used.add(key)
                for modality in self.modalities:
                    column = f"{modality}_valid"
                    if not hasattr(row, column):
                        raise ValueError(f"Shared mask lacks {column}")
                    external = _as_bool(getattr(row, column))
                    self.masks[modality][sample_index, window_index] &= external
        if strict:
            expected = {
                (participant, condition, window_index)
                for participant, condition in zip(self.participant_ids, self.conditions, strict=True)
                for window_index in range(next(iter(self.masks.values())).shape[1])
                if (participant, condition, window_index) in lookup
            }
            if used != expected:
                raise ValueError("Shared modality mask coverage is inconsistent with the cache")

    @property
    def embed_dims(self) -> dict[str, int]:
        return {modality: int(values.shape[-1]) for modality, values in self.embeddings.items()}

    def condition_keys(self) -> list[tuple[str, str]]:
        return list(zip(self.participant_ids, self.conditions, strict=True))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "participant_id": self.participant_ids[index],
            "condition": self.conditions[index],
            "presentation_position": self.presentation_positions[index],
            "targets": self.targets[index],
            "embeddings": {modality: values[index] for modality, values in self.embeddings.items()},
            "masks": {modality: values[index] for modality, values in self.masks.items()},
        }


__all__ = ["RelaxConditionEmbeddingDataset"]

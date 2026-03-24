"""Variable-length collation for multi-modal embeddings."""
from __future__ import annotations

from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_embeddings(
    batch: list[dict[str, Any]], pool: bool = False
) -> dict[str, Any]:
    modalities = list(batch[0]["embeddings"].keys())
    result: dict[str, Any] = {
        "subject_ids": [s["subject_id"] for s in batch],
        "segment_indices": [s["segment_idx"] for s in batch],
    }

    if pool:
        for mod in modalities:
            tensors = [s["embeddings"][mod].mean(dim=0) for s in batch]
            result[f"{mod}_emb"] = torch.stack(tensors)
    else:
        for mod in modalities:
            seqs = [s["embeddings"][mod] for s in batch]
            padded = pad_sequence(seqs, batch_first=True)
            lengths = torch.tensor([s.shape[0] for s in seqs])
            mask = torch.arange(padded.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
            result[f"{mod}_emb"] = padded
            result[f"{mod}_mask"] = mask

    if "labels" in batch[0] and batch[0]["labels"] is not None:
        label_keys = batch[0]["labels"].keys()
        for key in label_keys:
            vals = [s["labels"][key] for s in batch]
            if isinstance(vals[0], (int, float)):
                result[key] = torch.tensor(vals)
            elif isinstance(vals[0], torch.Tensor):
                result[key] = torch.stack(vals)

    return result

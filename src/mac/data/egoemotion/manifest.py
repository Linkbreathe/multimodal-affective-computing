"""Checked loader for task-aware egoEMOTION 10s embeddings."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch


REQUIRED_10S_COLUMNS = {
    "subject",
    "global_seq",
    "task_name",
    "chunk_idx_in_task",
    "emotion_label",
    "emotion_name",
    "soft_label",
    "valence",
    "arousal",
    "dominance",
}


class Ego10sManifestError(ValueError):
    """Raised when the egoEMOTION 10s manifest and embeddings disagree."""


@dataclass
class Ego10sLoadReport:
    """Summary of what the checked 10s loader accepted and dropped."""

    manifest_rows: int
    loaded_rows: int
    dropped_rows: int
    subjects_in_manifest: int
    subjects_loaded: int
    missing_by_encoder: dict[str, int]
    missing_by_subject: dict[str, int]
    encoder_dir_names: list[str]
    modality_names: list[str]
    manifest_hash: str
    max_missing_fraction: float

    @property
    def missing_fraction(self) -> float:
        if self.manifest_rows == 0:
            return 0.0
        return self.dropped_rows / self.manifest_rows

    def as_dict(self) -> dict[str, Any]:
        report = asdict(self)
        report["missing_fraction"] = self.missing_fraction
        return report


def compute_manifest_hash(manifest: pd.DataFrame) -> str:
    """Return a stable content hash for a manifest dataframe."""
    canonical = manifest.drop(columns=["manifest_hash"], errors="ignore").copy()
    canonical = canonical.reindex(sorted(canonical.columns), axis=1)
    payload = canonical.to_csv(index=False, lineterminator="\n")
    return sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_egoemotion_10s_by_subject(
    embeddings_dir: str | Path,
    encoder_dir_names: list[str],
    config_mod_names: list[str],
    manifest: pd.DataFrame,
    *,
    normalize_embedding: Callable[[torch.Tensor, str], torch.Tensor] | None = None,
    pool_clips: bool = False,
    max_missing_fraction: float = 0.01,
    require_manifest_hash: bool = False,
) -> tuple[dict[str, list[dict]], Ego10sLoadReport]:
    """Load checked task-aware egoEMOTION 10s embeddings grouped by subject.

    Missing modality files are reported and bounded. Present files are checked
    against the manifest metadata before they can enter LOSO training.
    """
    if len(encoder_dir_names) != len(config_mod_names):
        raise Ego10sManifestError(
            "encoder_dir_names and config_mod_names must have the same length"
        )
    if not 0.0 <= max_missing_fraction <= 1.0:
        raise Ego10sManifestError("max_missing_fraction must be between 0 and 1")

    _validate_manifest_columns(manifest)

    emb_path = Path(embeddings_dir)
    data_by_subject: dict[str, list[dict]] = {}
    missing_by_encoder: Counter[str] = Counter()
    missing_by_subject: Counter[str] = Counter()
    manifest_hash = compute_manifest_hash(manifest)

    for _, row in manifest.sort_values(["subject", "global_seq"]).iterrows():
        subj = _normalize_subject(row["subject"])
        global_seq = int(row["global_seq"])
        missing = [
            enc for enc in encoder_dir_names
            if not _embedding_path(emb_path, enc, subj, global_seq).exists()
        ]

        if missing:
            missing_by_subject[subj] += 1
            missing_by_encoder.update(missing)
            continue

        emb_list = []
        for enc in encoder_dir_names:
            path = _embedding_path(emb_path, enc, subj, global_seq)
            data = torch.load(path, weights_only=False)
            _validate_embedding_metadata(
                path=path,
                payload=data,
                subject=subj,
                global_seq=global_seq,
                emotion_label=int(row["emotion_label"]),
                emotion_name=str(row["emotion_name"]),
                manifest_hash=manifest_hash,
                require_manifest_hash=require_manifest_hash,
            )

            if "embedding" not in data:
                raise Ego10sManifestError(f"{path}: missing 'embedding' tensor")
            emb = data["embedding"]
            if not isinstance(emb, torch.Tensor):
                raise Ego10sManifestError(f"{path}: 'embedding' is not a tensor")
            if normalize_embedding is not None:
                emb = normalize_embedding(emb, enc)
            if pool_clips and enc == "video_mae_v2" and emb.dim() == 2:
                emb = emb.mean(dim=0)
            emb_list.append(emb)

        sample = {
            "embeddings": emb_list,
            "modality_ids": config_mod_names,
            "labels": {
                "emotion_label": torch.tensor(int(row["emotion_label"]), dtype=torch.long),
                "soft_label": _parse_soft_label(row["soft_label"]),
                "vad": torch.tensor([
                    float(row["valence"]),
                    float(row["arousal"]),
                    float(row["dominance"]),
                ], dtype=torch.float32),
            },
            "meta": {
                "subject": subj,
                "global_seq": global_seq,
                "task_name": row["task_name"],
                "chunk_idx_in_task": int(row["chunk_idx_in_task"]),
                "emotion_name": str(row["emotion_name"]),
            },
        }
        data_by_subject.setdefault(subj, []).append(sample)

    manifest_rows = len(manifest)
    loaded_rows = sum(len(samples) for samples in data_by_subject.values())
    dropped_rows = manifest_rows - loaded_rows
    report = Ego10sLoadReport(
        manifest_rows=manifest_rows,
        loaded_rows=loaded_rows,
        dropped_rows=dropped_rows,
        subjects_in_manifest=manifest["subject"].map(_normalize_subject).nunique(),
        subjects_loaded=len(data_by_subject),
        missing_by_encoder=dict(sorted(missing_by_encoder.items())),
        missing_by_subject=dict(sorted(missing_by_subject.items())),
        encoder_dir_names=list(encoder_dir_names),
        modality_names=list(config_mod_names),
        manifest_hash=manifest_hash,
        max_missing_fraction=max_missing_fraction,
    )

    if report.missing_fraction > max_missing_fraction:
        raise Ego10sManifestError(
            "missing embedding fraction "
            f"{report.missing_fraction:.4f} exceeds max_missing_fraction "
            f"{max_missing_fraction:.4f}; missing_by_encoder={report.missing_by_encoder}"
        )

    return data_by_subject, report


def _validate_manifest_columns(manifest: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_10S_COLUMNS - set(manifest.columns))
    if missing:
        raise Ego10sManifestError(f"manifest missing required columns: {missing}")


def _embedding_path(root: Path, encoder: str, subject: str, global_seq: int) -> Path:
    return root / encoder / subject / f"segment_{global_seq:04d}.pt"


def _parse_soft_label(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        label = value.detach().clone().float()
    elif isinstance(value, (list, tuple)):
        label = torch.tensor([float(v) for v in value], dtype=torch.float32)
    else:
        label = torch.tensor(
            [float(v.strip()) for v in str(value).split(",")],
            dtype=torch.float32,
        )

    if label.numel() != 9:
        raise Ego10sManifestError(f"soft_label must contain 9 values, got {label.numel()}")
    return label.reshape(9)


def _validate_embedding_metadata(
    *,
    path: Path,
    payload: dict[str, Any],
    subject: str,
    global_seq: int,
    emotion_label: int,
    emotion_name: str,
    manifest_hash: str,
    require_manifest_hash: bool,
) -> None:
    if not isinstance(payload, dict):
        raise Ego10sManifestError(f"{path}: expected a dict payload")

    if _metadata_value(payload, "subject") is not None:
        found_subject = _normalize_subject(_metadata_value(payload, "subject"))
        if found_subject != subject:
            raise Ego10sManifestError(
                f"{path}: subject mismatch, manifest={subject}, embedding={found_subject}"
            )

    if _metadata_value(payload, "segment_idx") is not None:
        found_seq = int(_metadata_value(payload, "segment_idx"))
        if found_seq != global_seq:
            raise Ego10sManifestError(
                f"{path}: segment_idx mismatch, manifest={global_seq}, embedding={found_seq}"
            )

    if _metadata_value(payload, "label") is not None:
        found_label = int(_metadata_value(payload, "label"))
        if found_label != emotion_label:
            raise Ego10sManifestError(
                f"{path}: label mismatch, manifest={emotion_label}, embedding={found_label}"
            )

    if _metadata_value(payload, "emotion") is not None:
        found_emotion = str(_metadata_value(payload, "emotion"))
        if found_emotion != emotion_name:
            raise Ego10sManifestError(
                f"{path}: emotion mismatch, manifest={emotion_name}, embedding={found_emotion}"
            )

    found_hash = _metadata_value(payload, "manifest_hash")
    if require_manifest_hash and found_hash is None:
        raise Ego10sManifestError(f"{path}: missing manifest_hash")
    if found_hash is not None and str(found_hash) != manifest_hash:
        raise Ego10sManifestError(
            f"{path}: manifest_hash mismatch, manifest={manifest_hash}, embedding={found_hash}"
        )


def _metadata_value(payload: dict[str, Any], key: str) -> Any:
    if key in payload:
        return payload[key]
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        return metadata.get(key)
    return None


def _normalize_subject(value: Any) -> str:
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text.zfill(3) if text.isdigit() else text
    if number.is_integer():
        return f"{int(number):03d}"
    return text

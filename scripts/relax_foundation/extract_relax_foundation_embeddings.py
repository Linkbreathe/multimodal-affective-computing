#!/usr/bin/env python
"""Assemble Relax condition-level foundation embedding caches from 10s windows.

This script consumes Relax-Model windows and auditable per-window modality
outputs. It does not silently create foundation embeddings. If a requested
window tensor is missing, strict mode fails unless the missingness is an
explicit Relax EEG-disabled participant or ``--allow-missing-window-files`` is
given for sensitivity analyses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mac.data.relax_foundation import (  # noqa: E402
    CONDITIONS,
    EEG_DISABLED_PARTICIPANTS,
    MODALITIES,
    RELAX_EEG_EMBED_DIM,
    RELAX_EEG_MONTAGE,
    RELAX_EEG_RUN_TAG,
    RELAX_EEG_XDF_COLUMNS,
    RELAX_REVE_MODEL,
    RELAX_REVE_POSITIONS,
    RelaxHardFailure,
    RelaxRunPaths,
    assert_relax_modalities,
    read_relax_csv,
    relax_eeg_contract_payload,
    validate_phase0_inputs,
    write_json,
)
from mac.data.relax_attention_video import (  # noqa: E402
    AttentionVideoConfig,
    RelaxAttentionVideoExtractor,
)


def _sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_revision(source: str | Path) -> str | None:
    path = Path(source)
    if path.exists():
        return _sha256(path) if path.is_file() else None
    try:
        from huggingface_hub import HfApi

        return str(HfApi().model_info(str(source)).sha)
    except Exception:
        return None


def _safe_zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True).clamp(min=eps)
    return (x - mean) / std


def _select_relax_eeg(window_samples: np.ndarray) -> np.ndarray:
    """Select EEG columns without changing their acquisition order."""
    values = np.asarray(window_samples)
    if values.ndim != 2 or values.shape[1] <= max(RELAX_EEG_XDF_COLUMNS):
        raise RelaxHardFailure(
            f"Relax XDF window cannot provide EEG columns {list(RELAX_EEG_XDF_COLUMNS)}: {values.shape}"
        )
    return values[:, RELAX_EEG_XDF_COLUMNS].T


def _fixed_reve_embedding_dim(x: torch.Tensor) -> torch.Tensor:
    """Return the pre-registered 1024-D representation from REVE-large output.

    The current public REVE-large checkpoint has a native width of 1216.  The
    run contract predates that checkpoint width, so a deterministic adaptive
    average reduction is recorded explicitly in provenance instead of adding
    trainable parameters or fitting a projection on Relax data.
    """
    if x.ndim != 2:
        raise RelaxHardFailure(f"Pooled REVE output must be [batch,dim], got {tuple(x.shape)}")
    if x.shape[-1] == RELAX_EEG_EMBED_DIM:
        return x
    return F.adaptive_avg_pool1d(x.unsqueeze(1), RELAX_EEG_EMBED_DIM).squeeze(1)


def _merge_window_manifest(root: Path, modality_payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Merge modality provenance so later attention-video work cannot erase EEG."""
    path = root / "window_embedding_manifest.json"
    existing: dict[str, Any] = {}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    modalities = existing.get("modalities", {})
    if isinstance(modalities, list):
        modalities = {
            str(name): {"legacy_manifest": True}
            for name in modalities
        }
    if not isinstance(modalities, dict):
        raise RelaxHardFailure(f"Invalid modality provenance in {path}")
    modalities.update(modality_payloads)
    merged = {
        "schema_version": "2.0",
        "run_tag": existing.get("run_tag", RELAX_EEG_RUN_TAG),
        "relax_run_dir": existing.get("relax_run_dir"),
        "window_cache_root": str(root),
        "modalities": modalities,
    }
    write_json(path, merged)
    return merged


def copy_non_eeg_window_cache(
    *,
    source_root: str | Path,
    destination_root: str | Path,
    modalities: list[str],
    run_tag: str = RELAX_EEG_RUN_TAG,
) -> dict[str, Any]:
    """Copy immutable non-EEG tensors and verify every destination checksum."""
    source_root = Path(source_root)
    destination_root = Path(destination_root)
    copied: dict[str, Any] = {}
    for modality in assert_relax_modalities(modalities):
        if modality == "eeg":
            raise RelaxHardFailure("Corrected cache initialization must not copy legacy EEG tensors")
        source_dir = source_root / modality
        if not source_dir.exists():
            raise RelaxHardFailure(f"Legacy {modality} cache is missing: {source_dir}")
        records: list[dict[str, str]] = []
        for source in sorted(source_dir.rglob("*.pt")):
            relative = source.relative_to(source_root)
            destination = destination_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            source_sha = _sha256(source)
            if destination.exists() and _sha256(destination) != source_sha:
                raise RelaxHardFailure(f"Refusing to overwrite mismatched corrected-cache file: {destination}")
            if not destination.exists():
                shutil.copy2(source, destination)
            destination_sha = _sha256(destination)
            if destination_sha != source_sha:
                raise RelaxHardFailure(f"Checksum mismatch after copying {source} -> {destination}")
            records.append({"path": str(relative), "sha256": str(source_sha)})
        copied[modality] = {
            "source": "copied_from_legacy_cache",
            "source_root": str(source_root),
            "file_count": len(records),
            "files": records,
        }
    destination_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "1.0",
        "run_tag": run_tag,
        "source_root": str(source_root),
        "destination_root": str(destination_root),
        "modalities": copied,
    }
    copy_manifest_path = destination_root / "cache_copy_manifest.json"
    write_json(copy_manifest_path, manifest)
    provenance = {
        modality: {
            "source": payload["source"],
            "source_root": payload["source_root"],
            "file_count": payload["file_count"],
            "cache_copy_manifest": str(copy_manifest_path),
            "cache_copy_manifest_sha256": _sha256(copy_manifest_path),
        }
        for modality, payload in copied.items()
    }
    _merge_window_manifest(destination_root, provenance)
    return manifest


def _resample_last_axis(
    x: torch.Tensor,
    *,
    source_rate: float,
    target_rate: float,
) -> torch.Tensor:
    if source_rate <= 0 or target_rate <= 0:
        raise RelaxHardFailure(f"Invalid resampling rates: {source_rate} -> {target_rate}")
    if abs(source_rate - target_rate) < 1e-6:
        return x
    target_len = int(round(x.shape[-1] * target_rate / source_rate))
    if target_len <= 0:
        raise RelaxHardFailure("Resampling produced a non-positive target length")
    flat = x.reshape(-1, 1, x.shape[-1])
    out = F.interpolate(flat, size=target_len, mode="linear", align_corners=False)
    return out.reshape(*x.shape[:-1], target_len)


def _resample_matrix(values: np.ndarray, target_len: int) -> np.ndarray:
    if values.ndim != 2:
        raise RelaxHardFailure(f"Expected 2D time series, got shape {values.shape}")
    if len(values) == 0:
        return np.zeros((target_len, values.shape[1]), dtype=np.float32)
    if len(values) == target_len:
        return values.astype(np.float32, copy=False)
    source_x = np.linspace(0.0, 1.0, num=len(values), endpoint=True)
    target_x = np.linspace(0.0, 1.0, num=target_len, endpoint=True)
    out = np.empty((target_len, values.shape[1]), dtype=np.float32)
    for col in range(values.shape[1]):
        series = values[:, col]
        finite = np.isfinite(series)
        if finite.sum() == 0:
            out[:, col] = 0.0
        elif finite.sum() == 1:
            out[:, col] = float(series[finite][0])
        else:
            out[:, col] = np.interp(target_x, source_x[finite], series[finite])
    return out


def _save_window_tensor(
    root: Path,
    modality: str,
    participant: str,
    condition: str,
    window_index: int,
    tensor: torch.Tensor,
    *,
    valid: bool,
    metadata: dict[str, Any] | None = None,
) -> None:
    path = root / modality / participant / f"{condition}_w{window_index:03d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embedding" if tensor.ndim == 1 else "signal": tensor.detach().cpu().float().contiguous(),
            "valid": bool(valid),
            "metadata": metadata or {},
        },
        path,
    )


def _quaternion_to_euler_xyz(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q.T
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    t2 = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(t2)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return np.stack([roll, pitch, yaw], axis=1)


def _eye_signal(rows: list[dict[str, Any]], *, target_len: int = 900) -> tuple[torch.Tensor, bool]:
    if not rows:
        return torch.zeros(2, target_len), False
    raw = np.asarray(
        [
            [
                float(row.get("gaze_direction_x", np.nan)),
                float(row.get("gaze_direction_y", np.nan)),
                float(row.get("gaze_direction_z", np.nan)),
            ]
            for row in rows
        ],
        dtype=float,
    )
    finite = np.isfinite(raw).all(axis=1)
    if finite.sum() < 10:
        return torch.zeros(2, target_len), False
    raw = raw[finite]
    azimuth = np.arctan2(raw[:, 0], raw[:, 2])
    elevation = np.arctan2(raw[:, 1], np.sqrt(raw[:, 0] ** 2 + raw[:, 2] ** 2))
    signal = _resample_matrix(np.stack([azimuth, elevation], axis=1), target_len).T
    return torch.from_numpy(signal), True


def _head_signal(rows: list[dict[str, Any]], *, target_len: int = 500) -> tuple[torch.Tensor, bool]:
    if not rows:
        return torch.zeros(10, target_len), False
    columns = [
        "head_position_x", "head_position_y", "head_position_z",
        "head_rotation_x", "head_rotation_y", "head_rotation_z", "head_rotation_w",
        "head_velocity_x", "head_velocity_y", "head_velocity_z",
        "head_angular_velocity_deg_s",
    ]
    raw = np.asarray([[float(row.get(column, np.nan)) for column in columns] for row in rows], dtype=float)
    finite = np.isfinite(raw).all(axis=1)
    if finite.sum() < 10:
        return torch.zeros(10, target_len), False
    raw = raw[finite]
    position = raw[:, 0:3]
    position = position - position[0:1]
    euler = _quaternion_to_euler_xyz(raw[:, 3:7])
    euler = np.unwrap(euler, axis=0) - euler[0:1]
    velocity = raw[:, 7:10]
    angular = raw[:, 10:11]
    signal = _resample_matrix(np.concatenate([position, euler, velocity, angular], axis=1), target_len).T
    return torch.from_numpy(signal), True


def _iso_to_unix_ms_python311_compatible(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    if "." in text:
        prefix, suffix = text.split(".", 1)
        tz_pos = min(
            [pos for pos in (suffix.find("+"), suffix.find("-"), suffix.find("Z")) if pos >= 0],
            default=len(suffix),
        )
        fraction = suffix[:tz_pos][:6]
        text = f"{prefix}.{fraction}{suffix[tz_pos:]}"
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(round(parsed.timestamp() * 1000.0))


def _patch_relax_video_iso_parser(relax_video_module: Any) -> None:
    relax_video_module._iso_to_unix_ms = _iso_to_unix_ms_python311_compatible


def _window_cache_candidates(
    root: Path,
    modality: str,
    participant: str,
    condition: str,
    window_index: int,
    global_index: int,
) -> list[Path]:
    mod_dir = root / modality / participant
    return [
        mod_dir / f"{condition}_w{window_index:03d}.pt",
        mod_dir / f"{condition}_window_{window_index:03d}.pt",
        mod_dir / f"window_{global_index:04d}.pt",
    ]


def _load_tensor(path: Path) -> tuple[torch.Tensor, bool]:
    item = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(item, torch.Tensor):
        return item.float(), True
    if not isinstance(item, dict):
        raise RelaxHardFailure(f"Window cache item must be a Tensor or dict: {path}")
    key = "signal" if "signal" in item else "embedding" if "embedding" in item else None
    if key is None:
        raise RelaxHardFailure(f"Window cache item lacks 'embedding' or 'signal': {path}")
    valid = bool(item.get("valid", True))
    return torch.as_tensor(item[key]).float(), valid


def _infer_modality_shape(cache_root: Path, modality: str) -> tuple[int, ...]:
    files = sorted((cache_root / modality).rglob("*.pt"))
    for file in files:
        tensor, _ = _load_tensor(file)
        if tensor.ndim == 1:
            return (int(tensor.shape[0]),)
        if tensor.ndim == 2:
            return (int(tensor.shape[0]), int(tensor.shape[1]))
        raise RelaxHardFailure(f"Unsupported {modality} tensor rank {tensor.ndim} in {file}")
    raise RelaxHardFailure(f"No cached window tensors found for requested modality: {modality}")


def _video_frame_rows_for_attention(
    selected_frames: list[Any],
    video_frame_table: pd.DataFrame,
) -> list[dict[str, Any]]:
    required = {
        "unix_time_ms",
        "width",
        "height",
        "camera_forward_x",
        "camera_forward_y",
        "camera_forward_z",
        "camera_up_x",
        "camera_up_y",
        "camera_up_z",
    }
    missing = sorted(required - set(video_frame_table.columns))
    if missing:
        raise RelaxHardFailure(f"video_frames.csv missing attention-video fields: {missing}")

    table = video_frame_table.copy()
    by_index: dict[int, dict[str, Any]] = {}
    if "frame_index" in table.columns:
        frame_indices = pd.to_numeric(table["frame_index"], errors="coerce")
        for idx, row in table.iterrows():
            value = frame_indices.loc[idx]
            if np.isfinite(value):
                by_index[int(value)] = row.to_dict()

    time_values = pd.to_numeric(table["unix_time_ms"], errors="coerce").to_numpy(dtype=float)
    finite_time_indices = np.flatnonzero(np.isfinite(time_values))
    rows: list[dict[str, Any]] = []
    for frame in selected_frames:
        raw_index = getattr(frame, "frame_index", None)
        row: dict[str, Any] | None = None
        if raw_index is not None:
            try:
                row = by_index.get(int(raw_index))
            except Exception:
                row = None
        if row is None:
            frame_time = getattr(frame, "unix_time_ms", None)
            if frame_time is not None and finite_time_indices.size:
                nearest = int(finite_time_indices[np.argmin(np.abs(time_values[finite_time_indices] - float(frame_time)))])
                row = table.iloc[nearest].to_dict()
        if row is None:
            raise RelaxHardFailure(f"Could not map selected video frame to video_frames.csv row: {frame}")
        row = dict(row)
        row["unix_time_ms"] = getattr(frame, "unix_time_ms")
        row["path"] = getattr(frame, "path")
        rows.append(row)
    return rows


def _load_attention_video_frame_table(
    path: Path,
    session_dir: Path | None,
    relax_video_module: Any,
) -> pd.DataFrame:
    if session_dir is None:
        raise RelaxHardFailure("attention_video requires source_manifest.session_dir")
    required = {
        "unix_time_ms",
        "frame_index",
        "width",
        "height",
        "camera_forward_x",
        "camera_forward_y",
        "camera_forward_z",
        "camera_up_x",
        "camera_up_y",
        "camera_up_z",
    }
    _, decimal, _ = relax_video_module.sniff_csv(path)
    rows: list[dict[str, Any]] = []
    for raw in relax_video_module.iter_csv(path):
        missing = sorted(required - set(raw))
        if missing:
            raise RelaxHardFailure(f"video_frames.csv missing attention-video fields: {missing}")
        row: dict[str, Any] = {}
        for field in required:
            value = relax_video_module.parse_float(raw.get(field), decimal)
            row[field] = float(value) if value is not None else float("nan")
        if not np.isfinite(row["unix_time_ms"]) or row["unix_time_ms"] < 100_000_000_000:
            timestamp = relax_video_module._iso_to_unix_ms(raw.get("utc_timestamp_iso"))
            row["unix_time_ms"] = float(timestamp) if timestamp is not None else float("nan")
        resolved_path = relax_video_module.resolve_video_path(session_dir, raw)
        row["path"] = resolved_path
        rows.append(row)
    if not rows:
        raise RelaxHardFailure(f"video_frames.csv has no auditable rows for attention_video: {path}")
    return pd.DataFrame(rows)


def assemble_condition_cache(
    *,
    relax_run_dir: str | Path,
    window_cache_root: str | Path,
    modalities: list[str],
    output_path: str | Path,
    allow_missing_window_files: bool = False,
    eeg_weights: str | Path | None = None,
    eeg_pos_bank: str | Path | None = None,
    run_tag: str = RELAX_EEG_RUN_TAG,
) -> dict[str, Any]:
    modalities = assert_relax_modalities(modalities)
    paths = RelaxRunPaths.from_root(relax_run_dir)
    cache_root = Path(window_cache_root)
    output_path = Path(output_path)
    if not cache_root.exists():
        raise RelaxHardFailure(f"Window embedding cache root is missing: {cache_root}")

    labels = read_relax_csv(paths.preprocessed / "condition_labels.csv")
    windows = read_relax_csv(paths.preprocessed / "windows.csv")
    required = {"participant_id", "condition", "relaxation", "discomfort"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise RelaxHardFailure(f"condition_labels.csv missing columns: {missing}")
    missing = sorted({"participant_id", "condition", "condition_window_index"} - set(windows.columns))
    if missing:
        raise RelaxHardFailure(f"windows.csv missing columns: {missing}")

    labels = labels.copy()
    labels["participant_id"] = labels["participant_id"].astype(str)
    labels["condition"] = labels["condition"].astype(str)
    windows = windows.copy()
    windows["participant_id"] = windows["participant_id"].astype(str)
    windows["condition"] = windows["condition"].astype(str)
    windows["condition_window_index"] = pd.to_numeric(windows["condition_window_index"], errors="raise").astype(int)
    windows = windows.sort_values(["participant_id", "condition", "condition_window_index"]).reset_index(drop=True)

    modality_shapes = {modality: _infer_modality_shape(cache_root, modality) for modality in modalities}
    samples: list[dict[str, Any]] = []
    missing_records: list[dict[str, Any]] = []

    label_by_key = {
        (row["participant_id"], row["condition"]): row
        for _, row in labels.iterrows()
    }
    for (participant, condition), group in windows.groupby(["participant_id", "condition"], sort=True):
        if condition not in CONDITIONS:
            raise RelaxHardFailure(f"Unexpected Relax Condition in windows.csv: {condition}")
        label = label_by_key.get((participant, condition))
        if label is None:
            raise RelaxHardFailure(f"Missing label for {participant}/{condition}")
        sample = {
            "participant_id": participant,
            "condition": condition,
            "condition_index": CONDITIONS.index(condition),
            "labels": {
                "relaxation": float(label["relaxation"]),
                "discomfort": float(label["discomfort"]),
            },
            "embeddings": {},
            "masks": {},
        }
        for modality in modalities:
            tensors: list[torch.Tensor] = []
            masks: list[bool] = []
            zero_shape = modality_shapes[modality]
            for global_idx, row in group.iterrows():
                window_idx = int(row["condition_window_index"])
                candidates = _window_cache_candidates(
                    cache_root, modality, participant, condition, window_idx, int(global_idx)
                )
                found = next((path for path in candidates if path.exists()), None)
                expected_missing = modality == "eeg" and participant in EEG_DISABLED_PARTICIPANTS
                if found is None:
                    if not (allow_missing_window_files or expected_missing):
                        raise RelaxHardFailure(
                            f"Missing {modality} window cache for {participant}/{condition}/w{window_idx:03d}"
                        )
                    tensors.append(torch.zeros(*zero_shape, dtype=torch.float32))
                    masks.append(False)
                    missing_records.append(
                        {
                            "participant_id": participant,
                            "condition": condition,
                            "window_index": window_idx,
                            "modality": modality,
                            "reason": "eeg_disabled_by_relax" if expected_missing else "missing_window_cache",
                        }
                    )
                    continue
                tensor, valid = _load_tensor(found)
                if tuple(tensor.shape) != zero_shape:
                    raise RelaxHardFailure(
                        f"{found} shape {tuple(tensor.shape)} does not match inferred {modality} shape {zero_shape}"
                    )
                tensors.append(tensor)
                masks.append(valid)
            sample["embeddings"][modality] = torch.stack(tensors)
            sample["masks"][modality] = torch.tensor(masks, dtype=torch.bool)
        samples.append(sample)

    embedding_dims: dict[str, Any] = {
        modality: (shape[0] if len(shape) == 1 else list(shape))
        for modality, shape in modality_shapes.items()
    }
    window_manifest_path = cache_root / "window_embedding_manifest.json"
    if not window_manifest_path.exists():
        raise RelaxHardFailure(f"Window embedding manifest is missing: {window_manifest_path}")
    window_manifest = json.loads(window_manifest_path.read_text(encoding="utf-8"))
    modality_provenance = window_manifest.get("modalities", {})
    if not isinstance(modality_provenance, dict):
        raise RelaxHardFailure("Window embedding manifest must contain per-modality provenance")
    missing_provenance = sorted(set(modalities) - set(modality_provenance))
    if missing_provenance:
        raise RelaxHardFailure(f"Window embedding manifest lacks provenance for {missing_provenance}")

    payload = {
        "schema_version": "2.0",
        "samples": samples,
        "embedding_dims": embedding_dims,
        "modalities": modalities,
        "manifest": {
            "run_tag": run_tag,
            "relax_run_dir": str(paths.root),
            "window_cache_root": str(cache_root),
            "eeg_model": "reve" if "eeg" in modalities else None,
            "eeg_channel_contract": relax_eeg_contract_payload() if "eeg" in modalities else None,
            "eeg_montage": list(RELAX_EEG_MONTAGE) if "eeg" in modalities else None,
            "eeg_weights_sha256": _sha256(eeg_weights),
            "eeg_pos_bank_sha256": _sha256(eeg_pos_bank),
            "eeg_weights_revision": _model_revision(eeg_weights) if eeg_weights is not None else None,
            "eeg_pos_bank_revision": _model_revision(eeg_pos_bank) if eeg_pos_bank is not None else None,
            "window_manifest": str(window_manifest_path),
            "window_manifest_sha256": _sha256(window_manifest_path),
            "modality_provenance": {name: modality_provenance[name] for name in modalities},
            "missing_records": missing_records,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    write_json(output_path.with_suffix(".manifest.json"), payload["manifest"])
    return payload["manifest"]


class RelaxWindowEmbeddingExtractor:
    """Extract auditable per-window tensors from Relax-Model outputs and raw sources."""

    def __init__(
        self,
        *,
        relax_run_dir: str | Path,
        window_cache_root: str | Path,
        modalities: list[str],
        device: str,
        batch_size: int,
        reve_model: str,
        reve_positions: str,
        relax_model_src: str | Path,
        ecg_weights: str,
        eye_weights: str,
        attention_config: AttentionVideoConfig | None = None,
        participants: list[str] | None = None,
        run_tag: str = RELAX_EEG_RUN_TAG,
    ) -> None:
        self.paths = RelaxRunPaths.from_root(relax_run_dir)
        self.window_cache_root = Path(window_cache_root)
        self.modalities = assert_relax_modalities(modalities)
        self.participants = {str(participant) for participant in participants} if participants else None
        self.run_tag = str(run_tag)
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.reve_model_name = reve_model
        self.reve_positions_name = reve_positions
        self.relax_model_src = Path(relax_model_src)
        if str(self.relax_model_src) not in sys.path:
            sys.path.insert(0, str(self.relax_model_src))
        self.ecg_weights = ecg_weights
        self.eye_weights = eye_weights
        self.attention_video = RelaxAttentionVideoExtractor(attention_config or AttentionVideoConfig())

        self._ecg_encoder = None
        self._eye_encoder = None
        self._video_encoder = None
        self._reve_model = None
        self._reve_positions = None
        self._reve_model_revision = _model_revision(reve_model)
        self._reve_positions_revision = _model_revision(reve_positions)

    def _load_source_manifest(self) -> pd.DataFrame:
        path = self.paths.root / "manifests" / "source_manifest.csv"
        if not path.exists():
            path = self.paths.root / "source_manifest.csv"
        if not path.exists():
            path = self.paths.root / "manifests" / "source_manifest.csv"
        if not path.exists():
            raise RelaxHardFailure(f"Relax source manifest is missing: {path}")
        frame = pd.read_csv(path)
        if "participant_id" not in frame.columns:
            raise RelaxHardFailure(f"Source manifest lacks participant_id: {path}")
        return frame

    def _load_ecg_encoder(self):
        if self._ecg_encoder is None:
            from mac.encoders.ecgfounder import ECGFounderEncoder

            self._ecg_encoder = ECGFounderEncoder(weights_path=self.ecg_weights).to(self.device)
            self._ecg_encoder.eval()
        return self._ecg_encoder

    def _load_eye_encoder(self):
        if self._eye_encoder is None:
            from mac.encoders.inceptiontime import InceptionTimeGazeEncoder

            encoder = InceptionTimeGazeEncoder().to(self.device)
            weights = Path(self.eye_weights)
            if not weights.exists():
                raise RelaxHardFailure(f"Eye encoder checkpoint is missing: {weights}")
            encoder.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
            encoder.eval()
            self._eye_encoder = encoder
        return self._eye_encoder

    def _load_video_encoder(self):
        if self._video_encoder is None:
            from mac.encoders.video_mae import VideoMAEV2Encoder

            encoder = VideoMAEV2Encoder().to(self.device)
            encoder.eval()
            self._video_encoder = encoder
        return self._video_encoder

    def _load_reve(self):
        if self._reve_model is None or self._reve_positions is None:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            from transformers import AutoConfig, AutoModel

            pos_config = AutoConfig.from_pretrained(
                self.reve_positions_name,
                trust_remote_code=True,
            )
            pos_bank = AutoModel.from_config(pos_config, trust_remote_code=True)
            pos_state = load_file(
                hf_hub_download(self.reve_positions_name, filename="model.safetensors"),
                device="cpu",
            )
            missing, unexpected = pos_bank.load_state_dict(pos_state, strict=False)
            if "embedding" in missing or unexpected:
                raise RelaxHardFailure(
                    f"Could not load REVE position bank cleanly; missing={missing}, unexpected={unexpected}"
                )
            pos_bank = pos_bank.to(self.device)
            model = AutoModel.from_pretrained(
                self.reve_model_name,
                trust_remote_code=True,
            ).to(self.device)
            for parameter in pos_bank.parameters():
                parameter.requires_grad_(False)
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            pos_bank.eval()
            model.eval()
            positions = pos_bank(list(RELAX_EEG_MONTAGE)).to(self.device)
            if positions.shape[0] != len(RELAX_EEG_MONTAGE):
                raise RelaxHardFailure(
                    f"REVE position bank did not resolve montage {list(RELAX_EEG_MONTAGE)}"
                )
            self._reve_model = model
            self._reve_positions = positions
        return self._reve_model, self._reve_positions

    def _encode_batches(
        self,
        tensors: list[torch.Tensor],
        encode_fn,
    ) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        for start in range(0, len(tensors), self.batch_size):
            batch = torch.stack(tensors[start : start + self.batch_size]).to(self.device)
            with torch.no_grad():
                out = encode_fn(batch)
            outputs.extend(out.detach().cpu())
        return outputs

    def _encode_reve(self, batch: torch.Tensor) -> torch.Tensor:
        model, positions = self._load_reve()
        pos = positions.unsqueeze(0).expand(batch.size(0), -1, -1)
        out = model(batch, pos)
        if hasattr(model, "attention_pooling"):
            out = model.attention_pooling(out)
        elif out.ndim == 4:
            out = out.mean(dim=(1, 2))
        return _fixed_reve_embedding_dim(out)

    def _extract_video_clip(self, paths: list[Path]) -> torch.Tensor:
        import cv2
        from mac.data.video_transforms import normalize_clip, preprocess_frame

        frames = []
        for path in paths:
            frame = cv2.imread(str(path))
            if frame is None:
                raise RelaxHardFailure(f"Could not read video frame: {path}")
            frames.append(preprocess_frame(frame))
        if len(frames) != 16:
            raise RelaxHardFailure(f"Video clip must have 16 frames, got {len(frames)}")
        return normalize_clip(np.stack(frames, axis=0))

    def run(self) -> dict[str, Any]:
        import mac.data.video as relax_video
        from mac.features.extract import NUMERIC_FIELDS, _load_log_rows, _load_physio, _slice_rows

        _patch_relax_video_iso_parser(relax_video)
        load_video_index = relax_video.load_video_index
        uniform_clip_frames = relax_video.uniform_clip_frames

        windows = read_relax_csv(self.paths.preprocessed / "windows.csv")
        windows["participant_id"] = windows["participant_id"].astype(str)
        windows["condition"] = windows["condition"].astype(str)
        windows["condition_window_index"] = pd.to_numeric(
            windows["condition_window_index"], errors="raise"
        ).astype(int)
        if self.participants is not None:
            windows = windows[windows["participant_id"].isin(self.participants)].copy()
            missing = sorted(self.participants - set(windows["participant_id"]))
            if missing:
                raise RelaxHardFailure(f"Requested extraction participants missing from windows.csv: {missing}")
        source_manifest = self._load_source_manifest()
        source_by_participant = {
            str(row["participant_id"]): row for _, row in source_manifest.iterrows()
        }

        zero_dims = {
            "ecg": 1024,
            "eeg": RELAX_EEG_EMBED_DIM,
            "eye": 128,
            "video": 768,
            "attention_video": 768,
        }
        extracted = {modality: 0 for modality in self.modalities}
        invalid = {modality: 0 for modality in self.modalities}

        for participant, participant_windows in windows.groupby("participant_id", sort=True):
            if participant not in source_by_participant:
                raise RelaxHardFailure(f"Source manifest missing participant {participant}")
            source = source_by_participant[participant]
            xdf_path = Path(str(source.get("xdf_path")))
            if not xdf_path.exists():
                raise RelaxHardFailure(f"{participant}: XDF source missing: {xdf_path}")
            if "ecg" in self.modalities or "eeg" in self.modalities:
                samples, timestamps, sample_rate = _load_physio(xdf_path, "eeg")
            else:
                samples = timestamps = None
                sample_rate = None
            samples_csv = Path(str(source.get("samples_csv"))) if pd.notna(source.get("samples_csv")) else None
            eye_tracking_csv = (
                Path(str(source.get("eye_tracking_csv"))) if pd.notna(source.get("eye_tracking_csv")) else None
            )
            video_frames_csv = (
                Path(str(source.get("video_frames_csv"))) if pd.notna(source.get("video_frames_csv")) else None
            )
            session_dir = Path(str(source.get("session_dir"))) if pd.notna(source.get("session_dir")) else None
            head_rows = _load_log_rows(
                samples_csv,
                NUMERIC_FIELDS,
            )
            eye_rows = _load_log_rows(
                eye_tracking_csv,
                NUMERIC_FIELDS,
            )
            if "attention_video" in self.modalities:
                if eye_tracking_csv is None or not eye_tracking_csv.exists():
                    raise RelaxHardFailure(f"{participant}: attention_video requires eye_tracking.csv")
                if video_frames_csv is None or not video_frames_csv.exists():
                    raise RelaxHardFailure(f"{participant}: attention_video requires video_frames.csv")
                eye_table = read_relax_csv(eye_tracking_csv)
                video_frame_table = _load_attention_video_frame_table(video_frames_csv, session_dir, relax_video)
            else:
                eye_table = pd.DataFrame()
                video_frame_table = pd.DataFrame()
            video_index = load_video_index(
                video_frames_csv,
                session_dir,
                participant,
            )

            ecg_inputs: list[tuple[dict[str, Any], torch.Tensor]] = []
            eeg_inputs: list[tuple[dict[str, Any], torch.Tensor]] = []
            eye_inputs: list[tuple[dict[str, Any], torch.Tensor]] = []
            video_inputs: list[tuple[dict[str, Any], torch.Tensor]] = []
            attention_video_inputs: list[tuple[dict[str, Any], torch.Tensor]] = []

            for _, row in participant_windows.sort_values("window_start_xdf").iterrows():
                condition = str(row["condition"])
                window_index = int(row["condition_window_index"])
                metadata = {
                    "run_tag": self.run_tag,
                    "participant_id": participant,
                    "condition": condition,
                    "condition_window_index": window_index,
                    "window_start_xdf": float(row["window_start_xdf"]),
                    "window_end_xdf": float(row["window_end_xdf"]),
                    "window_start_unix_ms": float(row["window_start_unix_ms"]),
                    "window_end_unix_ms": float(row["window_end_unix_ms"]),
                }
                if samples is not None and timestamps is not None:
                    left, right = np.searchsorted(
                        timestamps,
                        [metadata["window_start_xdf"], metadata["window_end_xdf"]],
                        side="left",
                    )
                    window_samples = samples[int(left) : int(right)]
                else:
                    window_samples = None

                if "ecg" in self.modalities:
                    if window_samples is None or sample_rate is None:
                        raise RelaxHardFailure("ECG extraction requested without loaded physiology samples")
                    try:
                        ecg = window_samples[:, 7] - window_samples[:, 8]
                        ecg_t = torch.as_tensor(ecg, dtype=torch.float32).view(1, 1, -1)
                        ecg_t = _resample_last_axis(ecg_t, source_rate=float(sample_rate), target_rate=500.0)
                        if ecg_t.shape[-1] != 5000:
                            ecg_t = F.interpolate(ecg_t, size=5000, mode="linear", align_corners=False)
                        ecg_inputs.append((metadata, _safe_zscore(ecg_t.squeeze(0))))
                    except Exception as error:
                        raise RelaxHardFailure(
                            f"{participant}/{condition}/w{window_index:03d}: ECG extraction failed: {error}"
                        ) from error

                if "eeg" in self.modalities and participant not in EEG_DISABLED_PARTICIPANTS:
                    if window_samples is None or sample_rate is None:
                        raise RelaxHardFailure("EEG extraction requested without loaded physiology samples")
                    try:
                        eeg = _select_relax_eeg(window_samples)
                        eeg_t = torch.as_tensor(eeg, dtype=torch.float32).unsqueeze(0)
                        eeg_t = _resample_last_axis(eeg_t, source_rate=float(sample_rate), target_rate=200.0)
                        if eeg_t.shape[-1] != 2000:
                            eeg_t = F.interpolate(eeg_t, size=2000, mode="linear", align_corners=False)
                        eeg_metadata = {
                            **metadata,
                            "source_xdf": str(xdf_path),
                            "source_sample_rate_hz": float(sample_rate),
                            "target_sample_rate_hz": 200.0,
                            "source_window_sample_indices": [int(left), int(right)],
                            "eeg_channel_contract": relax_eeg_contract_payload(),
                            "xdf_columns": list(RELAX_EEG_XDF_COLUMNS),
                            "reve_montage": list(RELAX_EEG_MONTAGE),
                            "reve_model": self.reve_model_name,
                            "reve_model_revision": self._reve_model_revision,
                            "reve_positions": self.reve_positions_name,
                            "reve_positions_revision": self._reve_positions_revision,
                            "native_embedding_dim": int(getattr(self._load_reve()[0].config, "embed_dim", 0)),
                            "output_embedding_dim": RELAX_EEG_EMBED_DIM,
                            "embedding_reduction": "adaptive_avg_pool1d" if int(
                                getattr(self._load_reve()[0].config, "embed_dim", 0)
                            ) != RELAX_EEG_EMBED_DIM else "none",
                            "preprocessing": "10s; resample_to_200hz; per_channel_zscore",
                        }
                        eeg_inputs.append((eeg_metadata, _safe_zscore(eeg_t.squeeze(0))))
                    except Exception as error:
                        raise RelaxHardFailure(
                            f"{participant}/{condition}/w{window_index:03d}: EEG extraction failed: {error}"
                        ) from error

                if "eye" in self.modalities:
                    eye_tensor, eye_valid = _eye_signal(
                        _slice_rows(eye_rows, metadata["window_start_unix_ms"], metadata["window_end_unix_ms"])
                    )
                    if eye_valid:
                        eye_inputs.append((metadata, eye_tensor))
                    else:
                        invalid["eye"] += 1
                        _save_window_tensor(
                            self.window_cache_root, "eye", participant, condition, window_index,
                            torch.zeros(zero_dims["eye"]), valid=False, metadata=metadata,
                        )

                if "head" in self.modalities:
                    head_tensor, head_valid = _head_signal(
                        _slice_rows(head_rows, metadata["window_start_unix_ms"], metadata["window_end_unix_ms"])
                    )
                    _save_window_tensor(
                        self.window_cache_root, "head", participant, condition, window_index,
                        head_tensor, valid=head_valid, metadata=metadata,
                    )
                    extracted["head"] += int(head_valid)
                    invalid["head"] += int(not head_valid)

                if "video" in self.modalities or "attention_video" in self.modalities:
                    try:
                        frames = uniform_clip_frames(
                            video_index,
                            metadata["window_start_unix_ms"],
                            metadata["window_end_unix_ms"],
                            count=16,
                        )
                        if len(frames) != 16:
                            if "video" in self.modalities:
                                invalid["video"] += 1
                                video_metadata = {**metadata, "reason": "insufficient_video_frames"}
                                _save_window_tensor(
                                    self.window_cache_root, "video", participant, condition, window_index,
                                    torch.zeros(zero_dims["video"]), valid=False, metadata=video_metadata,
                                )
                            if "attention_video" in self.modalities:
                                invalid["attention_video"] += 1
                                attention_metadata = {**metadata, "reason": "insufficient_video_frames"}
                                _save_window_tensor(
                                    self.window_cache_root, "attention_video", participant, condition, window_index,
                                    torch.zeros(zero_dims["attention_video"]), valid=False, metadata=attention_metadata,
                                )
                            continue
                        if "video" in self.modalities:
                            video_inputs.append((metadata, self._extract_video_clip([frame.path for frame in frames])))
                        if "attention_video" in self.modalities:
                            attention_frame_rows = _video_frame_rows_for_attention(frames, video_frame_table)
                            clip, valid, attention_metadata = self.attention_video.build_clip(
                                frame_rows=attention_frame_rows,
                                eye_rows=eye_table,
                            )
                            merged_metadata = {**metadata, **attention_metadata}
                            if valid:
                                attention_video_inputs.append((merged_metadata, clip))
                            else:
                                invalid["attention_video"] += 1
                                _save_window_tensor(
                                    self.window_cache_root, "attention_video", participant, condition, window_index,
                                    torch.zeros(zero_dims["attention_video"]), valid=False, metadata=merged_metadata,
                                )
                    except Exception as error:
                        modality_label = "video/attention_video" if "attention_video" in self.modalities else "video"
                        raise RelaxHardFailure(
                            f"{participant}/{condition}/w{window_index:03d}: {modality_label} extraction failed: {error}"
                        ) from error

            if ecg_inputs:
                encoder = self._load_ecg_encoder()
                outputs = self._encode_batches([item[1] for item in ecg_inputs], encoder)
                for (metadata, _), output in zip(ecg_inputs, outputs):
                    _save_window_tensor(
                        self.window_cache_root, "ecg", participant, metadata["condition"],
                        metadata["condition_window_index"], output, valid=True, metadata=metadata,
                    )
                    extracted["ecg"] += 1

            if eeg_inputs:
                outputs = self._encode_batches([item[1] for item in eeg_inputs], self._encode_reve)
                for (metadata, _), output in zip(eeg_inputs, outputs):
                    _save_window_tensor(
                        self.window_cache_root, "eeg", participant, metadata["condition"],
                        metadata["condition_window_index"], output, valid=True, metadata=metadata,
                    )
                    extracted["eeg"] += 1

            if eye_inputs:
                encoder = self._load_eye_encoder()
                outputs = self._encode_batches([item[1] for item in eye_inputs], encoder)
                for (metadata, _), output in zip(eye_inputs, outputs):
                    _save_window_tensor(
                        self.window_cache_root, "eye", participant, metadata["condition"],
                        metadata["condition_window_index"], output, valid=True, metadata=metadata,
                    )
                    extracted["eye"] += 1

            if video_inputs:
                encoder = self._load_video_encoder()
                outputs = self._encode_batches([item[1] for item in video_inputs], encoder)
                for (metadata, _), output in zip(video_inputs, outputs):
                    _save_window_tensor(
                        self.window_cache_root, "video", participant, metadata["condition"],
                        metadata["condition_window_index"], output, valid=True, metadata=metadata,
                    )
                    extracted["video"] += 1

            if attention_video_inputs:
                encoder = self._load_video_encoder()
                outputs = self._encode_batches([item[1] for item in attention_video_inputs], encoder)
                for (metadata, _), output in zip(attention_video_inputs, outputs):
                    _save_window_tensor(
                        self.window_cache_root, "attention_video", participant, metadata["condition"],
                        metadata["condition_window_index"], output, valid=True, metadata=metadata,
                    )
                    extracted["attention_video"] += 1

        modality_payloads: dict[str, dict[str, Any]] = {}
        for modality in self.modalities:
            payload: dict[str, Any] = {
                "source": "fresh_extraction",
                "run_tag": self.run_tag,
                "extracted_valid_windows": extracted[modality],
                "invalid_windows": invalid[modality],
            }
            if modality == "eeg":
                payload.update(
                    {
                        "channel_contract": relax_eeg_contract_payload(),
                        "xdf_columns": list(RELAX_EEG_XDF_COLUMNS),
                        "reve_montage": list(RELAX_EEG_MONTAGE),
                        "reve_model": self.reve_model_name,
                        "reve_model_revision": self._reve_model_revision,
                        "reve_positions": self.reve_positions_name,
                        "reve_positions_revision": self._reve_positions_revision,
                        "output_embedding_dim": RELAX_EEG_EMBED_DIM,
                        "preprocessing": "10s; source_rate_to_200hz; per_channel_zscore",
                    }
                )
            if modality == "attention_video":
                payload["attention_video_config"] = self.attention_video.config.__dict__
            modality_payloads[modality] = payload
        manifest = _merge_window_manifest(self.window_cache_root, modality_payloads)
        manifest["relax_run_dir"] = str(self.paths.root)
        write_json(self.window_cache_root / "window_embedding_manifest.json", manifest)
        return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relax-run-dir", required=True)
    parser.add_argument(
        "--window-cache-root",
        default="artifacts/relax/window_embeddings_reve_large",
        help="Root containing or receiving per-window modality tensors.",
    )
    parser.add_argument("--output", default="artifacts/relax/condition_embeddings.pt")
    parser.add_argument("--modalities", nargs="+", default=list(MODALITIES))
    parser.add_argument("--raw-root", default=None, help="Optional raw root for Phase 0 source checks.")
    parser.add_argument("--labels-root", default=None, help="Optional labels root for Phase 0 workbook checks.")
    parser.add_argument("--eeg-model", default="reve", choices=["reve"])
    parser.add_argument("--eeg-weights", default=RELAX_REVE_MODEL)
    parser.add_argument("--eeg-pos-bank", default=RELAX_REVE_POSITIONS)
    parser.add_argument("--run-tag", default=RELAX_EEG_RUN_TAG)
    parser.add_argument("--extract-window-embeddings", action="store_true")
    parser.add_argument("--skip-assemble", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--participants", nargs="*", default=None)
    parser.add_argument("--relax-model-src", default="/home/link/Wei/Models/core/Relax-Model/src")
    parser.add_argument("--ecg-weights", default="weights/ecgfounder/1_lead_ECGFounder.pth")
    parser.add_argument("--eye-weights", default="checkpoints/inceptiontime_gaze_pretrained.pt")
    parser.add_argument("--attention-hfov-deg", type=float, default=90.0)
    parser.add_argument("--attention-max-gaze-lag-ms", type=float, default=150.0)
    parser.add_argument("--attention-min-valid-frames", type=int, default=12)
    parser.add_argument("--attention-sigma-fraction", type=float, default=0.10)
    parser.add_argument("--attention-periphery-scale", type=float, default=0.35)
    parser.add_argument("--allow-missing-window-files", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Kept for CLI parity; extraction is strict by default.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        modalities = assert_relax_modalities(args.modalities)
        validate_phase0_inputs(
            args.relax_run_dir,
            raw_root=args.raw_root,
            labels_root=args.labels_root,
            eeg_weights=args.eeg_weights if "eeg" in modalities else None,
            eeg_pos_bank=args.eeg_pos_bank if "eeg" in modalities else None,
            require_reve="eeg" in modalities,
        )
        if args.extract_window_embeddings:
            device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
            extraction_manifest = RelaxWindowEmbeddingExtractor(
                relax_run_dir=args.relax_run_dir,
                window_cache_root=args.window_cache_root,
                modalities=modalities,
                device=device,
                batch_size=args.batch_size,
                reve_model=args.eeg_weights,
                reve_positions=args.eeg_pos_bank,
                relax_model_src=args.relax_model_src,
                ecg_weights=args.ecg_weights,
                eye_weights=args.eye_weights,
                attention_config=AttentionVideoConfig(
                    hfov_deg=args.attention_hfov_deg,
                    max_gaze_lag_ms=args.attention_max_gaze_lag_ms,
                    min_valid_projected_frames=args.attention_min_valid_frames,
                    sigma_fraction=args.attention_sigma_fraction,
                    periphery_scale=args.attention_periphery_scale,
                ),
                participants=args.participants,
                run_tag=args.run_tag,
            ).run()
            print(json.dumps(extraction_manifest, indent=2))
        if args.skip_assemble:
            return 0
        manifest = assemble_condition_cache(
            relax_run_dir=args.relax_run_dir,
            window_cache_root=args.window_cache_root,
            modalities=modalities,
            output_path=args.output,
            allow_missing_window_files=args.allow_missing_window_files,
            eeg_weights=args.eeg_weights,
            eeg_pos_bank=args.eeg_pos_bank,
            run_tag=args.run_tag,
        )
        print(json.dumps(manifest, indent=2))
        return 0
    except RelaxHardFailure as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

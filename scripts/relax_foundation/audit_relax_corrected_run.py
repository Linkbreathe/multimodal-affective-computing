#!/usr/bin/env python
"""Strict acceptance audit for the corrected Relax EEG mapping artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mac.data.relax_foundation import (  # noqa: E402
    EEG_DISABLED_PARTICIPANTS,
    RELAX_EEG_MONTAGE,
    RELAX_EEG_RUN_TAG,
    RELAX_EEG_XDF_COLUMNS,
)


def _tensor_item(path: Path) -> tuple[torch.Tensor, bool, dict[str, Any]]:
    item = torch.load(path, map_location="cpu", weights_only=False)
    key = "embedding" if "embedding" in item else "signal"
    return torch.as_tensor(item[key]).float(), bool(item.get("valid", True)), dict(item.get("metadata", {}))


def _window_key(path: Path, modality_root: Path) -> tuple[str, str]:
    relative = path.relative_to(modality_root)
    return relative.parts[0], relative.stem


def audit(
    *,
    old_window_root: Path,
    new_run_root: Path,
    relax_model_run: Path,
    output_dir: Path,
) -> dict[str, Any]:
    window_root = new_run_root / "window_embeddings"
    standard_cache = new_run_root / "condition_embeddings.pt"
    attention_cache = new_run_root / "condition_embeddings_attention_video.pt"
    copy_manifest = json.loads((window_root / "cache_copy_manifest.json").read_text(encoding="utf-8"))
    window_manifest = json.loads((window_root / "window_embedding_manifest.json").read_text(encoding="utf-8"))

    checksum_records = []
    for modality in ("ecg", "eye", "head", "video", "attention_video"):
        payload = copy_manifest["modalities"][modality]
        if payload["file_count"] != 946:
            raise RuntimeError(f"{modality}: expected 946 copied files, got {payload['file_count']}")
        for record in payload["files"]:
            source = old_window_root / record["path"]
            destination = window_root / record["path"]
            if not source.exists() or not destination.exists() or source.read_bytes() != destination.read_bytes():
                raise RuntimeError(f"Copied tensor is not byte-identical: {record['path']}")
            checksum_records.append(
                {
                    "modality": modality,
                    "path": record["path"],
                    "sha256": record["sha256"],
                    "identical": True,
                }
            )

    eeg_files = sorted((window_root / "eeg").rglob("*.pt"))
    if len(eeg_files) != 567:
        raise RuntimeError(f"Expected 567 corrected EEG tensors, got {len(eeg_files)}")
    drift_records = []
    for new_path in eeg_files:
        new_tensor, valid, metadata = _tensor_item(new_path)
        if not valid or new_tensor.shape != (1024,) or not torch.isfinite(new_tensor).all():
            raise RuntimeError(f"Invalid corrected EEG tensor: {new_path}")
        if metadata.get("reve_montage") != list(RELAX_EEG_MONTAGE):
            raise RuntimeError(f"Incorrect REVE montage provenance: {new_path}")
        if metadata.get("xdf_columns") != list(RELAX_EEG_XDF_COLUMNS):
            raise RuntimeError(f"Incorrect XDF column provenance: {new_path}")
        if metadata.get("run_tag") != RELAX_EEG_RUN_TAG or not metadata.get("source_xdf"):
            raise RuntimeError(f"Incomplete corrected EEG provenance: {new_path}")
        relative = new_path.relative_to(window_root / "eeg")
        old_tensor, old_valid, _ = _tensor_item(old_window_root / "eeg" / relative)
        if not old_valid:
            raise RuntimeError(f"Legacy EEG comparison tensor unexpectedly invalid: {relative}")
        old_reduced = F.adaptive_avg_pool1d(old_tensor.view(1, 1, -1), 1024).view(-1)
        cosine = float(F.cosine_similarity(old_reduced, new_tensor, dim=0))
        l2 = float(torch.linalg.vector_norm(old_reduced - new_tensor))
        drift_records.append(
            {
                "participant_id": relative.parts[0],
                "condition": relative.stem.split("_", 1)[0],
                "window": relative.stem,
                "cosine_similarity": cosine,
                "l2_drift": l2,
                "exactly_equal": bool(torch.equal(old_reduced, new_tensor)),
            }
        )

    drift = pd.DataFrame(drift_records)
    if drift["exactly_equal"].all() or not np.isfinite(drift[["cosine_similarity", "l2_drift"]]).all().all():
        raise RuntimeError("Corrected EEG embeddings did not exhibit finite, non-zero mapping drift")

    invalid_keys: dict[str, list[tuple[str, str]]] = {}
    for modality in ("ecg", "eye", "head", "video", "attention_video"):
        modality_root = window_root / modality
        invalid = []
        for path in sorted(modality_root.rglob("*.pt")):
            _, valid, _ = _tensor_item(path)
            if not valid:
                invalid.append(_window_key(path, modality_root))
        invalid_keys[modality] = invalid
    non_eeg_anomaly_keys = set().union(*(set(values) for values in invalid_keys.values()))
    if len(non_eeg_anomaly_keys) != 1:
        raise RuntimeError(f"Expected one unique non-EEG anomalous window, got {sorted(non_eeg_anomaly_keys)}")

    cache_summaries: dict[str, Any] = {}
    for name, path, expected_modalities in (
        ("standard", standard_cache, {"ecg", "eeg", "eye", "head", "video"}),
        (
            "attention_video",
            attention_cache,
            {"ecg", "eeg", "eye", "head", "video", "attention_video"},
        ),
    ):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if len(payload["samples"]) != 135 or set(payload["modalities"]) != expected_modalities:
            raise RuntimeError(f"Invalid {name} condition cache sample or modality count")
        valid_eeg = sum(int(sample["masks"]["eeg"].sum()) for sample in payload["samples"])
        missing_eeg = sum(int((~sample["masks"]["eeg"]).sum()) for sample in payload["samples"])
        disabled_mask = sum(
            int((~sample["masks"]["eeg"]).sum())
            for sample in payload["samples"]
            if sample["participant_id"] in EEG_DISABLED_PARTICIPANTS
        )
        if (valid_eeg, missing_eeg, disabled_mask) != (567, 379, 379):
            raise RuntimeError(
                f"{name}: EEG mask counts {(valid_eeg, missing_eeg, disabled_mask)} != (567,379,379)"
            )
        cache_summaries[name] = {
            "samples": len(payload["samples"]),
            "valid_eeg_windows": valid_eeg,
            "missing_eeg_windows": missing_eeg,
            "modalities": payload["modalities"],
        }

    window_features = pd.read_csv(relax_model_run / "features" / "window_features.csv")
    feature_columns = set(window_features.columns)
    required_names = {f"eeg_{name.lower()}_alpha_power" for name in RELAX_EEG_MONTAGE}
    forbidden_names = {"eeg_t7_alpha_power", "eeg_t8_alpha_power", "eeg_tp7_alpha_power", "eeg_tp8_alpha_power"}
    if not required_names.issubset(feature_columns) or feature_columns & forbidden_names:
        raise RuntimeError("Corrected handcrafted feature names failed the EEG mapping contract")

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(checksum_records).to_csv(output_dir / "non_eeg_checksum_registry.csv", index=False)
    drift.to_csv(output_dir / "eeg_embedding_drift.csv", index=False)
    participant_condition = (
        drift.groupby(["participant_id", "condition"], as_index=False)
        .agg(
            cosine_similarity_mean=("cosine_similarity", "mean"),
            cosine_similarity_std=("cosine_similarity", "std"),
            l2_drift_mean=("l2_drift", "mean"),
            l2_drift_std=("l2_drift", "std"),
        )
    )
    participant_condition.to_csv(output_dir / "eeg_embedding_drift_participant_condition.csv", index=False)

    summary = {
        "status": "ok",
        "run_tag": RELAX_EEG_RUN_TAG,
        "channel_contract": {
            "xdf_columns": list(RELAX_EEG_XDF_COLUMNS),
            "reve_montage": list(RELAX_EEG_MONTAGE),
        },
        "window_manifest_modalities": sorted(window_manifest["modalities"]),
        "copied_non_eeg_tensors": len(checksum_records),
        "non_eeg_checksums_identical": True,
        "corrected_eeg_tensors": len(eeg_files),
        "eeg_drift": {
            "cosine_mean": float(drift["cosine_similarity"].mean()),
            "cosine_min": float(drift["cosine_similarity"].min()),
            "cosine_max": float(drift["cosine_similarity"].max()),
            "l2_mean": float(drift["l2_drift"].mean()),
            "l2_min": float(drift["l2_drift"].min()),
            "l2_max": float(drift["l2_drift"].max()),
            "exactly_equal_windows": int(drift["exactly_equal"].sum()),
        },
        "condition_caches": cache_summaries,
        "non_eeg_invalid_windows": {name: len(values) for name, values in invalid_keys.items()},
        "unique_non_eeg_anomaly": [list(value) for value in sorted(non_eeg_anomaly_keys)],
        "handcrafted_feature_names": sorted(required_names),
    }
    (output_dir / "acceptance_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-window-root", required=True, type=Path)
    parser.add_argument("--new-run-root", required=True, type=Path)
    parser.add_argument("--relax-model-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    summary = audit(
        old_window_root=args.old_window_root,
        new_run_root=args.new_run_root,
        relax_model_run=args.relax_model_run,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

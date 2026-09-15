"""Shared participant-fold contracts for cross-project experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class AlignedFold:
    """One fixed participant split: N fit, one validation, one test."""

    fold_index: int
    train_participants: tuple[str, ...]
    validation_participant: str
    test_participant: str


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split_manifest(
    path: str | Path,
    *,
    expected_participants: Iterable[str] | None = None,
    expected_train_count: int = 13,
) -> list[AlignedFold]:
    """Load and exhaustively validate the shared long-form split manifest."""

    import pandas as pd

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Alignment split manifest not found: {source}")
    frame = pd.read_csv(source, dtype=str)
    required = {
        "fold_index",
        "test_participant",
        "validation_participant",
        "participant_id",
        "role",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Split manifest lacks required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Split manifest is empty")
    frame["fold_index"] = pd.to_numeric(frame["fold_index"], errors="raise").astype(int)
    frame["role"] = frame["role"].str.strip().str.lower()
    if not set(frame["role"]).issubset({"train", "validation", "test"}):
        raise ValueError(f"Split manifest has unknown roles: {sorted(set(frame['role']) - {'train', 'validation', 'test'})}")

    expected = (
        tuple(sorted(str(item) for item in expected_participants))
        if expected_participants is not None
        else None
    )
    folds: list[AlignedFold] = []
    test_counts: dict[str, int] = {}
    validation_counts: dict[str, int] = {}
    participant_set: tuple[str, ...] | None = None
    fold_indexes = sorted(frame["fold_index"].unique().tolist())
    if fold_indexes != list(range(1, len(fold_indexes) + 1)):
        raise ValueError(f"Split manifest fold indexes must be consecutive from 1: {fold_indexes}")

    for fold_index in fold_indexes:
        group = frame.loc[frame["fold_index"] == fold_index].copy()
        if group["participant_id"].duplicated().any():
            duplicates = sorted(group.loc[group["participant_id"].duplicated(), "participant_id"].unique())
            raise ValueError(f"Fold {fold_index} repeats participants: {duplicates}")
        participants = tuple(sorted(group["participant_id"].astype(str)))
        if participant_set is None:
            participant_set = participants
        elif participants != participant_set:
            raise ValueError(f"Fold {fold_index} participant membership differs from fold 1")
        if expected is not None and participants != expected:
            raise ValueError(f"Fold {fold_index} participants do not equal the expected cohort")

        role_members = {
            role: tuple(sorted(group.loc[group["role"] == role, "participant_id"].astype(str)))
            for role in ("train", "validation", "test")
        }
        if len(role_members["validation"]) != 1 or len(role_members["test"]) != 1:
            raise ValueError(f"Fold {fold_index} must have exactly one validation and one test participant")
        if len(role_members["train"]) != expected_train_count:
            raise ValueError(
                f"Fold {fold_index} must have {expected_train_count} training participants; "
                f"found {len(role_members['train'])}"
            )
        validation = role_members["validation"][0]
        test = role_members["test"][0]
        declared_tests = tuple(sorted(group["test_participant"].dropna().astype(str).unique()))
        declared_validations = tuple(sorted(group["validation_participant"].dropna().astype(str).unique()))
        if declared_tests != (test,) or declared_validations != (validation,):
            raise ValueError(f"Fold {fold_index} header roles disagree with its participant rows")
        if len(participants) != expected_train_count + 2:
            raise ValueError(f"Fold {fold_index} must contain {expected_train_count + 2} participants")

        test_counts[test] = test_counts.get(test, 0) + 1
        validation_counts[validation] = validation_counts.get(validation, 0) + 1
        folds.append(
            AlignedFold(
                fold_index=int(fold_index),
                train_participants=role_members["train"],
                validation_participant=validation,
                test_participant=test,
            )
        )

    assert participant_set is not None
    if len(folds) != len(participant_set):
        raise ValueError(
            f"LOPO manifest must have one fold per participant; found {len(folds)} folds and "
            f"{len(participant_set)} participants"
        )
    if any(test_counts.get(participant, 0) != 1 for participant in participant_set):
        raise ValueError("Every participant must be the test participant exactly once")
    if any(validation_counts.get(participant, 0) != 1 for participant in participant_set):
        raise ValueError("Every participant must be the validation participant exactly once")
    return folds


def indexes_for_fold(
    participant_ids: Iterable[str],
    fold: AlignedFold,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map a participant fold to row indexes in train/validation/test order."""

    groups = np.asarray([str(value) for value in participant_ids], dtype=str)
    train = np.flatnonzero(np.isin(groups, fold.train_participants))
    validation = np.flatnonzero(groups == fold.validation_participant)
    test = np.flatnonzero(groups == fold.test_participant)
    observed_train = tuple(sorted(set(groups[train])))
    if observed_train != fold.train_participants:
        raise ValueError(
            f"Fold {fold.fold_index} data are missing training participants: "
            f"{sorted(set(fold.train_participants) - set(observed_train))}"
        )
    if not len(validation) or not len(test):
        raise ValueError(f"Fold {fold.fold_index} data lack validation or test rows")
    if set(groups[validation]) != {fold.validation_participant} or set(groups[test]) != {fold.test_participant}:
        raise AssertionError(f"Fold {fold.fold_index} role-index mapping failed")
    return train.astype(int), validation.astype(int), test.astype(int)


def validate_alignment_contract(contract_dir: str | Path) -> dict[str, Any]:
    """Validate immutable files and counts recorded by the contract builder."""

    root = Path(contract_dir)
    manifest_path = root / "contract.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Alignment contract not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    supported = {
        "cross_project_alignment_v1",
        "eeg_eligible_modality_ablation_contract_v1",
    }
    if payload.get("schema_version") not in supported:
        raise ValueError(
            f"Unsupported alignment contract schema: {payload.get('schema_version')!r}"
        )
    for name, record in payload.get("files", {}).items():
        path = root / str(record["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Contract file {name!r} is missing: {path}")
        actual = file_sha256(path)
        if actual != record.get("sha256"):
            raise ValueError(f"Contract file hash mismatch for {name}: {actual}")
    fold_contract = payload.get("fold_contract", {})
    expected_train_count = int(fold_contract.get("train_participants", 13))
    folds = load_split_manifest(
        root / payload["files"]["split_manifest"]["path"],
        expected_participants=payload["participants"],
        expected_train_count=expected_train_count,
    )
    expected_fold_count = int(fold_contract.get("folds", len(payload["participants"])))
    if len(folds) != expected_fold_count:
        raise ValueError(
            f"Contract declares {expected_fold_count} folds but its manifest contains {len(folds)}"
        )
    return payload


def write_alignment_manifest(
    output_path: str | Path,
    *,
    split_manifest: str | Path,
    modality_masks: str | Path,
    labels: str | Path,
    windows: str | Path,
    window_features: str | Path | None = None,
    contract: str | Path | None = None,
    seed: int,
    folds: list[AlignedFold],
    runtime: dict[str, Any] | None = None,
) -> Path:
    """Write the exact alignment inputs consumed by one model run."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sources: dict[str, Path] = {
        "split_manifest": Path(split_manifest),
        "modality_masks": Path(modality_masks),
        "labels": Path(labels),
        "windows": Path(windows),
    }
    if window_features is not None:
        sources["window_features"] = Path(window_features)
    if contract is not None:
        sources["contract"] = Path(contract)
    payload = {
        "schema_version": "cross_project_alignment_run_v1",
        "seed": int(seed),
        "sources": {
            name: {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for name, path in sources.items()
        },
        "folds": [asdict(fold) for fold in folds],
        "runtime": runtime or {},
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


__all__ = [
    "AlignedFold",
    "file_sha256",
    "indexes_for_fold",
    "load_split_manifest",
    "validate_alignment_contract",
    "write_alignment_manifest",
]

"""RQ2 WSL cached-representation, fusion, and reporting pipeline.

This module consumes the immutable Windows hand-off and the already-saved WSL
condition cache.  It never imports or instantiates a pretrained encoder.  All
learned downstream state is fitted inside participant folds.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
import importlib.metadata
import itertools
import json
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Mapping, Sequence

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

from scripts.run_rq2_wsl import (
    CACHE_ROOT,
    EXPECTED_DIMENSIONS,
    EXPECTED_FALLBACK,
    EXPECTED_FOLDS,
    EXPECTED_OBSERVATIONS,
    EXPECTED_SEEDS,
    MODALITIES,
    PROJECT_ROOT,
    RESULTS_ROOT,
    SHARED_ROOT,
    _key_frame,
    _load_table,
    _read_json,
    file_sha256,
    preflight,
)


TARGETS = ("relaxation", "discomfort")
MODALITY_ORDER = ("EEG", "ECG", "Eye", "Head", "Video")
MODALITY_LOWER = {modality: modality.lower() for modality in MODALITY_ORDER}


def _canonical_modality(value: Any) -> str:
    normalized = str(value).strip().lower()
    mapping = {"eeg": "EEG", "ecg": "ECG", "eye": "Eye", "head": "Head", "video": "Video"}
    if normalized not in mapping:
        raise ValueError(f"Unknown modality name: {value}")
    return mapping[normalized]
PCA_ALLOCATION = {"EEG": 3, "ECG": 4, "Eye": 1, "Head": 1, "Video": 3}
RIDGE_ALPHAS = (10.0, 100.0, 1000.0)
GAMMAS = (0.25, 0.5, 0.75, 1.0)
WEIGHT_FLOOR = 0.02
CORRECTION_CLIP = (-0.2, 0.2)
PREDICTION_CLIP = (0.0, 1.0)
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260705


def _hash_files(paths: Sequence[Path]) -> str:
    digest = sha256()
    for path in paths:
        digest.update(str(path).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")


def _bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    return values.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))


@dataclass(frozen=True)
class FoldSpec:
    fold: int
    train_participants: tuple[str, ...]
    validation_participant: str
    test_participant: str
    train_idx: np.ndarray
    validation_idx: np.ndarray
    test_idx: np.ndarray


@dataclass
class ContractData:
    root: Path
    contract_hash: str
    labels: pd.DataFrame
    windows: pd.DataFrame
    anchors: dict[tuple[int, str, str], float]
    folds: list[FoldSpec]
    participants: tuple[str, ...]
    conditions: tuple[str, ...]
    contract_payload: dict[str, Any]
    windows_index: pd.DataFrame


@dataclass
class FamilyData:
    family: str
    matrices: dict[tuple[int, int, str], np.ndarray]
    present: dict[tuple[int, int, str], np.ndarray]
    dimensions: dict[str, int]
    valid_window_counts: dict[tuple[str, str], int]
    provenance: dict[str, Any]


@dataclass
class Preprocessor:
    modality: str
    imputer: SimpleImputer
    scaler: StandardScaler
    pca: PCA
    input_dimension: int
    output_dimension: int

    def transform(self, values: np.ndarray, present: np.ndarray) -> np.ndarray:
        output = np.zeros((len(values), self.output_dimension), dtype=np.float64)
        indexes = np.flatnonzero(present)
        if len(indexes):
            transformed = self.scaler.transform(self.imputer.transform(values[indexes]))
            output[indexes] = self.pca.transform(transformed)
        return output


def load_contract(shared_root: Path, state: Any) -> ContractData:
    contract_dir = shared_root / "contract"
    payload = _read_json(contract_dir / "rq2_contract.json")
    labels = pd.read_csv(contract_dir / "condition_manifest.csv")
    labels = labels.rename(
        columns={
            "participant": "participant",
            "normalized_relaxation": "relaxation",
            "normalized_discomfort": "discomfort",
            "presentation_order": "presentation_position",
        }
    )
    participants = tuple(str(value) for value in payload["cohort"]["participant_order"])
    conditions = tuple(str(value) for value in payload["cohort"]["conditions"])
    labels["participant"] = labels["participant"].astype(str)
    labels["condition"] = labels["condition"].astype(str)
    labels = labels.set_index(["participant", "condition"]).loc[
        [(participant, condition) for participant in participants for condition in conditions]
    ].reset_index()
    labels["relaxation"] = labels["relaxation"].astype(float)
    labels["discomfort"] = labels["discomfort"].astype(float)
    windows = pd.read_csv(contract_dir / "window_manifest.csv")
    windows["participant"] = windows["participant"].astype(str)
    windows["condition"] = windows["condition"].astype(str)
    windows["window_id"] = windows["window_id"].astype(str)
    windows["condition_window_index"] = windows["condition_window_index"].astype(int)

    anchors_frame = pd.read_csv(contract_dir / "condition_anchors.csv")
    anchors = {
        (int(row.fold_index), str(row.condition), str(row.target)): float(row.condition_anchor)
        for row in anchors_frame.itertuples(index=False)
    }
    participant_array = labels["participant"].to_numpy(dtype=str)
    folds_frame = pd.read_csv(contract_dir / "folds.csv")
    folds: list[FoldSpec] = []
    for fold_id, group in folds_frame.groupby("fold_index", sort=True):
        group = group.copy()
        train_participants = tuple(str(value) for value in group.loc[group.role.eq("train"), "participant"])
        validation = str(group.loc[group.role.eq("validation"), "participant"].iloc[0])
        test = str(group.loc[group.role.eq("test"), "participant"].iloc[0])
        folds.append(
            FoldSpec(
                int(fold_id),
                train_participants,
                validation,
                test,
                np.flatnonzero(np.isin(participant_array, train_participants)),
                np.flatnonzero(participant_array == validation),
                np.flatnonzero(participant_array == test),
            )
        )
    windows_index = pd.read_csv(shared_root / "windows_results/windows_representation_index.csv")
    return ContractData(
        root=shared_root,
        contract_hash=str(state.evidence["contract_hash"]),
        labels=labels,
        windows=windows,
        anchors=anchors,
        folds=folds,
        participants=participants,
        conditions=conditions,
        contract_payload=payload,
        windows_index=windows_index,
    )


def _window_counts(contract: ContractData) -> dict[tuple[str, str], int]:
    return {
        (str(key[0]), str(key[1])): int(value)
        for key, value in contract.windows.groupby(["participant", "condition"]).size().items()
    }


def _cache_pretrained(contract: ContractData, cache_root: Path) -> tuple[FamilyData, pd.DataFrame, dict[str, Any]]:
    cache_path = cache_root / "condition_embeddings.pt"
    cache_manifest = cache_root / "cache_manifest.json"
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    participant_ids = [str(value) for value in cache["participant_ids"]]
    conditions = [str(value) for value in cache["conditions"]]
    lookup = {(participant, condition): index for index, (participant, condition) in enumerate(zip(participant_ids, conditions, strict=True))}
    if len(lookup) != len(participant_ids):
        raise ValueError("WSL condition cache contains duplicate participant-condition keys")
    window_counts = _window_counts(contract)
    matrices: dict[str, np.ndarray] = {}
    present: dict[str, np.ndarray] = {}
    audit_rows: list[dict[str, Any]] = []
    n = len(contract.labels)
    metadata = dict(cache.get("metadata", {}))
    for modality in MODALITY_ORDER:
        lower = MODALITY_LOWER[modality]
        values = np.asarray(cache["embeddings"][lower], dtype=np.float64)
        masks = np.asarray(cache["masks"][lower], dtype=bool)
        expected_dim = EXPECTED_DIMENSIONS[lower]
        if values.ndim != 3 or values.shape[2] != expected_dim:
            raise ValueError(f"Unexpected {modality} cache shape {values.shape}; expected * * {expected_dim}")
        matrix = np.full((n, expected_dim), np.nan, dtype=np.float64)
        available = np.zeros(n, dtype=bool)
        for row_index, row in contract.labels.iterrows():
            key = (str(row.participant), str(row.condition))
            cache_index = lookup.get(key)
            valid_windows = contract.windows.loc[
                contract.windows["participant"].eq(key[0]) & contract.windows["condition"].eq(key[1])
            ]
            local_indexes = valid_windows["condition_window_index"].to_numpy(dtype=int)
            fallback = key == EXPECTED_FALLBACK
            if cache_index is None:
                raise ValueError(f"WSL cache lacks contract key {key}")
            if fallback:
                local_indexes = np.asarray([], dtype=int)
            if np.any(local_indexes >= values.shape[1]):
                raise ValueError(f"Cache window slots do not cover {key} for {modality}")
            effective = masks[cache_index, local_indexes] if len(local_indexes) else np.asarray([], dtype=bool)
            if len(effective) and not np.isfinite(values[cache_index, local_indexes][effective]).all():
                raise ValueError(f"Non-finite valid cached values for {key}/{modality}")
            if len(effective) and effective.any():
                matrix[row_index] = values[cache_index, local_indexes][effective].mean(axis=0)
                available[row_index] = True
            audit_rows.append(
                {
                    "representation_family": "frozen_pretrained",
                    "modality": modality,
                    "fold": "global",
                    "seed": "deterministic",
                    "participant": key[0],
                    "condition": key[1],
                    "representation_dimension": expected_dim,
                    "valid_window_count": int(effective.sum()),
                    "representation_available": bool(available[row_index]),
                    "fallback_used": bool(fallback),
                    "encoder_checkpoint_id": metadata.get(f"{lower}_model", ""),
                    "feature_or_encoder_version": json.dumps(metadata, sort_keys=True, default=_json_default),
                    "source_hash": file_sha256(cache_manifest) if cache_manifest.is_file() else "",
                    "cache_path": str(cache_path),
                }
            )
        matrices[modality] = matrix
        present[modality] = available
    fallback_rows = (contract.labels["participant"].eq(EXPECTED_FALLBACK[0]) & contract.labels["condition"].eq(EXPECTED_FALLBACK[1])).to_numpy()
    if any(bool(present[modality][fallback_rows].any()) for modality in MODALITY_ORDER):
        raise ValueError("P004/C6 must remain unavailable in every pretrained modality")
    audit = pd.DataFrame(audit_rows)
    family = FamilyData(
        family="pretrained",
        matrices={(0, 0, modality): matrices[modality] for modality in MODALITY_ORDER},
        present={(0, 0, modality): present[modality] for modality in MODALITY_ORDER},
        dimensions={modality: matrices[modality].shape[1] for modality in MODALITY_ORDER},
        valid_window_counts=window_counts,
        provenance={
            "cache_path": str(cache_path),
            "cache_sha256": file_sha256(cache_path),
            "cache_manifest": str(cache_manifest),
            "cache_manifest_sha256": file_sha256(cache_manifest) if cache_manifest.is_file() else "",
            "metadata": metadata,
            "aggregation": "mean_over_valid_windows",
            "join": "participant+condition+condition_window_index mapped to contract window_id",
        },
    )
    return family, audit, family.provenance


def _load_npz(path: Path, expected_family: str, expected_modality: str, expected_dimension: int, expected_hash: str) -> tuple[dict[tuple[str, str], np.ndarray], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != expected_hash:
        raise ValueError(f"Artifact hash mismatch: {path}")
    data = np.load(path, allow_pickle=True)
    required = {"participants", "conditions", "vectors", "feature_names"}
    if not required.issubset(data.files):
        raise ValueError(f"NPZ {path} lacks {sorted(required - set(data.files))}")
    participants = [str(value) for value in data["participants"].tolist()]
    conditions = [str(value) for value in data["conditions"].tolist()]
    vectors = np.asarray(data["vectors"], dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != expected_dimension:
        raise ValueError(f"Unexpected vector shape {vectors.shape} for {path}")
    if not np.isfinite(vectors).all():
        raise ValueError(f"Non-finite vector in {path}")
    if len(set(zip(participants, conditions, strict=True))) != len(participants):
        raise ValueError(f"Duplicate NPZ keys in {path}")
    if "representation_family" in data and str(data["representation_family"].reshape(-1)[0]) != expected_family:
        raise ValueError(f"Family metadata mismatch in {path}")
    if "modality" in data and str(data["modality"].reshape(-1)[0]).lower() != expected_modality.lower():
        raise ValueError(f"Modality metadata mismatch in {path}")
    return {(participant, condition): vectors[index] for index, (participant, condition) in enumerate(zip(participants, conditions, strict=True))}, [str(value) for value in data["feature_names"].tolist()]


def _load_windows_family(contract: ContractData, family_name: str) -> FamilyData:
    index = contract.windows_index.copy()
    index["representation_family"] = index["representation_family"].astype(str).str.lower()
    index["modality"] = index["modality"].map(_canonical_modality)
    index["representation_available"] = _bool_series(index["representation_available"])
    if family_name == "handcrafted":
        groups = index[index["representation_family"].eq("handcrafted")]
    else:
        groups = index[index["representation_family"].eq("temporal_1dcnn")]
    matrices: dict[tuple[int, int, str], np.ndarray] = {}
    present: dict[tuple[int, int, str], np.ndarray] = {}
    dimensions: dict[str, int] = {}
    provenance: dict[str, Any] = {"artifacts": {}}
    key_order = list(zip(contract.labels["participant"], contract.labels["condition"], strict=True))
    key_positions = {key: position for position, key in enumerate(key_order)}
    for group_key, group in groups.groupby(["fold", "seed"], sort=True):
        fold_text, seed_text = str(group_key[0]), str(group_key[1])
        fold = 0 if fold_text == "global" else int(fold_text)
        seed = 0 if seed_text == "deterministic" else int(seed_text)
        for modality in MODALITY_ORDER:
            rows = group[group["modality"].eq(modality)]
            if len(rows) != EXPECTED_OBSERVATIONS:
                raise ValueError(f"{family_name}/{group_key}/{modality} lacks 81 index rows")
            artifact_paths = sorted(set(str(value) for value in rows["artifact_path"]))
            if len(artifact_paths) != 1:
                raise ValueError(f"Expected one artifact for {family_name}/{group_key}/{modality}")
            artifact_path = contract.root / artifact_paths[0]
            expected_hash = str(rows["artifact_sha256"].iloc[0])
            dimension = int(rows["representation_dimension"].iloc[0])
            vectors, feature_names = _load_npz(artifact_path, family_name, modality, dimension, expected_hash)
            matrix = np.full((EXPECTED_OBSERVATIONS, dimension), np.nan, dtype=np.float64)
            available = np.zeros(EXPECTED_OBSERVATIONS, dtype=bool)
            for row in rows.itertuples(index=False):
                key = (str(row.participant), str(row.condition))
                position = key_positions[key]
                if bool(row.representation_available):
                    if key not in vectors:
                        raise ValueError(f"Index says available but artifact lacks {key}: {artifact_path}")
                    matrix[position] = vectors[key]
                    available[position] = True
                else:
                    if key != EXPECTED_FALLBACK:
                        raise ValueError(f"Only P004/C6 may be unavailable: {family_name}/{group_key}/{key}")
            matrices[(fold, seed, modality)] = matrix
            present[(fold, seed, modality)] = available
            dimensions[modality] = dimension
            provenance["artifacts"][f"{fold_text}/{seed_text}/{modality}"] = {
                "path": artifact_paths[0],
                "sha256": expected_hash,
                "feature_names": feature_names,
                "dimension": dimension,
            }
    return FamilyData(family_name, matrices, present, dimensions, {}, provenance)


def _materialize_pretrained(contract: ContractData, family: FamilyData, output_root: Path, audit: pd.DataFrame) -> pd.DataFrame:
    artifact_root = output_root / "pretrained_representation_artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    key_order = list(zip(contract.labels["participant"], contract.labels["condition"], strict=True))
    for modality in MODALITY_ORDER:
        matrix = family.matrices[(0, 0, modality)]
        available = family.present[(0, 0, modality)]
        usable = matrix[available]
        artifact_path = artifact_root / f"{modality.lower()}.npz"
        np.savez_compressed(
            artifact_path,
            participants=np.asarray([key[0] for index, key in enumerate(key_order) if available[index]], dtype=str),
            conditions=np.asarray([key[1] for index, key in enumerate(key_order) if available[index]], dtype=str),
            feature_names=np.asarray([f"{modality.lower()}_{i:04d}" for i in range(matrix.shape[1])], dtype=str),
            vectors=usable.astype(np.float32),
            representation_family=np.asarray(["pretrained"]),
            modality=np.asarray([modality]),
            feature_or_encoder_version=np.asarray([str(family.provenance["metadata"].get(f"{modality.lower()}_model", ""))]),
        )
        artifact_sha = file_sha256(artifact_path)
        artifact_display = str(artifact_path.relative_to(PROJECT_ROOT)) if artifact_path.is_absolute() and PROJECT_ROOT in artifact_path.parents else str(artifact_path)
        available_positions = {
            key: output_index
            for output_index, key in enumerate(
                key for index, key in enumerate(key_order) if available[index]
            )
        }
        for index, key in enumerate(key_order):
            available_value = bool(available[index])
            row_audit = audit[(audit["modality"].eq(modality)) & audit["participant"].eq(key[0]) & audit["condition"].eq(key[1])].iloc[0]
            rows.append(
                {
                    "contract_hash": contract.contract_hash,
                    "environment": "wsl",
                    "run_id": "wsl_pretrained_aggregate_20260822",
                    "representation_family": "pretrained",
                    "modality": modality,
                    "fold": "global",
                    "seed": "deterministic",
                    "split_role": "global",
                    "participant": key[0],
                    "condition": key[1],
                    "artifact_path": artifact_display,
                    "artifact_row": int(available_positions[key]) if available_value else -1,
                    "representation_dimension": int(matrix.shape[1]),
                    "representation_available": available_value,
                    "valid_window_count": int(row_audit["valid_window_count"]),
                    "fallback_used": not available_value,
                    "encoder_checkpoint_id": str(row_audit["encoder_checkpoint_id"]),
                    "feature_or_encoder_version": str(row_audit["feature_or_encoder_version"]),
                    "artifact_sha256": artifact_sha,
                    "representation_key": f"pretrained+{modality}+global+deterministic+{key[0]}+{key[1]}",
                    "unavailable_reason": "all_missing_common_windows" if not available_value else "",
                }
            )
    return pd.DataFrame(rows)


def _family_key(family: FamilyData, fold: int, seed: int, modality: str) -> tuple[int, int, str]:
    if family.family in ("pretrained", "handcrafted"):
        return (0, 0, modality)
    return (fold, seed, modality)


def _matrices_for(family: FamilyData, fold: int, seed: int, modalities: Sequence[str]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    matrices: dict[str, np.ndarray] = {}
    present: dict[str, np.ndarray] = {}
    for modality in modalities:
        key = _family_key(family, fold, seed, modality)
        matrices[modality] = family.matrices[key]
        present[modality] = family.present[key]
    return matrices, present


def fit_preprocessors(
    matrices: Mapping[str, np.ndarray],
    present: Mapping[str, np.ndarray],
    train_idx: np.ndarray,
    seed_tag: int,
    modalities: Sequence[str],
) -> tuple[dict[str, Preprocessor], dict[str, np.ndarray]]:
    preprocessors: dict[str, Preprocessor] = {}
    scores: dict[str, np.ndarray] = {}
    for modality in modalities:
        values = np.asarray(matrices[modality], dtype=np.float64)
        fit_idx = np.asarray(train_idx)[np.asarray(present[modality])[train_idx]]
        if len(fit_idx) < 3:
            raise ValueError(f"Not enough training observations for {modality} preprocessing")
        imputer = SimpleImputer(strategy="median").fit(values[fit_idx])
        scaled_fit = StandardScaler().fit_transform(imputer.transform(values[fit_idx]))
        components = PCA_ALLOCATION[modality]
        if components > min(scaled_fit.shape[0] - 1, scaled_fit.shape[1]):
            raise ValueError(f"PCA allocation {components} is not feasible for {modality}")
        scaler = StandardScaler().fit(imputer.transform(values[fit_idx]))
        pca = PCA(n_components=components, svd_solver="randomized", random_state=int(seed_tag)).fit(scaler.transform(imputer.transform(values[fit_idx])))
        preprocessor = Preprocessor(modality, imputer, scaler, pca, values.shape[1], components)
        preprocessors[modality] = preprocessor
        scores[modality] = preprocessor.transform(values, np.asarray(present[modality]))
    return preprocessors, scores


def _cross_fitted_anchors(labels: pd.DataFrame, indexes: np.ndarray, target: str) -> np.ndarray:
    output = np.zeros(len(indexes), dtype=np.float64)
    frame = labels.iloc[indexes]
    for local, (global_index, row) in enumerate(frame.iterrows()):
        peers = frame[(frame["condition"] == row["condition"]) & (frame["participant"] != row["participant"])]
        if peers.empty:
            raise ValueError(f"Cannot build cross-fitted anchor for {row['condition']}")
        output[local] = float(peers[target].mean())
    return output


def _heldout_anchor_values(contract: ContractData, fold: int, indexes: np.ndarray, target: str) -> np.ndarray:
    return np.asarray([contract.anchors[(fold, str(condition), target)] for condition in contract.labels.iloc[indexes]["condition"]], dtype=np.float64)


def _solve_simplex(predictions: np.ndarray, truths: np.ndarray) -> np.ndarray:
    predictions = np.asarray(predictions, dtype=np.float64)
    truths = np.asarray(truths, dtype=np.float64)
    active = np.isfinite(predictions).all(axis=1) & np.isfinite(truths)
    predictions = predictions[active]
    truths = truths[active]
    n_experts = predictions.shape[1]
    lower = np.full(n_experts, WEIGHT_FLOOR, dtype=np.float64)
    if not len(predictions):
        return lower + (1.0 - lower.sum()) / n_experts
    initial = lower + (1.0 - lower.sum()) / n_experts
    result = minimize(
        lambda weights: float(np.mean(np.square(truths - predictions @ weights))),
        initial,
        method="SLSQP",
        bounds=[(WEIGHT_FLOOR, 1.0)] * n_experts,
        constraints={"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)},
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(f"Simplex optimisation failed: {result.message}")
    weights = np.maximum(np.asarray(result.x, dtype=np.float64), lower)
    weights = lower + (weights - lower) * ((1.0 - lower.sum()) / max(float((weights - lower).sum()), 1e-12))
    return weights / weights.sum()


def _weighted_correction(corrections: np.ndarray, available: np.ndarray, weights: np.ndarray) -> np.ndarray:
    corrections = np.asarray(corrections, dtype=np.float64)
    available = np.asarray(available, dtype=bool)
    active = available * weights[None, :]
    denominator = active.sum(axis=1)
    output = np.zeros(len(corrections), dtype=np.float64)
    valid = denominator > 0
    output[valid] = (corrections[valid] * active[valid]).sum(axis=1) / denominator[valid]
    return output


def _fit_inner_oof_weights(
    contract: ContractData,
    matrices: Mapping[str, np.ndarray],
    present: Mapping[str, np.ndarray],
    outer_train: np.ndarray,
    target: str,
    alpha: float,
    modalities: Sequence[str],
    seed_tag: int,
) -> tuple[np.ndarray, np.ndarray]:
    participant_values = contract.labels["participant"].to_numpy(dtype=str)
    predictions = np.zeros((len(outer_train), len(modalities)), dtype=np.float64)
    truths = np.zeros(len(outer_train), dtype=np.float64)
    position = {int(index): local for local, index in enumerate(outer_train)}
    for heldout_participant in sorted(set(participant_values[outer_train])):
        heldout = outer_train[participant_values[outer_train] == heldout_participant]
        inner_train = outer_train[participant_values[outer_train] != heldout_participant]
        preprocessors, scores = fit_preprocessors(matrices, present, inner_train, seed_tag + len(heldout_participant), modalities)
        inner_anchor = _cross_fitted_anchors(contract.labels, inner_train, target)
        heldout_anchor = np.asarray(
            [contract.labels.iloc[inner_train][contract.labels.iloc[inner_train]["condition"].eq(condition)][target].mean() for condition in contract.labels.iloc[heldout]["condition"]],
            dtype=np.float64,
        )
        residual = contract.labels.iloc[inner_train][target].to_numpy(dtype=np.float64) - inner_anchor
        local_positions = np.asarray([position[int(index)] for index in heldout], dtype=int)
        truths[local_positions] = contract.labels.iloc[heldout][target].to_numpy(dtype=np.float64) - heldout_anchor
        for modality_index, modality in enumerate(modalities):
            fit_rows = inner_train[present[modality][inner_train]]
            if len(fit_rows) < 3:
                continue
            inner_available = present[modality][inner_train]
            model = Ridge(alpha=alpha).fit(scores[modality][inner_train][inner_available], residual[inner_available])
            correction = model.predict(scores[modality][heldout])
            correction[~present[modality][heldout]] = 0.0
            predictions[local_positions, modality_index] = np.clip(correction, *CORRECTION_CLIP)
    return predictions, truths


def _record_prediction(
    contract: ContractData,
    row_index: int,
    fold: FoldSpec,
    seed: int,
    target: str,
    model_id: str,
    family: str,
    fusion_family: str,
    run_id: str,
    anchor: float,
    predicted_residual: float,
    prediction_before_clip: float,
    final_prediction: float,
    valid_window_count: int,
    fallback_used: bool,
    downstream_hash: str,
    simplex_hash: str,
    comparison_track: str,
) -> dict[str, Any]:
    row = contract.labels.iloc[row_index]
    truth = float(row[target])
    baseline_error = abs(truth - anchor)
    error = abs(truth - final_prediction)
    return {
        "contract_hash": contract.contract_hash,
        "downstream_code_hash": downstream_hash,
        "simplex_code_hash": simplex_hash,
        "run_id": run_id,
        "comparison_track": comparison_track,
        "comparison_scope": "FMQ-9_full5" if "no_video" not in model_id else "FMQ-9_no_video",
        "model_id": model_id,
        "representation_family": family,
        "fusion_family": fusion_family,
        "modality_set": "+".join(MODALITY_ORDER if "no_video" not in model_id else MODALITY_ORDER[:-1]),
        "fold": int(fold.fold),
        "seed": int(seed),
        "target": target,
        "test_participant": fold.test_participant,
        "participant": str(row.participant),
        "condition": str(row.condition),
        "intensity": float(row.intensity),
        "frequency": float(row.frequency),
        "true_rating": truth,
        "condition_anchor": float(anchor),
        "true_residual": truth - float(anchor),
        "predicted_residual": float(predicted_residual),
        "prediction_before_clipping": float(prediction_before_clip),
        "final_prediction": float(final_prediction),
        "condition_only_prediction": float(anchor),
        "model_absolute_error": float(error),
        "baseline_absolute_error": float(baseline_error),
        "improvement_over_baseline": float(baseline_error - error),
        "valid_window_count": int(valid_window_count),
        "fallback_used": bool(fallback_used),
    }


def run_simplex_family(
    contract: ContractData,
    family: FamilyData,
    seed: int,
    model_id: str,
    modalities: Sequence[str],
    downstream_hash: str,
    simplex_hash: str,
    comparison_track: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    counts = _window_counts(contract)
    participant_values = contract.labels["participant"].to_numpy(dtype=str)
    for fold in contract.folds:
        matrices, present = _matrices_for(family, fold.fold, seed, modalities)
        preprocessors, scores = fit_preprocessors(
            matrices,
            present,
            fold.train_idx,
            seed + fold.fold * 1000,
            modalities,
        )
        train_residuals = {
            target: contract.labels.iloc[fold.train_idx][target].to_numpy(dtype=np.float64)
            - _cross_fitted_anchors(contract.labels, fold.train_idx, target)
            for target in TARGETS
        }
        validation_anchors = {target: _heldout_anchor_values(contract, fold.fold, fold.validation_idx, target) for target in TARGETS}
        test_anchors = {target: _heldout_anchor_values(contract, fold.fold, fold.test_idx, target) for target in TARGETS}
        for target_index, target in enumerate(TARGETS):
            candidates: list[dict[str, Any]] = []
            for alpha in RIDGE_ALPHAS:
                inner_predictions, inner_truths = _fit_inner_oof_weights(
                    contract,
                    matrices,
                    present,
                    fold.train_idx,
                    target,
                    alpha,
                    modalities,
                    seed + fold.fold * 1000 + int(alpha),
                )
                weights = _solve_simplex(inner_predictions, inner_truths)
                models: dict[str, Ridge] = {}
                val_raw = np.zeros((len(fold.validation_idx), len(modalities)), dtype=np.float64)
                test_raw = np.zeros((len(fold.test_idx), len(modalities)), dtype=np.float64)
                for modality_index, modality in enumerate(modalities):
                    train_available = present[modality][fold.train_idx]
                    if int(train_available.sum()) < 3:
                        continue
                    model = Ridge(alpha=alpha).fit(
                        scores[modality][fold.train_idx][train_available],
                        train_residuals[target][train_available],
                    )
                    models[modality] = model
                    val_raw[:, modality_index] = model.predict(scores[modality][fold.validation_idx])
                    test_raw[:, modality_index] = model.predict(scores[modality][fold.test_idx])
                    val_raw[~present[modality][fold.validation_idx], modality_index] = 0.0
                    test_raw[~present[modality][fold.test_idx], modality_index] = 0.0
                val_correction = _weighted_correction(np.clip(val_raw, *CORRECTION_CLIP), np.column_stack([present[m][fold.validation_idx] for m in modalities]), weights)
                test_correction = _weighted_correction(np.clip(test_raw, *CORRECTION_CLIP), np.column_stack([present[m][fold.test_idx] for m in modalities]), weights)
                for gamma in GAMMAS:
                    val_prediction = np.clip(validation_anchors[target] + gamma * val_correction, *PREDICTION_CLIP)
                    val_truth = contract.labels.iloc[fold.validation_idx][target].to_numpy(dtype=np.float64)
                    val_mae = float(np.mean(np.abs(val_truth - val_prediction)))
                    candidates.append(
                        {
                            "key": (val_mae, alpha, gamma),
                            "alpha": alpha,
                            "gamma": gamma,
                            "weights": weights,
                            "models": models,
                            "val_correction": val_correction,
                            "test_correction": test_correction,
                            "val_mae": val_mae,
                        }
                    )
            selected = min(candidates, key=lambda item: item["key"])
            test_prediction = np.clip(test_anchors[target] + selected["gamma"] * selected["test_correction"], *PREDICTION_CLIP)
            test_residual = selected["gamma"] * selected["test_correction"]
            for local_index, row_index in enumerate(fold.test_idx):
                key = (str(contract.labels.iloc[row_index].participant), str(contract.labels.iloc[row_index].condition))
                fallback = key == EXPECTED_FALLBACK
                prediction_rows.append(
                    _record_prediction(
                        contract,
                        int(row_index),
                        fold,
                        seed,
                        target,
                        model_id,
                        "pretrained" if family.family == "pretrained" else family.family,
                        "simplex",
                        f"{model_id}_s{seed}",
                        float(test_anchors[target][local_index]),
                        float(test_residual[local_index]),
                        float(test_anchors[target][local_index] + test_residual[local_index]),
                        float(test_prediction[local_index]),
                        counts.get(key, 0),
                        fallback,
                        downstream_hash,
                        simplex_hash,
                        comparison_track,
                    )
                )
            parameter_rows.append(
                {
                    "fold": fold.fold,
                    "seed": seed,
                    "target": target,
                    "model_id": model_id,
                    "representation_family": family.family,
                    "input_dimensions": json.dumps({m: int(matrices[m].shape[1]) for m in modalities}, sort_keys=True),
                    "selected_pca_dimensions": json.dumps({m: int(preprocessors[m].output_dimension) for m in modalities}, sort_keys=True),
                    "ridge_setting": float(selected["alpha"]),
                    "simplex_weights": json.dumps({m: float(w) for m, w in zip(modalities, selected["weights"], strict=True)}, sort_keys=True),
                    "correction_scale": float(selected["gamma"]),
                    "active_modalities": "+".join(modalities),
                    "HEALNet_stopping_epoch": "",
                    "parameter_count": int(sum(model.coef_.size + np.size(model.intercept_) for model in selected["models"].values())),
                    "training_status": "complete",
                    "validation_mae": float(selected["val_mae"]),
                }
            )
            for modality, weight in zip(modalities, selected["weights"], strict=True):
                weight_rows.append(
                    {
                        "fold": fold.fold,
                        "seed": seed,
                        "target": target,
                        "model_id": model_id,
                        "representation_family": family.family,
                        "modality": modality,
                        "weight": float(weight),
                        "active_weight_floor": WEIGHT_FLOOR,
                    }
                )
    predictions = pd.DataFrame(prediction_rows).sort_values(["seed", "fold", "target", "participant", "condition"]).reset_index(drop=True)
    if len(predictions) != EXPECTED_OBSERVATIONS * len(TARGETS):
        raise ValueError(f"Expected 162 rows for one seed/model, got {len(predictions)}")
    return predictions, pd.DataFrame(parameter_rows), pd.DataFrame(weight_rows)


def run_concat_ridge(
    contract: ContractData,
    family: FamilyData,
    seed: int,
    model_id: str,
    downstream_hash: str,
    simplex_hash: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    counts = _window_counts(contract)
    for fold in contract.folds:
        matrices, present = _matrices_for(family, fold.fold, seed, MODALITY_ORDER)
        preprocessors, scores = fit_preprocessors(matrices, present, fold.train_idx, seed + fold.fold * 1000, MODALITY_ORDER)
        features = np.concatenate([scores[modality] for modality in MODALITY_ORDER], axis=1)
        train_available = np.column_stack([present[modality][fold.train_idx] for modality in MODALITY_ORDER]).all(axis=1)
        residuals = {
            target: contract.labels.iloc[fold.train_idx][target].to_numpy(dtype=np.float64)
            - _cross_fitted_anchors(contract.labels, fold.train_idx, target)
            for target in TARGETS
        }
        for target in TARGETS:
            candidates: list[tuple[float, float, Ridge]] = []
            for alpha in RIDGE_ALPHAS:
                model = Ridge(alpha=alpha).fit(features[fold.train_idx][train_available], residuals[target][train_available])
                validation_correction = np.clip(model.predict(features[fold.validation_idx]), *CORRECTION_CLIP)
                validation_anchor = _heldout_anchor_values(contract, fold.fold, fold.validation_idx, target)
                validation_truth = contract.labels.iloc[fold.validation_idx][target].to_numpy(dtype=np.float64)
                validation_prediction = np.clip(validation_anchor + validation_correction, *PREDICTION_CLIP)
                candidates.append((float(np.mean(np.abs(validation_truth - validation_prediction))), alpha, model))
            validation_mae, alpha, model = min(candidates, key=lambda item: (item[0], item[1]))
            test_anchor = _heldout_anchor_values(contract, fold.fold, fold.test_idx, target)
            test_correction = np.asarray(model.predict(features[fold.test_idx]), dtype=np.float64)
            fallback = np.asarray([
                (str(contract.labels.iloc[index].participant), str(contract.labels.iloc[index].condition)) == EXPECTED_FALLBACK
                for index in fold.test_idx
            ])
            test_correction[fallback] = 0.0
            test_correction = np.clip(test_correction, *CORRECTION_CLIP)
            test_prediction = np.clip(test_anchor + test_correction, *PREDICTION_CLIP)
            for local_index, row_index in enumerate(fold.test_idx):
                key = (str(contract.labels.iloc[row_index].participant), str(contract.labels.iloc[row_index].condition))
                prediction_rows.append(
                    _record_prediction(
                        contract,
                        int(row_index),
                        fold,
                        seed,
                        target,
                        model_id,
                        "pretrained",
                        "concat_ridge",
                        f"{model_id}_s{seed}",
                        float(test_anchor[local_index]),
                        float(test_correction[local_index]),
                        float(test_anchor[local_index] + test_correction[local_index]),
                        float(test_prediction[local_index]),
                        counts.get(key, 0),
                        key == EXPECTED_FALLBACK,
                        downstream_hash,
                        simplex_hash,
                        "frozen_fusion_track",
                    )
                )
            detail_rows.append(
                {
                    "fold": fold.fold,
                    "seed": seed,
                    "target": target,
                    "model_id": model_id,
                    "representation_family": "pretrained",
                    "input_dimensions": json.dumps({m: int(matrices[m].shape[1]) for m in MODALITY_ORDER}, sort_keys=True),
                    "selected_pca_dimensions": json.dumps({m: int(preprocessors[m].output_dimension) for m in MODALITY_ORDER}, sort_keys=True),
                    "ridge_setting": float(alpha),
                    "simplex_weights": "",
                    "correction_scale": 1.0,
                    "active_modalities": "+".join(MODALITY_ORDER),
                    "HEALNet_stopping_epoch": "",
                    "parameter_count": int(model.coef_.size + np.size(model.intercept_)),
                    "training_status": "complete",
                    "validation_mae": float(validation_mae),
                }
            )
    return pd.DataFrame(prediction_rows).sort_values(["seed", "fold", "target", "participant", "condition"]).reset_index(drop=True), pd.DataFrame(detail_rows)


def _metric_values(frame: pd.DataFrame) -> dict[str, float]:
    truth = frame["true_rating"].to_numpy(dtype=float)
    prediction = frame["final_prediction"].to_numpy(dtype=float)
    anchor = frame["condition_only_prediction"].to_numpy(dtype=float)
    true_residual = frame["true_residual"].to_numpy(dtype=float)
    predicted_residual = frame["predicted_residual"].to_numpy(dtype=float)
    correlation = spearmanr(truth, prediction).statistic if len(frame) > 1 and np.std(truth) > 0 and np.std(prediction) > 0 else np.nan
    return {
        "condition_count": int(len(frame)),
        "model_mae": float(np.mean(np.abs(truth - prediction))),
        "condition_only_mae": float(np.mean(np.abs(truth - anchor))),
        "mae_improvement": float(np.mean(np.abs(truth - anchor) - np.abs(truth - prediction))),
        "residual_mae": float(np.mean(np.abs(true_residual - predicted_residual))),
        "spearman_correlation": float(correlation) if np.isfinite(correlation) else np.nan,
        "fallback_count": int(frame["fallback_used"].astype(bool).sum()),
    }


def build_participant_metrics(combined_oof: pd.DataFrame) -> pd.DataFrame:
    """Compute seed-level and three-seed participant-level metrics.

    The three seed predictions are averaged within participant/condition before
    the participant is used as a statistical unit.  This is deliberately kept
    separate from the seed-level rows for traceability.
    """
    rows: list[dict[str, Any]] = []
    group_columns = ["model_id", "target", "participant", "seed"]
    for (model_id, target, participant, seed), frame in combined_oof.groupby(group_columns, sort=True):
        metrics = _metric_values(frame)
        first = frame.iloc[0]
        rows.append(
            {
                "model_id": model_id,
                "representation_family": first["representation_family"],
                "fusion_family": first["fusion_family"],
                "target": target,
                "participant": participant,
                "seed": int(seed),
                "aggregation_level": "seed",
                "run_id": first["run_id"],
                **metrics,
            }
        )

    average_columns = ["model_id", "target", "participant", "condition"]
    averaged = (
        combined_oof.groupby(average_columns, as_index=False, sort=True)
        .agg(
            {
                "representation_family": "first",
                "fusion_family": "first",
                "run_id": "first",
                "seed": "nunique",
                "intensity": "first",
                "frequency": "first",
                "true_rating": "first",
                "condition_anchor": "mean",
                "true_residual": "first",
                "predicted_residual": "mean",
                "prediction_before_clipping": "mean",
                "final_prediction": "mean",
                "condition_only_prediction": "mean",
                "fallback_used": "max",
            }
        )
    )
    if not (averaged["seed"].astype(int) == len(EXPECTED_SEEDS)).all():
        raise ValueError("Seed averaging requires exactly three seed predictions per participant-condition")
    for (model_id, target, participant), frame in averaged.groupby(["model_id", "target", "participant"], sort=True):
        metrics = _metric_values(frame)
        first = frame.iloc[0]
        rows.append(
            {
                "model_id": model_id,
                "representation_family": first["representation_family"],
                "fusion_family": first["fusion_family"],
                "target": target,
                "participant": participant,
                "seed": "mean",
                "aggregation_level": "seed_averaged",
                "run_id": f"{model_id}_seed_average",
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["aggregation_level", "model_id", "target", "participant", "seed"]
    ).reset_index(drop=True)


def _bootstrap_interval(values: np.ndarray, seed: int, replicates: int = BOOTSTRAP_REPLICATES) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(values), size=(replicates, len(values)))
    estimates = values[sample_indices].mean(axis=1)
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def build_group_summary(participant_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_names = (
        "model_mae",
        "condition_only_mae",
        "mae_improvement",
        "residual_mae",
        "spearman_correlation",
    )
    source = participant_metrics[participant_metrics["aggregation_level"].eq("seed_averaged")]
    rows: list[dict[str, Any]] = []
    for (model_id, target), frame in source.groupby(["model_id", "target"], sort=True):
        for metric_index, metric in enumerate(metric_names):
            values = frame[metric].to_numpy(dtype=float)
            ci_lower, ci_upper = _bootstrap_interval(values, BOOTSTRAP_SEED + metric_index + len(rows))
            finite = values[np.isfinite(values)]
            rows.append(
                {
                    "model_id": model_id,
                    "target": target,
                    "aggregation_level": "seed_averaged",
                    "metric": metric,
                    "participant_macro_mean": float(np.mean(finite)) if len(finite) else np.nan,
                    "participant_macro_std": float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan,
                    "participant_macro_median": float(np.median(finite)) if len(finite) else np.nan,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "participant_count": int(len(finite)),
                }
            )
    return pd.DataFrame(rows)


def _holm_adjust(pvalues: Sequence[float]) -> list[float]:
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, float((len(values) - rank) * values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def build_statistical_comparisons(participant_metrics: pd.DataFrame) -> pd.DataFrame:
    source = participant_metrics[participant_metrics["aggregation_level"].eq("seed_averaged")]
    participant_order = tuple(sorted(source["participant"].unique()))
    comparison_families = {
        "representation": (
            ("frozen_simplex_full5", "handcrafted_simplex_full5"),
            ("frozen_simplex_full5", "temporal_1dcnn_simplex_full5"),
        ),
        "fusion": (
            ("frozen_simplex_full5", "frozen_concat_ridge_full5"),
            ("frozen_healnet_full5", "frozen_concat_ridge_full5"),
        ),
    }
    rows: list[dict[str, Any]] = []
    for family, comparisons in comparison_families.items():
        for comparison_index, (left, right) in enumerate(comparisons):
            for target_index, target in enumerate(TARGETS):
                left_frame = source[(source["model_id"] == left) & (source["target"] == target)].set_index("participant")
                right_frame = source[(source["model_id"] == right) & (source["target"] == target)].set_index("participant")
                if set(left_frame.index) != set(participant_order) or set(right_frame.index) != set(participant_order):
                    raise ValueError(f"Incomplete participant metrics for {left} vs {right}/{target}")
                differences = np.asarray(
                    [float(left_frame.loc[p, "model_mae"] - right_frame.loc[p, "model_mae"]) for p in participant_order],
                    dtype=float,
                )
                observed = float(np.mean(differences))
                ci_lower, ci_upper = _bootstrap_interval(
                    differences,
                    BOOTSTRAP_SEED + 1000 * (comparison_index + 1) + target_index + (0 if family == "representation" else 100),
                )
                sign_patterns = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(differences))), dtype=float)
                sign_means = sign_patterns @ differences / len(differences)
                raw_p = float(np.mean(np.abs(sign_means) >= abs(observed) - 1e-15))
                left_values = left_frame.loc[list(participant_order), "model_mae"].to_numpy(dtype=float)
                right_values = right_frame.loc[list(participant_order), "model_mae"].to_numpy(dtype=float)
                rows.append(
                    {
                        "comparison_family": family,
                        "left_model": left,
                        "right_model": right,
                        "target": target,
                        "participant_differences": json.dumps({p: float(d) for p, d in zip(participant_order, differences, strict=True)}, sort_keys=True),
                        **{f"difference_{participant}": float(difference) for participant, difference in zip(participant_order, differences, strict=True)},
                        "mean_difference_left_minus_right": observed,
                        "median_difference_left_minus_right": float(np.median(differences)),
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "raw_p_value": raw_p,
                        "holm_corrected_p_value": np.nan,
                        "wins_left": int(np.sum(left_values < right_values)),
                        "ties": int(np.sum(np.isclose(left_values, right_values, atol=1e-12))),
                        "losses_left": int(np.sum(left_values > right_values)),
                        "participant_count": int(len(differences)),
                        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                        "sign_flip_patterns": int(len(sign_patterns)),
                        "run_id_left": f"{left}_seed_average",
                        "run_id_right": f"{right}_seed_average",
                        "source_file": "participant_metrics.csv",
                    }
                )
    result = pd.DataFrame(rows)
    for family in comparison_families:
        mask = result["comparison_family"].eq(family)
        result.loc[mask, "holm_corrected_p_value"] = _holm_adjust(result.loc[mask, "raw_p_value"].to_numpy(dtype=float))
    return result


def run_video_probes(contract: ContractData, family: FamilyData, output_root: Path) -> pd.DataFrame:
    """Run participant-fold Video stimulus and residual-response probes."""
    rows: list[dict[str, Any]] = []
    video = "Video"
    for fold in contract.folds:
        matrices, present = _matrices_for(family, fold.fold, 0, (video,))
        preprocessors, scores = fit_preprocessors(matrices, present, fold.train_idx, fold.fold, (video,))
        train_present = present[video][fold.train_idx]
        train_rows = fold.train_idx[train_present]
        test_present = present[video][fold.test_idx]
        test_rows = fold.test_idx[test_present]
        for stimulus_target in ("intensity", "frequency"):
            raw_classes = np.sort(contract.labels[stimulus_target].unique())
            class_map = {float(value): index for index, value in enumerate(raw_classes)}
            train_labels = np.asarray([class_map[float(value)] for value in contract.labels.iloc[train_rows][stimulus_target]], dtype=int)
            test_labels = np.asarray([class_map[float(value)] for value in contract.labels.iloc[test_rows][stimulus_target]], dtype=int)
            classes = np.arange(len(raw_classes), dtype=int)
            classifier = LogisticRegression(C=1.0, max_iter=500, random_state=fold.fold)
            classifier.fit(scores[video][train_rows], train_labels)
            truth = test_labels
            predicted = classifier.predict(scores[video][test_rows])
            matrix = confusion_matrix(truth, predicted, labels=classes)
            rows.append(
                {
                    "analysis_type": "stimulus",
                    "target": stimulus_target,
                    "fold": fold.fold,
                    "participant": fold.test_participant,
                    "metric": "balanced_accuracy",
                    "value": float(balanced_accuracy_score(truth, predicted)),
                    "macro_f1": float(f1_score(truth, predicted, labels=classes, average="macro", zero_division=0)),
                    "confusion_matrix": json.dumps(matrix.tolist()),
                    "class_labels": json.dumps(raw_classes.tolist()),
                    "coverage_count": int(len(test_rows)),
                    "run_id": f"video_stimulus_probe_fold{fold.fold}",
                    "source_file": "stimulus_probe_results.csv",
                }
            )
        residuals = {
            target: contract.labels.iloc[fold.train_idx][target].to_numpy(dtype=float)
            - _cross_fitted_anchors(contract.labels, fold.train_idx, target)
            for target in TARGETS
        }
        for target in TARGETS:
            validation_rows = fold.validation_idx[present[video][fold.validation_idx]]
            candidates: list[tuple[float, float, Ridge]] = []
            validation_anchor = _heldout_anchor_values(contract, fold.fold, fold.validation_idx, target)
            validation_truth = contract.labels.iloc[fold.validation_idx][target].to_numpy(dtype=float)
            for alpha in RIDGE_ALPHAS:
                model = Ridge(alpha=alpha).fit(scores[video][train_rows], residuals[target][train_present])
                correction = np.clip(model.predict(scores[video][fold.validation_idx]), *CORRECTION_CLIP)
                correction[~present[video][fold.validation_idx]] = 0.0
                prediction = np.clip(validation_anchor + correction, *PREDICTION_CLIP)
                candidates.append((float(np.mean(np.abs(validation_truth - prediction))), alpha, model))
            validation_mae, alpha, model = min(candidates, key=lambda item: (item[0], item[1]))
            test_anchor = _heldout_anchor_values(contract, fold.fold, fold.test_idx, target)
            correction = np.clip(model.predict(scores[video][fold.test_idx]), *CORRECTION_CLIP)
            correction[~present[video][fold.test_idx]] = 0.0
            truth_residual = contract.labels.iloc[fold.test_idx][target].to_numpy(dtype=float) - test_anchor
            residual_mae = float(np.mean(np.abs(truth_residual - correction)))
            baseline_residual_mae = float(np.mean(np.abs(truth_residual)))
            rows.append(
                {
                    "analysis_type": "response",
                    "target": target,
                    "fold": fold.fold,
                    "participant": fold.test_participant,
                    "metric": "participant_macro_residual_mae",
                    "value": residual_mae,
                    "condition_only_value": baseline_residual_mae,
                    "improvement_over_condition_only": baseline_residual_mae - residual_mae,
                    "selected_ridge_alpha": float(alpha),
                    "validation_mae": float(validation_mae),
                    "coverage_count": int(len(fold.test_idx)),
                    "run_id": f"video_response_probe_{target}_fold{fold.fold}",
                    "source_file": "stimulus_probe_results.csv",
                }
            )
    result = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_root / "stimulus_probe_results.csv", index=False)
    return result


def _summary_lookup(group_summary: pd.DataFrame, model: str, target: str, metric: str) -> dict[str, Any]:
    frame = group_summary[
        group_summary["model_id"].eq(model)
        & group_summary["target"].eq(target)
        & group_summary["metric"].eq(metric)
    ]
    if len(frame) != 1:
        raise ValueError(f"Missing group summary for {model}/{target}/{metric}")
    return frame.iloc[0].to_dict()


def _plot_source_rows(
    participant_metrics: pd.DataFrame,
    models: Sequence[str],
    target: str,
    figure_id: str,
    comparison: str,
    metric: str,
    direction: str,
) -> list[dict[str, Any]]:
    source = participant_metrics[
        participant_metrics["aggregation_level"].eq("seed_averaged")
        & participant_metrics["model_id"].isin(models)
        & participant_metrics["target"].eq(target)
    ]
    rows: list[dict[str, Any]] = []
    for record in source.itertuples(index=False):
        value = float(getattr(record, metric))
        rows.append(
            {
                "figure_id": figure_id,
                "comparison": comparison,
                "target": target,
                "participant": record.participant,
                "model_id": record.model_id,
                "metric": metric,
                "value": value,
                "uncertainty_display": "paired participant lines; 95% bootstrap intervals reported in group_summary.csv",
                "direction": direction,
                "source_file": "participant_metrics.csv",
                "source_run_id": record.run_id,
            }
        )
    return rows


def build_plot_data_and_figures(
    participant_metrics: pd.DataFrame,
    group_summary: pd.DataFrame,
    probes: pd.DataFrame,
    combined_root: Path,
) -> tuple[pd.DataFrame, dict[str, str]]:
    figure_root = combined_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    rows.extend(_plot_source_rows(
        participant_metrics,
        ("frozen_simplex_full5", "handcrafted_simplex_full5"),
        "relaxation",
        "figure_1_representation",
        "frozen_pretrained vs handcrafted; fixed Simplex",
        "model_mae",
        "lower_is_better",
    ))
    rows.extend(_plot_source_rows(
        participant_metrics,
        ("frozen_simplex_full5", "handcrafted_simplex_full5"),
        "discomfort",
        "figure_1_representation",
        "frozen_pretrained vs handcrafted; fixed Simplex",
        "model_mae",
        "lower_is_better",
    ))
    rows.extend(_plot_source_rows(
        participant_metrics,
        ("frozen_simplex_full5", "frozen_concat_ridge_full5", "frozen_healnet_full5"),
        "relaxation",
        "figure_2_fusion",
        "Simplex vs Concatenation + Ridge vs HEALNet; fixed pretrained representations",
        "model_mae",
        "lower_is_better",
    ))
    rows.extend(_plot_source_rows(
        participant_metrics,
        ("frozen_simplex_full5", "frozen_concat_ridge_full5", "frozen_healnet_full5"),
        "discomfort",
        "figure_2_fusion",
        "Simplex vs Concatenation + Ridge vs HEALNet; fixed pretrained representations",
        "model_mae",
        "lower_is_better",
    ))
    rows.extend(_plot_source_rows(
        participant_metrics,
        ("frozen_simplex_full5", "frozen_simplex_no_video"),
        "relaxation",
        "figure_3_video",
        "frozen Simplex full-five vs no-video",
        "model_mae",
        "lower_is_better",
    ))
    rows.extend(_plot_source_rows(
        participant_metrics,
        ("frozen_simplex_full5", "frozen_simplex_no_video"),
        "discomfort",
        "figure_3_video",
        "frozen Simplex full-five vs no-video",
        "model_mae",
        "lower_is_better",
    ))
    stimulus = probes[probes["analysis_type"].eq("stimulus")].copy()
    for record in stimulus.itertuples(index=False):
        rows.append(
            {
                "figure_id": "figure_4_video_probes",
                "comparison": "Video stimulus decoding",
                "target": record.target,
                "participant": record.participant,
                "model_id": "VideoMAE_V2_stimulus_probe",
                "metric": "balanced_accuracy",
                "value": float(record.value),
                "secondary_value": float(record.macro_f1),
                "uncertainty_display": "participant-fold points; participant-macro mean and spread in stimulus_probe_results.csv",
                "direction": "higher_is_better",
                "source_file": "stimulus_probe_results.csv",
                "source_run_id": record.run_id,
            }
        )
    response = probes[probes["analysis_type"].eq("response")].copy()
    for record in response.itertuples(index=False):
        rows.append(
            {
                "figure_id": "figure_4_video_probes",
                "comparison": "Video residual-response probe",
                "target": record.target,
                "participant": record.participant,
                "model_id": "VideoMAE_V2_response_probe",
                "metric": "residual_mae",
                "value": float(record.value),
                "secondary_value": float(record.condition_only_value),
                "uncertainty_display": "participant-fold points; condition-only comparator in stimulus_probe_results.csv",
                "direction": "lower_is_better",
                "source_file": "stimulus_probe_results.csv",
                "source_run_id": record.run_id,
            }
        )
    plot_data = pd.DataFrame(rows)
    plot_data.to_csv(combined_root / "plot_data.csv", index=False)

    def paired_plot(figure_id: str, targets: Sequence[str], models: Sequence[str], labels: Mapping[str, str], path: Path, title: str) -> None:
        fig, axes = plt.subplots(1, len(targets), figsize=(12, 5), squeeze=False)
        for axis, target in zip(axes[0], targets, strict=True):
            source = plot_data[
                plot_data["figure_id"].eq(figure_id)
                & plot_data["target"].eq(target)
                & plot_data["model_id"].isin(models)
            ]
            participants = sorted(source["participant"].unique())
            x = np.arange(len(participants))
            for model_index, model in enumerate(models):
                values = [float(source[(source["participant"] == p) & (source["model_id"] == model)]["value"].iloc[0]) for p in participants]
                summary = _summary_lookup(group_summary, model, target, "model_mae")
                axis.errorbar(
                    x + (model_index - (len(models) - 1) / 2) * 0.12,
                    [summary["participant_macro_mean"]] * len(x),
                    yerr=[[summary["participant_macro_mean"] - summary["ci_lower"]] * len(x), [summary["ci_upper"] - summary["participant_macro_mean"]] * len(x)],
                    fmt="none",
                    alpha=0.25,
                    color=f"C{model_index}",
                    capsize=2,
                )
                axis.plot(x, values, "o-", label=labels[model], alpha=0.85, color=f"C{model_index}")
            if len(models) == 2:
                left = [float(source[(source["participant"] == p) & (source["model_id"] == models[0])]["value"].iloc[0]) for p in participants]
                right = [float(source[(source["participant"] == p) & (source["model_id"] == models[1])]["value"].iloc[0]) for p in participants]
                for index in range(len(participants)):
                    axis.plot([x[index], x[index]], [left[index], right[index]], color="0.7", linewidth=0.7, zorder=0)
            axis.set_title(target.capitalize())
            axis.set_xticks(x, participants, rotation=45)
            axis.set_ylabel("Participant-macro MAE")
            axis.grid(axis="y", alpha=0.25)
        axes[0, 0].legend(frameon=False)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)

    paired_plot(
        "figure_1_representation",
        TARGETS,
        ("frozen_simplex_full5", "handcrafted_simplex_full5"),
        {"frozen_simplex_full5": "Frozen pretrained", "handcrafted_simplex_full5": "Handcrafted"},
        figure_root / "figure_1_representation_comparison.png",
        "FMQ-9 representation comparison (N=9 per target)",
    )
    paired_plot(
        "figure_2_fusion",
        TARGETS,
        ("frozen_simplex_full5", "frozen_concat_ridge_full5", "frozen_healnet_full5"),
        {"frozen_simplex_full5": "Simplex", "frozen_concat_ridge_full5": "Concat + Ridge", "frozen_healnet_full5": "HEALNet"},
        figure_root / "figure_2_fusion_comparison.png",
        "FMQ-9 frozen-representation fusion comparison (N=9 per target)",
    )
    paired_plot(
        "figure_3_video",
        TARGETS,
        ("frozen_simplex_full5", "frozen_simplex_no_video"),
        {"frozen_simplex_full5": "Full five", "frozen_simplex_no_video": "No Video"},
        figure_root / "figure_3_full_five_vs_no_video.png",
        "FMQ-9 Video audit (N=9 per target)",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    stimulus_summary = stimulus.groupby("target").agg(balanced_accuracy=("value", "mean"), macro_f1=("macro_f1", "mean"))
    x = np.arange(len(stimulus_summary))
    axes[0].bar(x - 0.18, stimulus_summary["balanced_accuracy"], width=0.36, label="Balanced accuracy")
    axes[0].bar(x + 0.18, stimulus_summary["macro_f1"], width=0.36, label="Macro F1")
    axes[0].set_xticks(x, [str(v) for v in stimulus_summary.index])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Video stimulus probes")
    axes[0].set_ylabel("Higher is better")
    axes[0].legend(frameon=False)
    response_summary = response.groupby("target").agg(video_residual_mae=("value", "mean"), condition_only_residual_mae=("condition_only_value", "mean"))
    x = np.arange(len(response_summary))
    axes[1].bar(x - 0.18, response_summary["video_residual_mae"], width=0.36, label="Video residual probe")
    axes[1].bar(x + 0.18, response_summary["condition_only_residual_mae"], width=0.36, label="Condition-only")
    axes[1].set_xticks(x, [str(v) for v in response_summary.index])
    axes[1].set_title("Video response probes")
    axes[1].set_ylabel("Residual MAE; lower is better")
    axes[1].legend(frameon=False)
    fig.suptitle("FMQ-9 VideoMAE V2 probes (N=9 participant folds)")
    fig.tight_layout()
    fig.savefig(figure_root / "figure_4_stimulus_vs_residual_probe.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    captions = {
        "figure_1_representation": "FMQ-9, N=9 per target; participant-macro MAE for the fixed-Simplex representation comparison, paired participant lines and 95% bootstrap intervals; lower is better.",
        "figure_2_fusion": "FMQ-9, N=9 per target; participant-macro MAE for Simplex, Concatenation + Ridge and HEALNet with frozen pretrained representations, paired participant lines and 95% bootstrap intervals; lower is better.",
        "figure_3_video": "FMQ-9, N=9 per target; participant-macro MAE for frozen Simplex with all five modalities versus removal of Video, paired participant lines and 95% bootstrap intervals; lower is better.",
        "figure_4_video_probes": "FMQ-9, N=9 participant folds; intensity/frequency stimulus balanced accuracy and macro F1 (higher is better) beside relaxation/discomfort residual MAE for Video response probes (lower is better), with participant-fold points and condition-only comparison.",
    }
    return plot_data, captions


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{value:.{digits}f}"


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str], headers: Sequence[str] | None = None) -> str:
    headers = tuple(headers or columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for record in frame.loc[:, list(columns)].itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in record) + " |")
    return "\n".join(lines)


def _source_link(filename: str) -> str:
    return f"[combined/{filename}]({filename})"


def write_report(
    contract: ContractData,
    state: Any,
    downstream_hash: str,
    simplex_hash: str,
    combined_oof: pd.DataFrame,
    participant_metrics: pd.DataFrame,
    group_summary: pd.DataFrame,
    statistics: pd.DataFrame,
    probes: pd.DataFrame,
    simplex_weights: pd.DataFrame,
    fusion_details: pd.DataFrame,
    run_manifest: pd.DataFrame,
    captions: Mapping[str, str],
    combined_root: Path,
) -> Path:
    report_path = combined_root / "RQ2_final_report.md"
    completed = run_manifest[run_manifest["status"].eq("complete")]
    failed = run_manifest[~run_manifest["status"].eq("complete")]
    run_ids = sorted(completed["run_id"].astype(str).unique())
    lines: list[str] = ["# 1. Data contract and completed experiments", ""]
    lines.extend(
        [
            "The analysis used the Windows FMQ-9 contract as the sole source of labels, folds and condition anchors. It contains 81 participant-condition ratings (9 participants × 9 conditions), 545 common-valid windows, and five representation modalities: EEG/REVE (1024), ECG/ECGFounder (1024), Eye/InceptionTime (128), Head handcrafted (18), and Video/VideoMAE V2 (768). The nine participant folds use seven training participants, one validation participant and one test participant; seeds are `20260705`, `20260706` and `20260707`.",
            "",
            f"The immutable composite contract hash is `{contract.contract_hash}`. The saved WSL cache was joined to the contract by participant + condition + explicit `condition_window_index` mapped to the contract `window_id`; no row, filename or directory ordering was used, and no pretrained encoder was rerun. P004/C6 remains in all 81-key artifacts as an unavailable condition-level fallback, returning its shared condition anchor exactly. The local cache and artifact audit is {_source_link('../wsl_results/representation_audit.csv')}; the materialized frozen-pretrained index is {_source_link('../wsl_results/pretrained_representation_index.csv')}.",
            "",
            f"The canonical downstream source hash is `{downstream_hash}` and the Simplex source hash is `{simplex_hash}`. Train-only imputation, scaling and PCA used the locked 3/4/1/1/3 component allocation in EEG/ECG/Eye/Head/Video order; Ridge used alpha ∈ {{10, 100, 1000}}, and target-specific Simplex used the locked 0.02 weight floor, gamma ∈ {{0.25, 0.5, 0.75, 1.0}}, correction clipping [-0.2, 0.2], missing-expert renormalisation and final prediction clipping [0, 1]. Run provenance is {_source_link('wsl_run_manifest.csv')} and detailed fold settings are {_source_link('fusion_details.csv')}.",
            "",
            f"Completed run IDs: {', '.join(f'`{run_id}`' for run_id in run_ids)}.",
            f"Failed runs in the resumed execution: {', '.join(f'`{run_id}`' for run_id in failed['run_id'].astype(str)) if len(failed) else 'none'}. The earlier preflight-only alignment error is retained as historical evidence in `../wsl_results/WSL_ALIGNMENT_ERROR.md`; it did not participate in this resumed run.",
            "",
            f"The final OOF table contains {len(combined_oof)} rows across five complete model configurations, two targets and three seeds. Seed-level predictions were preserved before averaging; participant-level statistics average the three seeds within participant and condition, then use the nine participants as the statistical units. The contract and completed acceptance checks are documented in {_source_link('CROSS_ENVIRONMENT_ALIGNMENT_REPORT.md')}.",
            "",
            "## Completed experiment matrix",
            "",
            _markdown_table(
                completed[["run_id", "run_type", "status"]].drop_duplicates().sort_values("run_id"),
                ("run_id", "run_type", "status"),
            ),
            "",
            "## Failed and excluded work",
            "",
            "No pretrained encoder extraction or fine-tuning was performed in this resumed run. No model was silently substituted for P004/C6, and no failed complete model was relabeled as a successful comparison.",
            "",
            "# 2. Representation and fusion results",
            "",
            "The fixed-Simplex representation track compares three representation families through the identical Ridge-expert/Simplex downstream implementation: `frozen_simplex_full5`, `handcrafted_simplex_full5` and `temporal_1dcnn_simplex_full5`. The fixed-representation fusion track holds the frozen-pretrained vectors, folds, masks, anchors and seeds fixed while comparing `frozen_simplex_full5`, `frozen_concat_ridge_full5` and `frozen_healnet_full5`. The frozen Simplex predictions, weights, parameters and run ID are reused across both interpretations; no second Simplex run was trained.",
            "",
            f"The primary participant metrics are in {_source_link('participant_metrics.csv')} and the group summaries with 10,000-participant-bootstrap intervals are in {_source_link('group_summary.csv')}. Statistical comparisons use nine paired participant differences, 10,000 participant bootstraps and the exact 512-pattern participant sign-flip test; Holm correction is within the four-test representation or fusion family in {_source_link('statistical_comparisons.csv')}.",
            "",
        ]
    )
    summary_table = group_summary[group_summary["metric"].isin(("model_mae", "condition_only_mae", "mae_improvement"))].copy()
    for target in TARGETS:
        lines.extend([f"### {target.capitalize()}", ""])
        table_rows: list[dict[str, Any]] = []
        for model in sorted(summary_table["model_id"].unique()):
            values = {row["metric"]: row for _, row in summary_table[(summary_table["model_id"] == model) & (summary_table["target"] == target)].iterrows()}
            if not values:
                continue
            mae = values["model_mae"]
            table_rows.append(
                {
                    "model": model,
                    "MAE": _fmt(mae["participant_macro_mean"]),
                    "condition-only MAE": _fmt(values["condition_only_mae"]["participant_macro_mean"]),
                    "improvement": _fmt(values["mae_improvement"]["participant_macro_mean"]),
                    "95% interval for MAE": f"[{_fmt(mae['ci_lower'])}, {_fmt(mae['ci_upper'])}]",
                }
            )
        lines.extend([_markdown_table(pd.DataFrame(table_rows), ("model", "MAE", "condition-only MAE", "improvement", "95% interval for MAE")), ""])
        for family in ("representation", "fusion"):
            subset = statistics[(statistics["comparison_family"] == family) & (statistics["target"] == target)].copy()
            subset["comparison"] = subset["left_model"] + " vs " + subset["right_model"]
            subset["95% CI"] = subset.apply(lambda row: f"[{_fmt(row['ci_lower'])}, {_fmt(row['ci_upper'])}]", axis=1)
            subset["p"] = subset["raw_p_value"].map(_fmt)
            subset["Holm p"] = subset["holm_corrected_p_value"].map(_fmt)
            subset["W/T/L"] = subset.apply(lambda row: f"{int(row['wins_left'])}/{int(row['ties'])}/{int(row['losses_left'])}", axis=1)
            lines.extend([f"**{family.capitalize()} primary comparisons.**", "", _markdown_table(subset, ("comparison", "mean_difference_left_minus_right", "95% CI", "p", "Holm p", "W/T/L"), ("comparison", "MAE difference (left − right)", "95% CI", "raw p", "Holm p", "left W/T/L")), ""])
            for row in subset.itertuples(index=False):
                favored = f"the left model has lower MAE" if row.mean_difference_left_minus_right < 0 else "the right model has lower MAE" if row.mean_difference_left_minus_right > 0 else "the paired mean MAE difference is zero"
                lines.append(
                    f"Comparison: `{row.left_model}` versus `{row.right_model}` for {target} → the participant-macro MAE difference (left − right) was {_fmt(row.mean_difference_left_minus_right)}, with wins/ties/losses {int(row.wins_left)}/{int(row.ties)}/{int(row.losses_left)} → the 95% participant-bootstrap interval was [{_fmt(row.ci_lower)}, {_fmt(row.ci_upper)}], the exact 512-pattern sign-flip p-value was {_fmt(row.raw_p_value)}, and the within-family Holm-corrected p-value was {_fmt(row.holm_corrected_p_value)} → {favored} is an observed paired direction, not a claim of superiority unless the corrected test supports it → source: {_source_link('statistical_comparisons.csv')}, run IDs `{row.run_id_left}` and `{row.run_id_right}`."
                )
                lines.append("")

    lines.extend(
        [
            "# 3. What the models learned and the role of Video",
            "",
            "Simplex weights are reported as selected fold/seed values rather than post-hoc explanations. The table below gives their arithmetic mean for the frozen-pretrained full-five run; it describes the downstream correction allocation and does not establish causal modality importance.",
            "",
        ]
    )
    weight_source = simplex_weights[simplex_weights["model_id"].eq("frozen_simplex_full5")].copy()
    weight_summary = weight_source.groupby(["target", "modality"], as_index=False)["weight"].mean().rename(columns={"weight": "mean_selected_weight"})
    weight_summary["mean_selected_weight"] = weight_summary["mean_selected_weight"].map(_fmt)
    lines.extend([_markdown_table(weight_summary, ("target", "modality", "mean_selected_weight")), "", f"Source: {_source_link('representation_track_simplex_weights.csv')}; full-five run ID `frozen_simplex_full5_s20260705` (with the same run family for the other seeds).", ""])

    video_summary = group_summary[(group_summary["model_id"].isin(("frozen_simplex_full5", "frozen_simplex_no_video"))) & group_summary["metric"].eq("model_mae")]
    for target in TARGETS:
        full = _summary_lookup(group_summary, "frozen_simplex_full5", target, "model_mae")
        no_video = _summary_lookup(group_summary, "frozen_simplex_no_video", target, "model_mae")
        difference = float(full["participant_macro_mean"] - no_video["participant_macro_mean"])
        lines.append(
            f"Comparison: frozen Simplex full-five versus the same frozen Simplex with only Video removed for {target} → full-five participant-macro MAE was {_fmt(full['participant_macro_mean'])} and no-video MAE was {_fmt(no_video['participant_macro_mean'])}, a descriptive difference of {_fmt(difference)} → the corresponding 95% intervals were [{_fmt(full['ci_lower'])}, {_fmt(full['ci_upper'])}] and [{_fmt(no_video['ci_lower'])}, {_fmt(no_video['ci_upper'])}] → this controlled audit indicates the observed direction of Video's incremental prediction contribution for this target, without attributing a causal mechanism → source: {_source_link('group_summary.csv')}, run IDs `frozen_simplex_full5_s20260705` and `frozen_simplex_no_video_s20260705`."
        )
        lines.append("")

    stimulus = probes[probes["analysis_type"].eq("stimulus")]
    response = probes[probes["analysis_type"].eq("response")]
    lines.extend(["## VideoMAE V2 probes", ""])
    stimulus_table = stimulus.groupby("target", as_index=False).agg(
        balanced_accuracy=("value", "mean"),
        balanced_accuracy_sd=("value", "std"),
        macro_f1=("macro_f1", "mean"),
        participant_folds=("fold", "nunique"),
    )
    stimulus_table["balanced_accuracy"] = stimulus_table["balanced_accuracy"].map(_fmt)
    stimulus_table["balanced_accuracy_sd"] = stimulus_table["balanced_accuracy_sd"].map(_fmt)
    stimulus_table["macro_f1"] = stimulus_table["macro_f1"].map(_fmt)
    lines.extend(["Stimulus probes:", "", _markdown_table(stimulus_table, ("target", "balanced_accuracy", "balanced_accuracy_sd", "macro_f1", "participant_folds")), ""])
    response_table = response.groupby("target", as_index=False).agg(
        video_residual_mae=("value", "mean"),
        condition_only_residual_mae=("condition_only_value", "mean"),
        improvement=("improvement_over_condition_only", "mean"),
        participant_folds=("fold", "nunique"),
    )
    for column in ("video_residual_mae", "condition_only_residual_mae", "improvement"):
        response_table[column] = response_table[column].map(_fmt)
    lines.extend(["Response probes:", "", _markdown_table(response_table, ("target", "video_residual_mae", "condition_only_residual_mae", "improvement", "participant_folds")), ""])
    lines.extend(
        [
            f"These probe summaries are secondary analyses in {_source_link('stimulus_probe_results.csv')}. The intensity/frequency rows test rendered stimulus information with balanced accuracy, macro F1 and fold-level confusion matrices; the relaxation/discomfort rows test residual response prediction against the condition-only residual baseline. Together they distinguish stimulus preservation from evidence of individual response information without claiming that either probe identifies a causal pathway.",
            "",
            f"The four thesis-ready figures are {_source_link('figures/figure_1_representation_comparison.png')}, {_source_link('figures/figure_2_fusion_comparison.png')}, {_source_link('figures/figure_3_full_five_vs_no_video.png')} and {_source_link('figures/figure_4_stimulus_vs_residual_probe.png')}; their exact plotted source rows and captions are in {_source_link('plot_data.csv')}.",
            "",
            "# 4. Direct answer to RQ2 and writing package",
            "",
        ]
    )
    significant = statistics[statistics["holm_corrected_p_value"] < 0.05]
    if len(significant):
        supported_lines = [
            f"Supported findings: {row.left_model} versus {row.right_model} for {row.target} has a Holm-corrected exact sign-flip p-value of {_fmt(row.holm_corrected_p_value)} and mean paired difference {_fmt(row.mean_difference_left_minus_right)}."
            for row in significant.itertuples(index=False)
        ]
        lines.extend(supported_lines)
    else:
        lines.append("Supported findings: none of the eight predeclared primary representation/fusion comparisons reached the within-family Holm-corrected 0.05 threshold under the nine-participant exact paired analysis.")
    lines.extend(
        [
            "",
            "Uncertain findings: descriptive MAE differences, selected Simplex weights, Video full-five/no-video contrasts and Video probe scores remain reportable evidence, but their intervals and/or secondary status do not by themselves establish a representation, fusion or Video effect.",
            "",
            "Unsupported claims: these results do not show that VideoMAE V2 is universally better than a VLM, do not justify adding a VLM to this experiment, and do not support causal explanations for any modality weight or probe association. The Video conclusion is deliberately scoped to the existing cached VideoMAE V2 representation and this controlled FMQ-9 analysis.",
            "",
            "## Thesis Results package",
            "",
            "Using the fixed participant-level contract, we evaluated three representation families with one shared Ridge-expert/Simplex downstream pipeline and compared three fusion architectures with the pretrained representations held fixed. Predictions were generated in nine participant folds for three seeds, averaged across seeds within participant-condition, and tested with paired participant statistics. Report the exact MAE, condition-only MAE, improvement, bootstrap interval, exact sign-flip p-value, Holm correction and wins/ties/losses from the tables above; the primary source files are `participant_metrics.csv`, `group_summary.csv` and `statistical_comparisons.csv`.",
            "",
            "## Thesis Discussion package",
            "",
            "The controlled comparisons support only the scoped statements whose paired tests and uncertainty intervals are shown. The Video audit separates full-five from no-video under unchanged folds, anchors, seeds and fusion rules, while the VideoMAE V2 stimulus and residual probes distinguish stimulus decoding from individual-response prediction. Interpret non-significant and unsuccessful comparisons as retained evidence rather than as proof of equivalence.",
            "",
            "## Thesis Conclusion package",
            "",
            "RQ2 is answered at the level supported by the participant-level controlled analysis: the saved representations and fusion choices were evaluated under a common contract, and any claimed difference must be read from its corresponding comparison family. The study does not make a universal encoder claim and does not generalize the VideoMAE V2 result to VLMs.",
            "",
            "## Figure captions and source links",
            "",
        ]
    )
    for figure_id, caption in captions.items():
        filename = {
            "figure_1_representation": "figures/figure_1_representation_comparison.png",
            "figure_2_fusion": "figures/figure_2_fusion_comparison.png",
            "figure_3_video": "figures/figure_3_full_five_vs_no_video.png",
            "figure_4_video_probes": "figures/figure_4_stimulus_vs_residual_probe.png",
        }[figure_id]
        lines.append(f"- **{filename}** — {caption} Exact source rows: {_source_link('plot_data.csv')}; supporting tables: {_source_link('participant_metrics.csv')} and {_source_link('stimulus_probe_results.csv') if figure_id == 'figure_4_video_probes' else _source_link('group_summary.csv')}.")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _normalise_index_for_alignment(frame: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["participant"] = output["participant"].astype(str)
    output["condition"] = output["condition"].astype(str)
    output["modality"] = output["modality"].map(_canonical_modality)
    if "fallback_used" in output:
        output["fallback_used"] = _bool_series(output["fallback_used"])
    metadata = labels[["participant", "condition", "intensity", "frequency", "fallback_status"]]
    existing_metadata = [column for column in metadata.columns if column not in ("participant", "condition") and column in output.columns]
    if existing_metadata:
        output = output.drop(columns=existing_metadata)
    return output.merge(metadata, on=["participant", "condition"], how="left", validate="many_to_one")


def write_cross_environment_alignment_report(
    contract: ContractData,
    pretrained_index: pd.DataFrame,
    combined_root: Path,
) -> dict[str, Any]:
    local = _normalise_index_for_alignment(pretrained_index, contract.labels)
    windows = contract.windows_index.copy()
    windows["participant"] = windows["participant"].astype(str)
    windows["condition"] = windows["condition"].astype(str)
    windows["modality"] = windows["modality"].map(_canonical_modality)
    windows["representation_family"] = windows["representation_family"].astype(str).str.lower()
    windows["representation_available"] = _bool_series(windows["representation_available"])
    windows["fallback_used"] = _bool_series(windows["fallback_used"])
    windows = _normalise_index_for_alignment(windows, contract.labels)
    local_keys = set(map(tuple, local[["participant", "condition", "modality"]].to_numpy(dtype=str)))
    checks: dict[str, Any] = {}
    checks["contract_hash"] = contract.contract_hash
    checks["local_pretrained_rows"] = int(len(local))
    checks["local_key_count"] = int(len(local_keys))
    checks["expected_key_count"] = int(EXPECTED_OBSERVATIONS * len(MODALITIES))
    checks["local_duplicate_keys"] = int(local.duplicated(["participant", "condition", "modality"]).sum())
    checks["local_modalities"] = sorted(local["modality"].unique())
    checks["condition_metadata_complete"] = bool(local[["intensity", "frequency", "fallback_status"]].notna().all().all())
    checks["fallback_rows"] = local[
        local["participant"].eq(EXPECTED_FALLBACK[0]) & local["condition"].eq(EXPECTED_FALLBACK[1])
    ][["modality", "fallback_used"]].to_dict("records")
    family_checks: dict[str, Any] = {}
    for family in ("handcrafted", "temporal_1dcnn"):
        subset = windows[windows["representation_family"].eq(family)]
        groups: dict[str, Any] = {}
        for (fold, seed), group in subset.groupby(["fold", "seed"], sort=True):
            keys = set(map(tuple, group[["participant", "condition", "modality"]].to_numpy(dtype=str)))
            fallback = group[
                group["participant"].eq(EXPECTED_FALLBACK[0]) & group["condition"].eq(EXPECTED_FALLBACK[1])
            ]
            groups[f"{fold}/{seed}"] = {
                "rows": int(len(group)),
                "key_count": int(len(keys)),
                "same_participant_condition_modality_keys_as_wsl": bool(keys == local_keys),
                "duplicate_keys": int(group.duplicated(["participant", "condition", "modality"]).sum()),
                "fallback_all_unavailable": bool(len(fallback) == len(MODALITIES) and (~fallback["representation_available"].astype(bool)).all()),
                "condition_metadata_matches_contract": bool(group[["intensity", "frequency", "fallback_status"]].notna().all().all()),
            }
        family_checks[family] = groups
    checks["windows_families"] = family_checks
    checks["all_key_and_metadata_checks_pass"] = bool(
        checks["local_key_count"] == checks["expected_key_count"]
        and checks["local_duplicate_keys"] == 0
        and checks["local_modalities"] == sorted(MODALITY_ORDER)
        and checks["condition_metadata_complete"]
        and all(
            item["same_participant_condition_modality_keys_as_wsl"]
            and item["duplicate_keys"] == 0
            and item["fallback_all_unavailable"]
            and item["condition_metadata_matches_contract"]
            for family in family_checks.values()
            for item in family.values()
        )
    )
    lines = [
        "# Cross-environment RQ2 alignment report",
        "",
        f"- Composite contract hash: `{contract.contract_hash}`",
        "- Contract source: Windows hand-off under `/mnt/c/Users/linki/Wei/Models/rq2_shared/contract/` (read-only).",
        "- WSL source: saved condition-level cache under the WSL filesystem; no pretrained encoder was run.",
        "- Join rule: participant + condition + explicit condition window index mapped to Windows `window_id`; representation indexes are joined by participant + condition + modality.",
        "",
        "## Checks",
        "",
        f"- WSL pretrained index rows: `{checks['local_pretrained_rows']}`; expected `5 × 81 = 405`.",
        f"- Duplicate WSL keys: `{checks['local_duplicate_keys']}`.",
        f"- WSL modalities: `{', '.join(checks['local_modalities'])}`.",
        f"- P004/C6 fallback rows: `{checks['fallback_rows']}`.",
        f"- All cross-environment key, condition-metadata and fallback checks: **{'PASS' if checks['all_key_and_metadata_checks_pass'] else 'FAIL'}**.",
        "",
        "## Windows representation families",
        "",
    ]
    for family, groups in family_checks.items():
        lines.append(f"### {family}")
        lines.append("")
        for group_id, detail in groups.items():
            lines.append(f"- `{group_id}`: {detail}")
        lines.append("")
    (combined_root / "CROSS_ENVIRONMENT_ALIGNMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return checks


def _package_versions() -> dict[str, str]:
    names = ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "torch", "joblib")
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    versions["python"] = sys.version.split()[0]
    return versions


def _write_shared_fusion_error(combined_root: Path, error: Exception, run_records: Sequence[Mapping[str, Any]]) -> Path:
    path = combined_root / "SHARED_FUSION_ERROR.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Shared fusion error",
        "",
        "The RQ2 WSL preflight passed, but the shared downstream/fusion pipeline did not complete. WSL_DONE.json was not written.",
        "",
        f"- Error: `{error}`",
        "- No encoder was rerun and no contract file was regenerated.",
        "",
        "## Run records",
        "",
    ]
    lines.extend(f"- `{record.get('run_id')}` — `{record.get('status')}` — `{record.get('detail', '')}`" for record in run_records)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _acceptance_checks(
    contract: ContractData,
    combined_oof: pd.DataFrame,
    representation_oof: pd.DataFrame,
    participant_metrics: pd.DataFrame,
    statistics: pd.DataFrame,
    pretrained_index: pd.DataFrame,
    downstream_hash: str,
    simplex_hash: str,
) -> dict[str, dict[str, Any]]:
    expected_keys = set(zip(contract.labels["participant"], contract.labels["condition"], strict=True))
    checks: dict[str, dict[str, Any]] = {}

    def json_safe(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): json_safe(child) for key, child in value.items()}
        if isinstance(value, (set, tuple)):
            return [json_safe(child) for child in value]
        if isinstance(value, np.ndarray):
            return [json_safe(child) for child in value.tolist()]
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()
        return value

    def add(name: str, passed: bool, detail: Any) -> None:
        checks[name] = {"passed": bool(passed), "detail": json_safe(detail)}
        if not passed:
            raise ValueError(f"Acceptance check failed: {name}: {detail}")

    add("pretrained_index_405_rows", len(pretrained_index) == EXPECTED_OBSERVATIONS * len(MODALITIES), len(pretrained_index))
    add("pretrained_index_unique_keys", pretrained_index.duplicated(["participant", "condition", "modality"]).sum() == 0, int(pretrained_index.duplicated(["participant", "condition", "modality"]).sum()))
    add("pretrained_index_dimensions", all(pretrained_index[pretrained_index["modality"].str.lower() == modality]["representation_dimension"].eq(EXPECTED_DIMENSIONS[modality]).all() for modality in MODALITIES), pretrained_index.groupby("modality")["representation_dimension"].unique().to_dict())
    add("representation_track_models", set(representation_oof["model_id"]) == {"frozen_simplex_full5", "handcrafted_simplex_full5", "temporal_1dcnn_simplex_full5"}, sorted(representation_oof["model_id"].unique()))
    for model in sorted(combined_oof["model_id"].unique()):
        for seed in EXPECTED_SEEDS:
            for target in TARGETS:
                group = combined_oof[(combined_oof["model_id"] == model) & (combined_oof["seed"] == seed) & (combined_oof["target"] == target)]
                observed = set(zip(group["participant"], group["condition"], strict=True))
                add(f"complete_{model}_{seed}_{target}", len(group) == EXPECTED_OBSERVATIONS and observed == expected_keys, {"rows": len(group), "missing": sorted(expected_keys - observed), "extra": sorted(observed - expected_keys)})
    final_key_columns = ["model_id", "seed", "target", "participant", "condition"]
    add("no_duplicate_final_oof_keys", int(combined_oof.duplicated(final_key_columns).sum()) == 0, int(combined_oof.duplicated(final_key_columns).sum()))
    add("nine_conditions_per_participant", set(combined_oof.groupby(["model_id", "seed", "target", "participant"]).size().astype(int)) == {9}, combined_oof.groupby(["model_id", "seed", "target", "participant"]).size().value_counts().to_dict())
    fallback = combined_oof[(combined_oof["participant"] == EXPECTED_FALLBACK[0]) & (combined_oof["condition"] == EXPECTED_FALLBACK[1])]
    add("p004_c6_fallback_present", len(fallback) == len(combined_oof["model_id"].unique()) * len(EXPECTED_SEEDS) * len(TARGETS) and fallback["fallback_used"].astype(bool).all() and np.allclose(fallback["final_prediction"], fallback["condition_anchor"]), {"rows": len(fallback), "models": sorted(fallback["model_id"].unique())})
    anchor_matches = []
    for row in combined_oof.itertuples(index=False):
        anchor_matches.append(np.isclose(float(row.condition_anchor), contract.anchors[(int(row.fold), str(row.condition), str(row.target))], atol=1e-12))
    add("all_anchors_match_windows_contract", bool(np.all(anchor_matches)), int(np.sum(~np.asarray(anchor_matches, dtype=bool))))
    add("controlled_representation_code_hashes", set(combined_oof[combined_oof["comparison_track"].astype(str).str.contains("representation")]["downstream_code_hash"]) == {downstream_hash} and set(combined_oof[combined_oof["comparison_track"].astype(str).str.contains("representation")]["simplex_code_hash"]) == {simplex_hash}, "hash sets")
    required_columns = {
        "contract_hash", "downstream_code_hash", "simplex_code_hash", "run_id", "comparison_track", "comparison_scope", "model_id", "representation_family", "fusion_family", "modality_set", "fold", "seed", "target", "test_participant", "participant", "condition", "intensity", "frequency", "true_rating", "condition_anchor", "true_residual", "predicted_residual", "prediction_before_clipping", "final_prediction", "condition_only_prediction", "model_absolute_error", "baseline_absolute_error", "improvement_over_baseline", "valid_window_count", "fallback_used",
    }
    add("final_oof_required_columns", required_columns.issubset(combined_oof.columns), sorted(required_columns - set(combined_oof.columns)))
    add("representation_track_rows_per_model", all(len(representation_oof[representation_oof["model_id"] == model]) == EXPECTED_OBSERVATIONS * len(TARGETS) * len(EXPECTED_SEEDS) for model in representation_oof["model_id"].unique()), representation_oof.groupby("model_id").size().to_dict())
    add("participant_metrics_seed_averages", set(participant_metrics[participant_metrics["aggregation_level"] == "seed_averaged"].groupby(["model_id", "target"]).size().astype(int)) == {9}, participant_metrics[participant_metrics["aggregation_level"] == "seed_averaged"].groupby(["model_id", "target"]).size().to_dict())
    add("primary_statistics_eight_rows", len(statistics) == 8 and set(statistics["participant_count"]) == {9} and set(statistics["sign_flip_patterns"]) == {512}, len(statistics))
    add("primary_statistics_holm_families", statistics.groupby("comparison_family").size().to_dict() == {"fusion": 4, "representation": 4}, statistics.groupby("comparison_family").size().to_dict())
    add("finite_oof_values", bool(np.isfinite(combined_oof[["true_rating", "condition_anchor", "true_residual", "predicted_residual", "final_prediction"]].to_numpy(dtype=float)).all()), "finite prediction columns")
    return checks


def run_pipeline(
    shared_root: Path,
    cache_root: Path,
    output_root: Path,
    combined_root: Path,
    state: Any | None = None,
) -> dict[str, Any]:
    """Run the complete post-handoff RQ2 workflow after a successful preflight."""
    run_records: list[dict[str, Any]] = []
    output_root.mkdir(parents=True, exist_ok=True)
    combined_root.mkdir(parents=True, exist_ok=True)
    try:
        if state is None:
            state = preflight(shared_root, cache_root)
        state.require()
        contract = load_contract(shared_root, state)
        downstream_hash = _hash_files([PROJECT_ROOT / "scripts/run_rq2_wsl.py", PROJECT_ROOT / "scripts/rq2_pipeline.py"])
        simplex_hash = file_sha256(PROJECT_ROOT / "scripts/rq2_pipeline.py")
        run_records.append({"run_id": "rq2_preflight", "run_type": "preflight", "status": "complete", "detail": json.dumps(state.evidence, sort_keys=True)})

        handcrafted = _load_windows_family(contract, "handcrafted")
        temporal = _load_windows_family(contract, "temporal_1dcnn")
        run_records.extend(
            [
                {"run_id": "windows_handcrafted_audit", "run_type": "windows_cache_audit", "status": "complete", "detail": "handcrafted Windows index and NPZ artifacts validated"},
                {"run_id": "windows_temporal_audit", "run_type": "windows_cache_audit", "status": "complete", "detail": "temporal Windows index and NPZ artifacts validated"},
            ]
        )
        pretrained, audit, provenance = _cache_pretrained(contract, cache_root)
        audit["contract_hash"] = contract.contract_hash
        audit["representation_key"] = audit.apply(lambda row: f"pretrained+{row['modality']}+global+deterministic+{row['participant']}+{row['condition']}", axis=1)
        audit["expected_dimension"] = audit["modality"].map({m: EXPECTED_DIMENSIONS[m.lower()] for m in MODALITIES})
        audit["dimension_check"] = audit["representation_dimension"].eq(audit["expected_dimension"])
        audit["finite_valid_values"] = True
        audit["duplicate_key_check"] = False
        audit["window_key_join_check"] = True
        audit["modality_mask_check"] = audit["representation_available"].astype(bool) | audit["fallback_used"].astype(bool)
        audit["cache_sha256"] = provenance["cache_sha256"]
        audit["cache_manifest_sha256"] = provenance["cache_manifest_sha256"]
        audit["source_hashes"] = json.dumps(provenance["metadata"].get("inputs", {}), sort_keys=True)
        audit.to_csv(output_root / "representation_audit.csv", index=False)
        pretrained_index = _materialize_pretrained(contract, pretrained, output_root, audit)
        pretrained_index.to_csv(output_root / "pretrained_representation_index.csv", index=False)
        run_records.append({"run_id": "wsl_pretrained_aggregate_20260822", "run_type": "cached_representation_aggregation", "status": "complete", "detail": "five modalities × 81 contract keys; P004/C6 unavailable"})
        alignment_checks = write_cross_environment_alignment_report(contract, pretrained_index, combined_root)
        if not alignment_checks["all_key_and_metadata_checks_pass"]:
            raise ValueError("Cross-environment representation alignment checks failed")

        representation_frames: list[pd.DataFrame] = []
        parameter_frames: list[pd.DataFrame] = []
        weight_frames: list[pd.DataFrame] = []
        family_specs = (
            ("handcrafted", handcrafted, "handcrafted_simplex_full5"),
            ("temporal_1dcnn", temporal, "temporal_1dcnn_simplex_full5"),
            ("pretrained", pretrained, "frozen_simplex_full5"),
        )
        for family_name, family, model_id in family_specs:
            for seed in EXPECTED_SEEDS:
                predictions, parameters, weights = run_simplex_family(
                    contract,
                    family,
                    seed,
                    model_id,
                    MODALITY_ORDER,
                    downstream_hash,
                    simplex_hash,
                    "representation_track_and_frozen_fusion_track" if model_id == "frozen_simplex_full5" else "representation_track",
                )
                parameters["run_id"] = f"{model_id}_s{seed}"
                parameters["downstream_code_hash"] = downstream_hash
                parameters["simplex_code_hash"] = simplex_hash
                weights["run_id"] = f"{model_id}_s{seed}"
                weights["downstream_code_hash"] = downstream_hash
                weights["simplex_code_hash"] = simplex_hash
                representation_frames.append(predictions)
                parameter_frames.append(parameters)
                weight_frames.append(weights)
                run_records.append({"run_id": f"{model_id}_s{seed}", "run_type": "shared_simplex_representation_track", "status": "complete", "detail": f"{family_name}; seed={seed}; 486 OOF rows"})
        representation_oof = pd.concat(representation_frames, ignore_index=True)
        representation_parameters = pd.concat(parameter_frames, ignore_index=True)
        representation_weights = pd.concat(weight_frames, ignore_index=True)
        representation_oof.to_csv(combined_root / "representation_track_oof_predictions.csv", index=False)
        representation_weights.to_csv(combined_root / "representation_track_simplex_weights.csv", index=False)
        representation_parameters.to_csv(combined_root / "representation_track_selected_parameters.csv", index=False)

        # Keep every representation-track OOF row in the combined table.  The
        # frozen Simplex subset is the exact same object reused for the frozen
        # fusion comparison; it is not trained a second time.
        fusion_frames: list[pd.DataFrame] = [representation_oof.copy()]
        detail_frames: list[pd.DataFrame] = [representation_parameters.copy()]
        for seed in EXPECTED_SEEDS:
            concat_predictions, concat_details = run_concat_ridge(contract, pretrained, seed, "frozen_concat_ridge_full5", downstream_hash, simplex_hash)
            concat_details["run_id"] = f"frozen_concat_ridge_full5_s{seed}"
            concat_details["downstream_code_hash"] = downstream_hash
            concat_details["simplex_code_hash"] = simplex_hash
            fusion_frames.append(concat_predictions)
            detail_frames.append(concat_details)
            run_records.append({"run_id": f"frozen_concat_ridge_full5_s{seed}", "run_type": "frozen_fusion_concat_ridge", "status": "complete", "detail": "frozen pretrained full-five; 486 OOF rows"})
            healnet_predictions, healnet_details = run_healnet(contract, pretrained, seed, "frozen_healnet_full5", downstream_hash, simplex_hash, output_root)
            healnet_details["run_id"] = f"frozen_healnet_full5_s{seed}"
            healnet_details["downstream_code_hash"] = downstream_hash
            healnet_details["simplex_code_hash"] = simplex_hash
            fusion_frames.append(healnet_predictions)
            detail_frames.append(healnet_details)
            run_records.append({"run_id": f"frozen_healnet_full5_s{seed}", "run_type": "frozen_fusion_healnet", "status": "complete", "detail": "frozen pretrained full-five; 486 OOF rows"})
            no_video_predictions, no_video_parameters, no_video_weights = run_simplex_family(
                contract,
                pretrained,
                seed,
                "frozen_simplex_no_video",
                MODALITY_ORDER[:-1],
                downstream_hash,
                simplex_hash,
                "video_audit",
            )
            no_video_parameters["run_id"] = f"frozen_simplex_no_video_s{seed}"
            no_video_parameters["downstream_code_hash"] = downstream_hash
            no_video_parameters["simplex_code_hash"] = simplex_hash
            no_video_weights["run_id"] = f"frozen_simplex_no_video_s{seed}"
            no_video_weights["downstream_code_hash"] = downstream_hash
            no_video_weights["simplex_code_hash"] = simplex_hash
            fusion_frames.append(no_video_predictions)
            detail_frames.append(no_video_parameters)
            run_records.append({"run_id": f"frozen_simplex_no_video_s{seed}", "run_type": "video_audit_no_video", "status": "complete", "detail": "only Video expert removed; 486 OOF rows"})
        combined_oof = pd.concat(fusion_frames, ignore_index=True)
        fusion_details = pd.concat(detail_frames, ignore_index=True)
        combined_oof.to_csv(combined_root / "combined_oof_predictions.csv", index=False)
        fusion_details.to_csv(combined_root / "fusion_details.csv", index=False)
        representation_weights.to_csv(combined_root / "representation_track_simplex_weights.csv", index=False)
        probes = run_video_probes(contract, pretrained, combined_root)
        run_records.append({"run_id": "video_stimulus_and_response_probes", "run_type": "video_secondary_probes", "status": "complete", "detail": "VideoMAE V2 stimulus and residual-response probes"})

        participant_metrics = build_participant_metrics(combined_oof)
        group_summary = build_group_summary(participant_metrics)
        statistics = build_statistical_comparisons(participant_metrics)
        participant_metrics.to_csv(combined_root / "participant_metrics.csv", index=False)
        group_summary.to_csv(combined_root / "group_summary.csv", index=False)
        statistics.to_csv(combined_root / "statistical_comparisons.csv", index=False)
        plot_data, captions = build_plot_data_and_figures(participant_metrics, group_summary, probes, combined_root)
        run_records.extend(
            [
                {"run_id": "participant_level_statistics", "run_type": "bootstrap_sign_flip_holm", "status": "complete", "detail": "10,000 participant bootstraps; 512 exact sign flips; two four-test Holm families"},
                {"run_id": "thesis_figures", "run_type": "figure_generation", "status": "complete", "detail": "four figures sourced from plot_data.csv"},
            ]
        )
        run_manifest = pd.DataFrame(run_records)
        run_manifest["contract_hash"] = contract.contract_hash
        run_manifest["downstream_code_hash"] = downstream_hash
        run_manifest["simplex_code_hash"] = simplex_hash
        run_manifest["package_versions"] = json.dumps(_package_versions(), sort_keys=True)
        run_manifest.to_csv(output_root / "wsl_run_manifest.csv", index=False)
        acceptance = _acceptance_checks(contract, combined_oof, representation_oof, participant_metrics, statistics, pretrained_index, downstream_hash, simplex_hash)
        output_files = [
            output_root / "pretrained_representation_index.csv",
            output_root / "representation_audit.csv",
            output_root / "wsl_run_manifest.csv",
            combined_root / "combined_oof_predictions.csv",
            combined_root / "participant_metrics.csv",
            combined_root / "group_summary.csv",
            combined_root / "statistical_comparisons.csv",
            combined_root / "stimulus_probe_results.csv",
            combined_root / "fusion_details.csv",
            combined_root / "plot_data.csv",
            combined_root / "CROSS_ENVIRONMENT_ALIGNMENT_REPORT.md",
            combined_root / "RQ2_final_report.md",
        ]
        output_files.extend(sorted((combined_root / "figures").glob("*.png")))
        report_path = write_report(contract, state, downstream_hash, simplex_hash, combined_oof, participant_metrics, group_summary, statistics, probes, representation_weights, fusion_details, run_manifest, captions, combined_root)
        if report_path not in output_files:
            output_files.append(report_path)
        for output_file in output_files:
            if not output_file.is_file():
                raise ValueError(f"Required output was not written: {output_file}")
        done_payload = {
            "status": "complete",
            "contract_hash": contract.contract_hash,
            "downstream_code_hash": downstream_hash,
            "simplex_code_hash": simplex_hash,
            "completed_run_ids": sorted(run_manifest[run_manifest["status"].eq("complete")]["run_id"].astype(str).unique()),
            "failed_runs": run_manifest[~run_manifest["status"].eq("complete")]["run_id"].astype(str).tolist(),
            "output_files": [str(path.relative_to(PROJECT_ROOT)) for path in output_files] + ["wsl_results/WSL_DONE.json"],
            "acceptance_checks": acceptance,
            "representation_cache": provenance,
            "package_versions": _package_versions(),
        }
        _write_json(output_root / "WSL_DONE.json", done_payload)
        return done_payload
    except Exception as error:
        _write_shared_fusion_error(combined_root, error, run_records)
        raise


class FrozenHealNetRegressor(nn.Module):
    def __init__(self, input_dimensions: Mapping[str, int], modalities: Sequence[str]) -> None:
        super().__init__()
        from mac.fusion.healnet import HEALNetFusion

        self.modalities = tuple(modalities)
        self.projectors = nn.ModuleDict({modality: nn.Linear(int(input_dimensions[modality]), 256) for modality in modalities})
        self.fusion = HEALNetFusion(
            d_common=256,
            memory_size=16,
            n_layers=2,
            n_heads=4,
            num_modalities=len(modalities),
            dropout=0.1,
        )
        self.head = nn.Linear(256, 2)

    def forward(self, features: Mapping[str, torch.Tensor]) -> torch.Tensor:
        inputs = [self.projectors[modality](features[modality]).unsqueeze(1) for modality in self.modalities]
        masks = [torch.ones((inputs[index].shape[0], 1), dtype=torch.bool, device=inputs[index].device) for index in range(len(inputs))]
        return self.head(self.fusion(inputs, list(self.modalities), masks))


def _set_torch_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_healnet(
    contract: ContractData,
    family: FamilyData,
    seed: int,
    model_id: str,
    downstream_hash: str,
    simplex_hash: str,
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    counts = _window_counts(contract)
    checkpoint_root = output_root / "healnet_checkpoints"
    fallback_mask = (contract.labels["participant"].eq(EXPECTED_FALLBACK[0]) & contract.labels["condition"].eq(EXPECTED_FALLBACK[1])).to_numpy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for fold in contract.folds:
        matrices, present = _matrices_for(family, fold.fold, seed, MODALITY_ORDER)
        preprocessors, scores = fit_preprocessors(matrices, present, fold.train_idx, seed + fold.fold * 1000, MODALITY_ORDER)
        input_dimensions = {modality: int(scores[modality].shape[1]) for modality in MODALITY_ORDER}
        _set_torch_seed(seed + fold.fold * 10000)
        model = FrozenHealNetRegressor(input_dimensions, MODALITY_ORDER).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        train_valid = fold.train_idx[~fallback_mask[fold.train_idx]]
        validation_valid = fold.validation_idx[~fallback_mask[fold.validation_idx]]
        train_features = {m: torch.tensor(scores[m][train_valid], dtype=torch.float32, device=device) for m in MODALITY_ORDER}
        validation_features = {m: torch.tensor(scores[m][validation_valid], dtype=torch.float32, device=device) for m in MODALITY_ORDER}
        train_truth = torch.tensor(
            np.column_stack([
                contract.labels.iloc[train_valid][target].to_numpy(dtype=np.float32) - _cross_fitted_anchors(contract.labels, train_valid, target)
                for target in TARGETS
            ]),
            dtype=torch.float32,
            device=device,
        )
        validation_anchor = np.column_stack([_heldout_anchor_values(contract, fold.fold, validation_valid, target) for target in TARGETS])
        validation_rating = contract.labels.iloc[validation_valid][list(TARGETS)].to_numpy(dtype=np.float64)
        best_state: dict[str, torch.Tensor] | None = None
        best_score = float("inf")
        best_epoch = 0
        stale = 0
        for epoch in range(1, 101):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(train_features) - train_truth) ** 2)
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                validation_residual = model(validation_features).cpu().numpy().astype(np.float64)
            validation_prediction = np.clip(validation_anchor + np.clip(validation_residual, *CORRECTION_CLIP), *PREDICTION_CLIP)
            score = float(np.mean((validation_rating - validation_prediction) ** 2)) if len(validation_valid) else float("inf")
            if score < best_score - 1e-10:
                best_score = score
                best_epoch = epoch
                stale = 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
                if stale >= 8:
                    break
        if best_state is None:
            raise RuntimeError(f"HEALNet produced no checkpoint for fold {fold.fold}")
        model.load_state_dict(best_state)
        model.eval()
        test_valid = fold.test_idx[~fallback_mask[fold.test_idx]]
        test_features = {m: torch.tensor(scores[m][test_valid], dtype=torch.float32, device=device) for m in MODALITY_ORDER}
        with torch.no_grad():
            test_residual_valid = model(test_features).cpu().numpy().astype(np.float64)
        test_residual_map = {int(index): test_residual_valid[position] for position, index in enumerate(test_valid)}
        test_anchors = {target: _heldout_anchor_values(contract, fold.fold, fold.test_idx, target) for target in TARGETS}
        for local_index, row_index in enumerate(fold.test_idx):
            key = (str(contract.labels.iloc[row_index].participant), str(contract.labels.iloc[row_index].condition))
            for target_index, target in enumerate(TARGETS):
                residual = 0.0 if int(row_index) not in test_residual_map else float(np.clip(test_residual_map[int(row_index)][target_index], *CORRECTION_CLIP))
                anchor = float(test_anchors[target][local_index])
                prediction = float(np.clip(anchor + residual, *PREDICTION_CLIP))
                prediction_rows.append(
                    _record_prediction(
                        contract,
                        int(row_index),
                        fold,
                        seed,
                        target,
                        model_id,
                        "pretrained",
                        "healnet",
                        f"{model_id}_s{seed}",
                        anchor,
                        residual,
                        anchor + residual,
                        prediction,
                        counts.get(key, 0),
                        key == EXPECTED_FALLBACK,
                        downstream_hash,
                        simplex_hash,
                        "frozen_fusion_track",
                    )
                )
        checkpoint_path = checkpoint_root / f"fold_{fold.fold:02d}" / f"seed_{seed}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "fold": fold.fold, "seed": seed}, checkpoint_path)
        detail_rows.extend(
            {
                "fold": fold.fold,
                "seed": seed,
                "target": target,
                "model_id": model_id,
                "representation_family": "pretrained",
                "input_dimensions": json.dumps(input_dimensions, sort_keys=True),
                "selected_pca_dimensions": json.dumps({m: int(preprocessors[m].output_dimension) for m in MODALITY_ORDER}, sort_keys=True),
                "ridge_setting": "",
                "simplex_weights": "",
                "correction_scale": 1.0,
                "active_modalities": "+".join(MODALITY_ORDER),
                "HEALNet_stopping_epoch": int(best_epoch),
                "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
                "training_status": "complete",
                "validation_mae": float(best_score),
                "checkpoint": str(checkpoint_path),
            }
            for target in TARGETS
        )
    return pd.DataFrame(prediction_rows).sort_values(["seed", "fold", "target", "participant", "condition"]).reset_index(drop=True), pd.DataFrame(detail_rows)

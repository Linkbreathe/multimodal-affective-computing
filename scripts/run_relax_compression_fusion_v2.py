"""Run the preregistered July 18 five-modality frozen-feature experiments.

The foundation encoders are never updated.  Compression is fitted to aligned
windows from the outer-training participants with equal total weight for every
participant-condition observation, then pooled to the 81 label units.  Every
normalizer, unsupervised or supervised projection, Ridge head, expert weight,
and validation choice excludes the outer test participant.
"""

from __future__ import annotations

# ruff: noqa: E402

import argparse
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
import warnings

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from scripts.run_relax_condition_anchor_probe import (
    ALLOWED_SEEDS,
    EXPECTED_CACHE_SHA256,
    EXPECTED_OBSERVATIONS,
    EXPECTED_PARTICIPANTS,
    EXPECTED_VALID_WINDOWS,
    _git_state,
    _hash_array,
    _package_versions,
    _write_json,
    cross_fitted_condition_anchors,
    heldout_condition_anchors,
    validate_contract,
)
from scripts.run_relax_foundation_probe import TARGETS, Fold, _fold_indexes, _metrics, file_sha256
from mac.data.relax_dataset import RelaxConditionEmbeddingDataset
from mac.fusion.frozen_compression_v2 import (
    FIVE_MODALITIES,
    JointBlockBalancedWindowPCA,
    LinearGCCASharedPrivate,
    ModalityWindowPCABank,
    TargetwiseModalityPLS1,
    fit_modality_bank,
)


METHODS = (
    "joint_block_balanced_pca12",
    "modality_rank_alloc_pca12",
    "targetwise_modality_pls1",
    "linear_gcca_shared_private",
    "modality_expert_simplex5",
)
PRIMARY_METHOD = "modality_expert_simplex5"
FULL_MODALITIES = tuple(FIVE_MODALITIES)
ABLATION_VARIANTS = ("no_eeg", "no_ecg", "no_eye", "no_head", "no_video")
VARIANTS = ("full", *ABLATION_VARIANTS)
DEFAULT_CONTRACT_DIR = Path(
    "/mnt/c/Users/linki/amaster/data_collection_v3/analysis/real_time_inference/"
    "artifacts/cross_project_alignment_2026-07-16/eeg_eligible_ablation/contract"
)
DEFAULT_CACHE = ROOT / "artifacts/relax/aligned_20260716/condition_embeddings.pt"
DEFAULT_OUTPUT_ROOT = (
    ROOT / "artifacts/relax/foundation_compression_fusion_reinvestigation_20260718"
)


def variant_modalities(variant: str) -> tuple[str, ...]:
    if variant == "full":
        return FULL_MODALITIES
    if variant not in ABLATION_VARIANTS:
        raise ValueError(f"Unknown modality variant: {variant}")
    missing = variant.removeprefix("no_")
    return tuple(modality for modality in FULL_MODALITIES if modality != missing)


def _method_spec(payload: Mapping[str, Any], name: str) -> dict[str, Any]:
    records = {str(record["name"]): dict(record) for record in payload["methods"]}
    if name not in records:
        raise ValueError(f"Method {name!r} was not preregistered")
    return records[name]


def _validate_preregistration(args: argparse.Namespace, payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != "relax_foundation_compression_preregistration_v2":
        raise ValueError("Unexpected July18 preregistration schema")
    if not payload.get("frozen_before_candidate_outcomes"):
        raise ValueError("Preregistration is not marked frozen before candidate outcomes")
    protocol = payload["protocol"]
    if tuple(protocol["participants"]) != EXPECTED_PARTICIPANTS:
        raise ValueError("Preregistered cohort changed")
    if tuple(int(value) for value in protocol["seeds"]) != ALLOWED_SEEDS:
        raise ValueError("Preregistered seed set changed")
    if int(protocol["observations"]) != EXPECTED_OBSERVATIONS:
        raise ValueError("Preregistered observation count changed")
    if int(protocol["common_valid_windows"]) != EXPECTED_VALID_WINDOWS:
        raise ValueError("Preregistered common-window count changed")
    if args.seed not in ALLOWED_SEEDS:
        raise ValueError("Seed is outside the fixed protocol")
    if args.method not in METHODS:
        raise ValueError(f"Unsupported method: {args.method}")
    if args.variant != "full" and args.method != PRIMARY_METHOD:
        raise ValueError("Reduced modalities are permitted only as primary-method ablations")
    if tuple(payload["input_contract"]["formal_modalities"]) != FULL_MODALITIES:
        raise ValueError("The formal five-modality order changed")
    rules = payload["common_model_rules"]
    if any(float(value) <= 0.0 for value in rules["positive_gammas"]):
        raise ValueError("Every formal gamma must be strictly positive")
    if int(rules["total_latent_budget"]) != 12:
        raise ValueError("The formal latent budget must remain 12")
    expected = payload["input_contract"]
    observed = {
        "embedding_cache_sha256": file_sha256(args.embedding_cache),
        "labels_sha256": file_sha256(args.labels),
        "windows_sha256": file_sha256(args.windows),
        "split_manifest_sha256": file_sha256(args.split_manifest),
        "common_mask_sha256": file_sha256(args.mask_manifest),
        "cohorts_sha256": file_sha256(args.cohorts),
    }
    for key, value in observed.items():
        if expected.get(key) != value:
            raise ValueError(f"Preregistered input hash mismatch for {key}: {value}")
    if args.variant != "full":
        ablations = payload["modality_ablations"]
        declared = tuple(
            str(record["variant"]) for record in ablations
        ) if isinstance(ablations, list) else tuple(str(value) for value in ablations["variants"])
        if args.variant not in declared:
            raise ValueError(f"Ablation {args.variant} was not preregistered")
    return _method_spec(payload, args.method)


def _load_inputs(
    args: argparse.Namespace,
) -> tuple[
    RelaxConditionEmbeddingDataset,
    pd.DataFrame,
    list[str],
    list[Fold],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    labels, participants, folds, contract = validate_contract(
        SimpleNamespace(
            labels=args.labels,
            windows=args.windows,
            split_manifest=args.split_manifest,
            mask_manifest=args.mask_manifest,
            cohorts=args.cohorts,
            embedding_cache=args.embedding_cache,
            expected_cache_sha256=EXPECTED_CACHE_SHA256,
            cohort="eeg_eligible",
        )
    )
    dataset = RelaxConditionEmbeddingDataset(
        args.embedding_cache,
        modalities=FULL_MODALITIES,
        participants=participants,
        mask_manifest=args.mask_manifest,
        strict=True,
    )
    if not dataset.metadata.get("cuda_used") or not str(dataset.metadata.get("device", "")).startswith("cuda"):
        raise ValueError("The frozen cache does not prove CUDA encoder extraction")
    blocks = {
        modality: dataset.embeddings[modality].numpy().astype(np.float64, copy=False)
        for modality in FULL_MODALITIES
    }
    masks = {
        modality: dataset.masks[modality].numpy().astype(bool, copy=False)
        for modality in FULL_MODALITIES
    }
    shared_mask = masks[FULL_MODALITIES[0]].copy()
    for modality in FULL_MODALITIES[1:]:
        if not np.array_equal(shared_mask, masks[modality]):
            raise ValueError("Formal external masks must create identical aligned windows")
    common_valid = shared_mask.sum(axis=1).astype(int)
    if int(common_valid.sum()) != EXPECTED_VALID_WINDOWS:
        raise ValueError("The dataset does not contain exactly 545 common-valid windows")
    availability = {modality: common_valid > 0 for modality in FULL_MODALITIES}
    if any(not np.array_equal(availability[FULL_MODALITIES[0]], availability[m]) for m in FULL_MODALITIES[1:]):
        raise ValueError("Formal modality availability must be shared")

    windows = pd.read_csv(args.windows)
    windows = windows.loc[windows["participant_id"].astype(str).isin(participants)].copy()
    source_lookup = windows.groupby(["participant_id", "condition"]).size()
    source_windows = np.asarray(
        [int(source_lookup.loc[key]) for key in dataset.condition_keys()], dtype=int
    )
    if (source_windows <= 0).any() or (common_valid > source_windows).any():
        raise ValueError("Common/source window counts are inconsistent")
    quality = common_valid / source_windows.astype(np.float64)
    frame = pd.DataFrame(
        {
            "participant_id": dataset.participant_ids,
            "condition": dataset.conditions,
            "presentation_position": dataset.presentation_positions.numpy(),
            "relaxation": dataset.targets[:, 0].numpy(),
            "discomfort": dataset.targets[:, 1].numpy(),
            "common_valid_windows": common_valid,
            "source_windows": source_windows,
            "quality": quality,
            "presence": common_valid > 0,
        }
    )
    label_lookup = labels.set_index(["participant_id", "condition"])
    if set(dataset.condition_keys()) != set(label_lookup.index):
        raise ValueError("Cache and contract label keys differ")
    for row in frame.itertuples(index=False):
        expected = label_lookup.loc[(row.participant_id, row.condition), list(TARGETS)].to_numpy(dtype=float)
        if not np.allclose(expected, [row.relaxation, row.discomfort], atol=1e-6, rtol=0.0):
            raise ValueError(f"Cache target mismatch for {(row.participant_id, row.condition)}")
    zero = frame.loc[~frame["presence"], ["participant_id", "condition"]]
    if zero.astype(str).apply(tuple, axis=1).tolist() != [("P004", "C6")]:
        raise ValueError("Expected P004/C6 to be the sole all-missing observation")
    feature_metadata = {
        "modalities": {
            modality: {
                "embedding_dimension": int(blocks[modality].shape[2]),
                "effective_valid_windows": int(masks[modality].sum()),
                "zero_window_observations": int((~availability[modality]).sum()),
                "window_tensor_sha256": _hash_array(blocks[modality]),
                "mask_sha256": _hash_array(masks[modality]),
            }
            for modality in FULL_MODALITIES
        },
        "raw_coordinate_total": int(sum(blocks[m].shape[2] for m in FULL_MODALITIES)),
        "common_valid_windows": int(common_valid.sum()),
        "quality_definition": "common_valid_windows/source_windows",
        "quality_sha256": _hash_array(quality),
        "all_missing_observation": ["P004", "C6"],
    }
    return (
        dataset,
        frame,
        participants,
        folds,
        blocks,
        masks,
        availability,
        common_valid > 0,
        quality,
        {**contract, "features": feature_metadata},
    )


def _residual_targets(frame: pd.DataFrame, train: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    anchors: dict[str, np.ndarray] = {}
    columns: list[np.ndarray] = []
    for target in TARGETS:
        anchor = cross_fitted_condition_anchors(frame, train, target)
        anchors[target] = anchor
        columns.append(frame.iloc[train][target].to_numpy(dtype=float) - anchor)
    return np.column_stack(columns), anchors


def _heldout_anchors(frame: pd.DataFrame, train: np.ndarray, rows: np.ndarray) -> dict[str, np.ndarray]:
    return {target: heldout_condition_anchors(frame, train, rows, target) for target in TARGETS}


def _selection_key(macro_mae: float, alpha: float, gamma: float) -> tuple[float, float, float]:
    return float(macro_mae), -float(alpha), float(gamma)


def _augment_and_scale(
    latent: np.ndarray,
    presence: np.ndarray,
    quality: np.ndarray,
    train: np.ndarray,
) -> tuple[np.ndarray, StandardScaler]:
    raw = np.column_stack([np.asarray(latent, dtype=float), presence.astype(float), quality])
    scaler = StandardScaler().fit(raw[train])
    transformed = scaler.transform(raw)
    if not np.isfinite(transformed).all():
        raise ValueError("Latent representation is non-finite")
    return transformed, scaler


def _cache_key(kind: str, train: np.ndarray, participant_ids: np.ndarray) -> str:
    participants = sorted(set(participant_ids[np.asarray(train, dtype=int)]))
    digest = sha256()
    digest.update(kind.encode())
    digest.update(EXPECTED_CACHE_SHA256.encode())
    digest.update(file_sha256(ROOT / "src/fusion/frozen_compression_v2.py").encode())
    digest.update("|".join(participants).encode())
    return f"{kind}_{digest.hexdigest()[:24]}"


def _load_or_fit_modality_bank(
    blocks: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    train: np.ndarray,
    participant_ids: np.ndarray,
    cache_dir: Path | None,
) -> ModalityWindowPCABank:
    path = None if cache_dir is None else cache_dir / "modality_banks" / f"{_cache_key('modality12', train, participant_ids)}.joblib"
    if path is not None and path.is_file():
        bank = joblib.load(path)
        if not isinstance(bank, ModalityWindowPCABank):
            raise TypeError(f"Unexpected cached modality-bank type: {path}")
        if bank.fit_indexes_.tolist() != np.asarray(train, dtype=int).tolist():
            raise ValueError("Cached modality bank was fitted on different observation indexes")
        return bank
    bank = fit_modality_bank(
        blocks,
        masks,
        train,
        modalities=FULL_MODALITIES,
        max_components_per_modality=12,
        participant_ids=participant_ids,
    )
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bank, path)
    return bank


def _load_or_fit_joint_bank(
    blocks: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    train: np.ndarray,
    participant_ids: np.ndarray,
    cache_dir: Path | None,
) -> JointBlockBalancedWindowPCA:
    path = None if cache_dir is None else cache_dir / "joint_banks" / f"{_cache_key('joint12', train, participant_ids)}.joblib"
    if path is not None and path.is_file():
        bank = joblib.load(path)
        if not isinstance(bank, JointBlockBalancedWindowPCA):
            raise TypeError(f"Unexpected cached joint-bank type: {path}")
        if bank.fit_indexes_.tolist() != np.asarray(train, dtype=int).tolist():
            raise ValueError("Cached joint bank was fitted on different observation indexes")
        return bank
    bank = JointBlockBalancedWindowPCA(FULL_MODALITIES, n_components=12).fit(
        blocks, masks, train, participant_ids=participant_ids
    )
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bank, path)
    return bank


def _standard_representations(
    method: str,
    blocks: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    availability: Mapping[str, np.ndarray],
    presence: np.ndarray,
    quality: np.ndarray,
    frame: pd.DataFrame,
    train: np.ndarray,
    cache_dir: Path | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    participant_ids = frame["participant_id"].astype(str).to_numpy()
    residual, _ = _residual_targets(frame, train)
    full_residual = np.zeros((len(frame), len(TARGETS)), dtype=float)
    full_residual[train] = residual
    fitted: dict[str, Any] = {}
    if method == "joint_block_balanced_pca12":
        compressor = _load_or_fit_joint_bank(
            blocks, masks, train, participant_ids, cache_dir
        )
        base = compressor.transform(blocks, masks)
        latent, scaler = _augment_and_scale(base, presence, quality, train)
        representations = {target: latent for target in TARGETS}
        provenance = compressor.provenance()
        fitted = {"compressor": compressor, "latent_scalers": {"shared": scaler}}
    else:
        bank = _load_or_fit_modality_bank(
            blocks, masks, train, participant_ids, cache_dir
        )
        allocation = bank.allocation_for(
            FULL_MODALITIES, total_components=12, variant_name="full_five"
        )
        scores = bank.transform_condition_blocks(
            blocks, masks, allocation=allocation, modalities=FULL_MODALITIES
        )
        if method == "modality_rank_alloc_pca12":
            base = np.concatenate([scores[m] for m in FULL_MODALITIES], axis=1)
            latent, scaler = _augment_and_scale(base, presence, quality, train)
            representations = {target: latent for target in TARGETS}
            fitted = {"compressor": bank, "latent_scalers": {"shared": scaler}}
            provenance = bank.provenance()
        elif method == "targetwise_modality_pls1":
            supervised = TargetwiseModalityPLS1(FULL_MODALITIES, TARGETS).fit(
                scores,
                full_residual,
                train,
                availability=availability,
                participant_ids=participant_ids,
            )
            projected = supervised.transform(scores, availability=availability)
            representations = {}
            scalers: dict[str, StandardScaler] = {}
            for target in TARGETS:
                representations[target], scalers[target] = _augment_and_scale(
                    projected[target], presence, quality, train
                )
            fitted = {"compressor": bank, "supervised": supervised, "latent_scalers": scalers}
            provenance = {
                "method": method,
                "modality_pca": bank.provenance(),
                "supervised_projection": supervised.provenance(),
            }
        elif method == "linear_gcca_shared_private":
            gcca = LinearGCCASharedPrivate(FULL_MODALITIES, shared_components=2).fit(
                scores, train, availability=availability, participant_ids=participant_ids
            )
            base = gcca.transform(scores, availability=availability)
            latent, scaler = _augment_and_scale(base, presence, quality, train)
            representations = {target: latent for target in TARGETS}
            fitted = {"compressor": bank, "gcca": gcca, "latent_scalers": {"shared": scaler}}
            provenance = {
                "method": method,
                "modality_pca": bank.provenance(),
                "shared_private": gcca.provenance(),
            }
        else:
            raise ValueError(f"Unsupported standard method: {method}")
    return representations, provenance, fitted


def _fit_standard_fold(
    method: str,
    blocks: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    availability: Mapping[str, np.ndarray],
    presence: np.ndarray,
    quality: np.ndarray,
    frame: pd.DataFrame,
    fold: Fold,
    rules: Mapping[str, Any],
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    participants = frame["participant_id"].astype(str).to_numpy()
    train, validation, test = _fold_indexes(participants, fold)
    if (len(train), len(validation), len(test)) != (63, 9, 9):
        raise ValueError("Formal outer fold is not 63/9/9")
    residual, train_anchor = _residual_targets(frame, train)
    validation_anchor = _heldout_anchors(frame, train, validation)
    test_anchor = _heldout_anchors(frame, train, test)
    representations, provenance, fitted = _standard_representations(
        method, blocks, masks, availability, presence, quality, frame, train, cache_dir
    )
    cap = float(rules["residual_cap"])
    candidates: list[dict[str, Any]] = []
    for alpha in (float(value) for value in rules["ridge_alphas"]):
        models: dict[str, Ridge] = {}
        validation_corrections: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        test_corrections: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for target_index, target in enumerate(TARGETS):
            model = Ridge(alpha=alpha).fit(representations[target][train], residual[:, target_index])
            models[target] = model
            validation_raw = np.asarray(model.predict(representations[target][validation]), dtype=float)
            test_raw = np.asarray(model.predict(representations[target][test]), dtype=float)
            validation_raw[~presence[validation]] = 0.0
            test_raw[~presence[test]] = 0.0
            validation_corrections[target] = (validation_raw, np.clip(validation_raw, -cap, cap))
            test_corrections[target] = (test_raw, np.clip(test_raw, -cap, cap))
        for gamma in (float(value) for value in rules["positive_gammas"]):
            predictions = {
                target: np.clip(validation_anchor[target] + gamma * validation_corrections[target][1], 0.0, 1.0)
                for target in TARGETS
            }
            target_mae = {
                target: float(np.mean(np.abs(frame.iloc[validation][target].to_numpy(dtype=float) - predictions[target])))
                for target in TARGETS
            }
            candidates.append(
                {
                    "key": _selection_key(np.mean(list(target_mae.values())), alpha, gamma),
                    "alpha": alpha,
                    "gamma": gamma,
                    "models": models,
                    "validation_corrections": validation_corrections,
                    "test_corrections": test_corrections,
                    "validation_predictions": predictions,
                    "validation_target_mae": target_mae,
                }
            )
    selected = min(candidates, key=lambda record: record["key"])
    test_predictions = {
        target: np.clip(test_anchor[target] + selected["gamma"] * selected["test_corrections"][target][1], 0.0, 1.0)
        for target in TARGETS
    }
    allocation = (
        provenance.get("rank_allocation")
        or provenance.get("modality_pca", {}).get("rank_allocation")
        or provenance.get("modality_pca", {}).get("rank_allocation", {})
    )
    components = allocation.get("components", {}) if isinstance(allocation, Mapping) else {}
    return {
        "fold": fold,
        "train_indexes": train,
        "validation_indexes": validation,
        "test_indexes": test,
        "alpha": float(selected["alpha"]),
        "gamma": float(selected["gamma"]),
        "models": selected["models"],
        "fitted": fitted,
        "compression_provenance": provenance,
        "modality_components": dict(components),
        "validation_anchor": validation_anchor,
        "test_anchor": test_anchor,
        "validation_corrections": selected["validation_corrections"],
        "test_corrections": selected["test_corrections"],
        "validation_predictions": selected["validation_predictions"],
        "test_predictions": test_predictions,
        "validation_target_mae": selected["validation_target_mae"],
        "validation_macro_mae": float(np.mean(list(selected["validation_target_mae"].values()))),
        "train_anchor_sha256": {target: _hash_array(train_anchor[target]) for target in TARGETS},
        "modalities": FULL_MODALITIES,
    }


def _solve_expert_weights(predictions: np.ndarray, truths: np.ndarray, floor: float) -> np.ndarray:
    matrix = np.asarray(predictions, dtype=float)
    target = np.asarray(truths, dtype=float)
    n_experts = matrix.shape[1]
    if matrix.ndim != 2 or target.shape != (len(matrix),):
        raise ValueError("Invalid expert-weight training arrays")
    lower = np.full(n_experts, float(floor), dtype=float)
    if float(lower.sum()) >= 1.0:
        raise ValueError("Expert weight floor leaves no feasible simplex")
    initial = lower + (1.0 - lower.sum()) / n_experts
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"scipy\.optimize")
        solution = minimize(
            lambda weights: float(np.mean(np.square(target - matrix @ weights))),
            initial,
            method="SLSQP",
            bounds=[(float(value), 1.0) for value in lower],
            constraints={"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
            options={"ftol": 1e-12, "maxiter": 2000},
        )
    if not solution.success:
        raise RuntimeError(f"Expert simplex optimization failed: {solution.message}")
    weights = np.maximum(np.asarray(solution.x, dtype=float), lower)
    excess = weights - lower
    free = 1.0 - float(lower.sum())
    weights = lower + excess * (free / float(excess.sum())) if float(excess.sum()) > 0 else lower + free / n_experts
    if not np.isclose(weights.sum(), 1.0, atol=1e-9) or np.any(weights < lower - 1e-9):
        raise ValueError("Expert weights violate the preregistered simplex")
    return weights


def _weighted_experts(corrections: np.ndarray, available: np.ndarray, weights: np.ndarray) -> np.ndarray:
    active = np.asarray(available, dtype=bool).astype(float) * np.asarray(weights, dtype=float)[None, :]
    denominator = active.sum(axis=1)
    output = np.zeros(len(active), dtype=float)
    valid = denominator > 0
    output[valid] = (np.asarray(corrections)[valid] * active[valid]).sum(axis=1) / denominator[valid]
    return output


def _fit_expert_bank_cached(
    blocks: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    train: np.ndarray,
    modalities: tuple[str, ...],
    participants: np.ndarray,
    cache_dir: Path | None,
) -> tuple[ModalityWindowPCABank, dict[str, np.ndarray], dict[str, int]]:
    bank = _load_or_fit_modality_bank(
        blocks, masks, train, participants, cache_dir
    )
    variant_name = "full" if modalities == FULL_MODALITIES else "no_" + next(
        modality for modality in FULL_MODALITIES if modality not in modalities
    )
    allocation = bank.allocation_for(
        modalities, total_components=12, variant_name=variant_name
    )
    scores = bank.transform_condition_blocks(
        blocks, masks, allocation=allocation, modalities=modalities
    )
    return bank, scores, allocation


def _inner_expert_data(
    blocks: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    availability: Mapping[str, np.ndarray],
    modalities: tuple[str, ...],
    frame: pd.DataFrame,
    outer_train: np.ndarray,
    cache_dir: Path | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    participant_ids = frame["participant_id"].astype(str).to_numpy()
    records: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for heldout_participant in sorted(set(participant_ids[outer_train])):
        heldout = outer_train[participant_ids[outer_train] == heldout_participant]
        inner_train = outer_train[participant_ids[outer_train] != heldout_participant]
        if (len(inner_train), len(heldout)) != (54, 9):
            raise ValueError("Inner expert LOPO is not 54/9")
        bank, scores, allocation = _fit_expert_bank_cached(
            blocks, masks, inner_train, modalities, participant_ids, cache_dir
        )
        residual, _ = _residual_targets(frame, inner_train)
        anchors = _heldout_anchors(frame, inner_train, heldout)
        truths = {
            target: frame.iloc[heldout][target].to_numpy(dtype=float) - anchors[target]
            for target in TARGETS
        }
        records.append(
            {
                "inner_train": inner_train,
                "heldout": heldout,
                "scores": scores,
                "residual": residual,
                "truths": truths,
            }
        )
        provenance.append(
            {
                "heldout_participant": heldout_participant,
                "fit_participants": sorted(set(participant_ids[inner_train])),
                "fit_observations": int(len(inner_train)),
                "heldout_observations": int(len(heldout)),
                "compression": bank.provenance(),
                "modality_components": allocation,
            }
        )
    covered = np.concatenate([record["heldout"] for record in records])
    if set(covered) != set(outer_train) or len(covered) != len(set(covered)):
        raise ValueError("Inner expert OOF coverage is incomplete or duplicated")
    return records, provenance


def _expert_oof(
    inner: Sequence[Mapping[str, Any]],
    availability: Mapping[str, np.ndarray],
    modalities: tuple[str, ...],
    outer_train: np.ndarray,
    alpha: float,
    cap: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    lookup = {int(index): position for position, index in enumerate(outer_train)}
    predictions = {target: np.zeros((len(outer_train), len(modalities)), dtype=float) for target in TARGETS}
    truths = {target: np.zeros(len(outer_train), dtype=float) for target in TARGETS}
    for record in inner:
        heldout = np.asarray(record["heldout"], dtype=int)
        inner_train = np.asarray(record["inner_train"], dtype=int)
        positions = np.asarray([lookup[int(index)] for index in heldout], dtype=int)
        for target_index, target in enumerate(TARGETS):
            truths[target][positions] = record["truths"][target]
            for modality_index, modality in enumerate(modalities):
                score = record["scores"][modality]
                model = Ridge(alpha=alpha).fit(score[inner_train], record["residual"][:, target_index])
                correction = np.asarray(model.predict(score[heldout]), dtype=float)
                correction[~availability[modality][heldout]] = 0.0
                predictions[target][positions, modality_index] = np.clip(correction, -cap, cap)
    return predictions, truths


def _fit_expert_fold(
    blocks: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    availability: Mapping[str, np.ndarray],
    presence: np.ndarray,
    frame: pd.DataFrame,
    fold: Fold,
    modalities: tuple[str, ...],
    rules: Mapping[str, Any],
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    participant_ids = frame["participant_id"].astype(str).to_numpy()
    train, validation, test = _fold_indexes(participant_ids, fold)
    residual, train_anchor = _residual_targets(frame, train)
    validation_anchor = _heldout_anchors(frame, train, validation)
    test_anchor = _heldout_anchors(frame, train, test)
    bank, scores, allocation = _fit_expert_bank_cached(
        blocks, masks, train, modalities, participant_ids, cache_dir
    )
    inner, inner_provenance = _inner_expert_data(
        blocks, masks, availability, modalities, frame, train, cache_dir
    )
    cap = float(rules["residual_cap"])
    floor = float(rules["expert_weight_floor"])
    candidates: list[dict[str, Any]] = []
    for alpha in (float(value) for value in rules["ridge_alphas"]):
        inner_predictions, inner_truths = _expert_oof(
            inner, availability, modalities, train, alpha, cap
        )
        models: dict[str, dict[str, Ridge]] = {target: {} for target in TARGETS}
        weights: dict[str, np.ndarray] = {}
        validation_experts: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {target: {} for target in TARGETS}
        test_experts: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {target: {} for target in TARGETS}
        validation_combined: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        test_combined: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for target_index, target in enumerate(TARGETS):
            weights[target] = _solve_expert_weights(inner_predictions[target], inner_truths[target], floor)
            validation_raw = np.zeros((len(validation), len(modalities)), dtype=float)
            validation_bounded = np.zeros_like(validation_raw)
            test_raw = np.zeros((len(test), len(modalities)), dtype=float)
            test_bounded = np.zeros_like(test_raw)
            for modality_index, modality in enumerate(modalities):
                model = Ridge(alpha=alpha).fit(scores[modality][train], residual[:, target_index])
                models[target][modality] = model
                vr = np.asarray(model.predict(scores[modality][validation]), dtype=float)
                tr = np.asarray(model.predict(scores[modality][test]), dtype=float)
                vr[~availability[modality][validation]] = 0.0
                tr[~availability[modality][test]] = 0.0
                vb, tb = np.clip(vr, -cap, cap), np.clip(tr, -cap, cap)
                validation_experts[target][modality] = (vr, vb)
                test_experts[target][modality] = (tr, tb)
                validation_raw[:, modality_index], validation_bounded[:, modality_index] = vr, vb
                test_raw[:, modality_index], test_bounded[:, modality_index] = tr, tb
            val_available = np.column_stack([availability[m][validation] for m in modalities])
            test_available = np.column_stack([availability[m][test] for m in modalities])
            validation_combined[target] = (
                _weighted_experts(validation_raw, val_available, weights[target]),
                _weighted_experts(validation_bounded, val_available, weights[target]),
            )
            test_combined[target] = (
                _weighted_experts(test_raw, test_available, weights[target]),
                _weighted_experts(test_bounded, test_available, weights[target]),
            )
        for gamma in (float(value) for value in rules["positive_gammas"]):
            predictions = {
                target: np.clip(validation_anchor[target] + gamma * validation_combined[target][1], 0.0, 1.0)
                for target in TARGETS
            }
            target_mae = {
                target: float(np.mean(np.abs(frame.iloc[validation][target].to_numpy(dtype=float) - predictions[target])))
                for target in TARGETS
            }
            candidates.append(
                {
                    "key": _selection_key(np.mean(list(target_mae.values())), alpha, gamma),
                    "alpha": alpha,
                    "gamma": gamma,
                    "models": models,
                    "weights": weights,
                    "inner_predictions": inner_predictions,
                    "inner_truths": inner_truths,
                    "validation_experts": validation_experts,
                    "test_experts": test_experts,
                    "validation_combined": validation_combined,
                    "test_combined": test_combined,
                    "validation_predictions": predictions,
                    "validation_target_mae": target_mae,
                }
            )
    selected = min(candidates, key=lambda record: record["key"])
    test_predictions = {
        target: np.clip(test_anchor[target] + selected["gamma"] * selected["test_combined"][target][1], 0.0, 1.0)
        for target in TARGETS
    }
    return {
        "fold": fold,
        "train_indexes": train,
        "validation_indexes": validation,
        "test_indexes": test,
        "alpha": float(selected["alpha"]),
        "gamma": float(selected["gamma"]),
        "models": selected["models"],
        "weights": selected["weights"],
        "fitted": {"compressor": bank},
        "compression_provenance": bank.provenance(),
        "modality_components": dict(allocation),
        "inner_provenance": inner_provenance,
        "inner_oof_sha256": {
            target: {
                "predictions": _hash_array(selected["inner_predictions"][target]),
                "truths": _hash_array(selected["inner_truths"][target]),
            }
            for target in TARGETS
        },
        "validation_anchor": validation_anchor,
        "test_anchor": test_anchor,
        "validation_corrections": selected["validation_combined"],
        "test_corrections": selected["test_combined"],
        "validation_experts": selected["validation_experts"],
        "test_experts": selected["test_experts"],
        "validation_predictions": selected["validation_predictions"],
        "test_predictions": test_predictions,
        "validation_target_mae": selected["validation_target_mae"],
        "validation_macro_mae": float(np.mean(list(selected["validation_target_mae"].values()))),
        "train_anchor_sha256": {target: _hash_array(train_anchor[target]) for target in TARGETS},
        "modalities": modalities,
        "weight_floor": floor,
    }


def _eligibility(method: str, variant: str, result: Mapping[str, Any], presence: np.ndarray) -> dict[str, Any]:
    test = np.asarray(result["test_indexes"], dtype=int)
    nonzero = {
        target: bool(np.any(np.abs(result["test_corrections"][target][1][presence[test]]) > 1e-12))
        for target in TARGETS
    }
    evidence: dict[str, Any] = {
        "modalities": list(result["modalities"]),
        "modality_count": len(result["modalities"]),
        "both_targets_learned": all(target in result["models"] for target in TARGETS),
        "gamma_strictly_positive": float(result["gamma"]) > 0.0,
        "condition_fallback_used": False,
        "nonzero_test_correction_for_both_targets": all(nonzero.values()),
        "all_missing_test_rows": int((~presence[test]).sum()),
        "primary_claim_eligible": method == PRIMARY_METHOD and variant == "full",
        "ablation_only": variant != "full",
    }
    if method == PRIMARY_METHOD:
        floor = float(result["weight_floor"])
        evidence["all_active_expert_weights_meet_floor"] = all(
            np.all(np.asarray(result["weights"][target]) >= floor - 1e-9) for target in TARGETS
        )
        evidence["all_active_expert_branches_trained"] = all(
            set(result["models"][target]) == set(result["modalities"]) for target in TARGETS
        )
        evidence["feature_coefficients_nonzero_for_both_targets"] = all(
            all(
                float(np.linalg.norm(result["models"][target][modality].coef_)) > 1e-12
                for modality in result["modalities"]
            )
            for target in TARGETS
        )
    else:
        evidence["all_active_expert_weights_meet_floor"] = True
        evidence["all_active_expert_branches_trained"] = True
        evidence["feature_coefficients_nonzero_for_both_targets"] = all(
            float(np.linalg.norm(np.asarray(result["models"][target].coef_)[:-2])) > 1e-12
            for target in TARGETS
        )
    evidence["eligible"] = bool(
        evidence["modality_count"] >= 2
        and evidence["both_targets_learned"]
        and evidence["gamma_strictly_positive"]
        and not evidence["condition_fallback_used"]
        and evidence["nonzero_test_correction_for_both_targets"]
        and evidence["all_active_expert_weights_meet_floor"]
        and evidence["all_active_expert_branches_trained"]
        and evidence["feature_coefficients_nonzero_for_both_targets"]
    )
    return evidence


def _prediction_rows(
    method: str,
    variant: str,
    seed: int,
    frame: pd.DataFrame,
    result: Mapping[str, Any],
    indexes: np.ndarray,
    split: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prediction_key = f"{split}_predictions"
    anchor_key = f"{split}_anchor"
    correction_key = f"{split}_corrections"
    expert_key = f"{split}_experts"
    for local_index, row_index in enumerate(indexes):
        source = frame.iloc[int(row_index)]
        record: dict[str, Any] = {
            "participant_id": str(source.participant_id),
            "condition": str(source.condition),
            "presentation_position": float(source.presentation_position),
            "fold_index": int(result["fold"].fold_index),
            "validation_participant": str(result["fold"].validation_participant),
            "test_participant": str(result["fold"].test_participant),
            "split": split,
            "seed": int(seed),
            "candidate": method,
            "variant": variant,
            "modalities": "+".join(result["modalities"]),
            "common_valid_windows": int(source.common_valid_windows),
            "source_windows": int(source.source_windows),
            "quality": float(source.quality),
            "presence": bool(source.presence),
            "neural_training_performed": False,
            "head_device": "cpu",
            "embedding_cuda_used": True,
        }
        for modality in FULL_MODALITIES:
            record[f"{modality}_included"] = modality in result["modalities"]
            record[f"{modality}_components"] = int(result["modality_components"].get(modality, 0))
        for target in TARGETS:
            raw, bounded = result[correction_key][target]
            record[f"{target}_true"] = float(source[target])
            record[f"{target}_pred"] = float(result[prediction_key][target][local_index])
            record[f"condition_only_{target}"] = float(result[anchor_key][target][local_index])
            record[f"{target}_raw_correction"] = float(raw[local_index])
            record[f"{target}_bounded_correction"] = float(bounded[local_index])
            record[f"{target}_alpha"] = float(result["alpha"])
            record[f"{target}_gamma"] = float(result["gamma"])
            if method == PRIMARY_METHOD:
                for modality_index, modality in enumerate(result["modalities"]):
                    expert_raw, expert_bounded = result[expert_key][target][modality]
                    record[f"{target}_{modality}_raw_correction"] = float(expert_raw[local_index])
                    record[f"{target}_{modality}_bounded_correction"] = float(expert_bounded[local_index])
                    record[f"{target}_{modality}_weight"] = float(result["weights"][target][modality_index])
        rows.append(record)
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    preregistration = json.loads(args.preregistration.read_text(encoding="utf-8"))
    method_spec = _validate_preregistration(args, preregistration)
    (
        dataset,
        frame,
        participants,
        folds,
        blocks,
        masks,
        availability,
        presence,
        quality,
        contract,
    ) = _load_inputs(args)
    modalities = variant_modalities(args.variant)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []
    for fold in folds:
        result = (
            _fit_expert_fold(blocks, masks, availability, presence, frame, fold, modalities, preregistration["common_model_rules"], args.compression_cache_dir)
            if args.method == PRIMARY_METHOD
            else _fit_standard_fold(args.method, blocks, masks, availability, presence, quality, frame, fold, preregistration["common_model_rules"], args.compression_cache_dir)
        )
        evidence = _eligibility(args.method, args.variant, result, presence)
        if not evidence["eligible"]:
            raise ValueError(f"Fold {fold.fold_index} violates feature-contribution eligibility: {evidence}")
        artifact = {
            "schema_version": "relax_compression_fusion_fold_v2",
            "candidate": args.method,
            "variant": args.variant,
            "seed": args.seed,
            "fold": asdict(fold),
            "modalities": list(modalities),
            "alpha": result["alpha"],
            "gamma": result["gamma"],
            "compression": result["fitted"],
            "compression_provenance": result["compression_provenance"],
            "models": result["models"],
            "expert_weights": result.get("weights"),
            "eligibility": evidence,
        }
        artifact_path = model_dir / f"fold_{fold.fold_index:02d}.joblib"
        joblib.dump(artifact, artifact_path)
        prediction_rows.extend(_prediction_rows(args.method, args.variant, args.seed, frame, result, result["test_indexes"], "test"))
        validation_rows.extend(_prediction_rows(args.method, args.variant, args.seed, frame, result, result["validation_indexes"], "validation"))
        target_records = {}
        for target in TARGETS:
            target_records[target] = {
                "learned": True,
                "fallback": False,
                "alpha": float(result["alpha"]),
                "gamma": float(result["gamma"]),
                "validation_mae": float(result["validation_target_mae"][target]),
                "expert_weights": (
                    {modality: float(result["weights"][target][index]) for index, modality in enumerate(modalities)}
                    if args.method == PRIMARY_METHOD
                    else None
                ),
                "ridge_coefficient_sha256": (
                    {modality: _hash_array(result["models"][target][modality].coef_) for modality in modalities}
                    if args.method == PRIMARY_METHOD
                    else _hash_array(result["models"][target].coef_)
                ),
            }
        fold_records.append(
            {
                **asdict(fold),
                "n_train_observations": 63,
                "n_validation_observations": 9,
                "n_test_observations": 9,
                "modalities": list(modalities),
                "compression": {
                    "modality_components": result["modality_components"],
                    "total_modality_components": int(sum(result["modality_components"].values())),
                    "provenance": result["compression_provenance"],
                },
                "targets": target_records,
                "validation_macro_mae": float(result["validation_macro_mae"]),
                "eligibility": evidence,
                "inner_provenance": result.get("inner_provenance"),
                "inner_oof_sha256": result.get("inner_oof_sha256"),
                "train_anchor_sha256": result["train_anchor_sha256"],
                "test_anchor_sha256": {target: _hash_array(result["test_anchor"][target]) for target in TARGETS},
                "model_artifact": str(artifact_path),
                "model_artifact_sha256": file_sha256(artifact_path),
            }
        )
    predictions = pd.DataFrame(prediction_rows).sort_values(["participant_id", "presentation_position"]).reset_index(drop=True)
    validation_predictions = pd.DataFrame(validation_rows).sort_values(["fold_index", "presentation_position"]).reset_index(drop=True)
    if len(predictions) != EXPECTED_OBSERVATIONS or predictions.duplicated(["participant_id", "condition"]).any():
        raise ValueError("Expected 81 unique outer-test predictions")
    for target in TARGETS:
        if not predictions[f"{target}_pred"].between(0.0, 1.0).all():
            raise ValueError(f"{target} prediction is outside [0,1]")
        if not (predictions[f"{target}_gamma"] > 0.0).all():
            raise ValueError(f"{target} includes a nonpositive gamma")
    run_name = f"{args.method}_{args.variant}_s{args.seed}"
    prediction_path = args.output_dir / f"{run_name}_predictions.csv"
    validation_path = args.output_dir / f"{run_name}_validation_predictions.csv"
    fold_path = args.output_dir / f"{run_name}_folds.json"
    result_path = args.output_dir / f"{run_name}_results.json"
    predictions.to_csv(prediction_path, index=False)
    validation_predictions.to_csv(validation_path, index=False)
    _write_json(fold_path, fold_records)
    metrics = _metrics(predictions)
    baseline = predictions.copy()
    for target in TARGETS:
        baseline[f"{target}_pred"] = baseline[f"condition_only_{target}"]
    baseline_metrics = _metrics(baseline)
    delta = {
        target: metrics["targets"][target]["participant_macro_mae"] - baseline_metrics["targets"][target]["participant_macro_mae"]
        for target in TARGETS
    }
    delta["macro"] = metrics["macro_mae"] - baseline_metrics["macro_mae"]
    run_eligibility = {
        "all_folds_eligible": all(record["eligibility"]["eligible"] for record in fold_records),
        "both_targets_learned_in_every_fold": all(all(record["targets"][target]["learned"] for target in TARGETS) for record in fold_records),
        "gamma_strictly_positive_in_every_fold": all(record["eligibility"]["gamma_strictly_positive"] for record in fold_records),
        "condition_fallback_used": False,
        "nonzero_correction_both_targets_in_every_fold": all(record["eligibility"]["nonzero_test_correction_for_both_targets"] for record in fold_records),
        "primary_claim_eligible": args.method == PRIMARY_METHOD and args.variant == "full",
        "ablation_only": args.variant != "full",
    }
    payload = {
        "schema_version": "relax_foundation_compression_fusion_run_v2",
        "run_name": run_name,
        "candidate": args.method,
        "variant": args.variant,
        "family": method_spec.get("family"),
        "modalities": list(modalities),
        "method_spec": method_spec,
        "seed": args.seed,
        "protocol": "fixed_9p_7_train_1_validation_1_test_81_labels_545_common_windows",
        "participant_count": len(participants),
        "observation_count": len(predictions),
        "fold_count": len(folds),
        "metrics": metrics,
        "condition_only_metrics": baseline_metrics,
        "delta_vs_condition": delta,
        "eligibility": run_eligibility,
        "feature_extraction": contract["features"],
        "embedding_provenance": dataset.metadata,
        "head_runtime": {
            "device": "cpu",
            "neural_training_performed": False,
            "embedding_extraction_cuda_used": True,
            "embedding_extraction_device": dataset.metadata.get("cuda_device_name"),
        },
        "contract": contract,
        "preregistration": {
            "path": str(args.preregistration.resolve()),
            "sha256": file_sha256(args.preregistration),
            "schema_version": preregistration["schema_version"],
        },
        "folds": fold_records,
        "outputs": {"predictions": str(prediction_path), "validation_predictions": str(validation_path), "folds": str(fold_path)},
        "prediction_sha256": file_sha256(prediction_path),
        "validation_prediction_sha256": file_sha256(validation_path),
        "source_sha256": {
            "runner": file_sha256(__file__),
            "compression": file_sha256(ROOT / "src/fusion/frozen_compression_v2.py"),
            "dataset": file_sha256(ROOT / "src/data/relax_dataset.py"),
            "anchor_runner": file_sha256(ROOT / "scripts/run_relax_condition_anchor_probe.py"),
        },
        "git": _git_state(),
        "packages": _package_versions(),
        "command": [sys.executable, *sys.argv],
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json(result_path, payload)
    return {
        "result_path": str(result_path),
        "prediction_path": str(prediction_path),
        "metrics": metrics,
        "delta_vs_condition": delta,
        "eligibility": run_eligibility,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--variant", choices=VARIANTS, default="full")
    parser.add_argument("--seed", choices=ALLOWED_SEEDS, type=int, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--mask-manifest", type=Path, required=True)
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compression-cache-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            args.output_dir / "hard_failure.json",
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "candidate": args.method,
                "variant": args.variant,
                "seed": args.seed,
            },
        )
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

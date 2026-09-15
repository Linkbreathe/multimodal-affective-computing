"""Run preregistered frozen-feature compression/fusion probes on Relax.

The foundation encoders are never retrained.  Their frozen cache must prove
CUDA extraction.  The deliberately small compression and Ridge heads run on
CPU.  Every learned transform, residual head, expert weight, and validation
choice excludes the outer test participant.
"""

from __future__ import annotations

# ruff: noqa: E402

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any
import warnings

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge

from scripts.run_relax_condition_anchor_probe import (
    ALLOWED_SEEDS,
    EXPECTED_CACHE_SHA256,
    EXPECTED_OBSERVATIONS,
    EXPECTED_PARTICIPANTS,
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
from mac.fusion import frozen_compression as compression


FORMAL_MODALITIES = ("ecg", "eye", "head", "video")
FOUNDATION_MODALITIES = ("ecg", "eye", "video")
EXPERT_METHODS = ("modality_expert_simplex", "modality_expert_simplex_foundation_only")
METHODS = (
    "joint_block_balanced_pca8",
    "modality_pca2_additive",
    "supervised_pls2",
    "linear_shared_private",
    *EXPERT_METHODS,
)


def _preregistered_method(payload: dict[str, Any], name: str) -> dict[str, Any]:
    declared = {str(record["name"]): deepcopy(record) for record in payload["methods"]}
    sensitivity = payload["foundation_only_sensitivity"]
    if name == str(sensitivity["name"]):
        record = deepcopy(declared[str(sensitivity["method_family"])])
        record.update(
            {
                "name": str(sensitivity["name"]),
                "family": str(record["family"]),
                "modalities": [str(value) for value in sensitivity["modalities"]],
                "purpose": str(sensitivity["purpose"]),
                "multiplicity_status": str(sensitivity["multiplicity_status"]),
            }
        )
        return record
    if name not in declared:
        raise ValueError(f"Method {name!r} is not preregistered")
    return declared[name]


def _validate_preregistration(
    args: argparse.Namespace,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if payload.get("schema_version") != "relax_foundation_compression_preregistration_v1":
        raise ValueError("Unexpected preregistration schema")
    if not payload.get("frozen_before_formal_candidate_outcomes"):
        raise ValueError("Preregistration is not marked frozen before formal outcomes")
    protocol = payload["protocol"]
    if tuple(protocol["participants"]) != EXPECTED_PARTICIPANTS:
        raise ValueError("Preregistration participant cohort changed")
    if tuple(int(value) for value in protocol["seeds"]) != ALLOWED_SEEDS:
        raise ValueError("Preregistration seed set changed")
    if args.seed not in protocol["seeds"]:
        raise ValueError(f"Seed {args.seed} is outside the preregistered set")
    rules = payload["common_model_rules"]
    if 0.0 in rules["positive_gammas"] or rules.get("gamma_zero_allowed", True):
        raise ValueError("Formal compression/fusion probes forbid gamma=0")
    if tuple(payload["input_contract"]["formal_modalities"]) != FORMAL_MODALITIES:
        raise ValueError("Formal modality order changed")
    expected_hashes = payload["input_contract"]
    observed = {
        "embedding_cache_sha256": file_sha256(args.embedding_cache),
        "labels_sha256": file_sha256(args.labels),
        "split_manifest_sha256": file_sha256(args.split_manifest),
        "common_mask_sha256": file_sha256(args.mask_manifest),
    }
    for key, value in observed.items():
        if value != expected_hashes[key]:
            raise ValueError(f"Preregistered input hash mismatch for {key}: {value}")
    method = _preregistered_method(payload, args.method)
    if args.method not in METHODS:
        raise ValueError(f"Unsupported formal method: {args.method}")
    return method


def _pool_blocks(
    dataset: RelaxConditionEmbeddingDataset,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    blocks: dict[str, np.ndarray] = {}
    availability: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {"modalities": {}}
    for modality in FORMAL_MODALITIES:
        values = dataset.embeddings[modality].numpy().astype(np.float64, copy=False)
        mask = dataset.masks[modality].numpy().astype(bool, copy=False)
        counts = mask.sum(axis=1).astype(np.int64)
        pooled = (values * mask[..., None]).sum(axis=1) / np.maximum(counts[:, None], 1)
        pooled[counts == 0] = 0.0
        if not np.isfinite(pooled).all():
            raise ValueError(f"Non-finite pooled features for {modality}")
        blocks[modality] = pooled
        availability[modality] = counts > 0
        metadata["modalities"][modality] = {
            "embedding_dimension": int(pooled.shape[1]),
            "effective_valid_windows": int(mask.sum()),
            "zero_window_observations": int((counts == 0).sum()),
            "feature_sha256": _hash_array(pooled),
            "availability_sha256": _hash_array(availability[modality]),
        }
    shared = availability[FORMAL_MODALITIES[0]].copy()
    for modality in FORMAL_MODALITIES[1:]:
        if not np.array_equal(shared, availability[modality]):
            raise ValueError("The formal common mask must yield identical modality availability")
    if int(shared.sum()) != EXPECTED_OBSERVATIONS - 1:
        raise ValueError("Expected exactly one all-missing participant-condition observation")
    metadata.update(
        {
            "shared_presence_policy": "one_common_presence_flag",
            "shared_presence_sha256": _hash_array(shared),
            "all_missing_observations": int((~shared).sum()),
        }
    )
    return blocks, availability, shared, metadata


def _fit_compressor(
    method: str,
    blocks: dict[str, np.ndarray],
    availability: dict[str, np.ndarray],
    train_indexes: np.ndarray,
    residual_targets: np.ndarray,
    seed: int,
) -> Any:
    """Narrow adapter to the shared frozen-compression implementation."""
    del seed  # All preregistered reducers use deterministic full SVD.
    fitted = compression.make_compressor(method, FORMAL_MODALITIES)
    full_targets = np.zeros((len(next(iter(blocks.values()))), len(TARGETS)), dtype=np.float64)
    full_targets[train_indexes] = residual_targets
    fitted.fit(
        blocks,
        availability,
        train_indexes,
        targets=full_targets if method == "supervised_pls2" else None,
    )
    return fitted


def _transform(
    compressor: Any,
    blocks: dict[str, np.ndarray],
    availability: dict[str, np.ndarray],
    indexes: np.ndarray,
) -> np.ndarray:
    values = np.asarray(compressor.transform(blocks, availability, indexes), dtype=np.float64)
    if values.ndim != 2 or len(values) != len(indexes) or not np.isfinite(values).all():
        raise ValueError("Compressor returned invalid latent features")
    return values


def _compressor_provenance(compressor: Any) -> dict[str, Any]:
    value = compressor.provenance() if callable(getattr(compressor, "provenance", None)) else compressor.provenance
    if not isinstance(value, dict):
        raise TypeError("Compressor provenance must be a dictionary")
    return deepcopy(value)


def _fit_expert_bank(
    blocks: dict[str, np.ndarray],
    availability: dict[str, np.ndarray],
    modalities: tuple[str, ...],
    train_indexes: np.ndarray,
    seed: int,
) -> Any:
    del seed  # Expert PCA reducers also use deterministic full SVD.
    return compression.fit_modality_reducers(
        blocks,
        availability,
        train_indexes,
        modalities,
        n_components=2,
    )


def _expert_latents(
    bank: dict[str, Any],
    blocks: dict[str, np.ndarray],
    availability: dict[str, np.ndarray],
    indexes: np.ndarray,
) -> dict[str, np.ndarray]:
    result = {
        modality: np.asarray(reducer.transform(blocks, availability, indexes), dtype=np.float64)
        for modality, reducer in bank.items()
    }
    for modality, value in result.items():
        if value.ndim != 2 or len(value) != len(indexes) or not np.isfinite(value).all():
            raise ValueError(f"Expert bank returned invalid {modality} latent values")
    return result


def _solve_expert_weights(
    predictions: np.ndarray,
    truths: np.ndarray,
    modalities: tuple[str, ...],
    minimum_foundation_weight: float,
) -> np.ndarray:
    lower_bounds = np.asarray(
        [minimum_foundation_weight if modality in FOUNDATION_MODALITIES else 0.0 for modality in modalities],
        dtype=np.float64,
    )
    if lower_bounds.sum() >= 1.0:
        raise ValueError("Expert lower bounds leave no feasible simplex")
    initial = lower_bounds + (1.0 - lower_bounds.sum()) / len(modalities)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Values in x were outside bounds during a minimize step, clipping to bounds",
            category=RuntimeWarning,
            module=r"scipy\.optimize",
        )
        solution = minimize(
            lambda value: float(np.mean(np.square(truths - predictions @ value))),
            initial,
            method="SLSQP",
            bounds=[(float(bound), 1.0) for bound in lower_bounds],
            constraints={"type": "eq", "fun": lambda value: float(value.sum() - 1.0)},
            options={"ftol": 1e-12, "maxiter": 2000},
        )
    if not solution.success:
        raise RuntimeError(f"Expert simplex optimization failed: {solution.message}")
    # Project numerical optimizer tolerance back onto the exact constrained
    # simplex.  This is a sub-nanoscopic correction, but makes the formal
    # >=0.05 audit deterministic instead of platform-tolerance dependent.
    weights = np.maximum(np.asarray(solution.x, dtype=np.float64), lower_bounds)
    excess = weights - lower_bounds
    free_mass = 1.0 - float(lower_bounds.sum())
    if float(excess.sum()) > 0.0:
        weights = lower_bounds + excess * (free_mass / float(excess.sum()))
    else:
        weights = lower_bounds + free_mass / len(weights)
    if weights.shape != (len(modalities),) or not np.isfinite(weights).all():
        raise ValueError("Invalid expert simplex weights")
    if not np.isclose(weights.sum(), 1.0, atol=1e-8):
        raise ValueError("Expert weights do not sum to one")
    if np.any(weights < lower_bounds - 1e-8):
        raise ValueError("A foundation expert violates its preregistered weight floor")
    return weights


def _weighted_experts(
    corrections: np.ndarray,
    available: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    weighted = available.astype(np.float64) * weights[None, :]
    denominator = weighted.sum(axis=1)
    result = np.zeros(len(corrections), dtype=np.float64)
    valid = denominator > 0.0
    result[valid] = (corrections[valid] * weighted[valid]).sum(axis=1) / denominator[valid]
    return result


def _residual_targets(
    frame: pd.DataFrame,
    train_indexes: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    anchors: dict[str, np.ndarray] = {}
    residuals: list[np.ndarray] = []
    for target in TARGETS:
        anchor = cross_fitted_condition_anchors(frame, train_indexes, target)
        anchors[target] = anchor
        residuals.append(frame.iloc[train_indexes][target].to_numpy(dtype=np.float64) - anchor)
    return np.column_stack(residuals), anchors


def _heldout_anchors(
    frame: pd.DataFrame,
    train_indexes: np.ndarray,
    indexes: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        target: heldout_condition_anchors(frame, train_indexes, indexes, target) for target in TARGETS
    }


def _selection_key(macro_mae: float, alpha: float, gamma: float) -> tuple[float, float, float]:
    # Exact preregistered tie policy: stronger alpha, then smaller positive gamma.
    return float(macro_mae), -float(alpha), float(gamma)


def _model_corrections(
    model: Ridge,
    latent: np.ndarray,
    shared_presence: np.ndarray,
    cap: float,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(model.predict(latent), dtype=np.float64)
    raw[~shared_presence] = 0.0
    return raw, np.clip(raw, -cap, cap)


def _fit_standard_fold(
    method: str,
    blocks: dict[str, np.ndarray],
    availability: dict[str, np.ndarray],
    shared_presence: np.ndarray,
    frame: pd.DataFrame,
    fold: Fold,
    seed: int,
    rules: dict[str, Any],
) -> dict[str, Any]:
    participants = frame["participant_id"].astype(str).to_numpy()
    train, validation, test = _fold_indexes(participants, fold)
    residual, train_anchor = _residual_targets(frame, train)
    validation_anchor = _heldout_anchors(frame, train, validation)
    test_anchor = _heldout_anchors(frame, train, test)
    fitted = _fit_compressor(
        method,
        blocks,
        availability,
        train,
        residual,
        seed + fold.fold_index,
    )
    # Shared compressors already apply their train-fitted latent scaler.
    train_latent = _transform(fitted, blocks, availability, train)
    validation_latent = _transform(fitted, blocks, availability, validation)
    test_latent = _transform(fitted, blocks, availability, test)
    cap = float(rules["residual_cap"])
    candidates: list[dict[str, Any]] = []
    for alpha in (float(value) for value in rules["ridge_alphas"]):
        models: dict[str, Ridge] = {}
        validation_corrections: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        test_corrections: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for target_index, target in enumerate(TARGETS):
            model = Ridge(alpha=alpha).fit(train_latent, residual[:, target_index])
            models[target] = model
            validation_corrections[target] = _model_corrections(
                model, validation_latent, shared_presence[validation], cap
            )
            test_corrections[target] = _model_corrections(
                model, test_latent, shared_presence[test], cap
            )
        for gamma in (float(value) for value in rules["positive_gammas"]):
            if gamma <= 0.0:
                raise ValueError("Gamma must be strictly positive")
            validation_predictions = {
                target: np.clip(
                    validation_anchor[target] + gamma * validation_corrections[target][1], 0.0, 1.0
                )
                for target in TARGETS
            }
            target_mae = {
                target: float(
                    np.mean(
                        np.abs(
                            frame.iloc[validation][target].to_numpy(dtype=np.float64)
                            - validation_predictions[target]
                        )
                    )
                )
                for target in TARGETS
            }
            macro = float(np.mean(list(target_mae.values())))
            candidates.append(
                {
                    "key": _selection_key(macro, alpha, gamma),
                    "alpha": alpha,
                    "gamma": gamma,
                    "models": models,
                    "validation_corrections": validation_corrections,
                    "test_corrections": test_corrections,
                    "validation_predictions": validation_predictions,
                    "validation_target_mae": target_mae,
                    "validation_macro_mae": macro,
                }
            )
    selected = min(candidates, key=lambda value: value["key"])
    test_predictions = {
        target: np.clip(
            test_anchor[target] + selected["gamma"] * selected["test_corrections"][target][1],
            0.0,
            1.0,
        )
        for target in TARGETS
    }
    return {
        "train_indexes": train,
        "validation_indexes": validation,
        "test_indexes": test,
        "fold_index": fold.fold_index,
        "validation_participant": fold.validation_participant,
        "test_participant": fold.test_participant,
        "compressor": fitted,
        "compressor_provenance": _compressor_provenance(fitted),
        "latent_scaler": None,
        "latent_dimension": int(train_latent.shape[1]),
        "models": selected["models"],
        "alpha": float(selected["alpha"]),
        "gamma": float(selected["gamma"]),
        "validation_anchor": validation_anchor,
        "test_anchor": test_anchor,
        "validation_corrections": selected["validation_corrections"],
        "test_corrections": selected["test_corrections"],
        "validation_predictions": selected["validation_predictions"],
        "test_predictions": test_predictions,
        "validation_target_mae": selected["validation_target_mae"],
        "validation_macro_mae": float(selected["validation_macro_mae"]),
        "train_anchor_sha256": {target: _hash_array(train_anchor[target]) for target in TARGETS},
    }


def _inner_expert_data(
    blocks: dict[str, np.ndarray],
    availability: dict[str, np.ndarray],
    modalities: tuple[str, ...],
    frame: pd.DataFrame,
    outer_train: np.ndarray,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    participants = frame["participant_id"].astype(str).to_numpy()
    inner_data: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    outer_participants = sorted(set(participants[outer_train]))
    for inner_index, heldout_participant in enumerate(outer_participants, start=1):
        heldout = outer_train[participants[outer_train] == heldout_participant]
        inner_train = outer_train[participants[outer_train] != heldout_participant]
        if len(inner_train) != 54 or len(heldout) != 9:
            raise ValueError("Inner participant LOPO must map to 54 train / 9 heldout rows")
        bank = _fit_expert_bank(
            blocks,
            availability,
            modalities,
            inner_train,
            seed + 100 * inner_index,
        )
        train_latents = _expert_latents(bank, blocks, availability, inner_train)
        heldout_latents = _expert_latents(bank, blocks, availability, heldout)
        residual, _train_anchor = _residual_targets(frame, inner_train)
        heldout_anchor = _heldout_anchors(frame, inner_train, heldout)
        heldout_truth = {
            target: frame.iloc[heldout][target].to_numpy(dtype=np.float64) - heldout_anchor[target]
            for target in TARGETS
        }
        inner_data.append(
            {
                "inner_train": inner_train,
                "heldout": heldout,
                "bank": bank,
                "train_latents": train_latents,
                "heldout_latents": heldout_latents,
                "residual": residual,
                "heldout_truth": heldout_truth,
            }
        )
        bank_provenance = {
            modality: reducer.provenance() for modality, reducer in bank.items()
        }
        provenance.append(
            {
                "heldout_participant": heldout_participant,
                "fit_participants": sorted(set(participants[inner_train])),
                "fit_observations": int(len(inner_train)),
                "heldout_observations": int(len(heldout)),
                "compressor": deepcopy(bank_provenance),
            }
        )
    coverage = np.concatenate([record["heldout"] for record in inner_data])
    if set(coverage) != set(outer_train) or len(coverage) != len(set(coverage)):
        raise ValueError("Inner LOPO expert predictions do not exactly cover outer training rows")
    return inner_data, provenance


def _expert_oof_for_alpha(
    inner_data: list[dict[str, Any]],
    availability: dict[str, np.ndarray],
    modalities: tuple[str, ...],
    frame: pd.DataFrame,
    outer_train: np.ndarray,
    alpha: float,
    cap: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    local_lookup = {int(index): position for position, index in enumerate(outer_train)}
    predictions = {
        target: np.zeros((len(outer_train), len(modalities)), dtype=np.float64) for target in TARGETS
    }
    truths = {target: np.zeros(len(outer_train), dtype=np.float64) for target in TARGETS}
    for record in inner_data:
        heldout = record["heldout"]
        positions = np.asarray([local_lookup[int(index)] for index in heldout], dtype=np.int64)
        for target_index, target in enumerate(TARGETS):
            truths[target][positions] = record["heldout_truth"][target]
            for modality_index, modality in enumerate(modalities):
                model = Ridge(alpha=alpha).fit(
                    record["train_latents"][modality], record["residual"][:, target_index]
                )
                raw = np.asarray(model.predict(record["heldout_latents"][modality]), dtype=np.float64)
                available = np.asarray(availability[modality][heldout], dtype=bool)
                raw[~available] = 0.0
                predictions[target][positions, modality_index] = np.clip(raw, -cap, cap)
    return predictions, truths


def _fit_expert_fold(
    method: str,
    blocks: dict[str, np.ndarray],
    availability: dict[str, np.ndarray],
    shared_presence: np.ndarray,
    frame: pd.DataFrame,
    fold: Fold,
    seed: int,
    rules: dict[str, Any],
    minimum_foundation_weight: float,
) -> dict[str, Any]:
    participants = frame["participant_id"].astype(str).to_numpy()
    train, validation, test = _fold_indexes(participants, fold)
    modalities = FOUNDATION_MODALITIES if method.endswith("foundation_only") else FORMAL_MODALITIES
    residual, train_anchor = _residual_targets(frame, train)
    validation_anchor = _heldout_anchors(frame, train, validation)
    test_anchor = _heldout_anchors(frame, train, test)
    inner_data, inner_provenance = _inner_expert_data(
        blocks, availability, modalities, frame, train, seed + 1000 * fold.fold_index
    )
    outer_bank = _fit_expert_bank(
        blocks, availability, modalities, train, seed + fold.fold_index
    )
    train_latents = _expert_latents(outer_bank, blocks, availability, train)
    validation_latents = _expert_latents(outer_bank, blocks, availability, validation)
    test_latents = _expert_latents(outer_bank, blocks, availability, test)
    cap = float(rules["residual_cap"])
    candidates: list[dict[str, Any]] = []
    for alpha in (float(value) for value in rules["ridge_alphas"]):
        inner_predictions, inner_truths = _expert_oof_for_alpha(
            inner_data, availability, modalities, frame, train, alpha, cap
        )
        models: dict[str, dict[str, Ridge]] = {target: {} for target in TARGETS}
        weights: dict[str, np.ndarray] = {}
        validation_experts: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {
            target: {} for target in TARGETS
        }
        test_experts: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {
            target: {} for target in TARGETS
        }
        validation_combined: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        test_combined: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for target_index, target in enumerate(TARGETS):
            weights[target] = _solve_expert_weights(
                inner_predictions[target],
                inner_truths[target],
                modalities,
                minimum_foundation_weight,
            )
            validation_raw_matrix = np.zeros((len(validation), len(modalities)), dtype=np.float64)
            validation_bounded_matrix = np.zeros_like(validation_raw_matrix)
            test_raw_matrix = np.zeros((len(test), len(modalities)), dtype=np.float64)
            test_bounded_matrix = np.zeros_like(test_raw_matrix)
            for modality_index, modality in enumerate(modalities):
                model = Ridge(alpha=alpha).fit(train_latents[modality], residual[:, target_index])
                models[target][modality] = model
                validation_raw = np.asarray(
                    model.predict(validation_latents[modality]), dtype=np.float64
                )
                test_raw = np.asarray(model.predict(test_latents[modality]), dtype=np.float64)
                validation_available = availability[modality][validation]
                test_available = availability[modality][test]
                validation_raw[~validation_available] = 0.0
                test_raw[~test_available] = 0.0
                validation_bounded = np.clip(validation_raw, -cap, cap)
                test_bounded = np.clip(test_raw, -cap, cap)
                validation_experts[target][modality] = (validation_raw, validation_bounded)
                test_experts[target][modality] = (test_raw, test_bounded)
                validation_raw_matrix[:, modality_index] = validation_raw
                validation_bounded_matrix[:, modality_index] = validation_bounded
                test_raw_matrix[:, modality_index] = test_raw
                test_bounded_matrix[:, modality_index] = test_bounded
            validation_available = np.column_stack(
                [availability[modality][validation] for modality in modalities]
            )
            test_available = np.column_stack([availability[modality][test] for modality in modalities])
            validation_combined[target] = (
                _weighted_experts(validation_raw_matrix, validation_available, weights[target]),
                _weighted_experts(validation_bounded_matrix, validation_available, weights[target]),
            )
            test_combined[target] = (
                _weighted_experts(test_raw_matrix, test_available, weights[target]),
                _weighted_experts(test_bounded_matrix, test_available, weights[target]),
            )
        for gamma in (float(value) for value in rules["positive_gammas"]):
            if gamma <= 0.0:
                raise ValueError("Gamma must be strictly positive")
            validation_predictions = {
                target: np.clip(
                    validation_anchor[target] + gamma * validation_combined[target][1], 0.0, 1.0
                )
                for target in TARGETS
            }
            target_mae = {
                target: float(
                    np.mean(
                        np.abs(
                            frame.iloc[validation][target].to_numpy(dtype=np.float64)
                            - validation_predictions[target]
                        )
                    )
                )
                for target in TARGETS
            }
            macro = float(np.mean(list(target_mae.values())))
            candidates.append(
                {
                    "key": _selection_key(macro, alpha, gamma),
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
                    "validation_predictions": validation_predictions,
                    "validation_target_mae": target_mae,
                    "validation_macro_mae": macro,
                }
            )
    selected = min(candidates, key=lambda value: value["key"])
    test_predictions = {
        target: np.clip(
            test_anchor[target] + selected["gamma"] * selected["test_combined"][target][1],
            0.0,
            1.0,
        )
        for target in TARGETS
    }
    modality_ablations: dict[str, dict[str, Any]] = {}
    for target in TARGETS:
        full_mae = selected["validation_target_mae"][target]
        modality_ablations[target] = {"full_validation_mae": full_mae, "without": {}}
        bounded_matrix = np.column_stack(
            [selected["validation_experts"][target][modality][1] for modality in modalities]
        )
        base_available = np.column_stack(
            [availability[modality][validation] for modality in modalities]
        )
        truth = frame.iloc[validation][target].to_numpy(dtype=np.float64)
        for modality_index, modality in enumerate(modalities):
            ablated_available = base_available.copy()
            ablated_available[:, modality_index] = False
            correction = _weighted_experts(
                bounded_matrix, ablated_available, selected["weights"][target]
            )
            prediction = np.clip(
                validation_anchor[target] + selected["gamma"] * correction, 0.0, 1.0
            )
            mae = float(np.mean(np.abs(truth - prediction)))
            modality_ablations[target]["without"][modality] = {
                "validation_mae": mae,
                "delta_without_minus_full": mae - full_mae,
            }
    bank_provenance = {
        modality: reducer.provenance() for modality, reducer in outer_bank.items()
    }
    return {
        "train_indexes": train,
        "validation_indexes": validation,
        "test_indexes": test,
        "fold_index": fold.fold_index,
        "validation_participant": fold.validation_participant,
        "test_participant": fold.test_participant,
        "modalities": modalities,
        "outer_bank": outer_bank,
        "compressor_provenance": deepcopy(bank_provenance),
        "inner_provenance": inner_provenance,
        "latent_dimension": int(2 * len(modalities)),
        "models": selected["models"],
        "weights": selected["weights"],
        "alpha": float(selected["alpha"]),
        "gamma": float(selected["gamma"]),
        "validation_anchor": validation_anchor,
        "test_anchor": test_anchor,
        "validation_corrections": selected["validation_combined"],
        "test_corrections": selected["test_combined"],
        "validation_experts": selected["validation_experts"],
        "test_experts": selected["test_experts"],
        "validation_predictions": selected["validation_predictions"],
        "test_predictions": test_predictions,
        "validation_target_mae": selected["validation_target_mae"],
        "validation_macro_mae": float(selected["validation_macro_mae"]),
        "modality_ablations": modality_ablations,
        "train_anchor_sha256": {target: _hash_array(train_anchor[target]) for target in TARGETS},
        "inner_oof_sha256": {
            target: {
                "predictions": _hash_array(selected["inner_predictions"][target]),
                "truths": _hash_array(selected["inner_truths"][target]),
            }
            for target in TARGETS
        },
    }


def _eligibility(
    method: str,
    result: dict[str, Any],
    shared_presence: np.ndarray,
) -> dict[str, Any]:
    test = result["test_indexes"]
    learned = {target: target in result["models"] for target in TARGETS}
    nonzero = {
        target: bool(
            np.any(
                np.abs(result["test_corrections"][target][1][shared_presence[test]]) > 1e-12
            )
        )
        for target in TARGETS
    }
    evidence: dict[str, Any] = {
        "minimum_modalities": len(result.get("modalities", FORMAL_MODALITIES)),
        "both_targets_learned": bool(all(learned.values())),
        "gamma_strictly_positive": bool(result["gamma"] > 0.0),
        "condition_fallback_used": False,
        "nonzero_test_correction_for_both_targets": bool(all(nonzero.values())),
        "all_missing_test_rows": int((~shared_presence[test]).sum()),
    }
    if method in EXPERT_METHODS:
        weights = result["weights"]
        evidence["foundation_modalities_with_positive_weight_per_target"] = {
            target: sum(
                float(weights[target][index]) > 0.0
                for index, modality in enumerate(result["modalities"])
                if modality in FOUNDATION_MODALITIES
            )
            for target in TARGETS
        }
        evidence["foundation_weight_floor_satisfied"] = bool(
            all(
                float(weights[target][index]) >= 0.05 - 1e-8
                for target in TARGETS
                for index, modality in enumerate(result["modalities"])
                if modality in FOUNDATION_MODALITIES
            )
        )
    else:
        evidence["foundation_modalities_in_compressor"] = list(FOUNDATION_MODALITIES)
    evidence["eligible"] = bool(
        evidence["minimum_modalities"] >= 2
        and evidence["both_targets_learned"]
        and evidence["gamma_strictly_positive"]
        and not evidence["condition_fallback_used"]
        and evidence["nonzero_test_correction_for_both_targets"]
        and evidence.get("foundation_weight_floor_satisfied", True)
    )
    return evidence


def _rows_for_fold(
    method: str,
    family: str,
    seed: int,
    frame: pd.DataFrame,
    result: dict[str, Any],
    availability: dict[str, np.ndarray],
    indexes: np.ndarray,
    kind: str,
) -> list[dict[str, Any]]:
    anchor_key = f"{kind}_anchor"
    correction_key = f"{kind}_corrections"
    prediction_key = f"{kind}_predictions"
    expert_key = f"{kind}_experts"
    records: list[dict[str, Any]] = []
    for local_index, row_index in enumerate(indexes):
        row = frame.iloc[row_index]
        record: dict[str, Any] = {
            "candidate": method,
            "family": family,
            "seed": seed,
            "participant_id": str(row["participant_id"]),
            "condition": str(row["condition"]),
            "presentation_position": float(row["presentation_position"]),
            "fold_index": int(result["fold_index"]),
            "test_participant": str(result["test_participant"]),
            "validation_participant": str(result["validation_participant"]),
            "modalities": "+".join(result.get("modalities", FORMAL_MODALITIES)),
            "relaxation_true": float(row["relaxation"]),
            "discomfort_true": float(row["discomfort"]),
            "alpha": float(result["alpha"]),
            "gamma": float(result["gamma"]),
            "compression_dimension": int(result["latent_dimension"]),
            "head_device": "cpu",
            "neural_training_performed": False,
            "embedding_cuda_used": True,
        }
        for modality in FORMAL_MODALITIES:
            record[f"{modality}_available"] = bool(availability[modality][row_index])
        for target in TARGETS:
            raw, bounded = result[correction_key][target]
            record[f"{target}_pred"] = float(result[prediction_key][target][local_index])
            record[f"condition_only_{target}"] = float(result[anchor_key][target][local_index])
            record[f"{target}_raw_correction"] = float(raw[local_index])
            record[f"{target}_bounded_correction"] = float(bounded[local_index])
            record[f"{target}_alpha"] = float(result["alpha"])
            record[f"{target}_gamma"] = float(result["gamma"])
            if method in EXPERT_METHODS:
                for modality in result["modalities"]:
                    expert_raw, expert_bounded = result[expert_key][target][modality]
                    record[f"{target}_{modality}_raw_correction"] = float(
                        expert_raw[local_index]
                    )
                    record[f"{target}_{modality}_bounded_correction"] = float(
                        expert_bounded[local_index]
                    )
                    modality_index = result["modalities"].index(modality)
                    record[f"{target}_{modality}_weight"] = float(
                        result["weights"][target][modality_index]
                    )
        records.append(record)
    return records


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    preregistration = json.loads(args.preregistration.read_text(encoding="utf-8"))
    method_spec = _validate_preregistration(args, preregistration)
    labels, participants, folds, contract = validate_contract(
        SimpleNamespace(
            labels=args.labels,
            windows=args.windows,
            split_manifest=args.split_manifest,
            mask_manifest=args.mask_manifest,
            cohorts=args.cohorts,
            embedding_cache=args.embedding_cache,
            expected_cache_sha256=EXPECTED_CACHE_SHA256,
            cohort=args.cohort,
        )
    )
    dataset = RelaxConditionEmbeddingDataset(
        args.embedding_cache,
        modalities=FORMAL_MODALITIES,
        participants=participants,
        mask_manifest=args.mask_manifest,
        strict=True,
    )
    if not dataset.metadata.get("cuda_used") or not str(dataset.metadata.get("device", "")).startswith(
        "cuda"
    ):
        raise ValueError("Frozen embedding cache does not prove CUDA extraction")
    blocks, availability, shared_presence, feature_metadata = _pool_blocks(dataset)
    frame = pd.DataFrame(
        {
            "participant_id": dataset.participant_ids,
            "condition": dataset.conditions,
            "presentation_position": dataset.presentation_positions.numpy(),
            "relaxation": dataset.targets[:, 0].numpy(),
            "discomfort": dataset.targets[:, 1].numpy(),
        }
    )
    label_lookup = labels.set_index(["participant_id", "condition"])
    if set(dataset.condition_keys()) != set(label_lookup.index):
        raise ValueError("Cache and formal label keys differ")
    for row in frame.itertuples(index=False):
        expected = label_lookup.loc[(row.participant_id, row.condition), list(TARGETS)].to_numpy(
            dtype=np.float64
        )
        if not np.allclose(expected, [row.relaxation, row.discomfort], atol=1e-6, rtol=0.0):
            raise ValueError(f"Cache target mismatch for {(row.participant_id, row.condition)}")
    rules = preregistration["common_model_rules"]
    prediction_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []
    model_dir = args.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        if args.method in EXPERT_METHODS:
            result = _fit_expert_fold(
                args.method,
                blocks,
                availability,
                shared_presence,
                frame,
                fold,
                args.seed,
                rules,
                minimum_foundation_weight=0.05,
            )
        else:
            result = _fit_standard_fold(
                args.method,
                blocks,
                availability,
                shared_presence,
                frame,
                fold,
                args.seed,
                rules,
            )
        if args.method not in EXPERT_METHODS and int(result["latent_dimension"]) != int(
            method_spec["compression_dimension"]
        ):
            raise ValueError(
                f"{args.method} produced {result['latent_dimension']} latent dimensions; "
                f"preregistered {method_spec['compression_dimension']}"
            )
        expected_expert_dimension = 2 * len(result.get("modalities", ()))
        if args.method in EXPERT_METHODS and int(result["latent_dimension"]) != expected_expert_dimension:
            raise ValueError("Expert latent dimension does not equal two per modality")
        evidence = _eligibility(args.method, result, shared_presence)
        if not evidence["eligible"]:
            raise ValueError(f"Fold {fold.fold_index} violates primary eligibility: {evidence}")
        artifact = {
            "schema_version": "relax_compression_fusion_fold_v1",
            "candidate": args.method,
            "method_spec": method_spec,
            "seed": args.seed,
            "fold": asdict(fold),
            "alpha": result["alpha"],
            "gamma": result["gamma"],
            "compressor": result.get("compressor", result.get("outer_bank")),
            "compressor_provenance": result["compressor_provenance"],
            "latent_scaler": result.get("latent_scaler"),
            "models": result["models"],
            "expert_weights": result.get("weights"),
            "inner_provenance": result.get("inner_provenance"),
            "eligibility": evidence,
        }
        artifact_path = model_dir / f"fold_{fold.fold_index:02d}.joblib"
        joblib.dump(artifact, artifact_path)
        prediction_rows.extend(
            _rows_for_fold(
                args.method,
                str(method_spec["family"]),
                args.seed,
                frame,
                result,
                availability,
                result["test_indexes"],
                "test",
            )
        )
        validation_rows.extend(
            _rows_for_fold(
                args.method,
                str(method_spec["family"]),
                args.seed,
                frame,
                result,
                availability,
                result["validation_indexes"],
                "validation",
            )
        )
        targets: dict[str, Any] = {}
        for target in TARGETS:
            targets[target] = {
                "learned": True,
                "fallback": False,
                "alpha": float(result["alpha"]),
                "gamma": float(result["gamma"]),
                "validation_mae": float(result["validation_target_mae"][target]),
                "expert_weights": (
                    {
                        modality: float(result["weights"][target][index])
                        for index, modality in enumerate(result["modalities"])
                    }
                    if args.method in EXPERT_METHODS
                    else None
                ),
                "expert_weight_fit": (
                    {
                        "source": "participant_wise_inner_lopo_predictions",
                        "loss": "mean_squared_residual_error",
                        "solver": "SLSQP_nonnegative_simplex",
                        "foundation_weight_lower_bound": 0.05,
                        "inner_prediction_sha256": result["inner_oof_sha256"][target][
                            "predictions"
                        ],
                        "inner_truth_sha256": result["inner_oof_sha256"][target]["truths"],
                    }
                    if args.method in EXPERT_METHODS
                    else None
                ),
                "ridge_coefficient_sha256": (
                    {
                        modality: _hash_array(result["models"][target][modality].coef_)
                        for modality in result["modalities"]
                    }
                    if args.method in EXPERT_METHODS
                    else _hash_array(result["models"][target].coef_)
                ),
            }
        fold_records.append(
            {
                **asdict(fold),
                "n_train_participants": 7,
                "n_validation_participants": 1,
                "n_test_participants": 1,
                "n_train_observations": 63,
                "n_validation_observations": 9,
                "n_test_observations": 9,
                "compression_dimension": int(result["latent_dimension"]),
                "compression": result["compressor_provenance"],
                "inner_provenance": result.get("inner_provenance"),
                "inner_oof_sha256": result.get("inner_oof_sha256"),
                "targets": targets,
                "validation_macro_mae": float(result["validation_macro_mae"]),
                "modality_ablations": result.get("modality_ablations"),
                "eligibility": evidence,
                "train_anchor_sha256": result["train_anchor_sha256"],
                "test_anchor_sha256": {
                    target: _hash_array(result["test_anchor"][target]) for target in TARGETS
                },
                "model_artifact": str(artifact_path),
                "model_artifact_sha256": file_sha256(artifact_path),
            }
        )
    predictions = pd.DataFrame(prediction_rows).sort_values(
        ["participant_id", "presentation_position"]
    ).reset_index(drop=True)
    validation_predictions = pd.DataFrame(validation_rows).sort_values(
        ["fold_index", "presentation_position"]
    ).reset_index(drop=True)
    if len(predictions) != EXPECTED_OBSERVATIONS or predictions.duplicated(
        ["participant_id", "condition"]
    ).any():
        raise ValueError("Expected 81 unique OOF predictions")
    for target in TARGETS:
        if not predictions[f"{target}_pred"].between(0.0, 1.0).all():
            raise ValueError(f"{target} prediction is outside [0,1]")
        if not (predictions[f"{target}_gamma"] > 0.0).all():
            raise ValueError(f"{target} contains a nonpositive gamma")
    run_name = f"{args.method}_s{args.seed}"
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
        target: metrics["targets"][target]["participant_macro_mae"]
        - baseline_metrics["targets"][target]["participant_macro_mae"]
        for target in TARGETS
    }
    delta["macro"] = metrics["macro_mae"] - baseline_metrics["macro_mae"]
    eligibility = {
        "all_folds_eligible": bool(all(record["eligibility"]["eligible"] for record in fold_records)),
        "both_targets_learned_in_every_fold": bool(
            all(
                all(record["targets"][target]["learned"] for target in TARGETS)
                for record in fold_records
            )
        ),
        "gamma_strictly_positive_in_every_fold": bool(
            all(record["eligibility"]["gamma_strictly_positive"] for record in fold_records)
        ),
        "condition_fallback_used": False,
        "nonzero_correction_both_targets_in_every_fold": bool(
            all(
                record["eligibility"]["nonzero_test_correction_for_both_targets"]
                for record in fold_records
            )
        ),
    }
    payload = {
        "schema_version": "relax_foundation_compression_fusion_run_v1",
        "run_name": run_name,
        "candidate": args.method,
        "family": method_spec["family"],
        "modalities": list(
            FOUNDATION_MODALITIES if args.method.endswith("foundation_only") else FORMAL_MODALITIES
        ),
        "method_spec": method_spec,
        "seed": args.seed,
        "protocol": "fixed_9p_7_train_1_validation_1_test_545_common_windows",
        "participant_count": len(participants),
        "observation_count": len(predictions),
        "fold_count": len(folds),
        "metrics": metrics,
        "condition_only_metrics": baseline_metrics,
        "delta_vs_condition": delta,
        "compression_dimensions": [record["compression_dimension"] for record in fold_records],
        "eligibility": eligibility,
        "feature_extraction": feature_metadata,
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
        "outputs": {
            "predictions": str(prediction_path),
            "validation_predictions": str(validation_path),
            "folds": str(fold_path),
        },
        "prediction_sha256": file_sha256(prediction_path),
        "validation_prediction_sha256": file_sha256(validation_path),
        "source_sha256": {
            "runner": file_sha256(__file__),
            "compression": file_sha256(ROOT / "src/fusion/frozen_compression.py"),
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
        "eligibility": eligibility,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, choices=ALLOWED_SEEDS, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--mask-manifest", type=Path, required=True)
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument("--cohort", default="eeg_eligible")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
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
                "seed": args.seed,
            },
        )
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

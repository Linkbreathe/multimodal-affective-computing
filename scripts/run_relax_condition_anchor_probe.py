"""Run leakage-safe condition-anchored linear probes over frozen Relax embeddings.

The neural encoders are frozen.  Their cached embeddings were extracted on CUDA;
the deliberately small Ridge heads in this script run on CPU.  Every learned
transform, residual head, and calibration choice is fitted without the outer
test participant.

The condition anchor is a train-only estimate of the expected label for the
same visual condition. The learned model predicts a correction to that anchor,
which makes this probe comparable with the classical residual model.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from scripts.run_relax_foundation_probe import (
    Fold,
    TARGETS,
    _fold_indexes,
    _load_split_manifest,
    _metrics,
    file_sha256,
)
from mac.data.relax_dataset import RelaxConditionEmbeddingDataset


ALLOWED_SEEDS = (20260705, 20260706, 20260707)
FULL_MODALITIES = ("eeg", "ecg", "eye", "head", "video")
EXPECTED_CACHE_SHA256 = "827488833a32d98d67c8e7c3477b86c87b384a9746542a57ad0448acccc2c12e"
EXPECTED_PARTICIPANTS = ("P003", "P004", "P007", "P008", "P009", "P011", "P012", "P013", "P015")
EXPECTED_OBSERVATIONS = 81
EXPECTED_VALID_WINDOWS = 545


@dataclass(frozen=True)
class Candidate:
    name: str
    modalities: tuple[str, ...]
    learned_targets: tuple[str, ...]
    representation: str
    quality_features: str
    pca_fit_components: int
    pca_prefixes: tuple[int, ...]
    alphas: tuple[float, ...]
    gammas: tuple[float, ...]
    residual_cap: float | None
    selection_tie_policy: str
    family: str


def _screen_candidate(name: str, modalities: tuple[str, ...], representation: str) -> Candidate:
    return Candidate(
        name=name,
        modalities=modalities,
        learned_targets=TARGETS,
        representation=representation,
        quality_features="presence",
        pca_fit_components=8,
        pca_prefixes=(8,),
        alphas=(10.0, 100.0, 1000.0),
        gammas=(0.0, 0.25, 0.5, 0.75, 1.0),
        residual_cap=0.2,
        selection_tie_policy="validation_mae_then_alpha_then_gamma",
        family="adaptive_six_candidate_screen",
    )


CANDIDATES: dict[str, Candidate] = {
    candidate.name: candidate
    for candidate in (
        _screen_candidate("ecg_raw_dual", ("ecg",), "raw"),
        _screen_candidate("ecg_centered_dual", ("ecg",), "train_condition_centered"),
        _screen_candidate("ecg_eye_raw_dual", ("ecg", "eye"), "raw"),
        _screen_candidate("ecg_eye_centered_dual", ("ecg", "eye"), "train_condition_centered"),
        _screen_candidate("no_eeg_raw_dual", ("ecg", "eye", "head", "video"), "raw"),
        _screen_candidate(
            "no_eeg_centered_dual", ("ecg", "eye", "head", "video"), "train_condition_centered"
        ),
        Candidate(
            name="eye_discomfort_compact",
            modalities=("eye",),
            learned_targets=("discomfort",),
            representation="raw",
            quality_features="count_and_missing",
            pca_fit_components=32,
            pca_prefixes=(4, 16, 32),
            alphas=(0.1, 10.0, 100.0),
            gammas=(1.0,),
            residual_cap=None,
            selection_tie_policy="validation_mae_then_components_then_stronger_alpha",
            family="target_specific_eye_followup",
        ),
        Candidate(
            name="eye_discomfort_wide",
            modalities=("eye",),
            learned_targets=("discomfort",),
            representation="raw",
            quality_features="count_and_missing",
            pca_fit_components=32,
            pca_prefixes=(2, 4, 8, 16, 32),
            alphas=(0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0),
            gammas=(0.0, 0.05, 0.1, 0.25, 0.5, 1.0),
            residual_cap=None,
            selection_tie_policy="validation_mae_then_gamma_components_stronger_alpha",
            family="target_specific_eye_followup",
        ),
    )
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _hash_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _git_state() -> dict[str, Any]:
    def run(*command: str) -> str:
        return subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "head": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "branch", "--show-current"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def _as_bool_series(series: pd.Series) -> pd.Series:
    lowered = series.astype(str).str.strip().str.lower()
    allowed = {"true": True, "1": True, "1.0": True, "false": False, "0": False, "0.0": False}
    unknown = sorted(set(lowered) - set(allowed))
    if unknown:
        raise ValueError(f"Unrecognized boolean mask values: {unknown}")
    return lowered.map(allowed).astype(bool)


def validate_contract(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], list[Fold], dict[str, Any]]:
    """Validate the immutable formal cohort, hashes, masks, labels, and folds."""
    paths = {
        "labels": args.labels,
        "windows": args.windows,
        "split_manifest": args.split_manifest,
        "mask_manifest": args.mask_manifest,
        "cohorts": args.cohorts,
        "embedding_cache": args.embedding_cache,
    }
    for name, path in paths.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    cache_sha = file_sha256(args.embedding_cache)
    if cache_sha != args.expected_cache_sha256:
        raise ValueError(
            f"Frozen cache SHA-256 mismatch: expected {args.expected_cache_sha256}, observed {cache_sha}"
        )

    cohorts = json.loads(Path(args.cohorts).read_text(encoding="utf-8"))
    participants = [str(value) for value in cohorts.get(args.cohort, [])]
    if tuple(participants) != EXPECTED_PARTICIPANTS:
        raise ValueError(f"Formal cohort must be exactly {EXPECTED_PARTICIPANTS}; got {tuple(participants)}")

    labels = pd.read_csv(args.labels)
    labels = labels.loc[labels["participant_id"].astype(str).isin(participants)].copy()
    labels = labels.sort_values(["participant_id", "presentation_position"]).reset_index(drop=True)
    if len(labels) != EXPECTED_OBSERVATIONS or labels.duplicated(["participant_id", "condition"]).any():
        raise ValueError("Formal labels must contain 81 unique participant-condition observations")
    if set(labels.groupby("participant_id").size()) != {9}:
        raise ValueError("Every formal participant must have nine labels")

    windows = pd.read_csv(args.windows)
    masks = pd.read_csv(args.mask_manifest)
    keys = ["participant_id", "condition", "condition_window_index"]
    windows = windows.loc[windows["participant_id"].astype(str).isin(participants)].copy()
    masks = masks.loc[masks["participant_id"].astype(str).isin(participants)].copy()
    if windows.duplicated(keys).any() or masks.duplicated(keys).any():
        raise ValueError("Window or mask contract contains duplicate keys")
    if set(map(tuple, windows[keys].to_numpy())) != set(map(tuple, masks[keys].to_numpy())):
        raise ValueError("Window and mask keys do not match")
    validity = np.ones(len(masks), dtype=bool)
    for modality in FULL_MODALITIES:
        column = f"{modality}_valid"
        if column not in masks:
            raise ValueError(f"Mask contract lacks {column}")
        validity &= _as_bool_series(masks[column]).to_numpy()
    if int(validity.sum()) != EXPECTED_VALID_WINDOWS:
        raise ValueError(f"Expected 545 common-valid windows, observed {int(validity.sum())}")
    zero = (
        masks.assign(_valid=validity)
        .groupby(["participant_id", "condition"], sort=True)["_valid"]
        .sum()
    )
    zero_keys = [tuple(map(str, key)) for key in zero.index[zero == 0]]
    if zero_keys != [("P004", "C6")]:
        raise ValueError(f"Unexpected zero-window observations: {zero_keys}")

    folds = _load_split_manifest(args.split_manifest, participants, strict=True)
    if len(folds) != 9 or any(len(fold.train_participants) != 7 for fold in folds):
        raise ValueError("Formal split must contain nine 7/1/1 folds")
    hashes = {name: {"path": str(Path(path).resolve()), "sha256": file_sha256(path)} for name, path in paths.items()}
    return labels, participants, folds, {
        "inputs": hashes,
        "common_valid_window_count": int(validity.sum()),
        "zero_common_window_observations": zero_keys,
    }


def pooled_features(
    dataset: RelaxConditionEmbeddingDataset,
    quality_features: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Mask-aware arithmetic means with explicit quality/missingness features."""
    blocks: list[np.ndarray] = []
    modality_records: dict[str, Any] = {}
    sequence_length = next(iter(dataset.embeddings.values())).shape[1]
    for modality in dataset.modalities:
        # Pool only valid aligned windows. A zero-window observation is retained
        # as an explicit zero vector plus quality metadata, not silently dropped.
        values = dataset.embeddings[modality].numpy().astype(np.float64, copy=False)
        mask = dataset.masks[modality].numpy().astype(bool, copy=False)
        count = mask.sum(axis=1).astype(np.float64)
        pooled = (values * mask[..., None]).sum(axis=1) / np.maximum(count[:, None], 1.0)
        pooled[count == 0] = 0.0
        blocks.append(pooled)
        if quality_features == "presence":
            blocks.append((count > 0).astype(np.float64)[:, None])
        elif quality_features == "count_and_missing":
            blocks.append((count / float(sequence_length))[:, None])
            blocks.append((count == 0).astype(np.float64)[:, None])
        else:
            raise ValueError(f"Unsupported quality feature policy: {quality_features}")
        modality_records[modality] = {
            "embedding_dimension": int(values.shape[-1]),
            "effective_valid_windows": int(mask.sum()),
            "zero_window_observations": int((count == 0).sum()),
            "minimum_windows": int(count.min()),
            "maximum_windows": int(count.max()),
        }
    features = np.concatenate(blocks, axis=1)
    if not np.isfinite(features).all():
        raise ValueError("Pooled features contain non-finite values")
    return features, {
        "feature_dimension": int(features.shape[1]),
        "quality_features": quality_features,
        "modalities": modality_records,
        "feature_sha256": _hash_array(features),
    }


def condition_feature_means(
    features: np.ndarray,
    conditions: np.ndarray,
    train_indexes: np.ndarray,
) -> dict[str, np.ndarray]:
    """Fit condition centroids using only the outer-training observations."""
    means: dict[str, np.ndarray] = {}
    for condition in sorted(set(conditions[train_indexes])):
        indexes = train_indexes[conditions[train_indexes] == condition]
        if len(indexes) == 0:
            raise ValueError(f"No training feature rows for {condition}")
        means[str(condition)] = features[indexes].mean(axis=0)
    return means


def apply_condition_centering(
    features: np.ndarray,
    conditions: np.ndarray,
    means: dict[str, np.ndarray],
) -> np.ndarray:
    centered = np.empty_like(features)
    for index, condition in enumerate(conditions):
        if str(condition) not in means:
            raise ValueError(f"Condition {condition} has no train-fitted feature centroid")
        centered[index] = features[index] - means[str(condition)]
    return centered


def cross_fitted_condition_anchors(
    frame: pd.DataFrame,
    train_indexes: np.ndarray,
    target: str,
) -> np.ndarray:
    """Anchor each training row using same-condition labels from other participants."""
    # Leaving the current participant out is essential: otherwise the residual
    # target would contain information from the very label being predicted.
    anchors = np.empty(len(train_indexes), dtype=np.float64)
    train = frame.iloc[train_indexes]
    for local_index, (_, row) in enumerate(train.iterrows()):
        others = train.loc[
            (train["condition"].astype(str) == str(row["condition"]))
            & (train["participant_id"].astype(str) != str(row["participant_id"])),
            target,
        ]
        if len(others) != len(set(train["participant_id"].astype(str))) - 1:
            raise ValueError("Cross-fitted train anchor does not contain exactly the other six participants")
        anchors[local_index] = float(others.mean())
    return anchors


def heldout_condition_anchors(
    frame: pd.DataFrame,
    train_indexes: np.ndarray,
    heldout_indexes: np.ndarray,
    target: str,
) -> np.ndarray:
    """Apply train-participant condition means to validation or test rows."""
    train = frame.iloc[train_indexes]
    means = train.groupby(train["condition"].astype(str))[target].mean()
    result = frame.iloc[heldout_indexes]["condition"].astype(str).map(means)
    if result.isna().any():
        raise ValueError("A held-out condition has no outer-training anchor")
    return result.to_numpy(dtype=np.float64)


def _candidate_key(
    candidate: Candidate,
    validation_mae: float,
    gamma: float,
    components: int,
    alpha: float,
) -> tuple[float, ...]:
    if candidate.selection_tie_policy == "validation_mae_then_alpha_then_gamma":
        return validation_mae, alpha, gamma
    if candidate.selection_tie_policy == "validation_mae_then_components_then_stronger_alpha":
        return validation_mae, components, -alpha
    if candidate.selection_tie_policy == "validation_mae_then_gamma_components_stronger_alpha":
        return validation_mae, gamma, components, -alpha
    raise ValueError(f"Unknown tie policy: {candidate.selection_tie_policy}")


def _bounded(values: np.ndarray, cap: float | None) -> np.ndarray:
    return np.clip(values, -cap, cap) if cap is not None else values


def fit_fold(
    candidate: Candidate,
    features: np.ndarray,
    frame: pd.DataFrame,
    fold: Fold,
    seed: int,
) -> dict[str, Any]:
    """Fit one probe fold: anchor first, learn residual correction second."""
    participants = frame["participant_id"].astype(str).to_numpy()
    conditions = frame["condition"].astype(str).to_numpy()
    train, validation, test = _fold_indexes(participants, fold)
    if len(train) != 63 or len(validation) != 9 or len(test) != 9:
        raise ValueError(f"Fold {fold.fold_index} does not map to 63/9/9 rows")

    fitted_condition_means: dict[str, np.ndarray] | None = None
    model_features = features
    if candidate.representation == "train_condition_centered":
        fitted_condition_means = condition_feature_means(features, conditions, train)
        model_features = apply_condition_centering(features, conditions, fitted_condition_means)
    elif candidate.representation != "raw":
        raise ValueError(f"Unknown representation: {candidate.representation}")

    scaler = StandardScaler().fit(model_features[train])
    scaled_train = scaler.transform(model_features[train])
    scaled_validation = scaler.transform(model_features[validation])
    scaled_test = scaler.transform(model_features[test])
    pca_components = min(candidate.pca_fit_components, len(train) - 1, scaled_train.shape[1])
    if pca_components < max(candidate.pca_prefixes):
        raise ValueError("Not enough train rows/features for the requested PCA prefixes")
    pca = PCA(
        n_components=pca_components,
        svd_solver="randomized",
        random_state=seed + fold.fold_index,
    ).fit(scaled_train)
    train_pca = pca.transform(scaled_train)
    validation_pca = pca.transform(scaled_validation)
    test_pca = pca.transform(scaled_test)

    target_outputs: dict[str, Any] = {}
    selected_models: dict[str, Ridge | None] = {}
    for target in TARGETS:
        target_index = TARGETS.index(target)
        train_anchor = cross_fitted_condition_anchors(frame, train, target)
        validation_anchor = heldout_condition_anchors(frame, train, validation, target)
        test_anchor = heldout_condition_anchors(frame, train, test, target)
        train_truth = frame.iloc[train][target].to_numpy(dtype=np.float64)
        validation_truth = frame.iloc[validation][target].to_numpy(dtype=np.float64)
        residual = train_truth - train_anchor
        baseline_validation_mae = float(np.mean(np.abs(validation_truth - validation_anchor)))

        if target not in candidate.learned_targets:
            selected_models[target] = None
            target_outputs[target] = {
                "prediction": test_anchor.copy(),
                "test_anchor": test_anchor,
                "raw_test_correction": np.zeros_like(test_anchor),
                "bounded_test_correction": np.zeros_like(test_anchor),
                "validation_prediction": validation_anchor.copy(),
                "validation_anchor": validation_anchor,
                "raw_validation_correction": np.zeros_like(validation_anchor),
                "selection": {
                    "target": target,
                    "learned": False,
                    "fallback": True,
                    "fallback_reason": "target_specific_condition_only",
                    "gamma": 0.0,
                    "alpha": None,
                    "pca_components": None,
                    "validation_baseline_mae": baseline_validation_mae,
                    "validation_candidate_mae": baseline_validation_mae,
                },
            }
            continue

        candidates: list[dict[str, Any]] = []
        for components in candidate.pca_prefixes:
            for alpha in candidate.alphas:
                model = Ridge(alpha=alpha)
                model.fit(train_pca[:, :components], residual)
                raw_validation = model.predict(validation_pca[:, :components])
                bounded_validation = _bounded(raw_validation, candidate.residual_cap)
                for gamma in candidate.gammas:
                    validation_prediction = np.clip(validation_anchor + gamma * bounded_validation, 0.0, 1.0)
                    validation_mae = float(np.mean(np.abs(validation_truth - validation_prediction)))
                    candidates.append(
                        {
                            "key": _candidate_key(candidate, validation_mae, gamma, components, alpha),
                            "validation_mae": validation_mae,
                            "gamma": gamma,
                            "components": components,
                            "alpha": alpha,
                            "model": model,
                            "raw_validation": raw_validation,
                            "bounded_validation": bounded_validation,
                            "validation_prediction": validation_prediction,
                        }
                    )
        selected = min(candidates, key=lambda item: item["key"])
        model = selected["model"]
        raw_test = model.predict(test_pca[:, : selected["components"]])
        bounded_test = _bounded(raw_test, candidate.residual_cap)
        if selected["gamma"] == 0.0:
            prediction = test_anchor.copy()
        else:
            prediction = np.clip(test_anchor + selected["gamma"] * bounded_test, 0.0, 1.0)
        selected_models[target] = model
        target_outputs[target] = {
            "prediction": prediction,
            "test_anchor": test_anchor,
            "raw_test_correction": raw_test,
            "bounded_test_correction": bounded_test,
            "validation_prediction": selected["validation_prediction"],
            "validation_anchor": validation_anchor,
            "raw_validation_correction": selected["raw_validation"],
            "selection": {
                "target": target,
                "learned": True,
                "fallback": bool(selected["gamma"] == 0.0),
                "fallback_reason": "validation_selected_gamma_zero" if selected["gamma"] == 0.0 else None,
                "gamma": float(selected["gamma"]),
                "alpha": float(selected["alpha"]),
                "pca_components": int(selected["components"]),
                "validation_baseline_mae": baseline_validation_mae,
                "validation_candidate_mae": float(selected["validation_mae"]),
                "validation_improvement": baseline_validation_mae - float(selected["validation_mae"]),
                "ridge_coefficient_sha256": _hash_array(model.coef_),
                "ridge_intercept": float(model.intercept_),
            },
        }

    return {
        "fold": fold,
        "train_indexes": train,
        "validation_indexes": validation,
        "test_indexes": test,
        "scaler": scaler,
        "pca": pca,
        "condition_feature_means": fitted_condition_means,
        "models": selected_models,
        "targets": target_outputs,
        "transform_provenance": {
            "fit_participants": sorted(set(participants[train])),
            "fit_observation_count": int(len(train)),
            "scaler_mean_sha256": _hash_array(scaler.mean_),
            "scaler_scale_sha256": _hash_array(scaler.scale_),
            "pca_components_sha256": _hash_array(pca.components_),
            "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "pca_solver": "randomized",
            "pca_random_state": seed + fold.fold_index,
        },
    }


def _package_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "cuda_runtime": str(torch.version.cuda),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.seed not in ALLOWED_SEEDS:
        raise ValueError(f"Formal seed must be one of {ALLOWED_SEEDS}")
    candidate = CANDIDATES[args.candidate]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels, participants, folds, contract = validate_contract(args)
    dataset = RelaxConditionEmbeddingDataset(
        args.embedding_cache,
        modalities=candidate.modalities,
        participants=participants,
        mask_manifest=args.mask_manifest,
        strict=True,
    )
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
        raise ValueError("Embedding cache and formal labels have different observation keys")
    for row in frame.itertuples(index=False):
        expected = label_lookup.loc[(row.participant_id, row.condition), list(TARGETS)].to_numpy(dtype=float)
        if not np.allclose(expected, [row.relaxation, row.discomfort], atol=1e-6, rtol=0.0):
            raise ValueError(f"Cache target mismatch for {(row.participant_id, row.condition)}")
    for modality in candidate.modalities:
        if int(dataset.masks[modality].sum()) != EXPECTED_VALID_WINDOWS:
            raise ValueError(f"{modality} effective mask is not the frozen 545-window mask")
    features, feature_metadata = pooled_features(dataset, candidate.quality_features)

    started = time.perf_counter()
    prediction_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []
    model_dir = args.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        result = fit_fold(candidate, features, frame, fold, args.seed)
        test = result["test_indexes"]
        validation = result["validation_indexes"]
        artifact_path = model_dir / f"fold_{fold.fold_index:02d}.joblib"
        joblib.dump(
            {
                "candidate": asdict(candidate),
                "fold": asdict(fold),
                "seed": args.seed,
                "scaler": result["scaler"],
                "pca": result["pca"],
                "condition_feature_means": result["condition_feature_means"],
                "models": result["models"],
                "selections": {target: result["targets"][target]["selection"] for target in TARGETS},
            },
            artifact_path,
        )
        for local_index, row_index in enumerate(test):
            row = frame.iloc[row_index]
            record = {
                "participant_id": str(row["participant_id"]),
                "condition": str(row["condition"]),
                "presentation_position": float(row["presentation_position"]),
                "relaxation_true": float(row["relaxation"]),
                "discomfort_true": float(row["discomfort"]),
                "fold_index": fold.fold_index,
                "test_participant": fold.test_participant,
                "validation_participant": fold.validation_participant,
                "seed": args.seed,
                "candidate": candidate.name,
                "modalities": "+".join(candidate.modalities),
                "representation": candidate.representation,
                "head_device": "cpu",
                "neural_training_performed": False,
                "embedding_cuda_used": bool(dataset.metadata.get("cuda_used")),
            }
            for target in TARGETS:
                output = result["targets"][target]
                selection = output["selection"]
                record[f"{target}_pred"] = float(output["prediction"][local_index])
                record[f"condition_only_{target}"] = float(output["test_anchor"][local_index])
                record[f"{target}_raw_correction"] = float(output["raw_test_correction"][local_index])
                record[f"{target}_bounded_correction"] = float(output["bounded_test_correction"][local_index])
                record[f"{target}_gamma"] = float(selection["gamma"])
                record[f"{target}_alpha"] = selection["alpha"]
                record[f"{target}_pca_components"] = selection["pca_components"]
                record[f"{target}_fallback"] = bool(selection["fallback"])
            prediction_rows.append(record)
        for local_index, row_index in enumerate(validation):
            row = frame.iloc[row_index]
            record = {
                "participant_id": str(row["participant_id"]),
                "condition": str(row["condition"]),
                "presentation_position": float(row["presentation_position"]),
                "relaxation_true": float(row["relaxation"]),
                "discomfort_true": float(row["discomfort"]),
                "fold_index": fold.fold_index,
                "test_participant": fold.test_participant,
                "validation_participant": fold.validation_participant,
                "seed": args.seed,
                "candidate": candidate.name,
            }
            for target in TARGETS:
                output = result["targets"][target]
                record[f"{target}_pred"] = float(output["validation_prediction"][local_index])
                record[f"condition_only_{target}"] = float(output["validation_anchor"][local_index])
                record[f"{target}_raw_correction"] = float(output["raw_validation_correction"][local_index])
            validation_rows.append(record)
        fold_records.append(
            {
                **asdict(fold),
                "n_train_participants": 7,
                "n_validation_participants": 1,
                "n_test_participants": 1,
                "n_train_observations": 63,
                "n_validation_observations": 9,
                "n_test_observations": 9,
                "model_artifact": str(artifact_path),
                "model_artifact_sha256": file_sha256(artifact_path),
                "transform": result["transform_provenance"],
                "targets": {target: result["targets"][target]["selection"] for target in TARGETS},
                "test_anchor_sha256": {
                    target: _hash_array(result["targets"][target]["test_anchor"]) for target in TARGETS
                },
            }
        )

    predictions = pd.DataFrame(prediction_rows).sort_values(
        ["participant_id", "presentation_position"]
    ).reset_index(drop=True)
    validation_predictions = pd.DataFrame(validation_rows).sort_values(
        ["fold_index", "presentation_position"]
    ).reset_index(drop=True)
    if len(predictions) != EXPECTED_OBSERVATIONS or predictions.duplicated(["participant_id", "condition"]).any():
        raise ValueError("Run did not produce exactly 81 unique OOF predictions")
    if set(map(tuple, predictions[["participant_id", "condition"]].to_numpy())) != set(label_lookup.index):
        raise ValueError("OOF prediction keys differ from the formal labels")
    for target in TARGETS:
        if not predictions[f"{target}_pred"].between(0.0, 1.0).all():
            raise ValueError(f"{target} predictions leave [0,1]")
    if not dataset.metadata.get("cuda_used") or not str(dataset.metadata.get("device", "")).startswith("cuda"):
        raise ValueError("Frozen neural embedding cache does not prove CUDA extraction")

    run_name = f"{candidate.name}_s{args.seed}"
    prediction_path = args.output_dir / f"{run_name}_predictions.csv"
    validation_path = args.output_dir / f"{run_name}_validation_predictions.csv"
    fold_path = args.output_dir / f"{run_name}_folds.json"
    result_path = args.output_dir / f"{run_name}_results.json"
    predictions.to_csv(prediction_path, index=False)
    validation_predictions.to_csv(validation_path, index=False)
    _write_json(fold_path, fold_records)

    metrics = _metrics(predictions)
    baseline_frame = predictions.copy()
    for target in TARGETS:
        baseline_frame[f"{target}_pred"] = baseline_frame[f"condition_only_{target}"]
    baseline_metrics = _metrics(baseline_frame)
    payload = {
        "schema_version": "relax_condition_anchor_probe_v1",
        "run_name": run_name,
        "candidate": asdict(candidate),
        "seed": args.seed,
        "protocol": "shared_7_train_1_validation_1_test",
        "participant_count": len(participants),
        "observation_count": len(predictions),
        "fold_count": len(folds),
        "metrics": metrics,
        "condition_only_metrics": baseline_metrics,
        "delta_vs_condition": {
            "relaxation": metrics["targets"]["relaxation"]["participant_macro_mae"]
            - baseline_metrics["targets"]["relaxation"]["participant_macro_mae"],
            "discomfort": metrics["targets"]["discomfort"]["participant_macro_mae"]
            - baseline_metrics["targets"]["discomfort"]["participant_macro_mae"],
            "macro": metrics["macro_mae"] - baseline_metrics["macro_mae"],
        },
        "feature_extraction": feature_metadata,
        "embedding_provenance": dataset.metadata,
        "head_runtime": {
            "device": "cpu",
            "neural_training_performed": False,
            "embedding_extraction_cuda_used": True,
            "embedding_extraction_device": dataset.metadata.get("cuda_device_name"),
        },
        "contract": contract,
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
            "dataset": file_sha256(ROOT / "src/data/relax_dataset.py"),
            "aligned_runner": file_sha256(ROOT / "scripts/run_relax_foundation_probe.py"),
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
        "delta_vs_condition": payload["delta_vs_condition"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=sorted(CANDIDATES), required=True)
    parser.add_argument("--seed", type=int, choices=ALLOWED_SEEDS, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--mask-manifest", type=Path, required=True)
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument("--cohort", default="eeg_eligible")
    parser.add_argument("--expected-cache-sha256", default=EXPECTED_CACHE_SHA256)
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
                "candidate": args.candidate,
                "seed": args.seed,
            },
        )
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

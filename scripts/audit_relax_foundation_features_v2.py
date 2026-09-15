"""July 18 audit of the fixed five-modality Relax feature contract.

The audit is deliberately outcome-independent except for fixed, fold-safe Ridge
diagnostics.  It does not fit candidate models.  Window covariance uses weights
that give every participant-condition with at least one valid window equal total
mass; supervised diagnostics operate only on the 81 labelled condition rows.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from scripts.run_relax_foundation_probe import _fold_indexes, _load_split_manifest, file_sha256
from scripts.run_relax_condition_anchor_probe import (
    EXPECTED_CACHE_SHA256,
    EXPECTED_OBSERVATIONS,
    EXPECTED_PARTICIPANTS,
    EXPECTED_VALID_WINDOWS,
    TARGETS,
    cross_fitted_condition_anchors,
    heldout_condition_anchors,
)
from mac.data.relax_dataset import RelaxConditionEmbeddingDataset


MODALITIES = ("eeg", "ecg", "eye", "head", "video")
EXPECTED_DIMS = {"eeg": 1024, "ecg": 1024, "eye": 128, "head": 18, "video": 768}
HISTORICAL_NO_EEG = ("ecg", "eye", "head", "video")
NOMINAL_WINDOWS = 567


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _array_sha(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _as_bool(series: pd.Series) -> np.ndarray:
    mapping = {"true": True, "1": True, "1.0": True, "false": False, "0": False, "0.0": False}
    lowered = series.astype(str).str.strip().str.lower()
    unknown = sorted(set(lowered) - set(mapping))
    if unknown:
        raise ValueError(f"Unrecognized booleans: {unknown}")
    return lowered.map(mapping).to_numpy(dtype=bool)


def _spectrum_from_eigen(eigenvalues: np.ndarray) -> dict[str, Any]:
    eigen = np.asarray(eigenvalues, dtype=np.float64)
    eigen = np.sort(eigen[np.isfinite(eigen) & (eigen > 0.0)])[::-1]
    if not len(eigen):
        return {
            "matrix_rank": 0,
            "participation_ratio": 0.0,
            "entropy_effective_rank": 0.0,
            "pc_count_80": 0,
            "pc_count_90": 0,
            "pc_count_95": 0,
            "top_2_variance": 0.0,
            "top_8_variance": 0.0,
            "eigenvalues": [],
            "explained_variance_ratio": [],
        }
    tolerance = max(float(eigen[0]), 1.0) * 1e-12
    eigen = eigen[eigen > tolerance]
    proportions = eigen / eigen.sum()
    cumulative = np.cumsum(proportions)

    def count(threshold: float) -> int:
        return int(np.searchsorted(cumulative, threshold) + 1)

    return {
        "matrix_rank": int(len(eigen)),
        "participation_ratio": float(eigen.sum() ** 2 / np.square(eigen).sum()),
        "entropy_effective_rank": float(np.exp(-(proportions * np.log(proportions)).sum())),
        "pc_count_80": count(0.80),
        "pc_count_90": count(0.90),
        "pc_count_95": count(0.95),
        "top_2_variance": float(cumulative[min(1, len(cumulative) - 1)]),
        "top_8_variance": float(cumulative[min(7, len(cumulative) - 1)]),
        "eigenvalues": eigen.tolist(),
        "explained_variance_ratio": proportions.tolist(),
    }


def _unweighted_spectrum(values: np.ndarray) -> dict[str, Any]:
    scaled = StandardScaler().fit_transform(np.asarray(values, dtype=np.float64))
    singular = np.linalg.svd(scaled - scaled.mean(axis=0, keepdims=True), compute_uv=False)
    return _spectrum_from_eigen(singular**2 / max(len(values) - 1, 1))


def _weighted_window_spectrum(
    values: np.ndarray,
    mask: np.ndarray,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Condition-balanced spectrum of valid windows.

    Each nonempty condition contributes total weight one regardless of whether
    it has one or seven common-valid windows.
    """
    counts = mask.sum(axis=1).astype(np.int64)
    row_index, window_index = np.nonzero(mask)
    flat = values[row_index, window_index].astype(np.float64, copy=False)
    weights = 1.0 / counts[row_index]
    weights /= weights.sum()
    mean = np.sum(flat * weights[:, None], axis=0)
    variance = np.sum(np.square(flat - mean) * weights[:, None], axis=0)
    scale = np.sqrt(np.maximum(variance, 0.0))
    safe_scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (flat - mean) / safe_scale
    weighted = standardized * np.sqrt(weights[:, None])
    singular = np.linalg.svd(weighted, compute_uv=False)
    spectrum = _spectrum_from_eigen(singular**2)
    scale_summary = {
        "valid_window_rows": int(len(flat)),
        "nonempty_condition_rows": int((counts > 0).sum()),
        "window_row_l2_median": float(np.median(np.linalg.norm(flat, axis=1))),
        "window_coordinate_sd_median": float(np.median(scale)),
        "window_coordinate_sd_mean": float(np.mean(scale)),
    }
    return spectrum, scale_summary


def _pooled(dataset: RelaxConditionEmbeddingDataset, modality: str) -> tuple[np.ndarray, np.ndarray]:
    values = dataset.embeddings[modality].numpy().astype(np.float64, copy=False)
    mask = dataset.masks[modality].numpy().astype(bool, copy=False)
    counts = mask.sum(axis=1).astype(np.int64)
    pooled = (values * mask[..., None]).sum(axis=1) / np.maximum(counts[:, None], 1)
    pooled[counts == 0] = 0.0
    if not np.isfinite(pooled).all():
        raise ValueError(f"Non-finite pooled values for {modality}")
    return pooled, counts


def _factor_r2(values: np.ndarray, factor: np.ndarray) -> float:
    scaled = StandardScaler().fit_transform(values)
    design = pd.get_dummies(pd.Series(factor, dtype=str), drop_first=True).to_numpy(dtype=np.float64)
    design = np.column_stack([np.ones(len(design)), design])
    fitted = design @ np.linalg.pinv(design) @ scaled
    centered = scaled - scaled.mean(axis=0)
    denominator = float(np.square(centered).sum())
    return float(np.square(fitted - fitted.mean(axis=0)).sum() / denominator) if denominator else 0.0


def _continuous_feature_r2(values: np.ndarray, target: np.ndarray) -> float:
    scaled = StandardScaler().fit_transform(values)
    centered_target = np.asarray(target, dtype=np.float64) - float(np.mean(target))
    design = np.column_stack([np.ones(len(target)), centered_target])
    fitted = design @ np.linalg.pinv(design) @ scaled
    centered = scaled - scaled.mean(axis=0)
    denominator = float(np.square(centered).sum())
    return float(np.square(fitted - fitted.mean(axis=0)).sum() / denominator) if denominator else 0.0


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _fixed_target_diagnostic(
    values: np.ndarray,
    counts: np.ndarray,
    frame: pd.DataFrame,
    folds: list[Any],
) -> list[dict[str, Any]]:
    """Fixed PCA8/Ridge100 residual diagnostic with zero all-missing correction."""
    participants = frame["participant_id"].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        truth_parts: list[np.ndarray] = []
        anchor_parts: list[np.ndarray] = []
        corrected_parts: list[np.ndarray] = []
        legacy_parts: list[np.ndarray] = []
        raw_parts: list[np.ndarray] = []
        residual_parts: list[np.ndarray] = []
        missing_parts: list[np.ndarray] = []
        keys: list[str] = []
        for fold in folds:
            train, _validation, test = _fold_indexes(participants, fold)
            scaler = StandardScaler().fit(values[train])
            train_scaled = scaler.transform(values[train])
            test_scaled = scaler.transform(values[test])
            components = min(8, len(train) - 1, train_scaled.shape[1])
            pca = PCA(n_components=components, svd_solver="full").fit(train_scaled)
            train_latent = pca.transform(train_scaled)
            test_latent = pca.transform(test_scaled)
            train_anchor = cross_fitted_condition_anchors(frame, train, target)
            test_anchor = heldout_condition_anchors(frame, train, test, target)
            train_truth = frame.iloc[train][target].to_numpy(dtype=np.float64)
            test_truth = frame.iloc[test][target].to_numpy(dtype=np.float64)
            model = Ridge(alpha=100.0).fit(train_latent, train_truth - train_anchor)
            raw = np.clip(model.predict(test_latent), -0.2, 0.2)
            missing = counts[test] == 0
            corrected = raw.copy()
            corrected[missing] = 0.0
            truth_parts.append(test_truth)
            anchor_parts.append(test_anchor)
            corrected_parts.append(np.clip(test_anchor + corrected, 0.0, 1.0))
            legacy_parts.append(np.clip(test_anchor + raw, 0.0, 1.0))
            raw_parts.append(corrected)
            residual_parts.append(test_truth - test_anchor)
            missing_parts.append(missing)
            keys.extend(
                f"{frame.iloc[index]['participant_id']}/{frame.iloc[index]['condition']}" for index in test
            )
        truth = np.concatenate(truth_parts)
        anchor = np.concatenate(anchor_parts)
        prediction = np.concatenate(corrected_parts)
        legacy_prediction = np.concatenate(legacy_parts)
        correction = np.concatenate(raw_parts)
        residual = np.concatenate(residual_parts)
        missing = np.concatenate(missing_parts)
        missing_keys = [key for key, flag in zip(keys, missing, strict=True) if flag]
        rows.append(
            {
                "target": target,
                "fixed_components": 8,
                "fixed_alpha": 100.0,
                "fixed_gamma": 1.0,
                "condition_anchor_mae": float(np.mean(np.abs(truth - anchor))),
                "corrected_oof_mae": float(np.mean(np.abs(truth - prediction))),
                "corrected_delta_vs_condition": float(
                    np.mean(np.abs(truth - prediction)) - np.mean(np.abs(truth - anchor))
                ),
                "legacy_unforced_oof_mae": float(np.mean(np.abs(truth - legacy_prediction))),
                "forced_zero_minus_legacy_mae": float(
                    np.mean(np.abs(truth - prediction)) - np.mean(np.abs(truth - legacy_prediction))
                ),
                "correction_vs_condition_residual_correlation": _safe_corr(correction, residual),
                "mean_absolute_feature_correction_nonmissing": float(np.mean(np.abs(correction[~missing]))),
                "all_missing_count": int(missing.sum()),
                "all_missing_keys": "|".join(missing_keys),
                "all_missing_corrected_max_abs_correction": float(
                    np.max(np.abs(correction[missing]), initial=0.0)
                ),
            }
        )
    return rows


def _joint_fold_audit(
    blocks: dict[str, np.ndarray],
    counts: dict[str, np.ndarray],
    participants: np.ndarray,
    folds: list[Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        train, _validation, _test = _fold_indexes(participants, fold)
        historical_parts: list[np.ndarray] = []
        historical_spans: dict[str, tuple[int, int]] = {}
        offset = 0
        for modality in HISTORICAL_NO_EEG:
            part = np.column_stack([blocks[modality], (counts[modality] > 0).astype(float)])
            historical_parts.append(part)
            historical_spans[modality] = (offset, offset + part.shape[1])
            offset += part.shape[1]
        historical = np.concatenate(historical_parts, axis=1)
        if historical.shape != (81, 1942):
            raise ValueError(f"Historical no-EEG input changed: {historical.shape}")
        historical_scaled = StandardScaler().fit_transform(historical[train])
        historical_pca = PCA(n_components=8, svd_solver="full").fit(historical_scaled)

        full_scaled_parts: list[np.ndarray] = []
        full_spans: dict[str, tuple[int, int]] = {}
        offset = 0
        for modality in MODALITIES:
            scaled = StandardScaler().fit_transform(blocks[modality][train])
            full_scaled_parts.append(scaled)
            full_spans[modality] = (offset, offset + scaled.shape[1])
            offset += scaled.shape[1]
        full_scaled = np.concatenate(full_scaled_parts, axis=1)
        if full_scaled.shape[1] != sum(EXPECTED_DIMS.values()):
            raise ValueError("Full raw coordinate width changed")
        full_unbalanced = PCA(n_components=12, svd_solver="full").fit(full_scaled)
        balanced = np.concatenate(
            [part / np.sqrt(EXPECTED_DIMS[modality]) for modality, part in zip(MODALITIES, full_scaled_parts, strict=True)],
            axis=1,
        )
        full_balanced = PCA(n_components=12, svd_solver="full").fit(balanced)
        row: dict[str, Any] = {
            "fold_index": int(fold.fold_index),
            "test_participant": fold.test_participant,
            "validation_participant": fold.validation_participant,
            "historical_input_dimension": int(historical.shape[1]),
            "historical_pca8_explained_variance": float(historical_pca.explained_variance_ratio_.sum()),
            "full_raw_input_dimension": int(full_scaled.shape[1]),
            "full_unbalanced_pca12_explained_variance": float(full_unbalanced.explained_variance_ratio_.sum()),
            "full_block_balanced_pca12_explained_variance": float(full_balanced.explained_variance_ratio_.sum()),
        }
        for modality in HISTORICAL_NO_EEG:
            start, stop = historical_spans[modality]
            row[f"historical_{modality}_loading_mass_share"] = float(
                np.square(historical_pca.components_[:, start:stop]).sum() / historical_pca.n_components_
            )
        for modality in MODALITIES:
            start, stop = full_spans[modality]
            row[f"full_unbalanced_{modality}_loading_mass_share"] = float(
                np.square(full_unbalanced.components_[:, start:stop]).sum() / full_unbalanced.n_components_
            )
            row[f"full_balanced_{modality}_loading_mass_share"] = float(
                np.square(full_balanced.components_[:, start:stop]).sum() / full_balanced.n_components_
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _quality_frame(
    mask_manifest: pd.DataFrame,
    labels: pd.DataFrame,
    participants: tuple[str, ...],
) -> pd.DataFrame:
    masks = mask_manifest.loc[mask_manifest["participant_id"].astype(str).isin(participants)].copy()
    keys = ["participant_id", "condition"]
    rows: list[dict[str, Any]] = []
    for (participant, condition), group in masks.groupby(keys, sort=False):
        source_windows = len(group)
        common_count = int(_as_bool(group["common_valid"]).sum())
        row: dict[str, Any] = {
            "participant_id": str(participant),
            "condition": str(condition),
            "source_windows": int(source_windows),
            "common_valid_windows": common_count,
            "shared_quality": float(common_count / source_windows),
            "shared_presence": int(common_count > 0),
        }
        for modality in MODALITIES:
            row[f"source_{modality}_valid_windows"] = int(_as_bool(group[f"source_{modality}_valid"]).sum())
            row[f"formal_{modality}_valid_windows"] = int(_as_bool(group[f"{modality}_valid"]).sum())
        rows.append(row)
    quality = pd.DataFrame(rows)
    label_columns = ["participant_id", "condition", "presentation_position", *TARGETS]
    quality = quality.merge(labels[label_columns], on=keys, how="left", validate="one_to_one")
    if len(quality) != EXPECTED_OBSERVATIONS or quality[list(TARGETS)].isna().any().any():
        raise ValueError("Quality rows do not align one-to-one with formal labels")
    return quality.sort_values(["participant_id", "presentation_position"]).reset_index(drop=True)


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in (args.embedding_cache, args.labels, args.windows, args.mask_manifest, args.split_manifest, args.cohorts):
        if not path.is_file():
            raise FileNotFoundError(path)
    if file_sha256(args.embedding_cache) != EXPECTED_CACHE_SHA256:
        raise ValueError("Embedding cache hash does not match the frozen formal cache")
    cohorts = json.loads(args.cohorts.read_text(encoding="utf-8"))
    participants = tuple(str(value) for value in cohorts[args.cohort])
    if participants != EXPECTED_PARTICIPANTS:
        raise ValueError(f"Expected {EXPECTED_PARTICIPANTS}, got {participants}")
    folds = _load_split_manifest(args.split_manifest, list(participants), strict=True)
    if len(folds) != 9:
        raise ValueError("Expected exactly nine LOPO folds")

    labels = pd.read_csv(args.labels)
    labels = labels.loc[labels["participant_id"].astype(str).isin(participants)].copy()
    labels = labels.sort_values(["participant_id", "presentation_position"]).reset_index(drop=True)
    if len(labels) != EXPECTED_OBSERVATIONS or labels.duplicated(["participant_id", "condition"]).any():
        raise ValueError("Expected 81 unique participant-condition labels")

    masks = pd.read_csv(args.mask_manifest)
    formal_masks = masks.loc[masks["participant_id"].astype(str).isin(participants)].copy()
    if len(formal_masks) != NOMINAL_WINDOWS:
        raise ValueError(f"Expected {NOMINAL_WINDOWS} mask rows")
    common_manifest = _as_bool(formal_masks["common_valid"])
    if int(common_manifest.sum()) != EXPECTED_VALID_WINDOWS:
        raise ValueError("Common-valid mask count changed")
    for modality in MODALITIES:
        if not np.array_equal(_as_bool(formal_masks[f"{modality}_valid"]), common_manifest):
            raise ValueError(f"Formal {modality} mask is not the shared common mask")

    internal = RelaxConditionEmbeddingDataset(
        args.embedding_cache, modalities=MODALITIES, participants=participants, strict=True
    )
    common = RelaxConditionEmbeddingDataset(
        args.embedding_cache,
        modalities=MODALITIES,
        participants=participants,
        mask_manifest=args.mask_manifest,
        strict=True,
    )
    if len(common) != EXPECTED_OBSERVATIONS:
        raise ValueError("Condition embedding row count changed")
    for modality, dimension in EXPECTED_DIMS.items():
        if common.embeddings[modality].shape[-1] != dimension:
            raise ValueError(f"{modality} dimension changed")
        if int(common.masks[modality].sum()) != EXPECTED_VALID_WINDOWS:
            raise ValueError(f"{modality} common-valid count changed")
        if not torch.equal(common.masks[modality], common.masks["eeg"]):
            raise ValueError("Dataset masks are not identical across modalities")

    frame = pd.DataFrame(
        {
            "participant_id": common.participant_ids,
            "condition": common.conditions,
            "relaxation": common.targets[:, 0].numpy(),
            "discomfort": common.targets[:, 1].numpy(),
        }
    )
    frame = frame.merge(
        labels[["participant_id", "condition", "presentation_position"]],
        on=["participant_id", "condition"],
        how="left",
        validate="one_to_one",
    )
    participants_array = frame["participant_id"].astype(str).to_numpy()
    conditions_array = frame["condition"].astype(str).to_numpy()

    quality = _quality_frame(masks, labels, participants)
    quality.to_csv(args.output_dir / "condition_quality_and_missingness.csv", index=False)

    blocks: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    pooled_spectra: dict[str, dict[str, Any]] = {}
    window_spectra: dict[str, dict[str, Any]] = {}
    modality_rows: list[dict[str, Any]] = []
    for modality in MODALITIES:
        pooled, count = _pooled(common, modality)
        blocks[modality] = pooled
        counts[modality] = count
        pooled_spectrum = _unweighted_spectrum(pooled)
        window_spectrum, window_scale = _weighted_window_spectrum(
            common.embeddings[modality].numpy(), common.masks[modality].numpy().astype(bool)
        )
        pooled_spectra[modality] = pooled_spectrum
        window_spectra[modality] = window_spectrum
        internal_counts = internal.masks[modality].numpy().sum(axis=1)
        source_valid = _as_bool(formal_masks[f"source_{modality}_valid"])
        coordinate_sd = pooled.std(axis=0)
        row_norm = np.linalg.norm(pooled, axis=1)
        modality_rows.append(
            {
                "modality": modality,
                "raw_dimension": EXPECTED_DIMS[modality],
                "raw_width_share": EXPECTED_DIMS[modality] / sum(EXPECTED_DIMS.values()),
                "internal_valid_windows": int(internal.masks[modality].sum()),
                "source_valid_windows": int(source_valid.sum()),
                "common_valid_windows": int(common.masks[modality].sum()),
                "internal_missing_windows": int(NOMINAL_WINDOWS - int(internal.masks[modality].sum())),
                "source_missing_windows": int(NOMINAL_WINDOWS - int(source_valid.sum())),
                "common_masked_windows": int(NOMINAL_WINDOWS - int(common.masks[modality].sum())),
                "internal_zero_observations": int((internal_counts == 0).sum()),
                "common_zero_observations": int((count == 0).sum()),
                "common_count_min": int(count.min()),
                "common_count_median": float(np.median(count)),
                "common_count_max": int(count.max()),
                "pooled_row_l2_median": float(np.median(row_norm)),
                "pooled_coordinate_sd_median": float(np.median(coordinate_sd)),
                "pooled_coordinate_sd_mean": float(coordinate_sd.mean()),
                "participant_feature_r2_descriptive": _factor_r2(pooled, participants_array),
                "condition_feature_r2_descriptive": _factor_r2(pooled, conditions_array),
                "relaxation_feature_r2_descriptive": _continuous_feature_r2(
                    pooled, frame["relaxation"].to_numpy(dtype=float)
                ),
                "discomfort_feature_r2_descriptive": _continuous_feature_r2(
                    pooled, frame["discomfort"].to_numpy(dtype=float)
                ),
                "pooled_entropy_effective_rank": pooled_spectrum["entropy_effective_rank"],
                "pooled_pc95": pooled_spectrum["pc_count_95"],
                "pooled_matrix_rank": pooled_spectrum["matrix_rank"],
                "window_entropy_effective_rank_condition_balanced": window_spectrum[
                    "entropy_effective_rank"
                ],
                "window_pc95_condition_balanced": window_spectrum["pc_count_95"],
                "window_matrix_rank_condition_balanced": window_spectrum["matrix_rank"],
                **window_scale,
                "pooled_feature_sha256": _array_sha(pooled),
            }
        )
    modality_frame = pd.DataFrame(modality_rows)
    modality_frame.to_csv(args.output_dir / "modality_audit.csv", index=False)

    spectrum_rows: list[dict[str, Any]] = []
    for level, spectra in (("condition_pooled", pooled_spectra), ("window_condition_balanced", window_spectra)):
        for modality, spectrum in spectra.items():
            for component, (eigenvalue, ratio) in enumerate(
                zip(spectrum["eigenvalues"], spectrum["explained_variance_ratio"], strict=True), start=1
            ):
                spectrum_rows.append(
                    {
                        "analysis_level": level,
                        "modality": modality,
                        "component": component,
                        "eigenvalue": eigenvalue,
                        "explained_ratio": ratio,
                    }
                )
    pd.DataFrame(spectrum_rows).to_csv(args.output_dir / "variance_spectra.csv", index=False)

    fold_rank_rows: list[dict[str, Any]] = []
    for fold in folds:
        train, _validation, _test = _fold_indexes(participants_array, fold)
        for modality in MODALITIES:
            spectrum = _unweighted_spectrum(blocks[modality][train])
            fold_rank_rows.append(
                {
                    "fold_index": fold.fold_index,
                    "test_participant": fold.test_participant,
                    "validation_participant": fold.validation_participant,
                    "modality": modality,
                    "analysis_level": "condition_pooled_outer_train_only",
                    "matrix_rank": spectrum["matrix_rank"],
                    "entropy_effective_rank": spectrum["entropy_effective_rank"],
                    "pc_count_80": spectrum["pc_count_80"],
                    "pc_count_90": spectrum["pc_count_90"],
                    "pc_count_95": spectrum["pc_count_95"],
                }
            )
    pd.DataFrame(fold_rank_rows).to_csv(args.output_dir / "fold_local_pooled_rank.csv", index=False)

    joint_frame = _joint_fold_audit(blocks, counts, participants_array, folds)
    joint_frame.to_csv(args.output_dir / "joint_pca_dominance_audit.csv", index=False)

    target_rows: list[dict[str, Any]] = []
    for modality in MODALITIES:
        for record in _fixed_target_diagnostic(blocks[modality], counts[modality], frame, folds):
            target_rows.append({"modality": modality, **record})
    target_frame = pd.DataFrame(target_rows)
    target_frame.to_csv(args.output_dir / "fixed_target_information_probe.csv", index=False)

    zero_keys = quality.loc[quality["common_valid_windows"] == 0, ["participant_id", "condition"]]
    zero_key_records = zero_keys.to_dict(orient="records")
    if zero_key_records != [{"participant_id": "P004", "condition": "C6"}]:
        raise ValueError(f"Expected only P004/C6 all-missing, got {zero_key_records}")
    discomfort_zero = int(np.isclose(labels["discomfort"].to_numpy(dtype=float), 0.0).sum())
    discomfort_nonzero = int(len(labels) - discomfort_zero)
    if (discomfort_zero, discomfort_nonzero) != (65, 16):
        raise ValueError("Discomfort zero/nonzero composition changed")

    quality_counts = (
        quality.groupby(["common_valid_windows", "shared_quality"], as_index=False)
        .size()
        .sort_values("common_valid_windows")
    )
    quality_counts.to_csv(args.output_dir / "quality_count_distribution.csv", index=False)

    summary = {
        "schema_version": "relax_foundation_feature_audit_v2",
        "status": "pre_candidate descriptive audit plus fixed fold-safe target diagnostic",
        "protocol": {
            "participants": list(participants),
            "participant_count": 9,
            "observations": 81,
            "nominal_windows": NOMINAL_WINDOWS,
            "common_valid_windows": EXPECTED_VALID_WINDOWS,
            "folds": 9,
            "outer_training_participants": 7,
            "outer_validation_participants": 1,
            "outer_test_participants": 1,
        },
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for name, path in {
                "embedding_cache": args.embedding_cache,
                "labels": args.labels,
                "windows": args.windows,
                "mask_manifest": args.mask_manifest,
                "split_manifest": args.split_manifest,
                "cohorts": args.cohorts,
            }.items()
        },
        "modalities": modality_frame.to_dict(orient="records"),
        "shared_missingness": {
            "formal_masks_identical_across_modalities": True,
            "zero_common_window_observations": zero_key_records,
            "quality_definition": "common_valid_windows / source_windows",
            "quality_is_shared_not_modality_specific": True,
            "quality_distribution": quality_counts.to_dict(orient="records"),
        },
        "labels": {
            "discomfort_zero_count": discomfort_zero,
            "discomfort_nonzero_count": discomfort_nonzero,
        },
        "historical_joint_pca8_fold_mean": joint_frame.mean(numeric_only=True).to_dict(),
        "fixed_target_information_probe": target_rows,
        "cache_provenance": {
            "device": common.metadata.get("device"),
            "cuda_used": common.metadata.get("cuda_used"),
            "cuda_device_name": common.metadata.get("cuda_device_name"),
        },
        "audit_source_sha256": file_sha256(__file__),
    }
    _write_json(args.output_dir / "feature_audit_summary.json", summary)

    compact = modality_frame[
        [
            "modality",
            "raw_dimension",
            "internal_valid_windows",
            "source_valid_windows",
            "common_valid_windows",
            "common_zero_observations",
            "pooled_row_l2_median",
            "pooled_coordinate_sd_median",
            "pooled_entropy_effective_rank",
            "pooled_pc95",
            "window_entropy_effective_rank_condition_balanced",
            "window_pc95_condition_balanced",
            "participant_feature_r2_descriptive",
            "condition_feature_r2_descriptive",
            "relaxation_feature_r2_descriptive",
            "discomfort_feature_r2_descriptive",
        ]
    ]
    report = [
        "# July 18 Five-Modality Frozen-Feature Audit",
        "",
        "## Contract checks",
        "",
        "The frozen contract contains exactly nine EEG-eligible participants, 81 unique participant-condition labels, nine fixed 7/1/1 LOPO folds, 567 nominal windows, and 545 common-valid windows. EEG, ECG, eye, head, and video use the identical formal common mask. The only all-missing observation is `P004/C6`; `P004/C7` has one common-valid window.",
        "",
        "Head contributes 18 handcrafted Project-A window features. Therefore an all-five model is a hybrid foundation-feature pipeline, even though the other branches are frozen encoder embeddings.",
        "",
        "## Dimensions, missingness, scale, rank, and descriptive information",
        "",
        compact.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Window spectra standardize coordinates with weights that give each nonempty participant-condition total mass one. Condition-pooled spectra use the 81 condition rows and explicitly retain the zero vector for P004/C6. These are descriptive whole-cohort measurements, not held-out performance estimates.",
        "",
        "Participant, condition, relaxation, and discomfort R2 entries are fractions of standardized feature sum-of-squares fit by a one-factor linear design over all 81 rows. They measure association, not generalizable prediction.",
        "",
        "## Shared quality and missingness",
        "",
        quality_counts.to_markdown(index=False, floatfmt=".4f"),
        "",
        "`shared_quality = common_valid_windows / 7`. Because every formal modality mask is identical, a sample-dependent modality-quality gate cannot learn which modality is low quality. The model may append shared presence and quality fields, while modality-specific reliability must come from train-only global expert weights and deletion ablations.",
        "",
        f"Discomfort is zero for {discomfort_zero}/81 observations and nonzero for {discomfort_nonzero}/81. Final evaluation must therefore report zero/nonzero discomfort strata in addition to aggregate MAE.",
        "",
        "## Joint PCA width dominance",
        "",
        f"The historical no-EEG input is 1,942 dimensions: 1,938 pooled coordinates plus four duplicated presence flags. Across outer-training folds, PCA8 explains {joint_frame['historical_pca8_explained_variance'].mean():.4%} of standardized variance.",
        "",
        "| historical block | raw coordinates | mean PCA8 loading-mass share |",
        "| --- | ---: | ---: |",
    ]
    for modality in HISTORICAL_NO_EEG:
        report.append(
            f"| {modality} | {EXPECTED_DIMS[modality]} | {joint_frame[f'historical_{modality}_loading_mass_share'].mean():.4%} |"
        )
    report.extend(
        [
            "",
            "A coordinate-standardized joint PCA still lets wide ECG/video blocks supply most axes; coordinate scaling is not block balancing. Compressing approximately 1,942 dimensions to eight axes discards about one third of standardized train variance on average and gives no protected capacity to eye or head. This does not prove that discarded variance is target-relevant, but it makes PCA8 an unsafe default.",
            "",
            f"For the complete five-modality 2,962-coordinate input, unbalanced PCA12 explains {joint_frame['full_unbalanced_pca12_explained_variance'].mean():.4%}, while explicit `1/sqrt(block width)` balancing explains {joint_frame['full_block_balanced_pca12_explained_variance'].mean():.4%} in the balanced metric. These percentages are not directly comparable objectives; the latter deliberately gives each modality equal total standardized block variance.",
            "",
            "## Fixed fold-safe target-information diagnostic",
            "",
            target_frame[
                [
                    "modality",
                    "target",
                    "condition_anchor_mae",
                    "corrected_oof_mae",
                    "corrected_delta_vs_condition",
                    "correction_vs_condition_residual_correlation",
                    "mean_absolute_feature_correction_nonmissing",
                    "forced_zero_minus_legacy_mae",
                ]
            ].to_markdown(index=False, floatfmt=".6f"),
            "",
            "Every scaler, PCA, and Ridge diagnostic was fitted on only the seven outer-training participants. Its fixed PCA8/Ridge100 definition was not tuned. Unlike the July 17 diagnostic, this version forces the P004/C6 feature correction to zero and correlates the feature correction with the Condition-anchor residual rather than correlating the complete prediction with the target.",
            "",
            "## Audit-led method decision",
            "",
            "- Compare block-balanced joint PCA12 directly with modality-wise PCA dimensions summing to the same 12-axis budget.",
            "- Allocate modality-wise dimensions from train-fold condition-balanced window effective rank, with one protected axis per modality and deterministic largest-remainder allocation.",
            "- Restrict supervised reduction to one PLS1 direction per modality and target after unsupervised precompression.",
            "- Restrict shared/private fusion to two regularized shared factors and one private factor per modality.",
            "- Use the all-five modality-expert simplex as the primary synthesis, with global target-specific weights and systematic leave-one-modality-out retraining.",
            "- Do not use neural attention, learned sample-wise gates, tensor fusion, or architecture/rank search.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "/home/link/miniconda3/envs/egoEMOTION/bin/python scripts/audit_relax_foundation_features_v2.py \\",
            f"  --embedding-cache {args.embedding_cache} \\",
            f"  --labels {args.labels} \\",
            f"  --windows {args.windows} \\",
            f"  --mask-manifest {args.mask_manifest} \\",
            f"  --split-manifest {args.split_manifest} \\",
            f"  --cohorts {args.cohorts} \\",
            f"  --output-dir {args.output_dir}",
            "```",
            "",
        ]
    )
    (args.output_dir / "feature_audit_report.md").write_text("\n".join(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--mask-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument("--cohort", default="eeg_eligible")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

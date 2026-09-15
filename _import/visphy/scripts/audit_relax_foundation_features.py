"""Reproducible audit of frozen Relax multimodal condition embeddings.

This is a read-only diagnostic.  It never retrains an encoder and never writes
outside ``--output-dir``.  Descriptive whole-cohort quantities are explicitly
separated from fold-local quantities.  Every fold-local scaler, PCA and fixed
Ridge diagnostic excludes the outer validation and test participants.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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
from src.data.relax_dataset import RelaxConditionEmbeddingDataset


MODALITIES = ("eeg", "ecg", "eye", "head", "video")
CURRENT_NO_EEG = ("ecg", "eye", "head", "video")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _array_sha(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _pooled(dataset: RelaxConditionEmbeddingDataset, modality: str) -> tuple[np.ndarray, np.ndarray]:
    values = dataset.embeddings[modality].numpy().astype(np.float64, copy=False)
    mask = dataset.masks[modality].numpy().astype(bool, copy=False)
    counts = mask.sum(axis=1).astype(np.int64)
    pooled = (values * mask[..., None]).sum(axis=1) / np.maximum(counts[:, None], 1)
    pooled[counts == 0] = 0.0
    if not np.isfinite(pooled).all():
        raise ValueError(f"Non-finite pooled values for {modality}")
    return pooled, counts


def _spectrum(values: np.ndarray) -> dict[str, Any]:
    scaled = StandardScaler().fit_transform(values)
    singular = np.linalg.svd(scaled - scaled.mean(axis=0), compute_uv=False)
    eigen = singular**2 / max(len(values) - 1, 1)
    tolerance = max(float(eigen.max(initial=0.0)), 1.0) * 1e-12
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


def _factor_r2(values: np.ndarray, factor: np.ndarray) -> float:
    """Descriptive fraction of standardized feature sum-of-squares fit by a factor."""
    scaled = StandardScaler().fit_transform(values)
    design = pd.get_dummies(pd.Series(factor, dtype=str), drop_first=True).to_numpy(dtype=np.float64)
    design = np.column_stack([np.ones(len(design)), design])
    fitted = design @ np.linalg.pinv(design) @ scaled
    centered = scaled - scaled.mean(axis=0)
    return float(np.square(fitted - fitted.mean(axis=0)).sum() / np.square(centered).sum())


def _residualize(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    """Remove descriptive participant/condition fixed effects from a feature matrix."""
    coefficients = np.linalg.pinv(design) @ values
    return values - design @ coefficients


def _balanced_pca8_linear_cka(
    left: np.ndarray,
    right: np.ndarray,
    design: np.ndarray | None = None,
) -> float:
    """Equal-width PCA8 linear CKA; descriptive, not a held-out statistic."""

    def latent(values: np.ndarray) -> np.ndarray:
        scaled = StandardScaler().fit_transform(values)
        if design is not None:
            scaled = _residualize(scaled, design)
        components = min(8, len(scaled) - 1, scaled.shape[1])
        reduced = PCA(n_components=components, svd_solver="full").fit_transform(scaled)
        return reduced - reduced.mean(axis=0, keepdims=True)

    left_latent = latent(left)
    right_latent = latent(right)
    left_gram = left_latent @ left_latent.T
    right_gram = right_latent @ right_latent.T
    numerator = float(np.sum(left_gram * right_gram))
    denominator = float(
        np.sqrt(np.sum(np.square(left_gram)) * np.sum(np.square(right_gram)))
    )
    return numerator / denominator if denominator else 0.0


def _joint_blocks(
    blocks: dict[str, np.ndarray],
    counts: dict[str, np.ndarray],
    modalities: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    pieces: list[np.ndarray] = []
    spans: dict[str, tuple[int, int]] = {}
    offset = 0
    for modality in modalities:
        piece = np.column_stack([blocks[modality], (counts[modality] > 0).astype(np.float64)])
        pieces.append(piece)
        spans[modality] = (offset, offset + piece.shape[1])
        offset += piece.shape[1]
    return np.concatenate(pieces, axis=1), spans


def _reconstruction_fraction(original: np.ndarray, reconstructed: np.ndarray) -> float:
    denominator = float(np.square(original).sum())
    return 1.0 - float(np.square(original - reconstructed).sum()) / denominator if denominator else 1.0


def _target_diagnostic(
    values: np.ndarray,
    frame: pd.DataFrame,
    folds: list[Any],
) -> list[dict[str, Any]]:
    """Fixed PCA8/Ridge100 diagnostic; no validation or test labels affect fitting."""
    participants = frame["participant_id"].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        predictions: list[float] = []
        truths: list[float] = []
        anchors: list[float] = []
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
            residual = frame.iloc[train][target].to_numpy(dtype=np.float64) - train_anchor
            model = Ridge(alpha=100.0).fit(train_latent, residual)
            correction = np.clip(model.predict(test_latent), -0.2, 0.2)
            prediction = np.clip(test_anchor + correction, 0.0, 1.0)
            predictions.extend(prediction.tolist())
            anchors.extend(test_anchor.tolist())
            truths.extend(frame.iloc[test][target].to_numpy(dtype=np.float64).tolist())
        truth = np.asarray(truths)
        prediction = np.asarray(predictions)
        anchor = np.asarray(anchors)
        rows.append(
            {
                "target": target,
                "fixed_components": 8,
                "fixed_alpha": 100.0,
                "fixed_gamma": 1.0,
                "oof_mae": float(np.mean(np.abs(truth - prediction))),
                "condition_anchor_mae": float(np.mean(np.abs(truth - anchor))),
                "delta_vs_condition": float(
                    np.mean(np.abs(truth - prediction)) - np.mean(np.abs(truth - anchor))
                ),
                "oof_prediction_target_correlation": float(np.corrcoef(prediction, truth)[0, 1]),
            }
        )
    return rows


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if file_sha256(args.embedding_cache) != EXPECTED_CACHE_SHA256:
        raise ValueError("Embedding cache hash does not match the frozen formal cache")
    cohorts = json.loads(args.cohorts.read_text(encoding="utf-8"))
    participants = tuple(str(value) for value in cohorts[args.cohort])
    if participants != EXPECTED_PARTICIPANTS:
        raise ValueError(f"Expected formal cohort {EXPECTED_PARTICIPANTS}, got {participants}")
    folds = _load_split_manifest(args.split_manifest, list(participants), strict=True)
    if len(folds) != 9:
        raise ValueError("Expected nine outer folds")

    internal = RelaxConditionEmbeddingDataset(
        args.embedding_cache,
        modalities=MODALITIES,
        participants=participants,
        strict=True,
    )
    common = RelaxConditionEmbeddingDataset(
        args.embedding_cache,
        modalities=MODALITIES,
        participants=participants,
        mask_manifest=args.mask_manifest,
        strict=True,
    )
    if len(common) != EXPECTED_OBSERVATIONS:
        raise ValueError(f"Expected {EXPECTED_OBSERVATIONS} observations")

    frame = pd.DataFrame(
        {
            "participant_id": common.participant_ids,
            "condition": common.conditions,
            "relaxation": common.targets[:, 0].numpy(),
            "discomfort": common.targets[:, 1].numpy(),
        }
    )
    participant_array = frame["participant_id"].astype(str).to_numpy()
    condition_array = frame["condition"].astype(str).to_numpy()

    blocks: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    modality_rows: list[dict[str, Any]] = []
    spectra: dict[str, dict[str, Any]] = {}
    for modality in MODALITIES:
        pooled, count = _pooled(common, modality)
        blocks[modality] = pooled
        counts[modality] = count
        internal_count = internal.masks[modality].numpy().sum(axis=1)
        spectrum = _spectrum(pooled)
        spectra[modality] = spectrum
        row_norm = np.linalg.norm(pooled, axis=1)
        coordinate_sd = pooled.std(axis=0)
        modality_rows.append(
            {
                "modality": modality,
                "dimension": int(pooled.shape[1]),
                "internal_valid_windows": int(internal.masks[modality].sum()),
                "internal_missing_vs_567": int(567 - int(internal.masks[modality].sum())),
                "internal_zero_observations": int((internal_count == 0).sum()),
                "common_valid_windows": int(common.masks[modality].sum()),
                "common_masked_vs_567": int(567 - int(common.masks[modality].sum())),
                "common_zero_observations": int((count == 0).sum()),
                "valid_windows_min": int(count.min()),
                "valid_windows_median": float(np.median(count)),
                "valid_windows_max": int(count.max()),
                "row_l2_median": float(np.median(row_norm)),
                "coordinate_sd_median": float(np.median(coordinate_sd)),
                "coordinate_sd_mean": float(coordinate_sd.mean()),
                "participant_feature_r2_descriptive": _factor_r2(pooled, participant_array),
                "condition_feature_r2_descriptive": _factor_r2(pooled, condition_array),
                **{key: value for key, value in spectrum.items() if not isinstance(value, list)},
                "feature_sha256": _array_sha(pooled),
            }
        )
    modality_frame = pd.DataFrame(modality_rows)
    modality_frame.to_csv(args.output_dir / "modality_audit.csv", index=False)

    participant_design = pd.get_dummies(
        pd.Series(participant_array, dtype=str), drop_first=True
    ).to_numpy(dtype=np.float64)
    condition_design = pd.get_dummies(
        pd.Series(condition_array, dtype=str), drop_first=True
    ).to_numpy(dtype=np.float64)
    nuisance_design = np.column_stack(
        [np.ones(len(frame), dtype=np.float64), participant_design, condition_design]
    )
    redundancy_rows: list[dict[str, Any]] = []
    for left, right in combinations(MODALITIES, 2):
        redundancy_rows.append(
            {
                "modality_a": left,
                "modality_b": right,
                "balanced_pca_components_per_modality": 8,
                "linear_cka_balanced_pca8": _balanced_pca8_linear_cka(
                    blocks[left], blocks[right]
                ),
                "linear_cka_balanced_pca8_after_participant_condition_residualization": (
                    _balanced_pca8_linear_cka(
                        blocks[left], blocks[right], nuisance_design
                    )
                ),
                "status": "descriptive_posthoc_reproducibility_supplement",
            }
        )
    redundancy_frame = pd.DataFrame(redundancy_rows).sort_values(
        ["modality_a", "modality_b"]
    )
    redundancy_frame.to_csv(
        args.output_dir / "cross_modal_redundancy_balanced_pca8_cka.csv", index=False
    )

    spectrum_rows: list[dict[str, Any]] = []
    for modality, spectrum in spectra.items():
        for index, (eigen, ratio) in enumerate(
            zip(spectrum["eigenvalues"], spectrum["explained_variance_ratio"], strict=True), start=1
        ):
            spectrum_rows.append(
                {"modality": modality, "component": index, "eigenvalue": eigen, "explained_ratio": ratio}
            )
    pd.DataFrame(spectrum_rows).to_csv(args.output_dir / "global_variance_spectra.csv", index=False)

    fold_spectrum_rows: list[dict[str, Any]] = []
    joint_rows: list[dict[str, Any]] = []
    joint, spans = _joint_blocks(blocks, counts, CURRENT_NO_EEG)
    if joint.shape != (81, 1942):
        raise ValueError(f"Current no-EEG feature shape must be (81, 1942), got {joint.shape}")
    for fold in folds:
        train, validation, test = _fold_indexes(participant_array, fold)
        for modality in MODALITIES:
            fold_spectrum_rows.append(
                {
                    "fold_index": fold.fold_index,
                    "test_participant": fold.test_participant,
                    "validation_participant": fold.validation_participant,
                    "modality": modality,
                    **{key: value for key, value in _spectrum(blocks[modality][train]).items() if not isinstance(value, list)},
                }
            )
        scaler = StandardScaler().fit(joint[train])
        train_scaled = scaler.transform(joint[train])
        pca = PCA(n_components=8, svd_solver="full").fit(train_scaled)
        reconstruction = pca.inverse_transform(pca.transform(train_scaled))
        row: dict[str, Any] = {
            "fold_index": fold.fold_index,
            "test_participant": fold.test_participant,
            "validation_participant": fold.validation_participant,
            "joint_input_dimension": int(joint.shape[1]),
            "joint_components": 8,
            "joint_explained_variance": float(pca.explained_variance_ratio_.sum()),
        }
        for modality, (start, stop) in spans.items():
            row[f"{modality}_dimension_with_presence"] = stop - start
            row[f"{modality}_loading_mass_share"] = float(
                np.square(pca.components_[:, start:stop]).sum() / pca.n_components_
            )
            row[f"{modality}_reconstruction_fraction"] = _reconstruction_fraction(
                train_scaled[:, start:stop], reconstruction[:, start:stop]
            )

        total_ss = 0.0
        retained_two = 0.0
        retained_eight = 0.0
        for modality in CURRENT_NO_EEG:
            values = blocks[modality][train]
            scaled = StandardScaler().fit_transform(values)
            total_ss += float(np.square(scaled).sum())
            for components, key in ((2, "two"), (8, "eight")):
                count_components = min(components, len(train) - 1, scaled.shape[1])
                model = PCA(n_components=count_components, svd_solver="full").fit(scaled)
                reconstructed = model.inverse_transform(model.transform(scaled))
                retained = float(np.square(scaled).sum() - np.square(scaled - reconstructed).sum())
                if key == "two":
                    retained_two += retained
                else:
                    retained_eight += retained
        row["modalitywise_2_each_total_components"] = 8
        row["modalitywise_2_each_total_reconstruction"] = retained_two / total_ss
        row["modalitywise_8_each_total_components"] = 32
        row["modalitywise_8_each_total_reconstruction"] = retained_eight / total_ss
        joint_rows.append(row)

    pd.DataFrame(fold_spectrum_rows).to_csv(args.output_dir / "fold_local_variance_spectra.csv", index=False)
    joint_frame = pd.DataFrame(joint_rows)
    joint_frame.to_csv(args.output_dir / "joint_pca8_fold_audit.csv", index=False)

    target_rows: list[dict[str, Any]] = []
    for modality in MODALITIES:
        for record in _target_diagnostic(blocks[modality], frame, folds):
            target_rows.append({"modality": modality, **record})
    target_frame = pd.DataFrame(target_rows)
    target_frame.to_csv(args.output_dir / "fixed_target_information_probe.csv", index=False)

    zero_keys = [
        {"participant_id": frame.iloc[index]["participant_id"], "condition": frame.iloc[index]["condition"]}
        for index in np.flatnonzero(counts["eeg"] == 0)
    ]
    summary = {
        "schema_version": "relax_foundation_feature_audit_v1",
        "scope": "descriptive audit plus fixed fold-safe diagnostics; not confirmatory method selection",
        "inputs": {
            "embedding_cache": {"path": str(args.embedding_cache.resolve()), "sha256": file_sha256(args.embedding_cache)},
            "mask_manifest": {"path": str(args.mask_manifest.resolve()), "sha256": file_sha256(args.mask_manifest)},
            "split_manifest": {"path": str(args.split_manifest.resolve()), "sha256": file_sha256(args.split_manifest)},
            "cohorts": {"path": str(args.cohorts.resolve()), "sha256": file_sha256(args.cohorts)},
        },
        "participants": list(participants),
        "observation_count": len(frame),
        "nominal_internal_windows": 567,
        "common_valid_windows": int(common.masks["eeg"].sum()),
        "zero_common_window_observations": zero_keys,
        "current_no_eeg_joint_dimension": int(joint.shape[1]),
        "modalities": modality_frame.to_dict(orient="records"),
        "cross_modal_redundancy_balanced_pca8_cka": redundancy_frame.to_dict(
            orient="records"
        ),
        "joint_pca8_fold_mean": joint_frame.mean(numeric_only=True).to_dict(),
        "joint_pca8_fold_min": joint_frame.min(numeric_only=True).to_dict(),
        "joint_pca8_fold_max": joint_frame.max(numeric_only=True).to_dict(),
        "fixed_target_information_probe": target_rows,
        "torch_cuda_cache_provenance": {
            "device": common.metadata.get("device"),
            "cuda_used": common.metadata.get("cuda_used"),
            "cuda_device_name": common.metadata.get("cuda_device_name"),
        },
    }
    if summary["common_valid_windows"] != EXPECTED_VALID_WINDOWS:
        raise ValueError("Common-valid mask count changed")
    _write_json(args.output_dir / "feature_audit_summary.json", summary)

    modality_table = modality_frame[
        [
            "modality",
            "dimension",
            "row_l2_median",
            "coordinate_sd_median",
            "internal_valid_windows",
            "common_valid_windows",
            "common_zero_observations",
            "entropy_effective_rank",
            "pc_count_80",
            "pc_count_90",
            "pc_count_95",
            "participant_feature_r2_descriptive",
            "condition_feature_r2_descriptive",
        ]
    ].copy()
    report = [
        "# Frozen Multimodal Feature Audit",
        "",
        "## Scope labels",
        "",
        "- **Facts** below are computed from the frozen cache and fixed nine-participant contract.",
        "- **Inferences** interpret those measurements but are not independent validation.",
        "- The fixed PCA8/Ridge100 target probe is a diagnostic chosen before reading its scores; it is not a candidate sweep.",
        "",
        "## Facts: dimensions, missingness, rank, and information structure",
        "",
        modality_table.to_markdown(index=False, floatfmt=".4f"),
        "",
        f"The formal cohort has 81 observations and 567 nominal internal windows. The common-valid contract retains {EXPECTED_VALID_WINDOWS} windows for every modality. The only zero-window observation is `{zero_keys[0]['participant_id']}/{zero_keys[0]['condition']}`.",
        "",
        "Participant/condition R2 values are descriptive fractions of standardized feature sum-of-squares explained by one-hot factors on all 81 rows; they are not held-out prediction scores.",
        "",
        "Raw row norms and coordinate standard deviations are reported to expose scale differences before the train-fold standardization used by every candidate.",
        "",
        "## Facts: cross-modal redundancy supplement",
        "",
        "The following equal-width linear CKA values use eight descriptive PCA axes per modality. The residualized column first removes participant and condition fixed effects. This reproducibility supplement was generated after the formal matrix and did not affect candidate definitions or inference.",
        "",
        redundancy_frame.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Facts: current 1,942-to-8 joint PCA",
        "",
        f"Across the nine train-only fits, PCA8 explains {joint_frame['joint_explained_variance'].mean():.4%} of the 1,942-D standardized training variance (range {joint_frame['joint_explained_variance'].min():.4%}–{joint_frame['joint_explained_variance'].max():.4%}).",
        "",
        "| block | mean squared-loading mass | mean train reconstruction |",
        "| --- | ---: | ---: |",
    ]
    for modality in CURRENT_NO_EEG:
        report.append(
            f"| {modality} | {joint_frame[f'{modality}_loading_mass_share'].mean():.4%} | {joint_frame[f'{modality}_reconstruction_fraction'].mean():.4%} |"
        )
    report.extend(
        [
            "",
            f"At the same total eight axes, two independent PCs per modality retain {joint_frame['modalitywise_2_each_total_reconstruction'].mean():.4%} of standardized variance. Eight PCs per modality (32 total) retain {joint_frame['modalitywise_8_each_total_reconstruction'].mean():.4%}.",
            "",
            "## Facts: fixed fold-safe target-information diagnostic",
            "",
            target_frame[["modality", "target", "oof_mae", "condition_anchor_mae", "delta_vs_condition", "oof_prediction_target_correlation"]].to_markdown(index=False, floatfmt=".6f"),
            "",
            "Every diagnostic scaler/PCA/Ridge fit used only the seven outer-training participants. Validation and test labels did not affect fitting or hyperparameters.",
            "",
            "## Inferences",
            "",
            "- Coordinate-wise standardization fixes raw scale differences but does not fix block-size imbalance: ECG and video own most joint PCA8 loading mass because they contribute most coordinates.",
            "- Joint PCA8 is more variance-efficient than allocating only two PCs per no-EEG modality, but it provides almost no protected capacity to head and little to eye. Variance efficiency is not the same as target efficiency.",
            "- Eye and head are each extremely low-rank internally; ECG and video need materially more axes to reach the same variance threshold. A single shared component count is therefore a poor modality-neutral compression rule.",
            "- Participant structure exceeds condition structure in most modalities, so aggressive unsupervised compression can preserve participant identity rather than the cross-participant residual signal needed by LOPO prediction.",
            "- Equal-width CKA quantifies representation similarity but does not establish interchangeability or causal redundancy. Residualized CKA is the more relevant descriptive check for signal beyond participant and condition structure.",
            "- These measurements justify a small, preregistered comparison of equal-block joint PCA, modality-wise PCA, supervised linear compression, constrained shared/private factors, and low-capacity modality experts. They do not justify a neural architecture sweep.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "/home/link/miniconda3/envs/egoEMOTION/bin/python scripts/audit_relax_foundation_features.py \\",
            f"  --embedding-cache {args.embedding_cache} \\",
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
    parser.add_argument("--mask-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument("--cohort", default="eeg_eligible")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

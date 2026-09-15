"""Audit numerical integrity and isolation of the Relax embedding ladder."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as torch_functional

from mac.data.relax_dataset import RelaxConditionEmbeddingDataset


MODALITIES = ("eeg", "ecg", "eye", "head", "video")
DEFAULT_ROOT = ROOT / "artifacts/relax/neurorvq_embedding_ladder_20260718"
DEFAULT_CONTRACT = Path(
    "/mnt/c/Users/linki/amaster/data_collection_v3/analysis/real_time_inference/"
    "artifacts/cross_project_alignment_2026-07-16/eeg_eligible_ablation/contract"
)


def _hash_array(values: np.ndarray | torch.Tensor) -> str:
    array = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
    contiguous = np.ascontiguousarray(array)
    digest = sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(str(contiguous.shape).encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _matrix_metrics(values: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not len(matrix):
        raise ValueError(f"Expected a nonempty feature matrix, got {matrix.shape}")
    finite = np.isfinite(matrix)
    coordinate_std = np.std(matrix, axis=0)
    norms = np.linalg.norm(matrix, axis=1)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    tolerance = (
        max(centered.shape) * np.finfo(np.float64).eps * singular[0]
        if len(singular) and singular[0] > 0
        else 0.0
    )
    eigenvalues = singular**2
    positive = eigenvalues[eigenvalues > 0]
    if len(positive):
        probability = positive / positive.sum()
        effective_rank = float(np.exp(-np.sum(probability * np.log(probability))))
    else:
        effective_rank = 0.0
    unique = np.unique(matrix.astype(np.float32, copy=False), axis=0)
    return {
        "vectors": len(matrix),
        "dimension": matrix.shape[1],
        "all_finite": bool(finite.all()),
        "nonfinite_values": int((~finite).sum()),
        "all_zero_vectors": int(np.sum(np.all(matrix == 0.0, axis=1))),
        "exact_unique_vectors": len(unique),
        "exact_duplicate_vectors": len(matrix) - len(unique),
        "zero_variance_coordinates": int(np.sum(coordinate_std <= 1e-12)),
        "coordinate_std_mean": float(coordinate_std.mean()),
        "coordinate_std_median": float(np.median(coordinate_std)),
        "coordinate_std_min": float(coordinate_std.min()),
        "coordinate_std_max": float(coordinate_std.max()),
        "vector_norm_mean": float(norms.mean()),
        "vector_norm_std": float(norms.std()),
        "vector_norm_min": float(norms.min()),
        "vector_norm_max": float(norms.max()),
        "numeric_rank": int(np.sum(singular > tolerance)),
        "entropy_effective_rank": effective_rank,
        "matrix_sha256": _hash_array(matrix.astype(np.float32)),
    }


def _cosine_rows(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError(f"Cosine comparison shapes differ: {first.shape} vs {second.shape}")
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    cosine = np.divide(
        np.sum(first * second, axis=1),
        denominator,
        out=np.zeros(len(first), dtype=float),
        where=denominator > 0,
    )
    return {
        "count": len(cosine),
        "mean": float(cosine.mean()),
        "std": float(cosine.std()),
        "min": float(cosine.min()),
        "median": float(np.median(cosine)),
        "max": float(cosine.max()),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cohorts = json.loads(args.cohorts.read_text(encoding="utf-8"))
    participants = [str(value) for value in cohorts["eeg_eligible"]]
    stage_data: dict[str, RelaxConditionEmbeddingDataset] = {}
    feature_rows: list[dict[str, Any]] = []
    feature_matrices: dict[tuple[str, str], np.ndarray] = {}

    reference_keys: list[tuple[str, str]] | None = None
    reference_masks: dict[str, str] | None = None
    for stage, stage_manifest in manifest["stages"].items():
        cache = Path(stage_manifest["cache_path"])
        dataset = RelaxConditionEmbeddingDataset(
            cache,
            modalities=MODALITIES,
            participants=participants,
            mask_manifest=args.mask_manifest,
            strict=True,
        )
        stage_data[stage] = dataset
        if reference_keys is None:
            reference_keys = dataset.condition_keys()
        elif dataset.condition_keys() != reference_keys:
            raise ValueError(f"Condition order changed at stage {stage}")
        stage_mask_hashes = {modality: _hash_array(dataset.masks[modality]) for modality in MODALITIES}
        if reference_masks is None:
            reference_masks = stage_mask_hashes
        elif stage_mask_hashes != reference_masks:
            raise ValueError(f"Effective formal masks changed at stage {stage}")
        for modality in MODALITIES:
            mask = dataset.masks[modality].numpy().astype(bool)
            values = dataset.embeddings[modality].numpy()[mask]
            feature_matrices[(stage, modality)] = values
            feature_rows.append(
                {
                    "stage": stage,
                    "modality": modality,
                    "effective_windows": int(mask.sum()),
                    **_matrix_metrics(values),
                }
            )
    feature_frame = pd.DataFrame(feature_rows)
    if not feature_frame["all_finite"].all():
        raise ValueError("At least one stage/modality contains non-finite embeddings")
    if (feature_frame["all_zero_vectors"] > 0).any():
        raise ValueError("At least one common-valid embedding vector is all-zero")
    if (feature_frame["exact_unique_vectors"] < 2).any():
        raise ValueError("At least one stage/modality is constant")

    stage_pairs = (
        ("s1_reve_native_raw4", "s2_reve_native_linked2", "eeg"),
        ("s2_reve_native_linked2", "s3_reve_clean_linked2", "eeg"),
        ("s3_reve_clean_linked2", "r3_reve_official_session", "eeg"),
        ("s4_neurorvq_clean_z", "s5_neurorvq_clean_native", "eeg"),
        ("s3_reve_clean_linked2", "ecg_only_neurorvq", "eeg"),
        ("s5_neurorvq_clean_native", "s6_neurorvq_eeg_ecg", "eeg"),
        ("s0_legacy", "s1_reve_native_raw4", "ecg"),
        ("s0_legacy", "s2_reve_native_linked2", "ecg"),
        ("s6_neurorvq_eeg_ecg", "ecg_only_neurorvq", "ecg"),
    )
    cosine_records = []
    for left, right, modality in stage_pairs:
        first, second = feature_matrices[(left, modality)], feature_matrices[(right, modality)]
        if first.shape != second.shape:
            continue
        cosine_records.append(
            {
                "left_stage": left,
                "right_stage": right,
                "modality": modality,
                **_cosine_rows(first, second),
                "exactly_identical": _hash_array(first) == _hash_array(second),
            }
        )

    # Verify that S1 really is the native precursor of the legacy feature-axis pool.
    legacy = feature_matrices[("s0_legacy", "eeg")]
    native = feature_matrices[("s1_reve_native_raw4", "eeg")]
    reduced = (
        torch_functional.adaptive_avg_pool1d(
            torch.from_numpy(native).unsqueeze(1),
            1024,
        )
        .squeeze(1)
        .numpy()
    )
    difference = reduced.astype(np.float64) - legacy.astype(np.float64)
    legacy_reconstruction = {
        "cosine": _cosine_rows(reduced, legacy),
        "mean_absolute_error": float(np.mean(np.abs(difference))),
        "root_mean_square_error": float(np.sqrt(np.mean(np.square(difference)))),
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "note": "Tiny differences may arise from float64 signal preparation and a fresh deterministic CUDA pass; S0 remains the exact baseline cache.",
    }

    raw_qc = pd.read_csv(args.window_qc)
    qc_rows = []
    numeric_qc = [
        "std",
        "rms",
        "absolute_max",
        "fraction_absolute_ge_500",
        "delta_power_0p5_4",
        "theta_power_4_8",
        "alpha_power_8_13",
        "beta_power_13_30",
        "line_power_49_51",
    ]
    for stage_signal, frame in raw_qc.groupby("stage_signal", sort=True):
        record: dict[str, Any] = {"stage_signal": stage_signal, "windows": len(frame)}
        for column in numeric_qc:
            values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
            record[f"{column}_mean"] = float(values.mean())
            record[f"{column}_median"] = float(np.median(values))
            record[f"{column}_p95"] = float(np.quantile(values, 0.95))
            record[f"{column}_max"] = float(values.max())
        qc_rows.append(record)

    audit_dir = args.output_dir
    audit_dir.mkdir(parents=True, exist_ok=True)
    feature_frame.to_csv(audit_dir / "embedding_numerical_qc.csv", index=False)
    pd.DataFrame(cosine_records).to_csv(audit_dir / "stage_cosine_similarity.csv", index=False)
    pd.DataFrame(qc_rows).to_csv(audit_dir / "signal_qc_summary.csv", index=False)
    result = {
        "schema_version": "relax_embedding_ladder_audit_v1",
        "stages": list(manifest["stages"]),
        "participants": participants,
        "conditions": len(reference_keys or []),
        "common_valid_windows": int(stage_data["s0_legacy"].masks["eeg"].sum()),
        "all_effective_masks_identical": True,
        "all_embeddings_finite": True,
        "no_common_valid_all_zero_vectors": True,
        "unchanged_eye_head_video_tensor_hashes": manifest[
            "untouched_eye_head_video_tensor_sha256"
        ],
        "legacy_pool_reconstruction": legacy_reconstruction,
        "outputs": {
            "embedding_numerical_qc": str((audit_dir / "embedding_numerical_qc.csv").resolve()),
            "stage_cosine_similarity": str((audit_dir / "stage_cosine_similarity.csv").resolve()),
            "signal_qc_summary": str((audit_dir / "signal_qc_summary.csv").resolve()),
        },
    }
    _write_json(audit_dir / "audit_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cohorts", type=Path, default=DEFAULT_CONTRACT / "cohorts.json")
    parser.add_argument(
        "--mask-manifest",
        type=Path,
        default=DEFAULT_CONTRACT / "common_valid_window_masks.csv",
    )
    parser.add_argument("--window-qc", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    args.manifest = args.manifest or args.root / "embedding_ladder_manifest.json"
    args.window_qc = args.window_qc or args.root / "qc/preprocessed_window_qc.csv"
    args.output_dir = args.output_dir or args.root / "audit"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    audit(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

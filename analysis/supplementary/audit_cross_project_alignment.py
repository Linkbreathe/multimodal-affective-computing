"""Audit and rescore existing cross-project Relax artifacts without training models.

Source reports, labels, predictions, registries, and embedding caches are read-only.
All generated audit tables and figures are isolated under the requested output
directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr


TARGETS = ("relaxation", "discomfort")
CONDITIONS = tuple(f"C{index}" for index in range(1, 10))
RUN_TAG = "eegmap_m2_tp9_tp10_m1_20260714"
AUDIT_DATE = "2026-07-16"


@dataclass(frozen=True)
class Roots:
    project_a: Path
    project_b: Path
    output: Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def safe_spearman(truth: np.ndarray, prediction: np.ndarray) -> float:
    valid = np.isfinite(truth) & np.isfinite(prediction)
    if valid.sum() < 2 or np.unique(truth[valid]).size < 2 or np.unique(prediction[valid]).size < 2:
        return float("nan")
    return safe_float(spearmanr(truth[valid], prediction[valid]).statistic)


def safe_pearson(truth: np.ndarray, prediction: np.ndarray) -> float:
    valid = np.isfinite(truth) & np.isfinite(prediction)
    if valid.sum() < 2 or np.std(truth[valid]) <= 1e-15 or np.std(prediction[valid]) <= 1e-15:
        return float("nan")
    return safe_float(pearsonr(truth[valid], prediction[valid]).statistic)


def concordance_correlation(truth: np.ndarray, prediction: np.ndarray) -> float:
    valid = np.isfinite(truth) & np.isfinite(prediction)
    truth = truth[valid]
    prediction = prediction[valid]
    if truth.size < 2:
        return float("nan")
    covariance = float(np.mean((truth - truth.mean()) * (prediction - prediction.mean())))
    denominator = float(truth.var() + prediction.var() + (truth.mean() - prediction.mean()) ** 2)
    return (2.0 * covariance / denominator) if denominator > 0 else float("nan")


def r2_score_local(truth: np.ndarray, prediction: np.ndarray) -> float:
    valid = np.isfinite(truth) & np.isfinite(prediction)
    truth = truth[valid]
    prediction = prediction[valid]
    if truth.size < 2:
        return float("nan")
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    if denominator <= 1e-15:
        return float("nan")
    return 1.0 - float(np.sum((truth - prediction) ** 2)) / denominator


def pairwise_accuracy(truth: np.ndarray, prediction: np.ndarray) -> float:
    correct = 0.0
    total = 0
    for left in range(len(truth)):
        for right in range(left + 1, len(truth)):
            truth_delta = truth[left] - truth[right]
            prediction_delta = prediction[left] - prediction[right]
            if abs(truth_delta) <= 1e-12:
                continue
            total += 1
            if abs(prediction_delta) <= 1e-12:
                correct += 0.5
            elif np.sign(truth_delta) == np.sign(prediction_delta):
                correct += 1.0
    return (correct / total) if total else float("nan")


def score_target(
    frame: pd.DataFrame,
    truth_column: str,
    prediction_column: str,
    extra_group_columns: Iterable[str] = (),
) -> tuple[dict[str, float], pd.DataFrame]:
    truth = pd.to_numeric(frame[truth_column], errors="coerce").to_numpy(dtype=float)
    prediction = pd.to_numeric(frame[prediction_column], errors="coerce").to_numpy(dtype=float)
    group_columns = ["participant_id", *extra_group_columns]
    participant_rows: list[dict[str, Any]] = []
    for group_key, group in frame.groupby(group_columns, sort=True, dropna=False):
        group_key = group_key if isinstance(group_key, tuple) else (group_key,)
        group_truth = pd.to_numeric(group[truth_column], errors="coerce").to_numpy(dtype=float)
        group_prediction = pd.to_numeric(group[prediction_column], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(group_truth) & np.isfinite(group_prediction)
        group_truth = group_truth[valid]
        group_prediction = group_prediction[valid]
        if group_truth.size == 0:
            continue
        row: dict[str, Any] = {
            name: value for name, value in zip(group_columns, group_key, strict=True)
        }
        row.update(
            {
                "n_rows": int(group_truth.size),
                "mae": float(np.mean(np.abs(group_truth - group_prediction))),
                "rmse": float(np.sqrt(np.mean((group_truth - group_prediction) ** 2))),
                "spearman": safe_spearman(group_truth, group_prediction),
                "pairwise_accuracy": pairwise_accuracy(group_truth, group_prediction),
                "r2": r2_score_local(group_truth, group_prediction),
            }
        )
        participant_rows.append(row)
    participant = pd.DataFrame(participant_rows)
    valid = np.isfinite(truth) & np.isfinite(prediction)
    truth = truth[valid]
    prediction = prediction[valid]
    metrics = {
        "n_rows": int(truth.size),
        "mae": float(np.mean(np.abs(truth - prediction))),
        "participant_macro_mae": float(participant["mae"].mean()),
        "pooled_rmse": float(np.sqrt(np.mean((truth - prediction) ** 2))),
        "participant_macro_rmse": float(participant["rmse"].mean()),
        "global_spearman": safe_spearman(truth, prediction),
        "within_participant_spearman": float(participant["spearman"].mean(skipna=True)),
        "pairwise_accuracy": float(participant["pairwise_accuracy"].mean(skipna=True)),
        "global_pearson": safe_pearson(truth, prediction),
        "global_ccc": concordance_correlation(truth, prediction),
        "participant_r2_mean": float(participant["r2"].mean(skipna=True)),
    }
    return metrics, participant


def normalize_prediction_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    truth_options = {
        "relaxation": ("relaxation_true", "relaxation"),
        "discomfort": ("discomfort_true", "discomfort"),
    }
    prediction_options = {
        "relaxation": ("relaxation_pred", "pred_relaxation"),
        "discomfort": ("discomfort_pred", "pred_discomfort"),
    }
    truth_columns: dict[str, str] = {}
    prediction_columns: dict[str, str] = {}
    for target in TARGETS:
        truth_columns[target] = next(
            (column for column in truth_options[target] if column in frame.columns), ""
        )
        prediction_columns[target] = next(
            (column for column in prediction_options[target] if column in frame.columns), ""
        )
        if not truth_columns[target] or not prediction_columns[target]:
            raise ValueError(f"Prediction table lacks {target} truth/prediction columns")
    return frame, truth_columns, prediction_columns


def score_configuration(
    frame: pd.DataFrame,
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    frame, truth_columns, prediction_columns = normalize_prediction_frame(frame)
    extra_groups = ["seed"] if "seed" in frame.columns else []
    summary = dict(metadata)
    target_rows: list[dict[str, Any]] = []
    participant_by_target: dict[str, pd.DataFrame] = {}
    for target in TARGETS:
        metrics, participant = score_target(
            frame,
            truth_columns[target],
            prediction_columns[target],
            extra_group_columns=extra_groups,
        )
        row = {**metadata, "target": target, **metrics}
        target_rows.append(row)
        participant["target"] = target
        participant_by_target[target] = participant
        for key, value in metrics.items():
            summary[f"{target}_{key}"] = value
    summary["macro_mae"] = float(
        np.mean([summary[f"{target}_participant_macro_mae"] for target in TARGETS])
    )
    summary["n_rows"] = int(len(frame))
    participant = participant_by_target["relaxation"].merge(
        participant_by_target["discomfort"],
        on=["participant_id", *extra_groups],
        suffixes=("_relaxation", "_discomfort"),
        validate="one_to_one",
    )
    participant["macro_mae"] = (
        participant["mae_relaxation"] + participant["mae_discomfort"]
    ) / 2.0
    for key, value in metadata.items():
        participant[key] = value
    return summary, target_rows, participant


def check_record(
    name: str,
    passed: bool,
    observed: Any,
    expected: Any,
    evidence: str,
    category: str = "contract",
) -> dict[str, Any]:
    return {
        "category": category,
        "check": name,
        "status": "PASS" if passed else "FAIL",
        "observed": observed,
        "expected": expected,
        "evidence": evidence,
    }


def source_provenance(roots: Roots) -> pd.DataFrame:
    paths = {
        "file_1_current_project_a": roots.project_a
        / "artifacts/reports/PROJECT_COMPREHENSIVE_REPORT_2026-07-14_zh.md",
        "file_2_project_a_timeline": roots.project_a
        / "artifacts/reports/project_timeline_chronological_2026-07-14_zh.md",
        "file_3_legacy_project_b": roots.project_b
        / "reports/relax_foundation_probe/relax_foundation_report.md",
        "project_b_corrected_successor": roots.project_b
        / f"reports/relax_{RUN_TAG}/full_experiment_report_zh.md",
        "project_a_labels": roots.project_a / "artifacts/preprocessed/condition_labels.csv",
        "project_a_windows": roots.project_a / "artifacts/preprocessed/windows.csv",
        "project_b_labels": roots.project_b
        / "artifacts/relax_model/runs/relax-foundation-reve-large-20260705/preprocessed/condition_labels.csv",
        "project_b_windows": roots.project_b
        / "artifacts/relax_model/runs/relax-foundation-reve-large-20260705/preprocessed/windows.csv",
        "project_a_classical_oof": roots.project_a
        / "artifacts/reports/condition_level_lopo_predictions.csv",
        "project_a_dcnn_oof": roots.project_a
        / "artifacts/reports/dcnn_condition_lopo_predictions.csv",
        "project_b_corrected_registry": roots.project_b
        / f"reports/relax_{RUN_TAG}/result_registry.csv",
        "project_b_corrected_cohorts": roots.project_b
        / f"artifacts/relax/runs/{RUN_TAG}/cohorts.json",
        "project_b_corrected_qc": roots.project_b
        / f"artifacts/relax/runs/{RUN_TAG}/qc_manifest.json",
        "project_b_corrected_condition_cache": roots.project_b
        / f"artifacts/relax/runs/{RUN_TAG}/condition_embeddings.pt",
    }
    rows = []
    for source_id, path in paths.items():
        rows.append(
            {
                "source_id": source_id,
                "path": str(path),
                "exists": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else None,
                "sha256": sha256(path) if path.is_file() else None,
            }
        )
    return pd.DataFrame(rows)


def audit_data_contract(roots: Roots) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    project_a_labels_path = roots.project_a / "artifacts/preprocessed/condition_labels.csv"
    project_a_windows_path = roots.project_a / "artifacts/preprocessed/windows.csv"
    project_b_run = roots.project_b / "artifacts/relax_model/runs/relax-foundation-reve-large-20260705"
    project_b_labels_path = project_b_run / "preprocessed/condition_labels.csv"
    project_b_windows_path = project_b_run / "preprocessed/windows.csv"
    labels_a = pd.read_csv(project_a_labels_path)
    labels_b = pd.read_csv(project_b_labels_path)
    windows_a = pd.read_csv(project_a_windows_path)
    windows_b = pd.read_csv(project_b_windows_path)
    label_keys = ["participant_id", "condition"]
    window_keys = ["participant_id", "condition", "condition_window_index"]
    labels_a_sorted = labels_a.sort_values(label_keys).reset_index(drop=True)
    labels_b_sorted = labels_b.sort_values(label_keys).reset_index(drop=True)
    windows_a_sorted = windows_a.sort_values(window_keys).reset_index(drop=True)
    windows_b_sorted = windows_b.sort_values(window_keys).reset_index(drop=True)
    checks = [
        check_record(
            "labels_file_sha256_identity",
            sha256(project_a_labels_path) == sha256(project_b_labels_path),
            sha256(project_a_labels_path),
            sha256(project_b_labels_path),
            f"{project_a_labels_path} | {project_b_labels_path}",
        ),
        check_record(
            "windows_file_sha256_identity",
            sha256(project_a_windows_path) == sha256(project_b_windows_path),
            sha256(project_a_windows_path),
            sha256(project_b_windows_path),
            f"{project_a_windows_path} | {project_b_windows_path}",
        ),
        check_record("labels_row_count", len(labels_a) == len(labels_b) == 135, len(labels_a), 135, str(project_a_labels_path)),
        check_record("windows_row_count", len(windows_a) == len(windows_b) == 946, len(windows_a), 946, str(project_a_windows_path)),
        check_record(
            "project_a_label_duplicate_keys",
            not labels_a.duplicated(label_keys).any(),
            int(labels_a.duplicated(label_keys).sum()),
            0,
            str(project_a_labels_path),
        ),
        check_record(
            "project_b_label_duplicate_keys",
            not labels_b.duplicated(label_keys).any(),
            int(labels_b.duplicated(label_keys).sum()),
            0,
            str(project_b_labels_path),
        ),
        check_record(
            "project_a_window_duplicate_keys",
            not windows_a.duplicated(window_keys).any(),
            int(windows_a.duplicated(window_keys).sum()),
            0,
            str(project_a_windows_path),
        ),
        check_record(
            "project_b_window_duplicate_keys",
            not windows_b.duplicated(window_keys).any(),
            int(windows_b.duplicated(window_keys).sum()),
            0,
            str(project_b_windows_path),
        ),
        check_record(
            "label_key_identity",
            labels_a_sorted[label_keys].equals(labels_b_sorted[label_keys]),
            "identical" if labels_a_sorted[label_keys].equals(labels_b_sorted[label_keys]) else "different",
            "identical",
            "sorted participant_id+condition keys",
        ),
        check_record(
            "target_value_identity",
            np.allclose(
                labels_a_sorted[list(TARGETS)].to_numpy(dtype=float),
                labels_b_sorted[list(TARGETS)].to_numpy(dtype=float),
                atol=0.0,
                rtol=0.0,
            ),
            float(
                np.max(
                    np.abs(
                        labels_a_sorted[list(TARGETS)].to_numpy(dtype=float)
                        - labels_b_sorted[list(TARGETS)].to_numpy(dtype=float)
                    )
                )
            ),
            0.0,
            "sorted relaxation/discomfort arrays",
        ),
        check_record(
            "window_key_identity",
            windows_a_sorted[window_keys].equals(windows_b_sorted[window_keys]),
            "identical" if windows_a_sorted[window_keys].equals(windows_b_sorted[window_keys]) else "different",
            "identical",
            "sorted participant_id+condition+condition_window_index keys",
        ),
    ]
    return checks, labels_a_sorted, windows_a_sorted


def audit_cohorts_and_masks(
    roots: Roots,
    labels: pd.DataFrame,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    window_features_path = roots.project_a / "artifacts/features/window_features.csv"
    window_features = pd.read_csv(window_features_path)
    cohorts_path = roots.project_b / f"artifacts/relax/runs/{RUN_TAG}/cohorts.json"
    qc_path = roots.project_b / f"artifacts/relax/runs/{RUN_TAG}/qc_manifest.json"
    cache_path = roots.project_b / f"artifacts/relax/runs/{RUN_TAG}/condition_embeddings.pt"
    cohorts = json_load(cohorts_path)
    qc = json_load(qc_path)
    all_a = sorted(labels["participant_id"].astype(str).unique())
    all_b = sorted(cohorts["all_135"]["participants"])
    eeg_feature_columns = [
        column
        for column in window_features.columns
        if column.startswith(("eeg_tp9_", "eeg_tp10_")) and not column.startswith("qc_")
    ]
    has_eeg = window_features[eeg_feature_columns].notna().any(axis=1)
    eeg_a = sorted(window_features.loc[has_eeg, "participant_id"].astype(str).unique())
    eeg_b = sorted(cohorts["eeg_eligible"]["participants"])
    checks = [
        check_record("all_135_cohort_identity", all_a == all_b, ",".join(all_a), ",".join(all_b), str(cohorts_path), "cohort"),
        check_record("eeg_eligible_cohort_identity", eeg_a == eeg_b, ",".join(eeg_a), ",".join(eeg_b), str(cohorts_path), "cohort"),
        check_record("project_b_eeg_disabled_count", len(qc["eeg_disabled_participants"]) == 6, len(qc["eeg_disabled_participants"]), 6, str(qc_path), "cohort"),
    ]
    cohort_rows = []
    for source, cohort_name, participants in (
        ("project_a_derived", "all_135", all_a),
        ("project_b_manifest", "all_135", all_b),
        ("project_a_nonmissing_eeg", "eeg_eligible", eeg_a),
        ("project_b_manifest", "eeg_eligible", eeg_b),
        ("project_b_manifest", "foundation_complete", sorted(cohorts["foundation_complete"]["participants"])),
        ("project_b_manifest", "high_quality_pilot", list(cohorts["high_quality_pilot"]["participants"])),
    ):
        cohort_rows.append(
            {
                "source": source,
                "cohort": cohort_name,
                "participant_count": len(participants),
                "participant_ids": ",".join(participants),
            }
        )
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    mask_rows: list[dict[str, Any]] = []
    for modality in cache["modalities"]:
        valid = 0
        total = 0
        for sample in cache["samples"]:
            mask = torch.as_tensor(sample["masks"][modality], dtype=torch.bool)
            valid += int(mask.sum().item())
            total += int(mask.numel())
        mask_rows.append(
            {
                "source": "project_b_corrected_embedding_mask",
                "modality": modality,
                "total_windows": total,
                "valid_windows": valid,
                "invalid_windows": total - valid,
                "definition": "condition cache boolean mask",
            }
        )
    representative_columns = {
        "eeg": eeg_feature_columns,
        "ecg": [column for column in window_features.columns if column.startswith("ecg_") and not column.startswith("qc_")],
        "eye": [column for column in window_features.columns if column.startswith("eye_") and not column.startswith("qc_")],
        "head": [column for column in window_features.columns if column.startswith("head_") and not column.startswith("qc_")],
        "video": [column for column in window_features.columns if column.startswith("video_") and not column.startswith("qc_")],
    }
    qc_columns = {
        "eeg": "qc_eeg_usable",
        "ecg": "qc_ecg_usable",
        "eye": "qc_eye_usable",
        "head": "qc_head_usable",
        "video": "qc_video_usable",
    }
    for modality, columns in representative_columns.items():
        actual = window_features[columns].notna().any(axis=1) if columns else pd.Series(False, index=window_features.index)
        mask_rows.append(
            {
                "source": "project_a_feature_nonmissing",
                "modality": modality,
                "total_windows": len(window_features),
                "valid_windows": int(actual.sum()),
                "invalid_windows": int((~actual).sum()),
                "definition": f"any non-missing real {modality} feature ({len(columns)} columns)",
            }
        )
        qc_column = qc_columns[modality]
        if qc_column in window_features.columns:
            usable = pd.to_numeric(window_features[qc_column], errors="coerce").fillna(0.0) > 0.5
            mask_rows.append(
                {
                    "source": "project_a_qc_usable_flag",
                    "modality": modality,
                    "total_windows": len(window_features),
                    "valid_windows": int(usable.sum()),
                    "invalid_windows": int((~usable).sum()),
                    "definition": qc_column,
                }
            )
    mask_frame = pd.DataFrame(mask_rows)
    expected_b = {"eeg": 567, "ecg": 946, "eye": 945, "head": 945, "video": 945}
    for modality, expected in expected_b.items():
        observed = int(
            mask_frame.loc[
                (mask_frame["source"] == "project_b_corrected_embedding_mask")
                & (mask_frame["modality"] == modality),
                "valid_windows",
            ].iloc[0]
        )
        checks.append(
            check_record(
                f"project_b_{modality}_mask_count",
                observed == expected,
                observed,
                expected,
                str(cache_path),
                "mask",
            )
        )
    return checks, pd.DataFrame(cohort_rows), mask_frame


def recommended_splits(participants: list[str]) -> pd.DataFrame:
    ordered = sorted(participants)
    rows = []
    for index, test_participant in enumerate(ordered):
        validation_participant = ordered[(index + 1) % len(ordered)]
        for participant in ordered:
            role = (
                "test"
                if participant == test_participant
                else "validation"
                if participant == validation_participant
                else "train"
            )
            rows.append(
                {
                    "fold_index": index + 1,
                    "test_participant": test_participant,
                    "validation_participant": validation_participant,
                    "participant_id": participant,
                    "role": role,
                }
            )
    return pd.DataFrame(rows)


def verify_fold_payload(payload: dict[str, Any], cohort_participants: list[str]) -> tuple[int, int]:
    folds = payload.get("folds", [])
    ordered = sorted(cohort_participants)
    issues = 0
    checked = 0
    for index, fold in enumerate(folds):
        if "test_participant" not in fold:
            continue
        checked += 1
        expected_test = ordered[index]
        expected_validation = ordered[(index + 1) % len(ordered)]
        expected_train = sorted(set(ordered) - {expected_test, expected_validation})
        if str(fold.get("test_participant")) != expected_test:
            issues += 1
        if str(fold.get("val_participant")) != expected_validation:
            issues += 1
        if sorted(fold.get("train_participants", [])) != expected_train:
            issues += 1
    return checked, issues


def truth_mismatch_count(frame: pd.DataFrame, labels: pd.DataFrame) -> int:
    _, truth_columns, _ = normalize_prediction_frame(frame)
    lookup = labels[["participant_id", "condition", *TARGETS]].copy()
    merged = frame[["participant_id", "condition", *truth_columns.values()]].merge(
        lookup,
        on=["participant_id", "condition"],
        how="left",
        validate="many_to_one",
        suffixes=("_prediction", "_label"),
    )
    mismatch = np.zeros(len(merged), dtype=bool)
    for target in TARGETS:
        mismatch |= ~np.isclose(
            pd.to_numeric(merged[truth_columns[target]], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(merged[target], errors="coerce").to_numpy(dtype=float),
            # Neural prediction CSVs serialize truths from float32 tensors, while
            # condition_labels.csv stores the original float64 questionnaire values.
            atol=1e-6,
            rtol=0.0,
            equal_nan=False,
        )
    return int(mismatch.sum())


def audit_project_b_registry(
    roots: Roots,
    labels: pd.DataFrame,
    cohorts: dict[str, Any],
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    registry_path = roots.project_b / f"reports/relax_{RUN_TAG}/result_registry.csv"
    registry = pd.read_csv(registry_path)
    audit_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    participant_rows: list[pd.DataFrame] = []
    for _, row in registry.iterrows():
        result_path = roots.project_b / str(row["result_json"])
        prediction_path = roots.project_b / str(row["predictions_csv"])
        result_sha_match = result_path.is_file() and sha256(result_path) == str(row["result_sha256"])
        prediction_sha_match = prediction_path.is_file() and sha256(prediction_path) == str(row["predictions_sha256"])
        frame = pd.read_csv(prediction_path)
        unique_columns = ["participant_id", "condition"] + (["seed"] if "seed" in frame.columns else [])
        duplicate_count = int(frame.duplicated(unique_columns).sum())
        truth_mismatches = truth_mismatch_count(frame, labels)
        fold_mismatches = int((frame["fold"].astype(str) != frame["participant_id"].astype(str)).sum()) if "fold" in frame.columns else -1
        metadata = {
            "project": "project_b_corrected",
            "status": "current_corrected",
            "job_id": row["job_id"],
            "suite": row["suite"],
            "cohort": row["cohort"],
            "model": row["fusion"] if row["baseline"] == "none" else row["baseline"],
            "fusion": row["fusion"],
            "baseline": row["baseline"],
            "variant": row["variant"],
            "modalities": row["modalities"],
            "seed": int(row["seed"]),
            "split_protocol": "13_train+1_validation+1_test_LOPO",
            "prediction_path": str(prediction_path),
        }
        summary, targets, participant = score_configuration(frame, metadata)
        registry_macro = safe_float(row["macro_mae"])
        macro_delta = abs(summary["macro_mae"] - registry_macro)
        payload = json_load(result_path)
        cohort_participants = list(cohorts[str(row["cohort"])]["participants"])
        folds_checked, fold_payload_issues = verify_fold_payload(payload, cohort_participants)
        passed = all(
            (
                bool(row["complete"]),
                result_sha_match,
                prediction_sha_match,
                len(frame) == int(row["prediction_rows"]),
                duplicate_count == 0,
                truth_mismatches == 0,
                fold_mismatches in (-1, 0),
                macro_delta <= 5e-5 + 1e-6 * abs(registry_macro),
                fold_payload_issues == 0,
            )
        )
        audit_rows.append(
            {
                "job_id": row["job_id"],
                "suite": row["suite"],
                "cohort": row["cohort"],
                "variant": row["variant"],
                "baseline": row["baseline"],
                "expected_rows": int(row["prediction_rows"]),
                "actual_rows": len(frame),
                "duplicate_keys": duplicate_count,
                "truth_mismatches": truth_mismatches,
                "fold_column_mismatches": fold_mismatches,
                "folds_checked": folds_checked,
                "fold_payload_issues": fold_payload_issues,
                "result_sha_match": result_sha_match,
                "prediction_sha_match": prediction_sha_match,
                "registry_macro_mae": registry_macro,
                "rescored_macro_mae": summary["macro_mae"],
                "metric_abs_delta": macro_delta,
                "pass": passed,
                "prediction_path": str(prediction_path),
            }
        )
        summary_rows.append(summary)
        metric_rows.extend(targets)
        if (
            row["suite"] == "claim_validation"
            and row["cohort"] in {"all_135", "eeg_eligible"}
            and row["variant"] in {"full", "with_eeg", "without_eeg", "baseline"}
        ):
            participant_rows.append(participant)
    checks = [
        check_record("corrected_registry_result_count", len(registry) == 252, len(registry), 252, str(registry_path), "prediction"),
        check_record(
            "corrected_registry_all_outputs_pass",
            all(row["pass"] for row in audit_rows),
            sum(row["pass"] for row in audit_rows),
            len(audit_rows),
            str(registry_path),
            "prediction",
        ),
        check_record(
            "corrected_registry_prediction_rows_total",
            sum(row["actual_rows"] for row in audit_rows) == sum(row["expected_rows"] for row in audit_rows),
            sum(row["actual_rows"] for row in audit_rows),
            sum(row["expected_rows"] for row in audit_rows),
            str(registry_path),
            "prediction",
        ),
    ]
    participants = pd.concat(participant_rows, ignore_index=True) if participant_rows else pd.DataFrame()
    return checks, pd.DataFrame(audit_rows), pd.DataFrame(summary_rows), participants


def add_project_a_and_legacy_metrics(
    roots: Roots,
    labels: pd.DataFrame,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    participant_rows: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []

    classical_path = roots.project_a / "artifacts/reports/condition_level_lopo_predictions.csv"
    classical = pd.read_csv(classical_path)
    project_a_specs = [
        ("classical_residual_ensemble", "pred_relaxation", "pred_discomfort", "current"),
        ("condition_only", "condition_only_relaxation", "condition_only_discomfort", "baseline"),
        ("history", "history_relaxation", "history_discomfort", "baseline"),
    ]
    for model, relaxation_prediction, discomfort_prediction, variant in project_a_specs:
        frame = classical.copy()
        frame["relaxation_pred"] = frame[relaxation_prediction]
        frame["discomfort_pred"] = frame[discomfort_prediction]
        frame["relaxation_true"] = frame["relaxation"]
        frame["discomfort_true"] = frame["discomfort"]
        metadata = {
            "project": "project_a",
            "status": "current",
            "job_id": f"project_a/{model}",
            "suite": "project_a_current",
            "cohort": "all_135",
            "model": model,
            "fusion": "none",
            "baseline": model if variant == "baseline" else "none",
            "variant": variant,
            "modalities": "native_project_a",
            "seed": 20260621,
            "split_protocol": "14_non_test_LOPO_with_nested_selection",
            "prediction_path": str(classical_path),
        }
        summary, targets, participant = score_configuration(frame, metadata)
        summary_rows.append(summary)
        metric_rows.extend(targets)
        participant_rows.append(participant)
    duplicate_count = int(classical.duplicated(["participant_id", "condition"]).sum())
    truth_mismatches = truth_mismatch_count(
        classical.rename(
            columns={
                "relaxation": "relaxation_true",
                "discomfort": "discomfort_true",
                "pred_relaxation": "relaxation_pred",
                "pred_discomfort": "discomfort_pred",
            }
        ),
        labels,
    )
    audit_rows.append(
        {
            "job_id": "project_a/classical_oof",
            "expected_rows": 135,
            "actual_rows": len(classical),
            "duplicate_keys": duplicate_count,
            "truth_mismatches": truth_mismatches,
            "pass": len(classical) == 135 and duplicate_count == 0 and truth_mismatches == 0,
            "prediction_path": str(classical_path),
        }
    )

    dcnn_path = roots.project_a / "artifacts/reports/dcnn_condition_lopo_predictions.csv"
    dcnn = pd.read_csv(dcnn_path)
    for variant, frame in dcnn.groupby("model_variant", sort=False):
        normalized = frame.rename(
            columns={
                "relaxation": "relaxation_true",
                "discomfort": "discomfort_true",
                "pred_relaxation": "relaxation_pred",
                "pred_discomfort": "discomfort_pred",
            }
        )
        metadata = {
            "project": "project_a",
            "status": "current",
            "job_id": f"project_a/dcnn/{variant}",
            "suite": "project_a_current",
            "cohort": "all_135",
            "model": f"dcnn_{variant}",
            "fusion": "dcnn",
            "baseline": "none",
            "variant": variant,
            "modalities": "native_project_a",
            "seed": 20260621,
            "split_protocol": "project_a_native_LOPO",
            "prediction_path": str(dcnn_path),
        }
        summary, targets, participant = score_configuration(normalized, metadata)
        summary_rows.append(summary)
        metric_rows.extend(targets)
        participant_rows.append(participant)
        duplicates = int(frame.duplicated(["participant_id", "condition"]).sum())
        mismatches = truth_mismatch_count(normalized, labels)
        audit_rows.append(
            {
                "job_id": f"project_a/dcnn/{variant}",
                "expected_rows": 135,
                "actual_rows": len(frame),
                "duplicate_keys": duplicates,
                "truth_mismatches": mismatches,
                "pass": len(frame) == 135 and duplicates == 0 and mismatches == 0,
                "prediction_path": str(dcnn_path),
            }
        )

    legacy_path = roots.project_b / "logs/relax_foundation_probe/formal/fusions/all_135_healnet_original_sequence_predictions.csv"
    legacy = pd.read_csv(legacy_path)
    metadata = {
        "project": "project_b_legacy",
        "status": "superseded_invalid_eeg_map",
        "job_id": "project_b_legacy/healnet/all_135",
        "suite": "legacy_file_3",
        "cohort": "all_135",
        "model": "healnet",
        "fusion": "healnet",
        "baseline": "none",
        "variant": "legacy_full",
        "modalities": "eeg+ecg+eye+head+video",
        "seed": 20260705,
        "split_protocol": "13_train+1_validation+1_test_LOPO",
        "prediction_path": str(legacy_path),
    }
    summary, targets, participant = score_configuration(legacy, metadata)
    summary_rows.append(summary)
    metric_rows.extend(targets)
    participant_rows.append(participant)
    duplicates = int(legacy.duplicated(["participant_id", "condition"]).sum())
    mismatches = truth_mismatch_count(legacy, labels)
    audit_rows.append(
        {
            "job_id": "project_b_legacy/healnet/all_135",
            "expected_rows": 135,
            "actual_rows": len(legacy),
            "duplicate_keys": duplicates,
            "truth_mismatches": mismatches,
            "pass": len(legacy) == 135 and duplicates == 0 and mismatches == 0,
            "prediction_path": str(legacy_path),
        }
    )
    checks = [
        check_record(
            "project_a_and_legacy_primary_oof_coverage",
            all(row["pass"] for row in audit_rows),
            sum(row["pass"] for row in audit_rows),
            len(audit_rows),
            f"{classical_path} | {dcnn_path} | {legacy_path}",
            "prediction",
        )
    ]
    return (
        checks,
        pd.DataFrame(audit_rows),
        pd.DataFrame(summary_rows),
        pd.concat(participant_rows, ignore_index=True),
    )


def reproduce_baselines(
    roots: Roots,
    labels: pd.DataFrame,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    project_a_path = roots.project_a / "artifacts/reports/condition_level_lopo_predictions.csv"
    project_a = pd.read_csv(project_a_path)
    for target in TARGETS:
        expected_condition = np.empty(len(project_a), dtype=float)
        expected_history = np.empty(len(project_a), dtype=float)
        for participant, indexes in project_a.groupby("participant_id", sort=True).groups.items():
            indexes = np.asarray(list(indexes), dtype=int)
            train = labels.loc[labels["participant_id"] != participant]
            condition_means = train.groupby("condition")[target].mean()
            expected_condition[indexes] = project_a.loc[indexes, "condition"].map(condition_means).to_numpy(dtype=float)
            ordered = project_a.loc[indexes].sort_values("presentation_position")
            fallback = float(train[target].mean())
            history = np.r_[fallback, ordered[target].to_numpy(dtype=float)[:-1]]
            expected_history[ordered.index.to_numpy(dtype=int)] = history
        for baseline, expected, stored_column in (
            ("condition_only_14_fit", expected_condition, f"condition_only_{target}"),
            ("history_previous_label_with_train_fallback", expected_history, f"history_{target}"),
        ):
            stored = pd.to_numeric(project_a[stored_column], errors="coerce").to_numpy(dtype=float)
            delta = float(np.max(np.abs(stored - expected)))
            rows.append(
                {
                    "project": "project_a",
                    "baseline": baseline,
                    "target": target,
                    "rows": len(stored),
                    "max_abs_prediction_delta": delta,
                    "reproduced": delta <= 1e-12,
                    "evidence": str(project_a_path),
                }
            )

    project_b_condition_path = roots.project_b / f"logs/relax_{RUN_TAG}/foundation/base/baselines/all_135/condition/all_135_condition_predictions.csv"
    project_b_condition = pd.read_csv(project_b_condition_path)
    participants = sorted(labels["participant_id"].unique())
    for target in TARGETS:
        expected = np.empty(len(project_b_condition), dtype=float)
        for index, test_participant in enumerate(participants):
            validation_participant = participants[(index + 1) % len(participants)]
            train = labels.loc[~labels["participant_id"].isin([test_participant, validation_participant])]
            condition_means = train.groupby("condition")[target].mean()
            test_indexes = project_b_condition.index[project_b_condition["participant_id"] == test_participant]
            expected[test_indexes] = project_b_condition.loc[test_indexes, "condition"].map(condition_means).to_numpy(dtype=float)
        stored = pd.to_numeric(project_b_condition[f"{target}_pred"], errors="coerce").to_numpy(dtype=float)
        delta = float(np.max(np.abs(stored - expected)))
        rows.append(
            {
                "project": "project_b_corrected",
                "baseline": "condition_only_13_fit_1_validation",
                "target": target,
                "rows": len(stored),
                "max_abs_prediction_delta": delta,
                "reproduced": delta <= 1e-12,
                "evidence": str(project_b_condition_path),
            }
        )

    random_path = roots.project_b / f"logs/relax_{RUN_TAG}/foundation/base/baselines/all_135/random_9_condition/all_135_random_9_condition_predictions.csv"
    random_frame = pd.read_csv(random_path)
    for target in TARGETS:
        expected = np.empty(len(random_frame), dtype=float)
        for index, test_participant in enumerate(participants):
            validation_participant = participants[(index + 1) % len(participants)]
            train = labels.loc[~labels["participant_id"].isin([test_participant, validation_participant])]
            condition_means = train.groupby("condition")[target].mean()
            test_indexes = random_frame.index[random_frame["participant_id"] == test_participant]
            expected[test_indexes] = random_frame.loc[test_indexes, "chosen_condition"].map(condition_means).to_numpy(dtype=float)
        stored = pd.to_numeric(random_frame[f"{target}_pred"], errors="coerce").to_numpy(dtype=float)
        delta = float(np.max(np.abs(stored - expected)))
        rows.append(
            {
                "project": "project_b_corrected",
                "baseline": "random_9_condition_1000_seeds",
                "target": target,
                "rows": len(stored),
                "max_abs_prediction_delta": delta,
                "reproduced": delta <= 1e-12,
                "evidence": str(random_path),
            }
        )
    checks = [
        check_record(
            "all_named_baselines_reproduced",
            all(row["reproduced"] for row in rows),
            sum(row["reproduced"] for row in rows),
            len(rows),
            "condition/history/random baselines",
            "baseline",
        )
    ]
    return checks, pd.DataFrame(rows)


def plot_selected_metrics(summary: pd.DataFrame, participants: pd.DataFrame, output: Path) -> list[str]:
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    selections = [
        ("Project A Condition", (summary["project"] == "project_a") & (summary["model"] == "condition_only")),
        ("Project A history", (summary["project"] == "project_a") & (summary["model"] == "history")),
        ("Project A classical", (summary["project"] == "project_a") & (summary["model"] == "classical_residual_ensemble")),
        ("Project A DCNN full", (summary["project"] == "project_a") & (summary["model"] == "dcnn_full")),
        (
            "Project B Condition",
            (summary["project"] == "project_b_corrected")
            & (summary["suite"] == "claim_validation")
            & (summary["cohort"] == "all_135")
            & (summary["baseline"] == "condition"),
        ),
        (
            "Project B Ridge-CV",
            (summary["project"] == "project_b_corrected")
            & (summary["suite"] == "claim_validation")
            & (summary["cohort"] == "all_135")
            & (summary["baseline"] == "relax_handcrafted_ridge_cv"),
        ),
        (
            "Project B corrected Late",
            (summary["project"] == "project_b_corrected")
            & (summary["suite"] == "claim_validation")
            & (summary["cohort"] == "all_135")
            & (summary["variant"] == "full")
            & (summary["fusion"] == "late"),
        ),
        (
            "Project B corrected HealNet",
            (summary["project"] == "project_b_corrected")
            & (summary["suite"] == "claim_validation")
            & (summary["cohort"] == "all_135")
            & (summary["variant"] == "full")
            & (summary["fusion"] == "healnet"),
        ),
    ]
    labels = []
    means = []
    errors = []
    for label, mask in selections:
        values = pd.to_numeric(summary.loc[mask, "macro_mae"], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size:
            labels.append(label)
            means.append(float(values.mean()))
            errors.append(float(values.std(ddof=1)) if values.size > 1 else 0.0)
    fig, axis = plt.subplots(figsize=(9.0, 5.2))
    positions = np.arange(len(labels))
    axis.errorbar(means, positions, xerr=errors, fmt="o", color="#1f4e79", capsize=3)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Common-evaluator participant-macro MAE (lower is better)")
    axis.set_title("Descriptive OOF rescore; training protocols are not aligned")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = figure_dir / "common_oof_macro_mae.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    generated.append(str(path))

    project_a = participants.loc[
        (participants["project"] == "project_a")
        & (participants["model"] == "classical_residual_ensemble"),
        ["participant_id", "macro_mae"],
    ].rename(columns={"macro_mae": "project_a_macro_mae"})
    project_b = participants.loc[
        (participants["project"] == "project_b_corrected")
        & (participants["suite"] == "claim_validation")
        & (participants["cohort"] == "all_135")
        & (participants["variant"] == "full")
        & (participants["fusion"] == "late"),
        ["participant_id", "seed", "macro_mae"],
    ]
    project_b = project_b.groupby("participant_id", as_index=False)["macro_mae"].mean().rename(
        columns={"macro_mae": "project_b_late_seed_mean_macro_mae"}
    )
    paired = project_a.merge(project_b, on="participant_id", validate="one_to_one")
    paired.to_csv(output / "participant_primary_pairing.csv", index=False)
    fig, axis = plt.subplots(figsize=(5.8, 5.5))
    axis.scatter(
        paired["project_a_macro_mae"],
        paired["project_b_late_seed_mean_macro_mae"],
        color="#8f2d56",
    )
    lower = float(min(paired.iloc[:, 1:].min()))
    upper = float(max(paired.iloc[:, 1:].max()))
    axis.plot([lower, upper], [lower, upper], linestyle="--", color="0.5")
    for _, row in paired.iterrows():
        axis.annotate(
            row["participant_id"],
            (row["project_a_macro_mae"], row["project_b_late_seed_mean_macro_mae"]),
            fontsize=7,
            alpha=0.8,
        )
    axis.set_xlabel("Project A classical participant macro MAE")
    axis.set_ylabel("Project B corrected Late seed-mean macro MAE")
    axis.set_title("Descriptive pairing only: fits used different training splits")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    path = figure_dir / "participant_primary_pairing.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    generated.append(str(path))
    return generated


def parse_args() -> argparse.Namespace:
    script_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Read-only cross-project Relax alignment audit")
    parser.add_argument("--project-a-root", type=Path, default=script_root)
    parser.add_argument(
        "--project-b-root",
        type=Path,
        default=Path("/home/link/.config/superpowers/worktrees/real-time-vis-physio-fusion/relax-foundation-probe-v3"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_root / f"artifacts/cross_project_alignment_{AUDIT_DATE}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roots = Roots(args.project_a_root.resolve(), args.project_b_root.resolve(), args.output_dir.resolve())
    roots.output.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []

    provenance = source_provenance(roots)
    provenance.to_csv(roots.output / "source_provenance.csv", index=False)
    checks.append(
        check_record(
            "all_authoritative_sources_exist",
            bool(provenance["exists"].all()),
            int(provenance["exists"].sum()),
            len(provenance),
            str(roots.output / "source_provenance.csv"),
            "provenance",
        )
    )

    contract_checks, labels, _windows = audit_data_contract(roots)
    checks.extend(contract_checks)
    cohort_checks, cohort_frame, mask_frame = audit_cohorts_and_masks(roots, labels)
    checks.extend(cohort_checks)
    cohort_frame.to_csv(roots.output / "cohort_audit.csv", index=False)
    mask_frame.to_csv(roots.output / "mask_audit.csv", index=False)

    cohorts = json_load(roots.project_b / f"artifacts/relax/runs/{RUN_TAG}/cohorts.json")
    split_manifest = recommended_splits(list(cohorts["all_135"]["participants"]))
    split_manifest.to_csv(roots.output / "recommended_split_manifest.csv", index=False)

    registry_checks, registry_audit, project_b_summary, project_b_participants = audit_project_b_registry(
        roots, labels, cohorts
    )
    checks.extend(registry_checks)
    registry_audit.to_csv(roots.output / "project_b_registry_audit.csv", index=False)

    other_checks, other_audit, other_summary, other_participants = add_project_a_and_legacy_metrics(
        roots, labels
    )
    checks.extend(other_checks)
    other_audit.to_csv(roots.output / "project_a_legacy_prediction_audit.csv", index=False)

    baseline_checks, baseline_frame = reproduce_baselines(roots, labels)
    checks.extend(baseline_checks)
    baseline_frame.to_csv(roots.output / "baseline_reproduction.csv", index=False)

    summary = pd.concat([other_summary, project_b_summary], ignore_index=True, sort=False)
    summary.to_csv(roots.output / "common_oof_summary.csv", index=False)
    participants = pd.concat(
        [other_participants, project_b_participants], ignore_index=True, sort=False
    )
    participants.to_csv(roots.output / "participant_common_metrics.csv", index=False)
    figure_paths = plot_selected_metrics(summary, participants, roots.output)

    checks_frame = pd.DataFrame(checks)
    checks_frame.to_csv(roots.output / "verification_checks.csv", index=False)
    failed = checks_frame.loc[checks_frame["status"] != "PASS"]
    registry_failures = int((~registry_audit["pass"]).sum())
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "torch": torch.__version__,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_runtime": torch.version.cuda,
    }
    audit_summary = {
        "audit_date": AUDIT_DATE,
        "scope": "read-only artifact/report audit; no model training",
        "project_a_root": str(roots.project_a),
        "project_b_root": str(roots.project_b),
        "output_dir": str(roots.output),
        "overall_pass": failed.empty and registry_failures == 0,
        "check_count": len(checks_frame),
        "passed_checks": int((checks_frame["status"] == "PASS").sum()),
        "failed_checks": int((checks_frame["status"] != "PASS").sum()),
        "project_b_registry_results": len(registry_audit),
        "project_b_registry_failures": registry_failures,
        "project_b_prediction_rows": int(registry_audit["actual_rows"].sum()),
        "common_oof_configurations": len(summary),
        "source_count": len(provenance),
        "figure_paths": figure_paths,
        "environment": environment,
        "no_new_model_training": True,
    }
    write_json(roots.output / "audit_summary.json", audit_summary)
    print(json.dumps(audit_summary, ensure_ascii=False, indent=2))
    return 0 if audit_summary["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

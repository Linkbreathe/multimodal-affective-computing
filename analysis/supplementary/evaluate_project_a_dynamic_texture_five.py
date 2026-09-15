"""Validate and evaluate the formal Project A dynamic-texture five-modality matrix."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from real_time_ml.evaluation.alignment import file_sha256, validate_alignment_contract
from real_time_ml.evaluation.dynamic_texture_five import (
    TARGETS,
    assert_saved_metrics,
    paired_holm_families,
    seed_averaged_participant_errors,
    validate_normalized_oof,
)
from real_time_ml.experiments.dynamic_texture_five import project_b_style_metrics
from real_time_ml.utils import write_json


SEEDS = (20260705, 20260706, 20260707)
A_MODELS = ("classical", "1dcnn")
VARIANTS = ("full_five", "no_video")
PARTICIPANTS = ("P003", "P004", "P007", "P008", "P009", "P011", "P012", "P013", "P015")
DEFAULT_OUTPUT = Path(
    r"C:\Users\linki\amaster\foundation model result\91_Project_A_Dynamic_Texture_Five_Modality_Comparison_20260719"
)
DEFAULT_CONTRACT = (
    ROOT / "artifacts" / "cross_project_alignment_2026-07-16" / "eeg_eligible_ablation" / "contract"
)
DEFAULT_PROJECT_B = Path(
    r"\\?\UNC\wsl.localhost\Ubuntu\home\link\Wei\Models\core\real-time-vis-physio-fusion\artifacts\relax\foundation_compression_fusion_reinvestigation_20260718"
)
MODEL_LABELS = {
    "a_classical_full_five": "A Classical full-five",
    "a_classical_no_video": "A Classical no-video",
    "a_1dcnn_full_five": "A 1D-CNN full-five",
    "a_1dcnn_no_video": "A 1D-CNN no-video",
    "b_simplex5_full_five": "B Simplex5 full-five",
    "b_simplex5_no_video": "B Simplex5 no-video",
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _a_run_id(model: str, variant: str, seed: int) -> str:
    return f"project-a-dynamic-texture-{model}-{variant}-s{seed}"


def _model_key(project: str, model: str, variant: str) -> str:
    if project == "A":
        return f"a_{model}_{variant}"
    return f"b_simplex5_{variant}"


def _normalize_project_b(source: pd.DataFrame, *, seed: int, variant: str) -> pd.DataFrame:
    normalized_variant = "full_five" if variant == "full" else "no_video"
    frame = source.rename(
        columns={
            "relaxation_true": "true_relaxation",
            "relaxation_pred": "pred_relaxation",
            "discomfort_true": "true_discomfort",
            "discomfort_pred": "pred_discomfort",
        }
    ).copy()
    frame["project"] = "B"
    frame["model_family"] = "modality_expert_simplex5"
    frame["variant"] = normalized_variant
    frame["seed"] = int(seed)
    frame["split_protocol"] = "shared_7_train_1_validation_1_test"
    frame["device"] = frame["head_device"].astype(str)
    frame["cuda_used"] = False
    frame["embedding_cuda_used"] = frame["embedding_cuda_used"].astype(bool)
    frame["video_representation"] = "VideoMAE2-Base" if variant == "full" else "excluded"
    frame["p004_c6_condition_only_fallback"] = frame["participant_id"].astype(str).eq(
        "P004"
    ) & frame["condition"].astype(str).eq("C6")
    keep = [
        "participant_id",
        "condition",
        "presentation_position",
        "true_relaxation",
        "true_discomfort",
        "pred_relaxation",
        "condition_only_relaxation",
        "pred_discomfort",
        "condition_only_discomfort",
        "project",
        "model_family",
        "variant",
        "modalities",
        "seed",
        "fold_index",
        "test_participant",
        "validation_participant",
        "split_protocol",
        "device",
        "cuda_used",
        "embedding_cuda_used",
        "video_representation",
        "p004_c6_condition_only_fallback",
        "common_valid_windows",
        "source_windows",
        "quality",
        "presence",
        "neural_training_performed",
    ]
    return frame[keep]


def _flatten_metrics(frame: pd.DataFrame, run_id: str) -> list[dict[str, Any]]:
    metrics = project_b_style_metrics(frame)
    base = {
        "run_id": run_id,
        "project": str(frame["project"].iloc[0]),
        "model_family": str(frame["model_family"].iloc[0]),
        "model_key": str(frame["model_key"].iloc[0]),
        "variant": str(frame["variant"].iloc[0]),
        "seed": int(frame["seed"].iloc[0]),
    }
    rows = []
    for target in TARGETS:
        rows.append({**base, "outcome": target, **metrics["targets"][target]})
    rows.append(
        {
            **base,
            "outcome": "macro",
            "participant_macro_mae": metrics["macro_mae"],
            "rmse": np.nan,
            "spearman": np.nan,
            "pearson": np.nan,
            "ccc": np.nan,
        }
    )
    return rows


def _validate_project_a(
    *,
    output_root: Path,
    labels: pd.DataFrame,
    split_manifest: pd.DataFrame,
    expected_sources: dict[str, str],
) -> tuple[list[pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    gates: list[pd.DataFrame] = []
    for model in A_MODELS:
        for variant in VARIANTS:
            for seed in SEEDS:
                run_id = _a_run_id(model, variant, seed)
                run_root = output_root / "runs" / run_id
                paths = {
                    "prediction": run_root / "predictions" / "oof_predictions.csv",
                    "metrics": run_root / "metrics" / "metrics.json",
                    "manifest": run_root / "manifests" / "run_manifest.json",
                }
                if model == "1dcnn":
                    paths["gating"] = run_root / "manifests" / "gating_audit.csv"
                missing = [name for name, path in paths.items() if not path.is_file()]
                if missing:
                    raise ValueError(f"{run_id} is missing formal outputs: {missing}")
                manifest = _json(paths["manifest"])
                metrics = _json(paths["metrics"])
                if (
                    manifest["run_id"] != run_id
                    or manifest["model_family"] != model
                    or manifest["variant"] != variant
                    or int(manifest["seed"]) != seed
                ):
                    raise ValueError(f"{run_id} manifest identity mismatch")
                for source_name, expected_hash in expected_sources.items():
                    observed = manifest["sources"][source_name]["sha256"]
                    if observed != expected_hash:
                        raise ValueError(f"{run_id} used the wrong {source_name} hash")
                for output_name in ("predictions", "metrics"):
                    manifest_output = manifest["outputs"][output_name]
                    local_name = "prediction" if output_name == "predictions" else output_name
                    if file_sha256(paths[local_name]) != manifest_output["sha256"]:
                        raise ValueError(f"{run_id} {output_name} hash differs from its manifest")
                frame = pd.read_csv(paths["prediction"])
                frame["model_key"] = _model_key("A", model, variant)
                frame["run_id"] = run_id
                validate_normalized_oof(
                    frame,
                    labels=labels,
                    split_manifest=split_manifest,
                    expected_seed=seed,
                    expected_variant=variant,
                )
                recomputed = assert_saved_metrics(frame, metrics)
                cuda_used = bool(manifest["runtime"]["cuda_used"])
                if (model == "1dcnn") != cuda_used:
                    raise ValueError(f"{run_id} has an invalid CUDA declaration")
                gate_ok: bool | None = None
                maximum_video_gradient: float | None = None
                if model == "1dcnn":
                    gate = pd.read_csv(paths["gating"])
                    if len(gate) != 9:
                        raise ValueError(f"{run_id} does not have nine gating audits")
                    zero_columns = [
                        column
                        for column in gate.columns
                        if column.endswith("max_abs")
                        and (
                            "no_video" in column
                            or "embedding" in column
                            or "video_encoder_gradient" in column
                        )
                    ]
                    gate_ok = bool(
                        (gate[zero_columns].astype(float) == 0.0).all().all()
                        and gate["parameter_signature_equal"].astype(bool).all()
                        and gate["nonvideo_context_equal"].astype(bool).all()
                        and gate["p004_c6_full_gate_zero"].astype(bool).all()
                        and gate["trained_prediction_parameter_invariant_exact"].astype(bool).all()
                        and gate["initialization_state_sha256_full"].equals(
                            gate["initialization_state_sha256_no_video"]
                        )
                        and gate["nonvideo_input_sha256_full"].equals(
                            gate["nonvideo_input_sha256_no_video"]
                        )
                        and gate["parameter_count_full"].equals(gate["parameter_count_no_video"])
                        and gate["fusion_input_dim_full"].equals(gate["fusion_input_dim_no_video"])
                    )
                    if not gate_ok:
                        raise ValueError(f"{run_id} failed the formal gating audit")
                    fold_gradients = np.asarray(
                        [fold["maximum_video_gradient"] for fold in metrics["folds"]],
                        dtype=float,
                    )
                    if not all(bool(fold["cuda_used"]) for fold in metrics["folds"]):
                        raise ValueError(f"{run_id} contains a non-CUDA formal fold")
                    if variant == "full_five" and not (fold_gradients > 0.0).all():
                        raise ValueError(f"{run_id} lacks nonzero full-five Video gradients")
                    if variant == "no_video" and not (fold_gradients == 0.0).all():
                        raise ValueError(f"{run_id} has nonzero no-video gradients")
                    maximum_video_gradient = float(fold_gradients.max())
                    gate = gate.assign(
                        run_id=run_id,
                        model_key=frame["model_key"].iloc[0],
                        seed=seed,
                        variant=variant,
                    )
                    gates.append(gate)
                prediction_path = output_root / "predictions" / f"{run_id}.csv"
                prediction_path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(prediction_path, index=False)
                frames.append(frame)
                metric_rows.extend(_flatten_metrics(frame, run_id))
                audits.append(
                    {
                        "run_id": run_id,
                        "project": "A",
                        "model_family": model,
                        "variant": variant,
                        "seed": seed,
                        "rows": len(frame),
                        "folds": int(frame["fold_index"].nunique()),
                        "truth_fold_validation_ok": True,
                        "source_hashes_ok": True,
                        "saved_metrics_recomputed_ok": True,
                        "p004_c6_fallback_ok": True,
                        "cuda_used": cuda_used,
                        "gating_ok": gate_ok,
                        "maximum_video_gradient": maximum_video_gradient,
                        "prediction_sha256": file_sha256(paths["prediction"]),
                        "copied_prediction_sha256": file_sha256(prediction_path),
                        "macro_mae_recomputed": recomputed["macro_mae"],
                    }
                )
    gate_frame = pd.concat(gates, ignore_index=True)
    for seed in SEEDS:
        subset = gate_frame.loc[gate_frame["seed"].eq(seed)]
        full = subset.loc[subset["variant"].eq("full_five")].sort_values("fold_index")
        no_video = subset.loc[subset["variant"].eq("no_video")].sort_values("fold_index")
        for prefix in ("initialization_state_sha256_full", "initialization_state_sha256_no_video"):
            if full[prefix].tolist() != no_video[prefix].tolist():
                raise ValueError(f"Seed {seed} paired CNN initialization hashes differ")
    return frames, pd.DataFrame(audits), pd.DataFrame(metric_rows), gate_frame


def _validate_project_b(
    *,
    source_root: Path,
    output_root: Path,
    labels: pd.DataFrame,
    split_manifest: pd.DataFrame,
    expected_sources: dict[str, str],
) -> tuple[list[pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    evaluation_dir = source_root / "evaluation"
    evaluation_manifest = _json(evaluation_dir / "evaluation_manifest.json")
    formal_audit_path = evaluation_dir / "formal_run_audit.csv"
    expected_formal_hash = evaluation_manifest["outputs"]["formal_run_audit.csv"]["sha256"]
    if file_sha256(formal_audit_path) != expected_formal_hash:
        raise ValueError("Project B formal run audit differs from its evaluation manifest")
    formal_audit = pd.read_csv(formal_audit_path)
    formal_audit = formal_audit.loc[
        formal_audit["candidate"].astype(str).eq("modality_expert_simplex5")
        & formal_audit["variant"].astype(str).isin(("full", "no_video"))
    ].copy()
    if len(formal_audit) != 6 or not formal_audit["valid"].astype(bool).all():
        raise ValueError("Project B does not expose six validated Simplex5 source runs")

    reference_root = output_root / "references" / "project_b_verified"
    reference_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        evaluation_dir / "evaluation_manifest.json",
        reference_root / "source_evaluation_manifest.json",
    )
    shutil.copy2(formal_audit_path, reference_root / "source_formal_run_audit.csv")
    shutil.copy2(evaluation_dir / "final_report.md", reference_root / "source_final_report.md")
    frames: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    expected_fold_rows = split_manifest.loc[
        split_manifest["role"].astype(str).eq("test")
    ].set_index("test_participant")
    for variant in ("full", "no_video"):
        normalized_variant = "full_five" if variant == "full" else "no_video"
        expected_modalities = (
            ["eeg", "ecg", "eye", "head", "video"]
            if variant == "full"
            else ["eeg", "ecg", "eye", "head"]
        )
        for seed in SEEDS:
            source_dir = (
                source_root / "runs" / "modality_expert_simplex5" / variant / f"seed_{seed}"
            )
            prefix = f"modality_expert_simplex5_{variant}_s{seed}"
            prediction_path = source_dir / f"{prefix}_predictions.csv"
            result_path = source_dir / f"{prefix}_results.json"
            folds_path = source_dir / f"{prefix}_folds.json"
            result = _json(result_path)
            source_prediction_hash = file_sha256(prediction_path)
            audit_row = formal_audit.loc[
                formal_audit["variant"].astype(str).eq(variant)
                & pd.to_numeric(formal_audit["seed"]).astype(int).eq(seed)
            ].iloc[0]
            if (
                source_prediction_hash != result["prediction_sha256"]
                or source_prediction_hash != audit_row["prediction_sha256"]
                or file_sha256(result_path) != audit_row["result_sha256"]
            ):
                raise ValueError(f"Project B {prefix} fails prediction/result hash verification")
            if (
                result["candidate"] != "modality_expert_simplex5"
                or result["variant"] != variant
                or int(result["seed"]) != seed
                or result["modalities"] != expected_modalities
                or result["protocol"]
                != "fixed_9p_7_train_1_validation_1_test_81_labels_545_common_windows"
            ):
                raise ValueError(f"Project B {prefix} method declaration mismatch")
            if int(result["contract"]["common_valid_window_count"]) != 545 or result["contract"][
                "zero_common_window_observations"
            ] != [["P004", "C6"]]:
                raise ValueError(f"Project B {prefix} common mask declaration mismatch")
            if (
                result["head_runtime"]["neural_training_performed"]
                or result["head_runtime"]["device"] != "cpu"
                or not result["embedding_provenance"]["cuda_used"]
            ):
                raise ValueError(f"Project B {prefix} runtime provenance mismatch")
            input_mapping = {
                "labels": "labels",
                "windows": "windows",
                "split_manifest": "split_manifest",
                "mask_manifest": "masks",
            }
            for project_b_name, local_name in input_mapping.items():
                if (
                    result["contract"]["inputs"][project_b_name]["sha256"]
                    != expected_sources[local_name]
                ):
                    raise ValueError(f"Project B {prefix} used the wrong {project_b_name} hash")
            if len(result["folds"]) != 9:
                raise ValueError(f"Project B {prefix} lacks nine folds")
            for fold in result["folds"]:
                test_participant = str(fold["test_participant"])
                expected_fold = expected_fold_rows.loc[test_participant]
                expected_train = sorted(
                    split_manifest.loc[
                        split_manifest["fold_index"].eq(int(expected_fold["fold_index"]))
                        & split_manifest["role"].astype(str).eq("train"),
                        "participant_id",
                    ].astype(str)
                )
                if (
                    int(fold["fold_index"]) != int(expected_fold["fold_index"])
                    or str(fold["validation_participant"])
                    != str(expected_fold["validation_participant"])
                    or sorted(str(value) for value in fold["train_participants"]) != expected_train
                    or fold["modalities"] != expected_modalities
                ):
                    raise ValueError(f"Project B {prefix} fold metadata mismatch")
            source_frame = pd.read_csv(prediction_path)
            frame = _normalize_project_b(source_frame, seed=seed, variant=variant)
            run_id = f"project-b-simplex5-{normalized_variant}-s{seed}"
            frame["model_key"] = _model_key("B", "simplex5", normalized_variant)
            frame["run_id"] = run_id
            validate_normalized_oof(
                frame,
                labels=labels,
                split_manifest=split_manifest,
                expected_seed=seed,
                expected_variant=normalized_variant,
            )
            recomputed = assert_saved_metrics(frame, result)
            normalized_path = output_root / "predictions" / f"{run_id}.csv"
            normalized_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(normalized_path, index=False)
            run_reference = reference_root / run_id
            run_reference.mkdir(parents=True, exist_ok=True)
            shutil.copy2(prediction_path, run_reference / prediction_path.name)
            shutil.copy2(result_path, run_reference / result_path.name)
            shutil.copy2(folds_path, run_reference / folds_path.name)
            frames.append(frame)
            metric_rows.extend(_flatten_metrics(frame, run_id))
            audits.append(
                {
                    "run_id": run_id,
                    "project": "B",
                    "model_family": "modality_expert_simplex5",
                    "variant": normalized_variant,
                    "seed": seed,
                    "rows": len(frame),
                    "folds": int(frame["fold_index"].nunique()),
                    "truth_fold_validation_ok": True,
                    "source_hashes_ok": True,
                    "source_evaluation_audit_ok": True,
                    "saved_metrics_recomputed_ok": True,
                    "p004_c6_fallback_ok": True,
                    "neural_training_performed": False,
                    "head_device": "cpu",
                    "embedding_cuda_used": True,
                    "source_prediction_sha256": source_prediction_hash,
                    "normalized_prediction_sha256": file_sha256(normalized_path),
                    "source_result_sha256": file_sha256(result_path),
                    "macro_mae_recomputed": recomputed["macro_mae"],
                }
            )
    return frames, pd.DataFrame(audits), pd.DataFrame(metric_rows)


def _condition_leakage(
    condition_features: Path,
    split_manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(condition_features)
    video_columns = sorted(
        column
        for column in frame.columns
        if column.startswith("video_dynamic_") and "validity" not in column
    )
    if len(frame) != 81 or len(video_columns) != 108:
        raise ValueError("Condition leakage audit requires 81 rows and 108 video aggregates")
    participant_fold = split_manifest.loc[split_manifest["role"].astype(str).eq("test")].set_index(
        "test_participant"
    )[["fold_index", "validation_participant"]]
    classifier_rows: list[dict[str, Any]] = []
    classifier_predictions: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    class_targets = {
        "condition": ("condition", [f"C{index}" for index in range(1, 10)]),
        "intensity": ("intensity_index", [0, 1, 2]),
        "frequency": ("frequency_index", [0, 1, 2]),
    }
    for target_name, (target_column, class_order) in class_targets.items():
        truth_all: list[Any] = []
        prediction_all: list[Any] = []
        for participant in PARTICIPANTS:
            fold = participant_fold.loc[participant]
            fold_index = int(fold["fold_index"])
            train_ids = split_manifest.loc[
                split_manifest["fold_index"].eq(fold_index)
                & split_manifest["role"].astype(str).eq("train"),
                "participant_id",
            ].astype(str)
            train = frame["participant_id"].astype(str).isin(train_ids)
            test = frame["participant_id"].astype(str).eq(participant)
            estimator = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(C=0.1, class_weight="balanced", max_iter=5_000),
            )
            estimator.fit(frame.loc[train, video_columns], frame.loc[train, target_column])
            predicted = estimator.predict(frame.loc[test, video_columns])
            truth = frame.loc[test, target_column].to_numpy()
            truth_all.extend(truth.tolist())
            prediction_all.extend(predicted.tolist())
            for row, observed, estimate in zip(
                frame.loc[test].itertuples(index=False), truth, predicted, strict=True
            ):
                classifier_predictions.append(
                    {
                        "target": target_name,
                        "participant_id": row.participant_id,
                        "condition": row.condition,
                        "fold_index": fold_index,
                        "validation_participant": fold["validation_participant"],
                        "truth": observed,
                        "prediction": estimate,
                    }
                )
        balanced = balanced_accuracy_score(truth_all, prediction_all)
        macro_f1 = f1_score(truth_all, prediction_all, labels=class_order, average="macro")
        classifier_rows.append(
            {
                "target": target_name,
                "n_observations": len(truth_all),
                "n_participants": 9,
                "balanced_accuracy": balanced,
                "macro_f1": macro_f1,
                "chance_balanced_accuracy": 1.0 / len(class_order),
                "features": "108 fold-local-imputed dynamic_texture_v1 condition aggregates",
                "classifier": "StandardScaler+LogisticRegression(C=0.1,class_weight=balanced)",
            }
        )
        matrix = confusion_matrix(truth_all, prediction_all, labels=class_order)
        for truth_index, truth_label in enumerate(class_order):
            for prediction_index, prediction_label in enumerate(class_order):
                confusion_rows.append(
                    {
                        "target": target_name,
                        "true_class": truth_label,
                        "predicted_class": prediction_label,
                        "count": int(matrix[truth_index, prediction_index]),
                    }
                )

    ridge_predictions: list[dict[str, Any]] = []
    condition_one_hot = np.eye(9, dtype=float)[frame["condition_index"].astype(int).to_numpy() - 1]
    video_values = frame[video_columns].to_numpy(dtype=float)
    representations = {
        "condition_only": condition_one_hot,
        "video_only": video_values,
        "condition_plus_video": np.column_stack([condition_one_hot, video_values]),
    }
    for representation, values in representations.items():
        for participant in PARTICIPANTS:
            fold = participant_fold.loc[participant]
            fold_index = int(fold["fold_index"])
            train_ids = split_manifest.loc[
                split_manifest["fold_index"].eq(fold_index)
                & split_manifest["role"].astype(str).eq("train"),
                "participant_id",
            ].astype(str)
            train = frame["participant_id"].astype(str).isin(train_ids).to_numpy()
            test = frame["participant_id"].astype(str).eq(participant).to_numpy()
            for target in TARGETS:
                estimator = make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    Ridge(alpha=10.0),
                )
                estimator.fit(values[train], frame.loc[train, target])
                predicted = np.clip(estimator.predict(values[test]), 0.0, 1.0)
                for row, estimate in zip(
                    frame.loc[test].itertuples(index=False), predicted, strict=True
                ):
                    ridge_predictions.append(
                        {
                            "representation": representation,
                            "target": target,
                            "participant_id": row.participant_id,
                            "condition": row.condition,
                            "fold_index": fold_index,
                            "truth": getattr(row, target),
                            "prediction": estimate,
                        }
                    )
    ridge_frame = pd.DataFrame(ridge_predictions)
    ridge_rows: list[dict[str, Any]] = []
    for representation, group in ridge_frame.groupby("representation", sort=False):
        target_maes: dict[str, float] = {}
        for target in TARGETS:
            subset = group.loc[group["target"].eq(target)]
            participant_mae = (
                subset.assign(error=np.abs(subset["prediction"] - subset["truth"]))
                .groupby("participant_id")["error"]
                .mean()
            )
            target_maes[target] = float(participant_mae.mean())
            ridge_rows.append(
                {
                    "representation": representation,
                    "outcome": target,
                    "participant_macro_mae": target_maes[target],
                    "ridge_alpha": 10.0,
                    "n_participants": 9,
                    "n_observations": 81,
                }
            )
        ridge_rows.append(
            {
                "representation": representation,
                "outcome": "macro",
                "participant_macro_mae": float(np.mean(tuple(target_maes.values()))),
                "ridge_alpha": 10.0,
                "n_participants": 9,
                "n_observations": 81,
            }
        )
    return (
        pd.DataFrame(classifier_rows),
        pd.DataFrame(classifier_predictions),
        pd.DataFrame(confusion_rows),
        pd.DataFrame(ridge_rows),
    )


def _cluster_bootstrap(
    frame: pd.DataFrame,
    value_column: str,
    *,
    bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    participants = sorted(frame["participant_id"].astype(str).unique())
    values = {
        participant: frame.loc[
            frame["participant_id"].astype(str).eq(participant), value_column
        ].to_numpy(dtype=float)
        for participant in participants
    }
    draws = np.empty(bootstrap, dtype=float)
    for index in range(bootstrap):
        selected = rng.choice(participants, size=len(participants), replace=True)
        draws[index] = float(np.concatenate([values[item] for item in selected]).mean())
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _discomfort_strata(
    predictions: pd.DataFrame,
    *,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    observation = predictions.assign(
        absolute_error=np.abs(
            predictions["pred_discomfort"].to_numpy(dtype=float)
            - predictions["true_discomfort"].to_numpy(dtype=float)
        )
    )
    observation = (
        observation.groupby(
            [
                "model_key",
                "project",
                "model_family",
                "variant",
                "participant_id",
                "condition",
                "true_discomfort",
            ],
            as_index=False,
        )["absolute_error"]
        .mean()
        .rename(columns={"absolute_error": "seed_averaged_absolute_error"})
    )
    observation["stratum"] = np.where(observation["true_discomfort"].eq(0.0), "zero", "nonzero")
    unique_counts = (
        observation[["participant_id", "condition", "stratum"]]
        .drop_duplicates()
        .groupby("stratum")
        .agg(observations=("condition", "size"), participants=("participant_id", "nunique"))
    )
    if (
        int(unique_counts.loc["zero", "observations"]) != 65
        or int(unique_counts.loc["zero", "participants"]) != 9
        or int(unique_counts.loc["nonzero", "observations"]) != 16
        or int(unique_counts.loc["nonzero", "participants"]) != 6
    ):
        raise ValueError("Discomfort strata do not match the frozen 65/16 and 9/6 counts")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for (model_key, stratum), group in observation.groupby(["model_key", "stratum"], sort=True):
        low, high = _cluster_bootstrap(
            group,
            "seed_averaged_absolute_error",
            bootstrap=bootstrap,
            rng=rng,
        )
        rows.append(
            {
                "row_type": "stratified_mae",
                "model_key": model_key,
                "model_label": MODEL_LABELS[model_key],
                "variant": str(group["variant"].iloc[0]),
                "stratum": stratum,
                "estimate": float(group["seed_averaged_absolute_error"].mean()),
                "ci95_low": low,
                "ci95_high": high,
                "delta_definition": "not_applicable",
                "n_observations": int(len(group)),
                "n_participants": int(group["participant_id"].nunique()),
                "bootstrap_replicates": bootstrap,
                "formal_holm_family": False,
            }
        )
    for prefix in ("a_classical", "a_1dcnn", "b_simplex5"):
        full_key = f"{prefix}_full_five"
        no_video_key = f"{prefix}_no_video"
        left = observation.loc[
            observation["model_key"].eq(no_video_key),
            ["participant_id", "condition", "stratum", "seed_averaged_absolute_error"],
        ].rename(columns={"seed_averaged_absolute_error": "no_video_error"})
        right = observation.loc[
            observation["model_key"].eq(full_key),
            ["participant_id", "condition", "stratum", "seed_averaged_absolute_error"],
        ].rename(columns={"seed_averaged_absolute_error": "full_five_error"})
        paired = left.merge(
            right,
            on=["participant_id", "condition", "stratum"],
            validate="one_to_one",
        )
        paired["delta"] = paired["full_five_error"] - paired["no_video_error"]
        for stratum, group in paired.groupby("stratum", sort=True):
            low, high = _cluster_bootstrap(group, "delta", bootstrap=bootstrap, rng=rng)
            rows.append(
                {
                    "row_type": "video_delta",
                    "model_key": prefix,
                    "model_label": MODEL_LABELS[full_key].removesuffix(" full-five"),
                    "variant": "full_five_minus_no_video",
                    "stratum": stratum,
                    "estimate": float(group["delta"].mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "delta_definition": "full_five_minus_no_video_observation_mae",
                    "n_observations": int(len(group)),
                    "n_participants": int(group["participant_id"].nunique()),
                    "bootstrap_replicates": bootstrap,
                    "formal_holm_family": False,
                }
            )
    return pd.DataFrame(rows)


def _seed_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = ["participant_macro_mae", "rmse", "spearman", "pearson", "ccc"]
    summary = metrics.groupby(
        ["project", "model_family", "model_key", "variant", "outcome"], as_index=False
    )[metric_columns].agg(["mean", "std", "min", "max"])
    summary.columns = [
        "_".join(str(value) for value in column if value)
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    return summary


def _publication_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 7
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["legend.frameon"] = False


def _save_publication_figure(figure: Any, base_path: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    figure.tight_layout(pad=1.2)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix in ("svg", "pdf", "tiff", "png"):
        path = base_path.with_suffix(f".{suffix}")
        kwargs = {"bbox_inches": "tight"}
        if suffix in {"tiff", "png"}:
            kwargs["dpi"] = 600 if suffix == "tiff" else 300
        figure.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(figure)
    return outputs


def _forest_figure(frame: pd.DataFrame, output: Path, *, title: str, xlabel: str) -> list[Path]:
    import matplotlib.pyplot as plt

    plot = frame.copy()
    plot["label"] = plot.apply(
        lambda row: f"{row['outcome'].title()} | {row['comparison']}", axis=1
    )
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    y = np.arange(len(plot))[::-1]
    colors = {
        "relaxation": "#0F4D92",
        "discomfort": "#9A4D8E",
        "macro": "#4D4D4D",
    }
    for position, row in zip(y, plot.itertuples(index=False), strict=True):
        color = colors[row.outcome]
        axis.plot([row.ci95_low, row.ci95_high], [position, position], color=color, lw=1.4)
        axis.plot(row.mean_delta, position, marker="o", ms=4.0, color=color)
    axis.axvline(0.0, color="#767676", linestyle="--", lw=0.9)
    axis.set_yticks(y)
    axis.set_yticklabels(plot["label"])
    axis.set_xlabel(xlabel)
    axis.set_title(title, loc="left", weight="bold")
    axis.text(
        0.0,
        -0.16,
        "Dots: mean paired participant MAE delta; bars: 95% participant bootstrap CI; n=9.",
        transform=axis.transAxes,
        color="#606060",
    )
    return _save_publication_figure(figure, output)


def _make_figures(
    *,
    output_root: Path,
    metrics: pd.DataFrame,
    video_effects: pd.DataFrame,
    cross_model: pd.DataFrame,
    confusion: pd.DataFrame,
    discomfort: pd.DataFrame,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _publication_style()
    figure_paths: list[Path] = []
    figure_dir = output_root / "figures"

    macro = metrics.loc[metrics["outcome"].eq("macro")].copy()
    order = list(MODEL_LABELS)
    means = macro.groupby("model_key")["participant_macro_mae"].mean().reindex(order)
    stds = macro.groupby("model_key")["participant_macro_mae"].std().reindex(order)
    figure, axis = plt.subplots(figsize=(7.2, 3.8))
    colors = ["#B4C0E4", "#D8D8D8", "#E4CCD8", "#D8D8D8", "#AADCA9", "#D8D8D8"]
    x = np.arange(len(order))
    axis.bar(x, means, yerr=stds, color=colors, edgecolor="#4D4D4D", linewidth=0.7, capsize=2)
    for index, model_key in enumerate(order):
        points = macro.loc[macro["model_key"].eq(model_key), "participant_macro_mae"]
        axis.scatter(np.full(len(points), index), points, color="#272727", s=9, zorder=3)
    axis.set_xticks(x)
    axis.set_xticklabels([MODEL_LABELS[key] for key in order], rotation=24, ha="right")
    axis.set_ylabel("Macro participant-MAE")
    axis.set_title("Five-modality comparison", loc="left", weight="bold")
    axis.text(
        0.0,
        -0.31,
        "Bars: mean across three seeds; error bars: seed SD; dots: individual seeds. Lower is better.",
        transform=axis.transAxes,
        color="#606060",
    )
    figure_paths.extend(_save_publication_figure(figure, figure_dir / "mae_comparison"))

    figure_paths.extend(
        _forest_figure(
            video_effects,
            figure_dir / "video_effect_forest",
            title="Video effect within each model family",
            xlabel="Full-five − no-video participant MAE (negative favors video)",
        )
    )
    figure_paths.extend(
        _forest_figure(
            cross_model,
            figure_dir / "cross_model_forest",
            title="Paired full-five model differences",
            xlabel="Named first model − second model participant MAE",
        )
    )

    condition_confusion = confusion.loc[confusion["target"].eq("condition")].copy()
    labels = [f"C{index}" for index in range(1, 10)]
    matrix = (
        condition_confusion.pivot(index="true_class", columns="predicted_class", values="count")
        .reindex(index=labels, columns=labels)
        .to_numpy(dtype=float)
    )
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1.0)
    figure, axis = plt.subplots(figsize=(4.2, 3.8))
    image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    for row, column in np.ndindex(normalized.shape):
        axis.text(
            column,
            row,
            str(int(matrix[row, column])),
            ha="center",
            va="center",
            color="white" if normalized[row, column] > 0.55 else "#272727",
            fontsize=6,
        )
    axis.set_xticks(range(9), labels=labels)
    axis.set_yticks(range(9), labels=labels)
    axis.set_xlabel("Predicted condition")
    axis.set_ylabel("True condition")
    axis.set_title("Dynamic-texture condition decoding", loc="left", weight="bold")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Row-normalized recall")
    figure_paths.extend(_save_publication_figure(figure, figure_dir / "condition_confusion_matrix"))

    stratified = discomfort.loc[discomfort["row_type"].eq("stratified_mae")].copy()
    figure, axis = plt.subplots(figsize=(7.2, 4.0))
    width = 0.36
    x = np.arange(len(order))
    for offset, stratum, color in (
        (-width / 2, "zero", "#B4C0E4"),
        (width / 2, "nonzero", "#E4CCD8"),
    ):
        subset = (
            stratified.loc[stratified["stratum"].eq(stratum)].set_index("model_key").reindex(order)
        )
        values = subset["estimate"].to_numpy(dtype=float)
        lower = values - subset["ci95_low"].to_numpy(dtype=float)
        upper = subset["ci95_high"].to_numpy(dtype=float) - values
        axis.bar(
            x + offset,
            values,
            width,
            yerr=np.vstack([lower, upper]),
            label=f"{stratum} discomfort",
            color=color,
            edgecolor="#4D4D4D",
            linewidth=0.7,
            capsize=2,
        )
    axis.set_xticks(x)
    axis.set_xticklabels([MODEL_LABELS[key] for key in order], rotation=24, ha="right")
    axis.set_ylabel("Seed-averaged discomfort MAE")
    axis.set_title("Exploratory discomfort strata", loc="left", weight="bold")
    axis.legend(ncols=2)
    axis.text(
        0.0,
        -0.31,
        "Zero: 65 observations/9 participants; nonzero: 16/6. Error bars: participant-cluster 95% CI.",
        transform=axis.transAxes,
        color="#606060",
    )
    figure_paths.extend(_save_publication_figure(figure, figure_dir / "discomfort_stratified"))
    return figure_paths


def _markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
    subset = frame[columns].copy()
    for column in subset.columns:
        if pd.api.types.is_bool_dtype(subset[column]):
            subset[column] = subset[column].map(lambda value: "true" if value else "false")
        elif pd.api.types.is_numeric_dtype(subset[column]):
            subset[column] = subset[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.{digits}f}"
            )
        else:
            subset[column] = subset[column].astype(str).str.replace("|", "\\|", regex=False)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in subset.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _model_structure_lines() -> list[str]:
    """Describe the exact fitted structures behind the three comparison labels."""

    return [
        "## 三类模型的实际结构",
        "",
        (
            "这里的三个名称代表三条不同的建模管线，而不是三个可直接互换的单一网络。"
            "Project A Classical ML 在 condition 聚合特征上学习残差集成；Project A 1D-CNN 在 8 个窗口的"
            "五分支序列上端到端训练；Project B 则先离线提取冻结表示，再训练低容量 PCA/Ridge/Simplex 头。"
        ),
        "",
        "| 方法 | 监督输入 | 表示/编码器 | 融合与输出 | 本实验中的训练范围 |",
        "| --- | --- | --- | --- | --- |",
        (
            "| Project A Classical ML | 每个 participant-condition 一行聚合特征 | "
            "fold-local 缺失处理、去冗余、标准化、互信息选特征 | Condition-only anchor + "
            "top-3 residual regressors 等权平均 | 每折重新拟合全部预处理和回归器 |"
        ),
        (
            "| Project A 1D-CNN | 每个 participant-condition 的五模态 `[feature, 8 windows]` 序列 | "
            "五个独立逐特征 grouped-Conv1D 分支 | 拼接五个 embedding 与 2 个刺激 context，MLP 输出两个 target | "
            "每折在 CUDA 上端到端训练 |"
        ),
        (
            "| Project B foundation-feature model | 共同有效窗口上的冻结表示 | REVE、ECGFounder、"
            "InceptionTime、VideoMAEv2，加 18 维手工 Head 特征；随后 fold-local PCA | "
            "每模态/每 target 的 Ridge residual expert + target-specific Simplex 权重 | 原 Project B 只训练低容量头；"
            "本比较不重训，只验证并复用 OOF |"
        ),
        "",
        "### 1. Project A Classical ML：Condition-anchor 残差集成",
        "",
        "```text",
        "10 秒窗口特征 -> participant-condition 聚合 -> fold-local 特征处理",
        "                                      -> validation 选择 k 和 top-3 回归器",
        "Condition-only anchor ----------------> anchor + 平均残差 -> clip[0,1]",
        "```",
        "",
        (
            "输入不是原始波形或图像，而是 condition-level 工程特征。`no_video` 有 771 个候选；"
            "`full_five` 在完全相同的公共窗口和非视频特征上，再加入 108 个视频候选，共 879 个。"
            "这 108 个视频候选来自 12 个 `dynamic_texture_v1` 描述符，每个按 "
            "mean、std、min、max、median、first、last、slope、missing_ratio 九种统计聚合。"
        ),
        "",
        "每个 target、每个 outer fold 的实际处理顺序为：",
        "",
        "1. 只用 7 名 outer-training participants 计算每个 condition 的均值，作为 Condition-only anchor；监督目标改为 `真实值 − anchor`。",
        "2. 删除训练折非缺失率低于 40% 的列；训练折中位数填补并增加缺失指示列；删除零方差列。",
        "3. 按固定列序删除绝对相关系数达到 0.95 的后续冗余列，再做 `StandardScaler`。",
        "4. 用互信息 `SelectKBest` 选特征。先在唯一 validation participant 上仅比较 Ridge 的 `k=20/30/50/80`，确定 k。",
        "5. 固定该 k，在同一 validation participant 上比较六个候选家族，选择排名最好的三个。",
        "6. 用 7 名训练参与者重新拟合这三个 residual models；测试预测为三者残差的等权平均加回 anchor，并限制在 `[0,1]`。",
        "",
        "| 候选回归器 | 固定结构/超参数 |",
        "| --- | --- |",
        "| Ridge | `alpha=10` |",
        "| ElasticNet | `alpha=0.05, l1_ratio=0.5, max_iter=30000` |",
        "| SVR | RBF，`C=1, epsilon=0.05, gamma=scale` |",
        "| RandomForest | 120 trees，`max_features=0.6, min_samples_leaf=3` |",
        "| ExtraTrees | 120 trees，`max_features=0.6, min_samples_leaf=3` |",
        "| HistGradientBoosting | `learning_rate=0.06, max_leaf_nodes=7, l2=1, max_iter=80` |",
        "",
        (
            "Relaxation 的 validation 排名以 MAE 为主，并用 participant 内排序准确率作 0.02 权重的次级修正；"
            "Discomfort 直接按 MAE 排名。因此“Classical”不是某一个固定回归器，而是每折、每 target 均可能选择不同"
            " top-3 成员的固定规则残差集成。P004/C6 不接受残差修正，精确使用本折 Condition-only anchor。"
        ),
        "",
        "### 2. Project A 1D-CNN：五个独立的逐特征时序分支",
        "",
        "1D-CNN 接收工程特征的 8-window 序列，不接收原始 EEG/ECG 波形，也不接收 RGB tensor。每个分支输入形状为 `[batch, F_m, 8]`。",
        "",
        "```text",
        "每个模态 [B, F, 8]",
        "  -> grouped Conv1D(F -> 16F, kernel=3, padding=1, groups=F)",
        "  -> ReLU -> MaxPool(2) -> Dropout(0.30)                 # 时间 8 -> 4",
        "  -> grouped Conv1D(16F -> 32F, kernel=3, padding=1, groups=F)",
        "  -> ReLU -> MaxPool(2) -> Dropout(0.30)                 # 时间 4 -> 2",
        "  -> flatten                                             # 64F 维",
        "```",
        "",
        (
            "`groups=F` 表示不同原始特征流在卷积编码阶段不互相混合；每个特征各自学习时间滤波器。"
            "跨特征和跨模态交互只在所有分支 flatten 后的融合层发生。"
        ),
        "",
        "| 分支 | 输入流数 F | 分支输出维度 64F |",
        "| --- | ---: | ---: |",
        "| EEG | 33 | 2,112 |",
        "| ECG | 18 | 1,152 |",
        "| Eye | 11 | 704 |",
        "| Head | 18 | 1,152 |",
        "| Video | 13（12 个描述符 + 1 个 validity stream） | 832 |",
        "| 合计 | 93 | 5,952 |",
        "",
        (
            "五个 embedding 共 5,952 维，再拼接 fold-local 标准化的 intensity、frequency 两个 context，"
            "形成 5,954 维融合向量。融合头为 `Linear(5954,64) -> ReLU -> Dropout(0.30) -> "
            "Linear(64,2) -> Sigmoid`，同时输出归一化 relaxation 和 discomfort。每折总参数量为 533,026。"
        ),
        "",
        (
            "训练使用 AdamW（learning rate 0.001、weight decay 0.0001）、MSE、batch size 16、最多 40 epochs，"
            "按 validation MSE 早停（patience 8）。每个 epoch 随机截取 1 到实际长度之间的序列前缀作时序增强。"
            "所有特征可用性筛选、均值和标准差只在 7 名训练参与者上拟合；非有限值标准化后置零。"
        ),
        "",
        "Video 的消融不是删除网络层，而是在 Video encoder 之后、融合之前做显式门控：",
        "",
        "```text",
        "video_embedding = video_encoder(video_input)",
        "video_embedding = video_embedding * video_available[:, None]",
        "fused = concat(eeg, ecg, eye, head, video_embedding, intensity, frequency)",
        "```",
        "",
        (
            "`full_five` 与 `no_video` 实例化完全相同的 Video encoder 和融合头，并从同折同一个 initialization state 开始。"
            "`no_video` 的视频输入、validity 与 gate 全为 0；后编码门控连 Conv1D/Linear bias 产生的常数也一并归零，"
            "所以进入融合层的 video embedding 逐元素精确为 0，Video encoder 无有效梯度，改变其任意参数也不改变预测。"
            "P004/C6 在 `full_five` 中 gate 也为 0，最终两变体均精确回退到 Condition-only。"
        ),
        "",
        "### 3. Project B foundation model：冻结表示 + modality-expert Simplex5",
        "",
        (
            "这里更准确的名称是 **hybrid frozen-feature pipeline**，而不是一个端到端五模态 foundation network。"
            "EEG、ECG、Eye、Video 使用冻结神经表示；Head 是 18 个 Project A 手工窗口特征，不是 foundation encoder。"
        ),
        "",
        "| 模态 | 冻结窗口表示 | 原始维度 | full-five 的 PCA 分配 |",
        "| --- | --- | ---: | ---: |",
        "| EEG | `brain-bzh/reve-large`；输出自 1,216 adaptive-average-pool 到 1,024 | 1,024 | 3 |",
        "| ECG | `ECGFounder 1-lead` | 1,024 | 4 |",
        "| Eye | pretrained `InceptionTime` gaze encoder | 128 | 1 |",
        "| Head | shared-mask Project A handcrafted head features | 18 | 1 |",
        "| Video | `OpenGVLab/VideoMAEv2-Base` | 768 | 3 |",
        "| 合计 | 四个冻结神经表示 + 一个手工分支 | 2,962 | 12 |",
        "",
        (
            "冻结 embedding 在离线阶段用 CUDA 提取并缓存；正式 head 在 CPU 上拟合，神经编码器没有 fine-tune。"
            "每个 outer fold 只用 7 名训练参与者的共同有效窗口拟合每模态 PCA，先压缩窗口再池化为 condition score；"
            "full-five 的 12 维 rank allocation 为 EEG/ECG/Eye/Head/Video = 3/4/1/1/3。"
        ),
        "",
        "```text",
        "每模态冻结窗口表示 -> fold-local modality PCA -> condition score",
        "                                           -> 每 target 独立 Ridge residual expert",
        "五个 expert correction -> target-specific nonnegative Simplex weights",
        "Condition-only anchor + gamma * 加权修正 -> clip[0,1]",
        "```",
        "",
        (
            "训练折的 residual anchor 是同 condition 其余 6 名训练参与者的均值；validation/test anchor 是同 condition "
            "全部 7 名训练参与者的均值。对每个模态和每个 target 分别拟合 Ridge expert。Simplex 权重由 7 名 "
            "outer-training participants 内部的 participant-wise LOPO 预测学习，relaxation 与 discomfort 各有一组全局权重；"
            "每个活跃权重必须 `>=0.02` 且总和为 1，缺失专家按可用权重确定性重新归一化。"
        ),
        "",
        (
            "Ridge `alpha` 从 10/100/1000 中选择；正的修正系数 `gamma` 从 0.25/0.5/0.75/1.0 中选择。"
            "一个 alpha 和一个 gamma 由唯一 validation participant 的两 target Macro MAE 共同选择并在两个 target 间共享；"
            "每个 expert 修正先限制到 `[-0.2,0.2]`。该头的近似 label-conditioned capacity 是 43：约 34 个"
            " expert 系数/截距、8 个独立 Simplex 自由度和 1 个共享 gamma。"
        ),
        "",
        (
            "Project B 原实验的 `no_video` 会删除 VideoMAEv2 expert，并在相同 12 维总预算、fold 和 seed 下重新拟合"
            " PCA allocation、Ridge experts 与 Simplex 权重；本 Project A 比较只对保存的 full/no_video OOF 做哈希和"
            "协议验证后复用。P004/C6 没有任何共同有效窗口，因此修正严格为 0，只保留 Condition-only anchor。"
        ),
        "",
    ]


def _write_report(
    *,
    output_root: Path,
    seed_summary: pd.DataFrame,
    video_effects: pd.DataFrame,
    cross_model: pd.DataFrame,
    leakage: pd.DataFrame,
    ridge: pd.DataFrame,
    discomfort: pd.DataFrame,
) -> Path:
    full = seed_summary.loc[
        seed_summary["variant"].eq("full_five") & seed_summary["outcome"].eq("macro")
    ].copy()
    full["model"] = full["model_key"].map(MODEL_LABELS)
    full = full.sort_values("participant_macro_mae_mean")
    best = full.iloc[0]
    significant_video = video_effects.loc[video_effects["holm_significant_0_05"]]
    significant_cross = cross_model.loc[cross_model["holm_significant_0_05"]]
    if len(significant_cross) == 1:
        significant_row = significant_cross.iloc[0]
        significant_cross_statement = (
            f"唯一校正后显著的跨模型结果是 {significant_row['comparison']} 的 full-five "
            f"{significant_row['outcome'].title()} participant-MAE 差为 "
            f"{significant_row['mean_delta']:+.4f}（95% participant-bootstrap CI "
            f"{significant_row['ci95_low']:.4f}–{significant_row['ci95_high']:.4f}；"
            f"exact sign-flip p={significant_row['exact_sign_flip_p']:.4f}；"
            f"Holm p={significant_row['holm_p']:.4f}）。"
            "正值表示比较名称中的第一模型误差更高。"
        )
    else:
        significant_cross_statement = (
            f"共有 {len(significant_cross)} 项 full-five 跨模型差异通过 Holm 校正；"
            "具体方向与不确定性见下表。"
        )
    condition_row = leakage.loc[leakage["target"].eq("condition")].iloc[0]
    ridge_macro = ridge.loc[ridge["outcome"].eq("macro")].set_index("representation")
    condition_plus_improves = float(
        ridge_macro.loc["condition_plus_video", "participant_macro_mae"]
    ) < float(ridge_macro.loc["condition_only", "participant_macro_mae"])
    if (
        condition_row["balanced_accuracy"] > condition_row["chance_balanced_accuracy"]
        and not condition_plus_improves
    ):
        leakage_conclusion = (
            "动态纹理对刺激条件的被试留出分类点估计高于均匀机会水平，但统一 Ridge 的 Condition+Video "
            "没有降低 Condition-only 的 Macro MAE。因此视频结果应限定为主要编码刺激条件，"
            "不能解释为生理状态识别。"
        )
    else:
        leakage_conclusion = (
            "动态纹理的条件可识别性与标签预测控制结果并不支持把它单独解释为生理状态证据；"
            "任何视频效应均保留刺激条件混杂这一限制。"
        )
    formal_full = full[["model", "participant_macro_mae_mean", "participant_macro_mae_std"]]
    video_view = video_effects[
        [
            "comparison",
            "outcome",
            "mean_delta",
            "ci95_low",
            "ci95_high",
            "exact_sign_flip_p",
            "holm_p",
            "holm_significant_0_05",
        ]
    ]
    cross_view = cross_model[
        [
            "comparison",
            "outcome",
            "mean_delta",
            "ci95_low",
            "ci95_high",
            "exact_sign_flip_p",
            "holm_p",
            "holm_significant_0_05",
        ]
    ]
    leakage_view = leakage[["target", "balanced_accuracy", "macro_f1", "chance_balanced_accuracy"]]
    ridge_view = ridge[["representation", "outcome", "participant_macro_mae"]]
    strata_view = discomfort.loc[
        discomfort["row_type"].eq("video_delta"),
        ["model_label", "stratum", "estimate", "ci95_low", "ci95_high"],
    ]
    lines = [
        "# Project A 动态纹理五模态对齐实验",
        "",
        "## 结论摘要",
        "",
        (
            f"本报告完成了冻结边界内的 18 个正式运行：Project A 12 个新运行与 Project B "
            f"6 个经哈希和独立指标复算验证的既有运行，共 1,458 行 OOF 预测。三种子平均后，"
            f"full-five Macro participant-MAE 最低的是 {best['model']} "
            f"（{best['participant_macro_mae_mean']:.4f}；种子 SD {best['participant_macro_mae_std']:.4f}）。"
            "该排序是此 9 人冻结队列上的描述性结果，不外推为总体优越性。"
        ),
        "",
        (
            f"视频效应的九项检验中有 {len(significant_video)} 项通过各 outcome 内三项 Holm 校正；"
            f"full-five 跨模型九项检验中有 {len(significant_cross)} 项通过相同校正规则。"
            "负的视频增量代表加入视频后误差下降。点估计若未通过校正，只作描述。"
        ),
        "",
        significant_cross_statement,
        "",
        leakage_conclusion,
        "",
        "## 冻结数据与方法",
        "",
        "- 队列：P003、P004、P007、P008、P009、P011、P012、P013、P015。",
        "- 监督单位：81 个 participant-condition 标签；567 个源 10 秒窗口，其中 545 个五模态公共有效、22 个冻结屏蔽。",
        "- 固定划分：9 个 7/1/1 folds；种子 20260705–20260707。",
        "- 动态纹理：每个有效窗口 16 个互异最近帧；短边 224、中心裁剪 224、112 灰度分析；12 个连续描述符。",
        "- Classical：现有 residual ensemble 与 fold-local validation 选择规则；full-five 新增最多 108 个聚合候选。",
        "- 1D-CNN：五分支同构配对；Video encoder 后、融合前显式门控。full/no-video 参数量、参数名、shape、融合维度和同折初始化一致。",
        "- P004/C6：唯一零公共窗口 observation；所有模型与变体均精确回退到对应 fold 的 Condition-only 预测。",
        "- Project B：不重训；复用 modality_expert_simplex5 的 full/no_video 三种子 OOF，并核验结果哈希、预测哈希、truth、fold、validation participant、模态和重算指标。",
        "",
        *_model_structure_lines(),
        "## Full-five 主指标",
        "",
        "Macro MAE 是 relaxation 与 discomfort 的 participant-macro MAE 平均；下表为三种子均值与种子 SD。",
        "",
        _markdown_table(
            formal_full,
            ["model", "participant_macro_mae_mean", "participant_macro_mae_std"],
        ),
        "",
        "完整 MAE、RMSE、Spearman、Pearson 和 CCC 见 `../tables/all_model_metrics.csv` 与 `../tables/model_metric_seed_summary.csv`。",
        "",
        "## 视频效应：full-five − no-video",
        "",
        "推断先对每个 participant 的绝对误差在三个种子间平均，再形成 9 个 participant 配对值。CI 为 10,000 次 participant bootstrap；p 值为精确 2^9 sign-flip；relaxation、discomfort、Macro 各自是一组恰好三项的 Holm family。",
        "",
        _markdown_table(video_view, list(video_view.columns)),
        "",
        "## Full-five 跨模型配对",
        "",
        "差值按比较名称中的“第一模型 − 第二模型”定义；负值有利于第一模型。每个 outcome 独立构成三项 Holm family。",
        "",
        _markdown_table(cross_view, list(cross_view.columns)),
        "",
        "## Condition leakage 审计",
        "",
        "分类器与 Ridge 控制均复用相同 9 folds，训练仅使用每折 7 名训练参与者；缺失视频聚合只在训练折内中位数填补。分类器固定为低容量 LogisticRegression(C=0.1)，控制回归固定 Ridge(alpha=10)。",
        "",
        _markdown_table(leakage_view, list(leakage_view.columns)),
        "",
        _markdown_table(ridge_view, list(ridge_view.columns)),
        "",
        leakage_conclusion,
        "",
        "## Discomfort 零值/非零值分层（探索性）",
        "",
        "Zero 层为 65 个 observation、9 名参与者；Nonzero 层为 16 个 observation、6 名参与者。CI 按 participant 聚类 bootstrap；这些结果不进入主 Holm families。下表只列视频增量，所有模型/变体分层 MAE 见 `../tables/discomfort_strata.csv`。",
        "",
        _markdown_table(strata_view, list(strata_view.columns)),
        "",
        "## 门控、容量与运行验证",
        "",
        "- 1D-CNN full/no-video：每折 533,026 个参数、相同参数签名、融合输入维度 5,954、相同初始化 state hash。",
        "- no-video：视频输入、validity、availability、最终 video embedding、Video encoder 梯度均精确为 0；任意扰动 Video encoder 参数不改变预测。",
        "- full-five：所有正式 fold 的 Video 分支均产生非零训练梯度。",
        "- 正式 1D-CNN：所有 fold 均记录 CUDA；Project B 只复用既有 CPU head 预测，其基础 embedding 来源记录为 CUDA，未重训。",
        "- 所有 18 个运行的指标均从统一预测表独立复算；truth、fold、validation participant 与 P004/C6 fallback 一致。",
        "",
        "## 限制",
        "",
        "1. 只有 9 名参与者，精确 sign-flip 的分辨率有限；CI 与校正后 p 值必须和效应方向共同解读。",
        "2. Discomfort 极度零膨胀，Nonzero 层仅 16 个 observation/6 人，因此分层仅探索性。",
        "3. 动态纹理手工描述符与 Project B 的 VideoMAE2 表示并不等价；这里比较的是固定数据边界上的两套不同视频表示与模型管线。",
        "4. 视频中的条件可识别性可能来自刺激本身。即使视频提高标签预测，也不能据此声称识别了生理状态或因果放松机制。",
        "5. 本路径为 research-only；没有覆盖现有四模态 checkpoint 或 runtime 默认配置。",
        "",
        "## 复现命令",
        "",
        "```powershell",
        'C:\\Users\\linki\\miniconda3\\envs\\rtml-p002-p016\\python.exe -m real_time_ml.cli extract-dynamic-texture --contract-dir artifacts/cross_project_alignment_2026-07-16/eeg_eligible_ablation/contract --base-window-features artifacts/cross_project_alignment_2026-07-16/eeg_eligible_ablation/project_a/inputs/project_a_window_features_common.csv --output-dir "C:\\Users\\linki\\amaster\\foundation model result\\91_Project_A_Dynamic_Texture_Five_Modality_Comparison_20260719\\features"',
        "C:\\Users\\linki\\miniconda3\\envs\\rtml-p002-p016\\python.exe analysis/supplementary/run_project_a_dynamic_texture_five_matrix.py --resume",
        "C:\\Users\\linki\\miniconda3\\envs\\rtml-p002-p016\\python.exe analysis/supplementary/evaluate_project_a_dynamic_texture_five.py",
        "```",
        "",
        "## 产物导航",
        "",
        "- `../features/`：窗口描述符、逐帧选择审计、公共 mask 合并、condition 聚合。",
        "- `../configs/`：正式 run YAML 与 resolved config。",
        "- `../runs/`：Project A 训练记录、模型、指标和 OOF。",
        "- `../references/`：经验证的 Project B 原始结果与旧 Project A 审计引用。",
        "- `../predictions/`：18 个统一 run CSV 与 1,458 行合并表。",
        "- `../tables/`：正式主表、Holm family、泄漏控制和分层结果。",
        "- `../figures/`：SVG/PDF/TIFF/PNG 四格式科研图。",
        "- `../provenance/`：环境、CUDA、源哈希、初始化和门控验证。",
    ]
    report = output_root / "report" / "summary_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return report


def _figure_qa(paths: list[Path]) -> pd.DataFrame:
    from PIL import Image

    rows: list[dict[str, Any]] = []
    for path in paths:
        record: dict[str, Any] = {
            "path": str(path.resolve()),
            "format": path.suffix.lower().lstrip("."),
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else 0,
            "sha256": file_sha256(path) if path.is_file() else None,
            "editable_svg_text": None,
            "pdf_header_valid": None,
            "pdf_embedded_truetype_font": None,
            "width_pixels": None,
            "height_pixels": None,
            "dpi_x": None,
            "dpi_y": None,
        }
        if path.suffix.lower() == ".svg":
            text = path.read_text(encoding="utf-8")
            record["editable_svg_text"] = "<text" in text
            if not record["editable_svg_text"]:
                raise ValueError(f"{path.name} lacks editable SVG text")
        if path.suffix.lower() in {".png", ".tiff"}:
            with Image.open(path) as image:
                record["width_pixels"], record["height_pixels"] = image.size
                dpi = image.info.get("dpi")
                if dpi:
                    record["dpi_x"], record["dpi_y"] = float(dpi[0]), float(dpi[1])
                minimum_dpi = 590.0 if path.suffix.lower() == ".tiff" else 290.0
                if dpi and min(float(dpi[0]), float(dpi[1])) < minimum_dpi:
                    raise ValueError(f"{path.name} has insufficient raster resolution")
                image.verify()
        if path.suffix.lower() == ".pdf":
            payload = path.read_bytes()
            record["pdf_header_valid"] = payload.startswith(b"%PDF-")
            record["pdf_embedded_truetype_font"] = (
                b"/FontFile2" in payload and b"/CIDFontType2" in payload
            )
            if not (record["pdf_header_valid"] and record["pdf_embedded_truetype_font"]):
                raise ValueError(f"{path.name} lacks a valid header or embedded TrueType font")
        if not record["exists"] or record["bytes"] <= 0:
            raise ValueError(f"Figure export failed: {path}")
        rows.append(record)
    return pd.DataFrame(rows)


def _copy_old_project_a_references(output_root: Path) -> list[Path]:
    source = (
        ROOT
        / "artifacts"
        / "cross_project_alignment_2026-07-16"
        / "eeg_eligible_ablation"
        / "project_a"
        / "evaluation"
    )
    destination = output_root / "references" / "old_project_a_no_video_audit"
    destination.mkdir(parents=True, exist_ok=True)
    names = (
        "project_a_eeg_eligible_ablation_report.md",
        "model_and_baseline_summary.csv",
        "seed_metrics.csv",
        "provenance_validation.csv",
    )
    copied = []
    for name in names:
        path = source / name
        if path.is_file():
            target = destination / name
            shutil.copy2(path, target)
            copied.append(target)
    if not copied:
        raise ValueError("No old Project A no-video audit references were found")
    return copied


def _environment_lock() -> dict[str, Any]:
    packages = sorted(
        (
            {"name": distribution.metadata["Name"], "version": distribution.version}
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        ),
        key=lambda item: item["name"].lower(),
    )
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    for name in (
        "features",
        "configs",
        "runs",
        "references",
        "predictions",
        "tables",
        "figures",
        "provenance",
        "report",
    ):
        (output_root / name).mkdir(parents=True, exist_ok=True)
    contract = validate_alignment_contract(args.contract_dir)
    contract_json = _json(args.contract_dir / "contract.json")
    files = contract_json["files"]
    source_paths = {
        "contract": args.contract_dir / "contract.json",
        "split_manifest": args.contract_dir / files["split_manifest"]["path"],
        "masks": args.contract_dir / files["common_masks"]["path"],
        "labels": args.contract_dir / files["labels"]["path"],
        "windows": args.contract_dir / files["windows"]["path"],
        "window_features": args.window_features,
    }
    expected_sources = {name: file_sha256(path) for name, path in source_paths.items()}
    if tuple(contract["participants"]) != PARTICIPANTS or tuple(contract["seeds"]) != SEEDS:
        raise ValueError("Evaluation contract differs from the frozen cohort/seeds")
    feature_manifest = _json(output_root / "features" / "dynamic_texture_manifest.json")
    shutil.copy2(
        output_root / "features" / "dynamic_texture_config.json",
        output_root / "configs" / "dynamic_texture_feature_config.json",
    )
    assertions = {
        "source_windows": 567,
        "common_valid_windows": 545,
        "masked_windows": 22,
        "descriptor_count": 12,
        "participant_condition_labels": 81,
    }
    for name, expected in assertions.items():
        if int(feature_manifest[name]) != expected:
            raise ValueError(f"Feature assertion {name} != {expected}")
    if feature_manifest["zero_common_window_observations"] != [
        {"participant_id": "P004", "condition": "C6"}
    ]:
        raise ValueError("Feature manifest lacks the unique P004/C6 zero-window observation")
    if (
        expected_sources["window_features"]
        != feature_manifest["outputs"]["merged_window_features"]["sha256"]
    ):
        raise ValueError("Evaluation window-feature hash differs from extraction manifest")
    matrix_status_path = output_root / "provenance" / "project_a_matrix_status.json"
    matrix_status = _json(matrix_status_path)
    if (
        not matrix_status["ok"]
        or int(matrix_status["successful_runs"]) != 12
        or len(matrix_status["runs"]) != 12
    ):
        raise ValueError("Project A matrix status is not a complete 12-run success")

    labels = pd.read_csv(source_paths["labels"])
    split_manifest = pd.read_csv(source_paths["split_manifest"])
    a_frames, a_audit, a_metrics, gates = _validate_project_a(
        output_root=output_root,
        labels=labels,
        split_manifest=split_manifest,
        expected_sources=expected_sources,
    )
    b_frames, b_audit, b_metrics = _validate_project_b(
        source_root=args.project_b_root,
        output_root=output_root,
        labels=labels,
        split_manifest=split_manifest,
        expected_sources=expected_sources,
    )
    predictions = pd.concat([*a_frames, *b_frames], ignore_index=True, sort=False)
    if (
        len(predictions) != 1_458
        or predictions["run_id"].nunique() != 18
        or not (predictions.groupby("run_id").size() == 81).all()
    ):
        raise ValueError("Unified predictions must contain 18 x 81 = 1,458 rows")
    combined_path = output_root / "predictions" / "all_18_runs_oof_predictions.csv"
    predictions.to_csv(combined_path, index=False)

    run_audit = pd.concat([a_audit, b_audit], ignore_index=True, sort=False)
    metrics = pd.concat([a_metrics, b_metrics], ignore_index=True)
    seed_summary = _seed_summary(metrics)
    full_comparison = seed_summary.loc[seed_summary["variant"].eq("full_five")].copy()
    participant_errors = seed_averaged_participant_errors(predictions)
    video_effects = paired_holm_families(
        participant_errors,
        [
            ("a_classical_no_video", "a_classical_full_five", "A Classical full-five - no-video"),
            ("a_1dcnn_no_video", "a_1dcnn_full_five", "A 1D-CNN full-five - no-video"),
            ("b_simplex5_no_video", "b_simplex5_full_five", "B Simplex5 full-five - no-video"),
        ],
        family_prefix="video_effect",
        delta_definition="full_five_minus_no_video_seed_averaged_participant_mae",
        bootstrap=args.bootstrap,
        seed=args.bootstrap_seed,
    )
    cross_model = paired_holm_families(
        participant_errors,
        [
            ("a_1dcnn_full_five", "a_classical_full_five", "A Classical - A 1D-CNN"),
            ("b_simplex5_full_five", "a_classical_full_five", "A Classical - Project B"),
            ("b_simplex5_full_five", "a_1dcnn_full_five", "A 1D-CNN - Project B"),
        ],
        family_prefix="full_five_cross_model",
        delta_definition="named_first_minus_named_second_seed_averaged_participant_mae",
        bootstrap=args.bootstrap,
        seed=args.bootstrap_seed + 1,
    )
    holm = pd.concat(
        [video_effects.assign(analysis="video_effect"), cross_model.assign(analysis="cross_model")],
        ignore_index=True,
    )
    if len(holm) != 18 or not (holm.groupby("holm_family").size() == 3).all():
        raise ValueError("Formal Holm tables must contain six three-test families")

    leakage, leakage_predictions, confusion, ridge = _condition_leakage(
        args.condition_features, split_manifest
    )
    discomfort = _discomfort_strata(
        predictions, bootstrap=args.bootstrap, seed=args.bootstrap_seed + 2
    )

    tables = {
        "all_model_metrics.csv": metrics,
        "model_metric_seed_summary.csv": seed_summary,
        "full_five_comparison.csv": full_comparison,
        "video_effects.csv": video_effects,
        "cross_model_paired.csv": cross_model,
        "holm_families.csv": holm,
        "participant_seed_averaged_errors.csv": participant_errors,
        "condition_leakage_classification.csv": leakage,
        "condition_leakage_predictions.csv": leakage_predictions,
        "condition_confusion_matrices.csv": confusion,
        "condition_ridge_controls.csv": ridge,
        "discomfort_strata.csv": discomfort,
        "provenance_validation.csv": run_audit,
    }
    for name, frame in tables.items():
        frame.to_csv(output_root / "tables" / name, index=False)

    gates.to_csv(output_root / "provenance" / "gating_verification.csv", index=False)
    initialization = gates[
        [
            "run_id",
            "variant",
            "seed",
            "fold_index",
            "test_participant",
            "initialization_state_sha256_full",
            "initialization_state_sha256_no_video",
            "parameter_count_full",
            "parameter_count_no_video",
            "fusion_input_dim_full",
            "fusion_input_dim_no_video",
            "parameter_signature_equal",
            "nonvideo_input_sha256_full",
            "nonvideo_input_sha256_no_video",
        ]
    ]
    initialization.to_csv(
        output_root / "provenance" / "initialization_and_capacity.csv", index=False
    )
    source_hashes = pd.DataFrame(
        [
            {"source": name, "path": str(path.resolve()), "sha256": expected_sources[name]}
            for name, path in source_paths.items()
        ]
    )
    source_hashes.to_csv(output_root / "provenance" / "source_file_hashes.csv", index=False)
    write_json(output_root / "provenance" / "environment_lock.json", _environment_lock())
    import torch

    cuda = {
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": str(torch.version.cuda),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "formal_project_a_dcnn_all_cuda": bool(
            a_audit.loc[a_audit["model_family"].eq("1dcnn"), "cuda_used"].all()
        ),
    }
    if not cuda["formal_project_a_dcnn_all_cuda"]:
        raise ValueError("Not all formal Project A CNN runs used CUDA")
    write_json(output_root / "provenance" / "cuda_record.json", cuda)
    figure_contract = {
        "backend": "python_matplotlib_exclusive",
        "archetype": "quantitative_grid_exported_as_five_single_claim_figures",
        "core_conclusion": "Quantify model accuracy, paired video effects, cross-model uncertainty, stimulus-condition decodability, and discomfort sparsity without interpreting condition decoding as physiology.",
        "evidence_chain": {
            "mae_comparison": "three-seed full/no-video descriptive accuracy",
            "video_effect_forest": "participant-paired video ablation inference",
            "cross_model_forest": "participant-paired full-five model inference",
            "condition_confusion_matrix": "stimulus-condition decodability risk",
            "discomfort_stratified": "zero-inflation sensitivity",
        },
        "export": {
            "width_inches": 7.2,
            "formats": ["svg", "pdf", "tiff", "png"],
            "svg_editable_text": True,
            "pdf_fonttype": 42,
            "tiff_dpi": 600,
            "source_data": "tables/*.csv",
        },
        "statistics": {
            "n_participants": 9,
            "seeds": list(SEEDS),
            "folds": 9,
            "paired_ci": f"{args.bootstrap} participant bootstrap replicates",
            "test": "exact two-sided 2^9 sign-flip",
            "multiple_comparison": "Holm within each three-comparison outcome family",
        },
        "review_risks": [
            "nine-participant inference",
            "discomfort zero inflation",
            "stimulus-condition leakage",
            "dynamic_texture_v1 is not equivalent to VideoMAE2",
        ],
    }
    write_json(output_root / "provenance" / "figure_contract.json", figure_contract)
    figure_paths = _make_figures(
        output_root=output_root,
        metrics=metrics,
        video_effects=video_effects,
        cross_model=cross_model,
        confusion=confusion,
        discomfort=discomfort,
    )
    figure_qa = _figure_qa(figure_paths)
    figure_qa.to_csv(output_root / "provenance" / "figure_qa.csv", index=False)
    old_references = _copy_old_project_a_references(output_root)
    report = _write_report(
        output_root=output_root,
        seed_summary=seed_summary,
        video_effects=video_effects,
        cross_model=cross_model,
        leakage=leakage,
        ridge=ridge,
        discomfort=discomfort,
    )

    manifest_files = [
        combined_path,
        report,
        matrix_status_path,
        output_root / "features" / "dynamic_texture_manifest.json",
        *[output_root / "tables" / name for name in tables],
        *figure_paths,
        *old_references,
    ]
    manifest = {
        "schema_version": "project_a_dynamic_texture_five_evaluation_manifest_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "formal_project_a_runs": 12,
        "validated_reused_project_b_runs": 6,
        "formal_runs_total": 18,
        "formal_oof_rows": len(predictions),
        "participants": list(PARTICIPANTS),
        "seeds": list(SEEDS),
        "source_windows": 567,
        "common_valid_windows": 545,
        "masked_windows": 22,
        "zero_window_observation": ["P004", "C6"],
        "bootstrap_replicates": args.bootstrap,
        "exact_sign_flip_assignments": 2**9,
        "holm_families": 6,
        "tests_per_holm_family": 3,
        "discomfort_strata": {
            "zero": {"observations": 65, "participants": 9},
            "nonzero": {"observations": 16, "participants": 6},
        },
        "source_hashes": expected_sources,
        "code_sha256": {
            "dynamic_texture_extractor": file_sha256(
                ROOT / "src" / "real_time_ml" / "features" / "dynamic_texture.py"
            ),
            "project_a_models": file_sha256(
                ROOT / "src" / "real_time_ml" / "experiments" / "dynamic_texture_five.py"
            ),
            "paired_statistics": file_sha256(
                ROOT / "src" / "real_time_ml" / "evaluation" / "dynamic_texture_five.py"
            ),
            "matrix_entrypoint": file_sha256(
                ROOT / "analysis" / "supplementary" / "run_project_a_dynamic_texture_five_matrix.py"
            ),
            "evaluation_entrypoint": file_sha256(Path(__file__).resolve()),
        },
        "outputs": {
            str(path.relative_to(output_root)).replace("\\", "/"): {
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in manifest_files
        },
        "assertions": {
            "all_18_runs_valid": bool(len(run_audit) == 18),
            "all_metrics_independently_recomputed": bool(
                run_audit["saved_metrics_recomputed_ok"].all()
            ),
            "all_truth_folds_validation_aligned": bool(run_audit["truth_fold_validation_ok"].all()),
            "all_p004_c6_condition_only": bool(run_audit["p004_c6_fallback_ok"].all()),
            "project_a_dcnn_all_cuda": cuda["formal_project_a_dcnn_all_cuda"],
            "gating_exact": True,
            "condition_discomfort_counts_fixed": True,
        },
    }
    write_json(output_root / "provenance" / "evaluation_manifest.json", manifest)
    summary = {
        "output_root": str(output_root),
        "report": str(report),
        "formal_runs": 18,
        "formal_oof_rows": len(predictions),
        "project_a_runs": 12,
        "project_b_reused_runs": 6,
        "figure_exports": len(figure_paths),
        "holm_families": 6,
        "tests_per_family": 3,
        "ok": True,
    }
    write_json(output_root / "provenance" / "evaluation_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contract-dir", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--window-features",
        type=Path,
        default=DEFAULT_OUTPUT / "features" / "project_a_window_features_dynamic_texture.csv",
    )
    parser.add_argument(
        "--condition-features",
        type=Path,
        default=DEFAULT_OUTPUT / "features" / "condition_dynamic_texture_v1.csv",
    )
    parser.add_argument("--project-b-root", type=Path, default=DEFAULT_PROJECT_B)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260719)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    if arguments.bootstrap < 1_000:
        raise SystemExit("--bootstrap must be at least 1000")
    print(json.dumps(run(arguments), indent=2, ensure_ascii=False))

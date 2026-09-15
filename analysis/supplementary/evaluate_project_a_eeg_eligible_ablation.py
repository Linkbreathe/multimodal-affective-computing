"""Evaluate the formal Project A nine-participant modality-ablation matrix."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import pandas as pd

from evaluate_aligned_comparison import (
    TARGETS,
    _markdown_table,
    _normalize_columns,
    _paired_comparisons,
    _participant_metrics,
    _seed_metrics,
    _write_forest,
)
from mac.evaluation.alignment import load_split_manifest, validate_alignment_contract


MODELS = ("classical", "dcnn")
VARIANTS = ("full", "no_eeg", "no_ecg", "no_eye", "no_head")
SEEDS = (20260705, 20260706, 20260707)
PARTICIPANTS = ("P003", "P004", "P007", "P008", "P009", "P011", "P012", "P013", "P015")
PROJECT_A_MODALITIES = ("eeg", "ecg", "eye", "head")
B_ARCHITECTURES = ("late", "early", "mid", "qformer", "healnet", "mm_lego")
KEYS = ["participant_id", "condition"]


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_true(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "1.0"})


def _modalities(variant: str) -> tuple[str, ...]:
    if variant == "full":
        return PROJECT_A_MODALITIES
    removed = variant.removeprefix("no_")
    return tuple(modality for modality in PROJECT_A_MODALITIES if modality != removed)


def _belongs_to_modality(feature: str, modality: str) -> bool:
    if feature.startswith((f"{modality}_", f"qc_{modality}_", f"mask_{modality}_")):
        return True
    return modality in {"eeg", "ecg"} and feature.startswith("qc_physio_complete")


def _validate_saved_feature_mask(model_path: Path, model: str, variant: str) -> dict[str, Any]:
    if model == "classical":
        import joblib

        bundle = joblib.load(model_path)
    else:
        import torch

        try:
            bundle = torch.load(model_path, map_location="cpu", weights_only=True)
        except TypeError:
            bundle = torch.load(model_path, map_location="cpu")
    if str(bundle.get("model_variant")) != variant:
        raise ValueError(f"{model_path} stores the wrong model variant")
    columns = tuple(str(value) for value in bundle.get("feature_columns", ()))
    if not columns:
        raise ValueError(f"{model_path} does not store its actual feature columns")
    if any(name.startswith(("video_", "qc_video_", "mask_video_")) for name in columns):
        raise ValueError(f"{model_path} includes a Project B/video feature in Project A")
    retained = set(_modalities(variant))
    removed = set(PROJECT_A_MODALITIES) - retained
    for modality in removed:
        if any(_belongs_to_modality(name, modality) for name in columns):
            raise ValueError(f"{model_path} retains a removed {modality} feature")
    for modality in retained:
        if not any(name.startswith(f"{modality}_") for name in columns):
            raise ValueError(f"{model_path} lacks all retained {modality} features")
    observed = [
        modality
        for modality in PROJECT_A_MODALITIES
        if any(name.startswith(f"{modality}_") for name in columns)
    ]
    return {
        "saved_feature_mask_ok": True,
        "saved_feature_count": len(columns),
        "saved_feature_modalities": "+".join(observed),
    }


def _summary(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "mae",
        "participant_macro_mae",
        "rmse",
        "spearman",
        "pearson",
        "ccc",
        "participant_macro_spearman",
        "participant_macro_pairwise_accuracy",
    ]
    summary = seed_metrics.groupby(["model", "target"])[metric_columns].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(part for part in column if part) if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    model_order = {
        f"{model}_{variant}": model_index * len(VARIANTS) + variant_index
        for model_index, model in enumerate(MODELS)
        for variant_index, variant in enumerate(VARIANTS)
    }
    model_order.update({"condition_baseline": 100, "history_baseline": 101})
    summary["architecture"] = summary["model"].map(
        lambda value: value.split("_", 1)[0] if value not in {"condition_baseline", "history_baseline"} else "baseline"
    )
    summary["configuration"] = summary["model"].map(
        lambda value: next((variant for variant in VARIANTS if value.endswith(f"_{variant}")), value)
    )
    summary["modalities"] = summary["configuration"].map(
        lambda value: "+".join(_modalities(value)) if value in VARIANTS else "labels+folds"
    )
    summary["_order"] = summary["model"].map(model_order)
    return summary.sort_values(["_order", "target"]).drop(columns="_order").reset_index(drop=True)


def _fold_context(contract_dir: Path) -> tuple[pd.DataFrame, dict[str, tuple[int, str]], dict[int, dict[str, list[str]]]]:
    splits = pd.read_csv(contract_dir / "split_manifest.csv")
    folds = load_split_manifest(
        contract_dir / "split_manifest.csv",
        expected_participants=PARTICIPANTS,
        expected_train_count=7,
    )
    if len(folds) != 9 or len(splits) != 81:
        raise ValueError("The shared split manifest is not nine 7/1/1 folds")
    by_test = {
        fold.test_participant: (fold.fold_index, fold.validation_participant)
        for fold in folds
    }
    roles = {
        int(fold_index): {
            role: sorted(group.loc[group["role"] == role, "participant_id"].astype(str))
            for role in ("train", "validation", "test")
        }
        for fold_index, group in splits.groupby("fold_index", sort=True)
    }
    return splits, by_test, roles


def _recompute_baselines(
    labels: pd.DataFrame,
    roles: dict[int, dict[str, list[str]]],
    seed: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for fold_index in range(1, 10):
        fold_roles = roles[fold_index]
        train = labels.loc[labels["participant_id"].astype(str).isin(fold_roles["train"])].copy()
        test_participant = fold_roles["test"][0]
        validation_participant = fold_roles["validation"][0]
        test = labels.loc[labels["participant_id"].astype(str) == test_participant].copy()
        test["presentation_position"] = pd.to_numeric(test["presentation_position"], errors="raise")
        test = test.sort_values("presentation_position")
        output = test[[*KEYS, "presentation_position", *TARGETS]].copy()
        for target in TARGETS:
            train_values = pd.to_numeric(train[target], errors="raise")
            fallback = float(train_values.mean())
            condition_means = train.assign(**{target: train_values}).groupby("condition")[target].mean()
            output[f"condition_only_{target}"] = [
                float(condition_means.get(condition, fallback)) for condition in test["condition"]
            ]
            history: list[float] = []
            previous: float | None = None
            for value in pd.to_numeric(test[target], errors="raise"):
                history.append(fallback if previous is None else previous)
                previous = float(value)
            output[f"history_{target}"] = history
        output["fold_index"] = fold_index
        output["test_participant"] = test_participant
        output["validation_participant"] = validation_participant
        output["seed"] = seed
        rows.append(output)
    baseline = pd.concat(rows, ignore_index=True)
    if len(baseline) != 81 or baseline.duplicated(KEYS).any():
        raise ValueError("Independent baseline recomputation did not produce 81 unique predictions")
    return baseline


def _baseline_model_frames(baselines: dict[int, pd.DataFrame]) -> list[pd.DataFrame]:
    output: list[pd.DataFrame] = []
    for seed, baseline in baselines.items():
        for prefix, model in (("condition_only", "condition_baseline"), ("history", "history_baseline")):
            frame = baseline[[*KEYS, "presentation_position", "fold_index", "test_participant", "validation_participant", "seed", *TARGETS]].copy()
            for target in TARGETS:
                frame[f"{target}_true"] = pd.to_numeric(frame[target], errors="raise")
                frame[f"{target}_pred"] = pd.to_numeric(baseline[f"{prefix}_{target}"], errors="raise")
            frame["model"] = model
            frame["project"] = "shared_baseline"
            frame["model_variant"] = model
            frame["modalities"] = "labels+folds"
            frame["split_protocol"] = "shared_7_train_1_validation_1_test"
            frame["device"] = "cpu"
            frame["cuda_used"] = False
            frame["source_file"] = "independently_recomputed_from_contract"
            output.append(frame)
    return output


def _validate_a_prediction(
    frame: pd.DataFrame,
    *,
    path: Path,
    model: str,
    variant: str,
    seed: int,
    labels: pd.DataFrame,
    fold_by_test: dict[str, tuple[int, str]],
    baseline: pd.DataFrame,
) -> dict[str, Any]:
    normalized = _normalize_columns(frame)
    if len(normalized) != 81 or normalized.duplicated(KEYS).any():
        raise ValueError(f"{path} does not contain 81 unique OOF predictions")
    if set(normalized["participant_id"].astype(str)) != set(PARTICIPANTS):
        raise ValueError(f"{path} has the wrong participant cohort")
    merged = normalized.merge(
        labels[[*KEYS, "presentation_position", *TARGETS]],
        on=KEYS,
        how="outer",
        suffixes=("", "_contract"),
        validate="one_to_one",
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        raise ValueError(f"{path} keys differ from the contract labels")
    for target in TARGETS:
        if not np.allclose(
            merged[f"{target}_true"], merged[f"{target}_contract"], atol=1e-6, rtol=0.0
        ):
            raise ValueError(f"{path} {target} truths differ from the contract")
        prediction = pd.to_numeric(merged[f"{target}_pred"], errors="raise")
        if prediction.isna().any() or not prediction.between(0.0, 1.0).all():
            raise ValueError(f"{path} contains invalid {target} predictions")
    if set(pd.to_numeric(normalized["seed"], errors="raise").astype(int)) != {seed}:
        raise ValueError(f"{path} has the wrong seed")
    if set(normalized["model_variant"].astype(str)) != {variant}:
        raise ValueError(f"{path} has the wrong model variant")
    expected_modalities = "+".join(_modalities(variant))
    if set(normalized["modalities"].astype(str)) != {expected_modalities}:
        raise ValueError(f"{path} has the wrong modalities")
    if set(normalized["split_protocol"].astype(str)) != {"shared_7_train_1_validation_1_test"}:
        raise ValueError(f"{path} has the wrong split protocol")
    cuda = _as_true(normalized["cuda_used"])
    if model == "dcnn" and not cuda.all():
        raise ValueError(f"{path} contains non-CUDA DCNN predictions")
    if model == "classical" and cuda.any():
        raise ValueError(f"{path} unexpectedly records CUDA")
    for participant, group in normalized.groupby("participant_id"):
        fold_index, validation = fold_by_test[str(participant)]
        if set(pd.to_numeric(group["fold_index"], errors="raise").astype(int)) != {fold_index}:
            raise ValueError(f"{path} has the wrong fold for {participant}")
        if set(group["validation_participant"].astype(str)) != {validation}:
            raise ValueError(f"{path} has the wrong validation participant for {participant}")
        if set(group["test_participant"].astype(str)) != {str(participant)}:
            raise ValueError(f"{path} has the wrong test participant for {participant}")
    baseline_columns = [f"{prefix}_{target}" for prefix in ("condition_only", "history") for target in TARGETS]
    checked = normalized.merge(
        baseline[[*KEYS, *baseline_columns]], on=KEYS, validate="one_to_one", suffixes=("", "_recomputed")
    )
    for column in baseline_columns:
        if not np.allclose(
            pd.to_numeric(checked[column], errors="raise"),
            pd.to_numeric(checked[f"{column}_recomputed"], errors="raise"),
            atol=1e-6,
            rtol=0.0,
        ):
            raise ValueError(f"{path} {column} differs from the independent recomputation")
    return {
        "project": "A",
        "model_family": model,
        "variant": variant,
        "model": f"{model}_{variant}",
        "modalities": expected_modalities,
        "seed": seed,
        "rows": len(normalized),
        "participants": int(normalized["participant_id"].nunique()),
        "folds": int(normalized["fold_index"].nunique()),
        "prediction_coverage_ok": True,
        "fold_mapping_ok": True,
        "baseline_recomputation_ok": True,
        "cuda_policy_ok": True,
        "source_file": str(path.resolve()),
        "source_sha256": file_sha256(path),
    }


def _load_and_validate_a(
    runs_root: Path,
    labels: pd.DataFrame,
    fold_by_test: dict[str, tuple[int, str]],
    baselines: dict[int, pd.DataFrame],
) -> tuple[list[pd.DataFrame], list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    validation: list[dict[str, Any]] = []
    for model in MODELS:
        for variant in VARIANTS:
            for seed in SEEDS:
                run_id = f"eeg9-project-a-{model}-{variant}-s{seed}"
                filename = (
                    "condition_level_lopo_predictions.csv"
                    if model == "classical"
                    else "dcnn_condition_lopo_predictions.csv"
                )
                path = runs_root / run_id / "predictions" / filename
                if not path.is_file():
                    raise FileNotFoundError(f"Missing formal OOF prediction file: {path}")
                source = pd.read_csv(path)
                validation.append(
                    _validate_a_prediction(
                        source,
                        path=path,
                        model=model,
                        variant=variant,
                        seed=seed,
                        labels=labels,
                        fold_by_test=fold_by_test,
                        baseline=baselines[seed],
                    )
                )
                frame = _normalize_columns(source)
                frame["model"] = f"{model}_{variant}"
                frame["project"] = "A"
                frame["source_file"] = str(path.resolve())
                frames.append(frame)
    return frames, validation


def _validate_provenance(
    runs_root: Path,
    status: dict[str, Any],
    contract_dir: Path,
    feature_hash: str,
    fold_by_test: dict[str, tuple[int, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected_keys = {(model, variant, seed) for model in MODELS for variant in VARIANTS for seed in SEEDS}
    observed_keys = {
        (str(row["model"]), str(row["variant"]), int(row["seed"]))
        for row in status.get("runs", [])
        if row.get("status") in {"ok", "reused"}
    }
    if observed_keys != expected_keys:
        raise ValueError(
            f"Matrix status coverage mismatch: missing={sorted(expected_keys - observed_keys)}, "
            f"extra={sorted(observed_keys - expected_keys)}"
        )
    contract = _json(contract_dir / "contract.json")
    expected_sources = {
        "contract": file_sha256(contract_dir / "contract.json"),
        "labels": contract["files"]["labels"]["sha256"],
        "windows": contract["files"]["windows"]["sha256"],
        "split_manifest": contract["files"]["split_manifest"]["sha256"],
        "modality_masks": contract["files"]["common_masks"]["sha256"],
        "window_features": feature_hash,
    }
    provenance_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for model, variant, seed in sorted(expected_keys):
        run_id = f"eeg9-project-a-{model}-{variant}-s{seed}"
        run_root = runs_root / run_id
        manifest_name = "classical_alignment_manifest.json" if model == "classical" else "dcnn_alignment_manifest.json"
        metrics_name = "condition_level_lopo_metrics.json" if model == "classical" else "dcnn_condition_lopo_metrics.json"
        model_path = run_root / "models" / (
            "state_model.joblib" if model == "classical" else f"dcnn_state_{variant}.pt"
        )
        paths = {
            "alignment_manifest": run_root / "manifests" / manifest_name,
            "run_manifest": run_root / "manifests" / "run_manifest.json",
            "metrics": run_root / "metrics" / metrics_name,
            "model": model_path,
            "runner_log": run_root / "logs" / "runner.log",
        }
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            raise ValueError(f"{run_id} lacks provenance artifacts: {missing}")
        manifest = _json(paths["alignment_manifest"])
        for name, expected_hash in expected_sources.items():
            if manifest.get("sources", {}).get(name, {}).get("sha256") != expected_hash:
                raise ValueError(f"{run_id} has the wrong {name} hash")
        if int(manifest["seed"]) != seed or len(manifest.get("folds", [])) != 9:
            raise ValueError(f"{run_id} manifest has the wrong seed/fold count")
        runtime = manifest.get("runtime", {})
        if model == "dcnn" and not runtime.get("cuda_used"):
            raise ValueError(f"{run_id} manifest does not confirm CUDA")
        if model == "classical" and runtime.get("cuda_used"):
            raise ValueError(f"{run_id} classical manifest unexpectedly records CUDA")
        metrics = _json(paths["metrics"])
        feature_mask = _validate_saved_feature_mask(model_path, model, variant)
        fold_payload = metrics["folds"] if model == "classical" else metrics["variants"][variant]["folds"]
        if len(fold_payload) != 9:
            raise ValueError(f"{run_id} metrics do not contain nine folds")
        for fold in fold_payload:
            test = str(fold["test_participant"])
            expected_fold, expected_validation = fold_by_test[test]
            if (
                int(fold["fold"]) != expected_fold
                or str(fold["validation_participant"]) != expected_validation
                or int(fold["n_train_participants"]) != 7
                or int(fold["n_validation_participants"]) != 1
                or int(fold["n_test_participants"]) != 1
            ):
                raise ValueError(f"{run_id} fold metadata violate the shared 7/1/1 manifest")
            if model == "dcnn" and not bool(fold.get("cuda_used")):
                raise ValueError(f"{run_id} has a non-CUDA fold")
            fold_rows.append(
                {
                    "run_id": run_id,
                    "model_family": model,
                    "variant": variant,
                    "seed": seed,
                    "fold_index": int(fold["fold"]),
                    "test_participant": test,
                    "validation_participant": str(fold["validation_participant"]),
                    "train_participants": "+".join(fold["train_participants"]),
                    "n_train_participants": int(fold["n_train_participants"]),
                    "n_validation_participants": int(fold["n_validation_participants"]),
                    "n_test_participants": int(fold["n_test_participants"]),
                    "training_seed": fold.get("training_seed", seed),
                    "feature_count": fold.get("feature_count"),
                    "validation_loss": fold.get("validation_loss"),
                    "best_epoch": fold.get("best_epoch"),
                    "epochs_ran": fold.get("epochs_ran"),
                    "device": fold.get("device", "cpu"),
                    "cuda_used": bool(fold.get("cuda_used", False)),
                }
            )
        provenance_rows.append(
            {
                "run_id": run_id,
                "model_family": model,
                "variant": variant,
                "modalities": "+".join(_modalities(variant)),
                "seed": seed,
                "device": runtime.get("device", "cpu"),
                "cuda_used": bool(runtime.get("cuda_used", False)),
                "input_hashes_ok": True,
                "fold_metadata_ok": True,
                **feature_mask,
                **{f"{name}_path": str(path.resolve()) for name, path in paths.items()},
                **{f"{name}_sha256": file_sha256(path) for name, path in paths.items()},
            }
        )
    return pd.DataFrame(provenance_rows), pd.DataFrame(fold_rows)


def _ablation_tests(participant: pd.DataFrame, bootstrap: int, bootstrap_seed: int) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for model_index, model in enumerate(MODELS):
        pairs = [
            (f"{model}_full", f"{model}_{variant}", f"{model}_{variant}_vs_full")
            for variant in VARIANTS
            if variant != "full"
        ]
        family = _paired_comparisons(
            participant,
            pairs,
            bootstrap=bootstrap,
            seed=bootstrap_seed + model_index,
            expected_participants=9,
        )
        family.insert(0, "model_family", model)
        family.insert(1, "variant", family["right_model"].str.removeprefix(f"{model}_"))
        family.insert(2, "holm_family", f"{model}_four_removals_two_targets")
        outputs.append(family)
    return pd.concat(outputs, ignore_index=True)


def _baseline_tests(participant: pd.DataFrame, bootstrap: int, bootstrap_seed: int) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for model_index, model in enumerate(MODELS):
        pairs: list[tuple[str, str, str]] = []
        for variant in VARIANTS:
            right = f"{model}_{variant}"
            pairs.extend(
                [
                    ("condition_baseline", right, f"{right}_vs_condition_baseline"),
                    ("history_baseline", right, f"{right}_vs_history_baseline"),
                ]
            )
        family = _paired_comparisons(
            participant,
            pairs,
            bootstrap=bootstrap,
            seed=bootstrap_seed + 100 + model_index,
            expected_participants=9,
        )
        family.insert(0, "model_family", model)
        family.insert(1, "holm_family", f"{model}_five_configs_two_baselines_two_targets")
        outputs.append(family)
    return pd.concat(outputs, ignore_index=True)


def _largest_degradation(tests: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, target), group in tests.groupby(["model_family", "target"], sort=False):
        row = group.loc[group["mean_delta"].idxmax()]
        rows.append(
            {
                "model_family": model,
                "target": target,
                "variant": row["variant"],
                "removed_modality": str(row["variant"]).removeprefix("no_"),
                "mean_delta": row["mean_delta"],
                "ci95_low": row["ci95_low"],
                "ci95_high": row["ci95_high"],
                "exact_sign_flip_p": row["exact_sign_flip_p"],
                "holm_p": row["holm_p"],
                "holm_significant_0_05": row["holm_significant_0_05"],
            }
        )
    return pd.DataFrame(rows)


def _compare_with_existing_15(summary: pd.DataFrame, reference_path: Path) -> pd.DataFrame:
    reference = pd.read_csv(reference_path)
    mapping = {
        "classical_full": "a_classical_residual_ensemble",
        "dcnn_full": "a_dcnn_full",
        "dcnn_no_eeg": "a_dcnn_no_eeg",
    }
    rows: list[dict[str, Any]] = []
    for nine_model, fifteen_model in mapping.items():
        for target in TARGETS:
            nine = summary.loc[(summary["model"] == nine_model) & (summary["target"] == target)].iloc[0]
            fifteen = reference.loc[(reference["model"] == fifteen_model) & (reference["target"] == target)].iloc[0]
            delta = float(nine["participant_macro_mae_mean"] - fifteen["participant_macro_mae_mean"])
            rows.append(
                {
                    "nine_participant_model": nine_model,
                    "fifteen_participant_model": fifteen_model,
                    "target": target,
                    "nine_participant_mae_mean": float(nine["participant_macro_mae_mean"]),
                    "nine_participant_mae_std": float(nine["participant_macro_mae_std"]),
                    "fifteen_participant_mae_mean": float(fifteen["participant_macro_mae_mean"]),
                    "fifteen_participant_mae_std": float(fifteen["participant_macro_mae_std"]),
                    "nine_minus_fifteen_mae": delta,
                    "nine_point_estimate_lower": bool(delta < 0),
                    "directly_comparable": False,
                    "limitation": "different cohort, train/validation counts, and window-mask policy; descriptive direction only",
                }
            )
    return pd.DataFrame(rows)


def _load_project_b_no_video(
    evaluation_dir: Path,
    labels: pd.DataFrame,
    fold_by_test: dict[str, tuple[int, str]],
) -> list[pd.DataFrame]:
    audit = _json(evaluation_dir / "evaluation_audit.json")
    if not audit.get("ok") or int(audit.get("prediction_runs", -1)) != 108:
        raise ValueError("The existing Project B nine-participant evaluation audit is not valid")
    source = pd.read_csv(evaluation_dir / "normalized_predictions.csv")
    source = source.loc[source["configuration"].astype(str) == "no_video"].copy()
    frames: list[pd.DataFrame] = []
    for architecture in B_ARCHITECTURES:
        for seed in SEEDS:
            frame = source.loc[
                (source["architecture"].astype(str) == architecture)
                & (pd.to_numeric(source["seed"], errors="raise").astype(int) == seed)
            ].copy()
            if len(frame) != 81 or frame.duplicated(KEYS).any():
                raise ValueError(f"Project B {architecture}/no_video seed {seed} lacks 81 OOF rows")
            if set(frame["modalities"].astype(str)) != {"eeg+ecg+eye+head"}:
                raise ValueError(f"Project B {architecture}/no_video seed {seed} has wrong modalities")
            if not _as_true(frame["cuda_used"]).all():
                raise ValueError(f"Project B {architecture}/no_video seed {seed} lacks CUDA provenance")
            checked = frame.merge(
                labels[[*KEYS, *TARGETS]].rename(
                    columns={target: f"{target}_contract" for target in TARGETS}
                ),
                on=KEYS,
                validate="one_to_one",
            )
            for target in TARGETS:
                if not np.allclose(
                    pd.to_numeric(checked[f"{target}_true"], errors="raise"),
                    pd.to_numeric(checked[f"{target}_contract"], errors="raise"),
                    atol=1e-6,
                    rtol=0.0,
                ):
                    raise ValueError(f"Project B {architecture}/no_video truths differ from the contract")
            for participant, group in frame.groupby("participant_id"):
                expected_fold, expected_validation = fold_by_test[str(participant)]
                if set(pd.to_numeric(group["fold_index"], errors="raise").astype(int)) != {expected_fold}:
                    raise ValueError("Project B no-video fold mapping differs from the contract")
                if set(group["validation_participant"].astype(str)) != {expected_validation}:
                    raise ValueError("Project B no-video validation mapping differs from the contract")
            frame["model"] = f"b_{architecture}_no_video"
            frame["project"] = "B"
            frames.append(frame)
    return frames


def _project_b_comparison(
    a_frames: list[pd.DataFrame],
    b_frames: list[pd.DataFrame],
    bootstrap: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    a_full = [frame for frame in a_frames if str(frame["model"].iloc[0]) in {"classical_full", "dcnn_full"}]
    normalized = pd.concat([*a_full, *b_frames], ignore_index=True, sort=False)
    participant = _participant_metrics(normalized)
    seed = _seed_metrics(normalized, participant)
    summary = _summary(seed)
    outputs: list[pd.DataFrame] = []
    for model_index, a_model in enumerate(("classical_full", "dcnn_full")):
        pairs = [
            (a_model, f"b_{architecture}_no_video", f"b_{architecture}_no_video_vs_{a_model}")
            for architecture in B_ARCHITECTURES
        ]
        family = _paired_comparisons(
            participant,
            pairs,
            bootstrap=bootstrap,
            seed=bootstrap_seed + 200 + model_index,
            expected_participants=9,
        )
        family.insert(0, "project_a_reference", a_model)
        family.insert(1, "holm_family", f"six_project_b_architectures_two_targets_vs_{a_model}")
        outputs.append(family)
    tests = pd.concat(outputs, ignore_index=True)
    return normalized, participant, summary, tests


def _configuration_map() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        exact = variant == "full"
        rows.append(
            {
                "project_a_configuration": variant,
                "project_a_modalities": "+".join(_modalities(variant)),
                "exact_project_b_configuration": "no_video" if exact else "none",
                "project_b_modalities_if_exact": "eeg+ecg+eye+head" if exact else "none",
                "direct_cross_project_comparison_allowed": exact,
                "reason": (
                    "Project B no_video has the same four retained modalities"
                    if exact
                    else "Project B contains only single-removal configurations; matching this Project A removal would also require no_video"
                ),
            }
        )
    return pd.DataFrame(rows)


def _write_mae_figure(summary: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    subset = summary.loc[summary["architecture"].isin(MODELS)].copy()
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for axis, target in zip(axes, TARGETS, strict=True):
        target_rows = subset.loc[subset["target"] == target]
        labels = target_rows["model"].str.replace("_", "\n", regex=False)
        values = target_rows["participant_macro_mae_mean"].to_numpy(dtype=float)
        errors = target_rows["participant_macro_mae_std"].to_numpy(dtype=float)
        positions = np.arange(len(values))
        axis.bar(positions, values, yerr=errors, capsize=3, color=["#4472C4"] * 5 + ["#ED7D31"] * 5)
        axis.set_xticks(positions, labels, rotation=45, ha="right", fontsize=8)
        axis.set_ylabel("Participant-macro MAE")
        axis.set_title(target.capitalize())
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Project A nine-participant common-window benchmark (mean +/- SD over 3 seeds)")
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract = validate_alignment_contract(args.contract_dir)
    if contract.get("schema_version") != "eeg_eligible_modality_ablation_contract_v1":
        raise ValueError("The evaluator requires the frozen nine-participant contract")
    if tuple(contract.get("participants", ())) != PARTICIPANTS:
        raise ValueError("The contract participant cohort differs from the benchmark")
    labels = pd.read_csv(args.contract_dir / contract["files"]["labels"]["path"])
    if len(labels) != 81 or labels.duplicated(KEYS).any():
        raise ValueError("The contract does not contain 81 unique participant-condition labels")
    _, fold_by_test, roles = _fold_context(args.contract_dir)
    baselines = {seed: _recompute_baselines(labels, roles, seed) for seed in SEEDS}

    status_path = args.runs_root.parent / "eeg9_project_a_matrix_status.json"
    status = _json(status_path)
    if (
        not status.get("ok")
        or int(status.get("successful_runs", -1)) != 30
        or int(status.get("failed_runs", -1)) != 0
        or len(status.get("runs", [])) != 30
    ):
        raise ValueError("Formal Project A matrix status is not 30 successful / 0 failed")
    input_manifest = _json(args.inputs_dir / "project_a_input_manifest.json")
    feature_path = args.inputs_dir / "project_a_window_features_common.csv"
    feature_hash = file_sha256(feature_path)
    if input_manifest["outputs"]["window_features"]["sha256"] != feature_hash:
        raise ValueError("Project A input feature hash differs from its manifest")

    a_frames, validation_rows = _load_and_validate_a(
        args.runs_root, labels, fold_by_test, baselines
    )
    provenance, fold_metadata = _validate_provenance(
        args.runs_root, status, args.contract_dir, feature_hash, fold_by_test
    )
    model_predictions = pd.concat(a_frames, ignore_index=True, sort=False)
    all_predictions = pd.concat(
        [model_predictions, *_baseline_model_frames(baselines)], ignore_index=True, sort=False
    )
    all_predictions = all_predictions.sort_values(
        ["model", "seed", "participant_id", "presentation_position"]
    ).reset_index(drop=True)
    participant = _participant_metrics(all_predictions)
    seed_metrics = _seed_metrics(all_predictions, participant)
    summary = _summary(seed_metrics)
    ablation_tests = _ablation_tests(participant, args.bootstrap, args.bootstrap_seed)
    baseline_tests = _baseline_tests(participant, args.bootstrap, args.bootstrap_seed)
    largest = _largest_degradation(ablation_tests)

    existing_15 = _compare_with_existing_15(summary, args.project_a_15_summary)
    b_frames = _load_project_b_no_video(args.project_b_evaluation, labels, fold_by_test)
    b_normalized, b_participant, b_summary, b_tests = _project_b_comparison(
        a_frames, b_frames, args.bootstrap, args.bootstrap_seed
    )
    configuration_map = _configuration_map()

    run_validation = pd.DataFrame(validation_rows)
    run_validation.to_csv(args.output_dir / "run_validation.csv", index=False)
    provenance.to_csv(args.output_dir / "provenance_validation.csv", index=False)
    fold_metadata.to_csv(args.output_dir / "fold_metadata.csv", index=False)
    model_predictions.to_csv(args.output_dir / "complete_project_a_oof_predictions.csv", index=False)
    all_predictions.to_csv(args.output_dir / "normalized_predictions_with_baselines.csv", index=False)
    participant.to_csv(args.output_dir / "participant_metrics.csv", index=False)
    seed_metrics.to_csv(args.output_dir / "seed_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "model_and_baseline_summary.csv", index=False)
    ablation_tests.to_csv(args.output_dir / "ablation_vs_full_paired_tests.csv", index=False)
    baseline_tests.to_csv(args.output_dir / "model_vs_baseline_paired_tests.csv", index=False)
    largest.to_csv(args.output_dir / "largest_degradation_by_model_target.csv", index=False)
    existing_15.to_csv(args.output_dir / "project_a_9_vs_15_descriptive.csv", index=False)
    configuration_map.to_csv(args.output_dir / "project_a_project_b_configuration_map.csv", index=False)
    b_normalized.to_csv(args.output_dir / "project_a_project_b_same_modality_oof.csv", index=False)
    b_participant.to_csv(args.output_dir / "project_a_project_b_same_modality_participant_metrics.csv", index=False)
    b_summary.to_csv(args.output_dir / "project_a_project_b_same_modality_summary.csv", index=False)
    b_tests.to_csv(args.output_dir / "project_a_vs_project_b_no_video_paired_tests.csv", index=False)

    _write_mae_figure(summary, args.output_dir / "project_a_three_seed_mae.png")
    _write_forest(
        ablation_tests,
        args.output_dir / "project_a_ablation_delta_forest.png",
        title="Project A removal minus full participant MAE",
    )
    _write_forest(
        b_tests,
        args.output_dir / "project_a_vs_project_b_no_video_forest.png",
        title="Project B no-video minus Project A full participant MAE",
    )

    significant_degradations = ablation_tests.loc[
        (ablation_tests["mean_delta"] > 0) & ablation_tests["holm_significant_0_05"]
    ]
    significant_improvements = ablation_tests.loc[
        (ablation_tests["mean_delta"] < 0) & ablation_tests["holm_significant_0_05"]
    ]
    baseline_wins = baseline_tests.loc[
        (baseline_tests["mean_delta"] < 0) & baseline_tests["holm_significant_0_05"]
    ]
    lower_9 = existing_15.loc[existing_15["nine_point_estimate_lower"]]
    cross_project_significant = b_tests.loc[b_tests["holm_significant_0_05"]]
    if len(cross_project_significant):
        significant_b_models = ", ".join(
            str(value).removeprefix("b_").removesuffix("_no_video")
            for value in cross_project_significant["right_model"].unique()
        )
        if (
            set(cross_project_significant["project_a_reference"]) == {"dcnn_full"}
            and set(cross_project_significant["target"]) == {"discomfort"}
            and (cross_project_significant["mean_delta"] > 0).all()
        ):
            cross_project_statement = (
                "After Holm correction, Project A DCNN full has lower discomfort MAE than "
                f"Project B no-video {significant_b_models}. No corrected relaxation "
                "difference is supported, and Project B late no-video is not significantly "
                "different from Project A DCNN full."
            )
        else:
            cross_project_statement = (
                f"Holm-corrected differences involve: {significant_b_models}. See the signed "
                "deltas in the table for direction and target."
            )
    else:
        cross_project_statement = "No Holm-corrected same-modality Project A/Project B difference is supported."

    report_summary = summary[
        [
            "model",
            "architecture",
            "configuration",
            "modalities",
            "target",
            "participant_macro_mae_mean",
            "participant_macro_mae_std",
        ]
    ]
    report_ablation = ablation_tests[
        [
            "model_family",
            "variant",
            "target",
            "mean_delta",
            "ci95_low",
            "ci95_high",
            "exact_sign_flip_p",
            "holm_p",
            "holm_significant_0_05",
        ]
    ]
    report_baseline = baseline_tests[
        [
            "model_family",
            "left_model",
            "right_model",
            "target",
            "mean_delta",
            "ci95_low",
            "ci95_high",
            "exact_sign_flip_p",
            "holm_p",
            "holm_significant_0_05",
        ]
    ]
    report_b_summary = b_summary[
        ["model", "target", "participant_macro_mae_mean", "participant_macro_mae_std"]
    ]
    report_b_tests = b_tests[
        [
            "project_a_reference",
            "right_model",
            "target",
            "mean_delta",
            "ci95_low",
            "ci95_high",
            "exact_sign_flip_p",
            "holm_p",
            "holm_significant_0_05",
        ]
    ]
    report_path = args.output_dir / "project_a_eeg_eligible_ablation_report.md"
    report_lines = [
        "# Project A nine-participant EEG-eligible common-window benchmark",
        "",
        "## Execution facts",
        "",
        "- Formal runs: 30 successful, 0 failed (2 model families x 5 modality configurations x 3 seeds).",
        "- Cohort: P003, P004, P007, P008, P009, P011, P012, P013, P015.",
        "- Protocol: 9 fixed folds, each 7 train / 1 validation / 1 test.",
        "- Supervision: 81 participant-condition observations.",
        "- Window policy: 567 source windows, exactly 545 fixed five-way-common valid windows.",
        "- Classical residual ensemble ran on CPU. Every DCNN prediction, fold record, and run manifest records CUDA.",
        "- P004/C6 remains in OOF evaluation and is represented by missing inputs because it has zero common-valid windows.",
        "- This benchmark and all outputs below are separate from the existing 15-participant Project A artifacts.",
        "",
        "## Three-seed MAE",
        "",
        _markdown_table(report_summary),
        "",
        "## Modality removal versus corresponding full model",
        "",
        "Delta is removal minus full participant MAE; positive values are degradations. Each model family is one Holm family of four removals x two targets. Confidence intervals use 10,000 participant bootstraps; p-values use the exact 2^9 sign-flip distribution.",
        "",
        _markdown_table(report_ablation),
        "",
        "### Largest observed degradation",
        "",
        _markdown_table(largest),
        "",
        f"Holm-corrected removal results: {len(significant_degradations)} significant degradations and {len(significant_improvements)} significant improvements. Point-estimate directions without corrected significance are descriptive only.",
        "",
        "## Models versus independently recomputed baselines",
        "",
        "Delta is model minus baseline participant MAE; negative values favor the model. Baselines were independently recomputed from the same seven training participants and the held-out participant history in every fold, then checked against every run's saved baseline columns. Each model family is one Holm family of five configurations x two baselines x two targets.",
        "",
        _markdown_table(report_baseline),
        "",
        f"There are {len(baseline_wins)} Holm-corrected model improvements over a corresponding condition/history baseline.",
        "",
        "## Descriptive nine-participant versus existing 15-participant Project A",
        "",
        _markdown_table(existing_15),
        "",
        f"The nine-participant point estimate is lower in {len(lower_9)} of {len(existing_15)} available model-target comparisons. This is not evidence that restricting to nine participants improves the model: cohort membership, training-set size (7 versus 13), and window-mask policy differ, so these rows are descriptive and unpaired.",
        "",
        "## Direct Project A versus Project B comparison",
        "",
        "Only Project A full and Project B no_video retain the identical modality set EEG+ECG+eye+head. Other Project A removals would require a two-modality Project B removal (the named modality plus video), which the existing Project B matrix did not run.",
        "",
        _markdown_table(configuration_map),
        "",
        "### Same-modality three-seed MAE",
        "",
        _markdown_table(report_b_summary),
        "",
        "### Same-modality paired tests",
        "",
        "Delta is Project B no_video minus Project A full participant MAE; negative values favor Project B. Separate 12-test Holm families are used for the classical-full and DCNN-full Project A references.",
        "",
        _markdown_table(report_b_tests),
        "",
        f"There are {len(cross_project_significant)} Holm-corrected same-modality Project A/Project B differences. {cross_project_statement} The scope is limited to these nine participants, 545 common-valid windows, fixed folds, labels, metrics, and three seeds.",
        "",
        "## Reproduction commands",
        "",
        f"Working directory: `{ROOT.resolve()}`",
        "",
        "```powershell",
        "conda activate rtml-p002-p016",
        "python analysis/supplementary/build_project_a_eeg_eligible_inputs.py --contract-dir artifacts/cross_project_alignment_2026-07-16/eeg_eligible_ablation/contract --output-dir artifacts/cross_project_alignment_2026-07-16/eeg_eligible_ablation/project_a/inputs",
        "python analysis/supplementary/run_project_a_eeg_eligible_matrix.py --contract-dir artifacts/cross_project_alignment_2026-07-16/eeg_eligible_ablation/contract --inputs-dir artifacts/cross_project_alignment_2026-07-16/eeg_eligible_ablation/project_a/inputs --output-root artifacts/cross_project_alignment_2026-07-16/eeg_eligible_ablation/project_a/runs --config-root artifacts/cross_project_alignment_2026-07-16/eeg_eligible_ablation/project_a/configs --resume",
        "python analysis/supplementary/evaluate_project_a_eeg_eligible_ablation.py --contract-dir artifacts/cross_project_alignment_2026-07-16/eeg_eligible_ablation/contract --inputs-dir artifacts/cross_project_alignment_2026-07-16/eeg_eligible_ablation/project_a/inputs --runs-root artifacts/cross_project_alignment_2026-07-16/eeg_eligible_ablation/project_a/runs --output-dir artifacts/cross_project_alignment_2026-07-16/eeg_eligible_ablation/project_a/evaluation --bootstrap 10000 --bootstrap-seed 20260717",
        "```",
        "",
        "## Validation and artifact paths",
        "",
        f"The evaluator validated {len(run_validation)} OOF files ({len(model_predictions)} model predictions), {len(provenance)} run provenance bundles, and {len(fold_metadata)} fold records. It checked exact participant/condition coverage, contract truths, fold and validation mapping, seeds, modality strings, independently recomputed baselines, finite [0,1] predictions, input hashes, CPU/CUDA policy, and saved models/logs.",
        "",
        f"- Contract: `{args.contract_dir.resolve()}`",
        f"- Project A common-window inputs: `{args.inputs_dir.resolve()}`",
        f"- Formal run root: `{args.runs_root.resolve()}`",
        f"- Matrix status: `{status_path.resolve()}`",
        f"- Evaluation outputs: `{args.output_dir.resolve()}`",
        f"- Report: `{report_path.resolve()}`",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    audit = {
        "schema_version": "project_a_eeg_eligible_ablation_evaluation_v1",
        "ok": True,
        "contract_sha256": file_sha256(args.contract_dir / "contract.json"),
        "project_a_input_manifest_sha256": file_sha256(args.inputs_dir / "project_a_input_manifest.json"),
        "project_a_window_features_sha256": feature_hash,
        "matrix_status_sha256": file_sha256(status_path),
        "participants": list(PARTICIPANTS),
        "seeds": list(SEEDS),
        "model_families": list(MODELS),
        "variants": list(VARIANTS),
        "successful_runs": len(run_validation),
        "failed_runs": 0,
        "model_prediction_rows": len(model_predictions),
        "fold_metadata_rows": len(fold_metadata),
        "common_valid_windows": 545,
        "bootstrap": args.bootstrap,
        "exact_sign_flip_assignments": 2**9,
        "ablation_holm_families": 2,
        "ablation_tests_per_family": 8,
        "baseline_holm_families": 2,
        "baseline_tests_per_family": 20,
        "significant_ablation_degradations": len(significant_degradations),
        "significant_ablation_improvements": len(significant_improvements),
        "significant_model_vs_baseline_improvements": len(baseline_wins),
        "nine_vs_fifteen_descriptive_rows": len(existing_15),
        "project_b_same_modality_models": len(B_ARCHITECTURES),
        "project_b_same_modality_paired_tests": len(b_tests),
        "project_b_same_modality_significant_tests": len(cross_project_significant),
        "outputs": sorted(
            {path.name for path in args.output_dir.iterdir() if path.is_file()}
            | {"evaluation_audit.json"}
        ),
    }
    (args.output_dir / "evaluation_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    base = ROOT / "artifacts" / "cross_project_alignment_2026-07-16" / "eeg_eligible_ablation"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", type=Path, default=base / "contract")
    parser.add_argument("--inputs-dir", type=Path, default=base / "project_a" / "inputs")
    parser.add_argument("--runs-root", type=Path, default=base / "project_a" / "runs")
    parser.add_argument("--output-dir", type=Path, default=base / "project_a" / "evaluation")
    parser.add_argument(
        "--project-a-15-summary",
        type=Path,
        default=ROOT / "artifacts" / "cross_project_alignment_2026-07-16" / "evaluation" / "existing_aligned_reference_summary.csv",
    )
    parser.add_argument("--project-b-evaluation", type=Path, default=base / "evaluation")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260717)
    args = parser.parse_args(argv)
    if args.bootstrap < 1000:
        parser.error("--bootstrap must be at least 1000")
    for name in (
        "contract_dir",
        "inputs_dir",
        "runs_root",
        "output_dir",
        "project_a_15_summary",
        "project_b_evaluation",
    ):
        setattr(args, name, getattr(args, name).resolve())
    return args


def main(argv: list[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

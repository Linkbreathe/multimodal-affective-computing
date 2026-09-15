"""Validate and evaluate all cross-project aligned predictions with one code path."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from hashlib import sha256
import itertools
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from real_time_ml.evaluation.alignment import load_split_manifest, validate_alignment_contract


TARGETS = ("relaxation", "discomfort")
SEEDS = (20260705, 20260706, 20260707)
NEW_FUSIONS = ("early", "mid", "qformer", "healnet", "mm_lego")
NEW_ALIGNED_MODELS = tuple(
    f"b_corrected_{fusion}_{mask}"
    for fusion in NEW_FUSIONS
    for mask in ("full", "no_eeg")
)
EXPECTED_MODELS = (
    "a_classical_residual_ensemble",
    "a_dcnn_full",
    "a_dcnn_no_eeg",
    "b_corrected_ridge_cv",
    "b_corrected_late_full",
    "b_corrected_late_no_eeg",
    *NEW_ALIGNED_MODELS,
)
EXISTING_ALIGNED_MODELS = EXPECTED_MODELS[:6]
NEURAL_MODELS = set(EXPECTED_MODELS) - {
    "a_classical_residual_ensemble",
    "b_corrected_ridge_cv",
}
B_VARIANT_TO_MODEL = {
    "ridge_cv": "b_corrected_ridge_cv",
    "late_full": "b_corrected_late_full",
    "late_no_eeg": "b_corrected_late_no_eeg",
    **{
        f"{fusion}_{mask}": f"b_corrected_{fusion}_{mask}"
        for fusion in NEW_FUSIONS
        for mask in ("full", "no_eeg")
    },
}
B_RESULT_TO_MODEL = {
    "corrected_ridge_cv": "b_corrected_ridge_cv",
    **{
        f"corrected_{variant}": model
        for variant, model in B_VARIANT_TO_MODEL.items()
        if variant != "ridge_cv"
    },
}
FULL_MODALITIES = ("eeg", "ecg", "eye", "head", "video")
NO_EEG_MODALITIES = ("ecg", "eye", "head", "video")
KEYS = ["participant_id", "condition"]


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _markdown_table(frame: pd.DataFrame) -> str:
    def render(value: Any) -> str:
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        return str(value).replace("|", "\\|")

    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def _safe_correlation(truth: np.ndarray, prediction: np.ndarray, method: str) -> float:
    if len(truth) < 2 or np.std(truth) <= 1e-15 or np.std(prediction) <= 1e-15:
        return float("nan")
    value = (
        spearmanr(truth, prediction).statistic
        if method == "spearman"
        else pearsonr(truth, prediction).statistic
    )
    return float(value) if np.isfinite(value) else float("nan")


def _ccc(truth: np.ndarray, prediction: np.ndarray) -> float:
    covariance = float(np.mean((truth - truth.mean()) * (prediction - prediction.mean())))
    denominator = float(truth.var() + prediction.var() + (truth.mean() - prediction.mean()) ** 2)
    return 2.0 * covariance / denominator if denominator > 0 else float("nan")


def _pairwise_accuracy(truth: np.ndarray, prediction: np.ndarray) -> float:
    scores: list[float] = []
    for left, right in itertools.combinations(range(len(truth)), 2):
        truth_delta = truth[left] - truth[right]
        prediction_delta = prediction[left] - prediction[right]
        if abs(truth_delta) <= 1e-12:
            continue
        if abs(prediction_delta) <= 1e-12:
            scores.append(0.5)
        else:
            scores.append(float(np.sign(truth_delta) == np.sign(prediction_delta)))
    return float(np.mean(scores)) if scores else float("nan")


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for target in TARGETS:
        truth = next(
            (column for column in (f"{target}_true", target, f"true_{target}") if column in output),
            None,
        )
        prediction = next(
            (column for column in (f"{target}_pred", f"pred_{target}", "prediction") if column in output),
            None,
        )
        if truth is None or prediction is None:
            raise ValueError(f"Prediction table lacks {target} truth/prediction columns")
        output[f"{target}_true"] = pd.to_numeric(output[truth], errors="raise")
        output[f"{target}_pred"] = pd.to_numeric(output[prediction], errors="raise")
    return output


def _read_a_predictions(root: Path) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for path in sorted(root.rglob("condition_level_lopo_predictions.csv")):
        frame = _normalize_columns(pd.read_csv(path))
        frame["model"] = "a_classical_residual_ensemble"
        frame["project"] = "A"
        frame["source_file"] = str(path.resolve())
        frame["device"] = "cpu"
        frame["cuda_used"] = False
        frames.append(frame)
    for path in sorted(root.rglob("dcnn_condition_lopo_predictions.csv")):
        source = _normalize_columns(pd.read_csv(path))
        for variant, model in (("full", "a_dcnn_full"), ("no_eeg", "a_dcnn_no_eeg")):
            frame = source.loc[source["model_variant"].astype(str) == variant].copy()
            if frame.empty:
                raise ValueError(f"{path} lacks requested DCNN variant {variant!r}")
            frame["model"] = model
            frame["project"] = "A"
            frame["source_file"] = str(path.resolve())
            frames.append(frame)
    return frames


def _read_b_predictions(root: Path) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for path in sorted(root.rglob("*_predictions.csv")):
        frame = _normalize_columns(pd.read_csv(path))
        variants = tuple(sorted(frame["model_variant"].astype(str).unique()))
        if len(variants) != 1 or variants[0] not in B_VARIANT_TO_MODEL:
            raise ValueError(f"Unexpected Project B model variants in {path}: {variants}")
        frame["model"] = B_VARIANT_TO_MODEL[variants[0]]
        frame["project"] = "B"
        frame["source_file"] = str(path.resolve())
        frames.append(frame)
    return frames


def _expected_source_hashes(contract: dict[str, Any]) -> dict[str, str]:
    files = contract["files"]
    return {
        "labels": files["labels"]["sha256"],
        "windows": files["windows"]["sha256"],
        "split_manifest": files["split_manifest"]["sha256"],
        "modality_masks": files["modality_masks"]["sha256"],
        "mask_manifest": files["modality_masks"]["sha256"],
    }


def _validate_a_manifests(root: Path, contract: dict[str, Any]) -> list[dict[str, Any]]:
    expected = _expected_source_hashes(contract)
    rows: list[dict[str, Any]] = []
    for kind in ("classical", "dcnn"):
        paths = sorted(root.rglob(f"{kind}_alignment_manifest.json"))
        by_seed: dict[int, list[Path]] = {seed: [] for seed in SEEDS}
        for path in paths:
            payload = _json(path)
            seed = int(payload["seed"])
            if seed in by_seed:
                by_seed[seed].append(path)
        for seed in SEEDS:
            if len(by_seed[seed]) != 1:
                raise ValueError(f"Expected one Project A {kind} manifest for seed {seed}; found {by_seed[seed]}")
            path = by_seed[seed][0]
            payload = _json(path)
            sources = payload.get("sources", {})
            for name in ("labels", "windows", "split_manifest", "modality_masks"):
                if sources.get(name, {}).get("sha256") != expected[name]:
                    raise ValueError(f"Project A {kind} seed {seed} has wrong {name} hash")
            folds = payload.get("folds", [])
            if len(folds) != 15 or any(len(fold["train_participants"]) != 13 for fold in folds):
                raise ValueError(f"Project A {kind} seed {seed} manifest is not 13/1/1")
            runtime = payload.get("runtime", {})
            if kind == "dcnn" and not runtime.get("cuda_used"):
                raise ValueError(f"Project A DCNN seed {seed} did not record CUDA")
            rows.append(
                {
                    "project": "A",
                    "kind": kind,
                    "seed": seed,
                    "path": str(path.resolve()),
                    "sha256": file_sha256(path),
                    "cuda_used": bool(runtime.get("cuda_used", False)),
                    "device": runtime.get("device", "cpu"),
                    "input_hashes_ok": True,
                }
            )
    return rows


def _validate_b_results(root: Path, contract: dict[str, Any]) -> list[dict[str, Any]]:
    expected = _expected_source_hashes(contract)
    rows: list[dict[str, Any]] = []
    paths = sorted(root.rglob("*_results.json"))
    seen: set[tuple[str, int]] = set()
    for path in paths:
        payload = _json(path)
        model = B_RESULT_TO_MODEL.get(str(payload.get("model")))
        if model is None:
            continue
        seed = int(payload["seed"])
        key = (model, seed)
        if key in seen:
            raise ValueError(f"Duplicate Project B result for {key}")
        seen.add(key)
        inputs = payload.get("inputs", {})
        for name in ("labels", "windows", "split_manifest", "mask_manifest"):
            if inputs.get(name, {}).get("sha256") != expected[name]:
                raise ValueError(f"Project B {model} seed {seed} has wrong {name} hash")
        folds = payload.get("folds", [])
        if len(folds) != 15 or any(fold["n_train_participants"] != 13 for fold in folds):
            raise ValueError(f"Project B {model} seed {seed} is not 13/1/1")
        runtime = payload.get("runtime", {})
        if model in NEURAL_MODELS and not runtime.get("cuda_used"):
            raise ValueError(f"Project B neural result {model} seed {seed} did not record CUDA")
        if model in NEURAL_MODELS:
            expected_modalities = FULL_MODALITIES if model.endswith("_full") else NO_EEG_MODALITIES
            if tuple(payload.get("modalities", ())) != expected_modalities:
                raise ValueError(
                    f"Project B {model} seed {seed} has wrong modalities: "
                    f"{payload.get('modalities')}"
                )
        rows.append(
            {
                "project": "B",
                "kind": model,
                "seed": seed,
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "cuda_used": bool(runtime.get("cuda_used", False)),
                "device": runtime.get("device", "cpu"),
                "input_hashes_ok": True,
            }
        )
    expected_keys = {(model, seed) for model in EXPECTED_MODELS if model.startswith("b_") for seed in SEEDS}
    if seen != expected_keys:
        raise ValueError(f"Project B result coverage mismatch: missing={sorted(expected_keys - seen)}, extra={sorted(seen - expected_keys)}")
    return rows


def _validate_prediction_frame(
    frame: pd.DataFrame,
    labels: pd.DataFrame,
    fold_by_participant: dict[str, tuple[int, str]],
) -> dict[str, Any]:
    model = str(frame["model"].iloc[0])
    seeds = tuple(sorted(pd.to_numeric(frame["seed"], errors="raise").astype(int).unique()))
    if len(seeds) != 1 or seeds[0] not in SEEDS:
        raise ValueError(f"{model} prediction file has invalid seed(s): {seeds}")
    seed = seeds[0]
    if len(frame) != len(labels) or frame.duplicated(KEYS).any():
        raise ValueError(f"{model} seed {seed} does not contain exactly 135 unique predictions")
    contract_labels = labels[[*KEYS, "presentation_position", *TARGETS]].rename(
        columns={
            "presentation_position": "presentation_position_contract",
            **{target: f"{target}_contract" for target in TARGETS},
        }
    )
    ordered = frame.merge(
        contract_labels,
        on=KEYS,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not (ordered["_merge"] == "both").all():
        raise ValueError(f"{model} seed {seed} prediction keys differ from the contract")
    for target in TARGETS:
        if not np.allclose(
            ordered[f"{target}_true"], ordered[f"{target}_contract"], atol=1e-6, rtol=0.0
        ):
            raise ValueError(f"{model} seed {seed} {target} truths differ from the contract")
        prediction = ordered[f"{target}_pred"].to_numpy(dtype=float)
        if not np.isfinite(prediction).all() or np.any((prediction < 0.0) | (prediction > 1.0)):
            raise ValueError(f"{model} seed {seed} has invalid {target} predictions")
    for participant, group in frame.groupby("participant_id"):
        fold_index, validation = fold_by_participant[str(participant)]
        if set(pd.to_numeric(group["fold_index"], errors="raise").astype(int)) != {fold_index}:
            raise ValueError(f"{model} seed {seed} has wrong fold for {participant}")
        if set(group["validation_participant"].astype(str)) != {validation}:
            raise ValueError(f"{model} seed {seed} has wrong validation participant for {participant}")
        if "test_participant" in group and set(group["test_participant"].astype(str)) != {str(participant)}:
            raise ValueError(f"{model} seed {seed} has wrong test participant field")
    if "split_protocol" in frame and set(frame["split_protocol"].astype(str)) != {
        "shared_13_train_1_validation_1_test"
    }:
        raise ValueError(f"{model} seed {seed} has the wrong split protocol")
    cuda_values = frame["cuda_used"].astype(str).str.lower().isin({"true", "1", "1.0"})
    if model in NEURAL_MODELS and not cuda_values.all():
        raise ValueError(f"{model} seed {seed} contains a non-CUDA neural prediction")
    if model not in NEURAL_MODELS and cuda_values.any():
        raise ValueError(f"{model} seed {seed} unexpectedly records CUDA")
    if model.startswith("b_corrected_") and model in NEURAL_MODELS:
        expected_modalities = FULL_MODALITIES if model.endswith("_full") else NO_EEG_MODALITIES
        observed_modalities = tuple(frame["modalities"].astype(str).unique())
        if observed_modalities != ("+".join(expected_modalities),):
            raise ValueError(
                f"{model} seed {seed} prediction mask is wrong: {observed_modalities}"
            )
    return {
        "project": str(frame["project"].iloc[0]),
        "model": model,
        "seed": seed,
        "rows": len(frame),
        "participants": frame["participant_id"].nunique(),
        "conditions": frame["condition"].nunique(),
        "folds": frame["fold_index"].nunique(),
        "prediction_coverage_ok": True,
        "fold_mapping_ok": True,
        "cuda_policy_ok": True,
        "source_file": str(frame["source_file"].iloc[0]),
        "source_sha256": file_sha256(Path(str(frame["source_file"].iloc[0]))),
    }


def _baseline_frames(frames: list[pd.DataFrame]) -> list[pd.DataFrame]:
    output: list[pd.DataFrame] = []
    for seed in SEEDS:
        candidates = [frame.sort_values(KEYS).reset_index(drop=True) for frame in frames if int(frame["seed"].iloc[0]) == seed]
        reference = candidates[0]
        for prefix, model in (("condition_only", "condition_only_baseline"), ("history", "history_baseline")):
            columns = [f"{prefix}_{target}" for target in TARGETS]
            if any(column not in reference for column in columns):
                raise ValueError(f"Reference predictions lack standardized {prefix} baseline columns")
            for candidate in candidates[1:]:
                if not candidate[KEYS].equals(reference[KEYS]):
                    raise ValueError("Baseline candidate prediction keys are not identical")
                for column in columns:
                    if column not in candidate or not np.allclose(
                        reference[column].to_numpy(dtype=float),
                        candidate[column].to_numpy(dtype=float),
                        atol=1e-6,
                        rtol=0.0,
                    ):
                        raise ValueError(f"Baseline {column} is not identical across projects/models for seed {seed}")
            baseline = reference[
                [*KEYS, "presentation_position", "fold_index", "validation_participant", "seed", *[f"{target}_true" for target in TARGETS]]
            ].copy()
            for target in TARGETS:
                baseline[f"{target}_pred"] = reference[f"{prefix}_{target}"].to_numpy(dtype=float)
            baseline["model"] = model
            baseline["project"] = "shared"
            baseline["source_file"] = reference["source_file"].iloc[0]
            baseline["device"] = "cpu"
            baseline["cuda_used"] = False
            output.append(baseline)
    return output


def _participant_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, seed, participant), group in frame.groupby(["model", "seed", "participant_id"], sort=True):
        for target in TARGETS:
            truth = group[f"{target}_true"].to_numpy(dtype=float)
            prediction = group[f"{target}_pred"].to_numpy(dtype=float)
            rows.append(
                {
                    "model": model,
                    "seed": int(seed),
                    "participant_id": participant,
                    "target": target,
                    "n_conditions": len(group),
                    "mae": float(np.mean(np.abs(truth - prediction))),
                    "rmse": float(np.sqrt(np.mean((truth - prediction) ** 2))),
                    "spearman": _safe_correlation(truth, prediction, "spearman"),
                    "pearson": _safe_correlation(truth, prediction, "pearson"),
                    "pairwise_accuracy": _pairwise_accuracy(truth, prediction),
                }
            )
    return pd.DataFrame(rows)


def _seed_metrics(frame: pd.DataFrame, participant: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, seed), group in frame.groupby(["model", "seed"], sort=True):
        for target in TARGETS:
            truth = group[f"{target}_true"].to_numpy(dtype=float)
            prediction = group[f"{target}_pred"].to_numpy(dtype=float)
            subset = participant.loc[
                (participant["model"] == model)
                & (participant["seed"] == seed)
                & (participant["target"] == target)
            ]
            rows.append(
                {
                    "model": model,
                    "seed": int(seed),
                    "target": target,
                    "n_predictions": len(group),
                    "mae": float(np.mean(np.abs(truth - prediction))),
                    "participant_macro_mae": float(subset["mae"].mean()),
                    "rmse": float(np.sqrt(np.mean((truth - prediction) ** 2))),
                    "spearman": _safe_correlation(truth, prediction, "spearman"),
                    "pearson": _safe_correlation(truth, prediction, "pearson"),
                    "ccc": _ccc(truth, prediction),
                    "participant_macro_spearman": float(subset["spearman"].mean()),
                    "participant_macro_pairwise_accuracy": float(subset["pairwise_accuracy"].mean()),
                }
            )
    return pd.DataFrame(rows)


def exact_sign_flip_pvalue(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan")
    observed = abs(float(values.mean()))
    total = 1 << len(values)
    extreme = 0
    for mask in range(total):
        signs = np.fromiter((1.0 if mask & (1 << index) else -1.0 for index in range(len(values))), dtype=float)
        extreme += abs(float(np.mean(values * signs))) >= observed - 1e-15
    return float(extreme / total)


def holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    adjusted_sorted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted_sorted[rank] = min(1.0, running)
    adjusted = np.empty(len(values), dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def _paired_comparisons(
    participant: pd.DataFrame,
    pairs: list[tuple[str, str, str]],
    *,
    bootstrap: int,
    seed: int,
    expected_participants: int = 15,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    averaged = participant.groupby(["model", "participant_id", "target"], as_index=False)["mae"].mean()
    for left, right, comparison in pairs:
        for target in TARGETS:
            subset = averaged.loc[averaged["target"] == target]
            left_values = subset.loc[subset["model"] == left, ["participant_id", "mae"]].rename(columns={"mae": "left_mae"})
            right_values = subset.loc[subset["model"] == right, ["participant_id", "mae"]].rename(columns={"mae": "right_mae"})
            paired = left_values.merge(right_values, on="participant_id", validate="one_to_one")
            if len(paired) != expected_participants:
                raise ValueError(
                    f"Comparison {comparison} {target} does not have "
                    f"{expected_participants} paired participants"
                )
            delta = paired["right_mae"].to_numpy(dtype=float) - paired["left_mae"].to_numpy(dtype=float)
            draws = rng.choice(delta, size=(bootstrap, len(delta)), replace=True).mean(axis=1)
            rows.append(
                {
                    "comparison": comparison,
                    "left_model": left,
                    "right_model": right,
                    "target": target,
                    "delta_definition": "right_minus_left_participant_mae",
                    "mean_delta": float(delta.mean()),
                    "median_delta": float(np.median(delta)),
                    "ci95_low": float(np.quantile(draws, 0.025)),
                    "ci95_high": float(np.quantile(draws, 0.975)),
                    "exact_sign_flip_p": exact_sign_flip_pvalue(delta),
                    "n_participants": len(delta),
                }
            )
    output = pd.DataFrame(rows)
    output["holm_p"] = holm_adjust(output["exact_sign_flip_p"].to_numpy(dtype=float))
    output["holm_significant_0_05"] = output["holm_p"] <= 0.05
    return output


def _write_forest(
    comparisons: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    height = max(5.0, 0.34 * len(comparisons))
    figure, axis = plt.subplots(figsize=(11, height))
    positions = np.arange(len(comparisons))
    means = comparisons["mean_delta"].to_numpy(dtype=float)
    low = means - comparisons["ci95_low"].to_numpy(dtype=float)
    high = comparisons["ci95_high"].to_numpy(dtype=float) - means
    axis.errorbar(means, positions, xerr=np.vstack([low, high]), fmt="o", color="#C00000", capsize=4)
    axis.axvline(0.0, color="black", linewidth=1)
    axis.set_yticks(
        positions,
        [f"{row.comparison}: {row.target}" for row in comparisons.itertuples()],
        fontsize=8,
    )
    axis.set_xlabel("Right minus left participant MAE (negative favors right)")
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _write_figures(
    summary: pd.DataFrame,
    cross: pd.DataFrame,
    versus_late: pd.DataFrame,
    versus_dcnn: pd.DataFrame,
    mask_comparisons: pd.DataFrame,
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = [*EXPECTED_MODELS, "condition_only_baseline", "history_baseline"]
    labels = [model.replace("_", "\n") for model in models]
    for metric, filename, ylabel in (
        ("participant_macro_mae", "model_mae_comparison.png", "Participant-macro MAE (lower is better)"),
        ("spearman", "model_spearman_comparison.png", "Pooled Spearman rho (higher is better)"),
    ):
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(max(18, len(models) * 1.3), 7),
            sharey=False,
        )
        for axis, target in zip(axes, TARGETS, strict=True):
            subset = summary.loc[summary["target"] == target].set_index("model")
            means = [subset.loc[model, f"{metric}_mean"] for model in models]
            errors = [subset.loc[model, f"{metric}_std"] for model in models]
            axis.bar(np.arange(len(models)), means, yerr=errors, capsize=3, color="#4472C4")
            axis.set_xticks(np.arange(len(models)), labels, rotation=45, ha="right", fontsize=8)
            axis.set_title(target.capitalize())
            axis.set_ylabel(ylabel)
            axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(output / filename, dpi=220, bbox_inches="tight")
        plt.close(figure)

    _write_forest(
        cross,
        output / "cross_project_mae_forest.png",
        title="Existing aligned cross-project comparisons",
    )
    _write_forest(
        versus_late,
        output / "new_fusion_vs_late_mae_forest.png",
        title="New Project B fusions versus aligned Project B Late fusion",
    )
    _write_forest(
        versus_dcnn,
        output / "new_fusion_vs_project_a_dcnn_mae_forest.png",
        title="New Project B fusions versus aligned Project A DCNN",
    )
    _write_forest(
        mask_comparisons,
        output / "full_vs_no_eeg_mae_forest.png",
        title="Aligned no-EEG minus full-modality comparisons",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract = validate_alignment_contract(args.contract_dir)
    labels = pd.read_csv(args.contract_dir / contract["files"]["labels"]["path"])
    split_path = args.contract_dir / contract["files"]["split_manifest"]["path"]
    folds = load_split_manifest(split_path, expected_participants=contract["participants"], expected_train_count=13)
    fold_by_participant = {
        fold.test_participant: (fold.fold_index, fold.validation_participant) for fold in folds
    }

    frames = [*_read_a_predictions(args.project_a_root), *_read_b_predictions(args.project_b_root)]
    observed = [(str(frame["model"].iloc[0]), int(frame["seed"].iloc[0])) for frame in frames]
    expected = [(model, seed) for model in EXPECTED_MODELS for seed in SEEDS]
    if sorted(observed) != sorted(expected):
        raise ValueError(f"Prediction run coverage mismatch: missing={sorted(set(expected)-set(observed))}, extra={sorted(set(observed)-set(expected))}")
    coverage = [_validate_prediction_frame(frame, labels, fold_by_participant) for frame in frames]
    baseline_frames = _baseline_frames(frames)
    all_frames = [*frames, *baseline_frames]
    normalized = pd.concat(all_frames, ignore_index=True, sort=False)
    normalized = normalized.sort_values(["model", "seed", "participant_id", "presentation_position"]).reset_index(drop=True)

    participant = _participant_metrics(normalized)
    seed_metrics = _seed_metrics(normalized, participant)
    metric_columns = [
        "mae", "participant_macro_mae", "rmse", "spearman", "pearson", "ccc",
        "participant_macro_spearman", "participant_macro_pairwise_accuracy",
    ]
    summary = seed_metrics.groupby(["model", "target"])[metric_columns].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(part for part in column if part) if isinstance(column, tuple) else column
        for column in summary.columns
    ]

    cross_pairs = [
        ("a_classical_residual_ensemble", "b_corrected_ridge_cv", "classical_vs_ridge"),
        ("a_dcnn_full", "b_corrected_late_full", "dcnn_full_vs_late_full"),
        ("a_dcnn_no_eeg", "b_corrected_late_no_eeg", "dcnn_no_eeg_vs_late_no_eeg"),
    ]
    cross = _paired_comparisons(participant, cross_pairs, bootstrap=args.bootstrap, seed=args.bootstrap_seed)
    versus_late_pairs = [
        (
            f"b_corrected_late_{mask}",
            f"b_corrected_{fusion}_{mask}",
            f"{fusion}_{mask}_vs_late_{mask}",
        )
        for fusion in NEW_FUSIONS
        for mask in ("full", "no_eeg")
    ]
    versus_late = _paired_comparisons(
        participant,
        versus_late_pairs,
        bootstrap=args.bootstrap,
        seed=args.bootstrap_seed + 1,
    )
    versus_dcnn_pairs = [
        (
            f"a_dcnn_{mask}",
            f"b_corrected_{fusion}_{mask}",
            f"{fusion}_{mask}_vs_project_a_dcnn_{mask}",
        )
        for fusion in NEW_FUSIONS
        for mask in ("full", "no_eeg")
    ]
    versus_dcnn = _paired_comparisons(
        participant,
        versus_dcnn_pairs,
        bootstrap=args.bootstrap,
        seed=args.bootstrap_seed + 2,
    )
    mask_pairs = [
        ("a_dcnn_full", "a_dcnn_no_eeg", "a_dcnn_no_eeg_vs_full"),
        ("b_corrected_late_full", "b_corrected_late_no_eeg", "b_late_no_eeg_vs_full"),
        *[
            (
                f"b_corrected_{fusion}_full",
                f"b_corrected_{fusion}_no_eeg",
                f"b_{fusion}_no_eeg_vs_full",
            )
            for fusion in NEW_FUSIONS
        ],
    ]
    mask_comparisons = _paired_comparisons(
        participant,
        mask_pairs,
        bootstrap=args.bootstrap,
        seed=args.bootstrap_seed + 3,
    )
    baseline_pairs = [
        (baseline, model, f"{model}_vs_{baseline}")
        for model in EXPECTED_MODELS
        for baseline in ("condition_only_baseline", "history_baseline")
    ]
    baseline = _paired_comparisons(
        participant, baseline_pairs, bootstrap=args.bootstrap, seed=args.bootstrap_seed + 4
    )

    new_summary = summary.loc[summary["model"].isin(NEW_ALIGNED_MODELS)].copy()
    reference_summary = summary.loc[summary["model"].isin(EXISTING_ALIGNED_MODELS)].copy()

    provenance = [
        *_validate_a_manifests(args.project_a_root, contract),
        *_validate_b_results(args.project_b_root, contract),
    ]
    normalized.to_csv(args.output_dir / "normalized_predictions.csv", index=False)
    pd.DataFrame(coverage).sort_values(["model", "seed"]).to_csv(
        args.output_dir / "run_validation.csv", index=False
    )
    pd.DataFrame(provenance).sort_values(["project", "kind", "seed"]).to_csv(
        args.output_dir / "provenance_validation.csv", index=False
    )
    participant.to_csv(args.output_dir / "participant_metrics.csv", index=False)
    seed_metrics.to_csv(args.output_dir / "seed_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "comparison_summary.csv", index=False)
    new_summary.to_csv(args.output_dir / "new_fusion_summary.csv", index=False)
    reference_summary.to_csv(args.output_dir / "existing_aligned_reference_summary.csv", index=False)
    cross.to_csv(args.output_dir / "cross_project_paired_tests.csv", index=False)
    versus_late.to_csv(args.output_dir / "new_fusion_vs_late_paired_tests.csv", index=False)
    versus_dcnn.to_csv(
        args.output_dir / "new_fusion_vs_project_a_dcnn_paired_tests.csv", index=False
    )
    mask_comparisons.to_csv(args.output_dir / "full_vs_no_eeg_paired_tests.csv", index=False)
    baseline.to_csv(args.output_dir / "baseline_paired_tests.csv", index=False)
    _write_figures(
        summary,
        cross,
        versus_late,
        versus_dcnn,
        mask_comparisons,
        args.output_dir,
    )

    report_lines = [
        "# Aligned cross-project comparison",
        "",
        f"- Contract: `{(args.contract_dir / 'contract.json').resolve()}`",
        f"- Seeds: {', '.join(map(str, SEEDS))}",
        f"- Folds: {len(folds)}; each fold is 13 train / 1 validation / 1 test.",
        f"- Bootstrap replicates: {args.bootstrap}; paired exact sign-flip tests use 2^15 assignments.",
        "- Negative paired MAE delta favors the right-hand model.",
        "",
        "## New aligned Project B fusion runs (current extension)",
        "",
        _markdown_table(new_summary),
        "",
        "## Existing aligned reference runs (previously completed)",
        "",
        _markdown_table(reference_summary),
        "",
        "## New fusions versus aligned Project B Late fusion",
        "",
        _markdown_table(versus_late),
        "",
        "## New fusions versus aligned Project A DCNN",
        "",
        _markdown_table(versus_dcnn),
        "",
        "## Full modalities versus no-EEG",
        "",
        _markdown_table(mask_comparisons),
        "",
        "## Existing aligned cross-project paired tests",
        "",
        _markdown_table(cross),
        "",
        "## Scope",
        "",
        "All values are out-of-fold participant-condition predictions under the immutable shared contract. "
        "Statistical inference treats participant as the independent unit and averages each participant across the three seeds. "
        "Each paired-test CSV is a separate Holm family. The evaluator does not ingest old Project B native or "
        "superseded runs outside the aligned run root; those results remain out of scope and must not be mixed with "
        "the aligned tables above.",
    ]
    (args.output_dir / "aligned_comparison_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    audit = {
        "schema_version": "aligned_comparison_evaluation_v2",
        "ok": True,
        "contract_sha256": file_sha256(args.contract_dir / "contract.json"),
        "models": list(EXPECTED_MODELS),
        "seeds": list(SEEDS),
        "prediction_runs": len(frames),
        "prediction_rows": int(sum(len(frame) for frame in frames)),
        "fold_count": len(folds),
        "bootstrap": args.bootstrap,
        "outputs": sorted(
            {path.name for path in args.output_dir.iterdir() if path.is_file()}
            | {"evaluation_audit.json"}
        ),
    }
    (args.output_dir / "evaluation_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return audit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = ROOT / "artifacts" / "cross_project_alignment_2026-07-16"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", type=Path, default=root / "alignment_contract")
    parser.add_argument("--project-a-root", type=Path, default=root / "aligned_runs" / "project_a")
    parser.add_argument("--project-b-root", type=Path, default=root / "aligned_runs" / "project_b")
    parser.add_argument("--output-dir", type=Path, default=root / "evaluation")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260716)
    args = parser.parse_args(argv)
    if args.bootstrap < 1000:
        parser.error("--bootstrap must be at least 1000")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

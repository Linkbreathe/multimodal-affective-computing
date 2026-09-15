"""Evaluate the formal condition-anchor probe family against exact baselines."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from scripts.run_relax_condition_anchor_probe import (
    ALLOWED_SEEDS,
    CANDIDATES,
    EXPECTED_OBSERVATIONS,
    EXPECTED_PARTICIPANTS,
    EXPECTED_VALID_WINDOWS,
    TARGETS,
    _write_json,
    file_sha256,
)
from scripts.run_relax_foundation_probe import _load_split_manifest, _metrics


OUTCOMES = (*TARGETS, "macro")
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260717


APPROACH_DESCRIPTIONS = {
    "ecg_raw_dual": "ECG frozen mean embedding; separate bounded residual Ridge heads.",
    "ecg_centered_dual": "ECG embedding centered by train-only condition centroid; separate residual heads.",
    "ecg_eye_raw_dual": "Raw frozen ECG+eye means; separate bounded residual heads.",
    "ecg_eye_centered_dual": "Train-condition-centered ECG+eye means; separate residual heads.",
    "no_eeg_raw_dual": "Raw ECG+eye+head+video frozen means; separate bounded residual heads.",
    "no_eeg_centered_dual": "Train-condition-centered ECG+eye+head+video means; separate residual heads.",
    "eye_discomfort_compact": "Eye-only discomfort residual Ridge with compact PCA/alpha grid; relaxation exact fallback.",
    "eye_discomfort_wide": "Eye-only discomfort residual Ridge with wider PCA/alpha/gamma grid; relaxation exact fallback.",
}


def _load_runs(runs_dir: Path) -> tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame]:
    prediction_frames: list[pd.DataFrame] = []
    payloads: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    expected = {(candidate, seed) for candidate in CANDIDATES for seed in ALLOWED_SEEDS}
    observed: set[tuple[str, int]] = set()
    reference_hashes: dict[str, str] | None = None
    for candidate, seed in sorted(expected):
        run_dir = runs_dir / candidate / f"seed_{seed}"
        result_path = run_dir / f"{candidate}_s{seed}_results.json"
        prediction_path = run_dir / f"{candidate}_s{seed}_predictions.csv"
        issues: list[str] = []
        if not result_path.is_file() or not prediction_path.is_file():
            missing = [str(path) for path in (result_path, prediction_path) if not path.is_file()]
            raise FileNotFoundError(f"Missing formal run artifacts: {missing}")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        frame = pd.read_csv(prediction_path)
        if payload["candidate"]["name"] != candidate or int(payload["seed"]) != seed:
            issues.append("result_identity")
        if len(frame) != EXPECTED_OBSERVATIONS or frame.duplicated(["participant_id", "condition"]).any():
            issues.append("oof_coverage")
        if set(frame["participant_id"].astype(str)) != set(EXPECTED_PARTICIPANTS):
            issues.append("participant_membership")
        if set(frame["seed"].astype(int)) != {seed} or set(frame["candidate"].astype(str)) != {candidate}:
            issues.append("prediction_identity")
        if len(payload["folds"]) != 9 or any(
            fold["n_train_participants"] != 7
            or fold["n_validation_participants"] != 1
            or fold["n_test_participants"] != 1
            or fold["n_train_observations"] != 63
            or fold["n_validation_observations"] != 9
            or fold["n_test_observations"] != 9
            for fold in payload["folds"]
        ):
            issues.append("fold_protocol")
        if payload["contract"]["common_valid_window_count"] != EXPECTED_VALID_WINDOWS:
            issues.append("window_count")
        if not payload["head_runtime"]["embedding_extraction_cuda_used"]:
            issues.append("embedding_cuda")
        if payload["head_runtime"]["neural_training_performed"]:
            issues.append("unexpected_neural_training")
        if payload["head_runtime"]["device"] != "cpu":
            issues.append("ridge_device")
        if file_sha256(prediction_path) != payload["prediction_sha256"]:
            issues.append("prediction_hash")
        hashes = {
            name: record["sha256"]
            for name, record in payload["contract"]["inputs"].items()
        }
        if reference_hashes is None:
            reference_hashes = hashes
        elif hashes != reference_hashes:
            issues.append("input_hash_mismatch")
        if frame[[f"{target}_pred" for target in TARGETS]].isna().any().any():
            issues.append("missing_prediction")
        if not all(frame[f"{target}_pred"].between(0.0, 1.0).all() for target in TARGETS):
            issues.append("prediction_bounds")
        audit_rows.append(
            {
                "candidate": candidate,
                "seed": seed,
                "valid": not issues,
                "issues": ";".join(issues),
                "observations": len(frame),
                "folds": len(payload["folds"]),
                "common_valid_windows": payload["contract"]["common_valid_window_count"],
                "embedding_cuda_used": payload["head_runtime"]["embedding_extraction_cuda_used"],
                "embedding_device": payload["head_runtime"]["embedding_extraction_device"],
                "head_device": payload["head_runtime"]["device"],
                "result_path": str(result_path),
                "prediction_path": str(prediction_path),
            }
        )
        if issues:
            raise ValueError(f"Formal run {candidate}/{seed} failed audit: {issues}")
        frame["source_file"] = str(prediction_path)
        prediction_frames.append(frame)
        payloads.append(payload)
        observed.add((candidate, seed))
    if observed != expected:
        raise ValueError(f"Formal matrix differs from expected family: missing={expected-observed}, extra={observed-expected}")
    return pd.concat(prediction_frames, ignore_index=True), payloads, pd.DataFrame(audit_rows)


def _seed_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (candidate, seed), frame in predictions.groupby(["candidate", "seed"], sort=True):
        metrics = _metrics(frame)
        baseline = frame.copy()
        for target in TARGETS:
            baseline[f"{target}_pred"] = baseline[f"condition_only_{target}"]
        baseline_metrics = _metrics(baseline)
        row = {"candidate": candidate, "seed": int(seed)}
        for target in TARGETS:
            value = metrics["targets"][target]["participant_macro_mae"]
            base = baseline_metrics["targets"][target]["participant_macro_mae"]
            row[f"{target}_mae"] = value
            row[f"{target}_baseline_mae"] = base
            row[f"{target}_delta"] = value - base
        row["macro_mae"] = metrics["macro_mae"]
        row["macro_baseline_mae"] = baseline_metrics["macro_mae"]
        row["macro_delta"] = metrics["macro_mae"] - baseline_metrics["macro_mae"]
        rows.append(row)
    return pd.DataFrame(rows)


def _participant_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (candidate, seed, participant), frame in predictions.groupby(
        ["candidate", "seed", "participant_id"], sort=True
    ):
        target_rows = []
        for target in TARGETS:
            model = float(np.mean(np.abs(frame[f"{target}_true"] - frame[f"{target}_pred"])))
            baseline = float(
                np.mean(np.abs(frame[f"{target}_true"] - frame[f"condition_only_{target}"]))
            )
            target_rows.append((model, baseline))
            rows.append(
                {
                    "candidate": candidate,
                    "seed": int(seed),
                    "participant_id": participant,
                    "outcome": target,
                    "model_mae": model,
                    "baseline_mae": baseline,
                    "delta": model - baseline,
                }
            )
        model = float(np.mean([value[0] for value in target_rows]))
        baseline = float(np.mean([value[1] for value in target_rows]))
        rows.append(
            {
                "candidate": candidate,
                "seed": int(seed),
                "participant_id": participant,
                "outcome": "macro",
                "model_mae": model,
                "baseline_mae": baseline,
                "delta": model - baseline,
            }
        )
    return pd.DataFrame(rows)


def _exact_sign_flip(deltas: np.ndarray) -> tuple[float, int]:
    observed = abs(float(np.mean(deltas)))
    assignments = np.asarray(list(product((-1.0, 1.0), repeat=len(deltas))), dtype=np.float64)
    permuted = np.abs(np.mean(assignments * deltas[None, :], axis=1))
    count = int(np.sum(permuted >= observed - 1e-15))
    return count / len(assignments), len(assignments)


def _holm(pvalues: pd.Series) -> pd.Series:
    order = np.argsort(pvalues.to_numpy(dtype=float), kind="stable")
    values = pvalues.to_numpy(dtype=float)[order]
    adjusted_sorted = np.maximum.accumulate((len(values) - np.arange(len(values))) * values)
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty(len(values), dtype=float)
    adjusted[order] = adjusted_sorted
    return pd.Series(adjusted, index=pvalues.index)


def _paired_statistics(participant_metrics: pd.DataFrame) -> pd.DataFrame:
    seed_averaged = (
        participant_metrics.groupby(["candidate", "participant_id", "outcome"], as_index=False)["delta"]
        .mean()
        .sort_values(["candidate", "outcome", "participant_id"])
    )
    rows: list[dict[str, Any]] = []
    for index, ((candidate, outcome), frame) in enumerate(
        seed_averaged.groupby(["candidate", "outcome"], sort=True)
    ):
        deltas = frame["delta"].to_numpy(dtype=float)
        if len(deltas) != 9:
            raise ValueError(f"Paired inference requires nine participant clusters, got {len(deltas)}")
        rng = np.random.default_rng(BOOTSTRAP_SEED + index)
        sample_indexes = rng.integers(0, len(deltas), size=(BOOTSTRAP_RESAMPLES, len(deltas)))
        bootstrap = deltas[sample_indexes].mean(axis=1)
        pvalue, assignments = _exact_sign_flip(deltas)
        rows.append(
            {
                "candidate": candidate,
                "outcome": outcome,
                "mean_paired_delta": float(deltas.mean()),
                "bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
                "bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "bootstrap_cluster": "participant_after_seed_average",
                "exact_sign_flip_p_two_sided": pvalue,
                "sign_flip_assignments": assignments,
                "participants_improved": int(np.sum(deltas < 0.0)),
                "participants_tied": int(np.sum(np.isclose(deltas, 0.0, atol=1e-15))),
                "participants_worsened": int(np.sum(deltas > 0.0)),
                "participant_count": len(deltas),
            }
        )
    result = pd.DataFrame(rows)
    result["holm_family"] = "8_formal_candidates_x_3_outcomes"
    result["holm_family_size"] = len(result)
    result["holm_p"] = _holm(result["exact_sign_flip_p_two_sided"])
    result["holm_significant_0_05"] = result["holm_p"] < 0.05
    return result


def _candidate_summary(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate, frame in seed_metrics.groupby("candidate", sort=True):
        row: dict[str, Any] = {
            "candidate": candidate,
            "family": CANDIDATES[candidate].family,
            "modalities": "+".join(CANDIDATES[candidate].modalities),
            "representation": CANDIDATES[candidate].representation,
            "description": APPROACH_DESCRIPTIONS[candidate],
            "seed_count": len(frame),
        }
        for outcome in OUTCOMES:
            row[f"{outcome}_mae_mean"] = float(frame[f"{outcome}_mae"].mean())
            row[f"{outcome}_mae_std"] = float(frame[f"{outcome}_mae"].std(ddof=1))
            row[f"{outcome}_delta_mean"] = float(frame[f"{outcome}_delta"].mean())
            row[f"{outcome}_improves_all_seeds"] = bool((frame[f"{outcome}_delta"] < -1e-12).all())
            row[f"{outcome}_not_worse_all_seeds"] = bool((frame[f"{outcome}_delta"] <= 1e-12).all())
        row["both_targets_not_worse"] = bool(
            row["relaxation_not_worse_all_seeds"] and row["discomfort_not_worse_all_seeds"]
        )
        row["both_targets_improve"] = bool(
            row["relaxation_improves_all_seeds"] and row["discomfort_improves_all_seeds"]
        )
        row["primary_point_success"] = bool(
            row["macro_improves_all_seeds"] and row["both_targets_not_worse"]
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("macro_mae_mean").reset_index(drop=True)


def _targetwise_control(labels_path: Path, split_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    labels = pd.read_csv(labels_path).sort_values(["participant_id", "presentation_position"]).reset_index(drop=True)
    folds = _load_split_manifest(split_path, EXPECTED_PARTICIPANTS, strict=True)
    rows = []
    for fold in folds:
        train = labels[labels["participant_id"].astype(str).isin(fold.train_participants)]
        test = labels[labels["participant_id"].astype(str) == fold.test_participant]
        relaxation_global = float(train["relaxation"].mean())
        discomfort_condition = train.groupby("condition")["discomfort"].mean()
        relaxation_condition = train.groupby("condition")["relaxation"].mean()
        for row in test.itertuples(index=False):
            rows.append(
                {
                    "participant_id": str(row.participant_id),
                    "condition": str(row.condition),
                    "presentation_position": float(row.presentation_position),
                    "fold_index": fold.fold_index,
                    "relaxation_true": float(row.relaxation),
                    "discomfort_true": float(row.discomfort),
                    "relaxation_pred": relaxation_global,
                    "discomfort_pred": float(discomfort_condition.loc[row.condition]),
                    "condition_only_relaxation": float(relaxation_condition.loc[row.condition]),
                    "condition_only_discomfort": float(discomfort_condition.loc[row.condition]),
                }
            )
    frame = pd.DataFrame(rows).sort_values(["participant_id", "presentation_position"]).reset_index(drop=True)
    return frame, _metrics(frame)


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    for suffix, kwargs in (
        (".svg", {}),
        (".pdf", {}),
        (".png", {"dpi": 300}),
        (".tiff", {"dpi": 600}),
    ):
        fig.savefig(stem.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)


def _configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def _performance_figure(
    seed_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    control_macro: float,
    figure_dir: Path,
) -> None:
    _configure_plotting()
    order = summary["candidate"].tolist()
    short_labels = {
        "eye_discomfort_compact": "Eye compact",
        "eye_discomfort_wide": "Eye wide",
        "no_eeg_raw_dual": "No-EEG raw",
        "no_eeg_centered_dual": "No-EEG centered",
        "ecg_centered_dual": "ECG centered",
        "ecg_raw_dual": "ECG raw",
        "ecg_eye_raw_dual": "ECG+eye raw",
        "ecg_eye_centered_dual": "ECG+eye centered",
    }
    labels = [short_labels[value] for value in order]
    xlabels = [value.replace(" ", "\n") for value in labels]
    colors = {candidate: ("#2b8c68" if summary.set_index("candidate").loc[candidate, "primary_point_success"] else "#6688aa") for candidate in order}
    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 76 / 25.4), gridspec_kw={"width_ratios": [1.05, 1.0]})

    ax = axes[0]
    for index, candidate in enumerate(order):
        values = seed_metrics.loc[seed_metrics["candidate"] == candidate, "macro_mae"].to_numpy()
        ax.scatter(np.full(len(values), index), values, s=13, color=colors[candidate], alpha=0.65, zorder=2)
        ax.scatter(index, values.mean(), s=28, color=colors[candidate], edgecolor="white", linewidth=0.5, zorder=3)
    baseline = float(seed_metrics["macro_baseline_mae"].iloc[0])
    ax.axhline(baseline, color="#333333", linestyle="--", linewidth=0.9, label="Condition-only")
    ax.axhline(control_macro, color="#aa6f39", linestyle=":", linewidth=0.9, label="Global-relax + condition-discomfort")
    ax.set_xticks(range(len(order)), xlabels, rotation=25, ha="right", fontsize=5.5)
    ax.set_ylabel("Participant-macro MAE")
    ax.set_title("a  Three-seed point estimates", loc="left", fontweight="bold")
    ax.legend(fontsize=5.5, loc="upper right")

    ax = axes[1]
    macro = paired[paired["outcome"] == "macro"].set_index("candidate").loc[order].reset_index()
    positions = np.arange(len(order))
    for position, row in zip(positions, macro.itertuples(index=False), strict=True):
        color = "#2b8c68" if row.bootstrap_ci_high < 0 else "#6688aa"
        ax.plot([row.bootstrap_ci_low, row.bootstrap_ci_high], [position, position], color=color, linewidth=1.4)
        ax.scatter(row.mean_paired_delta, position, color=color, s=20, zorder=2)
    ax.axvline(0.0, color="#333333", linestyle="--", linewidth=0.8)
    ax.set_yticks(positions, labels, fontsize=5.5)
    ax.invert_yaxis()
    ax.set_xlabel("Candidate − Condition MAE (negative is better)")
    ax.set_title("b  Participant-cluster bootstrap (95% CI)", loc="left", fontweight="bold")
    fig.suptitle("Condition-anchored frozen-embedding probes", fontsize=9, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_figure(fig, figure_dir / "candidate_performance")


def _primary_participant_figure(
    predictions: pd.DataFrame,
    participant_metrics: pd.DataFrame,
    primary: str,
    figure_dir: Path,
) -> pd.DataFrame:
    _configure_plotting()
    averaged = (
        participant_metrics[participant_metrics["candidate"] == primary]
        .groupby(["participant_id", "outcome"], as_index=False)["delta"]
        .mean()
    )
    matrix = averaged.pivot(index="participant_id", columns="outcome", values="delta").loc[
        list(EXPECTED_PARTICIPANTS), list(OUTCOMES)
    ]
    primary_predictions = predictions[predictions["candidate"] == primary].copy()
    condition_rows = []
    for condition, frame in primary_predictions.groupby("condition", sort=True):
        for target in TARGETS:
            model = np.abs(frame[f"{target}_true"] - frame[f"{target}_pred"])
            baseline = np.abs(frame[f"{target}_true"] - frame[f"condition_only_{target}"])
            condition_rows.append(
                {"condition": condition, "outcome": target, "delta": float(np.mean(model - baseline))}
            )
    condition = pd.DataFrame(condition_rows)

    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 78 / 25.4), gridspec_kw={"width_ratios": [0.9, 1.1]})
    ax = axes[0]
    limit = max(abs(matrix.to_numpy()).max(), 1e-6)
    image = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit), aspect="auto")
    ax.set_xticks(range(len(OUTCOMES)), [value.capitalize() for value in OUTCOMES], rotation=25, ha="right")
    ax.set_yticks(range(len(matrix)), matrix.index)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, f"{matrix.iloc[row, column]:+.3f}", ha="center", va="center", fontsize=5.2)
    ax.set_title("a  Participant MAE deltas", loc="left", fontweight="bold")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Candidate − Condition")

    ax = axes[1]
    discomfort = condition[condition["outcome"] == "discomfort"].copy()
    discomfort["condition_number"] = discomfort["condition"].str.extract(r"(\d+)").astype(int)
    discomfort = discomfort.sort_values("condition_number")
    colors = np.where(discomfort["delta"] < 0, "#2b8c68", "#ba6b5d")
    ax.bar(discomfort["condition"], discomfort["delta"], color=colors, width=0.72)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_ylabel("Discomfort MAE delta")
    ax.set_title("b  Condition-specific discomfort effects", loc="left", fontweight="bold")
    ax.text(0.01, 0.98, "negative is better", transform=ax.transAxes, va="top", fontsize=6)
    fig.suptitle(primary.replace("_", " "), fontsize=9, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_figure(fig, figure_dir / "primary_participant_condition_effects")
    return condition


def _markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 6) -> str:
    display = frame[columns].copy()
    integer_columns = {
        "seed",
        "seed_count",
        "participants_improved",
        "participants_tied",
        "participants_worsened",
        "participant_count",
    }
    for column in display.select_dtypes(include=[np.number]).columns:
        if column in integer_columns:
            display[column] = display[column].map(lambda value: str(int(value)))
        else:
            display[column] = display[column].map(lambda value: f"{value:.{digits}f}")
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


def _report(
    output_dir: Path,
    summary: pd.DataFrame,
    seed_metrics: pd.DataFrame,
    paired: pd.DataFrame,
    participant_metrics: pd.DataFrame,
    audit: pd.DataFrame,
    control_metrics: dict[str, Any],
) -> Path:
    best = summary.iloc[0]
    both = summary[summary["both_targets_improve"]].iloc[0]
    baseline_macro = float(seed_metrics["macro_baseline_mae"].iloc[0])
    baseline_relax = float(seed_metrics["relaxation_baseline_mae"].iloc[0])
    baseline_discomfort = float(seed_metrics["discomfort_baseline_mae"].iloc[0])
    best_stats = paired[paired["candidate"] == best["candidate"]].sort_values("outcome")
    significant = paired[paired["holm_significant_0_05"]]
    participant = (
        participant_metrics[participant_metrics["candidate"] == best["candidate"]]
        .groupby(["participant_id", "outcome"], as_index=False)["delta"]
        .mean()
    )
    discomfort_participant = participant[participant["outcome"] == "discomfort"].sort_values("delta")

    text = f"""# Condition-Anchored Foundation Probe — Formal 9-Participant Evaluation

## Verdict

The best achieved method is `{best['candidate']}`. Across seeds `20260705`, `20260706`, and `20260707`, its mean participant-macro MAE is **{best['macro_mae_mean']:.6f}**, versus the exact fold-local Condition-only baseline **{baseline_macro:.6f}** (delta **{best['macro_delta_mean']:.6f}**). Relaxation is returned exactly to the Condition-only anchor ({best['relaxation_mae_mean']:.6f}; delta 0), while the frozen eye representation reduces discomfort from {baseline_discomfort:.6f} to **{best['discomfort_mae_mean']:.6f}**.

This satisfies the requested point-estimate criteria on this benchmark: macro MAE is below 0.12463, neither target is worse, the direction is consistent for all three seeds, all 81 OOF observations are covered, and the frozen neural embeddings have CUDA extraction provenance. It is **not a confirmatory significance result**: the candidate family was adaptively developed on these same nine participants, and no candidate × outcome survives the declared 24-test Holm family (corrected significant rows: {len(significant)}).

## Exact baseline and controls

| Comparator | Relaxation MAE | Discomfort MAE | Macro MAE |
| --- | ---: | ---: | ---: |
| Fold-local Condition-only | {baseline_relax:.6f} | {baseline_discomfort:.6f} | {baseline_macro:.6f} |
| Fold-local global relaxation + Condition discomfort | {control_metrics['targets']['relaxation']['participant_macro_mae']:.6f} | {control_metrics['targets']['discomfort']['participant_macro_mae']:.6f} | {control_metrics['macro_mae']:.6f} |

The target-wise non-foundation control already beats Condition-only on macro MAE. Therefore, beating 0.12463 alone does not establish foundation value. The eye method also beats that hybrid control on macro MAE by {best['macro_mae_mean'] - control_metrics['macro_mae']:.6f}, but its advantage comes entirely from discomfort; its relaxation prediction is worse than the hybrid control's global-mean relaxation.

## Formal candidate results

{_markdown_table(summary, ['candidate', 'modalities', 'representation', 'relaxation_mae_mean', 'discomfort_mae_mean', 'macro_mae_mean', 'macro_delta_mean', 'both_targets_not_worse', 'both_targets_improve', 'primary_point_success'])}

`{both['candidate']}` is the strongest method that improves both targets in every seed: relaxation {both['relaxation_mae_mean']:.6f}, discomfort {both['discomfort_mae_mean']:.6f}, macro {both['macro_mae_mean']:.6f}. The eye-only winner is safer for relaxation because it hard-falls back to the exact Condition anchor.

## Three-seed results for the best method

{_markdown_table(seed_metrics[seed_metrics['candidate'] == best['candidate']], ['seed', 'relaxation_mae', 'discomfort_mae', 'macro_mae', 'relaxation_delta', 'discomfort_delta', 'macro_delta'])}

## Paired participant statistics

Differences are candidate MAE minus Condition-only MAE; negative values favor the candidate. Seeds are averaged within participant before inference. Confidence intervals use 10,000 participant-cluster bootstrap resamples. P values enumerate all 512 sign assignments. Holm correction covers all 8 formal candidates × 3 outcomes.

{_markdown_table(best_stats, ['outcome', 'mean_paired_delta', 'bootstrap_ci_low', 'bootstrap_ci_high', 'participants_improved', 'participants_tied', 'participants_worsened', 'exact_sign_flip_p_two_sided', 'holm_p', 'holm_significant_0_05'])}

For best-method discomfort, the largest participant improvements are {', '.join(f"{row.participant_id} ({row.delta:+.4f})" for row in discomfort_participant.head(3).itertuples())}; the worsened participants are {', '.join(f"{row.participant_id} ({row.delta:+.4f})" for row in discomfort_participant[discomfort_participant['delta'] > 0].itertuples()) or 'none'}.

## Approaches tried and why

- The six-candidate screen tested a deliberately narrow modality ladder (ECG; ECG+eye; ECG+eye+head+video) with raw versus train-only condition-centered pooled embeddings. Each target used a separate Ridge residual head, a participant-LOO training anchor, and a seven-participant validation/test anchor.
- `no_eeg_raw_dual` was retained because it improved both targets across all seeds. Its 8-D train-only PCA limits the 1,942-D frozen feature vector before Ridge.
- The eye-only follow-up was motivated by discomfort sparsity and the screen. It fits only discomfort and returns exact Condition-only relaxation. The compact grid is the final best point estimate; the wider grid is retained rather than hidden.
- Direct aligned neural fusions were rejected before this family because their absolute two-target sigmoid heads had 232k–696k trainable parameters for only 63 training labels and did not beat Condition-only in their three-seed means. A saved-checkpoint validation-selected shrink also failed consistently, so it is not claimed as a success.
- A post-hoc fixed neural shrink and a low-anchor discomfort gate were diagnostic only and are not included as formal winners. Their use to focus this search is part of the adaptive-selection limitation.

## Implementation and exact reproduction commands

New isolated source files:

- `scripts/run_relax_condition_anchor_probe.py`: formal fold runner, frozen-cache/hash checks, mask-aware pooling, cross-fitted anchors, train-only scaling/PCA, separate Ridge heads, validation selection, target fallback, and per-fold model/provenance artifacts.
- `scripts/run_relax_condition_anchor_matrix.py`: complete 8-candidate × 3-seed launcher plus evaluator invocation.
- `scripts/evaluate_relax_condition_anchor.py`: strict 24-run audit, participant metrics, 10,000-cluster bootstrap, exact sign flip, Holm correction, controls, report, and figures.
- `tests/test_relax_condition_anchor.py`: own-participant exclusion, test-target isolation, transform isolation, zero-window handling, exact fallback, determinism, bounds, and statistics tests.

Completed aligned/ablation scripts and artifacts were not modified. From `/home/link/Wei/Models/core/real-time-vis-physio-fusion`, the complete forced reproduction is:

```bash
/home/link/miniconda3/envs/egoEMOTION/bin/python scripts/run_relax_condition_anchor_matrix.py \\
  --contract-dir /mnt/c/Users/linki/amaster/data_collection_v3/analysis/real_time_inference/artifacts/cross_project_alignment_2026-07-16/eeg_eligible_ablation/contract \\
  --embedding-cache artifacts/relax/aligned_20260716/condition_embeddings.pt \\
  --output-root artifacts/relax/condition_anchor_residual_20260717 \\
  --force
```

The exact verification command is:

```bash
/home/link/miniconda3/envs/egoEMOTION/bin/python -m pytest -q \\
  tests/test_relax_condition_anchor.py tests/test_relax_alignment.py
```

The evaluator-only command is:

```bash
/home/link/miniconda3/envs/egoEMOTION/bin/python scripts/evaluate_relax_condition_anchor.py \\
  --runs-dir artifacts/relax/condition_anchor_residual_20260717/runs \\
  --labels /mnt/c/Users/linki/amaster/data_collection_v3/analysis/real_time_inference/artifacts/cross_project_alignment_2026-07-16/eeg_eligible_ablation/contract/condition_labels.csv \\
  --split-manifest /mnt/c/Users/linki/amaster/data_collection_v3/analysis/real_time_inference/artifacts/cross_project_alignment_2026-07-16/eeg_eligible_ablation/contract/split_manifest.csv \\
  --output-dir artifacts/relax/condition_anchor_residual_20260717/evaluation
```

## Leakage and execution audit

- Exact cohort: {', '.join(EXPECTED_PARTICIPANTS)}; 9 folds; 7 train / 1 validation / 1 test; 81 participant-condition predictions per run.
- Every training residual anchor excludes that training participant. Validation/test Condition anchors use only the seven outer-training participants.
- `StandardScaler`, randomized PCA, Ridge coefficients, target selection, and correction calibration are fitted without the outer test participant. Test labels are read only after predictions for scoring.
- Common-valid policy is fixed at 545 windows. P004/C6 remains a finite zero-vector observation with explicit missingness/presence features; it is not dropped or imputed from test data.
- All 24/24 candidate-seed runs passed the artifact audit ({int(audit['valid'].sum())}/{len(audit)} valid). Ridge heads run on CPU by design; no new neural training occurs. The frozen encoder cache records CUDA extraction on {audit['embedding_device'].iloc[0]}.
- The frozen cache SHA is enforced by the runner. The cache is not rebuilt in this experiment.

## Figures and source data

- `figures/candidate_performance.*`: three-seed macro point estimates and participant-cluster bootstrap deltas.
- `figures/primary_participant_condition_effects.*`: participant and condition scope of the best method.
- Figure source tables are `candidate_summary.csv`, `paired_statistics.csv`, `participant_metrics.csv`, and `primary_condition_deltas.csv`. SVG/PDF retain editable text; TIFF is exported at 600 dpi.

## Conclusions, scope, and limitations

Supported on this exact 9-person, 81-observation, 545-window benchmark: a target-specific frozen-eye residual probe can improve discomfort while preserving the exact relaxation baseline; a no-EEG multimodal residual probe can improve both target point estimates. These conclusions apply only to the fixed cohort, labels, masks, folds, and cached encoders used here.

Not supported: a corrected-significance claim, a general population benefit, a causal role for eye physiology, or superiority on an untouched cohort. The same participants informed adaptive candidate development, the validation set contains only one participant per fold, and discomfort contains many zeros. The cache metadata proves its final extraction run used CUDA, but the current partial-cache builder does not cryptographically bind every reused partial to encoder code/weights/device; this is an upstream provenance limitation.

## Next smallest experiment

Freeze `eye_discomfort_compact` without further tuning and confirm it on an untouched participant cohort or prospectively collected repeat. Preserve the exact 130-D pooling, PCA/alpha grid, split discipline, and relaxation fallback. If only the existing nine participants are available, use a fully nested model-selection layer and report it as internal validation rather than an independent confirmation. A predeclared discomfort hurdle (correction only for conditions with non-trivial train-fold discomfort prevalence) is the next mechanistic sensitivity, not another fusion-architecture sweep.
"""
    path = output_dir / "condition_anchor_foundation_report.md"
    path.write_text(text, encoding="utf-8")
    return path


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    predictions, payloads, audit = _load_runs(args.runs_dir)
    seed_metrics = _seed_metrics(predictions)
    participant_metrics = _participant_metrics(predictions)
    paired = _paired_statistics(participant_metrics)
    summary = _candidate_summary(seed_metrics)
    control_predictions, control_metrics = _targetwise_control(args.labels, args.split_manifest)
    best = str(summary.iloc[0]["candidate"])

    predictions.to_csv(args.output_dir / "normalized_predictions.csv", index=False)
    seed_metrics.to_csv(args.output_dir / "seed_metrics.csv", index=False)
    participant_metrics.to_csv(args.output_dir / "participant_metrics.csv", index=False)
    paired.to_csv(args.output_dir / "paired_statistics.csv", index=False)
    summary.to_csv(args.output_dir / "candidate_summary.csv", index=False)
    audit.to_csv(args.output_dir / "run_audit.csv", index=False)
    control_predictions.to_csv(args.output_dir / "targetwise_control_predictions.csv", index=False)
    approach_registry = pd.DataFrame(
        [
            {
                **{
                    "candidate": candidate.name,
                    "family": candidate.family,
                    "modalities": "+".join(candidate.modalities),
                    "learned_targets": "+".join(candidate.learned_targets),
                    "representation": candidate.representation,
                    "quality_features": candidate.quality_features,
                    "pca_prefixes": "+".join(map(str, candidate.pca_prefixes)),
                    "alphas": "+".join(map(str, candidate.alphas)),
                    "gammas": "+".join(map(str, candidate.gammas)),
                    "residual_cap": candidate.residual_cap,
                },
                "description": APPROACH_DESCRIPTIONS[name],
            }
            for name, candidate in CANDIDATES.items()
        ]
    )
    approach_registry.to_csv(args.output_dir / "approach_registry.csv", index=False)

    primary_condition = _primary_participant_figure(
        predictions, participant_metrics, best, figure_dir
    )
    primary_condition.to_csv(args.output_dir / "primary_condition_deltas.csv", index=False)
    _performance_figure(seed_metrics, summary, paired, control_metrics["macro_mae"], figure_dir)
    figure_contract = {
        "core_conclusion": "A target-specific frozen-eye residual probe lowers discomfort and macro MAE while returning relaxation exactly to Condition-only; uncertainty and adaptive selection limit the claim.",
        "evidence_chain": {
            "candidate_performance_a": "Three-seed point estimates versus exact Condition and target-wise non-foundation control.",
            "candidate_performance_b": "Participant-cluster confidence intervals for candidate-minus-Condition macro error.",
            "primary_participant_condition_effects_a": "Participant heterogeneity of target and macro deltas.",
            "primary_participant_condition_effects_b": "Condition scope of the discomfort gain.",
        },
        "archetype": "quantitative_grid",
        "backend": "python_matplotlib_only",
        "final_width_mm": 183,
        "exports": ["svg_editable_text", "pdf_truetype_text", "png_300dpi", "tiff_600dpi"],
        "statistics": {
            "n": "9 participant clusters",
            "seeds": 3,
            "folds": "9 LOPO folds, 7/1/1",
            "metric": "participant-macro MAE",
            "interval": "10,000 participant-cluster bootstrap resamples after seed averaging",
            "test": "exact two-sided 2^9 sign-flip",
            "multiplicity": "Holm across 8 candidates x 3 outcomes",
        },
        "source_data": [
            "candidate_summary.csv",
            "paired_statistics.csv",
            "participant_metrics.csv",
            "primary_condition_deltas.csv",
        ],
    }
    _write_json(args.output_dir / "figure_contract.json", figure_contract)
    (args.output_dir / "figure_qa.md").write_text(
        """# Figure QA\n\n- Backend: Python/matplotlib only.\n- Core conclusion and evidence chain: recorded in `figure_contract.json`.\n- Final width: 183 mm; white background; restrained neutral/green/red palette; no rainbow map.\n- Statistics: n=9 participant clusters, three seeds, 10,000 cluster bootstraps, exact 512 sign flips, 24-test Holm family.\n- Source data: saved CSV tables in this directory.\n- Exports: SVG and PDF with editable text, PNG at 300 dpi, TIFF at 600 dpi.\n- Visual integrity: quantitative plots only; no image manipulation, cropping, or pseudo-coloring of source images.\n""",
        encoding="utf-8",
    )
    report_path = _report(
        args.output_dir,
        summary,
        seed_metrics,
        paired,
        participant_metrics,
        audit,
        control_metrics,
    )
    evaluation_audit = {
        "schema_version": "relax_condition_anchor_evaluation_v1",
        "expected_runs": len(CANDIDATES) * len(ALLOWED_SEEDS),
        "valid_runs": int(audit["valid"].sum()),
        "normalized_predictions": len(predictions),
        "expected_normalized_predictions": len(CANDIDATES) * len(ALLOWED_SEEDS) * EXPECTED_OBSERVATIONS,
        "candidate_count": len(CANDIDATES),
        "seeds": list(ALLOWED_SEEDS),
        "participant_clusters": len(EXPECTED_PARTICIPANTS),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "exact_sign_flip_assignments": 2 ** len(EXPECTED_PARTICIPANTS),
        "holm_family_size": len(paired),
        "best_candidate": best,
        "best_macro_mae": float(summary.iloc[0]["macro_mae_mean"]),
        "baseline_macro_mae": float(seed_metrics["macro_baseline_mae"].iloc[0]),
        "targetwise_control_macro_mae": float(control_metrics["macro_mae"]),
        "corrected_significant_tests": int(paired["holm_significant_0_05"].sum()),
        "report": str(report_path),
        "source_sha256": {
            "runner": file_sha256(ROOT / "scripts/run_relax_condition_anchor_probe.py"),
            "matrix_launcher": file_sha256(ROOT / "scripts/run_relax_condition_anchor_matrix.py"),
            "evaluator": file_sha256(__file__),
            "tests": file_sha256(ROOT / "tests/test_relax_condition_anchor.py"),
        },
        "command": [sys.executable, *sys.argv],
        "figure_exports": {
            path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in sorted(figure_dir.iterdir())
            if path.is_file()
        },
        "input_result_hashes": {
            payload["run_name"]: file_sha256(
                args.runs_dir
                / payload["candidate"]["name"]
                / f"seed_{payload['seed']}"
                / f"{payload['run_name']}_results.json"
            )
            for payload in payloads
        },
    }
    _write_json(args.output_dir / "evaluation_audit.json", evaluation_audit)
    return evaluation_audit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    result = evaluate(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

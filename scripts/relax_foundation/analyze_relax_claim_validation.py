#!/usr/bin/env python
"""Analyze Relax claim-validation runs and write paired statistics."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mac.data.relax_foundation import RelaxHardFailure, write_json  # noqa: E402
from mac.tasks.relaxation import compute_relax_regression_metrics  # noqa: E402


KEY_COLUMNS = ["participant_id", "condition"]
TARGET_COLUMNS = ["relaxation_true", "discomfort_true"]
PRED_COLUMNS = ["relaxation_pred", "discomfort_pred"]


def _macro_abs_error(frame: pd.DataFrame) -> pd.Series:
    return (
        (frame["relaxation_pred"] - frame["relaxation_true"]).abs()
        + (frame["discomfort_pred"] - frame["discomfort_true"]).abs()
    ) / 2.0


def assert_prediction_alignment(left: pd.DataFrame, right: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    for name, frame in (("left", left), ("right", right)):
        missing = sorted(set(KEY_COLUMNS + TARGET_COLUMNS + PRED_COLUMNS) - set(frame.columns))
        if missing:
            raise RelaxHardFailure(f"{name} predictions missing columns: {missing}")
        duplicates = frame.duplicated(KEY_COLUMNS)
        if duplicates.any():
            raise RelaxHardFailure(f"{name} predictions contain duplicate participant-Condition rows")

    left_sorted = left.sort_values(KEY_COLUMNS).reset_index(drop=True)
    right_sorted = right.sort_values(KEY_COLUMNS).reset_index(drop=True)
    if left_sorted[KEY_COLUMNS].to_dict("records") != right_sorted[KEY_COLUMNS].to_dict("records"):
        raise RelaxHardFailure("Prediction rows do not align between paired runs")
    for col in TARGET_COLUMNS:
        if not np.allclose(left_sorted[col].to_numpy(dtype=float), right_sorted[col].to_numpy(dtype=float)):
            raise RelaxHardFailure(f"Prediction true-label column {col} does not align between paired runs")
    return left_sorted, right_sorted


def participant_cluster_stats(
    frame: pd.DataFrame,
    *,
    value_col: str,
    rng_seed: int = 20260705,
    bootstrap_samples: int = 10000,
) -> dict[str, float]:
    missing = sorted({"participant_id", value_col} - set(frame.columns))
    if missing:
        raise RelaxHardFailure(f"Cluster statistics frame missing columns: {missing}")
    participant_values = frame.groupby("participant_id", sort=True)[value_col].mean().to_numpy(dtype=float)
    participant_values = participant_values[np.isfinite(participant_values)]
    if participant_values.size == 0:
        raise RelaxHardFailure("No finite participant-level deltas for cluster statistics")

    observed = float(np.mean(participant_values))
    rng = np.random.default_rng(rng_seed)
    boot = np.empty(bootstrap_samples, dtype=float)
    for idx in range(bootstrap_samples):
        sample = rng.choice(participant_values, size=participant_values.size, replace=True)
        boot[idx] = float(np.mean(sample))
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

    if participant_values.size <= 20:
        signs = np.array(np.meshgrid(*([[-1.0, 1.0]] * participant_values.size))).T.reshape(-1, participant_values.size)
        flipped = (signs * participant_values).mean(axis=1)
    else:
        signs = rng.choice(np.array([-1.0, 1.0]), size=(bootstrap_samples, participant_values.size), replace=True)
        flipped = (signs * participant_values).mean(axis=1)
    p_two_sided = float(np.mean(np.abs(flipped) >= abs(observed) - 1e-12))
    return {
        "n_participants": float(participant_values.size),
        "mean_delta": observed,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_signflip_two_sided": p_two_sided,
    }


def _load_result(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    args = data.get("args", {})
    predictions_csv = data.get("predictions_csv")
    if not predictions_csv:
        raise RelaxHardFailure(f"{path}: missing predictions_csv")
    pred_path = Path(predictions_csv)
    if not pred_path.is_absolute() and not pred_path.exists():
        pred_path = path.parent / pred_path
    if not pred_path.exists():
        raise RelaxHardFailure(f"{path}: predictions CSV is missing: {pred_path}")
    pred = pd.read_csv(pred_path)
    y_true = pred[["relaxation_true", "discomfort_true"]].to_numpy(dtype=float)
    y_pred = pred[["relaxation_pred", "discomfort_pred"]].to_numpy(dtype=float)
    metrics = compute_relax_regression_metrics(y_true, y_pred)
    return {
        "path": str(path),
        "args": args,
        "predictions": pred,
        "metrics": metrics,
        "cohort": args.get("cohort"),
        "baseline": args.get("baseline", "none"),
        "fusion": args.get("fusion"),
        "seed": args.get("seed"),
        "modalities": tuple(args.get("modalities", [])),
        "condition_control": args.get("condition_control", "none"),
    }


def _result_index(results: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    index: dict[tuple[Any, ...], dict[str, Any]] = {}
    for result in results:
        key = (
            result["cohort"],
            result["baseline"],
            result["fusion"],
            result["seed"],
            result["modalities"],
            result["condition_control"],
        )
        if key in index:
            raise RelaxHardFailure(f"Duplicate result key: {key}")
        index[key] = result
    return index


def _paired_delta_frame(left: dict[str, Any], right: dict[str, Any], *, value_col: str) -> pd.DataFrame:
    left_pred, right_pred = assert_prediction_alignment(left["predictions"], right["predictions"])
    frame = left_pred[KEY_COLUMNS].copy()
    frame["left_error"] = _macro_abs_error(left_pred)
    frame["right_error"] = _macro_abs_error(right_pred)
    frame[value_col] = frame["left_error"] - frame["right_error"]
    return frame


def analyze_results(results_dir: Path, report_dir: Path) -> dict[str, pd.DataFrame]:
    result_paths = sorted(results_dir.rglob("*_results.json"))
    if not result_paths:
        raise RelaxHardFailure(f"No Relax claim-validation result JSON files found under {results_dir}")
    results = [_load_result(path) for path in result_paths]
    index = _result_index(results)
    report_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for result in results:
        if result["baseline"] == "none" and result["cohort"] == "all_135" and result["condition_control"] == "none":
            baseline = index.get(("all_135", "condition", "early", result["seed"], result["modalities"], "none"))
            if baseline is None:
                baseline = next(
                    (
                        item
                        for item in results
                        if item["cohort"] == "all_135" and item["baseline"] == "condition"
                    ),
                    None,
                )
            delta = math.nan
            if baseline is not None:
                pair = _paired_delta_frame(result, baseline, value_col="delta_vs_condition")
                delta = float(pair["delta_vs_condition"].mean())
            rows.append(
                {
                    "cohort": result["cohort"],
                    "fusion": result["fusion"],
                    "seed": result["seed"],
                    "modalities": "+".join(result["modalities"]),
                    "condition_control": result["condition_control"],
                    "macro_mae": result["metrics"]["macro_mae"],
                    "delta_vs_condition": delta,
                }
            )
    seed_stability = pd.DataFrame(rows).sort_values(["condition_control", "fusion", "seed", "modalities"])

    eeg_rows = []
    for seed in sorted({item["seed"] for item in results if item["seed"] is not None}):
        for fusion in sorted({item["fusion"] for item in results if item["fusion"]}):
            with_key = ("eeg_eligible", "none", fusion, seed, FULL_MODALITIES := ("eeg", "ecg", "eye", "head", "video"), "none")
            without_key = ("eeg_eligible", "none", fusion, seed, ("ecg", "eye", "head", "video"), "none")
            if with_key not in index or without_key not in index:
                continue
            pair = _paired_delta_frame(index[without_key], index[with_key], value_col="delta_without_minus_with_eeg")
            stats = participant_cluster_stats(pair, value_col="delta_without_minus_with_eeg")
            eeg_rows.append(
                {
                    "fusion": fusion,
                    "seed": seed,
                    "mean_without_minus_with_eeg": stats["mean_delta"],
                    "ci_low": stats["ci_low"],
                    "ci_high": stats["ci_high"],
                    "p_signflip_two_sided": stats["p_signflip_two_sided"],
                    "n_participants": stats["n_participants"],
                }
            )
    eeg_stats = pd.DataFrame(eeg_rows).sort_values(["fusion", "seed"]) if eeg_rows else pd.DataFrame()

    video_rows = []
    for seed in sorted({item["seed"] for item in results if item["seed"] is not None}):
        for fusion in sorted({item["fusion"] for item in results if item["fusion"]}):
            full_key = ("all_135", "none", fusion, seed, ("eeg", "ecg", "eye", "head", "video"), "none")
            residual_key = (
                "all_135",
                "none",
                fusion,
                seed,
                ("eeg", "ecg", "eye", "head", "video"),
                "video_condition_residualized",
            )
            no_video_key = ("all_135", "none", fusion, seed, ("eeg", "ecg", "eye", "head"), "none")
            if full_key in index and residual_key in index:
                pair = _paired_delta_frame(index[residual_key], index[full_key], value_col="delta_residual_minus_full")
                stats = participant_cluster_stats(pair, value_col="delta_residual_minus_full")
                video_rows.append(
                    {
                        "fusion": fusion,
                        "seed": seed,
                        "comparison": "video_condition_residualized_minus_full",
                        "mean_delta": stats["mean_delta"],
                        "ci_low": stats["ci_low"],
                        "ci_high": stats["ci_high"],
                        "p_signflip_two_sided": stats["p_signflip_two_sided"],
                        "n_participants": stats["n_participants"],
                    }
                )
            if full_key in index and no_video_key in index:
                pair = _paired_delta_frame(index[no_video_key], index[full_key], value_col="delta_no_video_minus_full")
                stats = participant_cluster_stats(pair, value_col="delta_no_video_minus_full")
                video_rows.append(
                    {
                        "fusion": fusion,
                        "seed": seed,
                        "comparison": "without_video_minus_full",
                        "mean_delta": stats["mean_delta"],
                        "ci_low": stats["ci_low"],
                        "ci_high": stats["ci_high"],
                        "p_signflip_two_sided": stats["p_signflip_two_sided"],
                        "n_participants": stats["n_participants"],
                    }
                )
    video_stats = pd.DataFrame(video_rows).sort_values(["comparison", "fusion", "seed"]) if video_rows else pd.DataFrame()

    handcrafted_rows = []
    for cohort in ("all_135", "eeg_eligible"):
        handcrafted = next(
            (
                item
                for item in results
                if item["cohort"] == cohort and item["baseline"] == "relax_handcrafted_ridge_cv"
            ),
            None,
        )
        if handcrafted is None:
            continue
        for result in results:
            if result["cohort"] != cohort or result["baseline"] != "none" or result["condition_control"] != "none":
                continue
            pair = _paired_delta_frame(result, handcrafted, value_col="delta_foundation_minus_handcrafted")
            stats = participant_cluster_stats(pair, value_col="delta_foundation_minus_handcrafted")
            handcrafted_rows.append(
                {
                    "cohort": cohort,
                    "fusion": result["fusion"],
                    "seed": result["seed"],
                    "modalities": "+".join(result["modalities"]),
                    "foundation_macro_mae": result["metrics"]["macro_mae"],
                    "handcrafted_macro_mae": handcrafted["metrics"]["macro_mae"],
                    "mean_delta_foundation_minus_handcrafted": stats["mean_delta"],
                    "ci_low": stats["ci_low"],
                    "ci_high": stats["ci_high"],
                    "p_signflip_two_sided": stats["p_signflip_two_sided"],
                }
            )
    handcrafted_stats = (
        pd.DataFrame(handcrafted_rows).sort_values(["cohort", "fusion", "seed", "modalities"])
        if handcrafted_rows
        else pd.DataFrame()
    )

    tables = {
        "seed_stability": seed_stability,
        "paired_eeg_stats": eeg_stats,
        "video_condition_control": video_stats,
        "handcrafted_comparison": handcrafted_stats,
    }
    aggregates = _build_aggregate_tables(tables)
    for name, table in tables.items():
        table.to_csv(report_dir / f"{name}.csv", index=False)
    for name, table in aggregates.items():
        table.to_csv(report_dir / f"{name}.csv", index=False)

    decision = _claim_decision(aggregates)
    write_json(report_dir / "claim_validation_summary.json", {"decision": decision})
    (report_dir / "claim_validation_report.md").write_text(
        _render_report(decision, tables, aggregates, results_dir),
        encoding="utf-8",
    )
    return tables


def _build_aggregate_tables(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    seed = tables["seed_stability"]
    full = seed[
        (seed["modalities"] == "eeg+ecg+eye+head+video")
        & (seed["condition_control"] == "none")
    ]
    seed_full_aggregate = (
        full.groupby("fusion", as_index=False)
        .agg(
            macro_mae_mean=("macro_mae", "mean"),
            macro_mae_std=("macro_mae", "std"),
            delta_vs_condition_mean=("delta_vs_condition", "mean"),
            delta_vs_condition_min=("delta_vs_condition", "min"),
            delta_vs_condition_max=("delta_vs_condition", "max"),
            n_seeds=("seed", "count"),
        )
        .sort_values("macro_mae_mean")
        if not full.empty
        else pd.DataFrame()
    )

    eeg = tables["paired_eeg_stats"]
    eeg_aggregate = (
        eeg.groupby("fusion", as_index=False)
        .agg(
            eeg_help_mean=("mean_without_minus_with_eeg", "mean"),
            eeg_help_min=("mean_without_minus_with_eeg", "min"),
            eeg_help_max=("mean_without_minus_with_eeg", "max"),
            n_seeds=("seed", "count"),
        )
        .sort_values("eeg_help_mean", ascending=False)
        if not eeg.empty
        else pd.DataFrame()
    )

    video = tables["video_condition_control"]
    video_aggregate = (
        video.groupby(["comparison", "fusion"], as_index=False)
        .agg(
            mean_delta=("mean_delta", "mean"),
            min_delta=("mean_delta", "min"),
            max_delta=("mean_delta", "max"),
            n_seeds=("seed", "count"),
        )
        .sort_values(["comparison", "mean_delta"], ascending=[True, False])
        if not video.empty
        else pd.DataFrame()
    )

    hand = tables["handcrafted_comparison"]
    hand_full = (
        hand[hand["modalities"] == "eeg+ecg+eye+head+video"]
        .groupby(["cohort", "fusion"], as_index=False)
        .agg(
            foundation_minus_handcrafted_mean=("mean_delta_foundation_minus_handcrafted", "mean"),
            foundation_minus_handcrafted_min=("mean_delta_foundation_minus_handcrafted", "min"),
            foundation_minus_handcrafted_max=("mean_delta_foundation_minus_handcrafted", "max"),
            n_seeds=("seed", "count"),
        )
        .sort_values(["cohort", "foundation_minus_handcrafted_mean"])
        if not hand.empty
        else pd.DataFrame()
    )
    return {
        "seed_stability_full_aggregate": seed_full_aggregate,
        "eeg_effect_aggregate": eeg_aggregate,
        "video_control_aggregate": video_aggregate,
        "handcrafted_full_aggregate": hand_full,
    }


def _claim_decision(aggregates: dict[str, pd.DataFrame]) -> str:
    seed = aggregates["seed_stability_full_aggregate"]
    eeg = aggregates["eeg_effect_aggregate"]
    video = aggregates["video_control_aggregate"]
    hand = aggregates["handcrafted_full_aggregate"]

    stable_vs_condition = (
        not seed.empty
        and ((seed["n_seeds"] >= 3) & (seed["delta_vs_condition_max"] < 0.0)).any()
    )
    eeg_consistent_positive = (
        not eeg.empty
        and ((eeg["n_seeds"] >= 3) & (eeg["eeg_help_min"] > 0.0)).any()
    )
    no_video = video[video["comparison"] == "without_video_minus_full"] if not video.empty else pd.DataFrame()
    residualized = (
        video[video["comparison"] == "video_condition_residualized_minus_full"] if not video.empty else pd.DataFrame()
    )
    video_consistent = (
        not no_video.empty
        and ((no_video["n_seeds"] >= 3) & (no_video["min_delta"] > 0.0)).any()
        and not residualized.empty
    )
    all135_handcrafted_beaten = (
        not hand.empty
        and (
            (hand["cohort"] == "all_135")
            & (hand["n_seeds"] >= 3)
            & (hand["foundation_minus_handcrafted_max"] < 0.0)
        ).any()
    )
    eeg_cohort_handcrafted_beaten = (
        not hand.empty
        and (
            (hand["cohort"] == "eeg_eligible")
            & (hand["n_seeds"] >= 3)
            & (hand["foundation_minus_handcrafted_max"] < 0.0)
        ).any()
    )

    if stable_vs_condition and eeg_consistent_positive and video_consistent and all135_handcrafted_beaten:
        return "supported_under_predefined_checks"
    if stable_vs_condition and video_consistent and (eeg_consistent_positive or eeg_cohort_handcrafted_beaten):
        return "partially_supported_all135_condition_stable_but_handcrafted_not_beaten"
    if stable_vs_condition:
        return "limited_support_condition_baseline_only"
    return "not_supported_by_current_runs"


def _format_top(table: pd.DataFrame, columns: list[str], n: int = 12) -> str:
    if table.empty:
        return "No rows available."
    view = table[columns].head(n)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for record in view.to_dict("records"):
        values = []
        for column in columns:
            value = record[column]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def _render_report(
    decision: str,
    tables: dict[str, pd.DataFrame],
    aggregates: dict[str, pd.DataFrame],
    results_dir: Path,
) -> str:
    return "\n".join(
        [
            "# Relax Claim-Validation Report",
            "",
            f"Results directory: `{results_dir}`",
            "",
            f"Predefined decision: `{decision}`",
            "",
            "## Full all_135 Seed Stability",
            "",
            "Negative `delta_vs_condition` means the foundation probe improved over the train-fold Condition baseline.",
            "",
            _format_top(
                aggregates["seed_stability_full_aggregate"],
                [
                    "fusion",
                    "macro_mae_mean",
                    "macro_mae_std",
                    "delta_vs_condition_mean",
                    "delta_vs_condition_min",
                    "delta_vs_condition_max",
                    "n_seeds",
                ],
            ),
            "",
            "## Best Seed-Level Runs",
            "",
            _format_top(
                tables["seed_stability"].sort_values("macro_mae") if not tables["seed_stability"].empty else tables["seed_stability"],
                ["fusion", "seed", "modalities", "condition_control", "macro_mae", "delta_vs_condition"],
            ),
            "",
            "## EEG Paired Comparison",
            "",
            "Positive `mean_without_minus_with_eeg` means adding EEG reduced macro absolute error.",
            "",
            _format_top(
                aggregates["eeg_effect_aggregate"],
                ["fusion", "eeg_help_mean", "eeg_help_min", "eeg_help_max", "n_seeds"],
            ),
            "",
            "Seed-level paired EEG rows:",
            "",
            _format_top(
                tables["paired_eeg_stats"].sort_values("mean_without_minus_with_eeg", ascending=False)
                if not tables["paired_eeg_stats"].empty
                else tables["paired_eeg_stats"],
                ["fusion", "seed", "mean_without_minus_with_eeg", "ci_low", "ci_high", "p_signflip_two_sided"],
            ),
            "",
            "## Video Condition Control",
            "",
            "Positive deltas mean the controlled/no-video variant is worse than the full model.",
            "",
            _format_top(
                aggregates["video_control_aggregate"],
                ["comparison", "fusion", "mean_delta", "min_delta", "max_delta", "n_seeds"],
            ),
            "",
            "Seed-level video-control rows:",
            "",
            _format_top(
                tables["video_condition_control"].sort_values("mean_delta", ascending=False)
                if not tables["video_condition_control"].empty
                else tables["video_condition_control"],
                ["comparison", "fusion", "seed", "mean_delta", "ci_low", "ci_high", "p_signflip_two_sided"],
            ),
            "",
            "## Foundation vs Handcrafted",
            "",
            "Negative `mean_delta_foundation_minus_handcrafted` means the foundation probe is better.",
            "",
            _format_top(
                aggregates["handcrafted_full_aggregate"],
                [
                    "cohort",
                    "fusion",
                    "foundation_minus_handcrafted_mean",
                    "foundation_minus_handcrafted_min",
                    "foundation_minus_handcrafted_max",
                    "n_seeds",
                ],
            ),
            "",
            "Best seed-level foundation-vs-handcrafted rows:",
            "",
            _format_top(
                tables["handcrafted_comparison"].sort_values("mean_delta_foundation_minus_handcrafted")
                if not tables["handcrafted_comparison"].empty
                else tables["handcrafted_comparison"],
                [
                    "cohort",
                    "fusion",
                    "seed",
                    "modalities",
                    "mean_delta_foundation_minus_handcrafted",
                    "ci_low",
                    "ci_high",
                ],
            ),
            "",
            "## Interpretation",
            "",
            f"The predefined aggregate decision is `{decision}`. The tables above are generated from the current run; no fusion-specific conclusion is hard-coded from an earlier mapping.",
            "",
            "Treat an effect as stable across the registered seeds only when the corresponding aggregate minimum and maximum keep the same direction. Participant-clustered intervals that cross zero are reported as trends.",
            "",
            "Interpret EEG effects primarily on `eeg_eligible`: six `all_135` participants have EEG disabled by the locked QC audit, so their masks dilute an all-participant EEG comparison.",
            "",
            "## Limitations",
            "",
            "The report uses participant-clustered summaries and paired rows where possible. "
            "It should not be used to claim full support unless all configured jobs completed, no hard-failure files remain, and the handcrafted comparison is favorable on the target cohort.",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="logs/relax_claim_validation")
    parser.add_argument("--report-dir", default="reports/relax_claim_validation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        analyze_results(Path(args.results_dir), Path(args.report_dir))
        return 0
    except RelaxHardFailure as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

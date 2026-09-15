#!/usr/bin/env python
"""Analyze Relax attention-video replacement experiments."""
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

from scripts.relax_foundation.analyze_relax_claim_validation import (  # noqa: E402
    KEY_COLUMNS,
    assert_prediction_alignment,
    participant_cluster_stats,
)
from scripts.relax_foundation.run_relax_attention_video_experiments import VARIANT_MODALITIES  # noqa: E402
from src.data.relax_foundation import RelaxHardFailure, write_json  # noqa: E402
from src.tasks.relaxation import compute_relax_regression_metrics  # noqa: E402


MODALITY_VARIANTS = {modalities: variant for variant, modalities in VARIANT_MODALITIES.items()}


def _macro_abs_error(frame: pd.DataFrame) -> pd.Series:
    return (
        (frame["relaxation_pred"] - frame["relaxation_true"]).abs()
        + (frame["discomfort_pred"] - frame["discomfort_true"]).abs()
    ) / 2.0


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
    modalities = tuple(args.get("modalities", []))
    return {
        "path": str(path),
        "args": args,
        "predictions": pred,
        "metrics": compute_relax_regression_metrics(y_true, y_pred),
        "cohort": args.get("cohort"),
        "baseline": args.get("baseline", "none"),
        "fusion": args.get("fusion"),
        "seed": args.get("seed"),
        "modalities": modalities,
        "variant": MODALITY_VARIANTS.get(modalities, "baseline" if args.get("baseline", "none") != "none" else "unknown"),
    }


def _result_index(results: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    index: dict[tuple[Any, ...], dict[str, Any]] = {}
    for result in results:
        key = (
            result["cohort"],
            result["baseline"],
            result["fusion"],
            result["seed"],
            result["variant"],
        )
        if key in index:
            raise RelaxHardFailure(f"Duplicate attention-video result key: {key}")
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
    hard_failures = sorted(results_dir.rglob("hard_failure.json"))
    if hard_failures:
        raise RelaxHardFailure(
            f"Attention-video suite has hard-failure files; inspect before analysis: {hard_failures[:5]}"
        )
    result_paths = sorted(results_dir.rglob("*_results.json"))
    if not result_paths:
        raise RelaxHardFailure(f"No Relax attention-video result JSON files found under {results_dir}")
    results = [_load_result(path) for path in result_paths]
    index = _result_index(results)
    report_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    for result in results:
        metric_rows.append(
            {
                "cohort": result["cohort"],
                "baseline": result["baseline"],
                "fusion": result["fusion"],
                "seed": result["seed"],
                "variant": result["variant"],
                "modalities": "+".join(result["modalities"]),
                **result["metrics"],
            }
        )
    metrics = pd.DataFrame(metric_rows).sort_values(["cohort", "baseline", "fusion", "seed", "variant"])

    replacement_rows = []
    for cohort in sorted({item["cohort"] for item in results if item["cohort"]}):
        for seed in sorted({item["seed"] for item in results if item["seed"] is not None}):
            for fusion in sorted({item["fusion"] for item in results if item["fusion"]}):
                full_key = (cohort, "none", fusion, seed, "full_eye_video")
                attention_key = (cohort, "none", fusion, seed, "attention_replacement")
                no_visual_key = (cohort, "none", fusion, seed, "no_visual")
                visual_pair_key = (cohort, "none", fusion, seed, "visual_pair_only")
                attention_only_key = (cohort, "none", fusion, seed, "attention_only")
                comparisons = [
                    (
                        "attention_replacement_minus_full_eye_video",
                        attention_key,
                        full_key,
                        "delta_attention_minus_full",
                    ),
                    (
                        "attention_replacement_minus_no_visual",
                        attention_key,
                        no_visual_key,
                        "delta_attention_minus_no_visual",
                    ),
                    (
                        "attention_only_minus_visual_pair_only",
                        attention_only_key,
                        visual_pair_key,
                        "delta_attention_only_minus_visual_pair",
                    ),
                ]
                for comparison, left_key, right_key, value_col in comparisons:
                    if left_key not in index or right_key not in index:
                        continue
                    pair = _paired_delta_frame(index[left_key], index[right_key], value_col=value_col)
                    stats = participant_cluster_stats(pair, value_col=value_col)
                    replacement_rows.append(
                        {
                            "cohort": cohort,
                            "fusion": fusion,
                            "seed": seed,
                            "comparison": comparison,
                            "mean_delta": stats["mean_delta"],
                            "ci_low": stats["ci_low"],
                            "ci_high": stats["ci_high"],
                            "p_signflip_two_sided": stats["p_signflip_two_sided"],
                            "n_participants": stats["n_participants"],
                        }
                    )
    replacement = (
        pd.DataFrame(replacement_rows).sort_values(["cohort", "comparison", "fusion", "seed"])
        if replacement_rows
        else pd.DataFrame()
    )

    baseline_rows = []
    for result in results:
        if result["baseline"] != "none":
            continue
        for baseline_name in ("condition", "relax_handcrafted_ridge_cv"):
            baseline = next(
                (
                    item
                    for item in results
                    if item["cohort"] == result["cohort"] and item["baseline"] == baseline_name
                ),
                None,
            )
            if baseline is None:
                continue
            value_col = f"delta_{result['variant']}_minus_{baseline_name}"
            pair = _paired_delta_frame(result, baseline, value_col=value_col)
            stats = participant_cluster_stats(pair, value_col=value_col)
            baseline_rows.append(
                {
                    "cohort": result["cohort"],
                    "fusion": result["fusion"],
                    "seed": result["seed"],
                    "variant": result["variant"],
                    "baseline": baseline_name,
                    "mean_delta": stats["mean_delta"],
                    "ci_low": stats["ci_low"],
                    "ci_high": stats["ci_high"],
                    "p_signflip_two_sided": stats["p_signflip_two_sided"],
                    "n_participants": stats["n_participants"],
                }
            )
    baseline_comparison = (
        pd.DataFrame(baseline_rows).sort_values(["cohort", "baseline", "fusion", "seed", "variant"])
        if baseline_rows
        else pd.DataFrame()
    )

    aggregate = _build_aggregate(metrics, replacement, baseline_comparison)
    tables = {
        "attention_video_variant_metrics": metrics,
        "attention_video_replacement_pairs": replacement,
        "attention_video_baseline_comparison": baseline_comparison,
        "attention_video_aggregate": aggregate,
    }
    for name, table in tables.items():
        table.to_csv(report_dir / f"{name}.csv", index=False)
    write_json(report_dir / "attention_video_summary.json", {"decision": _decision(aggregate)})
    (report_dir / "attention_video_report.md").write_text(
        _render_report(tables, results_dir, decision=_decision(aggregate)),
        encoding="utf-8",
    )
    return tables


def _build_aggregate(
    metrics: pd.DataFrame,
    replacement: pd.DataFrame,
    baseline_comparison: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    if not metrics.empty:
        model_metrics = metrics[metrics["baseline"] == "none"]
        for record in (
            model_metrics.groupby(["cohort", "fusion", "variant"], as_index=False)
            .agg(
                macro_mae_mean=("macro_mae", "mean"),
                macro_mae_std=("macro_mae", "std"),
                macro_mae_min=("macro_mae", "min"),
                macro_mae_max=("macro_mae", "max"),
                n_seeds=("seed", "count"),
            )
            .to_dict("records")
        ):
            rows.append({"summary_type": "metric", "comparison": record.pop("variant"), **record})
    if not replacement.empty:
        for record in (
            replacement.groupby(["cohort", "fusion", "comparison"], as_index=False)
            .agg(
                mean_delta=("mean_delta", "mean"),
                min_delta=("mean_delta", "min"),
                max_delta=("mean_delta", "max"),
                n_seeds=("seed", "count"),
            )
            .to_dict("records")
        ):
            rows.append({"summary_type": "replacement_pair", **record})
    if not baseline_comparison.empty:
        for record in (
            baseline_comparison.groupby(["cohort", "fusion", "variant", "baseline"], as_index=False)
            .agg(
                mean_delta=("mean_delta", "mean"),
                min_delta=("mean_delta", "min"),
                max_delta=("mean_delta", "max"),
                n_seeds=("seed", "count"),
            )
            .to_dict("records")
        ):
            rows.append(
                {
                    "summary_type": "baseline_pair",
                    "comparison": f"{record.pop('variant')}_minus_{record.pop('baseline')}",
                    **record,
                }
            )
    return pd.DataFrame(rows)


def _decision(aggregate: pd.DataFrame) -> str:
    if aggregate.empty:
        return "not_analyzable_no_aggregate_rows"

    def cohort_supported(cohort: str) -> bool:
        replacement = aggregate[
            (aggregate["summary_type"] == "replacement_pair")
            & (aggregate["cohort"] == cohort)
            & (aggregate["comparison"] == "attention_replacement_minus_full_eye_video")
        ]
        no_visual = aggregate[
            (aggregate["summary_type"] == "replacement_pair")
            & (aggregate["cohort"] == cohort)
            & (aggregate["comparison"] == "attention_replacement_minus_no_visual")
        ]
        preserves = (
            not replacement.empty
            and ((replacement["n_seeds"] >= 3) & (replacement["max_delta"].abs() <= 0.02)).any()
        )
        beats_no_visual = (
            not no_visual.empty
            and ((no_visual["n_seeds"] >= 3) & (no_visual["max_delta"] < 0.0)).any()
        )
        return bool(preserves and beats_no_visual)

    def cohort_adds_visual_signal(cohort: str) -> bool:
        no_visual = aggregate[
            (aggregate["summary_type"] == "replacement_pair")
            & (aggregate["cohort"] == cohort)
            & (aggregate["comparison"] == "attention_replacement_minus_no_visual")
        ]
        return bool(
            not no_visual.empty
            and ((no_visual["n_seeds"] >= 3) & (no_visual["max_delta"] < 0.0)).any()
        )

    all135_supported = cohort_supported("all_135")
    eeg_supported = cohort_supported("eeg_eligible")
    if all135_supported and eeg_supported:
        return "attention_video_replacement_supported_on_all_predefined_cohorts"
    if eeg_supported and not all135_supported:
        return "partially_supported_on_eeg_eligible_not_all135"
    if all135_supported and not eeg_supported:
        return "partially_supported_on_all135_not_eeg_eligible"
    if cohort_adds_visual_signal("all_135") or cohort_adds_visual_signal("eeg_eligible"):
        return "attention_video_adds_visual_signal_but_does_not_replace_eye_video"
    return "attention_video_not_supported_as_replacement_by_current_runs"


def _format_top(table: pd.DataFrame, columns: list[str], n: int = 16) -> str:
    if table.empty:
        return "No rows available."
    view = table.copy()
    for column in columns:
        if column not in view.columns:
            view[column] = np.nan
    view = view[columns].head(n)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for record in view.to_dict("records"):
        values = []
        for column in columns:
            value = record[column]
            if isinstance(value, float) and math.isfinite(value):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def _render_report(tables: dict[str, pd.DataFrame], results_dir: Path, *, decision: str) -> str:
    metrics = tables["attention_video_variant_metrics"]
    replacement = tables["attention_video_replacement_pairs"]
    aggregate = tables["attention_video_aggregate"]
    return "\n".join(
        [
            "# Relax Attention-Video Replacement Report",
            "",
            f"Results directory: `{results_dir}`",
            "",
            f"Predefined decision: `{decision}`",
            "",
            "## Variant Metrics",
            "",
            _format_top(
                metrics.sort_values("macro_mae") if not metrics.empty else metrics,
                ["cohort", "fusion", "seed", "variant", "macro_mae", "relaxation_mae", "discomfort_mae"],
            ),
            "",
            "## Paired Replacement Tests",
            "",
            "Positive `mean_delta` means the left-hand variant in the comparison has higher macro absolute error.",
            "",
            _format_top(
                replacement.sort_values(["comparison", "mean_delta"]) if not replacement.empty else replacement,
                ["cohort", "comparison", "fusion", "seed", "mean_delta", "ci_low", "ci_high", "p_signflip_two_sided"],
            ),
            "",
            "## Aggregate Rows",
            "",
            _format_top(
                _sort_with_optional_columns(
                    aggregate,
                    ["summary_type", "comparison", "mean_delta"],
                ),
                ["summary_type", "cohort", "fusion", "comparison", "mean_delta", "min_delta", "max_delta", "n_seeds"],
            ),
            "",
            "## Interpretation Rules",
            "",
            "`attention_replacement_minus_full_eye_video` near zero means the derived attention video preserves the separate eye+video signal.",
            "",
            "`attention_replacement_minus_no_visual` below zero means the derived attention video adds predictive signal over removing visual modalities.",
            "",
            "`attention_only_minus_visual_pair_only` below zero means attention video alone outperforms eye+video alone.",
            "",
        ]
    )


def _sort_with_optional_columns(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if table.empty:
        return table
    sortable = table.copy()
    for column in columns:
        if column not in sortable.columns:
            sortable[column] = np.nan
    return sortable.sort_values(columns, na_position="last")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="logs/relax_attention_video")
    parser.add_argument("--report-dir", default="reports/relax_attention_video")
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

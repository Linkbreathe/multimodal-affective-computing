from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import (
    CONDITIONS,
    INFERENCE_UNIT,
    NEGLIGIBLE_HETEROGENEITY_SD,
    REPO_ROOT,
    REQUIRED_TABLES,
    TARGETS,
    WINDOW_FEATURE_UNIT,
    add_empty_ci,
    assert_estimate_tables_have_ci,
    assert_no_window_n_as_inference,
    ci,
    condition_sort_key,
    ensure_phase_dirs,
    forbidden_feature_columns,
    format_ci,
    format_number,
    participant_bootstrap,
    read_required_csv,
    runtime_qc_feature_columns,
    sha256_file,
    software_versions,
    write_csv,
    write_json,
    write_text,
)

from real_time_ml.config import load_config
from real_time_ml.modeling.condition_train import train_condition_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase A decision reanalysis")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser.parse_args()


def table_integrity(paths: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for name, spec in REQUIRED_TABLES.items():
        path = Path(spec["path"])
        frame = pd.read_csv(path)
        row_count, column_count = frame.shape
        rows.append(
            {
                "table_name": name,
                "path": str(path.relative_to(REPO_ROOT)),
                "row_count": row_count,
                "expected_rows": spec["rows"],
                "column_count": column_count,
                "expected_columns": spec["columns"],
                "shape_pass": bool(row_count == spec["rows"] and column_count == spec["columns"]),
                "sha256": sha256_file(path),
                "inference_unit": spec["inference_unit"],
            }
        )
    output = pd.DataFrame(rows)
    write_csv(paths["tables"] / "data_integrity.csv", output)
    if not output["shape_pass"].all():
        raise AssertionError("At least one required CSV shape changed")
    return output


def condition_completeness(labels: pd.DataFrame, paths: dict[str, Path]) -> pd.DataFrame:
    rows = []
    participants = sorted(labels["participant_id"].astype(str).unique())
    expected = set(CONDITIONS)
    for participant in participants:
        group = labels[labels["participant_id"].astype(str) == participant]
        conditions = group["condition"].astype(str).tolist()
        missing = sorted(expected - set(conditions), key=condition_sort_key)
        duplicate_rows = int(group.duplicated(["participant_id", "condition"]).sum())
        rows.append(
            {
                "participant_id": participant,
                "condition_count": int(group["condition"].nunique()),
                "missing_conditions": ";".join(missing),
                "duplicate_condition_rows": duplicate_rows,
                "complete": bool(not missing and duplicate_rows == 0),
                "inference_unit": INFERENCE_UNIT,
            }
        )
    output = pd.DataFrame(rows)
    write_csv(paths["tables"] / "condition_completeness.csv", output)
    if not output["complete"].all():
        raise AssertionError("Participant x condition completeness failed")
    return output


def target_distribution(labels: pd.DataFrame, paths: dict[str, Path], seed: int, replicates: int) -> pd.DataFrame:
    rows = []
    for target_index, target in enumerate(TARGETS):
        values = pd.to_numeric(labels[target], errors="coerce").to_numpy(dtype=float)
        summaries = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "min": float(np.min(values)),
            "q25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)),
            "q75": float(np.quantile(values, 0.75)),
            "max": float(np.max(values)),
            "floor_count": float(np.sum(np.isclose(values, 0.0))),
            "ceiling_count": float(np.sum(np.isclose(values, 1.0))),
            "floor_rate": float(np.mean(np.isclose(values, 0.0))),
            "ceiling_rate": float(np.mean(np.isclose(values, 1.0))),
        }
        for metric, estimate in summaries.items():
            low = high = np.nan
            if metric in {"mean", "median", "floor_rate", "ceiling_rate"}:
                low, high, failures = participant_bootstrap(
                    labels[["participant_id", target]].copy(),
                    lambda sample, metric=metric, target=target: _distribution_boot_stat(sample, target, metric),
                    seed=seed + 100 * target_index + len(metric),
                    replicates=replicates,
                )
            else:
                failures = 0
            rows.append(
                {
                    "target": target,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_failures": failures,
                    "n_labels": len(values),
                    "inference_unit": INFERENCE_UNIT,
                }
            )
    output = pd.DataFrame(rows)
    write_csv(paths["tables"] / "target_distributions.csv", output)
    return output


def _distribution_boot_stat(sample: pd.DataFrame, target: str, metric: str) -> float:
    values = pd.to_numeric(sample[target], errors="coerce").to_numpy(dtype=float)
    if metric == "mean":
        return float(np.mean(values))
    if metric == "median":
        return float(np.median(values))
    if metric == "floor_rate":
        return float(np.mean(np.isclose(values, 0.0)))
    if metric == "ceiling_rate":
        return float(np.mean(np.isclose(values, 1.0)))
    raise ValueError(metric)


def order_balance(labels: pd.DataFrame, paths: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for axis in ("condition", "intensity", "frequency"):
        table = (
            labels.groupby([axis, "presentation_position"])
            .size()
            .reset_index(name="count")
            .rename(columns={axis: "level"})
        )
        for row in table.to_dict(orient="records"):
            rows.append(
                {
                    "balance_axis": axis,
                    "level": row["level"],
                    "presentation_position": int(row["presentation_position"]),
                    "count": int(row["count"]),
                    "inference_unit": INFERENCE_UNIT,
                }
            )
    output = pd.DataFrame(rows)
    write_csv(paths["tables"] / "order_balance.csv", output)
    return output


def eeg_split(labels: pd.DataFrame, config: Any, paths: dict[str, Path]) -> pd.DataFrame:
    disabled = set(config.get("participants.eeg_disabled"))
    working = labels.copy()
    working["eeg_status"] = np.where(working["participant_id"].isin(disabled), "eeg_disabled", "eeg_available")
    rows = []
    for status, group in working.groupby("eeg_status"):
        row = {
            "eeg_status": status,
            "participant_count": int(group["participant_id"].nunique()),
            "n_labels": int(len(group)),
            "participants": ";".join(sorted(group["participant_id"].astype(str).unique())),
            "inference_unit": INFERENCE_UNIT,
        }
        for target in TARGETS:
            row[f"{target}_mean"] = float(pd.to_numeric(group[target], errors="coerce").mean())
            row[f"{target}_std"] = float(pd.to_numeric(group[target], errors="coerce").std(ddof=1))
        rows.append(row)
    output = pd.DataFrame(rows)
    write_csv(paths["tables"] / "eeg_availability_split.csv", output)
    return output


def find_retained_oof(config: Any) -> Path | None:
    candidates = [
        config.path("reports") / "condition_level_lopo_predictions.csv",
        REPO_ROOT / "artifacts" / "predictions" / "condition_level_lopo_predictions.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def historical_recalc(config: Any, paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    retained = find_retained_oof(config)
    source_kind = "retained_oof"
    if retained is None:
        source_kind = "regenerated_isolated_lopo"
        train_condition_state(
            config,
            source=config.path("features") / "window_features.csv",
            condition_output=paths["isolated_features"] / "condition_features.csv",
            models_dir=paths["isolated_models"],
            reports_dir=paths["isolated_reports"],
        )
        retained = paths["isolated_reports"] / "condition_level_lopo_predictions.csv"
    else:
        copied = paths["isolated_reports"] / "condition_level_lopo_predictions.csv"
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(retained, copied)
        retained = copied

    predictions = pd.read_csv(retained)
    metrics = compute_prediction_metrics(predictions)
    embedded = load_embedded_metrics(config)
    comparison = compare_metrics(metrics, embedded, source_kind)
    write_csv(paths["tables"] / "historical_recalc_comparison.csv", comparison)

    metrics_rows = flatten_metrics(metrics, "regenerated_oof", source_kind)
    embedded_rows = flatten_metrics(embedded, "state_model_joblib", "embedded_bundle")
    metric_frame = pd.DataFrame(metrics_rows + embedded_rows)
    write_csv(paths["tables"] / "historical_metrics_long.csv", metric_frame)

    audit = feature_audit(paths)
    write_csv(paths["tables"] / "forbidden_feature_audit.csv", audit)
    if audit["forbidden_count"].sum() > 0:
        raise AssertionError("Forbidden label/audit features entered the regenerated prediction feature set")
    return comparison, metric_frame, audit


def load_embedded_metrics(config: Any) -> dict[str, Any]:
    import joblib

    bundle = joblib.load(config.path("models") / "state_model.joblib")
    return bundle["metrics"]


def compute_prediction_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    from scipy.stats import spearmanr
    from sklearn.metrics import average_precision_score, precision_score, recall_score

    metrics: dict[str, Any] = {"unit_of_analysis": INFERENCE_UNIT, "n_labels": int(len(predictions)), "targets": {}}
    for target in TARGETS:
        truth = pd.to_numeric(predictions[target], errors="coerce").to_numpy(dtype=float)
        pred = pd.to_numeric(predictions[f"pred_{target}"], errors="coerce").to_numpy(dtype=float)
        condition_only = pd.to_numeric(predictions[f"condition_only_{target}"], errors="coerce").to_numpy(dtype=float)
        history = pd.to_numeric(predictions[f"history_{target}"], errors="coerce").to_numpy(dtype=float)
        metrics["targets"][target] = {
            "mae": float(np.mean(np.abs(truth - pred))),
            "spearman": float(spearmanr(truth, pred).statistic),
            "ranking_accuracy": ranking_accuracy(predictions, pred, target),
            "condition_only_baseline_mae": float(np.mean(np.abs(truth - condition_only))),
            "history_baseline_mae": float(np.mean(np.abs(truth - history))),
        }
    high_risk = pd.to_numeric(predictions["high_discomfort"], errors="coerce").to_numpy(dtype=int)
    probability = pd.to_numeric(predictions["risk_probability"], errors="coerce").to_numpy(dtype=float)
    thresholds = pd.to_numeric(predictions["risk_threshold"], errors="coerce").to_numpy(dtype=float)
    threshold = float(np.nanmedian(thresholds))
    predicted = probability >= threshold
    discomfort = metrics["targets"]["discomfort"]
    discomfort["risk_at_fold_tuned_threshold"] = {
        "threshold": threshold,
        "recall": float(recall_score(high_risk, predicted, zero_division=0)),
        "precision": float(precision_score(high_risk, predicted, zero_division=0)),
        "false_negatives": int(np.sum((high_risk == 1) & ~predicted)),
        "pr_auc": float(average_precision_score(high_risk, probability)) if len(np.unique(high_risk)) > 1 else float("nan"),
        "per_row_threshold_recall": float(np.mean(probability[high_risk == 1] >= thresholds[high_risk == 1]))
        if np.any(high_risk)
        else float("nan"),
    }
    metrics["deployable"] = bool(
        metrics["targets"]["relaxation"]["mae"] < metrics["targets"]["relaxation"]["condition_only_baseline_mae"]
        and metrics["targets"]["relaxation"]["mae"] < metrics["targets"]["relaxation"]["history_baseline_mae"]
        and metrics["targets"]["relaxation"]["spearman"] > 0
        and metrics["targets"]["discomfort"]["mae"] < metrics["targets"]["discomfort"]["condition_only_baseline_mae"]
        and metrics["targets"]["discomfort"]["mae"] < metrics["targets"]["discomfort"]["history_baseline_mae"]
        and discomfort["risk_at_fold_tuned_threshold"]["per_row_threshold_recall"] >= 0.5
    )
    return metrics


def ranking_accuracy(frame: pd.DataFrame, prediction: np.ndarray, target: str) -> float:
    correct = 0
    total = 0
    working = frame[["participant_id", "presentation_position", target]].copy()
    working["prediction"] = prediction
    for _, group in working.groupby("participant_id"):
        true = group[target].to_numpy(dtype=float)
        pred = group["prediction"].to_numpy(dtype=float)
        for left in range(len(group)):
            for right in range(left + 1, len(group)):
                true_difference = true[left] - true[right]
                if abs(true_difference) < 1e-12:
                    continue
                total += 1
                if true_difference * (pred[left] - pred[right]) > 0:
                    correct += 1
    return float(correct / total) if total else float("nan")


def compare_metrics(metrics: dict[str, Any], embedded: dict[str, Any], source_kind: str) -> pd.DataFrame:
    rows = []
    metric_paths = [
        ("relaxation", "mae"),
        ("relaxation", "spearman"),
        ("relaxation", "ranking_accuracy"),
        ("relaxation", "condition_only_baseline_mae"),
        ("relaxation", "history_baseline_mae"),
        ("discomfort", "mae"),
        ("discomfort", "spearman"),
        ("discomfort", "ranking_accuracy"),
        ("discomfort", "condition_only_baseline_mae"),
        ("discomfort", "history_baseline_mae"),
        ("discomfort", "risk_at_fold_tuned_threshold.recall"),
        ("discomfort", "risk_at_fold_tuned_threshold.precision"),
        ("discomfort", "risk_at_fold_tuned_threshold.false_negatives"),
        ("discomfort", "risk_at_fold_tuned_threshold.pr_auc"),
        ("discomfort", "risk_at_fold_tuned_threshold.per_row_threshold_recall"),
    ]
    for target, path in metric_paths:
        regenerated = nested_get(metrics["targets"][target], path)
        baseline = nested_get(embedded["targets"][target], path)
        rows.append(
            {
                "target": target,
                "metric": path,
                "regenerated_value": regenerated,
                "embedded_state_model_value": baseline,
                "absolute_delta": abs(float(regenerated) - float(baseline))
                if _is_number(regenerated) and _is_number(baseline)
                else np.nan,
                "source_kind": source_kind,
                "n_labels": metrics["n_labels"],
                "ci_low": np.nan,
                "ci_high": np.nan,
                "inference_unit": INFERENCE_UNIT,
            }
        )
    return pd.DataFrame(rows)


def nested_get(payload: dict[str, Any], dotted: str) -> Any:
    value: Any = payload
    for part in dotted.split("."):
        value = value[part]
    return value


def _is_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return np.isfinite(number)


def flatten_metrics(metrics: dict[str, Any], source: str, source_kind: str) -> list[dict[str, Any]]:
    rows = []
    for target in TARGETS:
        for key, value in metrics["targets"][target].items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    rows.append(
                        {
                            "source": source,
                            "source_kind": source_kind,
                            "target": target,
                            "metric": f"{key}.{subkey}",
                            "estimate": subvalue,
                            "n_labels": metrics.get("n_labels", 135),
                            "ci_low": np.nan,
                            "ci_high": np.nan,
                            "inference_unit": INFERENCE_UNIT,
                        }
                    )
            else:
                rows.append(
                    {
                        "source": source,
                        "source_kind": source_kind,
                        "target": target,
                        "metric": key,
                        "estimate": value,
                        "n_labels": metrics.get("n_labels", 135),
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "inference_unit": INFERENCE_UNIT,
                    }
                )
    return rows


def feature_audit(paths: dict[str, Path]) -> pd.DataFrame:
    import joblib

    rows = []
    for model_path in sorted(paths["isolated_models"].glob("state_model*.joblib")):
        bundle = joblib.load(model_path)
        columns = list(bundle.get("feature_columns", []))
        forbidden = forbidden_feature_columns(columns)
        runtime_qc = runtime_qc_feature_columns(columns)
        rows.append(
            {
                "model_file": str(model_path.relative_to(paths["root"])),
                "feature_count": len(columns),
                "forbidden_count": len(forbidden),
                "forbidden_columns": ";".join(forbidden[:25]),
                "runtime_qc_numeric_feature_count": len(runtime_qc),
                "runtime_qc_numeric_feature_examples": ";".join(runtime_qc[:15]),
                "inference_unit": INFERENCE_UNIT,
            }
        )
    return add_empty_ci(pd.DataFrame(rows), INFERENCE_UNIT)


def mixedlm_variance_components(
    labels: pd.DataFrame,
    paths: dict[str, Path],
    seed: int,
    replicates: int,
) -> pd.DataFrame:
    working = labels.copy()
    working["intensity_c"] = pd.to_numeric(working["intensity"], errors="coerce") - pd.to_numeric(
        working["intensity"], errors="coerce"
    ).mean()
    working["frequency_c"] = pd.to_numeric(working["frequency"], errors="coerce") - pd.to_numeric(
        working["frequency"], errors="coerce"
    ).mean()
    grid = (
        working[["condition", "intensity_c", "frequency_c"]]
        .drop_duplicates()
        .sort_values("condition", key=lambda series: series.map(condition_sort_key))
    )

    rows = []
    failure_rows = []
    for target_index, target in enumerate(TARGETS):
        fit = fit_mixedlm(working, target, "participant_id", grid)
        if fit["status"] != "ok":
            for metric in mixed_metric_names():
                rows.append(not_estimable_row(target, metric, fit["status"], 1.0))
            failure_rows.append({"target": target, "bootstrap_failures": replicates, "failure_rate": 1.0})
            continue

        boot_values: dict[str, list[float]] = {metric: [] for metric in mixed_metric_names()}
        failures = 0
        rng = np.random.default_rng(seed + 2000 + target_index)
        participants = np.asarray(sorted(working["participant_id"].astype(str).unique()))
        for _ in range(replicates):
            sampled = rng.choice(participants, size=len(participants), replace=True)
            pieces = []
            for draw_index, participant in enumerate(sampled):
                piece = working[working["participant_id"].astype(str) == participant].copy()
                piece["bootstrap_cluster"] = f"{participant}__boot{draw_index:02d}"
                pieces.append(piece)
            sample = pd.concat(pieces, ignore_index=True)
            boot_fit = fit_mixedlm(sample, target, "bootstrap_cluster", grid)
            if boot_fit["status"] != "ok":
                failures += 1
                continue
            for metric in mixed_metric_names():
                boot_values[metric].append(float(boot_fit[metric]))

        failure_rate = failures / replicates if replicates else 1.0
        failure_rows.append({"target": target, "bootstrap_failures": failures, "failure_rate": failure_rate})
        if failure_rate > 0.20:
            for metric in mixed_metric_names():
                rows.append(not_estimable_row(target, metric, "bootstrap_failure_rate_gt_20pct", failure_rate))
            continue

        for metric in mixed_metric_names():
            low, high = ci(boot_values[metric])
            rows.append(
                {
                    "target": target,
                    "metric": metric,
                    "estimate": fit[metric],
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_failures": failures,
                    "bootstrap_failure_rate": failure_rate,
                    "status": "estimable",
                    "model": "MixedLM target ~ intensity_c * frequency_c; re=1+intensity_c+frequency_c",
                    "inference_unit": INFERENCE_UNIT,
                }
            )

    output = pd.DataFrame(rows)
    write_csv(paths["tables"] / "variance_components.csv", output)
    write_csv(paths["tables"] / "mixedlm_bootstrap_failures.csv", pd.DataFrame(failure_rows))
    return output


def fit_mixedlm(frame: pd.DataFrame, target: str, group_col: str, grid: pd.DataFrame) -> dict[str, Any]:
    import statsmodels.formula.api as smf

    columns = [target, "intensity_c", "frequency_c", group_col]
    data = frame[columns].dropna().copy()
    if data[group_col].nunique() < 3:
        return {"status": "too_few_groups"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = smf.mixedlm(
                f"{target} ~ intensity_c * frequency_c",
                data=data,
                groups=data[group_col],
                re_formula="1 + intensity_c + frequency_c",
            )
            result = model.fit(method="lbfgs", reml=True, maxiter=200, disp=False)
        except Exception:
            return {"status": "fit_failed"}
    if not bool(getattr(result, "converged", False)):
        return {"status": "not_converged"}
    cov = np.asarray(result.cov_re, dtype=float)
    if cov.shape != (3, 3) or not np.isfinite(cov).all():
        return {"status": "invalid_random_effect_covariance"}
    residual = float(result.scale)
    variances = grid_random_variances(cov, grid)
    between = float(np.mean(variances))
    return {
        "status": "ok",
        "residual_variance": residual,
        "random_intercept_variance": float(cov[0, 0]),
        "random_intensity_slope_variance": float(cov[1, 1]),
        "random_frequency_slope_variance": float(cov[2, 2]),
        "grid_avg_random_effect_variance": between,
        "grid_avg_random_effect_sd": float(np.sqrt(max(between, 0.0))),
        "icc_between_over_total": float(between / (between + residual)) if between + residual > 0 else float("nan"),
    }


def mixed_metric_names() -> tuple[str, ...]:
    return (
        "residual_variance",
        "random_intercept_variance",
        "random_intensity_slope_variance",
        "random_frequency_slope_variance",
        "grid_avg_random_effect_variance",
        "grid_avg_random_effect_sd",
        "icc_between_over_total",
    )


def grid_random_variances(cov: np.ndarray, grid: pd.DataFrame) -> np.ndarray:
    rows = []
    for _, row in grid.iterrows():
        z = np.asarray([1.0, float(row["intensity_c"]), float(row["frequency_c"])], dtype=float)
        rows.append(float(z @ cov @ z.T))
    return np.asarray(rows, dtype=float)


def not_estimable_row(target: str, metric: str, status: str, failure_rate: float) -> dict[str, Any]:
    return {
        "target": target,
        "metric": metric,
        "estimate": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "bootstrap_failures": np.nan,
        "bootstrap_failure_rate": failure_rate,
        "status": f"not_estimable:{status}",
        "model": "MixedLM target ~ intensity_c * frequency_c; re=1+intensity_c+frequency_c",
        "inference_unit": INFERENCE_UNIT,
    }


def participant_optima(labels: pd.DataFrame, paths: dict[str, Path], seed: int, replicates: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    long_rows = []
    for participant, group in labels.groupby("participant_id"):
        group = group.sort_values("condition", key=lambda series: series.map(condition_sort_key))
        relaxation = choose_optimum(group, target="relaxation", direction="max")
        discomfort = choose_optimum(group, target="discomfort", direction="min")
        rows.append(
            {
                "participant_id": participant,
                "relaxation_primary_condition": relaxation["primary_condition"],
                "relaxation_tied_conditions": ";".join(relaxation["tied_conditions"]),
                "relaxation_primary_value": relaxation["primary_value"],
                "relaxation_opposite_discomfort": relaxation["opposite_value"],
                "discomfort_primary_condition": discomfort["primary_condition"],
                "discomfort_tied_conditions": ";".join(discomfort["tied_conditions"]),
                "discomfort_primary_value": discomfort["primary_value"],
                "discomfort_opposite_relaxation": discomfort["opposite_value"],
                "inference_unit": "participant",
            }
        )
        long_rows.append({"participant_id": participant, "target": "relaxation", **relaxation})
        long_rows.append({"participant_id": participant, "target": "discomfort", **discomfort})
    output = pd.DataFrame(rows)
    write_csv(paths["tables"] / "participant_optima.csv", output)

    long = pd.DataFrame(long_rows)
    summaries = []
    for target, condition_col, tied_col in (
        ("relaxation", "relaxation_primary_condition", "relaxation_tied_conditions"),
        ("discomfort", "discomfort_primary_condition", "discomfort_tied_conditions"),
    ):
        counts = Counter(output[condition_col])
        modal_condition, modal_count = sorted(
            counts.items(), key=lambda item: (-item[1], condition_sort_key(item[0]))
        )[0]
        observed_share = modal_count / len(output)
        tied_sets = output[tied_col].astype(str).str.split(";")
        tie_aware_share = float(np.mean([modal_condition in values for values in tied_sets]))
        modal_low, modal_high, modal_failures = modal_share_ci(
            output[["participant_id", condition_col]].rename(columns={condition_col: "condition"}),
            seed=seed + len(target),
            replicates=replicates,
        )
        tie_low, tie_high, tie_failures = default_tie_share_ci(
            output[["participant_id", tied_col]].rename(columns={tied_col: "tied"}),
            modal_condition,
            seed=seed + 500 + len(target),
            replicates=replicates,
        )
        summaries.append(
            {
                "target": target,
                "metric": "modal_primary_optimum_share",
                "modal_condition": modal_condition,
                "estimate": observed_share,
                "ci_low": modal_low,
                "ci_high": modal_high,
                "bootstrap_failures": modal_failures,
                "tie_aware_default_share": tie_aware_share,
                "tie_aware_ci_low": tie_low,
                "tie_aware_ci_high": tie_high,
                "tie_bootstrap_failures": tie_failures,
                "n_participants": len(output),
                "inference_unit": "participant",
            }
        )
    summary = pd.DataFrame(summaries)
    write_csv(paths["tables"] / "participant_optima_summary.csv", summary)
    return output, summary


def choose_optimum(group: pd.DataFrame, target: str, direction: str) -> dict[str, Any]:
    opposite = "discomfort" if target == "relaxation" else "relaxation"
    values = pd.to_numeric(group[target], errors="coerce")
    optimum = values.max() if direction == "max" else values.min()
    tied = group[np.isclose(values, optimum)].copy()
    if target == "relaxation":
        tied = tied.sort_values(
            ["discomfort", "condition"],
            key=lambda series: series.map(condition_sort_key) if series.name == "condition" else series,
        )
    else:
        tied = tied.sort_values(
            ["relaxation", "condition"],
            ascending=[False, True],
            key=lambda series: series.map(condition_sort_key) if series.name == "condition" else series,
        )
    primary = tied.iloc[0]
    return {
        "primary_condition": str(primary["condition"]),
        "tied_conditions": sorted(tied["condition"].astype(str).tolist(), key=condition_sort_key),
        "primary_value": float(primary[target]),
        "opposite_value": float(primary[opposite]),
    }


def modal_share_ci(frame: pd.DataFrame, seed: int, replicates: int) -> tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    participants = np.asarray(sorted(frame["participant_id"].astype(str).unique()))
    values = []
    for _ in range(replicates):
        sampled = rng.choice(participants, size=len(participants), replace=True)
        sample = frame.set_index("participant_id").loc[sampled]["condition"].tolist()
        counts = Counter(sample)
        values.append(max(counts.values()) / len(sample))
    low, high = ci(values)
    return low, high, 0


def default_tie_share_ci(frame: pd.DataFrame, default_condition: str, seed: int, replicates: int) -> tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    participants = np.asarray(sorted(frame["participant_id"].astype(str).unique()))
    indexed = frame.set_index("participant_id")
    values = []
    for _ in range(replicates):
        sampled = rng.choice(participants, size=len(participants), replace=True)
        tied_sets = indexed.loc[sampled]["tied"].astype(str).str.split(";")
        values.append(float(np.mean([default_condition in tied for tied in tied_sets])))
    low, high = ci(values)
    return low, high, 0


def branch_decision(
    variance: pd.DataFrame,
    optima_summary: pd.DataFrame,
    paths: dict[str, Path],
) -> pd.DataFrame:
    modal_min = float(optima_summary["estimate"].min())
    modal_upper = float(optima_summary["ci_high"].min())
    observed_modal_ge_60 = bool(modal_min >= 0.60)
    clearly_below_60 = bool(modal_upper < 0.60)
    sd_rows = variance[
        (variance["metric"] == "grid_avg_random_effect_sd")
        & (variance["status"].astype(str) == "estimable")
    ].copy()
    route1_estimable_targets = set(sd_rows["target"])
    route1_all_estimable = route1_estimable_targets == set(TARGETS)
    route1_excludes_negligible = bool(
        route1_all_estimable and (pd.to_numeric(sd_rows["ci_low"], errors="coerce") > NEGLIGIBLE_HETEROGENEITY_SD).all()
    )
    route1_includes_negligible = bool(
        route1_all_estimable and (pd.to_numeric(sd_rows["ci_low"], errors="coerce") <= NEGLIGIBLE_HETEROGENEITY_SD).all()
    )

    if clearly_below_60 and route1_excludes_negligible:
        decision = "personalized_selective_adaptive"
        project_default = "personalized/selective adaptive can proceed to Phase B planning"
        reason = "modal optimum share is clearly below 60% and MixedLM heterogeneity excludes the negligible SD threshold"
    elif observed_modal_ge_60 and route1_includes_negligible:
        decision = "unified_default_plus_guardrails"
        project_default = "unified default + safety guardrails"
        reason = "observed modal optimum share is at least 60% and MixedLM heterogeneity includes the negligible SD threshold"
    else:
        decision = "indeterminate_with_15_participants"
        project_default = "unified default + safety guardrails"
        reason = "15 participants cannot determine whether personalization is worth it"

    rows = [
        {
            "criterion": "min_modal_primary_optimum_share",
            "estimate": modal_min,
            "ci_low": float(optima_summary["ci_low"].min()),
            "ci_high": modal_upper,
            "threshold": 0.60,
            "passes_personalized_gate": clearly_below_60,
            "passes_unified_gate": observed_modal_ge_60,
            "decision": decision,
            "conservative_project_default": project_default,
            "reason": reason,
            "inference_unit": "participant",
        },
        {
            "criterion": "route1_grid_avg_random_effect_sd",
            "estimate": float(sd_rows["estimate"].max()) if not sd_rows.empty else np.nan,
            "ci_low": float(sd_rows["ci_low"].min()) if not sd_rows.empty else np.nan,
            "ci_high": float(sd_rows["ci_high"].max()) if not sd_rows.empty else np.nan,
            "threshold": NEGLIGIBLE_HETEROGENEITY_SD,
            "passes_personalized_gate": route1_excludes_negligible,
            "passes_unified_gate": route1_includes_negligible,
            "decision": decision,
            "conservative_project_default": project_default,
            "reason": reason,
            "inference_unit": INFERENCE_UNIT,
        },
        {
            "criterion": "phase_a_branch_decision",
            "estimate": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "threshold": np.nan,
            "passes_personalized_gate": False,
            "passes_unified_gate": decision == "unified_default_plus_guardrails",
            "decision": decision,
            "conservative_project_default": project_default,
            "reason": reason,
            "inference_unit": INFERENCE_UNIT,
        },
    ]
    output = pd.DataFrame(rows)
    write_csv(paths["tables"] / "branch_decision.csv", output)
    return output


def evidence_ledger(
    config: Any,
    paths: dict[str, Path],
    integrity: pd.DataFrame,
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for row in integrity.to_dict(orient="records"):
        rows.append(
            {
                "evidence_id": f"input:{row['table_name']}",
                "role": "input_csv",
                "path": row["path"],
                "sha256": row["sha256"],
                "rows": row["row_count"],
                "columns": row["column_count"],
                "inference_unit": row["inference_unit"],
                "notes": "required Phase A source table",
            }
        )
    model_path = config.path("models") / "state_model.joblib"
    rows.append(
        {
            "evidence_id": "model:state_model",
            "role": "historical_embedded_metrics",
            "path": str(model_path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(model_path),
            "rows": np.nan,
            "columns": np.nan,
            "inference_unit": INFERENCE_UNIT,
            "notes": "embedded metrics used for historical comparison",
        }
    )
    for artifact in [
        paths["isolated_reports"] / "condition_level_lopo_predictions.csv",
        paths["isolated_reports"] / "condition_level_lopo_metrics.json",
    ]:
        if artifact.exists():
            rows.append(
                {
                    "evidence_id": f"phase_a:{artifact.name}",
                    "role": "regenerated_lopo",
                    "path": str(artifact.relative_to(paths["root"])),
                    "sha256": sha256_file(artifact),
                    "rows": len(pd.read_csv(artifact)) if artifact.suffix == ".csv" else np.nan,
                    "columns": len(pd.read_csv(artifact).columns) if artifact.suffix == ".csv" else np.nan,
                    "inference_unit": INFERENCE_UNIT,
                    "notes": "isolated under Phase A output",
                }
            )
    rows.append(
        {
            "evidence_id": "historical_metric_max_delta",
            "role": "derived_check",
            "path": "tables/historical_recalc_comparison.csv",
            "sha256": sha256_file(paths["tables"] / "historical_recalc_comparison.csv"),
            "rows": len(comparison),
            "columns": len(comparison.columns),
            "inference_unit": INFERENCE_UNIT,
            "notes": f"max_abs_delta={comparison['absolute_delta'].max()}",
        }
    )
    output = add_empty_ci(pd.DataFrame(rows), INFERENCE_UNIT)
    write_csv(paths["tables"] / "evidence_ledger.csv", output)
    return output


def make_figures(labels: pd.DataFrame, variance: pd.DataFrame, optima: pd.DataFrame, paths: dict[str, Path]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    for axis, target in zip(axes, TARGETS, strict=True):
        axis.hist(pd.to_numeric(labels[target], errors="coerce"), bins=np.linspace(0, 1, 8), color="#3b82f6", edgecolor="white")
        axis.set_title(target)
        axis.set_xlabel("normalized score")
        axis.set_ylabel("participant-condition count")
    fig.savefig(paths["figures"] / "target_distributions.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    for axis, target, column in zip(
        axes,
        TARGETS,
        ("relaxation_primary_condition", "discomfort_primary_condition"),
        strict=True,
    ):
        counts = optima[column].value_counts().reindex(CONDITIONS, fill_value=0)
        axis.bar(counts.index, counts.values, color="#10b981")
        axis.set_title(f"{target} primary optimum")
        axis.set_xlabel("condition")
        axis.set_ylabel("participants")
        axis.set_ylim(0, max(5, int(counts.max()) + 1))
    fig.savefig(paths["figures"] / "participant_optima.png", dpi=180)
    plt.close(fig)

    sd = variance[(variance["metric"] == "grid_avg_random_effect_sd") & (variance["status"] == "estimable")]
    fig, axis = plt.subplots(figsize=(6, 3.8), constrained_layout=True)
    if sd.empty:
        axis.text(0.5, 0.5, "MixedLM route not estimable", ha="center", va="center")
        axis.set_axis_off()
    else:
        x = np.arange(len(sd))
        estimates = pd.to_numeric(sd["estimate"], errors="coerce").to_numpy(dtype=float)
        lows = estimates - pd.to_numeric(sd["ci_low"], errors="coerce").to_numpy(dtype=float)
        highs = pd.to_numeric(sd["ci_high"], errors="coerce").to_numpy(dtype=float) - estimates
        axis.errorbar(x, estimates, yerr=[lows, highs], fmt="o", color="#111827", capsize=4)
        axis.axhline(NEGLIGIBLE_HETEROGENEITY_SD, color="#ef4444", linestyle="--", linewidth=1)
        axis.set_xticks(x, sd["target"].tolist())
        axis.set_ylabel("grid-averaged random-effect SD")
        axis.set_title("Route 1 heterogeneity")
    fig.savefig(paths["figures"] / "variance_components.png", dpi=180)
    plt.close(fig)


def write_reports(
    paths: dict[str, Path],
    integrity: pd.DataFrame,
    distributions: pd.DataFrame,
    comparison: pd.DataFrame,
    variance: pd.DataFrame,
    optima_summary: pd.DataFrame,
    decision: pd.DataFrame,
    audit: pd.DataFrame,
    seed: int,
    replicates: int,
) -> None:
    decision_row = decision[decision["criterion"] == "phase_a_branch_decision"].iloc[0]
    max_delta = comparison["absolute_delta"].max()
    shape_ok = "通过" if integrity["shape_pass"].all() else "未通过"
    audit_ok = "通过" if int(audit["forbidden_count"].sum()) == 0 else "未通过"
    route_rows = variance[variance["metric"] == "grid_avg_random_effect_sd"]
    route_lines = []
    for _, row in route_rows.iterrows():
        if str(row["status"]) == "estimable":
            route_lines.append(
                f"- {row['target']}：SD={format_number(row['estimate'])}，95% CI={format_ci(row['ci_low'], row['ci_high'])}。"
            )
        else:
            route_lines.append(f"- {row['target']}：{row['status']}。")
    optima_lines = [
        f"- {row['target']}：modal={row['modal_condition']}，share={format_number(row['estimate'])}，"
        f"95% CI={format_ci(row['ci_low'], row['ci_high'])}，tie-aware share={format_number(row['tie_aware_default_share'])}。"
        for _, row in optima_summary.iterrows()
    ]
    dist_lines = []
    for target in TARGETS:
        subset = distributions[(distributions["target"] == target) & (distributions["metric"].isin(["mean", "floor_rate", "ceiling_rate"]))]
        parts = [f"{row['metric']}={format_number(row['estimate'])}" for _, row in subset.iterrows()]
        dist_lines.append(f"- {target}：" + "；".join(parts) + "。")

    report = f"""# Phase A 决策再分析报告

本次只执行 Phase A：A0 数据完整性/历史结果重算，以及 A1 个体化分支门。未生成 Phase B 输出，也未修改运行时 API、adaptive-control 行为、配置或模型服务路径。

## A0 数据完整性

- 五个必需 CSV shape：{shape_ok}。`condition_labels`、`condition_boundaries`、`condition_features` 均为 135 条 participant-condition 证据；`windows` 与 `window_features` 的 946 条仅是 10 秒特征切片，不作为独立监督样本。
- participant x Condition 完整性：15 名参与者均有 C1-C9，未发现重复 participant-condition 键。
- EEG split 已按配置中的 `participants.eeg_disabled` 输出到 `tables/eeg_availability_split.csv`。
- 禁用特征审计：{audit_ok}。历史重算模型未纳入问卷标签、raw 问卷列或审计型 QC 字段；运行时可见的数值 QC 特征单独计数记录。

## 目标分布

{chr(10).join(dist_lines)}

## 历史 LOPO 重算

当前默认 reports 中没有保留 `condition_level_lopo_predictions.csv`，因此本次在 `isolated_condition_lopo/` 下重跑 condition-level LOPO。重算指标与 `artifacts/models/state_model.joblib` 内嵌历史指标的最大绝对差为 `{format_number(max_delta, 8)}`。详细对比见 `tables/historical_recalc_comparison.csv`。

## A1 个体化门

预声明的可忽略异质性阈值：grid-averaged random-effect SD <= {NEGLIGIBLE_HETEROGENEITY_SD:.2f}，目标量表为 0-1 normalized score。MixedLM 规格为 `target ~ intensity_c * frequency_c`，随机项为 participant random intercept + intensity/frequency random slopes；bootstrap 单位始终是 participant。

{chr(10).join(route_lines)}

Route 2 的最优 Condition 统计：

{chr(10).join(optima_lines)}

## Phase A 分支结论

结论：`{decision_row['decision']}`。保守项目默认：{decision_row['conservative_project_default']}。

理由：{decision_row['reason']}。

因此，本轮应停在 Phase A；是否进入 Phase B 需要先审阅这些表和报告。
"""
    write_text(paths["root"] / "report_zh.md", report)

    appendix = f"""# Phase A Reproducibility Appendix

- Command: `python analysis/decision_reanalysis/run_phase_a.py --output artifacts/decision_reanalysis/phase_a --seed {seed} --bootstrap {replicates}`
- Repository root: `{REPO_ROOT}`
- Output root: `{paths['root']}`
- Inference unit: participant-condition labels (`n=135`). The `n=946` windows are feature slices only.
- Bootstrap: participant resampling with replacement; duplicated sampled participants are recreated as unique bootstrap clusters.
- Route 1: statsmodels MixedLM, fixed `target ~ intensity_c * frequency_c`, random `1 + intensity_c + frequency_c` by participant.
- Route 2 tie policy: relaxation ties break by lower discomfort, then C1-C9; discomfort ties break by higher relaxation, then C1-C9.
- Branch rules: personalized/selective adaptive requires modal optimum share clearly below 60% and Route 1 heterogeneity CI excluding SD <= 0.05. Unified default + guardrails requires observed modal optimum share >= 60% and Route 1 heterogeneity CI including SD <= 0.05. Otherwise the report says 15 participants cannot determine whether personalization is worth it.

Key files:

- `tables/data_integrity.csv`
- `tables/target_distributions.csv`
- `tables/historical_recalc_comparison.csv`
- `tables/variance_components.csv`
- `tables/participant_optima.csv`
- `tables/branch_decision.csv`
- `tables/evidence_ledger.csv`
- `figures/target_distributions.png`
- `figures/participant_optima.png`
- `figures/variance_components.png`
"""
    write_text(paths["root"] / "repro_appendix.md", appendix)


def copy_script_snapshot(paths: dict[str, Path]) -> None:
    snapshot = paths["root"] / "script_snapshot.json"
    payload = {
        "script": str(Path(__file__).resolve()),
        "python": sys.executable,
        "git_commit": git_commit(),
    }
    write_json(snapshot, payload)


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def main() -> int:
    args = parse_args()
    paths = ensure_phase_dirs(args.output)
    config = load_config()

    print("Phase A: data integrity", flush=True)
    integrity = table_integrity(paths)
    labels = read_required_csv("condition_labels")
    completeness = condition_completeness(labels, paths)
    distributions = target_distribution(labels, paths, args.seed, args.bootstrap)
    order_balance(labels, paths)
    eeg_split(labels, config, paths)
    versions = software_versions()
    write_csv(paths["tables"] / "software_versions.csv", versions)

    print("Phase A: historical LOPO recalc", flush=True)
    comparison, _metrics_long, audit = historical_recalc(config, paths)

    print("Phase A: MixedLM heterogeneity route", flush=True)
    variance = mixedlm_variance_components(labels, paths, args.seed, args.bootstrap)

    print("Phase A: participant optima route", flush=True)
    optima, optima_summary = participant_optima(labels, paths, args.seed, args.bootstrap)
    decision = branch_decision(variance, optima_summary, paths)
    ledger = evidence_ledger(config, paths, integrity, comparison)
    make_figures(labels, variance, optima, paths)
    write_reports(
        paths,
        integrity,
        distributions,
        comparison,
        variance,
        optima_summary,
        decision,
        audit,
        args.seed,
        args.bootstrap,
    )
    copy_script_snapshot(paths)

    assert len(completeness) == 15
    assert len(ledger) >= 1
    assert_no_window_n_as_inference(paths["tables"])
    assert_estimate_tables_have_ci(
        paths["tables"],
        [
            "target_distributions",
            "historical_recalc_comparison",
            "variance_components",
            "participant_optima_summary",
            "branch_decision",
        ],
    )
    print(json.dumps({"ok": True, "output": str(paths["root"])}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

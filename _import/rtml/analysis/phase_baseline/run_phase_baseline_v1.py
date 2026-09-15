from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from real_time_ml.config import load_config  # noqa: E402
from real_time_ml.data.alignment import condition_boundaries, load_marker_events  # noqa: E402
from real_time_ml.data.index import build_index  # noqa: E402
from real_time_ml.data.io import normalize_condition  # noqa: E402
from real_time_ml.features.physio import EEGMontage, eeg_features, eeg_quality_coverage  # noqa: E402
from real_time_ml.modeling.condition_models import apply_condition_baseline, condition_baseline  # noqa: E402


DEFAULT_PARTICIPANTS = ("P003", "P007", "P008", "P009", "P011", "P013", "P015")
TARGETS = ("calm", "activated")
EPS = 1e-12


def parse_analysis_seconds(event: dict[str, Any], default: float = 15.0) -> float:
    notes = str(event.get("notes") or "")
    key = "analysis_window_seconds="
    if key not in notes:
        return default
    tail = notes.split(key, 1)[1]
    value = []
    for char in tail:
        if char.isdigit() or char == ".":
            value.append(char)
        else:
            break
    try:
        return float("".join(value))
    except ValueError:
        return default


def event_time(event: dict[str, Any]) -> float:
    return float(event["xdf_time"])


def event_condition(event: dict[str, Any]) -> str | None:
    raw = event.get("condition_id") or event.get("condition")
    return normalize_condition(raw) if raw else None


def paired_pre_baselines(events: list[dict[str, Any]], participant_id: str) -> list[dict[str, Any]]:
    boundaries = condition_boundaries(events, participant_id)
    starts = sorted(
        [event for event in events if str(event.get("event_type", "")).lower() == "pre_condition_baseline_start"],
        key=event_time,
    )
    ends = sorted(
        [event for event in events if str(event.get("event_type", "")).lower() == "pre_condition_baseline_end"],
        key=event_time,
    )
    if len(starts) != 9 or len(ends) != 9:
        raise ValueError(f"{participant_id}: expected 9 pre-condition baseline start/end markers")

    rows: list[dict[str, Any]] = []
    used_end_indexes: set[int] = set()
    for start in starts:
        condition = event_condition(start)
        match_index: int | None = None
        for index, end in enumerate(ends):
            if index in used_end_indexes or event_time(end) <= event_time(start):
                continue
            if condition and event_condition(end) and event_condition(end) != condition:
                continue
            match_index = index
            break
        if match_index is None:
            raise ValueError(f"{participant_id}: missing pre-condition baseline end after marker {start.get('marker_index')}")
        used_end_indexes.add(match_index)
        end = ends[match_index]
        if condition is None:
            next_boundary = next((item for item in boundaries if item.start_xdf >= event_time(end) - 1e-6), None)
            if next_boundary is None:
                raise ValueError(f"{participant_id}: cannot map baseline ending at {event_time(end):.3f} to a condition")
            condition = next_boundary.condition
        duration = event_time(end) - event_time(start)
        analysis_seconds = min(parse_analysis_seconds(start), duration)
        rows.append(
            {
                "participant_id": participant_id,
                "condition": condition,
                "baseline_full_start_xdf": event_time(start),
                "baseline_full_end_xdf": event_time(end),
                "baseline_analysis_start_xdf": event_time(end) - analysis_seconds,
                "baseline_analysis_end_xdf": event_time(end),
                "baseline_analysis_seconds": analysis_seconds,
                "baseline_phase_duration_seconds": duration,
                "baseline_start_marker_index": int(start["marker_index"]),
                "baseline_end_marker_index": int(end["marker_index"]),
                "baseline_frequency_applied_start": float(start.get("applied_frequency_value", np.nan)),
                "baseline_frequency_applied_end": float(end.get("applied_frequency_value", np.nan)),
                "baseline_intensity_applied_start": float(start.get("applied_intensity_value", np.nan)),
                "baseline_intensity_applied_end": float(end.get("applied_intensity_value", np.nan)),
            }
        )
    return sorted(rows, key=lambda row: int(row["condition"][1:]))


def global_baseline_interval(events: list[dict[str, Any]]) -> tuple[float, float] | None:
    starts = [
        event_time(event)
        for event in events
        if str(event.get("event_type", "")).lower() in {"baseline_start", "global_baseline_start"}
    ]
    ends = [
        event_time(event)
        for event in events
        if str(event.get("event_type", "")).lower() in {"baseline_end", "global_baseline_end"}
    ]
    if not starts or not ends:
        return None
    start = min(starts)
    end = next((value for value in sorted(ends) if value > start), None)
    return (start, end) if end is not None else None


def load_physio_stream(xdf_path: Path, stream_type: str) -> tuple[np.ndarray, np.ndarray, float]:
    import pyxdf

    streams, _ = pyxdf.load_xdf(str(xdf_path), select_streams=[{"type": stream_type}], verbose=False)
    if len(streams) != 1:
        raise ValueError(f"Expected one stream of type {stream_type!r}, found {len(streams)}")
    stream = streams[0]
    samples = np.asarray(stream["time_series"], dtype=float)
    timestamps = np.asarray(stream["time_stamps"], dtype=float)
    sample_rate = float(stream["info"]["nominal_srate"][0])
    return samples, timestamps, sample_rate


def slice_samples(
    samples: np.ndarray,
    timestamps: np.ndarray,
    start_xdf: float,
    end_xdf: float,
) -> np.ndarray:
    left, right = np.searchsorted(timestamps, [start_xdf, end_xdf], side="left")
    return samples[int(left) : int(right)]


def mean_feature(features: dict[str, float], suffix: str) -> float:
    values = [float(value) for name, value in features.items() if name.endswith(suffix) and np.isfinite(value)]
    return float(np.mean(values)) if values else float("nan")


def add_eeg_summaries(record: dict[str, Any], features: dict[str, float]) -> None:
    for band in ("delta", "theta", "alpha", "beta", "gamma"):
        record[f"eeg_{band}_power_mean"] = mean_feature(features, f"_{band}_power")
        record[f"eeg_{band}_relative_mean"] = mean_feature(features, f"_{band}_relative")
        value = record[f"eeg_{band}_power_mean"]
        record[f"eeg_{band}_log_power_mean"] = float(np.log(value + EPS)) if np.isfinite(value) and value > 0 else float("nan")
    record["eeg_spectral_entropy_mean"] = mean_feature(features, "_spectral_entropy")
    record["eeg_hjorth_activity_mean"] = mean_feature(features, "_hjorth_activity")
    record["eeg_hjorth_mobility_mean"] = mean_feature(features, "_hjorth_mobility")
    record["eeg_hjorth_complexity_mean"] = mean_feature(features, "_hjorth_complexity")
    record["eeg_robust_amplitude_uV_mean"] = mean_feature(features, "_robust_amplitude_uV")


def neurokit_hr_features(ecg_uv: np.ndarray, sample_rate: float) -> dict[str, float | str]:
    if len(ecg_uv) < int(sample_rate * 5):
        return {
            "ecg_hr_bpm": float("nan"),
            "ecg_rr_median_ms": float("nan"),
            "ecg_peak_count": 0.0,
            "ecg_peak_count_per_min": float("nan"),
            "ecg_detector": "neurokit_too_short",
        }
    import neurokit2 as nk

    try:
        cleaned = nk.ecg_clean(np.asarray(ecg_uv, dtype=float), sampling_rate=sample_rate, method="neurokit")
        _signals, info = nk.ecg_peaks(cleaned, sampling_rate=sample_rate, method="neurokit", correct_artifacts=True)
        peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=float)
    except Exception:
        return {
            "ecg_hr_bpm": float("nan"),
            "ecg_rr_median_ms": float("nan"),
            "ecg_peak_count": 0.0,
            "ecg_peak_count_per_min": float("nan"),
            "ecg_detector": "neurokit_failed",
        }
    rr_ms = np.diff(peaks) / float(sample_rate) * 1000.0
    plausible = rr_ms[(rr_ms >= 60_000.0 / 220.0) & (rr_ms <= 60_000.0 / 35.0)]
    median_rr = float(np.median(plausible)) if plausible.size else float("nan")
    return {
        "ecg_hr_bpm": float(60_000.0 / median_rr) if np.isfinite(median_rr) and median_rr > 0 else float("nan"),
        "ecg_rr_median_ms": median_rr,
        "ecg_peak_count": float(peaks.size),
        "ecg_peak_count_per_min": float(peaks.size / (len(ecg_uv) / sample_rate) * 60.0),
        "ecg_rr_plausible_fraction": float(plausible.size / rr_ms.size) if rr_ms.size else 0.0,
        "ecg_detector": "neurokit",
    }


def segment_features(
    participant: str,
    condition: str,
    phase: str,
    start_xdf: float,
    end_xdf: float,
    samples: np.ndarray,
    timestamps: np.ndarray,
    sample_rate: float,
    config: Any,
) -> dict[str, Any]:
    segment = slice_samples(samples, timestamps, start_xdf, end_xdf)
    duration = max(0.0, end_xdf - start_xdf)
    record: dict[str, Any] = {
        "participant_id": participant,
        "condition": condition,
        "phase": phase,
        "phase_start_xdf": start_xdf,
        "phase_end_xdf": end_xdf,
        "phase_duration_seconds": duration,
        "sample_count": int(len(segment)),
        "sample_rate_hz": sample_rate,
    }
    if len(segment) == 0:
        return record

    eeg_columns = list(config.get("streams.eeg_columns"))
    ecg_columns = list(config.get("streams.ecg_columns"))
    eeg = segment[:, eeg_columns]
    ecg = segment[:, ecg_columns[0]] - segment[:, ecg_columns[1]]
    record.update(neurokit_hr_features(ecg, sample_rate))
    if participant in set(config.get("participants.eeg_disabled")):
        record["qc_eeg_disabled_by_participant"] = 1.0
        record["qc_eeg_strict_coverage"] = 0.0
    else:
        coverage = eeg_quality_coverage(
            eeg,
            sample_rate,
            float(config.get("quality.eeg_abs_uV_max")),
            float(config.get("quality.eeg_flat_std_uV_min")),
        )
        record["qc_eeg_disabled_by_participant"] = 0.0
        record["qc_eeg_strict_coverage"] = coverage
        if coverage >= float(config.get("quality.eeg_strict_coverage_min")):
            features = eeg_features(
                eeg,
                sample_rate,
                dict(config.get("features.eeg.bands")),
                EEGMontage.from_config(config),
                bool(config.get("features.eeg.common_average_reference")),
            )
            record.update(features)
            add_eeg_summaries(record, features)
    return record


def build_phase_segments(config: Any, participants: list[str], output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_by_participant = {row["participant_id"]: row for row in build_index(config, participants)}
    segment_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for participant in participants:
        source = source_by_participant[participant]
        if not source.get("xdf_path"):
            errors.append({"participant_id": participant, "reason": "missing_xdf"})
            continue
        try:
            events = load_marker_events(Path(source["xdf_path"]), config.get("streams.marker_name"))
            boundaries = {item.condition: item for item in condition_boundaries(events, participant)}
            pre_rows = paired_pre_baselines(events, participant)
            samples, timestamps, sample_rate = load_physio_stream(Path(source["xdf_path"]), config.get("streams.physio_type"))

            global_interval = global_baseline_interval(events)
            if global_interval is not None:
                segment_rows.append(
                    segment_features(
                        participant,
                        "GLOBAL",
                        "global_baseline",
                        global_interval[0],
                        global_interval[1],
                        samples,
                        timestamps,
                        sample_rate,
                        config,
                    )
                )

            for pre in pre_rows:
                condition = str(pre["condition"])
                boundary = boundaries[condition]
                pair_rows.append(
                    {
                        **pre,
                        "viewing_start_xdf": boundary.start_xdf,
                        "viewing_end_xdf": boundary.end_xdf,
                        "viewing_duration_seconds": boundary.duration_seconds,
                    }
                )
                segment_rows.append(
                    segment_features(
                        participant,
                        condition,
                        "pre",
                        float(pre["baseline_analysis_start_xdf"]),
                        float(pre["baseline_analysis_end_xdf"]),
                        samples,
                        timestamps,
                        sample_rate,
                        config,
                    )
                )
                segment_rows.append(
                    segment_features(
                        participant,
                        condition,
                        "viewing",
                        boundary.start_xdf,
                        boundary.end_xdf,
                        samples,
                        timestamps,
                        sample_rate,
                        config,
                    )
                )
        except Exception as error:
            errors.append({"participant_id": participant, "reason": str(error)})

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase_extraction_errors.json").write_text(
        json.dumps(errors, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    segments = pd.DataFrame(segment_rows)
    pairs = pd.DataFrame(pair_rows)
    return segments, pairs


def feature_base_columns(segments: pd.DataFrame) -> list[str]:
    excluded = {
        "participant_id",
        "condition",
        "phase",
        "phase_start_xdf",
        "phase_end_xdf",
        "phase_duration_seconds",
        "sample_count",
        "sample_rate_hz",
        "ecg_detector",
    }
    columns: list[str] = []
    for name in segments.columns:
        if name in excluded:
            continue
        if name.startswith("ecg_") and name != "ecg_hr_bpm":
            continue
        if name.startswith("qc_"):
            continue
        if name.startswith("eeg_") or name == "ecg_hr_bpm":
            values = pd.to_numeric(segments[name], errors="coerce")
            if values.notna().any():
                columns.append(name)
    return sorted(columns)


def build_delta_frame(segments: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    condition_segments = segments[segments["condition"].astype(str).ne("GLOBAL")].copy()
    bases = feature_base_columns(condition_segments)
    pre = condition_segments[condition_segments["phase"].eq("pre")].set_index(["participant_id", "condition"])
    viewing = condition_segments[condition_segments["phase"].eq("viewing")].set_index(["participant_id", "condition"])
    common_index = pre.index.intersection(viewing.index)
    rows: list[dict[str, Any]] = []
    for participant, condition in common_index:
        record: dict[str, Any] = {"participant_id": participant, "condition": condition}
        for base in bases:
            pre_value = pd.to_numeric(pd.Series([pre.loc[(participant, condition), base]]), errors="coerce").iloc[0]
            viewing_value = pd.to_numeric(pd.Series([viewing.loc[(participant, condition), base]]), errors="coerce").iloc[0]
            record[f"{base}__pre"] = float(pre_value) if np.isfinite(pre_value) else float("nan")
            record[f"{base}__viewing"] = float(viewing_value) if np.isfinite(viewing_value) else float("nan")
            if np.isfinite(pre_value) and np.isfinite(viewing_value):
                delta = float(viewing_value - pre_value)
                record[f"{base}__delta"] = delta
                record[f"{base}__relative_delta"] = float(delta / max(abs(float(pre_value)), EPS))
                if pre_value > 0 and viewing_value > 0:
                    record[f"{base}__log_ratio"] = float(math.log(float(viewing_value) / float(pre_value)))
                else:
                    record[f"{base}__log_ratio"] = float("nan")
            else:
                record[f"{base}__delta"] = float("nan")
                record[f"{base}__relative_delta"] = float("nan")
                record[f"{base}__log_ratio"] = float("nan")
        rows.append(record)
    delta = pd.DataFrame(rows)
    merged = labels.merge(delta, on=["participant_id", "condition"], how="inner", validate="one_to_one")
    merged["activated"] = 1.0 - pd.to_numeric(merged["calm"], errors="coerce")

    for column in [name for name in merged.columns if name.endswith("__delta")]:
        z_col = column.replace("__delta", "__participant_delta_z")
        values = pd.to_numeric(merged[column], errors="coerce")
        grouped = merged.assign(_value=values).groupby("participant_id")["_value"]
        mean = grouped.transform("mean")
        std = grouped.transform(lambda item: float(np.nanstd(item.to_numpy(dtype=float), ddof=0)))
        merged[z_col] = (values - mean) / std.replace(0.0, np.nan)
    return merged


def model_feature_columns(frame: pd.DataFrame) -> list[str]:
    allowed_suffixes = ("__delta", "__relative_delta", "__log_ratio")
    columns = []
    for name in frame.columns:
        if not name.endswith(allowed_suffixes):
            continue
        if "ecg_" in name and not name.startswith("ecg_hr_bpm"):
            continue
        values = pd.to_numeric(frame[name], errors="coerce")
        if values.notna().sum() >= 6:
            columns.append(name)
    return sorted(columns)


def matrix(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")


def ranking_accuracy(frame: pd.DataFrame, prediction: np.ndarray, target: str) -> float:
    correct = 0
    total = 0
    work = frame[["participant_id", "presentation_position", target]].copy()
    work["prediction"] = prediction
    for _participant, group in work.groupby("participant_id"):
        true = group[target].to_numpy(dtype=float)
        pred = group["prediction"].to_numpy(dtype=float)
        for left in range(len(group)):
            for right in range(left + 1, len(group)):
                true_diff = true[left] - true[right]
                if abs(true_diff) < 1e-12:
                    continue
                total += 1
                if true_diff * (pred[left] - pred[right]) > 0:
                    correct += 1
    return float(correct / total) if total else float("nan")


def history_baseline(test: pd.DataFrame, target: str, fallback: float) -> np.ndarray:
    predictions = np.full(len(test), fallback, dtype=float)
    for _participant, group in test.groupby("participant_id"):
        ordered = group.sort_values("presentation_position")
        previous: float | None = None
        for index, row in ordered.iterrows():
            position = test.index.get_loc(index)
            predictions[position] = fallback if previous is None else previous
            previous = float(row[target])
    return predictions


def fit_ridge(train: pd.DataFrame, target_values: np.ndarray, columns: list[str], seed: int):
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    k = max(1, min(20, len(columns), len(train) - 2))
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("select", SelectKBest(f_regression, k=k)),
            ("model", Ridge(alpha=10.0, random_state=seed)),
        ]
    ).fit(matrix(train, columns), target_values)


def evaluate_lopo(frame: pd.DataFrame, feature_columns: list[str], seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    prediction_rows: list[pd.DataFrame] = []
    groups = sorted(frame["participant_id"].astype(str).unique())
    for group in groups:
        train = frame[frame["participant_id"].astype(str).ne(group)].copy()
        test = frame[frame["participant_id"].astype(str).eq(group)].copy()
        if train.empty or test.empty:
            continue
        out = test[["participant_id", "condition", "presentation_position", *TARGETS]].copy()
        for target in TARGETS:
            fallback = float(train[target].mean())
            baseline_map, baseline_fallback = condition_baseline(train, target)
            condition_pred = apply_condition_baseline(test["condition"], baseline_map, baseline_fallback)
            out[f"{target}_condition_baseline"] = np.clip(condition_pred, 0.0, 1.0)
            out[f"{target}_history_baseline"] = np.clip(history_baseline(test, target, fallback), 0.0, 1.0)

            phase_model = fit_ridge(train, train[target].to_numpy(dtype=float), feature_columns, seed)
            phase_pred = phase_model.predict(matrix(test, feature_columns))
            out[f"{target}_phase_only"] = np.clip(phase_pred, 0.0, 1.0)

            train_base = apply_condition_baseline(train["condition"], baseline_map, baseline_fallback)
            residual_model = fit_ridge(
                train,
                train[target].to_numpy(dtype=float) - train_base,
                feature_columns,
                seed + 101,
            )
            residual_pred = residual_model.predict(matrix(test, feature_columns))
            out[f"{target}_condition_plus_phase"] = np.clip(condition_pred + residual_pred, 0.0, 1.0)
        prediction_rows.append(out)
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    metrics: dict[str, Any] = {
        "n_rows": int(len(frame)),
        "n_participants": int(frame["participant_id"].nunique()),
        "n_features": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "targets": {},
    }
    for target in TARGETS:
        y = predictions[target].to_numpy(dtype=float)
        target_metrics: dict[str, Any] = {}
        for name in ("condition_baseline", "history_baseline", "phase_only", "condition_plus_phase"):
            pred = predictions[f"{target}_{name}"].to_numpy(dtype=float)
            abs_error = np.abs(y - pred)
            try:
                spearman = float(spearmanr(y, pred).statistic)
            except Exception:
                spearman = float("nan")
            target_metrics[name] = {
                "mae": float(np.mean(abs_error)),
                "spearman": spearman,
                "ranking_accuracy": ranking_accuracy(predictions, pred, target),
            }
        condition_error = np.abs(y - predictions[f"{target}_condition_baseline"].to_numpy(dtype=float))
        phase_error = np.abs(y - predictions[f"{target}_condition_plus_phase"].to_numpy(dtype=float))
        target_metrics["condition_plus_phase_delta_mae_vs_condition"] = float(np.mean(phase_error - condition_error))
        target_metrics["condition_plus_phase_improved_rows"] = int(np.sum(phase_error < condition_error))
        target_metrics["condition_plus_phase_worse_rows"] = int(np.sum(phase_error > condition_error))
        metrics["targets"][target] = target_metrics
    return predictions, metrics


def write_report(path: Path, metrics: dict[str, Any], participants: list[str]) -> None:
    lines = [
        "# phase_baseline_v1 report",
        "",
        f"participants: {', '.join(participants)}",
        f"rows: {metrics['n_rows']}",
        f"features: {metrics['n_features']}",
        "",
        "## LOPO metrics",
        "",
        "| target | model | MAE | Spearman | ranking |",
        "|---|---:|---:|---:|---:|",
    ]
    for target in TARGETS:
        for model_name in ("condition_baseline", "history_baseline", "phase_only", "condition_plus_phase"):
            row = metrics["targets"][target][model_name]
            lines.append(
                f"| {target} | {model_name} | {row['mae']:.4f} | {row['spearman']:.4f} | {row['ranking_accuracy']:.4f} |"
            )
        delta = metrics["targets"][target]["condition_plus_phase_delta_mae_vs_condition"]
        improved = metrics["targets"][target]["condition_plus_phase_improved_rows"]
        worse = metrics["targets"][target]["condition_plus_phase_worse_rows"]
        lines.extend(
            [
                "",
                f"{target}: condition_plus_phase mean abs-error change vs condition baseline = {delta:.4f}; "
                f"improved rows = {improved}, worse rows = {worse}.",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase-aware pre-baseline vs viewing physiology analysis.")
    parser.add_argument("--participants", nargs="*", default=list(DEFAULT_PARTICIPANTS))
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    participants = [str(item) for item in args.participants]
    config = load_config(ROOT / "configs" / "project.yaml")
    output_dir = args.output_dir or (config.path("artifacts") / "phase_baseline_v1")
    output_dir.mkdir(parents=True, exist_ok=True)

    segments, pairs = build_phase_segments(config, participants, output_dir)
    labels = pd.read_csv(config.path("preprocessed") / "condition_labels.csv")
    labels = labels[labels["participant_id"].isin(participants)].copy()
    delta = build_delta_frame(segments, labels)
    features = model_feature_columns(delta)
    predictions, metrics = evaluate_lopo(delta, features, int(config.get("modeling.random_seed")))

    segments.to_csv(output_dir / "phase_segment_features.csv", index=False)
    pairs.to_csv(output_dir / "phase_pairs.csv", index=False)
    delta.to_csv(output_dir / "condition_phase_delta_features.csv", index=False)
    predictions.to_csv(output_dir / "calm_activated_phase_lopo_predictions.csv", index=False)
    (output_dir / "calm_activated_phase_lopo_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(output_dir / "calm_activated_phase_lopo_report.md", metrics, participants)
    print(json.dumps({k: v for k, v in metrics.items() if k != "feature_columns"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

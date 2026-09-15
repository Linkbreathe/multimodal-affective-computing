from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np


LABEL_COLUMNS = {
    "relaxation", "discomfort", "calm", "pleasantness", "monotony", "visual_fit",
    # These are the original questionnaire response scales. They are labels/audit fields,
    # never signals available to a model at inference time.
    "relaxation_raw", "discomfort_raw", "arousal_raw", "pleasantness_raw", "monotony_raw",
    "label_source_row",
}
METADATA_COLUMNS = {
    "participant_id", "condition", "presentation_position", "intensity", "frequency",
    "intensity_index", "frequency_index", "condition_index", "window_id",
    "condition_window_index", "condition_window_count", "sample_weight",
    "start_xdf", "end_xdf", "start_unix_ms", "end_unix_ms", "start_marker_index",
    "end_marker_index", "duration_seconds", "window_start_xdf", "window_end_xdf",
    "window_start_unix_ms", "window_end_unix_ms",
    *LABEL_COLUMNS,
}
STATIC_COLUMNS = {
    "participant_id", "condition", "presentation_position", "intensity", "frequency",
    "intensity_index", "frequency_index", "condition_index", *LABEL_COLUMNS,
}


def _to_numeric(series):
    import pandas as pd

    if series.dtype == bool:
        return series.astype(float)
    cleaned = series.replace({"True": 1.0, "False": 0.0, "true": 1.0, "false": 0.0})
    return pd.to_numeric(cleaned, errors="coerce")


def _slope(values: np.ndarray) -> float:
    valid = np.isfinite(values)
    if valid.sum() < 2:
        return float("nan")
    x = np.arange(len(values), dtype=float)[valid]
    y = values[valid]
    return float(np.polyfit(x, y, 1)[0])


def aggregate_window_frame(window_frame):
    """Create exactly one condition-level row for every participant/Condition pair.

    Labels are copied once from their parent Condition. Every window-derived statistic is
    calculated within that parent only; it never creates another supervised sample.
    """
    import pandas as pd

    required = {"participant_id", "condition", "condition_window_index", "relaxation", "discomfort"}
    missing = required - set(window_frame.columns)
    if missing:
        raise ValueError(f"Window feature table lacks required columns: {sorted(missing)}")
    frame = window_frame.copy()
    frame["condition_window_index"] = pd.to_numeric(frame["condition_window_index"], errors="coerce")
    frame = frame.sort_values(["participant_id", "condition", "condition_window_index"])
    duplicate_labels = frame.groupby(["participant_id", "condition"])[["relaxation", "discomfort"]].nunique(dropna=False)
    if (duplicate_labels > 1).any().any():
        raise ValueError("A participant/Condition has inconsistent inherited labels")
    numeric_columns: list[str] = []
    for column in frame.columns:
        if column in METADATA_COLUMNS or column in STATIC_COLUMNS:
            continue
        converted = _to_numeric(frame[column])
        if converted.notna().any():
            frame[column] = converted
            numeric_columns.append(column)
    records: list[dict[str, Any]] = []
    for _, group in frame.groupby(["participant_id", "condition"], sort=True):
        record = {column: group.iloc[0][column] for column in STATIC_COLUMNS if column in group.columns}
        record["window_count"] = int(len(group))
        record["window_count_expected"] = int(pd.to_numeric(group.get("condition_window_count"), errors="coerce").max())
        for column in numeric_columns:
            values = group[column].to_numpy(dtype=float)
            valid = values[np.isfinite(values)]
            prefix = f"{column}__"
            record[prefix + "missing_ratio"] = float(1.0 - valid.size / len(values))
            if valid.size == 0:
                for statistic in ("mean", "std", "min", "max", "median", "first", "last", "slope"):
                    record[prefix + statistic] = float("nan")
                continue
            record[prefix + "mean"] = float(np.mean(valid))
            record[prefix + "std"] = float(np.std(valid, ddof=0))
            record[prefix + "min"] = float(np.min(valid))
            record[prefix + "max"] = float(np.max(valid))
            record[prefix + "median"] = float(np.median(valid))
            first = next((value for value in values if np.isfinite(value)), float("nan"))
            last = next((value for value in values[::-1] if np.isfinite(value)), float("nan"))
            record[prefix + "first"] = float(first)
            record[prefix + "last"] = float(last)
            record[prefix + "slope"] = _slope(values)
            # range (=max-min) and delta (=last-first) are dropped: exact linear combinations of
            # retained stats, so redundant for the linear candidates. Removing them cuts ~19% of
            # condition-level features and leaves the LOPO deployability verdict unchanged; the
            # residual metric differences stay within the n=15 noise band (see feature-pruning report).
        records.append(record)
    output = pd.DataFrame(records).sort_values(["participant_id", "presentation_position"]).reset_index(drop=True)
    keys = output[["participant_id", "condition"]]
    if len(output) != len(keys.drop_duplicates()):
        raise AssertionError("Condition aggregation did not produce unique label keys")
    return output


def build_condition_dataset(window_feature_path: Path, output_path: Path | None = None):
    import pandas as pd

    source = pd.read_csv(window_feature_path)
    condition_frame = aggregate_window_frame(source)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        condition_frame.to_csv(output_path, index=False)
        try:
            condition_frame.to_parquet(output_path.with_suffix(".parquet"), index=False)
        except (ImportError, ValueError):
            pass
    return condition_frame


def aggregate_realtime_history(
    history: Iterable[dict[str, Any]],
    feature_columns: Iterable[str],
    static: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Causal condition aggregation for a live partial Condition.

    Missing future windows remain missing; no statistic looks beyond the current 10-second
    cycle. This gives the deployed condition-level model the same feature schema as offline
    training without inventing future data.
    """
    records = list(history)
    if not records:
        return {name: float("nan") for name in feature_columns}
    values_by_base: dict[str, list[float]] = {}
    for record in records:
        for name, value in record.items():
            if name in METADATA_COLUMNS or name in STATIC_COLUMNS:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            values_by_base.setdefault(name, []).append(number)
    output: dict[str, float] = dict(static or {})
    for base, values_list in values_by_base.items():
        values = np.asarray(values_list, dtype=float)
        valid = values[np.isfinite(values)]
        prefix = f"{base}__"
        output[prefix + "missing_ratio"] = float(1.0 - valid.size / len(values))
        if valid.size:
            output.update({
                prefix + "mean": float(np.mean(valid)),
                prefix + "std": float(np.std(valid)),
                prefix + "min": float(np.min(valid)),
                prefix + "max": float(np.max(valid)),
                prefix + "median": float(np.median(valid)),
                prefix + "first": float(valid[0]),
                prefix + "last": float(valid[-1]),
                prefix + "slope": _slope(values),
            })
    output["window_count"] = float(len(records))
    return {name: output.get(name, float("nan")) for name in feature_columns}

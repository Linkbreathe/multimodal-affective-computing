from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mac.config import ProjectConfig
from mac.data.index import build_index
from mac.data.io import iter_csv, parse_float, sniff_csv
from mac.data.video import load_video_index, sample_window_frames
from mac.data.tables import read_rows, write_parquet_if_available, write_rows
from mac.features.eye import eye_features
from mac.features.head import head_features
from mac.features.physio import EEGMontage, StreamingPhysioProcessor, detect_r_peaks, hrv_features
from mac.features.video import video_features
from mac.data.condition_data import aggregate_window_frame
from mac.preprocessing.pipeline import preprocess


NUMERIC_FIELDS = {
    "unix_time_ms",
    "head_position_x", "head_position_y", "head_position_z",
    "head_rotation_x", "head_rotation_y", "head_rotation_z", "head_rotation_w",
    "head_velocity_x", "head_velocity_y", "head_velocity_z", "head_angular_velocity_deg_s",
    "gaze_direction_x", "gaze_direction_y", "gaze_direction_z", "gaze_on_painting",
}


def _boolean_number(value: Any) -> float:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return 1.0
    if text in {"false", "0", "no"}:
        return 0.0
    return float("nan")


def _load_log_rows(path: Path | None, fields: set[str]) -> list[dict[str, float | str]]:
    if path is None or not path.exists():
        return []
    _, decimal, _ = sniff_csv(path)
    output: list[dict[str, float | str]] = []
    for raw in iter_csv(path):
        record: dict[str, float | str] = {}
        for name in fields:
            value = raw.get(name)
            if name == "gaze_on_painting":
                record[name] = _boolean_number(value)
            else:
                number = parse_float(value, decimal)
                record[name] = float("nan") if number is None else number
        if np.isfinite(record.get("unix_time_ms", np.nan)):
            if float(record["unix_time_ms"]) < 1e11:
                continue
            output.append(record)
    return sorted(output, key=lambda row: float(row["unix_time_ms"]))


def _slice_rows(rows: list[dict[str, Any]], start_ms: float, end_ms: float) -> list[dict[str, Any]]:
    if not rows:
        return []
    times = np.fromiter((float(row["unix_time_ms"]) for row in rows), dtype=float)
    left, right = np.searchsorted(times, [start_ms, end_ms], side="left")
    return rows[int(left) : int(right)]


def _load_physio(xdf_path: Path, stream_type: str) -> tuple[np.ndarray, np.ndarray, float]:
    import pyxdf

    streams, _ = pyxdf.load_xdf(str(xdf_path), select_streams=[{"type": stream_type}], verbose=False)
    if len(streams) != 1:
        raise ValueError(f"Expected one stream of type {stream_type!r}, found {len(streams)}")
    stream = streams[0]
    samples = np.asarray(stream["time_series"], dtype=float)
    timestamps = np.asarray(stream["time_stamps"], dtype=float)
    sample_rate = float(stream["info"]["nominal_srate"][0])
    return samples, timestamps, sample_rate


def _processor(config: ProjectConfig, participant: str, sample_rate: float) -> StreamingPhysioProcessor:
    return StreamingPhysioProcessor(
        sample_rate=sample_rate,
        eeg_columns=list(config.get("streams.eeg_columns")),
        ecg_columns=list(config.get("streams.ecg_columns")),
        counter_column=int(config.get("streams.counter_column")),
        bands=dict(config.get("features.eeg.bands")),
        montage=EEGMontage.from_config(config),
        eeg_disabled=participant in set(config.get("participants.eeg_disabled")),
        strict_coverage_min=float(config.get("quality.eeg_strict_coverage_min")),
        eeg_abs_uV_max=float(config.get("quality.eeg_abs_uV_max")),
        eeg_flat_std_uV_min=float(config.get("quality.eeg_flat_std_uV_min")),
        car=bool(config.get("features.eeg.common_average_reference")),
    )


def _aggregate_conditions(window_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in window_rows:
        groups[(row["participant_id"], row["condition"])].append(row)
    output: list[dict[str, Any]] = []
    metadata = {
        "participant_id", "condition", "presentation_position", "intensity", "frequency",
        "intensity_index", "frequency_index", "condition_index", "relaxation", "calm",
        "pleasantness", "discomfort", "monotony", "visual_fit",
    }
    for key, rows in groups.items():
        record = {name: rows[0].get(name) for name in metadata if name in rows[0]}
        record["window_count"] = len(rows)
        numeric_fields = set().union(*(row.keys() for row in rows)) - metadata
        for field in numeric_fields:
            values = []
            for row in rows:
                try:
                    value = float(row.get(field, np.nan))
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    values.append(value)
            if values and (field.startswith(("eeg_", "ecg_", "head_", "eye_", "video_")) or field.endswith("_coverage")):
                record[field] = float(np.mean(values))
        output.append(record)
    return sorted(output, key=lambda row: (row["participant_id"], int(row["condition"][1:])))


def extract_features(
    config: ProjectConfig,
    participants: list[str] | None = None,
    include_video: bool | None = None,
    output_dir: Path | None = None,
    windows_path: Path | None = None,
) -> dict[str, Any]:
    selected = participants or config.participants
    source_windows_path = windows_path or (config.path("preprocessed") / "windows.csv")
    if windows_path is None and not source_windows_path.exists():
        preprocess(config, selected)
    windows = [row for row in read_rows(source_windows_path) if row["participant_id"] in set(selected)]
    source_by_participant = {row["participant_id"]: row for row in build_index(config, selected)}
    video_enabled = bool(config.get("features.video.enabled")) if include_video is None else include_video
    output: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for participant in selected:
        source = source_by_participant[participant]
        participant_windows = [row for row in windows if row["participant_id"] == participant]
        if not source.get("xdf_path"):
            errors.append({"participant_id": participant, "reason": "missing_xdf"})
            continue
        try:
            samples, timestamps, sample_rate = _load_physio(Path(source["xdf_path"]), config.get("streams.physio_type"))
            session_dir = Path(source["session_dir"]) if source.get("session_dir") else None
            head_rows = _load_log_rows(Path(source["samples_csv"]) if source.get("samples_csv") else None, NUMERIC_FIELDS)
            eye_rows = _load_log_rows(Path(source["eye_tracking_csv"]) if source.get("eye_tracking_csv") else None, NUMERIC_FIELDS)
            video_index = load_video_index(
                Path(source["video_frames_csv"]) if source.get("video_frames_csv") else None,
                session_dir,
                participant,
            ) if video_enabled else None
            processor = _processor(config, participant, sample_rate)
            historical_peaks: list[float] = []
            for raw_window in sorted(participant_windows, key=lambda row: float(row["window_start_xdf"])):
                record: dict[str, Any] = dict(raw_window)
                start_xdf, end_xdf = float(raw_window["window_start_xdf"]), float(raw_window["window_end_xdf"])
                start_ms, end_ms = float(raw_window["window_start_unix_ms"]), float(raw_window["window_end_unix_ms"])
                left, right = np.searchsorted(timestamps, [start_xdf, end_xdf], side="left")
                window_samples = samples[int(left) : int(right)]
                physio, physio_qc = processor.process_window(window_samples)
                record.update(physio)
                record.update({f"qc_{key}": value for key, value in physio_qc.items()})
                if len(window_samples):
                    ecg = window_samples[:, processor.ecg_columns[0]] - window_samples[:, processor.ecg_columns[1]]
                    peaks, _ = detect_r_peaks(ecg, sample_rate)
                    historical_peaks.extend((start_xdf + peaks / sample_rate).tolist())
                    for horizon in config.get("features.ecg.slow_hrv_windows_seconds"):
                        record.update(hrv_features(np.asarray(historical_peaks), float(horizon)))
                head, head_qc = head_features(_slice_rows(head_rows, start_ms, end_ms))
                eye, eye_qc = eye_features(
                    _slice_rows(eye_rows, start_ms, end_ms),
                    float(config.get("features.eye.ivt_velocity_threshold_deg_s")),
                )
                record.update(head)
                record.update(eye)
                record.update({f"qc_{key}": value for key, value in {**head_qc, **eye_qc}.items()})
                if video_enabled:
                    frames = sample_window_frames(video_index, start_ms, end_ms, float(config.get("features.video.sample_fps"))) if video_index else ()
                    paths = [frame.path for frame in frames]
                    video, video_qc = video_features(paths)
                    record.update(video)
                    record.update({f"qc_{key}": value for key, value in video_qc.items()})
                    record["qc_video_timestamp_source"] = video_index.timestamp_source if video_index else "none"
                    record["qc_video_index_reason"] = video_index.reason if video_index else "video_disabled"
                    if not video_qc.get("video_usable", False) and video_index and video_index.reason != "ok":
                        record["qc_video_reason"] = video_index.reason
                    record["qc_video_iso_timestamp_fallback"] = float(bool(video_index and video_index.timestamp_source == "utc_timestamp_iso"))
                else:
                    record.update({"qc_video_coverage": 0.0, "qc_video_usable": False, "qc_video_reason": "disabled", "qc_video_timestamp_source": "none", "qc_video_index_reason": "disabled", "qc_video_iso_timestamp_fallback": 0.0})
                output.append(record)
        except Exception as error:
            errors.append({"participant_id": participant, "reason": str(error)})
    target = output_dir or config.path("features")
    aggregate = aggregate_window_frame(pd.DataFrame(output)) if output else []
    write_rows(target / "window_features.csv", output)
    aggregate_rows = aggregate.to_dict(orient="records") if hasattr(aggregate, "to_dict") else aggregate
    write_rows(target / "condition_features.csv", aggregate_rows)
    write_parquet_if_available(target / "window_features.parquet", output)
    write_parquet_if_available(target / "condition_features.parquet", aggregate_rows)
    error_name = "feature_extraction_errors.json" if target == config.path("features") else f"{target.name}_feature_extraction_errors.json"
    (config.path("reports") / error_name).write_text(
        json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"window_rows": len(output), "condition_rows": len(aggregate_rows), "errors": errors}

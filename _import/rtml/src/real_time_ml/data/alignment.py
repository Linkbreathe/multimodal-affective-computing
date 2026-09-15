from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from real_time_ml.data.io import normalize_condition


@dataclass(frozen=True)
class ConditionBoundary:
    participant_id: str
    condition: str
    start_xdf: float
    end_xdf: float
    start_unix_ms: int
    end_unix_ms: int
    start_marker_index: int
    end_marker_index: int

    @property
    def duration_seconds(self) -> float:
        return self.end_xdf - self.start_xdf


def load_marker_events(xdf_path: Path, marker_name: str) -> list[dict[str, Any]]:
    import pyxdf

    streams, _ = pyxdf.load_xdf(
        str(xdf_path), select_streams=[{"name": marker_name}], dejitter_timestamps=False, verbose=False
    )
    if len(streams) != 1:
        raise ValueError(f"Expected one marker stream {marker_name!r}, found {len(streams)}")
    stream = streams[0]
    events: list[dict[str, Any]] = []
    for index, (sample, timestamp) in enumerate(zip(stream["time_series"], stream["time_stamps"])):
        raw = sample[0] if isinstance(sample, (list, tuple, np.ndarray)) else sample
        try:
            payload = json.loads(str(raw))
        except json.JSONDecodeError:
            payload = {"event_type": str(raw)}
        payload["xdf_time"] = float(timestamp)
        payload["marker_index"] = index
        events.append(payload)
    return events


def marker_alignment_qc(events: list[dict[str, Any]], outlier_ms: float = 50.0) -> dict[str, Any]:
    paired = [event for event in events if event.get("unix_time_ms") is not None]
    if len(paired) < 2:
        return {"available": False, "reason": "fewer_than_two_paired_markers"}
    xdf = np.asarray([event["xdf_time"] for event in paired], dtype=float)
    unix = np.asarray([float(event["unix_time_ms"]) / 1000.0 for event in paired], dtype=float)
    offset = float(np.median(unix - xdf))
    residual_ms = (unix - (xdf + offset)) * 1000.0
    outlier_indices = np.flatnonzero(np.abs(residual_ms) >= outlier_ms)
    return {
        "available": True,
        "n_markers": len(paired),
        "unix_minus_xdf_offset_seconds": offset,
        "median_abs_residual_ms": float(np.median(np.abs(residual_ms))),
        "max_abs_residual_ms": float(np.max(np.abs(residual_ms))),
        "outliers": [
            {
                "marker_index": int(paired[i]["marker_index"]),
                "event_type": paired[i].get("event_type"),
                "residual_ms": float(residual_ms[i]),
            }
            for i in outlier_indices
        ],
    }


def condition_boundaries(events: list[dict[str, Any]], participant_id: str) -> list[ConditionBoundary]:
    starts: dict[str, dict[str, Any]] = {}
    ends: dict[str, dict[str, Any]] = {}
    for event in events:
        event_type = str(event.get("event_type", "")).lower()
        raw_condition = event.get("condition_id") or event.get("condition")
        if not raw_condition:
            continue
        condition = normalize_condition(raw_condition)
        if event_type == "condition_start":
            if condition in starts:
                raise ValueError(f"Duplicate start marker for {participant_id}/{condition}")
            starts[condition] = event
        elif event_type == "condition_end":
            if condition in ends:
                raise ValueError(f"Duplicate end marker for {participant_id}/{condition}")
            ends[condition] = event
    if set(starts) != {f"C{i}" for i in range(1, 10)} or set(ends) != set(starts):
        raise ValueError(
            f"{participant_id}: expected complete C1-C9 start/end markers; "
            f"starts={sorted(starts)}, ends={sorted(ends)}"
        )
    output: list[ConditionBoundary] = []
    for condition, start in starts.items():
        end = ends[condition]
        boundary = ConditionBoundary(
            participant_id=participant_id,
            condition=condition,
            start_xdf=float(start["xdf_time"]),
            end_xdf=float(end["xdf_time"]),
            start_unix_ms=int(start["unix_time_ms"]),
            end_unix_ms=int(end["unix_time_ms"]),
            start_marker_index=int(start["marker_index"]),
            end_marker_index=int(end["marker_index"]),
        )
        if boundary.duration_seconds <= 0:
            raise ValueError(f"Non-positive boundary duration for {participant_id}/{condition}")
        output.append(boundary)
    return sorted(output, key=lambda item: item.start_xdf)


def make_windows(boundaries: list[ConditionBoundary], length_seconds: float = 10.0) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for boundary in boundaries:
        count = int(np.floor((boundary.duration_seconds + 1e-6) / length_seconds))
        if count < 1:
            continue
        for condition_window_index in range(count):
            start_xdf = boundary.start_xdf + condition_window_index * length_seconds
            end_xdf = start_xdf + length_seconds
            start_unix_ms = boundary.start_unix_ms + int(round(condition_window_index * length_seconds * 1000))
            record = {
                **asdict(boundary),
                "window_id": f"{boundary.participant_id}_{boundary.condition}_W{condition_window_index:02d}",
                "condition_window_index": condition_window_index,
                "condition_window_count": count,
                "window_start_xdf": start_xdf,
                "window_end_xdf": end_xdf,
                "window_start_unix_ms": start_unix_ms,
                "window_end_unix_ms": start_unix_ms + int(round(length_seconds * 1000)),
                "sample_weight": 1.0 / count,
            }
            windows.append(record)
    return windows


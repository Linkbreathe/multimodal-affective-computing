"""Shared IO + signal utilities for the idiographic EEG/ECG study.

Everything here is deliberately thin: it reuses the installed ``real_time_ml``
utilities for marker parsing / condition boundaries / csv sniffing, and adds the
few things the existing pipeline does NOT provide and that the plan requires:
zero-phase filtering, a pre-condition-baseline boundary parser, headset-off /
IMU-motion gating intervals, and an overlapping windower.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy import signal

from real_time_ml.data.alignment import (
    ConditionBoundary,
    condition_boundaries,
    load_marker_events,
    marker_alignment_qc,
)
from real_time_ml.data.io import condition_parameters, iter_csv, normalize_condition, parse_float, sniff_csv

IDIO_DIR = Path(__file__).resolve().parent
REPO_ROOT = IDIO_DIR.parents[1]  # analysis/idiographic -> analysis -> real_time_inference
ARTIFACTS = IDIO_DIR / "artifacts"
MANIFEST = REPO_ROOT / "artifacts" / "manifests" / "source_manifest.json"
CONDITION_LABELS = REPO_ROOT / "artifacts" / "preprocessed" / "condition_labels.csv"


# --------------------------------------------------------------------------- #
# config + manifest
# --------------------------------------------------------------------------- #
def load_idio_config() -> dict[str, Any]:
    with (IDIO_DIR / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_manifest() -> dict[str, dict[str, Any]]:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {row["participant_id"]: row for row in payload["participants"]}


def load_subjective_labels() -> dict[tuple[str, str], dict[str, float]]:
    """(participant, condition) -> subjective label dict from the frozen csv."""
    out: dict[tuple[str, str], dict[str, float]] = {}
    for row in iter_csv(CONDITION_LABELS):
        key = (row["participant_id"], normalize_condition(row["condition"]))
        out[key] = {
            name: parse_float(row.get(name))
            for name in ("relaxation", "discomfort", "calm", "pleasantness", "monotony", "visual_fit")
        }
    return out


# --------------------------------------------------------------------------- #
# raw stream loading
# --------------------------------------------------------------------------- #
@dataclass
class Streams:
    eeg: np.ndarray  # (N, n_eeg) microvolts
    ecg: np.ndarray  # (N,) bipolar microvolts
    counter: np.ndarray  # (N,) packet counter
    t_eeg: np.ndarray  # (N,) xdf time seconds
    sample_rate: float
    imu_mag: np.ndarray | None  # (M,) motion magnitude, or None
    t_imu: np.ndarray | None


def load_streams(xdf_path: str | Path, cfg: dict[str, Any]) -> Streams:
    import pyxdf

    xdf_path = str(xdf_path)
    streams, _ = pyxdf.load_xdf(xdf_path, select_streams=[{"type": "eeg"}], verbose=False)
    if len(streams) != 1:
        raise ValueError(f"Expected one eeg stream, found {len(streams)} in {xdf_path}")
    s = streams[0]
    ts = np.asarray(s["time_series"], dtype=float)
    t_eeg = np.asarray(s["time_stamps"], dtype=float)
    eeg = ts[:, cfg["eeg_columns"]]
    ecg = ts[:, cfg["ecg_columns"][0]] - ts[:, cfg["ecg_columns"][1]]
    counter = ts[:, cfg["counter_column"]]

    imu_mag = t_imu = None
    try:
        imu_streams, _ = pyxdf.load_xdf(xdf_path, select_streams=[{"type": cfg["imu_stream_type"]}], verbose=False)
        if imu_streams:
            im = np.asarray(imu_streams[0]["time_series"], dtype=float)
            t_imu = np.asarray(imu_streams[0]["time_stamps"], dtype=float)
            gyro = im[:, 1:4]  # deg/s
            imu_mag = np.linalg.norm(gyro, axis=1)
    except Exception:
        imu_mag = t_imu = None

    return Streams(eeg=eeg, ecg=ecg, counter=counter, t_eeg=t_eeg,
                   sample_rate=float(cfg["sample_rate_hz"]), imu_mag=imu_mag, t_imu=t_imu)


# --------------------------------------------------------------------------- #
# boundaries: conditions (reuse), pre-condition baselines, headset-off
# --------------------------------------------------------------------------- #
def get_events(xdf_path: str | Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return load_marker_events(Path(str(xdf_path)), cfg["marker_name"])


def get_condition_boundaries(events: list[dict[str, Any]], participant_id: str) -> list[ConditionBoundary]:
    return condition_boundaries(events, participant_id)


def _paired_intervals(events: list[dict[str, Any]], start_type: str, end_type: str) -> list[tuple[float, float]]:
    starts = [e["xdf_time"] for e in events if str(e.get("event_type", "")).lower() == start_type]
    ends = [e["xdf_time"] for e in events if str(e.get("event_type", "")).lower() == end_type]
    starts.sort()
    ends.sort()
    out: list[tuple[float, float]] = []
    j = 0
    for s in starts:
        while j < len(ends) and ends[j] <= s:
            j += 1
        if j < len(ends):
            out.append((float(s), float(ends[j])))
            j += 1
    return out


def baseline_boundaries(events: list[dict[str, Any]], boundaries: list[ConditionBoundary]) -> dict[str, tuple[float, float]]:
    """Map each condition to its immediately-preceding 20s pre_condition_baseline.

    Baseline markers carry no condition_id, so we associate each baseline
    interval with the condition whose ``condition_start`` first follows it.
    """
    intervals = _paired_intervals(events, "pre_condition_baseline_start", "pre_condition_baseline_end")
    out: dict[str, tuple[float, float]] = {}
    starts_sorted = sorted(boundaries, key=lambda b: b.start_xdf)
    for s, e in intervals:
        nxt = next((b for b in starts_sorted if b.start_xdf >= e - 1e-6), None)
        if nxt is not None and nxt.condition not in out:
            out[nxt.condition] = (s, e)
    return out


def headset_off_intervals(events: list[dict[str, Any]]) -> list[tuple[float, float]]:
    return _paired_intervals(events, "headset_removed", "headset_worn")


# --------------------------------------------------------------------------- #
# windowing
# --------------------------------------------------------------------------- #
def overlapping_windows(start_xdf: float, end_xdf: float, length: float, step: float) -> list[dict[str, Any]]:
    """Non-averaged overlapping windows fully contained in [start_xdf, end_xdf]."""
    duration = end_xdf - start_xdf
    count = int(np.floor((duration - length) / step + 1e-9)) + 1
    count = max(count, 0)
    out: list[dict[str, Any]] = []
    for i in range(count):
        ws = start_xdf + i * step
        we = ws + length
        if we > end_xdf + 1e-6:
            break
        out.append({"window_index": i, "window_start_xdf": ws, "window_end_xdf": we})
    for rec in out:
        rec["window_count"] = len(out)
        rec["sample_weight"] = 1.0 / max(len(out), 1)
    return out


def slice_by_xdf(t_eeg: np.ndarray, start_xdf: float, end_xdf: float) -> tuple[int, int]:
    left = int(np.searchsorted(t_eeg, start_xdf, side="left"))
    right = int(np.searchsorted(t_eeg, end_xdf, side="left"))
    return left, right


def interval_overlaps(win: tuple[float, float], intervals: list[tuple[float, float]]) -> bool:
    ws, we = win
    for s, e in intervals:
        if ws < e and s < we:
            return True
    return False


# --------------------------------------------------------------------------- #
# zero-phase filtering (offline; replaces the pipeline's causal sosfilt)
# --------------------------------------------------------------------------- #
def zerophase_bandpass(x: np.ndarray, lo: float, hi: float, fs: float, order: int = 4) -> np.ndarray:
    x = signal.detrend(np.asarray(x, dtype=float), axis=0)
    hi = min(hi, fs * 0.45)
    sos = signal.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


def zerophase_notch(x: np.ndarray, freqs: list[float], fs: float, q: float = 30.0) -> np.ndarray:
    y = np.asarray(x, dtype=float)
    for f0 in freqs:
        if f0 < fs * 0.45:
            b, a = signal.iirnotch(f0, q, fs)
            y = signal.filtfilt(b, a, y, axis=0)
    return y


def condition_params(condition: str, cfg: dict[str, Any]) -> dict[str, float | int]:
    return condition_parameters(condition, cfg["intensities"], cfg["frequencies"])


__all__ = [
    "ARTIFACTS", "REPO_ROOT", "Streams",
    "load_idio_config", "load_manifest", "load_subjective_labels",
    "load_streams", "get_events", "get_condition_boundaries",
    "baseline_boundaries", "headset_off_intervals",
    "overlapping_windows", "slice_by_xdf", "interval_overlaps",
    "zerophase_bandpass", "zerophase_notch", "condition_params",
    "marker_alignment_qc",
]

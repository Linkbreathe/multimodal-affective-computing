"""Extract independent pre-condition baseline embeddings for P009."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal
import torch

from scripts.build_relax_alignment_cache import (
    _extract_ecg,
    _extract_eeg,
    _extract_eye,
    _extract_video,
    _gaze_window,
    _load_physio,
    _read_logged_csv,
    _video_index,
    _video_window,
)
from mac.adaptive.offline.healnet_prefix import EXPECTED_MODALITIES


HEAD_COLUMNS = (
    "head_angular_speed_deg_s_iqr",
    "head_angular_speed_deg_s_mean",
    "head_angular_speed_deg_s_median",
    "head_angular_speed_deg_s_range",
    "head_angular_speed_deg_s_std",
    "head_jerk_iqr",
    "head_jerk_mean",
    "head_jerk_median",
    "head_jerk_range",
    "head_jerk_std",
    "head_motion_spectral_entropy",
    "head_position_range",
    "head_speed_iqr",
    "head_speed_mean",
    "head_speed_median",
    "head_speed_range",
    "head_speed_std",
    "head_stationary_fraction",
)


def _robust_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {f"{prefix}_{name}": float("nan") for name in ("mean", "std", "median", "iqr", "range")}
    q25, q75 = np.percentile(values, [25, 75])
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_iqr": float(q75 - q25),
        f"{prefix}_range": float(np.max(values) - np.min(values)),
    }


def _spectral_entropy(power: np.ndarray) -> float:
    values = np.asarray(power, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < 2 or values.sum() <= 0:
        return float("nan")
    probabilities = values / values.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)) / np.log2(probabilities.size))


def _head_features(frame: pd.DataFrame, start_ms: float, end_ms: float) -> tuple[np.ndarray, bool, dict[str, float]]:
    timestamp = pd.to_numeric(frame["unix_time_ms"], errors="coerce")
    selected = frame.loc[(timestamp >= start_ms) & (timestamp < end_ms)].copy()
    required = ["unix_time_ms", "head_position_x", "head_position_y", "head_position_z"]
    numeric = selected[required].apply(pd.to_numeric, errors="coerce").dropna()
    if len(numeric) < 3:
        return np.zeros(len(HEAD_COLUMNS), dtype=np.float32), False, {"coverage": 0.0}
    time_s = numeric["unix_time_ms"].to_numpy(dtype=float) / 1000.0
    position = numeric[["head_position_x", "head_position_y", "head_position_z"]].to_numpy(dtype=float)
    dt = np.diff(time_s)
    keep = dt > 1e-4
    usable_dt = dt[keep]
    velocity = np.linalg.norm(np.diff(position, axis=0)[keep] / usable_dt[:, None], axis=1)
    angular_frame = selected.loc[numeric.index, "head_angular_velocity_deg_s"]
    angular = pd.to_numeric(angular_frame, errors="coerce").to_numpy(dtype=float)
    angular = angular[np.isfinite(angular)]
    acceleration = np.diff(velocity) / usable_dt[1:] if velocity.size > 1 else np.asarray([])
    jerk = np.diff(acceleration) / usable_dt[2:] if acceleration.size > 1 else np.asarray([])
    features: dict[str, float] = {}
    features.update(_robust_stats(velocity, "head_speed"))
    features.update(_robust_stats(angular, "head_angular_speed_deg_s"))
    features.update(_robust_stats(jerk, "head_jerk"))
    features["head_stationary_fraction"] = float(np.mean(velocity < 0.01)) if velocity.size else float("nan")
    features["head_position_range"] = float(np.linalg.norm(np.ptp(position, axis=0)))
    if velocity.size >= 8:
        sampling_rate = 1.0 / np.median(usable_dt)
        _, power = signal.welch(velocity, fs=sampling_rate, nperseg=min(len(velocity), 64))
        features["head_motion_spectral_entropy"] = _spectral_entropy(power)
    coverage = min(1.0, (time_s[-1] - time_s[0]) / 10.0)
    values = np.asarray([features.get(column, float("nan")) for column in HEAD_COLUMNS], dtype=np.float32)
    # The formal alignment contract defines head usability by temporal
    # coverage; non-finite individual features are zero-filled downstream.
    valid = bool(coverage >= 0.6)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), valid, {"coverage": float(coverage)}


def _baseline_bounds(events_path: Path) -> tuple[int, int]:
    events = _read_logged_csv(events_path)
    starts = events.loc[events["event_type"].eq("baseline_start"), "unix_time_ms"]
    ends = events.loc[events["event_type"].eq("baseline_end"), "unix_time_ms"]
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError("Expected exactly one initial baseline_start/baseline_end pair")
    start, end = int(starts.iloc[0]), int(ends.iloc[0])
    if end - start < 40_000:
        raise ValueError(f"Initial baseline is shorter than four independent windows: {(end - start) / 1000:.3f}s")
    return start, end


def build_baseline_windows(events_path: Path, formal_windows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    start_ms, end_ms = _baseline_bounds(events_path)
    participant = formal_windows.loc[formal_windows["participant_id"].astype(str).eq("P009")].copy()
    if participant.empty:
        raise ValueError("Formal alignment windows do not contain P009")
    offsets = (
        pd.to_numeric(participant["window_start_xdf"], errors="raise").to_numpy(dtype=float)
        - pd.to_numeric(participant["window_start_unix_ms"], errors="raise").to_numpy(dtype=float) / 1000.0
    )
    offset = float(np.median(offsets))
    rows = []
    for index in range(4):
        window_start_ms = start_ms + index * 10_000
        window_end_ms = window_start_ms + 10_000
        rows.append(
            {
                "participant_id": "P009",
                "condition": "BASELINE",
                "condition_window_index": index,
                "window_id": f"P009_BASELINE_W{index:02d}",
                "window_start_unix_ms": window_start_ms,
                "window_end_unix_ms": window_end_ms,
                "window_start_xdf": window_start_ms / 1000.0 + offset,
                "window_end_xdf": window_end_ms / 1000.0 + offset,
            }
        )
    return pd.DataFrame(rows), {
        "baseline_event_start_unix_ms": start_ms,
        "baseline_event_end_unix_ms": end_ms,
        "baseline_event_duration_seconds": (end_ms - start_ms) / 1000.0,
        "used_duration_seconds": 40.0,
        "unused_tail_seconds": (end_ms - start_ms) / 1000.0 - 40.0,
        "xdf_unix_offset_seconds": offset,
        "xdf_unix_offset_max_deviation_seconds": float(np.max(np.abs(offsets - offset))),
    }


def extract_baseline_embeddings(
    *,
    events_path: Path,
    samples_path: Path,
    eye_tracking_path: Path,
    video_frames_path: Path,
    session_dir: Path,
    xdf_path: Path,
    formal_windows_path: Path,
    device: str | torch.device,
) -> dict[str, Any]:
    formal_windows = pd.read_csv(formal_windows_path)
    windows, timing = build_baseline_windows(events_path, formal_windows)
    device_object = torch.device(device)
    sources = pd.DataFrame(
        [
            {
                "participant_id": "P009",
                "session_dir": str(session_dir),
                "xdf_path": str(xdf_path),
                "eye_tracking_csv": str(eye_tracking_path),
                "video_frames_csv": str(video_frames_path),
            }
        ]
    ).set_index("participant_id")
    dimensions = {"eeg": 1024, "ecg": 1024, "eye": 128, "head": 18, "video": 768}
    masks = {modality: np.ones(len(windows), dtype=bool) for modality in EXPECTED_MODALITIES}
    extraction: dict[str, Any] = {}

    physio_samples, physio_timestamps, _ = _load_physio(xdf_path)
    for index, row in windows.iterrows():
        left, right = np.searchsorted(
            physio_timestamps,
            [float(row["window_start_xdf"]), float(row["window_end_xdf"])],
            side="left",
        )
        masks["eeg"][index] = right - left >= 100 and physio_samples.shape[1] > 4
        masks["ecg"][index] = right - left >= 100 and physio_samples.shape[1] > 8
    del physio_samples, physio_timestamps

    eye_log = _read_logged_csv(eye_tracking_path)
    for index, row in windows.iterrows():
        masks["eye"][index] = _gaze_window(
            eye_log, float(row["window_start_unix_ms"]), float(row["window_end_unix_ms"])
        ) is not None
    video_timestamps, video_paths = _video_index(video_frames_path, session_dir)
    for index, row in windows.iterrows():
        masks["video"][index] = _video_window(
            video_timestamps,
            video_paths,
            float(row["window_start_unix_ms"]),
            float(row["window_end_unix_ms"]),
        ) is not None
    samples = _read_logged_csv(samples_path)
    head_values = np.zeros((len(windows), len(HEAD_COLUMNS)), dtype=np.float32)
    head_quality = []
    for index, row in windows.iterrows():
        values, valid, quality = _head_features(
            samples, float(row["window_start_unix_ms"]), float(row["window_end_unix_ms"])
        )
        head_values[index] = values
        masks["head"][index] = valid
        head_quality.append(quality)

    embeddings: dict[str, np.ndarray] = {"head": head_values}
    extractors = {
        "eeg": lambda: _extract_eeg(windows, sources, masks["eeg"], device_object, 4),
        "ecg": lambda: _extract_ecg(windows, sources, masks["ecg"], device_object, 4),
        "eye": lambda: _extract_eye(windows, sources, masks["eye"], device_object, 4),
        "video": lambda: _extract_video(windows, sources, masks["video"], device_object, 1),
    }
    for modality, extractor in extractors.items():
        try:
            embeddings[modality] = extractor()
            extraction[modality] = {"status": "success", "valid_windows": int(masks[modality].sum())}
        except Exception as error:  # Preserve a machine-readable calibration failure instead of fabricating validity.
            embeddings[modality] = np.zeros((len(windows), dimensions[modality]), dtype=np.float32)
            masks[modality][:] = False
            extraction[modality] = {
                "status": "failed",
                "valid_windows": 0,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    extraction["head"] = {
        "status": "success" if masks["head"].any() else "failed",
        "valid_windows": int(masks["head"].sum()),
        "feature_columns": list(HEAD_COLUMNS),
        "window_quality": head_quality,
    }
    return {
        "windows": windows,
        "embeddings": {key: torch.from_numpy(value) for key, value in embeddings.items()},
        "masks": {key: torch.from_numpy(value) for key, value in masks.items()},
        "metadata": {
            **timing,
            "source": "initial_pre_condition_physiological_baseline",
            "participant_id": "P009",
            "n_non_overlapping_windows": len(windows),
            "modalities": list(EXPECTED_MODALITIES),
            "valid_windows": {key: int(value.sum()) for key, value in masks.items()},
            "all_modalities_valid_per_window": np.logical_and.reduce(list(masks.values())).tolist(),
            "extraction": extraction,
            "c1_safety_condition_baseline": False,
        },
    }


__all__ = ["HEAD_COLUMNS", "build_baseline_windows", "extract_baseline_embeddings"]

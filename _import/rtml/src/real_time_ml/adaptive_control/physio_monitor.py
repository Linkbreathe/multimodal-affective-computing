from __future__ import annotations

from typing import Any

import numpy as np

from real_time_ml.config import ProjectConfig
from real_time_ml.features.physio import causal_filter

from .contracts import AdaptivePhysioChannelSnapshot, AdaptivePhysioSnapshot, AdaptivePhysioStats


def _finite_float(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


def summarize_values(values: np.ndarray) -> AdaptivePhysioStats:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return AdaptivePhysioStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return AdaptivePhysioStats(
        sample_count=int(finite.size),
        mean=_finite_float(float(np.mean(finite))),
        std=_finite_float(float(np.std(finite))),
        min=_finite_float(float(np.min(finite))),
        max=_finite_float(float(np.max(finite))),
        rms=_finite_float(float(np.sqrt(np.mean(finite**2)))),
        peak_to_peak=_finite_float(float(np.ptp(finite))),
    )


def downsample_values(values: np.ndarray, max_points: int) -> list[float]:
    data = np.asarray(values, dtype=float).reshape(-1)
    if data.size == 0 or max_points <= 0:
        return []
    if data.size > max_points:
        indices = np.linspace(0, data.size - 1, max_points).astype(int)
        data = data[indices]
    return [_finite_float(float(value)) for value in data]


def _filter_or_raw(values: np.ndarray, low_hz: float, high_hz: float, sample_rate_hz: float) -> np.ndarray:
    if values.size == 0 or sample_rate_hz <= 0 or high_hz <= low_hz:
        return np.asarray(values, dtype=float)
    try:
        return causal_filter(values, low_hz, high_hz, sample_rate_hz)
    except (TypeError, ValueError):
        return np.asarray(values, dtype=float)


def _valid_sample_rows(rows: list[Any], expected_channel_count: int) -> np.ndarray:
    output = []
    for row in rows:
        try:
            values = np.asarray(row, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            continue
        if values.size >= expected_channel_count:
            output.append(values[:expected_channel_count])
    if not output:
        return np.empty((0, expected_channel_count), dtype=float)
    return np.vstack(output)


def _channel_snapshot(
    name: str,
    source: str,
    unit: str,
    raw: np.ndarray,
    filtered: np.ndarray,
    max_points: int,
) -> AdaptivePhysioChannelSnapshot:
    return AdaptivePhysioChannelSnapshot(
        name=name,
        source=source,
        unit=unit,
        raw_values=downsample_values(raw, max_points),
        filtered_values=downsample_values(filtered, max_points),
        raw=summarize_values(raw),
        filtered=summarize_values(filtered),
    )


def build_physio_snapshot(
    config: ProjectConfig,
    rows: list[Any],
    now_ms: int,
    *,
    stream_found: bool,
    sample_received: bool,
    last_sample_ms: int | None,
    stream_name: str = "",
    stream_type: str = "",
    channel_count: int = 0,
    nominal_srate: float = 0.0,
    window_seconds: float = 10.0,
    max_points: int = 240,
) -> AdaptivePhysioSnapshot:
    sample_rate_hz = float(config.get("streams.sample_rate_hz"))
    eeg_columns = [int(column) for column in config.get("streams.eeg_columns")]
    ecg_columns = [int(column) for column in config.get("streams.ecg_columns")]
    eeg_names = [str(name) for name in config.get("streams.eeg_names")]
    raw_unit = str(config.get("streams.raw_unit") or "uV")
    expected_channel_count = max([int(config.get("streams.counter_column")), *eeg_columns, *ecg_columns]) + 1
    values = _valid_sample_rows(rows, expected_channel_count)

    reasons: list[str] = []
    if not stream_found:
        reasons.append("no_type_eeg_lsl_stream")
    elif not sample_received:
        reasons.append("lsl_stream_found_but_no_sample")
    elif values.size == 0:
        reasons.append("physio_sample_too_short")
    elif values.shape[0] < int(sample_rate_hz * min(window_seconds, 1.0) * 0.5):
        reasons.append("insufficient_physio_window")
    if not reasons:
        reasons.append("ready")

    channels: list[AdaptivePhysioChannelSnapshot] = []
    if values.size:
        eeg_raw = values[:, eeg_columns]
        eeg_filtered = _filter_or_raw(eeg_raw, 1.0, min(45.0, sample_rate_hz * 0.45), sample_rate_hz)
        for index, column in enumerate(eeg_columns):
            name = eeg_names[index] if index < len(eeg_names) else f"ch{column}"
            channels.append(
                _channel_snapshot(name, "EEG", raw_unit, eeg_raw[:, index], eeg_filtered[:, index], max_points)
            )

        ecg_raw = values[:, ecg_columns[0]] - values[:, ecg_columns[1]]
        ecg_filtered = _filter_or_raw(ecg_raw, 5.0, min(25.0, sample_rate_hz * 0.45), sample_rate_hz)
        channels.append(
            _channel_snapshot(str(config.get("streams.ecg_bipolar") or "ECG"), "ECG", raw_unit, ecg_raw, ecg_filtered, max_points)
        )

    sample_age_ms = -1.0 if last_sample_ms is None else float(max(0, now_ms - last_sample_ms))
    return AdaptivePhysioSnapshot(
        unix_time_ms=now_ms,
        lsl_eeg_stream_found=stream_found,
        lsl_eeg_sample_received=sample_received,
        stream_name=str(stream_name or ""),
        stream_type=str(stream_type or ""),
        channel_count=int(channel_count),
        expected_channel_count=expected_channel_count,
        nominal_srate=_finite_float(float(nominal_srate or 0.0)),
        sample_rate_hz=_finite_float(sample_rate_hz),
        window_seconds=_finite_float(float(window_seconds)),
        sample_age_ms=_finite_float(sample_age_ms),
        sample_count=int(values.shape[0]),
        max_points=int(max_points),
        channels=channels,
        reasons=reasons,
    )

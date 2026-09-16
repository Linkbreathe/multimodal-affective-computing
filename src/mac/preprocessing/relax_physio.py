"""Leakage-aware preprocessing utilities for the Relax EEG/ECG ladder.

The Relax XDF stream stores a sample counter followed by four EEG electrodes
and two bipolar-source ECG electrodes.  The EEG acquisition order is
``M2, TP9, TP10, M1``.  M1/M2 are reference electrodes, so the two model
channels are constructed with a linked-mastoid reference before any
normalization::

    reference = (M1 + M2) / 2
    TP9_linked = TP9 - reference
    TP10_linked = TP10 - reference

All functions use channel-first arrays and preserve physical microvolt units
unless their name explicitly says otherwise.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import signal


RAW_EEG_CHANNELS = ("M2", "TP9", "TP10", "M1")
LINKED_EEG_CHANNELS = ("TP9", "TP10")
RAW_EEG_COLUMNS = (1, 2, 3, 4)
RAW_ECG_COLUMNS = (7, 8)
TARGET_SAMPLING_RATE = 200.0


def _as_channel_first(values: np.ndarray, *, channels: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Expected a [channels, samples] array, got {array.shape}")
    if channels is not None and array.shape[0] != channels:
        raise ValueError(f"Expected {channels} channels, got {array.shape[0]}")
    return array


def interpolate_nonfinite(values: np.ndarray) -> np.ndarray:
    """Linearly fill isolated non-finite samples without changing finite data."""

    array = _as_channel_first(values).copy()
    sample_index = np.arange(array.shape[1], dtype=np.float64)
    for channel in range(array.shape[0]):
        finite = np.isfinite(array[channel])
        if finite.all():
            continue
        if int(finite.sum()) < 2:
            raise ValueError(f"Channel {channel} has fewer than two finite samples")
        array[channel, ~finite] = np.interp(
            sample_index[~finite], sample_index[finite], array[channel, finite]
        )
    return array


def select_raw_eeg(stream_samples: np.ndarray) -> np.ndarray:
    """Select the four EEG columns as ``[M2, TP9, TP10, M1]``."""

    samples = np.asarray(stream_samples)
    if samples.ndim != 2 or samples.shape[1] <= max(RAW_EEG_COLUMNS):
        raise ValueError(f"Unexpected Relax stream shape: {samples.shape}")
    return interpolate_nonfinite(samples[:, RAW_EEG_COLUMNS].T)


def select_lead_i_like_ecg_uv(stream_samples: np.ndarray) -> np.ndarray:
    """Return the left-wrist minus right-wrist ECG derivation in microvolts."""

    samples = np.asarray(stream_samples)
    if samples.ndim != 2 or samples.shape[1] <= max(RAW_ECG_COLUMNS):
        raise ValueError(f"Unexpected Relax stream shape: {samples.shape}")
    ecg = samples[:, RAW_ECG_COLUMNS[0]] - samples[:, RAW_ECG_COLUMNS[1]]
    return interpolate_nonfinite(np.asarray(ecg, dtype=np.float64)[None, :])


def linked_mastoid_reference(raw_eeg_uv: np.ndarray) -> np.ndarray:
    """Construct linked-mastoid TP9/TP10 channels before normalization."""

    raw = _as_channel_first(raw_eeg_uv, channels=4)
    # M1 and M2 are references, not model channels.  Referencing before
    # filtering/normalization preserves the intended electrode relationship.
    reference = 0.5 * (raw[0] + raw[3])
    return np.stack((raw[1] - reference, raw[2] - reference), axis=0)


def exact_resample(values: np.ndarray, output_samples: int) -> np.ndarray:
    """FFT-resample to an exact length, matching the legacy Relax extractor."""

    array = interpolate_nonfinite(values)
    if output_samples <= 0:
        raise ValueError("output_samples must be positive")
    if array.shape[-1] == output_samples:
        return array.astype(np.float32, copy=False)
    return signal.resample(array, output_samples, axis=-1).astype(np.float32)


def window_zscore(values: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Per-window, per-channel z-score used by the legacy REVE pipeline."""

    array = interpolate_nonfinite(values)
    mean = array.mean(axis=-1, keepdims=True)
    scale = array.std(axis=-1, keepdims=True)
    scale = np.where(np.isfinite(scale) & (scale > epsilon), scale, 1.0)
    result = (array - mean) / scale
    if not np.isfinite(result).all():
        raise ValueError("Window z-score produced non-finite values")
    return result.astype(np.float32)


def _notch(values: np.ndarray, sampling_rate: float, frequencies: Iterable[float]) -> np.ndarray:
    output = interpolate_nonfinite(values)
    for frequency in frequencies:
        frequency = float(frequency)
        if sampling_rate / 2.0 <= frequency:
            continue
        # This is the exact NeuroRVQ example setting: Q = f / 2.
        numerator, denominator = signal.iirnotch(
            w0=frequency, Q=frequency / 2.0, fs=float(sampling_rate)
        )
        output = signal.filtfilt(numerator, denominator, output, axis=-1)
    return output


def neurorvq_eeg_filter(
    values_uv: np.ndarray,
    sampling_rate: float,
    *,
    clip_uv: float | None = 500.0,
) -> np.ndarray:
    """Apply the official NeuroRVQ EEG example filter in physical units.

    Order and parameters match the repository example: 50/60/100 Hz notch
    filters (when below Nyquist), third-order 0.5--45 Hz Butterworth, then an
    optional ±500 microvolt clip.  Resampling is intentionally separate so a
    caller can filter a padded segment and crop its center before resampling.
    """

    if sampling_rate <= 2.0:
        raise ValueError(f"Invalid sampling rate: {sampling_rate}")
    # Keep this order aligned with the NeuroRVQ example: line-noise removal,
    # broad band-pass, then amplitude guard.  Resampling is done by the caller.
    output = _notch(values_uv, sampling_rate, (50.0, 60.0, 100.0))
    lowpass_applied = min(45.0, sampling_rate / 2.0) - 0.5
    if lowpass_applied <= 0.5:
        raise ValueError(f"Sampling rate is too low for 0.5--45 Hz filtering: {sampling_rate}")
    numerator, denominator = signal.butter(
        N=3,
        Wn=(0.5, lowpass_applied),
        btype="bandpass",
        fs=float(sampling_rate),
    )
    output = signal.filtfilt(numerator, denominator, output, axis=-1)
    if clip_uv is not None:
        output = np.clip(output, -float(clip_uv), float(clip_uv))
    if not np.isfinite(output).all():
        raise ValueError("NeuroRVQ EEG filtering produced non-finite values")
    return output.astype(np.float32)


def reve_pretraining_filter(values_uv: np.ndarray, sampling_rate: float) -> np.ndarray:
    """Approximate REVE pretraining's 0.5--99.5 Hz passband at source rate."""

    if sampling_rate <= 2.0:
        raise ValueError(f"Invalid sampling rate: {sampling_rate}")
    upper = min(99.5, sampling_rate / 2.0 - 0.5)
    if upper <= 0.5:
        raise ValueError(f"Sampling rate is too low for REVE filtering: {sampling_rate}")
    # REVE's passband is intentionally wider than the NeuroRVQ example; these
    # two filters must not be treated as interchangeable preprocessing paths.
    sos = signal.butter(
        N=4,
        Wn=(0.5, upper),
        btype="bandpass",
        fs=float(sampling_rate),
        output="sos",
    )
    output = signal.sosfiltfilt(sos, interpolate_nonfinite(values_uv), axis=-1)
    if not np.isfinite(output).all():
        raise ValueError("REVE filtering produced non-finite values")
    return output.astype(np.float32)


def session_zscore_clip(
    values: np.ndarray,
    *,
    clip_standard_deviations: float = 15.0,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-session/channel z-score followed by the REVE ±15 SD clip."""

    array = interpolate_nonfinite(values)
    mean = array.mean(axis=-1, keepdims=True)
    scale = array.std(axis=-1, keepdims=True)
    scale = np.where(np.isfinite(scale) & (scale > epsilon), scale, 1.0)
    standardized = np.clip(
        (array - mean) / scale,
        -float(clip_standard_deviations),
        float(clip_standard_deviations),
    )
    if not np.isfinite(standardized).all():
        raise ValueError("Session standardization produced non-finite values")
    return standardized.astype(np.float32), mean[:, 0], scale[:, 0]


def crop_indexes_with_context(
    timestamps: np.ndarray,
    start: float,
    end: float,
    *,
    context_seconds: float,
) -> tuple[slice, slice]:
    """Return a padded source slice and the central-window slice within it."""

    stamps = np.asarray(timestamps, dtype=np.float64)
    if stamps.ndim != 1 or len(stamps) < 2 or not np.all(np.diff(stamps) >= 0):
        raise ValueError("Timestamps must be a monotonic one-dimensional array")
    if not end > start:
        raise ValueError("Window end must be later than start")
    # Filtering a padded segment reduces edge transients.  Only the central
    # requested window is returned to the encoder after filtering.
    padded_left = int(np.searchsorted(stamps, start - context_seconds, side="left"))
    padded_right = int(np.searchsorted(stamps, end + context_seconds, side="left"))
    center_left = int(np.searchsorted(stamps, start, side="left"))
    center_right = int(np.searchsorted(stamps, end, side="left"))
    if center_right - center_left < 100:
        raise ValueError("The requested physiological window is unexpectedly short")
    outer = slice(padded_left, padded_right)
    inner = slice(center_left - padded_left, center_right - padded_left)
    return outer, inner


def basic_signal_qc(values: np.ndarray, sampling_rate: float) -> dict[str, float]:
    """Compact deterministic QC statistics for one channel-first window."""

    array = interpolate_nonfinite(values)
    frequencies, psd = signal.welch(
        array,
        fs=float(sampling_rate),
        nperseg=min(array.shape[-1], int(round(sampling_rate * 2.0))),
        axis=-1,
    )

    def band_power(low: float, high: float) -> float:
        selected = (frequencies >= low) & (frequencies < high)
        if int(selected.sum()) < 2:
            return 0.0
        return float(np.mean(np.trapz(psd[:, selected], frequencies[selected], axis=-1)))

    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "rms": float(np.sqrt(np.mean(np.square(array)))),
        "absolute_max": float(np.max(np.abs(array))),
        "delta_power_0p5_4": band_power(0.5, 4.0),
        "theta_power_4_8": band_power(4.0, 8.0),
        "alpha_power_8_13": band_power(8.0, 13.0),
        "beta_power_13_30": band_power(13.0, 30.0),
        "line_power_49_51": band_power(49.0, 51.0),
    }


__all__ = [
    "LINKED_EEG_CHANNELS",
    "RAW_ECG_COLUMNS",
    "RAW_EEG_CHANNELS",
    "RAW_EEG_COLUMNS",
    "TARGET_SAMPLING_RATE",
    "basic_signal_qc",
    "crop_indexes_with_context",
    "exact_resample",
    "interpolate_nonfinite",
    "linked_mastoid_reference",
    "neurorvq_eeg_filter",
    "reve_pretraining_filter",
    "select_lead_i_like_ecg_uv",
    "select_raw_eeg",
    "session_zscore_clip",
    "window_zscore",
]

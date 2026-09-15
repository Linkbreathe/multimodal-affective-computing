"""Papagei-style PPG preprocessing: Chebyshev II bandpass + resampling to 125 Hz."""
from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import cheby2, filtfilt, resample_poly


def chebyshev_bandpass_filter(
    signal: np.ndarray,
    fs: int,
    f_low: float = 0.5,
    f_high: float = 12.0,
    order: int = 4,
    smoothing_ms: float = 50.0,
) -> np.ndarray:
    """Chebyshev Type II bandpass + moving-average smoothing (replicates pyPPG).

    Parameters
    ----------
    signal : 1-D array of raw PPG values.
    fs : Sampling frequency in Hz.
    f_low, f_high : Bandpass cutoff frequencies in Hz.
    order : Filter order.
    smoothing_ms : Moving-average window in milliseconds.
    """
    b, a = cheby2(order, 20, [f_low, f_high], "bandpass", fs=fs)
    filtered = filtfilt(b, a, signal)

    if fs >= 75:
        win = round(fs * smoothing_ms / 1000)
        if win >= 1:
            B = (1.0 / win) * np.ones(win)
            filtered = filtfilt(B, 1, filtered)

    return filtered


def resample_to_target(
    signal: np.ndarray,
    fs_original: int,
    fs_target: int = 125,
) -> np.ndarray:
    """Polyphase resampling via scipy (matches Papagei's ResampleSignal)."""
    if fs_original == fs_target:
        return signal.copy()
    g = gcd(fs_original, fs_target)
    return resample_poly(signal, up=fs_target // g, down=fs_original // g)


def compute_segment_quality(
    signal: np.ndarray,
    fs: int,
    segment_s: float = 10.0,
    min_std: float = 0.01,
) -> np.ndarray:
    """Return boolean mask (True = good) for each non-overlapping segment."""
    seg_len = int(fs * segment_s)
    n_segments = len(signal) // seg_len
    good = np.ones(n_segments, dtype=bool)
    for i in range(n_segments):
        chunk = signal[i * seg_len : (i + 1) * seg_len]
        if np.any(np.isnan(chunk)) or np.std(chunk) < min_std:
            good[i] = False
    return good


def detect_native_sample_rate(
    signal_len: int,
    signal_90fps_len: int,
) -> int:
    """Infer native Fs from the ratio of raw to 90fps signal lengths."""
    fs = round(signal_len / signal_90fps_len * 90)
    known = {128, 256, 500, 512, 1000, 1024}
    if fs not in known:
        raise ValueError(
            f"Detected Fs={fs} Hz is not in known rates {sorted(known)}. "
            f"Lengths: raw={signal_len}, 90fps={signal_90fps_len}"
        )
    return fs


def preprocess_ppg(
    raw_signal: np.ndarray,
    fs_native: int,
    fs_target: int = 125,
    f_low: float = 0.5,
    f_high: float = 12.0,
    filter_order: int = 4,
    smoothing_ms: float = 50.0,
) -> tuple[np.ndarray, dict]:
    """Full pipeline: bandpass filter at native Fs, then resample to target.

    Returns (preprocessed_signal, metadata_dict).
    """
    filtered = chebyshev_bandpass_filter(
        raw_signal, fs_native, f_low, f_high, filter_order, smoothing_ms
    )
    resampled = resample_to_target(filtered, fs_native, fs_target)

    metadata = {
        "fs_native": fs_native,
        "fs_target": fs_target,
        "filter": f"cheby2_order{filter_order}_bp{f_low}-{f_high}Hz",
        "smoothing_ms": smoothing_ms,
        "samples_raw": len(raw_signal),
        "samples_output": len(resampled),
        "duration_s": round(len(resampled) / fs_target, 2),
    }
    return resampled, metadata

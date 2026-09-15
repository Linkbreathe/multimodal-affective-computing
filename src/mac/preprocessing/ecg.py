"""Preprocessing helpers for ECGFounder ECG feature extraction."""
from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly


def preprocess_ecgfounder_segment(
    signal: np.ndarray,
    source_fs: int = 90,
    target_fs: int = 500,
    duration_sec: int = 10,
) -> np.ndarray:
    """Convert one ECG segment into ECGFounder input format.

    Args:
        signal: One ECG segment as ``[T]``, ``[T, 1]``, or ``[1, T]``.
        source_fs: Sampling rate of the segment. Local egoEMOTION ECG is aligned
            to the 90 Hz timeline.
        target_fs: ECGFounder expects 500 Hz.
        duration_sec: Expected segment duration.

    Returns:
        Float32 array shaped ``[1, target_fs * duration_sec]``.
    """
    ecg = np.asarray(signal, dtype=np.float32)
    if ecg.ndim == 2:
        if ecg.shape[0] == 1:
            ecg = ecg[0]
        elif ecg.shape[1] == 1:
            ecg = ecg[:, 0]
        else:
            raise ValueError("Expected a single ECG channel.")
    if ecg.ndim != 1:
        raise ValueError("Expected ECG input shaped [T], [T, 1], or [1, T].")

    ecg = np.nan_to_num(ecg, nan=0.0, posinf=0.0, neginf=0.0)
    target_len = int(target_fs * duration_sec)
    expected_source_len = int(source_fs * duration_sec)
    if len(ecg) != expected_source_len:
        raise ValueError(
            f"Expected {expected_source_len} ECG samples for {duration_sec}s at "
            f"{source_fs} Hz, got {len(ecg)}."
        )

    if source_fs != target_fs:
        factor = gcd(source_fs, target_fs)
        ecg = resample_poly(ecg, up=target_fs // factor, down=source_fs // factor)
    if len(ecg) != target_len:
        ecg = np.interp(
            np.linspace(0.0, 1.0, target_len, endpoint=False),
            np.linspace(0.0, 1.0, len(ecg), endpoint=False),
            ecg,
        )

    mean = float(ecg.mean())
    std = float(ecg.std())
    if std > 1e-8:
        ecg = (ecg - mean) / std
    else:
        ecg = ecg - mean

    return ecg.astype(np.float32, copy=False)[np.newaxis, :]

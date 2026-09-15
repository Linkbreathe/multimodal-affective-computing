from __future__ import annotations

from typing import Any

import numpy as np

from real_time_ml.features.physio import counter_qc


def infer_microvolt_scale(eeg: np.ndarray) -> tuple[np.ndarray, str]:
    values = np.asarray(eeg, dtype=float)
    robust = float(np.nanmedian(np.abs(values - np.nanmedian(values, axis=0))))
    if robust < 0.01:
        return values * 1e6, "V_to_uV"
    return values, "already_uV"


def continuous_mne_audit(
    samples: np.ndarray,
    sample_rate: float,
    eeg_columns: list[int],
    counter_column: int = 0,
    channel_names: list[str] | None = None,
) -> dict[str, Any]:
    """Continuous MNE audit only; training features never come from this path."""
    import mne

    # Raw electrode identity per eeg_columns order: [M2, TP9, TP10, M1] (M1/M2 are mastoid
    # references; TP9/TP10 the temporo-parietal signals). Audit-only path; not a feature source.
    names = channel_names or ["M2", "TP9", "TP10", "M1"]
    eeg_uV, unit_action = infer_microvolt_scale(np.asarray(samples)[:, eeg_columns])
    info = mne.create_info(names, sample_rate, ch_types="eeg")
    raw = mne.io.RawArray((eeg_uV * 1e-6).T, info, verbose="ERROR")
    filtered = raw.copy().filter(1.0, 45.0, method="iir", verbose="ERROR")
    data_uV = filtered.get_data() * 1e6
    high_frequency = raw.copy().filter(30.0, min(90.0, sample_rate * 0.45), method="iir", verbose="ERROR").get_data() * 1e6
    muscle_ratio = np.sqrt(np.mean(high_frequency**2, axis=1)) / np.maximum(np.sqrt(np.mean(data_uV**2, axis=1)), 1e-9)
    return {
        "unit_action": unit_action,
        "channel_names": names,
        "channel_rms_uV": {name: float(np.sqrt(np.mean(data_uV[i] ** 2))) for i, name in enumerate(names)},
        "muscle_suspicion_ratio": {name: float(muscle_ratio[i]) for i, name in enumerate(names)},
        **counter_qc(np.asarray(samples)[:, counter_column]),
    }


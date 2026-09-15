from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal

from mac.features.common import hjorth, safe_divide, spectral_entropy


def _sos_bandpass(low: float, high: float, sample_rate: float, order: int = 4) -> np.ndarray:
    return signal.butter(order, [low, high], btype="bandpass", fs=sample_rate, output="sos")


def causal_filter(values: np.ndarray, low: float, high: float, sample_rate: float) -> np.ndarray:
    values = signal.detrend(np.asarray(values, dtype=float), axis=0)
    return signal.sosfilt(_sos_bandpass(low, high, sample_rate), values, axis=0)


def counter_qc(counter: np.ndarray) -> dict[str, Any]:
    counter = np.asarray(counter, dtype=float)
    if counter.size < 2:
        return {"counter_bad_fraction": 1.0, "counter_discontinuities": 0}
    differences = np.diff(counter)
    positive = differences[differences > 0]
    if positive.size == 0:
        return {"counter_bad_fraction": 1.0, "counter_discontinuities": int(differences.size)}
    step = float(np.median(positive))
    bad = ~np.isclose(differences, step, atol=max(1e-3, abs(step) * 0.05))
    return {"counter_bad_fraction": float(np.mean(bad)), "counter_discontinuities": int(np.sum(bad))}


def eeg_quality_coverage(eeg_uV: np.ndarray, sample_rate: float, abs_uV_max: float, flat_std_uV_min: float) -> float:
    block = max(1, int(round(2.0 * sample_rate)))
    usable = []
    filtered = causal_filter(eeg_uV, 1.0, min(45.0, sample_rate * 0.45), sample_rate)
    for start in range(0, len(filtered) - block + 1, block):
        piece = filtered[start : start + block]
        channel_std = np.std(piece, axis=0)
        channel_peak = np.percentile(np.abs(piece), 99, axis=0)
        good = np.isfinite(piece).all(axis=0) & (channel_std >= flat_std_uV_min) & (channel_peak <= abs_uV_max)
        usable.append(float(np.mean(good) >= 0.75))
    return float(np.mean(usable)) if usable else 0.0


@dataclass(frozen=True)
class EEGMontage:
    """Electrode identity for the four raw EEG columns, aligned to ``eeg_columns``.

    The hardware records four electrodes per sample: two mastoid references (M1/M2) and
    two temporo-parietal signal electrodes (TP9 left, TP10 right). ``names``, ``roles`` and
    ``hemispheres`` are index-for-index with the raw column slice handed to
    :func:`eeg_features`. Only ``role == "signal"`` channels produce per-channel features;
    the ``reference`` channels form the reference that is subtracted from the signals.
    """

    names: tuple[str, ...]
    roles: tuple[str, ...]
    hemispheres: tuple[str, ...]
    reference_scheme: str = "linked_mastoid"

    @classmethod
    def from_config(cls, config: Any) -> "EEGMontage":
        names = tuple(str(name) for name in config.get("streams.eeg_names"))
        roles = tuple(str(role) for role in config.get("features.eeg.channel_roles"))
        hemispheres = tuple(str(side) for side in config.get("features.eeg.channel_hemispheres"))
        scheme = str(config.get("features.eeg.reference_scheme"))
        if not (len(names) == len(roles) == len(hemispheres)):
            raise ValueError("streams.eeg_names, features.eeg.channel_roles and channel_hemispheres must be equal length")
        return cls(names=names, roles=roles, hemispheres=hemispheres, reference_scheme=scheme)

    def _indices(self, role: str) -> list[int]:
        return [index for index, value in enumerate(self.roles) if value == role]

    @property
    def signal_indices(self) -> list[int]:
        return self._indices("signal")

    @property
    def reference_indices(self) -> list[int]:
        return self._indices("reference")


def apply_reference(values: np.ndarray, montage: EEGMontage) -> tuple[np.ndarray, list[str], list[str]]:
    """Re-reference the raw electrode matrix and return (signal_matrix, names, hemispheres).

    The returned matrix has one column per signal electrode, in ``eeg_columns`` order.
    """
    values = np.asarray(values, dtype=float)
    signal_index = montage.signal_indices
    reference_index = montage.reference_indices
    if not signal_index:
        raise ValueError("EEG montage defines no signal channels")
    scheme = montage.reference_scheme
    if scheme == "none" or not reference_index:
        referenced = values[:, signal_index]
    elif scheme == "linked_mastoid":
        reference = np.mean(values[:, reference_index], axis=1, keepdims=True)
        referenced = values[:, signal_index] - reference
    elif scheme == "ipsilateral_mastoid":
        columns = []
        for channel in signal_index:
            hemisphere = montage.hemispheres[channel]
            matched = [ref for ref in reference_index if montage.hemispheres[ref] == hemisphere]
            reference = np.mean(values[:, matched], axis=1) if matched else np.zeros(len(values))
            columns.append(values[:, channel] - reference)
        referenced = np.column_stack(columns)
    else:
        raise ValueError(f"Unknown EEG reference_scheme: {scheme!r}")
    names = [montage.names[index].lower() for index in signal_index]
    hemispheres = [montage.hemispheres[index] for index in signal_index]
    return referenced, names, hemispheres


def eeg_features(eeg_uV: np.ndarray, sample_rate: float, bands: dict[str, list[float]], montage: EEGMontage, car: bool = False) -> dict[str, float]:
    referenced, names, hemispheres = apply_reference(eeg_uV, montage)
    if car:
        referenced = referenced - np.mean(referenced, axis=1, keepdims=True)
    filtered = causal_filter(referenced, 1.0, min(45.0, sample_rate * 0.45), sample_rate)
    output: dict[str, float] = {}
    band_values: dict[str, list[float]] = {name: [] for name in bands}
    for channel_index, channel_name in enumerate(names):
        frequencies, psd = signal.welch(filtered[:, channel_index], fs=sample_rate, nperseg=min(len(filtered), int(sample_rate * 2)))
        total_mask = (frequencies >= 1.0) & (frequencies <= 45.0)
        total = float(np.trapezoid(psd[total_mask], frequencies[total_mask]))
        for band_name, limits in bands.items():
            mask = (frequencies >= limits[0]) & (frequencies < limits[1])
            absolute = float(np.trapezoid(psd[mask], frequencies[mask])) if np.any(mask) else float("nan")
            output[f"eeg_{channel_name}_{band_name}_power"] = absolute
            output[f"eeg_{channel_name}_{band_name}_relative"] = safe_divide(absolute, total)
            band_values[band_name].append(absolute)
        activity, mobility, complexity = hjorth(filtered[:, channel_index])
        output[f"eeg_{channel_name}_spectral_entropy"] = spectral_entropy(psd[total_mask])
        output[f"eeg_{channel_name}_hjorth_activity"] = activity
        output[f"eeg_{channel_name}_hjorth_mobility"] = mobility
        output[f"eeg_{channel_name}_hjorth_complexity"] = complexity
        output[f"eeg_{channel_name}_robust_amplitude_uV"] = float(np.percentile(filtered[:, channel_index], 95) - np.percentile(filtered[:, channel_index], 5))
    alpha = float(np.nanmean(band_values.get("alpha", [np.nan])))
    beta = float(np.nanmean(band_values.get("beta", [np.nan])))
    theta = float(np.nanmean(band_values.get("theta", [np.nan])))
    output["eeg_alpha_beta_ratio"] = safe_divide(alpha, beta)
    output["eeg_theta_beta_ratio"] = safe_divide(theta, beta)
    alpha_by_channel = band_values.get("alpha", [np.nan] * len(names))
    left_alpha = np.nanmean([alpha_by_channel[i] for i, side in enumerate(hemispheres) if side == "left"] or [np.nan])
    right_alpha = np.nanmean([alpha_by_channel[i] for i, side in enumerate(hemispheres) if side == "right"] or [np.nan])
    output["eeg_alpha_asymmetry_log_right_left"] = float(np.log(right_alpha + 1e-12) - np.log(left_alpha + 1e-12))
    return output


def detect_r_peaks(ecg_uV: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    filtered = causal_filter(np.asarray(ecg_uV, dtype=float), 5.0, min(25.0, sample_rate * 0.45), sample_rate)
    derivative = np.diff(filtered, prepend=filtered[0])
    energy = signal.sosfilt(signal.butter(2, 8.0, btype="lowpass", fs=sample_rate, output="sos"), derivative**2)
    threshold = float(np.median(energy) + 3.0 * np.median(np.abs(energy - np.median(energy))))
    peaks, _ = signal.find_peaks(energy, distance=int(0.28 * sample_rate), height=max(threshold, np.finfo(float).eps))
    return peaks.astype(int), filtered


def ecg_features(ecg_uV: np.ndarray, sample_rate: float) -> tuple[dict[str, float], dict[str, Any]]:
    peaks, filtered = detect_r_peaks(ecg_uV, sample_rate)
    rr_ms = np.diff(peaks) / sample_rate * 1000.0
    plausible = rr_ms[(rr_ms >= 60_000 / 220.0) & (rr_ms <= 60_000 / 35.0)]
    quality = float(plausible.size / rr_ms.size) if rr_ms.size else 0.0
    median_rr = float(np.median(plausible)) if plausible.size else float("nan")
    features = {
        "ecg_peak_count": float(len(peaks)),
        "ecg_rr_mean_ms": float(np.mean(plausible)) if plausible.size else float("nan"),
        "ecg_rr_median_ms": median_rr,
        "ecg_hr_bpm": safe_divide(60_000.0, median_rr),
        "ecg_rr_std_ms_audit_only": float(np.std(plausible)) if plausible.size >= 2 else float("nan"),
        "ecg_signal_iqr_uV": float(np.percentile(filtered, 75) - np.percentile(filtered, 25)),
    }
    return features, {"ecg_quality": quality, "ecg_usable": bool(plausible.size >= 5 and quality >= 0.7)}


def hrv_features(r_peak_times_seconds: np.ndarray, horizon_seconds: float) -> dict[str, float]:
    peaks = np.asarray(r_peak_times_seconds, dtype=float)
    if peaks.size < 3 or peaks[-1] - peaks[0] < horizon_seconds * 0.8:
        return {}
    cutoff = peaks[-1] - horizon_seconds
    rr = np.diff(peaks[peaks >= cutoff]) * 1000.0
    if rr.size < 2:
        return {}
    diff = np.diff(rr)
    prefix = f"ecg_hrv_{int(horizon_seconds)}s"
    return {
        f"{prefix}_rmssd_ms": float(np.sqrt(np.mean(diff**2))),
        f"{prefix}_sdnn_ms": float(np.std(rr, ddof=1)),
        f"{prefix}_pnn50": float(np.mean(np.abs(diff) > 50.0)) if diff.size else float("nan"),
    }


def peak_f1(reference_seconds: np.ndarray, detected_seconds: np.ndarray, tolerance_seconds: float = 0.1) -> float:
    reference = list(np.sort(np.asarray(reference_seconds, dtype=float)))
    detected = list(np.sort(np.asarray(detected_seconds, dtype=float)))
    used: set[int] = set()
    matches = 0
    for ref in reference:
        candidates = [(abs(value - ref), index) for index, value in enumerate(detected) if index not in used and abs(value - ref) <= tolerance_seconds]
        if candidates:
            _, index = min(candidates)
            used.add(index)
            matches += 1
    precision = matches / len(detected) if detected else 0.0
    recall = matches / len(reference) if reference else 0.0
    return safe_divide(2 * precision * recall, precision + recall) if precision + recall else 0.0


@dataclass
class StreamingPhysioProcessor:
    sample_rate: float
    eeg_columns: list[int]
    ecg_columns: list[int]
    counter_column: int
    bands: dict[str, list[float]]
    montage: EEGMontage
    eeg_disabled: bool = False
    strict_coverage_min: float = 0.60
    eeg_abs_uV_max: float = 350.0
    eeg_flat_std_uV_min: float = 0.2
    car: bool = False

    def process_window(self, samples: np.ndarray) -> tuple[dict[str, float], dict[str, Any]]:
        values = np.asarray(samples, dtype=float)
        expected = int(round(self.sample_rate * 10.0))
        qc: dict[str, Any] = {"sample_count": int(len(values)), "expected_sample_count": expected}
        qc.update(counter_qc(values[:, self.counter_column]))
        features: dict[str, float] = {}
        ecg = values[:, self.ecg_columns[0]] - values[:, self.ecg_columns[1]]
        ecg_values, ecg_qc = ecg_features(ecg, self.sample_rate)
        features.update(ecg_values)
        qc.update(ecg_qc)
        if self.eeg_disabled:
            qc.update({"eeg_usable": False, "eeg_disabled_by_participant_qc": True, "eeg_strict_coverage": 0.0})
        else:
            eeg = values[:, self.eeg_columns]
            coverage = eeg_quality_coverage(eeg, self.sample_rate, self.eeg_abs_uV_max, self.eeg_flat_std_uV_min)
            qc.update({"eeg_strict_coverage": coverage, "eeg_usable": coverage >= self.strict_coverage_min, "eeg_disabled_by_participant_qc": False})
            if qc["eeg_usable"]:
                features.update(eeg_features(eeg, self.sample_rate, self.bands, self.montage, self.car))
        qc["physio_complete"] = len(values) >= int(expected * 0.98)
        return features, qc


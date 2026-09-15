from __future__ import annotations

import numpy as np


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if np.isfinite(denominator) and abs(denominator) > 1e-12 else float("nan")


def robust_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
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


def spectral_entropy(power: np.ndarray) -> float:
    power = np.asarray(power, dtype=float)
    power = power[np.isfinite(power) & (power > 0)]
    if power.size < 2 or power.sum() <= 0:
        return float("nan")
    probabilities = power / power.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)) / np.log2(probabilities.size))


def hjorth(signal: np.ndarray) -> tuple[float, float, float]:
    signal = np.asarray(signal, dtype=float)
    d1, d2 = np.diff(signal), np.diff(signal, n=2)
    variance = float(np.var(signal))
    mobility = safe_divide(np.sqrt(np.var(d1)), np.sqrt(variance))
    complexity = safe_divide(safe_divide(np.sqrt(np.var(d2)), np.sqrt(np.var(d1))), mobility)
    return variance, mobility, complexity


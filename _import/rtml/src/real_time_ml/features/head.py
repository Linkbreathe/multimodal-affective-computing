from __future__ import annotations

import numpy as np
from scipy import signal

from real_time_ml.features.common import robust_stats, spectral_entropy


def head_features(rows: list[dict[str, float]]) -> tuple[dict[str, float], dict[str, float | bool]]:
    required = ("unix_time_ms", "head_position_x", "head_position_y", "head_position_z")
    valid = [row for row in rows if all(np.isfinite(row.get(name, np.nan)) for name in required)]
    if len(valid) < 3:
        return {}, {"head_coverage": 0.0, "head_usable": False}
    time_s = np.asarray([row["unix_time_ms"] for row in valid]) / 1000.0
    position = np.asarray([[row[f"head_position_{axis}"] for axis in "xyz"] for row in valid], dtype=float)
    dt = np.diff(time_s)
    keep = dt > 1e-4
    velocity = np.linalg.norm(np.diff(position, axis=0)[keep] / dt[keep, None], axis=1)
    angular = np.asarray([row.get("head_angular_velocity_deg_s", np.nan) for row in valid], dtype=float)
    angular = angular[np.isfinite(angular)]
    acceleration = np.diff(velocity) / dt[keep][1:] if velocity.size > 1 else np.asarray([])
    jerk = np.diff(acceleration) / dt[keep][2:] if acceleration.size > 1 else np.asarray([])
    features = {}
    features.update(robust_stats(velocity, "head_speed"))
    features.update(robust_stats(angular, "head_angular_speed_deg_s"))
    features.update(robust_stats(jerk, "head_jerk"))
    features["head_stationary_fraction"] = float(np.mean(velocity < 0.01)) if velocity.size else float("nan")
    features["head_position_range"] = float(np.linalg.norm(np.ptp(position, axis=0)))
    if velocity.size >= 8:
        fs = 1.0 / np.median(dt[keep])
        _, psd = signal.welch(velocity, fs=fs, nperseg=min(len(velocity), 64))
        features["head_motion_spectral_entropy"] = spectral_entropy(psd)
    coverage = min(1.0, (time_s[-1] - time_s[0]) / 10.0)
    return features, {"head_coverage": float(coverage), "head_usable": bool(coverage >= 0.6)}


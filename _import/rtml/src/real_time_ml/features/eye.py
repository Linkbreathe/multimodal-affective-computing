from __future__ import annotations

import numpy as np

from real_time_ml.features.common import robust_stats, spectral_entropy


def eye_features(rows: list[dict[str, float]], velocity_threshold_deg_s: float = 100.0) -> tuple[dict[str, float], dict[str, float | bool]]:
    required = ("unix_time_ms", "gaze_direction_x", "gaze_direction_y", "gaze_direction_z")
    valid = [row for row in rows if all(np.isfinite(row.get(name, np.nan)) for name in required)]
    total = max(1, len(rows))
    valid_fraction = len(valid) / total
    if len(valid) < 3:
        return {"eye_valid_fraction": valid_fraction}, {"eye_coverage": valid_fraction, "eye_usable": False}
    time_s = np.asarray([row["unix_time_ms"] for row in valid]) / 1000.0
    gaze = np.asarray([[row[f"gaze_direction_{axis}"] for axis in "xyz"] for row in valid], dtype=float)
    gaze /= np.maximum(np.linalg.norm(gaze, axis=1, keepdims=True), 1e-12)
    dt = np.diff(time_s)
    good = dt > 1e-4
    angles = np.degrees(np.arccos(np.clip(np.sum(gaze[1:] * gaze[:-1], axis=1), -1.0, 1.0)))
    velocity = angles[good] / dt[good]
    saccade = velocity >= velocity_threshold_deg_s
    mean_direction = np.mean(gaze, axis=0)
    dispersion = np.degrees(np.arccos(np.clip(gaze @ (mean_direction / np.linalg.norm(mean_direction)), -1.0, 1.0)))
    features = {"eye_valid_fraction": float(valid_fraction)}
    features.update(robust_stats(velocity, "eye_angular_velocity_deg_s"))
    features["eye_fixation_fraction_ivt"] = float(np.mean(~saccade)) if saccade.size else float("nan")
    features["eye_saccade_fraction_ivt"] = float(np.mean(saccade)) if saccade.size else float("nan")
    features["eye_scanpath_deg"] = float(np.sum(angles[good]))
    features["eye_direction_dispersion_deg"] = float(np.mean(dispersion))
    histogram, _, _ = np.histogram2d(gaze[:, 0], gaze[:, 1], bins=8)
    features["eye_spatial_entropy"] = spectral_entropy(histogram.ravel())
    # gaze_on_painting is never populated at collection, so eye_gaze_on_painting_fraction was a
    # constant (0 variance across all windows) dead feature; it is no longer emitted. Restore it
    # only after fixing gaze_on_painting capture/export upstream.
    coverage = min(valid_fraction, (time_s[-1] - time_s[0]) / 10.0)
    return features, {"eye_coverage": float(coverage), "eye_usable": bool(coverage >= 0.6)}

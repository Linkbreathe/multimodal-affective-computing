"""Time-weighted gaze/head descriptions from the retained Unity samples."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from .common import feature_result, truthy
from .specification import SPEC


def unit_vectors(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norm = np.linalg.norm(values, axis=1)
    valid = np.isfinite(values).all(axis=1) & (norm > 1e-12)
    result = np.zeros_like(values, dtype=float)
    result[valid] = values[valid] / norm[valid, None]
    return result, valid


def rotation_angles(quaternions: np.ndarray) -> np.ndarray:
    dots = np.abs(np.sum(quaternions[1:] * quaternions[:-1], axis=1))
    return np.degrees(2 * np.arccos(np.clip(dots, 0, 1)))


def endpoint_weights(dt: np.ndarray, valid_pairs: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(dt) + 1)
    weights[:-1] += np.where(valid_pairs, dt / 2, 0)
    weights[1:] += np.where(valid_pairs, dt / 2, 0)
    return weights


def behavioral_features(
    frame: pd.DataFrame,
    modality: str,
    start_ms: float,
    end_ms: float,
    presence: pd.DataFrame | None = None,
) -> tuple[list[dict], dict]:
    names = (
        ["gaze_mean_angular_velocity", "gaze_fixation_time_proportion"]
        if modality == "eye"
        else ["head_mean_angular_speed", "head_orientation_dispersion"]
    )
    if frame.empty:
        return [feature_result(f, reason="missing_source") for f in names], {}
    f = frame.loc[frame.unix_time_ms.ge(start_ms) & frame.unix_time_ms.lt(end_ms)].copy()
    duration = (end_ms - start_ms) / 1000
    if len(f) < 3:
        return [
            feature_result(
                x,
                reason="fewer_than_three_samples",
                sample_count=len(f),
                target_duration_s=duration,
            )
            for x in names
        ], {}
    time = f.unix_time_ms.to_numpy(float) / 1000
    dt = np.diff(time)
    prefix, axes = ("gaze_direction_", "xyz") if modality == "eye" else ("head_rotation_", "xyzw")
    directions, valid = unit_vectors(f[[prefix + a for a in axes]].to_numpy(float))
    flag = "gaze_available" if modality == "eye" else "head_pose_available"
    valid &= truthy(f[flag]) if flag in f else False
    if "formal_viewing" in f:
        valid &= truthy(f.formal_viewing)
    if "headset_user_present" in f:
        valid &= truthy(f.headset_user_present)
    elif presence is not None and "headset_user_present" in presence:
        mapping = (
            presence.drop_duplicates("unix_time_ms").set_index("unix_time_ms").headset_user_present
        )
        valid &= truthy(pd.Series(f.unix_time_ms.map(mapping)))
    pairs = (
        valid[:-1]
        & valid[1:]
        & np.isfinite(dt)
        & (dt > 0)
        & (dt <= SPEC["time"]["behavior_max_gap_s"])
    )
    good_time = float(dt[pairs].sum())
    coverage = good_time / duration
    quality = dict(
        sample_count=len(f),
        valid_sample_count=int(valid.sum()),
        valid_pair_count=int(pairs.sum()),
        target_duration_s=duration,
        valid_duration_s=good_time,
        valid_coverage=coverage,
        median_sampling_interval_s=float(np.median(dt[dt > 0])) if (dt > 0).any() else np.nan,
        rejected_gap_count=int((dt > SPEC["time"]["behavior_max_gap_s"]).sum()),
        nonpositive_interval_count=int((dt <= 0).sum()),
    )
    diagnostics = dict(modality=modality, **quality)
    if modality == "eye":
        quality["warning"] = "gaze_source_unknown;low_rate_ivt_proxy;smooth_pursuit_not_separated"
        quality["gaze_source_identifiable"] = False
        if all("gaze_hit_" + a in f and "gaze_origin_" + a in f for a in "xyz"):
            ray = f[["gaze_hit_" + a for a in "xyz"]].to_numpy(float) - f[
                ["gaze_origin_" + a for a in "xyz"]
            ].to_numpy(float)
            ray, ok = unit_vectors(ray)
            ok &= valid
            if "gaze_hit" in f:
                ok &= truthy(f.gaze_hit)
            if ok.any():
                alignment = np.degrees(
                    np.arccos(np.clip(np.sum(ray[ok] * directions[ok], axis=1), -1, 1))
                )
                diagnostics["gaze_hit_ray_angle_median_deg"] = float(np.median(alignment))
                diagnostics["gaze_hit_ray_angle_p99_deg"] = float(np.quantile(alignment, 0.99))
    if coverage < SPEC["time"]["coverage_min"] or pairs.sum() < 2:
        return [
            feature_result(x, reason="insufficient_valid_duration", **quality) for x in names
        ], diagnostics
    if modality == "eye":
        angle = np.degrees(
            np.arccos(np.clip(np.sum(directions[:-1] * directions[1:], axis=1), -1, 1))
        )
        velocity = angle[pairs] / dt[pairs]
        values = [
            float(angle[pairs].sum() / good_time),
            float(dt[pairs][velocity < SPEC["eye"]["threshold_deg_s"]].sum() / good_time),
        ]
    else:
        angle = rotation_angles(directions)
        speed = float(angle[pairs].sum() / good_time)
        weights = endpoint_weights(dt, pairs)
        rotations = Rotation.from_quat(directions[weights > 0])
        mean = rotations.mean(weights=weights[weights > 0])
        deviations = np.degrees((mean.inv() * rotations).magnitude())
        dispersion = float(np.sqrt(np.average(deviations**2, weights=weights[weights > 0])))
        values = [speed, dispersion]
    return [
        feature_result(name, value, **quality) for name, value in zip(names, values)
    ], diagnostics

"""Outcome-independent, versioned computational contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PARTICIPANTS = ("P003", "P004", "P007", "P008", "P009", "P011", "P012", "P013", "P015")
CONDITIONS = tuple(f"C{i}" for i in range(1, 10))
OUTCOMES = ("relaxation", "discomfort")
FEATURES = {
    "eeg_alpha_power": ("EEG", "µV²", "ln", "Alpha power, TP9/TP10 equal mean"),
    "eeg_beta_power": ("EEG", "µV²", "ln", "Beta power, TP9/TP10 equal mean"),
    "eeg_theta_power": ("EEG", "µV²", "ln", "Theta power, TP9/TP10 equal mean"),
    "ecg_mean_hr": ("ECG", "beats/min", "identity", "Time-weighted mean instantaneous heart rate"),
    "ecg_rmssd": ("ECG", "ms", "ln", "RMSSD of originally adjacent accepted RR intervals"),
    "gaze_mean_angular_velocity": (
        "Eye",
        "deg/s",
        "identity",
        "Time-weighted logged gaze angular velocity",
    ),
    "gaze_fixation_time_proportion": (
        "Eye",
        "proportion",
        "identity",
        "Valid time with angular velocity <100 deg/s",
    ),
    "head_mean_angular_speed": (
        "Head",
        "deg/s",
        "identity",
        "Time-weighted quaternion angular speed",
    ),
    "head_orientation_dispersion": (
        "Head",
        "deg",
        "identity",
        "Weighted RMS geodesic distance from chordal rotation mean",
    ),
    "video_mean_optical_flow": (
        "Video",
        "320x180 pixels/s",
        "identity",
        "Time-weighted mean spatial-average flow speed",
    ),
    "video_optical_flow_temporal_sd": (
        "Video",
        "320x180 pixels/s",
        "identity",
        "Time-weighted population SD of spatial-average flow speed",
    ),
}
VARIABLES = (*OUTCOMES, *FEATURES)
SPEC = {
    "version": "participant_exploration_v1",
    "status": "post_hoc_exploratory_after_main_prediction_analyses",
    "participants": PARTICIPANTS,
    "conditions": CONDITIONS,
    "features": FEATURES,
    "intensities": [0.08, 0.16, 0.25],
    "frequencies": [0.12, 0.26, 0.41],
    "participant_aliases": {"P003": ["P001", "P003"]},
    "time": {
        "coverage_min": 0.60,
        "behavior_max_gap_s": 0.250,
        "alignment_warning_ms": 50.0,
        "alignment_exclusion_ms": 250.0,
        "sample_counter_modulus": 256,
        "physio_max_gap_samples": 3,
    },
    "eeg": {
        "columns": [1, 2, 3, 4],
        "electrodes": ["M2", "TP9", "TP10", "M1"],
        "raw_unit": "uV",
        "reference": "linked_mastoid",
        "car": False,
        "filter": "linear detrend then causal fourth-order Butterworth SOS 1-45 Hz",
        "notch_applied": False,
        "notch_discrepancy": "Configured 50 Hz notch is not applied by historical handcrafted code; preserved by user decision.",
        "filter_window_s": 10.0,
        "qc_block_s": 2.0,
        "flat_std_min_uv": 0.2,
        "abs_p99_max_uv": 350.0,
        "raw_constant_run_max_s": 0.1,
        "bands": {"alpha": [8.0, 13.0], "beta": [13.0, 30.0], "theta": [4.0, 8.0]},
        "psd": {
            "window": "periodic Hann",
            "segment_s": 2.0,
            "overlap": 0.5,
            "detrend": "constant",
            "scaling": "density",
            "average": "mean",
            "integration": "trapezoid over bins low <= f < high",
        },
        "aggregation": "PSD within contiguous jointly valid two-second blocks; duration-weight raw powers; equal TP9/TP10 mean; then natural log",
        "coverage_min": 0.60,
    },
    "ecg": {
        "columns": [7, 8],
        "derivation": "column7-column8 (ch6-ch7; LA-RA per Methods)",
        "clean_method": "neurokit",
        "highpass_hz": 0.5,
        "highpass_order": 5,
        "powerline_hz": 50,
        "peak_method": "neurokit",
        "correct_artifacts": False,
        "smoothwindow": 0.1,
        "avgwindow": 0.75,
        "gradthreshweight": 1.5,
        "minlenweight": 0.4,
        "mindelay": 0.3,
        "rr_min_ms": 60000 / 220,
        "rr_max_ms": 60000 / 35,
        "local_median_width": 5,
        "local_median_fraction": 0.20,
        "hr_min_intervals": 5,
        "hr_coverage_min": 0.60,
        "rmssd_interval_s": [5.0, 65.0],
        "rmssd_min_adjacent_pairs": 20,
        "rmssd_accepted_rr_fraction_min": 0.80,
        "rmssd_requires_continuous_recording": True,
        "gap_policy": "No interpolation, peak insertion, or RR stitching; an excluded RR breaks adjacency.",
    },
    "eye": {
        "threshold_deg_s": 100.0,
        "minimum_fixation_duration_s": None,
        "coordinate_system": "recorded world-oriented direction; compare with hit-origin geometry; no sign correction",
        "fallback": "Source flags absent; retain logged gaze as mixed-source proxy, not verified eye-only movement.",
    },
    "head": {
        "quaternion_order": "xyzw",
        "speed": "2*acos(abs(dot(unit_q_i,unit_q_next)))/dt",
        "dispersion": "RMS shortest angular distance to weighted chordal L2 mean",
        "weights": "half of each accepted adjacent interval to each endpoint",
    },
    "video": {
        "sample_fps": 2.0,
        "target_tolerance_s": 0.150,
        "width": 320,
        "height": 180,
        "resize": "INTER_AREA",
        "algorithm": "Farneback",
        "pyr_scale": 0.5,
        "levels": 2,
        "winsize": 15,
        "iterations": 2,
        "poly_n": 5,
        "poly_sigma": 1.1,
        "flags": 0,
        "max_pair_gap_s": 0.750,
        "max_capture_gap_s": 0.250,
        "units": "analysis-resolution pixels per actual elapsed second",
        "temporal_sd": "sqrt(sum(dt*(pair_speed-weighted_mean)**2)/sum(dt))",
    },
    "statistics": {
        "minimum_paired_conditions": 6,
        "correlation": "Spearman average tied ranks",
        "sd_ddof": 1,
        "quartiles": "linear interpolation",
        "zero_tolerance": 1e-12,
        "outcomes": OUTCOMES,
        "primary_outcome": "relaxation",
        "p_values": False,
        "pooled_correlations": False,
    },
    "transform": "Natural logarithm of positive values expressed in the stated unit; no epsilon. Valid raw zero retained, logarithm missing.",
}


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def spec_hash() -> str:
    return hashlib.sha256(canonical_json(SPEC).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

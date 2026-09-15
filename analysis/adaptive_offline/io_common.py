"""Shared helpers for the offline adaptive (route B + C) deliverables.

This package implements ``Auxiliary/research/adaptive_plan_offline_BC_version_zh.md``.  Every
estimate that is reported as evidence carries a participant-level confidence
interval and an explicit ``inference_unit`` so that the 946 ten-second windows
are never silently promoted to independent supervised samples.  The inference
unit for participant x condition claims is ``participant_condition`` (n=135);
participant-level claims use ``participant`` (n=15).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ANALYSIS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ANALYSIS_DIR.parent
ARTIFACTS = REPO_ROOT / "artifacts"
OUTPUT_ROOT = ARTIFACTS / "adaptive_offline"

CONDITION_LABELS_CSV = ARTIFACTS / "preprocessed" / "condition_labels.csv"
WINDOW_FEATURES_CSV = ARTIFACTS / "features" / "window_features.csv"
WINDOWS_CSV = ARTIFACTS / "preprocessed" / "windows.csv"
CONDITION_FEATURES_CSV = ARTIFACTS / "features" / "condition_features.csv"

# ---------------------------------------------------------------------------
# Study constants (mirrors configs/project.yaml; kept local to avoid importing
# the runtime package and accidentally touching the serving path).
# ---------------------------------------------------------------------------
PARTICIPANTS = (
    "P002", "P003", "P004", "P005", "P006", "P007", "P008",
    "P009", "P010", "P011", "P012", "P013", "P014", "P015", "P016",
)
EEG_DISABLED = ("P002", "P005", "P006", "P010", "P014", "P016")
CONDITIONS = tuple(f"C{i}" for i in range(1, 10))
INTENSITIES = (0.08, 0.16, 0.25)
FREQUENCIES = (0.12, 0.26, 0.41)

LABELS = ("visual_fit", "pleasantness", "calm", "relaxation", "monotony", "discomfort")
PRIMARY_TARGETS = ("relaxation", "discomfort")
# config: modeling.condition_level.high_discomfort_label_threshold
HIGH_DISCOMFORT_THRESHOLD = 0.50

INFERENCE_PC = "participant_condition"  # n = 135
INFERENCE_P = "participant"             # n = 15
INFERENCE_WINDOW = "window_feature_slice"  # n = 946, NOT inference evidence

# Phase A pre-registered negligible-heterogeneity threshold (0-1 scale).
NEGLIGIBLE_HETEROGENEITY_SD = 0.05

DEFAULT_SEED = 20260627
DEFAULT_BOOTSTRAP = 2000

# Pre-specified reduced multimodal indices for I.6 (theory-driven, NOT a free
# search over the 127 usable window-feature columns).  Each index maps to a
# small set of source columns that are averaged after per-column z-scoring.
MULTIMODAL_INDICES: dict[str, dict[str, Any]] = {
    "eeg_temporoparietal_alpha": {
        "modality": "eeg",
        "columns": [
            "eeg_tp9_alpha_relative", "eeg_tp10_alpha_relative",
        ],
        "sign": 1.0,
        "rationale": "TP9/TP10 alpha relative power (mastoid-referenced): relaxation / low-arousal marker",
    },
    "eeg_beta_gamma_arousal": {
        "modality": "eeg",
        "columns": [
            "eeg_tp9_beta_relative", "eeg_tp10_beta_relative",
            "eeg_tp9_gamma_relative", "eeg_tp10_gamma_relative",
        ],
        "sign": 1.0,
        "rationale": "fast-band relative power: cortical arousal / discomfort proxy",
    },
    "ecg_hr": {
        "modality": "ecg",
        "columns": ["ecg_hr_bpm"],
        "sign": 1.0,
        "rationale": "heart rate: autonomic arousal",
    },
    "ecg_hrv_rmssd": {
        "modality": "ecg",
        "columns": ["ecg_hrv_60s_rmssd_ms", "ecg_hrv_120s_rmssd_ms"],
        "sign": 1.0,
        "rationale": "RMSSD: parasympathetic (rest) tone",
    },
    "eye_gaze_engagement": {
        "modality": "eye",
        "columns": ["eye_fixation_fraction_ivt", "eye_gaze_on_painting_fraction"],
        "sign": 1.0,
        "rationale": "fixation / on-stimulus gaze: visual engagement",
    },
    "eye_scan_exploration": {
        "modality": "eye",
        "columns": ["eye_saccade_fraction_ivt", "eye_scanpath_deg", "eye_spatial_entropy"],
        "sign": 1.0,
        "rationale": "saccade / scanpath / entropy: visual exploration vs settling",
    },
    "head_motion": {
        "modality": "head",
        "columns": ["head_speed_mean", "head_angular_speed_deg_s_mean", "head_jerk_mean"],
        "sign": 1.0,
        "rationale": "head motion: restlessness vs stillness",
    },
    "head_stillness": {
        "modality": "head",
        "columns": ["head_stationary_fraction"],
        "sign": 1.0,
        "rationale": "stationary fraction: postural settling",
    },
}

MODALITY_AVAILABILITY = {
    "eeg": "qc_eeg_usable",
    "ecg": "qc_ecg_usable",
    "eye": "qc_eye_usable",
    "head": "qc_head_usable",
}


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def ensure_dirs() -> dict[str, Path]:
    paths = {
        "root": OUTPUT_ROOT,
        "reports": OUTPUT_ROOT / "reports",
        "figures": OUTPUT_ROOT / "figures",
        "models": OUTPUT_ROOT / "models",
        "sim": OUTPUT_ROOT / "sim",
        "spec": OUTPUT_ROOT / "spec",
        "docs": OUTPUT_ROOT / "docs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Any) -> None:
    write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_labels() -> pd.DataFrame:
    frame = pd.read_csv(CONDITION_LABELS_CSV)
    frame["participant_id"] = frame["participant_id"].astype(str)
    frame["condition"] = frame["condition"].astype(str)
    frame["eeg_status"] = np.where(
        frame["participant_id"].isin(EEG_DISABLED), "eeg_disabled", "eeg_available"
    )
    frame["high_discomfort"] = (
        pd.to_numeric(frame["discomfort"], errors="coerce") >= HIGH_DISCOMFORT_THRESHOLD
    ).astype(int)
    return frame


def load_window_features() -> pd.DataFrame:
    frame = pd.read_csv(WINDOW_FEATURES_CSV)
    frame["participant_id"] = frame["participant_id"].astype(str)
    frame["condition"] = frame["condition"].astype(str)
    return frame


def condition_sort_key(condition: str) -> int:
    text = str(condition).strip().upper()
    if text.startswith("C") and text[1:].isdigit():
        return int(text[1:])
    return 999


def condition_grid() -> pd.DataFrame:
    """Canonical C1..C9 -> (intensity, frequency) lattice positions."""
    rows = []
    for condition in CONDITIONS:
        idx = condition_sort_key(condition) - 1
        intensity_index = idx // 3
        frequency_index = idx % 3
        rows.append(
            {
                "condition": condition,
                "intensity_index": intensity_index,
                "frequency_index": frequency_index,
                "intensity": INTENSITIES[intensity_index],
                "frequency": FREQUENCIES[frequency_index],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
def ci(values: Iterable[float], alpha: float = 0.05) -> tuple[float, float]:
    array = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if array.size == 0:
        return float("nan"), float("nan")
    return (
        float(np.quantile(array, alpha / 2.0)),
        float(np.quantile(array, 1.0 - alpha / 2.0)),
    )


def participant_clusters(frame: pd.DataFrame, column: str = "participant_id") -> np.ndarray:
    return np.asarray(sorted(frame[column].astype(str).unique()))


def cluster_bootstrap(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    *,
    seed: int,
    replicates: int = DEFAULT_BOOTSTRAP,
    participant_column: str = "participant_id",
    alpha: float = 0.05,
) -> dict[str, float]:
    """Participant-cluster bootstrap.  Resampled duplicate participants are
    re-labelled as fresh clusters so within-participant structure is preserved.
    Returns point estimate (on the observed data), CI, and failure count.
    """
    rng = np.random.default_rng(seed)
    participants = participant_clusters(frame, participant_column)
    grouped = {p: frame[frame[participant_column].astype(str) == p] for p in participants}

    try:
        point = float(statistic(frame))
    except Exception:
        point = float("nan")

    estimates: list[float] = []
    failures = 0
    for _ in range(replicates):
        sampled = rng.choice(participants, size=len(participants), replace=True)
        pieces = []
        for draw_index, participant in enumerate(sampled):
            piece = grouped[participant].copy()
            piece[participant_column] = f"{participant}__b{draw_index:02d}"
            pieces.append(piece)
        sample = pd.concat(pieces, ignore_index=True)
        try:
            value = float(statistic(sample))
        except Exception:
            failures += 1
            continue
        if np.isfinite(value):
            estimates.append(value)
        else:
            failures += 1
    low, high = ci(estimates, alpha=alpha)
    return {
        "estimate": point,
        "ci_low": low,
        "ci_high": high,
        "bootstrap_failures": failures,
        "bootstrap_replicates": replicates,
    }


def paired_participant_bootstrap(
    frame: pd.DataFrame,
    per_participant_stat: Callable[[pd.DataFrame], float],
    *,
    seed: int,
    replicates: int = DEFAULT_BOOTSTRAP,
    participant_column: str = "participant_id",
    alpha: float = 0.05,
) -> dict[str, float]:
    """Bootstrap a participant-level paired difference.

    ``per_participant_stat`` maps a single participant's rows to one scalar
    (e.g. model_mae - baseline_mae).  The reported estimate is the mean across
    participants; the CI is from resampling participants with replacement.
    """
    rng = np.random.default_rng(seed)
    participants = participant_clusters(frame, participant_column)
    per_value: dict[str, float] = {}
    for participant in participants:
        piece = frame[frame[participant_column].astype(str) == participant]
        try:
            per_value[participant] = float(per_participant_stat(piece))
        except Exception:
            per_value[participant] = float("nan")
    valid = {k: v for k, v in per_value.items() if np.isfinite(v)}
    if not valid:
        return {
            "estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
            "n_participants": 0, "bootstrap_replicates": replicates,
        }
    keys = np.asarray(list(valid.keys()))
    arr = np.asarray([valid[k] for k in keys], dtype=float)
    point = float(np.mean(arr))
    estimates = []
    for _ in range(replicates):
        sampled = rng.choice(len(keys), size=len(keys), replace=True)
        estimates.append(float(np.mean(arr[sampled])))
    low, high = ci(estimates, alpha=alpha)
    return {
        "estimate": point,
        "ci_low": low,
        "ci_high": high,
        "n_participants": int(len(keys)),
        "bootstrap_replicates": replicates,
        "per_participant": valid,
    }


def benjamini_hochberg(pvalues: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    """Return the BH-adjusted q-values for a list of p-values."""
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    if n == 0:
        return np.asarray([], dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    # enforce monotonicity from the largest rank downwards
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def format_number(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def format_ci(low: Any, high: Any, digits: int = 4) -> str:
    try:
        lo, hi = float(low), float(high)
    except (TypeError, ValueError):
        return "NA"
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "NA"
    return f"[{lo:.{digits}f}, {hi:.{digits}f}]"


def add_contract_columns(frame: pd.DataFrame, inference_unit: str) -> pd.DataFrame:
    out = frame.copy()
    if "ci_low" not in out.columns:
        out["ci_low"] = np.nan
    if "ci_high" not in out.columns:
        out["ci_high"] = np.nan
    if "inference_unit" not in out.columns:
        out["inference_unit"] = inference_unit
    return out

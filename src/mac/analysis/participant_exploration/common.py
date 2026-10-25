from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mac.data.io import parse_float, sniff_csv
from .specification import FEATURES, SPEC, canonical_json, spec_hash


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value), encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path, index=False, encoding="utf-8", na_rep="", lineterminator="\n", float_format="%.15g"
    )


def read_logged(path: Path) -> pd.DataFrame:
    sep, decimal, skip = sniff_csv(path)
    frame = pd.read_csv(
        path, sep=sep, decimal=decimal, skiprows=skip, encoding="utf-8-sig", low_memory=False
    )
    numeric = [
        c
        for c in frame
        if c == "unix_time_ms"
        or c.startswith(
            ("head_rotation_", "head_position_", "gaze_direction_", "gaze_origin_", "gaze_hit_")
        )
    ]
    for column in numeric:
        if frame[column].dtype == object:
            frame[column] = frame[column].map(lambda x: parse_float(x, decimal))
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def truthy(values: pd.Series) -> np.ndarray:
    return values.astype(str).str.lower().isin(["true", "1", "1.0", "yes"]).to_numpy()


def verify_lock(output: Path) -> dict:
    config = json.loads((output / "resolved_config.json").read_text(encoding="utf-8"))
    frozen = json.loads((output / "feature_specification.json").read_text(encoding="utf-8"))
    if (
        canonical_json(frozen) != canonical_json(SPEC)
        or config["specification_sha256"] != spec_hash()
    ):
        raise ValueError("Frozen feature specification differs from code; use a new run directory.")
    return config


def intervals(valid: np.ndarray, adjacent_ok: np.ndarray):
    """Contiguous half-open index spans, never joining across invalid rows or gaps."""
    start = None
    for i, good in enumerate(valid):
        if start is not None and (not good or (i > 0 and not adjacent_ok[i - 1])):
            yield start, i
            start = None
        if good and start is None:
            start = i
    if start is not None:
        yield start, len(valid)


def feature_result(feature: str, raw: float = np.nan, reason: str = "", **quality) -> dict:
    raw = float(raw)
    analysis = np.nan
    raw_available = np.isfinite(raw)
    if raw_available:
        if FEATURES[feature][2] == "ln":
            if raw > 0:
                analysis = float(np.log(raw))
            else:
                reason = "nonpositive_value_for_log"
        else:
            analysis = raw
    return dict(
        feature=feature,
        raw_value=raw,
        analysis_value=analysis,
        availability_status="available"
        if np.isfinite(analysis)
        else "raw_only"
        if raw_available
        else "unavailable",
        raw_available=raw_available,
        analysis_available=bool(np.isfinite(analysis)),
        reason=reason,
        **quality,
    )


def weighted_mean_sd(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    if not len(values) or weights.sum() <= 0:
        return np.nan, np.nan
    mean = float(np.average(values, weights=weights))
    return mean, float(np.sqrt(np.average((values - mean) ** 2, weights=weights)))

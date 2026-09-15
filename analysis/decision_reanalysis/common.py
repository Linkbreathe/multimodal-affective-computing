from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
    sys.path.insert(0, str(SRC_ROOT / "src"))
TARGETS = ("relaxation", "discomfort")
CONDITIONS = tuple(f"C{index}" for index in range(1, 10))
INFERENCE_UNIT = "participant_condition"
WINDOW_FEATURE_UNIT = "window_feature_slice"
NEGLIGIBLE_HETEROGENEITY_SD = 0.05

REQUIRED_TABLES = {
    "condition_labels": {
        "path": REPO_ROOT / "artifacts" / "preprocessed" / "condition_labels.csv",
        "rows": 135,
        "columns": 20,
        "inference_unit": INFERENCE_UNIT,
    },
    "condition_boundaries": {
        "path": REPO_ROOT / "artifacts" / "preprocessed" / "condition_boundaries.csv",
        "rows": 135,
        "columns": 9,
        "inference_unit": INFERENCE_UNIT,
    },
    "windows": {
        "path": REPO_ROOT / "artifacts" / "preprocessed" / "windows.csv",
        "rows": 946,
        "columns": 34,
        "inference_unit": WINDOW_FEATURE_UNIT,
    },
    "window_features": {
        "path": REPO_ROOT / "artifacts" / "features" / "window_features.csv",
        "rows": 946,
        "columns": 162,
        "inference_unit": WINDOW_FEATURE_UNIT,
    },
    "condition_features": {
        "path": REPO_ROOT / "artifacts" / "features" / "condition_features.csv",
        "rows": 135,
        "columns": 1419,
        "inference_unit": INFERENCE_UNIT,
    },
}

LABEL_OR_AUDIT_TOKENS = (
    "relaxation",
    "discomfort",
    "calm",
    "pleasantness",
    "monotony",
    "visual_fit",
    "_raw",
    "label_source_row",
)
AUDIT_QC_COLUMNS = {
    "qc_video_reason",
    "qc_video_index_reason",
    "qc_video_timestamp_source",
}


def ensure_phase_dirs(output: Path) -> dict[str, Path]:
    output = output.resolve()
    paths = {
        "root": output,
        "tables": output / "tables",
        "figures": output / "figures",
        "isolated": output / "isolated_condition_lopo",
        "isolated_features": output / "isolated_condition_lopo" / "features",
        "isolated_models": output / "isolated_condition_lopo" / "models",
        "isolated_reports": output / "isolated_condition_lopo" / "reports",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_required_csv(name: str) -> pd.DataFrame:
    spec = REQUIRED_TABLES[name]
    path = spec["path"]
    if not path.exists():
        raise FileNotFoundError(f"Required CSV is missing: {path}")
    return pd.read_csv(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def write_json(path: Path, payload: Any) -> None:
    write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def ci(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if array.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))


def participant_bootstrap(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    *,
    seed: int,
    replicates: int,
    participant_column: str = "participant_id",
) -> tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    participants = np.asarray(sorted(frame[participant_column].astype(str).unique()))
    estimates: list[float] = []
    failures = 0
    for _ in range(replicates):
        sampled = rng.choice(participants, size=len(participants), replace=True)
        pieces = []
        for draw_index, participant in enumerate(sampled):
            piece = frame[frame[participant_column].astype(str) == participant].copy()
            piece[participant_column] = f"{participant}__boot{draw_index:02d}"
            pieces.append(piece)
        sample = pd.concat(pieces, ignore_index=True)
        try:
            value = float(statistic(sample))
            if np.isfinite(value):
                estimates.append(value)
            else:
                failures += 1
        except Exception:
            failures += 1
    low, high = ci(estimates)
    return low, high, failures


def software_versions() -> pd.DataFrame:
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "joblib", "matplotlib"]
    rows = [
        {
            "component": "python",
            "version": platform.python_version(),
            "detail": sys.version.replace("\n", " "),
        },
        {
            "component": "platform",
            "version": platform.platform(),
            "detail": platform.machine(),
        },
    ]
    for package in packages:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "not_installed"
        rows.append({"component": package, "version": version, "detail": ""})
    return pd.DataFrame(rows)


def forbidden_feature_columns(columns: Iterable[str]) -> list[str]:
    forbidden = []
    for column in columns:
        base = str(column).split("__", 1)[0]
        if base in AUDIT_QC_COLUMNS:
            forbidden.append(str(column))
            continue
        if any(token in str(column) for token in LABEL_OR_AUDIT_TOKENS):
            forbidden.append(str(column))
    return sorted(set(forbidden))


def runtime_qc_feature_columns(columns: Iterable[str]) -> list[str]:
    return sorted({str(column) for column in columns if str(column).startswith("qc_")})


def add_empty_ci(frame: pd.DataFrame, inference_unit: str) -> pd.DataFrame:
    output = frame.copy()
    if "ci_low" not in output.columns:
        output["ci_low"] = np.nan
    if "ci_high" not in output.columns:
        output["ci_high"] = np.nan
    if "inference_unit" not in output.columns:
        output["inference_unit"] = inference_unit
    return output


def assert_no_window_n_as_inference(tables_dir: Path) -> None:
    count_columns = {"n", "n_rows", "n_labels", "sample_size", "row_count"}
    for csv_path in tables_dir.glob("*.csv"):
        frame = pd.read_csv(csv_path)
        if "inference_unit" not in frame.columns:
            continue
        non_window = frame[frame["inference_unit"].astype(str) != WINDOW_FEATURE_UNIT]
        for column in count_columns & set(non_window.columns):
            values = pd.to_numeric(non_window[column], errors="coerce")
            if (values == 946).any():
                raise AssertionError(f"{csv_path.name} reports n=946 as inference evidence")


def assert_estimate_tables_have_ci(tables_dir: Path, names: Iterable[str]) -> None:
    required = {"ci_low", "ci_high", "inference_unit"}
    for name in names:
        path = tables_dir / f"{name}.csv"
        frame = pd.read_csv(path)
        missing = required - set(frame.columns)
        if missing:
            raise AssertionError(f"{path.name} lacks estimate-contract columns: {sorted(missing)}")


def condition_sort_key(condition: str) -> int:
    text = str(condition).strip().upper()
    if text.startswith("C"):
        return int(text[1:])
    return 999


def format_number(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def format_ci(low: Any, high: Any, digits: int = 4) -> str:
    if not np.isfinite(float(low)) or not np.isfinite(float(high)):
        return "NA"
    return f"[{float(low):.{digits}f}, {float(high):.{digits}f}]"

"""Strict Relax P002-P016 audit and cohort utilities.

These helpers intentionally fail loudly. They are used before any model
training so cohort membership and data availability cannot be changed after
seeing probe performance.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.encoders.reve_pos_bank import electrode_list


PARTICIPANTS: tuple[str, ...] = tuple(f"P{i:03d}" for i in range(2, 17))
CONDITIONS: tuple[str, ...] = tuple(f"C{i}" for i in range(1, 10))
MODALITIES: tuple[str, ...] = ("ecg", "eeg", "eye", "head", "video")
DERIVED_MODALITIES: tuple[str, ...] = ("attention_video",)
ALLOWED_MODALITIES: tuple[str, ...] = (*MODALITIES, *DERIVED_MODALITIES)
FORBIDDEN_RELAX_MODALITY_TOKENS: tuple[str, ...] = ("ppg", "papagei", "pulseppg")
EEG_DISABLED_PARTICIPANTS: frozenset[str] = frozenset(
    {"P002", "P005", "P006", "P010", "P014", "P016"}
)
RELAX_EEG_RUN_TAG = "eegmap_m2_tp9_tp10_m1_20260714"
RELAX_REVE_MODEL = "brain-bzh/reve-large"
RELAX_REVE_POSITIONS = "brain-bzh/reve-positions"
RELAX_EEG_EMBED_DIM = 1024


@dataclass(frozen=True)
class RelaxEEGChannel:
    """One immutable XDF-column to electrode mapping entry."""

    xdf_column: int
    eeg_index: int
    electrode: str
    hemisphere: str


RELAX_EEG_CHANNELS: tuple[RelaxEEGChannel, ...] = (
    RelaxEEGChannel(xdf_column=1, eeg_index=0, electrode="M2", hemisphere="right"),
    RelaxEEGChannel(xdf_column=2, eeg_index=1, electrode="TP9", hemisphere="left"),
    RelaxEEGChannel(xdf_column=3, eeg_index=2, electrode="TP10", hemisphere="right"),
    RelaxEEGChannel(xdf_column=4, eeg_index=3, electrode="M1", hemisphere="left"),
)
RELAX_EEG_XDF_COLUMNS: tuple[int, ...] = tuple(channel.xdf_column for channel in RELAX_EEG_CHANNELS)
RELAX_EEG_MONTAGE: tuple[str, ...] = tuple(channel.electrode for channel in RELAX_EEG_CHANNELS)
RELAX_EEG_LEFT_INDICES: tuple[int, ...] = tuple(
    channel.eeg_index for channel in RELAX_EEG_CHANNELS if channel.hemisphere == "left"
)
RELAX_EEG_RIGHT_INDICES: tuple[int, ...] = tuple(
    channel.eeg_index for channel in RELAX_EEG_CHANNELS if channel.hemisphere == "right"
)


def relax_eeg_contract_payload() -> list[dict[str, int | str]]:
    return [asdict(channel) for channel in RELAX_EEG_CHANNELS]

DEFAULT_PILOT_THRESHOLDS: dict[str, float] = {
    "label_count": 9.0,
    "valid_window_ratio": 0.95,
    "ecg_usable_ratio": 0.80,
    "eeg_usable_ratio": 0.70,
    "gaze_usable_ratio": 0.80,
    "head_usable_ratio": 0.90,
    "video_usable_ratio": 0.90,
    "marker_median_abs_residual_ms": 10.0,
}

HEAD_MOTION_FIELDS: tuple[str, ...] = (
    "unix_time_ms",
    "head_pose_available",
    "head_position_x",
    "head_position_y",
    "head_position_z",
    "head_rotation_x",
    "head_rotation_y",
    "head_rotation_z",
    "head_rotation_w",
    "head_velocity_x",
    "head_velocity_y",
    "head_velocity_z",
    "head_angular_velocity_deg_s",
)


class RelaxHardFailure(RuntimeError):
    """Raised when a Relax audit or training precondition is violated."""


def assert_relax_modalities(modalities: Iterable[str]) -> list[str]:
    """Validate the Relax modality list and reject all PPG/Papagei aliases."""
    normalized = [str(modality).strip().lower() for modality in modalities]
    forbidden = [
        modality for modality in normalized
        if any(token in modality for token in FORBIDDEN_RELAX_MODALITY_TOKENS)
    ]
    if forbidden:
        raise RelaxHardFailure(
            "PPG/Papagei modalities are excluded from Relax evaluation: "
            f"{forbidden}"
        )
    unknown = [modality for modality in normalized if modality not in ALLOWED_MODALITIES]
    if unknown:
        raise RelaxHardFailure(
            f"Unknown Relax modalities {unknown}; allowed modalities are {list(ALLOWED_MODALITIES)}"
        )
    return normalized


@dataclass(frozen=True)
class RelaxRunPaths:
    root: Path
    preprocessed: Path
    features: Path
    metrics: Path
    reports: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "RelaxRunPaths":
        root = Path(root)
        return cls(
            root=root,
            preprocessed=root / "preprocessed",
            features=root / "features",
            metrics=root / "metrics",
            reports=root / "reports",
        )


def read_relax_csv(path: str | Path) -> pd.DataFrame:
    """Read Relax CSVs, including files with an Excel-style ``sep=,`` line."""
    path = Path(path)
    if not path.exists():
        raise RelaxHardFailure(f"Required Relax table is missing: {path}")
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        first = handle.readline()
    skiprows = 1 if first.lower().startswith("sep=") else 0
    return pd.read_csv(path, skiprows=skiprows)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _required_columns(frame: pd.DataFrame, columns: Iterable[str], table_name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RelaxHardFailure(f"{table_name} missing required columns: {missing}")


def _condition_order(condition: Any) -> int:
    text = str(condition)
    if text not in CONDITIONS:
        raise RelaxHardFailure(f"Unexpected Relax Condition label: {text!r}")
    return CONDITIONS.index(text)


def _as_bool_series(frame: pd.DataFrame, column_names: tuple[str, ...]) -> pd.Series | None:
    for column in column_names:
        if column in frame.columns:
            values = frame[column]
            if values.dtype == bool:
                return values.astype(float)
            return values.replace(
                {
                    True: 1.0,
                    False: 0.0,
                    "True": 1.0,
                    "False": 0.0,
                    "true": 1.0,
                    "false": 0.0,
                    "1": 1.0,
                    "0": 0.0,
                }
            ).astype(float)
    return None


def _numeric_mean(frame: pd.DataFrame, column_names: tuple[str, ...]) -> float:
    series = _as_bool_series(frame, column_names)
    if series is None:
        return float("nan")
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    valid = values[np.isfinite(values)]
    return float(np.mean(valid)) if valid.size else float("nan")


def _load_data_qc(reports_dir: Path) -> dict[str, dict[str, Any]]:
    path = reports_dir / "data_qc.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(row.get("participant_id")): row for row in data.get("participants", [])}


def _find_label_workbook(labels_root: str | Path | None) -> Path | None:
    if labels_root is None:
        return None
    root = Path(labels_root)
    workbook = root / "4. Painting Reflection (Responses).xlsx"
    if workbook.exists():
        return workbook
    candidates = sorted(root.rglob("*Painting Reflection*Responses*.xlsx")) if root.exists() else []
    return candidates[0] if candidates else None


def _looks_like_hf_repo_id(value: str | Path) -> bool:
    text = str(value)
    return "/" in text and not text.startswith("/") and not text.startswith(".") and "\\" not in text


def _validate_model_source(value: str | Path | None, label: str, errors: list[str]) -> str | None:
    if value is None:
        errors.append(f"{label} is required for mandatory EEG")
        return None
    path = Path(value)
    if path.exists():
        return str(path)
    if _looks_like_hf_repo_id(value):
        try:
            from huggingface_hub import HfApi

            HfApi().model_info(str(value))
            return str(value)
        except Exception as error:
            errors.append(f"{label} Hugging Face repo unavailable: {value} ({error})")
            return None
    errors.append(f"{label} not found: {value}")
    return None


def _audit_raw_sources(raw_root: str | Path, participants: Iterable[str]) -> list[str]:
    root = Path(raw_root)
    errors: list[str] = []
    if not root.exists():
        return [f"raw_root does not exist: {root}"]

    for participant in participants:
        participant_dirs = [p for p in root.rglob(participant) if p.is_dir()]
        if not participant_dirs:
            errors.append(f"{participant}: missing participant directory under {root}")
            continue
        pdir = participant_dirs[0]
        if not list(pdir.rglob("*.xdf")):
            errors.append(f"{participant}: missing XDF source")

        samples = sorted(pdir.rglob("samples.csv"))
        if not samples:
            errors.append(f"{participant}: missing samples.csv for head-motion")
        else:
            frame = read_relax_csv(samples[0])
            missing = sorted(set(HEAD_MOTION_FIELDS) - set(frame.columns))
            if missing:
                errors.append(f"{participant}: samples.csv missing head-motion fields {missing}")

        if not list(pdir.rglob("eye_tracking.csv")):
            errors.append(f"{participant}: missing eye_tracking.csv")
        if not (list(pdir.rglob("*video*frames*.csv")) or list(pdir.rglob("video_frames"))):
            errors.append(f"{participant}: missing auditable video frame source")
    return errors


def _validate_labels(labels: pd.DataFrame, participants: tuple[str, ...]) -> None:
    _required_columns(labels, ("participant_id", "condition", "relaxation", "discomfort"), "condition_labels.csv")
    labels = labels.copy()
    labels["participant_id"] = labels["participant_id"].astype(str)
    labels["condition"] = labels["condition"].astype(str)
    unexpected_participants = sorted(set(labels["participant_id"]) - set(participants))
    if unexpected_participants:
        raise RelaxHardFailure(f"condition_labels.csv has unexpected participants: {unexpected_participants}")
    unexpected_conditions = sorted(set(labels["condition"]) - set(CONDITIONS))
    if unexpected_conditions:
        raise RelaxHardFailure(f"condition_labels.csv has unexpected Conditions: {unexpected_conditions}")
    duplicates = labels.duplicated(["participant_id", "condition"], keep=False)
    if duplicates.any():
        rows = labels.loc[duplicates, ["participant_id", "condition"]].to_dict("records")
        raise RelaxHardFailure(f"Duplicate participant-Condition labels found: {rows[:5]}")
    expected = len(participants) * len(CONDITIONS)
    if len(labels) != expected:
        raise RelaxHardFailure(
            f"Expected {expected} participant-Condition labels, found {len(labels)}"
        )
    counts = labels.groupby("participant_id")["condition"].nunique()
    bad = counts[counts != len(CONDITIONS)]
    if not bad.empty:
        raise RelaxHardFailure(f"Participants without 9 Condition labels: {bad.to_dict()}")


def validate_phase0_inputs(
    relax_run_dir: str | Path,
    *,
    raw_root: str | Path | None = None,
    labels_root: str | Path | None = None,
    eeg_weights: str | Path | None = None,
    eeg_pos_bank: str | Path | None = None,
    participants: Iterable[str] = PARTICIPANTS,
    require_reve: bool = True,
) -> dict[str, Any]:
    """Run Phase 0 checks and raise ``RelaxHardFailure`` on any violation."""
    participants_tuple = tuple(participants)
    paths = RelaxRunPaths.from_root(relax_run_dir)
    errors: list[str] = []

    labels_path = paths.preprocessed / "condition_labels.csv"
    windows_path = paths.preprocessed / "windows.csv"
    window_features_path = paths.features / "window_features.csv"
    for path in (labels_path, windows_path, window_features_path):
        if not path.exists():
            errors.append(f"missing required Relax output: {path}")

    workbook = _find_label_workbook(labels_root)
    if labels_root is not None and workbook is None:
        errors.append(
            "missing Painting Reflection workbook: "
            f"{Path(labels_root) / '4. Painting Reflection (Responses).xlsx'}"
        )

    if labels_path.exists():
        try:
            _validate_labels(read_relax_csv(labels_path), participants_tuple)
        except RelaxHardFailure as error:
            errors.append(str(error))

    if windows_path.exists():
        try:
            windows = read_relax_csv(windows_path)
            _required_columns(windows, ("participant_id", "condition"), "windows.csv")
            bad_conditions = sorted(set(windows["condition"].astype(str)) - set(CONDITIONS))
            if bad_conditions:
                errors.append(f"windows.csv has unexpected Conditions: {bad_conditions}")
        except RelaxHardFailure as error:
            errors.append(str(error))

    if require_reve:
        resolved_eeg_weights = _validate_model_source(eeg_weights, "REVE weights/model", errors)
        resolved_eeg_pos_bank = _validate_model_source(eeg_pos_bank, "REVE position bank/model", errors)
        position_names = {name.upper() for name in electrode_list}
        missing_positions = [name for name in RELAX_EEG_MONTAGE if name.upper() not in position_names]
        if missing_positions:
            errors.append(f"REVE position bank lacks Relax EEG montage channels: {missing_positions}")
    else:
        resolved_eeg_weights = None
        resolved_eeg_pos_bank = None

    try:
        from src.fusion.multimodal_lego import MultimodalLegoFusion  # noqa: F401
    except Exception as error:  # pragma: no cover - import error is the audit payload
        errors.append(f"MM-Lego implementation unavailable: {error}")

    if raw_root is not None:
        errors.extend(_audit_raw_sources(raw_root, participants_tuple))

    audit = {
        "status": "failed" if errors else "ok",
        "errors": errors,
        "participants": list(participants_tuple),
        "conditions": list(CONDITIONS),
        "eeg_montage": list(RELAX_EEG_MONTAGE),
        "eeg_channel_contract": relax_eeg_contract_payload(),
        "painting_reflection_workbook": str(workbook) if workbook else None,
        "relax_run_dir": str(paths.root),
        "reve_model_source": resolved_eeg_weights,
        "reve_position_source": resolved_eeg_pos_bank,
    }
    if errors:
        raise RelaxHardFailure("; ".join(errors))
    return audit


def build_qc_manifest(
    relax_run_dir: str | Path,
    *,
    participants: Iterable[str] = PARTICIPANTS,
) -> dict[str, Any]:
    """Build auditable participant QC metrics from Relax outputs."""
    participants_tuple = tuple(participants)
    paths = RelaxRunPaths.from_root(relax_run_dir)
    labels = read_relax_csv(paths.preprocessed / "condition_labels.csv")
    windows = read_relax_csv(paths.preprocessed / "windows.csv")
    features = read_relax_csv(paths.features / "window_features.csv")
    _required_columns(labels, ("participant_id", "condition", "relaxation", "discomfort"), "condition_labels.csv")
    _required_columns(windows, ("participant_id", "condition"), "windows.csv")
    _required_columns(features, ("participant_id", "condition"), "window_features.csv")
    data_qc = _load_data_qc(paths.metrics) or _load_data_qc(paths.reports)

    labels["participant_id"] = labels["participant_id"].astype(str)
    windows["participant_id"] = windows["participant_id"].astype(str)
    features["participant_id"] = features["participant_id"].astype(str)

    records: list[dict[str, Any]] = []
    for participant in participants_tuple:
        label_rows = labels[labels["participant_id"] == participant]
        window_rows = windows[windows["participant_id"] == participant]
        feature_rows = features[features["participant_id"] == participant]
        qc_row = data_qc.get(participant, {})

        if "condition_window_count" in window_rows.columns:
            expected_windows = (
                pd.to_numeric(window_rows["condition_window_count"], errors="coerce")
                .groupby(window_rows["condition"])
                .max()
                .sum()
            )
        else:
            expected_windows = len(window_rows)
        expected_windows = float(expected_windows) if np.isfinite(expected_windows) else float(len(window_rows))

        label_count = int(label_rows["condition"].nunique())
        row = {
            "participant_id": participant,
            "label_count": label_count,
            "condition_labels_complete": bool(label_count == len(CONDITIONS)),
            "window_count": int(len(window_rows)),
            "expected_window_count": expected_windows,
            "valid_window_ratio": float(len(feature_rows) / expected_windows) if expected_windows else 0.0,
            "ecg_usable_ratio": _numeric_mean(feature_rows, ("qc_ecg_usable", "ecg_usable")),
            "eeg_usable_ratio": _numeric_mean(feature_rows, ("qc_eeg_usable", "eeg_usable")),
            "eeg_strict_coverage_mean": _numeric_mean(feature_rows, ("qc_eeg_strict_coverage", "eeg_strict_coverage")),
            "gaze_usable_ratio": _numeric_mean(feature_rows, ("qc_eye_usable", "eye_usable", "qc_gaze_usable", "gaze_usable")),
            "gaze_coverage_mean": _numeric_mean(feature_rows, ("qc_eye_coverage", "eye_coverage", "gaze_coverage")),
            "head_usable_ratio": _numeric_mean(feature_rows, ("qc_head_usable", "head_usable")),
            "head_coverage_mean": _numeric_mean(feature_rows, ("qc_head_coverage", "head_coverage")),
            "video_usable_ratio": _numeric_mean(feature_rows, ("qc_video_usable", "video_usable")),
            "video_coverage_mean": _numeric_mean(feature_rows, ("qc_video_coverage", "video_coverage")),
            "eeg_disabled_by_relax": participant in EEG_DISABLED_PARTICIPANTS,
            "marker_median_abs_residual_ms": float(qc_row.get("median_abs_residual_ms", np.nan)),
            "marker_max_abs_residual_ms": float(qc_row.get("max_abs_residual_ms", np.nan)),
            "marker_outlier_count": len(qc_row.get("outliers", []) or []),
            "preprocess_status": qc_row.get("status", "missing"),
            "preprocess_reason": qc_row.get("reason"),
        }
        row["high_quality_score"] = _quality_score(row)
        records.append(row)

    return {
        "schema_version": "1.0",
        "participants": records,
        "conditions": list(CONDITIONS),
        "modalities": list(MODALITIES),
        "eeg_disabled_participants": sorted(EEG_DISABLED_PARTICIPANTS),
        "pilot_thresholds": dict(DEFAULT_PILOT_THRESHOLDS),
    }


def _quality_score(row: dict[str, Any]) -> float:
    higher = [
        "valid_window_ratio",
        "ecg_usable_ratio",
        "eeg_usable_ratio",
        "gaze_usable_ratio",
        "head_usable_ratio",
        "video_usable_ratio",
    ]
    score = 0.0
    for key in higher:
        value = float(row.get(key, float("nan")))
        if np.isfinite(value):
            score += value
    residual = float(row.get("marker_median_abs_residual_ms", float("nan")))
    if np.isfinite(residual):
        score -= residual / 100.0
    return float(score)


def _passes_pilot_thresholds(row: dict[str, Any], thresholds: dict[str, float]) -> bool:
    if row.get("eeg_disabled_by_relax"):
        return False
    if int(row.get("label_count", 0)) < int(thresholds["label_count"]):
        return False
    residual = float(row.get("marker_median_abs_residual_ms", float("nan")))
    if not np.isfinite(residual) or residual > thresholds["marker_median_abs_residual_ms"]:
        return False
    for key in (
        "valid_window_ratio",
        "ecg_usable_ratio",
        "eeg_usable_ratio",
        "gaze_usable_ratio",
        "head_usable_ratio",
        "video_usable_ratio",
    ):
        value = float(row.get(key, float("nan")))
        if not np.isfinite(value) or value < thresholds[key]:
            return False
    return True


def select_high_quality_pilot(
    qc_manifest: dict[str, Any],
    *,
    min_count: int = 4,
    max_count: int = 6,
    thresholds: dict[str, float] | None = None,
) -> list[str]:
    thresholds = thresholds or DEFAULT_PILOT_THRESHOLDS
    eligible = [
        row for row in qc_manifest.get("participants", [])
        if _passes_pilot_thresholds(row, thresholds)
    ]
    eligible = sorted(
        eligible,
        key=lambda row: (-float(row.get("high_quality_score", 0.0)), str(row["participant_id"])),
    )
    if len(eligible) < min_count:
        raise RelaxHardFailure(
            "high_quality_pilot has fewer participants than required "
            f"({len(eligible)} < {min_count}) under pre-registered QC thresholds"
        )
    return [str(row["participant_id"]) for row in eligible[:max_count]]


def build_cohorts(
    qc_manifest: dict[str, Any],
    *,
    high_quality_min_count: int = 4,
    high_quality_max_count: int = 6,
) -> dict[str, Any]:
    records = list(qc_manifest.get("participants", []))

    def complete_labels(row: dict[str, Any]) -> bool:
        return bool(row.get("condition_labels_complete"))

    def usable(row: dict[str, Any], key: str, threshold: float) -> bool:
        value = float(row.get(key, float("nan")))
        return bool(np.isfinite(value) and value >= threshold)

    all_135 = [row["participant_id"] for row in records if complete_labels(row)]
    eeg_eligible = [
        row["participant_id"] for row in records
        if complete_labels(row)
        and not row.get("eeg_disabled_by_relax")
        and usable(row, "eeg_usable_ratio", DEFAULT_PILOT_THRESHOLDS["eeg_usable_ratio"])
    ]
    foundation_complete = [
        row["participant_id"] for row in records
        if complete_labels(row)
        and usable(row, "ecg_usable_ratio", DEFAULT_PILOT_THRESHOLDS["ecg_usable_ratio"])
        and usable(row, "eeg_usable_ratio", DEFAULT_PILOT_THRESHOLDS["eeg_usable_ratio"])
        and usable(row, "gaze_usable_ratio", DEFAULT_PILOT_THRESHOLDS["gaze_usable_ratio"])
        and usable(row, "video_usable_ratio", DEFAULT_PILOT_THRESHOLDS["video_usable_ratio"])
    ]
    video_complete = [
        row["participant_id"] for row in records
        if complete_labels(row) and usable(row, "video_usable_ratio", DEFAULT_PILOT_THRESHOLDS["video_usable_ratio"])
    ]
    high_quality = select_high_quality_pilot(
        qc_manifest,
        min_count=high_quality_min_count,
        max_count=high_quality_max_count,
    )

    return {
        "schema_version": "1.0",
        "high_quality_pilot": {
            "participants": high_quality,
            "purpose": "engineering pilot only; not a final scientific cohort",
            "selection_rule": "pre-training QC thresholds and deterministic QC score",
        },
        "all_135": {"participants": list(all_135)},
        "eeg_eligible": {"participants": list(eeg_eligible)},
        "foundation_complete": {"participants": list(foundation_complete)},
        "video_complete_sensitivity": {"participants": list(video_complete)},
    }

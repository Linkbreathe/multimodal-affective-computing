from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.supplementary.run_attended_video_preview import (  # noqa: E402
    _current_gaze,
    _estimate_vmax,
    _gaze_pixel,
    _heatmap_mask,
    _nearest_video_frame,
    _read_video_frames,
    _rolling_gaze,
)
from analysis.supplementary.run_four_channel_attended_video_preview import (  # noqa: E402
    _attention_center,
    _expand_attention_mask,
    _make_four_channel_crop,
    run as run_four_channel_preview,
)
from analysis.supplementary.run_gaze_heatmap_pilot import (  # noqa: E402
    DEFAULT_MANIFEST,
    _participant_source,
    _read_manifest,
    add_normalized_coordinates,
    add_time_weights,
    coordinate_bounds,
    load_viewing_gaze,
)
from analysis.supplementary.run_gaze_heatmap_video import _build_formal_timeline  # noqa: E402
from real_time_ml.features.common import robust_stats, safe_divide  # noqa: E402
from real_time_ml.modeling import minimal_fusion as old_fusion  # noqa: E402
from real_time_ml.data.io import normalize_participant_id  # noqa: E402


DEFAULT_BASE_FEATURES = ROOT / "artifacts" / "features" / "ecg_neurokit" / "condition_features.csv"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "fusion_four_channel_ranking"
DEFAULT_VIDEO_OUTPUT_DIR = ROOT / "artifacts" / "four_channel_attended_video"
DEFAULT_REPORT = ROOT / "artifacts" / "reports" / "four_channel_attended_video_ranking_benchmark_zh.md"

A4_PREFIX = "a4_"
CONTINUOUS_TARGETS = ("relaxation", "pleasantness", "calm")
BINARY_TARGET = "high_discomfort"
DISCOMFORT_THRESHOLD = 0.50

MODALITY_PREFIXES = {
    "P": ("eeg_", "ecg_"),
    "H": ("head_",),
    "E": ("eye_",),
    "V": ("video_",),
    "A": (A4_PREFIX,),
}
MODALITY_ORDER = tuple(MODALITY_PREFIXES)
COMBINATIONS = tuple(
    "".join(parts)
    for size in range(1, len(MODALITY_ORDER) + 1)
    for parts in combinations(MODALITY_ORDER, size)
)

IDENTIFIER_AND_LABEL_COLUMNS = set(old_fusion.IDENTIFIER_AND_LABEL_COLUMNS) | {BINARY_TARGET}


def _dependencies():
    try:
        from scipy.stats import spearmanr
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.metrics import (
            average_precision_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Benchmark dependencies are missing; use the rtml-p002-p016 env") from error
    return (
        spearmanr,
        SimpleImputer,
        LogisticRegression,
        Ridge,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        Pipeline,
        StandardScaler,
    )


def _fmt(value: Any, digits: int = 4, *, percent: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(number):
        return "NA"
    return f"{number:.1%}" if percent else f"{number:.{digits}f}"


def _condition_at_time(segments: list[dict[str, Any]], time_s: float) -> str:
    for segment in segments:
        start = float(segment["formal_start_s"])
        end = float(segment["formal_end_s"])
        if start <= time_s <= end:
            return str(segment["condition"])
    return str(segments[-1]["condition"])


def _safe_spearman(truth: np.ndarray, prediction: np.ndarray) -> float:
    spearmanr, *_ = _dependencies()
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    valid = np.isfinite(truth) & np.isfinite(prediction)
    if valid.sum() < 3 or np.nanstd(truth[valid]) <= 0 or np.nanstd(prediction[valid]) <= 0:
        return 0.0
    value = float(spearmanr(truth[valid], prediction[valid]).statistic)
    return value if np.isfinite(value) else 0.0


def _colorfulness(frame: np.ndarray) -> float:
    values = frame.astype(float)
    rg = values[:, :, 2] - values[:, :, 1]
    yb = 0.5 * (values[:, :, 2] + values[:, :, 1]) - values[:, :, 0]
    return float(
        np.sqrt(np.var(rg) + np.var(yb))
        + 0.3 * np.sqrt(float(np.mean(rg)) ** 2 + float(np.mean(yb)) ** 2)
    )


def _visual_snapshot(frame_bgr: np.ndarray, prefix: str = "") -> dict[str, float]:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    metrics = {
        "brightness": float(np.mean(hsv[:, :, 2])),
        "saturation": float(np.mean(hsv[:, :, 1])),
        "colorfulness": _colorfulness(frame_bgr),
        "texture": float(np.std(gray)),
        "edge": float(np.mean(cv2.Canny(gray, 80, 160) > 0)),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }
    return {f"{prefix}{name}": value for name, value in metrics.items()}


def _weighted_visual_snapshot(frame_bgr: np.ndarray, mask_u8: np.ndarray, prefix: str) -> dict[str, float]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(float)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(float)
    mask = mask_u8.astype(float) / 255.0
    total = float(mask.sum())
    if total <= 1e-9:
        return {
            f"{prefix}brightness": float("nan"),
            f"{prefix}saturation": float("nan"),
            f"{prefix}texture": float("nan"),
        }
    brightness = float(np.sum(hsv[:, :, 2] * mask) / total)
    saturation = float(np.sum(hsv[:, :, 1] * mask) / total)
    gray_mean = float(np.sum(gray * mask) / total)
    texture = float(np.sqrt(np.sum(mask * (gray - gray_mean) ** 2) / total))
    return {
        f"{prefix}brightness": brightness,
        f"{prefix}saturation": saturation,
        f"{prefix}texture": texture,
    }


def _small_gray(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (112, 112), interpolation=cv2.INTER_AREA)


def _motion_metrics(previous_gray: np.ndarray | None, gray: np.ndarray, prefix: str) -> dict[str, float]:
    if previous_gray is None:
        return {f"{prefix}optical_flow": float("nan"), f"{prefix}scene_change": float("nan")}
    flow = cv2.calcOpticalFlowFarneback(previous_gray, gray, None, 0.5, 2, 15, 2, 5, 1.1, 0)
    return {
        f"{prefix}optical_flow": float(np.mean(np.linalg.norm(flow, axis=2))),
        f"{prefix}scene_change": float(np.mean(cv2.absdiff(previous_gray, gray))),
    }


def _mask_entropy(mask: np.ndarray) -> float:
    values = np.asarray(mask, dtype=float).ravel()
    total = float(values.sum())
    if total <= 0:
        return 0.0
    probabilities = values[values > 0] / total
    return float(-(probabilities * np.log2(probabilities)).sum() / np.log2(values.size))


def _mask_center(mask: np.ndarray) -> tuple[float, float]:
    values = np.asarray(mask, dtype=float)
    total = float(values.sum())
    if total <= 0:
        return float("nan"), float("nan")
    height, width = values.shape
    ys, xs = np.indices(values.shape, dtype=float)
    return (
        float(np.sum(xs * values) / total / max(1, width - 1)),
        float(np.sum(ys * values) / total / max(1, height - 1)),
    )


def _mask_spread(mask: np.ndarray) -> tuple[float, float]:
    values = np.asarray(mask, dtype=float)
    total = float(values.sum())
    if total <= 0:
        return float("nan"), float("nan")
    height, width = values.shape
    ys, xs = np.indices(values.shape, dtype=float)
    cx = float(np.sum(xs * values) / total)
    cy = float(np.sum(ys * values) / total)
    sx = float(np.sqrt(np.sum(values * (xs - cx) ** 2) / total) / max(1, width - 1))
    sy = float(np.sqrt(np.sum(values * (ys - cy) ** 2) / total) / max(1, height - 1))
    return sx, sy


def _mask_snapshot(mask_u8: np.ndarray, prefix: str = "mask_") -> dict[str, float]:
    mask = mask_u8.astype(float) / 255.0
    center_x, center_y = _mask_center(mask)
    spread_x, spread_y = _mask_spread(mask)
    return {
        f"{prefix}mean": float(np.mean(mask)),
        f"{prefix}std": float(np.std(mask)),
        f"{prefix}max": float(np.max(mask)),
        f"{prefix}coverage_025": float(np.mean(mask >= 0.25)),
        f"{prefix}coverage_050": float(np.mean(mask >= 0.50)),
        f"{prefix}entropy": _mask_entropy(mask),
        f"{prefix}center_x": center_x,
        f"{prefix}center_y": center_y,
        f"{prefix}spread_x": spread_x,
        f"{prefix}spread_y": spread_y,
    }


def _append_metrics(
    store: dict[tuple[str, str], dict[str, list[float]]],
    key: tuple[str, str],
    metrics: dict[str, float],
) -> None:
    for name, value in metrics.items():
        store[key][name].append(float(value))


def _summarize_condition_values(
    values: dict[tuple[str, str], dict[str, list[float]]],
    sample_counts: dict[tuple[str, str], int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (participant_id, condition), metrics in sorted(values.items()):
        row: dict[str, Any] = {
            "participant_id": participant_id,
            "condition": condition,
            "a4_sample_count": int(sample_counts[(participant_id, condition)]),
        }
        for name, series in sorted(metrics.items()):
            row.update(robust_stats(np.asarray(series, dtype=float), f"{A4_PREFIX}{name}"))
        rows.append(row)
    return pd.DataFrame(rows)


def extract_a4_features_for_participant(
    participant_id: str,
    source: pd.Series,
    *,
    bins: int,
    window_s: float,
    step_s: float,
    crop_size: int,
    model_size: int,
    color_quantile: float,
    expand_sigma: float,
    mask_gamma: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    participant_id = normalize_participant_id(participant_id)
    session_dir = Path(str(source["session_dir"]))
    eye_tracking_csv = Path(str(source["eye_tracking_csv"]))
    video_frames_csv = Path(str(source["video_frames_csv"]))

    gaze = load_viewing_gaze(eye_tracking_csv, participant_id)
    gaze = add_time_weights(gaze)
    bounds = coordinate_bounds(gaze)
    gaze = add_normalized_coordinates(gaze, bounds)
    gaze, segments = _build_formal_timeline(gaze)
    video = _read_video_frames(video_frames_csv, session_dir)
    vmax = _estimate_vmax(
        gaze,
        bins=bins,
        window_s=window_s,
        step_s=step_s,
        quantile=color_quantile,
    )

    values: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    sample_counts: dict[tuple[str, str], int] = defaultdict(int)
    previous_rgb_gray: dict[str, np.ndarray] = {}
    previous_mask_gray: dict[str, np.ndarray] = {}
    total_s = float(gaze["formal_elapsed_s"].max())
    readable = 0

    for time_s in np.arange(0.0, total_s + 1e-9, step_s):
        current = _current_gaze(gaze, float(time_s))
        video_row = _nearest_video_frame(video, float(current["session_elapsed_seconds"]))
        frame = cv2.imread(str(video_row["path"]))
        if frame is None:
            continue
        readable += 1
        rolling = _rolling_gaze(gaze, float(time_s), window_s)
        base_mask = _heatmap_mask(rolling, frame.shape[1], frame.shape[0], bins=bins, vmax=vmax)
        mask = _expand_attention_mask(base_mask, expand_sigma=expand_sigma, gamma=mask_gamma)
        current_gaze = _gaze_pixel(current, frame.shape[1], frame.shape[0])
        center = _attention_center(mask, fallback=current_gaze)
        _, rgb_model_bgr, mask_u8 = _make_four_channel_crop(
            frame,
            mask,
            center=center,
            crop_size=crop_size,
            model_size=model_size,
        )
        condition = _condition_at_time(segments, float(time_s))
        key = (participant_id, condition)
        sample_counts[key] += 1

        rgb_metrics = _visual_snapshot(rgb_model_bgr, prefix="rgb_")
        mask_metrics = _mask_snapshot(mask_u8)
        weighted_metrics = _weighted_visual_snapshot(rgb_model_bgr, mask_u8, prefix="mask_weighted_")
        contrasts = {
            "mask_weighted_minus_rgb_brightness": weighted_metrics["mask_weighted_brightness"] - rgb_metrics["rgb_brightness"],
            "mask_weighted_minus_rgb_saturation": weighted_metrics["mask_weighted_saturation"] - rgb_metrics["rgb_saturation"],
            "mask_weighted_minus_rgb_texture": weighted_metrics["mask_weighted_texture"] - rgb_metrics["rgb_texture"],
            "mask_weighted_to_rgb_brightness": safe_divide(weighted_metrics["mask_weighted_brightness"], rgb_metrics["rgb_brightness"]),
            "mask_weighted_to_rgb_saturation": safe_divide(weighted_metrics["mask_weighted_saturation"], rgb_metrics["rgb_saturation"]),
            "attention_center_x_px": float(center[0]),
            "attention_center_y_px": float(center[1]),
            "current_gaze_x_px": float(current_gaze[0]),
            "current_gaze_y_px": float(current_gaze[1]),
            "attention_to_gaze_distance_px": float(np.hypot(center[0] - current_gaze[0], center[1] - current_gaze[1])),
        }
        rgb_gray = _small_gray(rgb_model_bgr)
        mask_gray = cv2.resize(mask_u8, (112, 112), interpolation=cv2.INTER_AREA)
        rgb_motion = _motion_metrics(previous_rgb_gray.get(condition), rgb_gray, prefix="rgb_")
        mask_motion = {
            "mask_scene_change": float("nan")
            if condition not in previous_mask_gray
            else float(np.mean(cv2.absdiff(previous_mask_gray[condition], mask_gray))),
        }
        previous_rgb_gray[condition] = rgb_gray
        previous_mask_gray[condition] = mask_gray
        _append_metrics(values, key, rgb_metrics)
        _append_metrics(values, key, mask_metrics)
        _append_metrics(values, key, weighted_metrics)
        _append_metrics(values, key, contrasts)
        _append_metrics(values, key, rgb_motion)
        _append_metrics(values, key, mask_motion)

    features = _summarize_condition_values(values, sample_counts)
    metadata = {
        "participant_id": participant_id,
        "formal_duration_s": total_s,
        "sample_step_s": step_s,
        "readable_samples": readable,
        "vmax_s_per_bin": float(vmax),
        "eye_tracking_csv": str(eye_tracking_csv),
        "video_frames_csv": str(video_frames_csv),
        "coordinate_bounds": bounds.__dict__,
    }
    return features, metadata


def build_a4_features(
    *,
    manifest_path: Path,
    base_features_path: Path,
    output_path: Path,
    metadata_path: Path,
    bins: int,
    window_s: float,
    step_s: float,
    crop_size: int,
    model_size: int,
    color_quantile: float,
    expand_sigma: float,
    mask_gamma: float,
    force: bool,
) -> pd.DataFrame:
    if output_path.exists() and not force:
        return pd.read_csv(output_path)

    base = pd.read_csv(base_features_path, usecols=["participant_id", "condition"])
    base["participant_id"] = base["participant_id"].map(normalize_participant_id)
    base["condition"] = base["condition"].astype(str)
    participants = sorted(base["participant_id"].unique())
    manifest = _read_manifest(manifest_path)
    feature_frames: list[pd.DataFrame] = []
    metadata: list[dict[str, Any]] = []
    for participant_id in participants:
        source = _participant_source(manifest, participant_id)
        frame, record = extract_a4_features_for_participant(
            participant_id,
            source,
            bins=bins,
            window_s=window_s,
            step_s=step_s,
            crop_size=crop_size,
            model_size=model_size,
            color_quantile=color_quantile,
            expand_sigma=expand_sigma,
            mask_gamma=mask_gamma,
        )
        feature_frames.append(frame)
        metadata.append(record)
        print(
            f"A4 features {participant_id}: {len(frame)} condition rows, "
            f"{record['readable_samples']} readable samples"
        )

    features = pd.concat(feature_frames, ignore_index=True)
    features = base.merge(features, on=["participant_id", "condition"], how="left", validate="one_to_one")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    try:
        features.to_parquet(output_path.with_suffix(".parquet"), index=False)
    except Exception:
        pass
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return features


def _video_artifact_complete(output_dir: Path, participant_id: str) -> bool:
    required = [
        output_dir / f"{participant_id}_four_channel_rgb_heatmap_preview.mp4",
        output_dir / f"{participant_id}_four_channel_rgb_crop.mp4",
        output_dir / f"{participant_id}_four_channel_attention_mask.mp4",
        output_dir / f"{participant_id}_four_channel_rgb_heatmap_tensor.npz",
        output_dir / f"{participant_id}_four_channel_rgb_heatmap_preview.json",
    ]
    return all(path.exists() and path.stat().st_size > 0 for path in required)


def build_four_channel_video_artifacts(
    *,
    participants: list[str],
    manifest_path: Path,
    video_output_dir: Path,
    report_dir: Path,
    bins: int,
    window_s: float,
    step_s: float,
    fps: float,
    crop_size: int,
    model_size: int,
    color_quantile: float,
    expand_sigma: float,
    mask_gamma: float,
    force: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    report_dir.mkdir(parents=True, exist_ok=True)
    for participant_id in participants:
        participant_id = normalize_participant_id(participant_id)
        if _video_artifact_complete(video_output_dir, participant_id) and not force:
            metadata_path = video_output_dir / f"{participant_id}_four_channel_rgb_heatmap_preview.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["skipped_existing"] = True
        else:
            metadata = run_four_channel_preview(
                participant_id=participant_id,
                manifest_path=manifest_path,
                output_dir=video_output_dir,
                report_path=report_dir / f"{participant_id}_four_channel_attended_video_zh.md",
                bins=bins,
                window_s=window_s,
                step_s=step_s,
                fps=fps,
                crop_size=crop_size,
                model_size=model_size,
                color_quantile=color_quantile,
                expand_sigma=expand_sigma,
                mask_gamma=mask_gamma,
                export_tensor=True,
            )
            metadata["skipped_existing"] = False
        row = {
            "participant_id": participant_id,
            "skipped_existing": bool(metadata.get("skipped_existing", False)),
            "n_video_frames": int(metadata["n_video_frames"]),
            "video_duration_s": float(metadata["video_duration_s"]),
            "tensor_shape": "x".join(str(value) for value in metadata.get("tensor_shape") or []),
            "preview_video": metadata["preview_video"],
            "rgb_crop_video": metadata["rgb_crop_video"],
            "attention_mask_video": metadata["attention_mask_video"],
            "four_channel_tensor_npz": metadata["four_channel_tensor_npz"],
            "preview_frame": metadata["preview_frame"],
        }
        for key in ("preview_video", "rgb_crop_video", "attention_mask_video", "four_channel_tensor_npz"):
            path = Path(str(row[key]))
            row[f"{key}_bytes"] = path.stat().st_size if path.exists() else 0
        rows.append(row)
        print(f"4ch video {participant_id}: frames={row['n_video_frames']} skipped={row['skipped_existing']}")
    return pd.DataFrame(rows)


def _validate_frame(frame: pd.DataFrame, expected_labels: int = 135) -> pd.DataFrame:
    required = {
        "participant_id",
        "condition",
        "presentation_position",
        "relaxation",
        "pleasantness",
        "calm",
        "discomfort",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Input frame missing required columns: {missing}")
    frame = frame.copy()
    frame["participant_id"] = frame["participant_id"].map(normalize_participant_id)
    frame["condition"] = frame["condition"].astype(str)
    for name in ("presentation_position", "relaxation", "pleasantness", "calm", "discomfort"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    if frame[list(required)].isna().any().any():
        raise ValueError("Input has missing identifiers or targets")
    if frame[["participant_id", "condition"]].duplicated().any():
        raise ValueError("Input must contain one row per participant-condition")
    if len(frame) != expected_labels:
        raise ValueError(f"Expected {expected_labels} participant-condition rows; found {len(frame)}")
    frame[BINARY_TARGET] = (frame["discomfort"] >= DISCOMFORT_THRESHOLD).astype(int)
    return frame.reset_index(drop=True)


def _modal_columns(frame: pd.DataFrame, modality: str) -> list[str]:
    prefixes = MODALITY_PREFIXES[modality]
    return sorted(
        name
        for name in frame.columns
        if name not in IDENTIFIER_AND_LABEL_COLUMNS and name.startswith(prefixes)
    )


def _condition_baseline(train: pd.DataFrame, test: pd.DataFrame, target: str) -> tuple[np.ndarray, dict[str, float], float]:
    fallback = float(train[target].mean())
    by_condition = train.groupby("condition", sort=True)[target].mean().astype(float).to_dict()
    prediction = np.asarray(
        [float(by_condition.get(condition, fallback)) for condition in test["condition"]],
        dtype=float,
    )
    return prediction, by_condition, fallback


def _history_baseline(test: pd.DataFrame, fallback: float, target: str) -> np.ndarray:
    output = np.full(len(test), fallback, dtype=float)
    for _, group in test.groupby("participant_id", sort=False):
        previous: float | None = None
        for index in group.sort_values("presentation_position", kind="stable").index:
            position = test.index.get_loc(index)
            output[position] = fallback if previous is None else previous
            previous = float(test.loc[index, target])
    return output


def _rank_features(train: pd.DataFrame, columns: list[str], signal: np.ndarray, limit: int) -> list[str]:
    ranked: list[tuple[float, str]] = []
    signal = np.asarray(signal, dtype=float)
    for name in columns:
        values = pd.to_numeric(train[name], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(values) & np.isfinite(signal)
        if valid.sum() < 3:
            continue
        feature_values = values[valid]
        signal_values = signal[valid]
        if np.nanstd(feature_values) <= 0 or np.nanstd(signal_values) <= 0:
            continue
        correlation = float(np.corrcoef(feature_values, signal_values)[0, 1])
        if np.isfinite(correlation):
            ranked.append((abs(correlation), name))
    return [name for _, name in sorted(ranked, key=lambda item: (-item[0], item[1]))[:limit]]


def _selected_by_modality(
    train: pd.DataFrame,
    target: str,
    *,
    feature_limit_per_modality: int,
    binary: bool,
) -> dict[str, list[str]]:
    if binary:
        baseline, _, _ = _condition_baseline(train, train, target)
        signal = train[target].to_numpy(dtype=float) - baseline
    else:
        baseline, _, _ = _condition_baseline(train, train, target)
        signal = train[target].to_numpy(dtype=float) - baseline
    return {
        modality: _rank_features(
            train,
            _modal_columns(train, modality),
            signal,
            feature_limit_per_modality,
        )
        for modality in MODALITY_ORDER
    }


def _combine_selected(selected_by_modality: dict[str, list[str]], combination: str) -> tuple[list[str], dict[str, int]]:
    selected: list[str] = []
    counts: dict[str, int] = {}
    for modality in MODALITY_ORDER:
        retained = selected_by_modality[modality] if modality in combination else []
        selected.extend(retained)
        counts[modality] = len(retained)
    return sorted(selected), counts


def _ridge_pipeline(alpha: float):
    _, SimpleImputer, _, Ridge, *_, Pipeline, StandardScaler = _dependencies()
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def _logistic_pipeline():
    _, SimpleImputer, LogisticRegression, *_rest, Pipeline, StandardScaler = _dependencies()
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=5000,
                    solver="liblinear",
                    random_state=20260704,
                ),
            ),
        ]
    )


def _matrix(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")


def _within_participant_spearman(frame: pd.DataFrame, truth: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    values: list[float] = []
    for participant in sorted(frame["participant_id"].unique()):
        mask = frame["participant_id"].eq(participant).to_numpy()
        values.append(_safe_spearman(truth[mask], prediction[mask]))
    return float(np.mean(values)), float(np.median(values))


def _pairwise_ranking_accuracy(frame: pd.DataFrame, truth: np.ndarray, prediction: np.ndarray) -> float:
    scores: list[float] = []
    for participant in sorted(frame["participant_id"].unique()):
        mask = np.flatnonzero(frame["participant_id"].eq(participant).to_numpy())
        correct = 0.0
        total = 0
        for left_index, left in enumerate(mask):
            for right in mask[left_index + 1 :]:
                truth_delta = truth[left] - truth[right]
                pred_delta = prediction[left] - prediction[right]
                if abs(truth_delta) <= 1e-12:
                    continue
                total += 1
                if abs(pred_delta) <= 1e-12:
                    correct += 0.5
                elif np.sign(truth_delta) == np.sign(pred_delta):
                    correct += 1.0
        if total:
            scores.append(correct / total)
    return float(np.mean(scores)) if scores else float("nan")


def _top_condition_metrics(frame: pd.DataFrame, truth: np.ndarray, prediction: np.ndarray) -> tuple[float, float, float]:
    hits = []
    top3_hits = []
    regrets = []
    for participant in sorted(frame["participant_id"].unique()):
        mask = np.flatnonzero(frame["participant_id"].eq(participant).to_numpy())
        participant_truth = truth[mask]
        participant_prediction = prediction[mask]
        true_best = np.flatnonzero(participant_truth == np.max(participant_truth))
        predicted_order = np.argsort(-participant_prediction, kind="stable")
        predicted_best = int(predicted_order[0])
        hits.append(float(predicted_best in set(int(value) for value in true_best)))
        top3_hits.append(float(any(int(value) in set(int(item) for item in true_best) for value in predicted_order[:3])))
        regrets.append(float(np.max(participant_truth) - participant_truth[predicted_best]))
    return float(np.mean(hits)), float(np.mean(top3_hits)), float(np.mean(regrets))


def _continuous_metrics(frame: pd.DataFrame, target: str, prediction: np.ndarray) -> dict[str, float]:
    truth = frame[target].to_numpy(dtype=float)
    within_mean, within_median = _within_participant_spearman(frame, truth, prediction)
    top_hit, top3_hit, regret = _top_condition_metrics(frame, truth, prediction)
    return {
        "spearman": _safe_spearman(truth, prediction),
        "within_participant_spearman_mean": within_mean,
        "within_participant_spearman_median": within_median,
        "pairwise_ranking_accuracy": _pairwise_ranking_accuracy(frame, truth, prediction),
        "top1_condition_hit": top_hit,
        "top3_condition_hit": top3_hit,
        "top1_regret": regret,
    }


def _binary_metrics(truth: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    _, _, _, _, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score, *_ = _dependencies()
    prediction = (probability >= threshold).astype(int)
    if len(np.unique(truth)) == 2:
        roc_auc = float(roc_auc_score(truth, probability))
        pr_auc = float(average_precision_score(truth, probability))
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "recall": float(recall_score(truth, prediction, zero_division=0)),
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "f1": float(f1_score(truth, prediction, zero_division=0)),
        "false_negatives": float(np.sum((truth == 1) & (prediction == 0))),
        "positives_predicted": float(np.sum(prediction == 1)),
        "threshold": float(threshold),
    }


def evaluate_ranking_and_classification(
    frame: pd.DataFrame,
    *,
    ridge_alpha: float,
    feature_limit_per_modality: int,
    binary_threshold: float,
) -> dict[str, Any]:
    frame = _validate_frame(frame)
    participants = sorted(frame["participant_id"].unique())
    continuous_predictions = {
        combination: {target: np.full(len(frame), np.nan, dtype=float) for target in CONTINUOUS_TARGETS}
        for combination in COMBINATIONS
    }
    binary_probabilities = {combination: np.full(len(frame), np.nan, dtype=float) for combination in COMBINATIONS}
    continuous_baselines = {
        name: {target: np.full(len(frame), np.nan, dtype=float) for target in CONTINUOUS_TARGETS}
        for name in ("condition_only", "history")
    }
    binary_baselines = {name: np.full(len(frame), np.nan, dtype=float) for name in ("condition_only", "history")}
    selection_counts: dict[str, dict[str, dict[str, list[int]]]] = {
        combination: {
            **{target: {modality: [] for modality in MODALITY_ORDER} for target in CONTINUOUS_TARGETS},
            BINARY_TARGET: {modality: [] for modality in MODALITY_ORDER},
        }
        for combination in COMBINATIONS
    }

    for participant in participants:
        test_mask = frame["participant_id"].eq(participant).to_numpy()
        test_indexes = np.flatnonzero(test_mask)
        train = frame.loc[~test_mask].reset_index(drop=True)
        test = frame.loc[test_mask].reset_index(drop=True)

        for target in CONTINUOUS_TARGETS:
            baseline, _, fallback = _condition_baseline(train, test, target)
            continuous_baselines["condition_only"][target][test_indexes] = baseline
            continuous_baselines["history"][target][test_indexes] = _history_baseline(test, fallback, target)
            train_baseline, _, _ = _condition_baseline(train, train, target)
            residual = train[target].to_numpy(dtype=float) - train_baseline
            selected_by_modality = _selected_by_modality(
                train,
                target,
                feature_limit_per_modality=feature_limit_per_modality,
                binary=False,
            )
            for combination in COMBINATIONS:
                columns, counts = _combine_selected(selected_by_modality, combination)
                for modality, count in counts.items():
                    selection_counts[combination][target][modality].append(count)
                if columns:
                    model = _ridge_pipeline(ridge_alpha)
                    model.fit(_matrix(train, columns), residual)
                    held_out_residual = model.predict(_matrix(test, columns))
                else:
                    held_out_residual = np.zeros(len(test), dtype=float)
                continuous_predictions[combination][target][test_indexes] = np.clip(
                    baseline + held_out_residual,
                    0.0,
                    1.0,
                )

        baseline, _, fallback = _condition_baseline(train, test, BINARY_TARGET)
        binary_baselines["condition_only"][test_indexes] = np.clip(baseline, 0.0, 1.0)
        binary_baselines["history"][test_indexes] = np.clip(_history_baseline(test, fallback, BINARY_TARGET), 0.0, 1.0)
        selected_by_modality = _selected_by_modality(
            train,
            BINARY_TARGET,
            feature_limit_per_modality=feature_limit_per_modality,
            binary=True,
        )
        y_train = train[BINARY_TARGET].to_numpy(dtype=int)
        for combination in COMBINATIONS:
            columns, counts = _combine_selected(selected_by_modality, combination)
            for modality, count in counts.items():
                selection_counts[combination][BINARY_TARGET][modality].append(count)
            if columns and len(np.unique(y_train)) == 2:
                model = _logistic_pipeline()
                model.fit(_matrix(train, columns), y_train)
                probability = model.predict_proba(_matrix(test, columns))[:, 1]
            else:
                probability = np.full(len(test), float(np.mean(y_train)), dtype=float)
            binary_probabilities[combination][test_indexes] = np.clip(probability, 0.0, 1.0)

    for target in CONTINUOUS_TARGETS:
        for name, predictions in continuous_baselines.items():
            if np.isnan(predictions[target]).any():
                raise AssertionError(f"Missing continuous baseline predictions for {name}/{target}")
    for name, probability in binary_baselines.items():
        if np.isnan(probability).any():
            raise AssertionError(f"Missing binary baseline predictions for {name}")

    continuous_metric_rows: list[dict[str, Any]] = []
    binary_metric_rows: list[dict[str, Any]] = []
    for baseline_name in ("condition_only", "history"):
        for target in CONTINUOUS_TARGETS:
            continuous_metric_rows.append(
                {
                    "record_type": "baseline",
                    "baseline": baseline_name,
                    "combination": "",
                    "target": target,
                    **_continuous_metrics(frame, target, continuous_baselines[baseline_name][target]),
                }
            )
        binary_metric_rows.append(
            {
                "record_type": "baseline",
                "baseline": baseline_name,
                "combination": "",
                "target": BINARY_TARGET,
                **_binary_metrics(
                    frame[BINARY_TARGET].to_numpy(dtype=int),
                    binary_baselines[baseline_name],
                    threshold=binary_threshold,
                ),
            }
        )

    for combination in COMBINATIONS:
        for target in CONTINUOUS_TARGETS:
            prediction = continuous_predictions[combination][target]
            if np.isnan(prediction).any():
                raise AssertionError(f"Missing predictions for {combination}/{target}")
            row = {
                "record_type": "combination",
                "baseline": "",
                "combination": combination,
                "target": target,
                "n_labels": len(frame),
                "ridge_alpha": ridge_alpha,
                "feature_limit_per_modality": feature_limit_per_modality,
                **_continuous_metrics(frame, target, prediction),
            }
            for modality in MODALITY_ORDER:
                counts = np.asarray(selection_counts[combination][target][modality], dtype=float)
                row[f"selected_{modality}_min"] = float(counts.min())
                row[f"selected_{modality}_mean"] = float(counts.mean())
                row[f"selected_{modality}_max"] = float(counts.max())
            continuous_metric_rows.append(row)

        probability = binary_probabilities[combination]
        if np.isnan(probability).any():
            raise AssertionError(f"Missing binary predictions for {combination}")
        binary_metric_rows.append(
            {
                "record_type": "combination",
                "baseline": "",
                "combination": combination,
                "target": BINARY_TARGET,
                "n_labels": len(frame),
                "positive_labels": int(frame[BINARY_TARGET].sum()),
                "feature_limit_per_modality": feature_limit_per_modality,
                **_binary_metrics(
                    frame[BINARY_TARGET].to_numpy(dtype=int),
                    probability,
                    threshold=binary_threshold,
                ),
            }
        )

    oof_rows: list[dict[str, Any]] = []
    for row_index, source_row in frame.iterrows():
        for combination in COMBINATIONS:
            for target in CONTINUOUS_TARGETS:
                oof_rows.append(
                    {
                        "task": "continuous_ranking",
                        "combination": combination,
                        "participant_id": source_row["participant_id"],
                        "condition": source_row["condition"],
                        "presentation_position": int(source_row["presentation_position"]),
                        "target": target,
                        "truth": float(source_row[target]),
                        "prediction": float(continuous_predictions[combination][target][row_index]),
                        "condition_only_prediction": float(continuous_baselines["condition_only"][target][row_index]),
                        "history_prediction": float(continuous_baselines["history"][target][row_index]),
                    }
                )
            oof_rows.append(
                {
                    "task": "binary_classification",
                    "combination": combination,
                    "participant_id": source_row["participant_id"],
                    "condition": source_row["condition"],
                    "presentation_position": int(source_row["presentation_position"]),
                    "target": BINARY_TARGET,
                    "truth": int(source_row[BINARY_TARGET]),
                    "prediction": float(binary_probabilities[combination][row_index]),
                    "condition_only_prediction": float(binary_baselines["condition_only"][row_index]),
                    "history_prediction": float(binary_baselines["history"][row_index]),
                }
            )

    return {
        "frame": frame,
        "continuous_metrics": pd.DataFrame(continuous_metric_rows),
        "binary_metrics": pd.DataFrame(binary_metric_rows),
        "oof_predictions": pd.DataFrame(oof_rows),
    }


def _head_to_head_continuous(continuous: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("V", "A", "single video"),
        ("PV", "PA", "physio + video"),
        ("HV", "HA", "head + video"),
        ("EV", "EA", "eye + video"),
        ("PHEV", "PHEA", "replace original video in PHEV"),
        ("PHEV", "PHEVA", "add A4 on top of original video"),
        ("PHE", "PHEA", "add A4 to PHE"),
    ]
    rows: list[dict[str, Any]] = []
    combos = continuous.loc[continuous["record_type"].eq("combination")]
    for target in CONTINUOUS_TARGETS:
        target_frame = combos.loc[combos["target"].eq(target)].set_index("combination")
        for original, attended, comparison in pairs:
            if original not in target_frame.index or attended not in target_frame.index:
                continue
            left = target_frame.loc[original]
            right = target_frame.loc[attended]
            rows.append(
                {
                    "target": target,
                    "comparison": comparison,
                    "original_combination": original,
                    "attended_combination": attended,
                    "original_spearman": float(left["spearman"]),
                    "attended_spearman": float(right["spearman"]),
                    "delta_spearman": float(right["spearman"] - left["spearman"]),
                    "original_within_spearman": float(left["within_participant_spearman_mean"]),
                    "attended_within_spearman": float(right["within_participant_spearman_mean"]),
                    "delta_within_spearman": float(
                        right["within_participant_spearman_mean"] - left["within_participant_spearman_mean"]
                    ),
                    "original_pairwise_accuracy": float(left["pairwise_ranking_accuracy"]),
                    "attended_pairwise_accuracy": float(right["pairwise_ranking_accuracy"]),
                    "delta_pairwise_accuracy": float(
                        right["pairwise_ranking_accuracy"] - left["pairwise_ranking_accuracy"]
                    ),
                    "original_top1_hit": float(left["top1_condition_hit"]),
                    "attended_top1_hit": float(right["top1_condition_hit"]),
                    "delta_top1_hit": float(right["top1_condition_hit"] - left["top1_condition_hit"]),
                    "original_top1_regret": float(left["top1_regret"]),
                    "attended_top1_regret": float(right["top1_regret"]),
                    "delta_top1_regret": float(right["top1_regret"] - left["top1_regret"]),
                }
            )
    return pd.DataFrame(rows)


def _head_to_head_binary(binary: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("V", "A", "single video"),
        ("PV", "PA", "physio + video"),
        ("HV", "HA", "head + video"),
        ("EV", "EA", "eye + video"),
        ("PHEV", "PHEA", "replace original video in PHEV"),
        ("PHEV", "PHEVA", "add A4 on top of original video"),
        ("PHE", "PHEA", "add A4 to PHE"),
    ]
    combos = binary.loc[binary["record_type"].eq("combination")].set_index("combination")
    rows: list[dict[str, Any]] = []
    for original, attended, comparison in pairs:
        if original not in combos.index or attended not in combos.index:
            continue
        left = combos.loc[original]
        right = combos.loc[attended]
        rows.append(
            {
                "comparison": comparison,
                "original_combination": original,
                "attended_combination": attended,
                "original_roc_auc": float(left["roc_auc"]),
                "attended_roc_auc": float(right["roc_auc"]),
                "delta_roc_auc": float(right["roc_auc"] - left["roc_auc"]),
                "original_pr_auc": float(left["pr_auc"]),
                "attended_pr_auc": float(right["pr_auc"]),
                "delta_pr_auc": float(right["pr_auc"] - left["pr_auc"]),
                "original_recall": float(left["recall"]),
                "attended_recall": float(right["recall"]),
                "delta_recall": float(right["recall"] - left["recall"]),
                "original_precision": float(left["precision"]),
                "attended_precision": float(right["precision"]),
                "delta_precision": float(right["precision"] - left["precision"]),
                "original_false_negatives": float(left["false_negatives"]),
                "attended_false_negatives": float(right["false_negatives"]),
                "delta_false_negatives": float(right["false_negatives"] - left["false_negatives"]),
            }
        )
    return pd.DataFrame(rows)


def _continuous_row(row: pd.Series) -> str:
    return (
        f"| {row['combination']} | {_fmt(row['spearman'])} | "
        f"{_fmt(row['within_participant_spearman_mean'])} | "
        f"{_fmt(row['pairwise_ranking_accuracy'], percent=True)} | "
        f"{_fmt(row['top1_condition_hit'], percent=True)} | "
        f"{_fmt(row['top1_regret'])} |"
    )


def _binary_row(row: pd.Series) -> str:
    return (
        f"| {row['combination']} | {_fmt(row['roc_auc'])} | {_fmt(row['pr_auc'])} | "
        f"{_fmt(row['recall'], percent=True)} | {_fmt(row['precision'], percent=True)} | "
        f"{_fmt(row['f1'])} | {int(row['false_negatives'])} |"
    )


def write_report(
    report_path: Path,
    *,
    continuous: pd.DataFrame,
    binary: pd.DataFrame,
    head_continuous: pd.DataFrame,
    head_binary: pd.DataFrame,
    video_manifest: pd.DataFrame | None,
    output_dir: Path,
    parameters: dict[str, Any],
) -> None:
    key_continuous = continuous.loc[continuous["record_type"].eq("combination")]
    key_binary = binary.loc[binary["record_type"].eq("combination")]
    target_sections: list[str] = []
    for target in CONTINUOUS_TARGETS:
        rows = key_continuous.loc[key_continuous["target"].eq(target)].sort_values(
            ["within_participant_spearman_mean", "spearman"],
            ascending=[False, False],
        )
        target_sections.extend(
            [
                f"## {target} ranking top combinations",
                "",
                "| Combination | Global Spearman | Within Spearman mean | Pairwise acc | Top1 hit | Top1 regret |",
                "|---|---:|---:|---:|---:|---:|",
                *[_continuous_row(row) for _, row in rows.head(10).iterrows()],
                "",
            ]
        )

    binary_top = key_binary.sort_values(["pr_auc", "roc_auc"], ascending=[False, False])
    head_cont_rows = [
        "| Target | 对照 | 原组合 | A4组合 | Δ global rho | Δ within rho | Δ pairwise | Δ top1 | Δ regret |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in head_continuous.iterrows():
        head_cont_rows.append(
            "| {target} | {comparison} | {original} | {attended} | {d_global:+.4f} | "
            "{d_within:+.4f} | {d_pair:+.1%} | {d_top:+.1%} | {d_regret:+.4f} |".format(
                target=row["target"],
                comparison=row["comparison"],
                original=row["original_combination"],
                attended=row["attended_combination"],
                d_global=row["delta_spearman"],
                d_within=row["delta_within_spearman"],
                d_pair=row["delta_pairwise_accuracy"],
                d_top=row["delta_top1_hit"],
                d_regret=row["delta_top1_regret"],
            )
        )
    head_bin_rows = [
        "| 对照 | 原组合 | A4组合 | Δ ROC-AUC | Δ PR-AUC | Δ recall | Δ precision | Δ FN |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in head_binary.iterrows():
        head_bin_rows.append(
            "| {comparison} | {original} | {attended} | {d_roc:+.4f} | {d_pr:+.4f} | "
            "{d_rec:+.1%} | {d_prec:+.1%} | {d_fn:+.0f} |".format(
                comparison=row["comparison"],
                original=row["original_combination"],
                attended=row["attended_combination"],
                d_roc=row["delta_roc_auc"],
                d_pr=row["delta_pr_auc"],
                d_rec=row["delta_recall"],
                d_prec=row["delta_precision"],
                d_fn=row["delta_false_negatives"],
            )
        )

    video_summary = []
    if video_manifest is not None and not video_manifest.empty:
        total_bytes = int(
            video_manifest[
                [
                    "preview_video_bytes",
                    "rgb_crop_video_bytes",
                    "attention_mask_video_bytes",
                    "four_channel_tensor_npz_bytes",
                ]
            ]
            .sum()
            .sum()
        )
        video_summary = [
            "## Four-channel artifacts",
            "",
            f"- Participants: {video_manifest['participant_id'].nunique()}.",
            f"- Total generated/verified bytes: {total_bytes / 1024**3:.2f} GiB.",
            f"- Manifest: `{(output_dir / 'four_channel_video_manifest.csv').as_posix()}`",
            "",
        ]

    lines = [
        "# Four-channel attended video ranking/classification benchmark",
        "",
        "## 口径",
        "",
        "- A4 = 224x224x4, channels are R/G/B/attention_mask; RGB is not darkened.",
        "- Base table uses NeuroKit ECG features: `artifacts/features/ecg_neurokit/condition_features.csv`.",
        "- Continuous targets are evaluated by ranking metrics, not MAE: relaxation, pleasantness, calm.",
        "- Discomfort is evaluated as binary high_discomfort = discomfort >= 0.50.",
        "- Inference unit is 135 participant-condition rows; LOPO leaves one participant out.",
        "",
        "## Continuous A4 head-to-head",
        "",
        *head_cont_rows,
        "",
        "## High-discomfort A4 head-to-head",
        "",
        *head_bin_rows,
        "",
        *target_sections,
        "## High-discomfort top combinations",
        "",
        "| Combination | ROC-AUC | PR-AUC | Recall | Precision | F1 | FN |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *[_binary_row(row) for _, row in binary_top.head(12).iterrows()],
        "",
        *video_summary,
        "## Outputs",
        "",
        f"- Continuous metrics: `{(output_dir / 'continuous_ranking_metrics.csv').as_posix()}`",
        f"- Binary metrics: `{(output_dir / 'binary_high_discomfort_metrics.csv').as_posix()}`",
        f"- Continuous head-to-head: `{(output_dir / 'head_to_head_continuous.csv').as_posix()}`",
        f"- Binary head-to-head: `{(output_dir / 'head_to_head_binary.csv').as_posix()}`",
        f"- OOF predictions: `{(output_dir / 'oof_predictions.csv').as_posix()}`",
        f"- A4 features: `{(output_dir / 'a4_condition_features.csv').as_posix()}`",
        "",
        "## Parameters",
        "",
        f"- Feature cap per modality: {parameters['feature_limit_per_modality']}.",
        f"- Ridge alpha: {parameters['ridge_alpha']}.",
        f"- Binary threshold: {parameters['binary_threshold']}.",
        f"- A4 rolling window: {parameters['window_s']} s.",
        f"- A4 crop source: {parameters['crop_size']} px; model size: {parameters['model_size']} px.",
        f"- A4 attention expansion sigma: {parameters['expand_sigma']} px; gamma: {parameters['mask_gamma']}.",
        "- Spatial mapping remains pilot gaze_hit_y/z plane normalized to frame pixels, not calibrated Unity projection.",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run(
    *,
    manifest_path: Path,
    base_features_path: Path,
    output_dir: Path,
    video_output_dir: Path,
    report_path: Path,
    bins: int,
    window_s: float,
    step_s: float,
    fps: float,
    crop_size: int,
    model_size: int,
    color_quantile: float,
    expand_sigma: float,
    mask_gamma: float,
    ridge_alpha: float,
    feature_limit_per_modality: int,
    binary_threshold: float,
    force_features: bool,
    force_videos: bool,
    skip_videos: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_ids = pd.read_csv(base_features_path, usecols=["participant_id"])
    participants = sorted(base_ids["participant_id"].map(normalize_participant_id).unique())
    video_manifest: pd.DataFrame | None = None
    if not skip_videos:
        video_manifest = build_four_channel_video_artifacts(
            participants=participants,
            manifest_path=manifest_path,
            video_output_dir=video_output_dir,
            report_dir=output_dir / "participant_video_reports",
            bins=bins,
            window_s=window_s,
            step_s=step_s,
            fps=fps,
            crop_size=crop_size,
            model_size=model_size,
            color_quantile=color_quantile,
            expand_sigma=expand_sigma,
            mask_gamma=mask_gamma,
            force=force_videos,
        )
        video_manifest.to_csv(output_dir / "four_channel_video_manifest.csv", index=False)

    a4_path = output_dir / "a4_condition_features.csv"
    a4_features = build_a4_features(
        manifest_path=manifest_path,
        base_features_path=base_features_path,
        output_path=a4_path,
        metadata_path=output_dir / "a4_feature_metadata.json",
        bins=bins,
        window_s=window_s,
        step_s=step_s,
        crop_size=crop_size,
        model_size=model_size,
        color_quantile=color_quantile,
        expand_sigma=expand_sigma,
        mask_gamma=mask_gamma,
        force=force_features,
    )
    base = pd.read_csv(base_features_path)
    frame = base.merge(a4_features, on=["participant_id", "condition"], how="left", validate="one_to_one")
    a4_cols = [name for name in frame.columns if name.startswith(A4_PREFIX)]
    missing_a4_rows = int(frame[a4_cols].isna().all(axis=1).sum())
    if missing_a4_rows:
        raise ValueError(f"{missing_a4_rows} rows have no A4 features")

    result = evaluate_ranking_and_classification(
        frame,
        ridge_alpha=ridge_alpha,
        feature_limit_per_modality=feature_limit_per_modality,
        binary_threshold=binary_threshold,
    )
    evaluated_frame = result["frame"]
    continuous = result["continuous_metrics"]
    binary = result["binary_metrics"]
    oof = result["oof_predictions"]
    head_continuous = _head_to_head_continuous(continuous)
    head_binary = _head_to_head_binary(binary)

    continuous.to_csv(output_dir / "continuous_ranking_metrics.csv", index=False)
    binary.to_csv(output_dir / "binary_high_discomfort_metrics.csv", index=False)
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)
    head_continuous.to_csv(output_dir / "head_to_head_continuous.csv", index=False)
    head_binary.to_csv(output_dir / "head_to_head_binary.csv", index=False)
    write_report(
        report_path,
        continuous=continuous,
        binary=binary,
        head_continuous=head_continuous,
        head_binary=head_binary,
        video_manifest=video_manifest,
        output_dir=output_dir,
        parameters={
            "feature_limit_per_modality": feature_limit_per_modality,
            "ridge_alpha": ridge_alpha,
            "binary_threshold": binary_threshold,
            "window_s": window_s,
            "crop_size": crop_size,
            "model_size": model_size,
            "expand_sigma": expand_sigma,
            "mask_gamma": mask_gamma,
        },
    )
    summary = {
        "output_dir": str(output_dir),
        "report": str(report_path),
        "n_rows": int(len(evaluated_frame)),
        "participants": int(evaluated_frame["participant_id"].nunique()),
        "a4_feature_columns": int(len(a4_cols)),
        "positive_high_discomfort": int(evaluated_frame[BINARY_TARGET].sum()),
        "continuous_metrics": str(output_dir / "continuous_ranking_metrics.csv"),
        "binary_metrics": str(output_dir / "binary_high_discomfort_metrics.csv"),
        "head_to_head_continuous": str(output_dir / "head_to_head_continuous.csv"),
        "head_to_head_binary": str(output_dir / "head_to_head_binary.csv"),
        "skip_videos": bool(skip_videos),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run four-channel A4 ranking/classification benchmark.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--base-features", type=Path, default=DEFAULT_BASE_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--video-output-dir", type=Path, default=DEFAULT_VIDEO_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--bins", type=int, default=72)
    parser.add_argument("--window-s", type=float, default=20.0)
    parser.add_argument("--step-s", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--crop-size", type=int, default=520)
    parser.add_argument("--model-size", type=int, default=224)
    parser.add_argument("--color-quantile", type=float, default=0.995)
    parser.add_argument("--expand-sigma", type=float, default=28.0)
    parser.add_argument("--mask-gamma", type=float, default=0.65)
    parser.add_argument("--ridge-alpha", type=float, default=old_fusion.RIDGE_ALPHA)
    parser.add_argument("--feature-limit-per-modality", type=int, default=old_fusion.FEATURES_PER_MODALITY)
    parser.add_argument("--binary-threshold", type=float, default=0.50)
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--force-videos", action="store_true")
    parser.add_argument("--skip-videos", action="store_true")
    args = parser.parse_args()
    summary = run(
        manifest_path=args.manifest,
        base_features_path=args.base_features,
        output_dir=args.output_dir,
        video_output_dir=args.video_output_dir,
        report_path=args.report,
        bins=args.bins,
        window_s=args.window_s,
        step_s=args.step_s,
        fps=args.fps,
        crop_size=args.crop_size,
        model_size=args.model_size,
        color_quantile=args.color_quantile,
        expand_sigma=args.expand_sigma,
        mask_gamma=args.mask_gamma,
        ridge_alpha=args.ridge_alpha,
        feature_limit_per_modality=args.feature_limit_per_modality,
        binary_threshold=args.binary_threshold,
        force_features=args.force_features,
        force_videos=args.force_videos,
        skip_videos=args.skip_videos,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

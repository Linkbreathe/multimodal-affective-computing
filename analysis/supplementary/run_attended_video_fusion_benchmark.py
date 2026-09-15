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
sys.path.insert(0, str(ROOT / "src"))
from analysis.supplementary.run_attended_video_preview import (  # noqa: E402
    _current_gaze,
    _estimate_vmax,
    _heatmap_mask,
    _make_model_input,
    _nearest_video_frame,
    _read_video_frames,
    _rolling_gaze,
    _soft_attend,
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
from mac.features.common import robust_stats, safe_divide  # noqa: E402
from mac.models import minimal_fusion as fusion  # noqa: E402
from mac.data.io import normalize_participant_id  # noqa: E402


DEFAULT_BASE_FEATURES = ROOT / "artifacts" / "features" / "video_ml" / "condition_features.csv"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "fusion_attended_video"
DEFAULT_REPORT = ROOT / "artifacts" / "reports" / "attended_video_fusion_benchmark_zh.md"

ATTENDED_MODALITY = "A"
ATTENDED_PREFIX = "att_"
MODALITY_PREFIXES = {
    "P": ("eeg_", "ecg_"),
    "H": ("head_",),
    "E": ("eye_",),
    "V": ("video_",),
    "A": (ATTENDED_PREFIX,),
}
MODALITY_ORDER = tuple(MODALITY_PREFIXES)
COMBINATIONS = tuple(
    "".join(parts)
    for size in range(1, len(MODALITY_ORDER) + 1)
    for parts in combinations(MODALITY_ORDER, size)
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


def _colorfulness(frame: np.ndarray) -> float:
    values = frame.astype(float)
    rg = values[:, :, 2] - values[:, :, 1]
    yb = 0.5 * (values[:, :, 2] + values[:, :, 1]) - values[:, :, 0]
    return float(
        np.sqrt(np.var(rg) + np.var(yb))
        + 0.3 * np.sqrt(float(np.mean(rg)) ** 2 + float(np.mean(yb)) ** 2)
    )


def _visual_snapshot(frame: np.ndarray) -> dict[str, float]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return {
        "brightness": float(np.mean(hsv[:, :, 2])),
        "saturation": float(np.mean(hsv[:, :, 1])),
        "colorfulness": _colorfulness(frame),
        "texture": float(np.std(gray)),
        "edge": float(np.mean(cv2.Canny(gray, 80, 160) > 0)),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }


def _small_gray(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (112, 112), interpolation=cv2.INTER_AREA)


def _motion_metrics(previous_gray: np.ndarray | None, gray: np.ndarray) -> dict[str, float]:
    if previous_gray is None:
        return {"optical_flow": float("nan"), "scene_change": float("nan")}
    flow = cv2.calcOpticalFlowFarneback(previous_gray, gray, None, 0.5, 2, 15, 2, 5, 1.1, 0)
    return {
        "optical_flow": float(np.mean(np.linalg.norm(flow, axis=2))),
        "scene_change": float(np.mean(cv2.absdiff(previous_gray, gray))),
    }


def _mask_entropy(mask: np.ndarray) -> float:
    values = mask.astype(float).ravel()
    total = float(values.sum())
    if total <= 0:
        return 0.0
    probabilities = values[values > 0] / total
    return float(-(probabilities * np.log2(probabilities)).sum() / np.log2(values.size))


def _mask_center(mask: np.ndarray) -> tuple[float, float]:
    total = float(mask.sum())
    if total <= 0:
        return float("nan"), float("nan")
    height, width = mask.shape
    ys, xs = np.indices(mask.shape, dtype=float)
    return (
        float((xs * mask).sum() / total / max(1, width - 1)),
        float((ys * mask).sum() / total / max(1, height - 1)),
    )


def _append_prefixed(
    store: dict[tuple[str, str], dict[str, list[float]]],
    key: tuple[str, str],
    prefix: str,
    metrics: dict[str, float],
) -> None:
    for name, value in metrics.items():
        store[key][f"{prefix}_{name}"].append(float(value))


def _summarize_condition_values(
    values: dict[tuple[str, str], dict[str, list[float]]],
    sample_counts: dict[tuple[str, str], int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (participant_id, condition), metrics in sorted(values.items()):
        row: dict[str, Any] = {
            "participant_id": participant_id,
            "condition": condition,
            "att_sample_count": int(sample_counts[(participant_id, condition)]),
        }
        for name, series in sorted(metrics.items()):
            row.update(robust_stats(np.asarray(series, dtype=float), f"att_{name}"))
        rows.append(row)
    return pd.DataFrame(rows)


def extract_attended_features_for_participant(
    participant_id: str,
    source: pd.Series,
    *,
    bins: int,
    window_s: float,
    step_s: float,
    crop_size: int,
    model_size: int,
    color_quantile: float,
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
    previous_crop_gray: dict[str, np.ndarray] = {}
    previous_weighted_gray: dict[str, np.ndarray] = {}
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
        mask = _heatmap_mask(rolling, frame.shape[1], frame.shape[0], bins=bins, vmax=vmax)
        model_input = _make_model_input(
            frame,
            mask,
            current,
            crop_size=crop_size,
            model_size=model_size,
        )
        weighted = _soft_attend(frame, mask)
        weighted_small = cv2.resize(weighted, (model_size, model_size), interpolation=cv2.INTER_AREA)
        original_small = cv2.resize(frame, (model_size, model_size), interpolation=cv2.INTER_AREA)

        condition = _condition_at_time(segments, float(time_s))
        key = (participant_id, condition)
        sample_counts[key] += 1

        crop_metrics = _visual_snapshot(model_input)
        weighted_metrics = _visual_snapshot(weighted_small)
        original_metrics = _visual_snapshot(original_small)
        _append_prefixed(values, key, "crop", crop_metrics)
        _append_prefixed(values, key, "weighted", weighted_metrics)

        contrasts = {
            f"crop_minus_global_{name}": crop_metrics[name] - original_metrics[name]
            for name in crop_metrics
        }
        ratios = {
            f"crop_to_global_{name}": safe_divide(crop_metrics[name], original_metrics[name])
            for name in crop_metrics
        }
        _append_prefixed(values, key, "contrast", contrasts)
        _append_prefixed(values, key, "contrast", ratios)

        crop_gray = _small_gray(model_input)
        weighted_gray = _small_gray(weighted_small)
        crop_motion = _motion_metrics(previous_crop_gray.get(condition), crop_gray)
        weighted_motion = _motion_metrics(previous_weighted_gray.get(condition), weighted_gray)
        previous_crop_gray[condition] = crop_gray
        previous_weighted_gray[condition] = weighted_gray
        _append_prefixed(values, key, "crop", crop_motion)
        _append_prefixed(values, key, "weighted", weighted_motion)

        mask_center_x, mask_center_y = _mask_center(mask)
        mask_metrics = {
            "mean": float(np.mean(mask)),
            "std": float(np.std(mask)),
            "max": float(np.max(mask)),
            "coverage_025": float(np.mean(mask >= 0.25)),
            "coverage_050": float(np.mean(mask >= 0.50)),
            "entropy": _mask_entropy(mask),
            "center_x": mask_center_x,
            "center_y": mask_center_y,
            "current_gaze_x": float(current["gaze_x_norm"]),
            "current_gaze_y": float(current["gaze_y_norm"]),
        }
        _append_prefixed(values, key, "mask", mask_metrics)

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


def build_attended_features(
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
        frame, record = extract_attended_features_for_participant(
            participant_id,
            source,
            bins=bins,
            window_s=window_s,
            step_s=step_s,
            crop_size=crop_size,
            model_size=model_size,
            color_quantile=color_quantile,
        )
        feature_frames.append(frame)
        metadata.append(record)
        print(
            f"{participant_id}: {len(frame)} condition rows, "
            f"{record['readable_samples']} readable attended samples"
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


def _evaluate_with_attended(frame: pd.DataFrame, *, random_seed: int, random_simulations: int) -> dict[str, Any]:
    original_prefixes = fusion.MODALITY_PREFIXES
    original_order = fusion.MODALITY_ORDER
    original_combinations = fusion.COMBINATIONS
    try:
        fusion.MODALITY_PREFIXES = MODALITY_PREFIXES
        fusion.MODALITY_ORDER = MODALITY_ORDER
        fusion.COMBINATIONS = COMBINATIONS
        return fusion.evaluate_minimal_fusion_frame(
            frame,
            random_seed=random_seed,
            random_simulations=random_simulations,
            expected_labels=fusion.EXPECTED_LABELS,
        )
    finally:
        fusion.MODALITY_PREFIXES = original_prefixes
        fusion.MODALITY_ORDER = original_order
        fusion.COMBINATIONS = original_combinations


def _head_to_head_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    combinations_to_compare = [
        ("V", "A", "single video"),
        ("PV", "PA", "physio + video"),
        ("HV", "HA", "head + video"),
        ("EV", "EA", "eye + video"),
        ("PHEV", "PHEA", "all non-video + replacement video"),
        ("PHE", "PHEA", "add attended video to PHE"),
        ("PHEV", "PHEVA", "add attended video on top of original video"),
    ]
    combos = metrics.loc[metrics["record_type"].eq("combination")].set_index("combination")
    rows: list[dict[str, Any]] = []
    for original, attended, comparison in combinations_to_compare:
        if original not in combos.index or attended not in combos.index:
            continue
        left = combos.loc[original]
        right = combos.loc[attended]
        rows.append(
            {
                "comparison": comparison,
                "original_combination": original,
                "attended_combination": attended,
                "original_relaxation_mae": float(left["relaxation_mae"]),
                "attended_relaxation_mae": float(right["relaxation_mae"]),
                "delta_relaxation_mae_attended_minus_original": float(
                    right["relaxation_mae"] - left["relaxation_mae"]
                ),
                "original_discomfort_mae": float(left["discomfort_mae"]),
                "attended_discomfort_mae": float(right["discomfort_mae"]),
                "delta_discomfort_mae_attended_minus_original": float(
                    right["discomfort_mae"] - left["discomfort_mae"]
                ),
                "original_discomfort_recall": float(left["discomfort_high_recall"]),
                "attended_discomfort_recall": float(right["discomfort_high_recall"]),
                "delta_discomfort_recall_attended_minus_original": float(
                    right["discomfort_high_recall"] - left["discomfort_high_recall"]
                ),
                "original_false_negatives": float(left["discomfort_high_false_negatives"]),
                "attended_false_negatives": float(right["discomfort_high_false_negatives"]),
                "delta_false_negatives_attended_minus_original": float(
                    right["discomfort_high_false_negatives"] - left["discomfort_high_false_negatives"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _best_table(metrics: pd.DataFrame, key: str, *, ascending: bool, n: int = 8) -> pd.DataFrame:
    combos = metrics.loc[metrics["record_type"].eq("combination")].copy()
    return combos.sort_values(key, ascending=ascending).head(n)


def _combination_row(row: pd.Series) -> str:
    return (
        f"| {row['combination']} | {_fmt(row['relaxation_mae'])} | "
        f"{_fmt(row['relaxation_spearman'])} | {_fmt(row['discomfort_mae'])} | "
        f"{_fmt(row['discomfort_spearman'])} | "
        f"{_fmt(row['discomfort_high_recall'], percent=True)} | "
        f"{_fmt(row['discomfort_high_precision'], percent=True)} | "
        f"{int(row['discomfort_high_false_negatives'])} |"
    )


def write_report(
    report_path: Path,
    *,
    metrics: pd.DataFrame,
    head_to_head: pd.DataFrame,
    attended_features: pd.DataFrame,
    output_dir: Path,
    parameters: dict[str, Any],
) -> None:
    combos = metrics.loc[metrics["record_type"].eq("combination")].set_index("combination")
    has_v_or_a = metrics.loc[metrics["record_type"].eq("combination")].copy()
    has_v_or_a["has_A"] = has_v_or_a["combination"].str.contains("A", regex=False)
    has_v_or_a["has_V"] = has_v_or_a["combination"].str.contains("V", regex=False)
    best_relax = _best_table(metrics, "relaxation_mae", ascending=True)
    best_discomfort = _best_table(metrics, "discomfort_mae", ascending=True)
    best_recall = _best_table(metrics, "discomfort_high_recall", ascending=False)

    def combo_value(name: str, metric: str) -> str:
        if name not in combos.index:
            return "NA"
        return _fmt(combos.loc[name, metric], percent="recall" in metric or "precision" in metric)

    head_rows = [
        "| 对照 | 原组合 | A 组合 | Δ Relax MAE | Δ Discomfort MAE | Δ recall | Δ false negatives |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for _, row in head_to_head.iterrows():
        head_rows.append(
            "| {comparison} | {original} | {attended} | {d_rel:+.4f} | {d_dis:+.4f} | "
            "{d_recall:+.1%} | {d_fn:+.0f} |".format(
                comparison=row["comparison"],
                original=row["original_combination"],
                attended=row["attended_combination"],
                d_rel=row["delta_relaxation_mae_attended_minus_original"],
                d_dis=row["delta_discomfort_mae_attended_minus_original"],
                d_recall=row["delta_discomfort_recall_attended_minus_original"],
                d_fn=row["delta_false_negatives_attended_minus_original"],
            )
        )

    ranking_header = [
        "| Combination | Relax MAE | Relax rho | Discomfort MAE | Discomfort rho | "
        "High recall | Precision | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines = [
        "# Attended video vs original modality fusion benchmark",
        "",
        "## 结论",
        "",
        "- 这次比较把 A 定义为 `egocentric video + rolling gaze heatmap` 生成的 attended-video 特征；",
        "  V 仍然是原来的全局 egocentric video 手工视觉特征。",
        "- 评估单位仍是 135 个 participant-condition；没有把帧或 10 秒窗口当成独立标签。",
        "- 当前结果是 research-only，不代表可部署模型。",
        "",
        "## 最直接对照",
        "",
        *head_rows,
        "",
        "## 关键组合",
        "",
        "| 组合 | Relax MAE | Discomfort MAE | Discomfort recall | FN |",
        "|---|---:|---:|---:|---:|",
        (
            f"| V | {combo_value('V', 'relaxation_mae')} | {combo_value('V', 'discomfort_mae')} | "
            f"{combo_value('V', 'discomfort_high_recall')} | "
            f"{int(combos.loc['V', 'discomfort_high_false_negatives']) if 'V' in combos.index else 'NA'} |"
        ),
        (
            f"| A | {combo_value('A', 'relaxation_mae')} | {combo_value('A', 'discomfort_mae')} | "
            f"{combo_value('A', 'discomfort_high_recall')} | "
            f"{int(combos.loc['A', 'discomfort_high_false_negatives']) if 'A' in combos.index else 'NA'} |"
        ),
        (
            f"| PHE | {combo_value('PHE', 'relaxation_mae')} | "
            f"{combo_value('PHE', 'discomfort_mae')} | {combo_value('PHE', 'discomfort_high_recall')} | "
            f"{int(combos.loc['PHE', 'discomfort_high_false_negatives']) if 'PHE' in combos.index else 'NA'} |"
        ),
        (
            f"| PHEV | {combo_value('PHEV', 'relaxation_mae')} | "
            f"{combo_value('PHEV', 'discomfort_mae')} | {combo_value('PHEV', 'discomfort_high_recall')} | "
            f"{int(combos.loc['PHEV', 'discomfort_high_false_negatives']) if 'PHEV' in combos.index else 'NA'} |"
        ),
        (
            f"| PHEA | {combo_value('PHEA', 'relaxation_mae')} | "
            f"{combo_value('PHEA', 'discomfort_mae')} | {combo_value('PHEA', 'discomfort_high_recall')} | "
            f"{int(combos.loc['PHEA', 'discomfort_high_false_negatives']) if 'PHEA' in combos.index else 'NA'} |"
        ),
        "",
        "## Relaxation MAE 排名前列",
        "",
        *ranking_header,
        *[_combination_row(row) for _, row in best_relax.iterrows()],
        "",
        "## Discomfort MAE 排名前列",
        "",
        *ranking_header,
        *[_combination_row(row) for _, row in best_discomfort.iterrows()],
        "",
        "## High-discomfort recall 排名前列",
        "",
        *ranking_header,
        *[_combination_row(row) for _, row in best_recall.iterrows()],
        "",
        "## 输出",
        "",
        f"- Metrics: `{(output_dir / 'metrics.csv').as_posix()}`",
        f"- OOF predictions: `{(output_dir / 'oof_predictions.csv').as_posix()}`",
        f"- Head-to-head table: `{(output_dir / 'head_to_head.csv').as_posix()}`",
        f"- Attended features: `{(output_dir / 'attended_condition_features.csv').as_posix()}`",
        "",
        "## 特征与参数",
        "",
        f"- A 特征列数: {sum(str(c).startswith(ATTENDED_PREFIX) for c in attended_features.columns)}.",
        f"- A 特征样本行数: {len(attended_features)}.",
        f"- Rolling heatmap window: {parameters['window_s']} s.",
        f"- Sampling step: {parameters['step_s']} s.",
        f"- Crop source size: {parameters['crop_size']} px; model crop: {parameters['model_size']} px.",
        "- A 的当前空间映射仍是 normalized gaze_hit_y/z 到 frame pixel 的 pilot approximation，",
        "  不是完整 Unity camera projection。",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run(
    *,
    manifest_path: Path,
    base_features_path: Path,
    output_dir: Path,
    report_path: Path,
    bins: int,
    window_s: float,
    step_s: float,
    crop_size: int,
    model_size: int,
    color_quantile: float,
    random_seed: int,
    random_simulations: int,
    force: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    attended_path = output_dir / "attended_condition_features.csv"
    attended_metadata = output_dir / "attended_feature_metadata.json"
    attended = build_attended_features(
        manifest_path=manifest_path,
        base_features_path=base_features_path,
        output_path=attended_path,
        metadata_path=attended_metadata,
        bins=bins,
        window_s=window_s,
        step_s=step_s,
        crop_size=crop_size,
        model_size=model_size,
        color_quantile=color_quantile,
        force=force,
    )

    base = pd.read_csv(base_features_path)
    frame = base.merge(attended, on=["participant_id", "condition"], how="left", validate="one_to_one")
    missing_attended = frame.filter(regex=f"^{ATTENDED_PREFIX}").isna().all(axis=1).sum()
    if missing_attended:
        raise ValueError(f"{missing_attended} rows have no attended-video features")

    result = _evaluate_with_attended(
        frame,
        random_seed=random_seed,
        random_simulations=random_simulations,
    )
    metrics = result["metrics"]
    oof = result["oof_predictions"]
    head_to_head = _head_to_head_rows(metrics)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)
    head_to_head.to_csv(output_dir / "head_to_head.csv", index=False)
    write_report(
        report_path,
        metrics=metrics,
        head_to_head=head_to_head,
        attended_features=attended,
        output_dir=output_dir,
        parameters={
            "bins": bins,
            "window_s": window_s,
            "step_s": step_s,
            "crop_size": crop_size,
            "model_size": model_size,
            "color_quantile": color_quantile,
        },
    )
    summary = {
        "metrics": str(output_dir / "metrics.csv"),
        "oof_predictions": str(output_dir / "oof_predictions.csv"),
        "head_to_head": str(output_dir / "head_to_head.csv"),
        "attended_features": str(attended_path),
        "report": str(report_path),
        "n_rows": int(len(frame)),
        "n_attended_feature_columns": int(sum(str(c).startswith(ATTENDED_PREFIX) for c in frame.columns)),
        "n_combinations": int(metrics.loc[metrics["record_type"].eq("combination")].shape[0]),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark attended video against original modalities.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--base-features", type=Path, default=DEFAULT_BASE_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--bins", type=int, default=72)
    parser.add_argument("--window-s", type=float, default=20.0)
    parser.add_argument("--step-s", type=float, default=1.0)
    parser.add_argument("--crop-size", type=int, default=360)
    parser.add_argument("--model-size", type=int, default=224)
    parser.add_argument("--color-quantile", type=float, default=0.995)
    parser.add_argument("--random-seed", type=int, default=20260703)
    parser.add_argument("--random-simulations", type=int, default=fusion.RANDOM_SIMULATIONS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run(
        manifest_path=args.manifest,
        base_features_path=args.base_features,
        output_dir=args.output_dir,
        report_path=args.report,
        bins=args.bins,
        window_s=args.window_s,
        step_s=args.step_s,
        crop_size=args.crop_size,
        model_size=args.model_size,
        color_quantile=args.color_quantile,
        random_seed=args.random_seed,
        random_simulations=args.random_simulations,
        force=args.force,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

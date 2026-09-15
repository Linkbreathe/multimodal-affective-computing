from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from real_time_ml.data.io import iter_csv, normalize_condition, normalize_participant_id, parse_float, sniff_csv


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "artifacts" / "manifests" / "source_manifest.csv"
DEFAULT_OUTPUT = ROOT / "artifacts" / "reports" / "gaze_heatmap_pilot"
DEFAULT_REPORT = ROOT / "artifacts" / "reports" / "gaze_heatmap_pilot_zh.md"


@dataclass(frozen=True)
class CoordinateBounds:
    z_min: float
    z_max: float
    y_min: float
    y_max: float


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    if values.size == 0 or total <= 0:
        return float("nan")
    return float(np.sum(values * weights) / total)


def _weighted_sd(values: np.ndarray, weights: np.ndarray) -> float:
    mean = _weighted_mean(values, weights)
    total = float(np.sum(weights))
    if values.size == 0 or total <= 0 or not math.isfinite(mean):
        return float("nan")
    return float(np.sqrt(np.sum(weights * (values - mean) ** 2) / total))


def _weighted_fraction(mask: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    if total <= 0:
        return float("nan")
    return float(np.sum(weights[mask]) / total)


def _entropy_from_histogram(histogram: np.ndarray) -> float:
    values = histogram.ravel().astype(float)
    total = float(values.sum())
    if total <= 0:
        return float("nan")
    probs = values[values > 0] / total
    return float(-(probs * np.log2(probs)).sum())


def _read_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing source manifest: {path}")
    frame = pd.read_csv(path)
    required = {"participant_id", "eye_tracking_csv"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"source_manifest.csv missing columns: {sorted(missing)}")
    frame["participant_id"] = frame["participant_id"].map(normalize_participant_id)
    return frame


def _participant_source(manifest: pd.DataFrame, participant_id: str) -> pd.Series:
    participant_id = normalize_participant_id(participant_id)
    matches = manifest.loc[manifest["participant_id"].eq(participant_id)]
    if matches.empty:
        raise ValueError(f"Participant {participant_id} not found in source manifest")
    if len(matches) > 1:
        raise ValueError(f"Participant {participant_id} has multiple manifest rows")
    return matches.iloc[0]


def load_viewing_gaze(eye_tracking_csv: Path, participant_id: str) -> pd.DataFrame:
    if not eye_tracking_csv.exists():
        raise FileNotFoundError(f"Missing eye tracking CSV: {eye_tracking_csv}")
    _, decimal, _ = sniff_csv(eye_tracking_csv)
    rows: list[dict[str, Any]] = []
    for row in iter_csv(eye_tracking_csv):
        if row.get("phase") != "ConditionViewing":
            continue
        if not _truthy(row.get("formal_viewing")):
            continue
        if not _truthy(row.get("gaze_available")) or not _truthy(row.get("gaze_hit")):
            continue
        try:
            condition = normalize_condition(row.get("condition_id", ""))
        except ValueError:
            continue
        time_s = parse_float(row.get("session_elapsed_seconds"), decimal)
        hit_y = parse_float(row.get("gaze_hit_y"), decimal)
        hit_z = parse_float(row.get("gaze_hit_z"), decimal)
        if not (_finite(time_s) and _finite(hit_y) and _finite(hit_z)):
            continue
        rows.append(
            {
                "participant_id": participant_id,
                "condition": condition,
                "session_elapsed_seconds": float(time_s),
                "gaze_hit_y": float(hit_y),
                "gaze_hit_z": float(hit_z),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No formal ConditionViewing gaze-hit rows in {eye_tracking_csv}")
    frame = frame.sort_values(["condition", "session_elapsed_seconds"]).reset_index(drop=True)
    return frame


def add_time_weights(frame: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for condition, group in frame.groupby("condition", sort=True):
        group = group.sort_values("session_elapsed_seconds").copy()
        times = group["session_elapsed_seconds"].to_numpy(dtype=float)
        deltas = np.diff(times)
        valid = deltas[(deltas > 0) & (deltas <= 0.5)]
        median_delta = float(np.median(valid)) if valid.size else 0.1
        weights = np.full(len(group), median_delta, dtype=float)
        if len(group) > 1:
            clipped = np.clip(deltas, 0.0, 0.5)
            clipped[clipped <= 0] = median_delta
            weights[:-1] = clipped
        group["time_weight_s"] = weights
        group["condition_order"] = int(condition[1:])
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["condition_order", "session_elapsed_seconds"]
    )


def coordinate_bounds(frame: pd.DataFrame) -> CoordinateBounds:
    z_min = float(frame["gaze_hit_z"].min())
    z_max = float(frame["gaze_hit_z"].max())
    y_min = float(frame["gaze_hit_y"].min())
    y_max = float(frame["gaze_hit_y"].max())
    if z_max <= z_min or y_max <= y_min:
        raise ValueError("Gaze hit coordinates do not span a 2D plane")
    return CoordinateBounds(z_min=z_min, z_max=z_max, y_min=y_min, y_max=y_max)


def add_normalized_coordinates(frame: pd.DataFrame, bounds: CoordinateBounds) -> pd.DataFrame:
    frame = frame.copy()
    frame["gaze_x_norm"] = (frame["gaze_hit_z"] - bounds.z_min) / (bounds.z_max - bounds.z_min)
    frame["gaze_y_norm"] = (frame["gaze_hit_y"] - bounds.y_min) / (bounds.y_max - bounds.y_min)
    frame["gaze_x_norm"] = frame["gaze_x_norm"].clip(0.0, 1.0)
    frame["gaze_y_norm"] = frame["gaze_y_norm"].clip(0.0, 1.0)
    return frame


def _histogram(
    frame: pd.DataFrame,
    bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.histogram2d(
        frame["gaze_y_norm"].to_numpy(dtype=float),
        frame["gaze_x_norm"].to_numpy(dtype=float),
        bins=bins,
        range=[[0.0, 1.0], [0.0, 1.0]],
        weights=frame["time_weight_s"].to_numpy(dtype=float),
    )


def summarize(frame: pd.DataFrame, bins: int) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    groups: list[tuple[str, pd.DataFrame]] = [("overall", frame)]
    groups.extend((condition, group) for condition, group in frame.groupby("condition", sort=True))
    for condition, group in groups:
        weights = group["time_weight_s"].to_numpy(dtype=float)
        x = group["gaze_x_norm"].to_numpy(dtype=float)
        y = group["gaze_y_norm"].to_numpy(dtype=float)
        hist, _, _ = _histogram(group, bins)
        records.append(
            {
                "participant_id": str(group["participant_id"].iloc[0]),
                "condition": condition,
                "n_samples": int(len(group)),
                "duration_s": float(np.sum(weights)),
                "gaze_hit_y_min": float(group["gaze_hit_y"].min()),
                "gaze_hit_y_median": float(group["gaze_hit_y"].median()),
                "gaze_hit_y_max": float(group["gaze_hit_y"].max()),
                "gaze_hit_z_min": float(group["gaze_hit_z"].min()),
                "gaze_hit_z_median": float(group["gaze_hit_z"].median()),
                "gaze_hit_z_max": float(group["gaze_hit_z"].max()),
                "x_norm_weighted_mean": _weighted_mean(x, weights),
                "y_norm_weighted_mean": _weighted_mean(y, weights),
                "x_norm_weighted_sd": _weighted_sd(x, weights),
                "y_norm_weighted_sd": _weighted_sd(y, weights),
                "left_third_fraction": _weighted_fraction(x < 1 / 3, weights),
                "middle_x_third_fraction": _weighted_fraction((x >= 1 / 3) & (x < 2 / 3), weights),
                "right_third_fraction": _weighted_fraction(x >= 2 / 3, weights),
                "lower_third_fraction": _weighted_fraction(y < 1 / 3, weights),
                "middle_y_third_fraction": _weighted_fraction((y >= 1 / 3) & (y < 2 / 3), weights),
                "upper_third_fraction": _weighted_fraction(y >= 2 / 3, weights),
                "center_ninth_fraction": _weighted_fraction(
                    (x >= 1 / 3) & (x < 2 / 3) & (y >= 1 / 3) & (y < 2 / 3),
                    weights,
                ),
                "heatmap_entropy_bits": _entropy_from_histogram(hist),
            }
        )
    summary = pd.DataFrame(records)
    summary["condition_sort"] = summary["condition"].map(
        lambda value: 0 if value == "overall" else int(str(value)[1:])
    )
    return summary.sort_values("condition_sort").drop(columns=["condition_sort"])


def plot_overall(frame: pd.DataFrame, output: Path, bins: int, participant_id: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist, _, _ = _histogram(frame, bins)
    fig, ax = plt.subplots(figsize=(7.5, 6.2), dpi=160)
    image = ax.imshow(
        hist,
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap="magma",
        aspect="equal",
    )
    ax.set_title(f"{participant_id} gaze heatmap, formal viewing")
    ax.set_xlabel("normalized gaze_hit_z")
    ax.set_ylabel("normalized gaze_hit_y")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(color="white", alpha=0.18, linewidth=0.7)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("weighted viewing time (s/bin)")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_conditions(frame: pd.DataFrame, output: Path, bins: int, participant_id: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    histograms = {}
    vmax = 0.0
    for condition, group in frame.groupby("condition", sort=True):
        hist, _, _ = _histogram(group, bins)
        histograms[condition] = hist
        vmax = max(vmax, float(np.quantile(hist[hist > 0], 0.98)) if np.any(hist > 0) else 0.0)
    if vmax <= 0:
        vmax = None

    fig, axes = plt.subplots(3, 3, figsize=(10.5, 10), dpi=160, sharex=True, sharey=True)
    for ax, condition in zip(axes.ravel(), [f"C{i}" for i in range(1, 10)], strict=True):
        hist = histograms.get(condition, np.zeros((bins, bins)))
        group = frame.loc[frame["condition"].eq(condition)]
        image = ax.imshow(
            hist,
            origin="lower",
            extent=[0, 1, 0, 1],
            cmap="magma",
            aspect="equal",
            vmax=vmax,
        )
        ax.scatter(
            [_weighted_mean(group["gaze_x_norm"].to_numpy(float), group["time_weight_s"].to_numpy(float))],
            [_weighted_mean(group["gaze_y_norm"].to_numpy(float), group["time_weight_s"].to_numpy(float))],
            s=18,
            c="cyan",
            edgecolors="black",
            linewidths=0.4,
            label="weighted center",
        )
        ax.set_title(f"{condition}  n={len(group)}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(color="white", alpha=0.16, linewidth=0.5)
    for ax in axes[-1, :]:
        ax.set_xlabel("normalized gaze_hit_z")
    for ax in axes[:, 0]:
        ax.set_ylabel("normalized gaze_hit_y")
    fig.suptitle(f"{participant_id} gaze heatmaps by condition", y=0.995)
    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.028, pad=0.02)
    cbar.set_label("weighted viewing time (s/bin)")
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def write_report(
    report_path: Path,
    participant_id: str,
    eye_tracking_csv: Path,
    output_dir: Path,
    summary: pd.DataFrame,
    bounds: CoordinateBounds,
    bins: int,
) -> None:
    overall = summary.loc[summary["condition"].eq("overall")].iloc[0]
    condition = summary.loc[summary["condition"].ne("overall")].copy()
    most_center = condition.sort_values("center_ninth_fraction", ascending=False).iloc[0]
    most_upper = condition.sort_values("upper_third_fraction", ascending=False).iloc[0]
    most_lower = condition.sort_values("lower_third_fraction", ascending=False).iloc[0]
    lines = [
        f"# {participant_id} gaze heatmap pilot",
        "",
        "## 方法",
        "",
        f"- 输入：`{eye_tracking_csv}`。",
        "- 只保留 `phase == ConditionViewing` 且 `formal_viewing == true` 的正式观看段。",
        "- 不使用 `gaze_on_painting`，因为它在当前数据中不提供空间差异。",
        "- 空间坐标使用 `gaze_hit_z` 作为横轴代理、`gaze_hit_y` 作为纵轴代理，并按本参与者有效范围归一化到 `[0, 1]`。",
        "- 每个 gaze sample 用相邻 `session_elapsed_seconds` 间隔加权，因此 heatmap 表示近似观看时长。",
        "",
        "## 产物",
        "",
        f"- Overall heatmap: `{(output_dir / f'{participant_id}_overall_gaze_heatmap.png').as_posix()}`",
        f"- Condition heatmaps: `{(output_dir / f'{participant_id}_condition_gaze_heatmaps.png').as_posix()}`",
        f"- Summary CSV: `{(output_dir / f'{participant_id}_gaze_heatmap_summary.csv').as_posix()}`",
        f"- Filtered points: `{(output_dir / f'{participant_id}_filtered_gaze_points.csv').as_posix()}`",
        "",
        "## QC",
        "",
        f"- 有效正式观看 gaze samples：{int(overall['n_samples'])}。",
        f"- 加权观看时长：{_fmt(overall['duration_s'], 1)} s。",
        f"- `gaze_hit_z` 范围：{_fmt(bounds.z_min)} 到 {_fmt(bounds.z_max)}。",
        f"- `gaze_hit_y` 范围：{_fmt(bounds.y_min)} 到 {_fmt(bounds.y_max)}。",
        f"- heatmap bins：{bins} x {bins}。",
        "",
        "## 粗略读数",
        "",
        f"- 整体加权中心：x={_fmt(overall['x_norm_weighted_mean'])}, y={_fmt(overall['y_norm_weighted_mean'])}。",
        f"- 中央九宫格占比最高的 condition：{most_center['condition']}（{_fmt(most_center['center_ninth_fraction'] * 100, 1)}%）。",
        f"- 上三分区占比最高的 condition：{most_upper['condition']}（{_fmt(most_upper['upper_third_fraction'] * 100, 1)}%）。",
        f"- 下三分区占比最高的 condition：{most_lower['condition']}（{_fmt(most_lower['lower_third_fraction'] * 100, 1)}%）。",
        "",
        "## 图",
        "",
        f"![{participant_id} overall gaze heatmap](gaze_heatmap_pilot/{participant_id}_overall_gaze_heatmap.png)",
        "",
        f"![{participant_id} condition gaze heatmaps](gaze_heatmap_pilot/{participant_id}_condition_gaze_heatmaps.png)",
        "",
        "## 边界",
        "",
        "- 这版不是 video 像素级 overlay；它是画作命中平面上的 gaze heatmap。",
        "- 旧 eye CSV 没导出逐行 `gaze_source`，因此不能逐行区分真实 eye gaze 与 head-forward fallback。",
        "- 若要解释成画面中的具体物体，需要再把画作平面坐标映射到材质/视频像素坐标。",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run(
    participant_id: str,
    manifest_path: Path,
    output_dir: Path,
    report_path: Path,
    bins: int,
) -> dict[str, Any]:
    participant_id = normalize_participant_id(participant_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest(manifest_path)
    source = _participant_source(manifest, participant_id)
    eye_tracking_csv = Path(str(source["eye_tracking_csv"]))
    frame = load_viewing_gaze(eye_tracking_csv, participant_id)
    frame = add_time_weights(frame)
    bounds = coordinate_bounds(frame)
    frame = add_normalized_coordinates(frame, bounds)
    summary = summarize(frame, bins)

    points_path = output_dir / f"{participant_id}_filtered_gaze_points.csv"
    summary_path = output_dir / f"{participant_id}_gaze_heatmap_summary.csv"
    overall_png = output_dir / f"{participant_id}_overall_gaze_heatmap.png"
    conditions_png = output_dir / f"{participant_id}_condition_gaze_heatmaps.png"
    qc_path = output_dir / f"{participant_id}_gaze_heatmap_qc.json"

    frame.to_csv(points_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_overall(frame, overall_png, bins, participant_id)
    plot_conditions(frame, conditions_png, bins, participant_id)

    qc = {
        "participant_id": participant_id,
        "eye_tracking_csv": str(eye_tracking_csv),
        "n_filtered_samples": int(len(frame)),
        "conditions": sorted(frame["condition"].unique().tolist()),
        "bins": int(bins),
        "coordinate_bounds": bounds.__dict__,
        "outputs": {
            "points_csv": str(points_path),
            "summary_csv": str(summary_path),
            "overall_png": str(overall_png),
            "condition_png": str(conditions_png),
            "report": str(report_path),
        },
        "excluded_fields": ["gaze_on_painting"],
    }
    qc_path.write_text(json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(report_path, participant_id, eye_tracking_csv, output_dir, summary, bounds, bins)
    return qc


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate participant gaze heatmap pilot outputs.")
    parser.add_argument("--participant", default="P007")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--bins", type=int, default=48)
    args = parser.parse_args()
    qc = run(args.participant, args.manifest, args.output_dir, args.report, args.bins)
    print(json.dumps(qc, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

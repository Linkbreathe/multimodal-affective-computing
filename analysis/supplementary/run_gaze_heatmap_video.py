from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from analysis.supplementary.run_gaze_heatmap_pilot import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT,
    _participant_source,
    _read_manifest,
    add_normalized_coordinates,
    add_time_weights,
    coordinate_bounds,
    load_viewing_gaze,
)
from mac.data.io import normalize_participant_id  # noqa: E402


DEFAULT_REPORT = ROOT / "artifacts" / "reports" / "gaze_heatmap_video_zh.md"


def _condition_number(condition: str) -> int:
    return int(str(condition).replace("C", ""))


def _build_formal_timeline(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    pieces: list[pd.DataFrame] = []
    segments: list[dict[str, Any]] = []
    elapsed = 0.0
    for condition, group in sorted(frame.groupby("condition"), key=lambda item: _condition_number(item[0])):
        group = group.sort_values("session_elapsed_seconds").copy()
        start = float(group["session_elapsed_seconds"].iloc[0])
        local = group["session_elapsed_seconds"].to_numpy(float) - start
        group["condition_elapsed_s"] = local
        group["formal_elapsed_s"] = elapsed + local
        duration = float(np.sum(group["time_weight_s"].to_numpy(float)))
        segments.append(
            {
                "condition": condition,
                "formal_start_s": elapsed,
                "formal_end_s": elapsed + duration,
                "duration_s": duration,
                "n_samples": int(len(group)),
            }
        )
        pieces.append(group)
        elapsed += duration
    return pd.concat(pieces, ignore_index=True), segments


def _active_condition(segments: list[dict[str, Any]], time_s: float) -> dict[str, Any]:
    for segment in segments:
        if segment["formal_start_s"] <= time_s <= segment["formal_end_s"]:
            return segment
    return segments[-1]


def _histogram(frame: pd.DataFrame, bins: int) -> np.ndarray:
    hist, _, _ = np.histogram2d(
        frame["gaze_y_norm"].to_numpy(dtype=float),
        frame["gaze_x_norm"].to_numpy(dtype=float),
        bins=bins,
        range=[[0.0, 1.0], [0.0, 1.0]],
        weights=frame["time_weight_s"].to_numpy(dtype=float),
    )
    return hist


def _dynamic_histograms(
    frame: pd.DataFrame,
    bins: int,
    window_s: float,
    step_s: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    total_duration = float(frame["formal_elapsed_s"].max())
    times = np.arange(0.0, total_duration + 1e-9, step_s)
    histograms: list[np.ndarray] = []
    for time_s in times:
        start = max(0.0, time_s - window_s)
        window = frame.loc[
            frame["formal_elapsed_s"].between(start, time_s, inclusive="both")
        ]
        histograms.append(_histogram(window, bins) if not window.empty else np.zeros((bins, bins)))
    return times, histograms


def _colorize(histogram: np.ndarray, vmax: float, size: int) -> np.ndarray:
    import cv2

    normalized = np.clip(histogram / max(vmax, 1e-9), 0.0, 1.0)
    image = (np.flipud(normalized) * 255).astype(np.uint8)
    cmap = getattr(cv2, "COLORMAP_MAGMA", cv2.COLORMAP_INFERNO)
    colored = cv2.applyColorMap(image, cmap)
    return cv2.resize(colored, (size, size), interpolation=cv2.INTER_LINEAR)


def _point_to_pixel(x_norm: float, y_norm: float, left: int, top: int, size: int) -> tuple[int, int]:
    x = int(round(left + np.clip(x_norm, 0.0, 1.0) * size))
    y = int(round(top + (1.0 - np.clip(y_norm, 0.0, 1.0)) * size))
    return x, y


def _put_text(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    scale: float = 0.65,
    color: tuple[int, int, int] = (230, 230, 230),
    thickness: int = 1,
) -> None:
    import cv2

    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_frame(
    heatmap: np.ndarray,
    gaze_frame: pd.DataFrame,
    current: pd.Series | None,
    participant_id: str,
    time_s: float,
    total_s: float,
    active_segment: dict[str, Any],
    fps: float,
    step_s: float,
    window_s: float,
    vmax: float,
) -> np.ndarray:
    import cv2

    width, height = 1280, 720
    canvas = np.full((height, width, 3), (12, 14, 18), dtype=np.uint8)
    left, top, size = 55, 65, 590
    canvas[top : top + size, left : left + size] = heatmap

    cv2.rectangle(canvas, (left, top), (left + size, top + size), (185, 185, 185), 1)
    for fraction in (1 / 3, 2 / 3):
        x = int(left + size * fraction)
        y = int(top + size * fraction)
        cv2.line(canvas, (x, top), (x, top + size), (95, 95, 95), 1)
        cv2.line(canvas, (left, y), (left + size, y), (95, 95, 95), 1)

    if not gaze_frame.empty:
        recent = gaze_frame.tail(90)
        for _, row in recent.iterrows():
            px, py = _point_to_pixel(row["gaze_x_norm"], row["gaze_y_norm"], left, top, size)
            cv2.circle(canvas, (px, py), 2, (210, 210, 210), -1)

    if current is not None:
        px, py = _point_to_pixel(current["gaze_x_norm"], current["gaze_y_norm"], left, top, size)
        cv2.circle(canvas, (px, py), 8, (255, 255, 255), 2)
        cv2.circle(canvas, (px, py), 4, (255, 255, 0), -1)

    _put_text(canvas, f"{participant_id} gaze heatmap tracking", 55, 35, 0.85, (245, 245, 245), 2)
    _put_text(canvas, "x: normalized gaze_hit_z", 205, 680, 0.55, (205, 205, 205))
    _put_text(canvas, "y: normalized gaze_hit_y", 20, 55, 0.55, (205, 205, 205))

    panel_x = 700
    condition = active_segment["condition"]
    condition_elapsed = max(0.0, time_s - float(active_segment["formal_start_s"]))
    _put_text(canvas, f"Condition: {condition}", panel_x, 105, 0.85, (255, 255, 255), 2)
    _put_text(canvas, f"Formal time: {time_s:6.1f} / {total_s:6.1f} s", panel_x, 150)
    _put_text(canvas, f"Condition time: {condition_elapsed:5.1f} s", panel_x, 185)
    _put_text(canvas, f"Window: last {window_s:.0f} s", panel_x, 220)
    _put_text(canvas, f"Playback: {step_s:.1f} data-s/frame @ {fps:.1f} fps", panel_x, 255)
    _put_text(canvas, f"Color scale fixed: vmax={vmax:.2f} s/bin", panel_x, 290)
    _put_text(canvas, "gaze_on_painting excluded", panel_x, 325, color=(175, 210, 255))
    _put_text(canvas, "white/cyan marker = latest gaze point", panel_x, 360, color=(175, 255, 255))

    # Progress bar.
    bar_x, bar_y, bar_w, bar_h = panel_x, 410, 480, 18
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (70, 70, 70), 1)
    progress = min(1.0, max(0.0, time_s / max(total_s, 1e-9)))
    cv2.rectangle(
        canvas,
        (bar_x + 1, bar_y + 1),
        (bar_x + int((bar_w - 2) * progress), bar_y + bar_h - 1),
        (255, 180, 70),
        -1,
    )

    # Color bar.
    cmap_values = np.linspace(1.0, 0.0, 180).reshape(-1, 1)
    colorbar = _colorize(cmap_values, 1.0, 24)
    colorbar = cv2.resize(colorbar, (24, 180), interpolation=cv2.INTER_LINEAR)
    cb_x, cb_y = panel_x, 470
    canvas[cb_y : cb_y + 180, cb_x : cb_x + 24] = colorbar
    cv2.rectangle(canvas, (cb_x, cb_y), (cb_x + 24, cb_y + 180), (185, 185, 185), 1)
    _put_text(canvas, f"{vmax:.1f}", cb_x + 35, cb_y + 10, 0.5)
    _put_text(canvas, "0", cb_x + 35, cb_y + 180, 0.5)
    _put_text(canvas, "weighted seconds/bin", cb_x + 70, cb_y + 95, 0.55)
    return canvas


def write_video(
    frame: pd.DataFrame,
    segments: list[dict[str, Any]],
    participant_id: str,
    output_path: Path,
    bins: int,
    window_s: float,
    step_s: float,
    fps: float,
    color_quantile: float,
) -> dict[str, Any]:
    import cv2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    times, histograms = _dynamic_histograms(frame, bins, window_s, step_s)
    nonzero = np.concatenate([hist[hist > 0] for hist in histograms if np.any(hist > 0)])
    vmax = float(np.quantile(nonzero, color_quantile)) if nonzero.size else 1.0
    total_s = float(frame["formal_elapsed_s"].max())
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (1280, 720))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    try:
        formal_times = frame["formal_elapsed_s"].to_numpy(float)
        for time_s, histogram in zip(times, histograms, strict=True):
            heatmap = _colorize(histogram, vmax, 590)
            start = max(0.0, float(time_s) - window_s)
            gaze_window = frame.loc[frame["formal_elapsed_s"].between(start, time_s, inclusive="both")]
            current = None
            previous = np.flatnonzero(formal_times <= time_s)
            if previous.size:
                current = frame.iloc[int(previous[-1])]
            canvas = _draw_frame(
                heatmap=heatmap,
                gaze_frame=gaze_window,
                current=current,
                participant_id=participant_id,
                time_s=float(time_s),
                total_s=total_s,
                active_segment=_active_condition(segments, float(time_s)),
                fps=fps,
                step_s=step_s,
                window_s=window_s,
                vmax=vmax,
            )
            writer.write(canvas)
    finally:
        writer.release()

    return {
        "participant_id": participant_id,
        "output_video": str(output_path),
        "n_video_frames": int(len(times)),
        "output_fps": float(fps),
        "video_duration_s": float(len(times) / fps),
        "formal_duration_s": total_s,
        "step_s": float(step_s),
        "window_s": float(window_s),
        "bins": int(bins),
        "color_quantile": float(color_quantile),
        "vmax_s_per_bin": vmax,
        "segments": segments,
    }


def write_report(report_path: Path, metadata: dict[str, Any]) -> None:
    participant_id = metadata["participant_id"]
    video_path = Path(metadata["output_video"])
    lines = [
        f"# {participant_id} gaze heatmap tracking video",
        "",
        "## Output",
        "",
        f"- Video: `{video_path.as_posix()}`",
        f"- Metadata: `{video_path.with_suffix('.json').as_posix()}`",
        "",
        "## Method",
        "",
        "- Input is the same formal-viewing gaze-hit stream used for the static heatmaps.",
        "- `gaze_on_painting` is excluded.",
        f"- The video uses a rolling {metadata['window_s']:.0f}s heatmap window.",
        f"- The formal viewing timeline is compressed from {metadata['formal_duration_s']:.1f}s to {metadata['video_duration_s']:.1f}s.",
        "- Coordinates are normalized painting-plane coordinates: x=`gaze_hit_z`, y=`gaze_hit_y`.",
        "- The color scale is fixed within this video, so brightness changes across time are meaningful for this participant.",
        "",
        "## Segment Durations",
        "",
        "| Condition | Formal start s | Formal end s | Duration s | Samples |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for segment in metadata["segments"]:
        lines.append(
            f"| {segment['condition']} | {segment['formal_start_s']:.1f} | "
            f"{segment['formal_end_s']:.1f} | {segment['duration_s']:.1f} | {segment['n_samples']} |"
        )
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run(
    participant_id: str,
    manifest_path: Path,
    output_dir: Path,
    output_video: Path | None,
    report_path: Path,
    bins: int,
    window_s: float,
    step_s: float,
    fps: float,
    color_quantile: float,
) -> dict[str, Any]:
    participant_id = normalize_participant_id(participant_id)
    manifest = _read_manifest(manifest_path)
    source = _participant_source(manifest, participant_id)
    eye_tracking_csv = Path(str(source["eye_tracking_csv"]))
    frame = load_viewing_gaze(eye_tracking_csv, participant_id)
    frame = add_time_weights(frame)
    bounds = coordinate_bounds(frame)
    frame = add_normalized_coordinates(frame, bounds)
    frame, segments = _build_formal_timeline(frame)
    target = output_video or output_dir / f"{participant_id}_gaze_heatmap_tracking.mp4"
    metadata = write_video(
        frame=frame,
        segments=segments,
        participant_id=participant_id,
        output_path=target,
        bins=bins,
        window_s=window_s,
        step_s=step_s,
        fps=fps,
        color_quantile=color_quantile,
    )
    metadata["eye_tracking_csv"] = str(eye_tracking_csv)
    metadata["coordinate_bounds"] = bounds.__dict__
    metadata["excluded_fields"] = ["gaze_on_painting"]
    metadata_path = target.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(report_path, metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a rolling gaze heatmap tracking video.")
    parser.add_argument("--participant", default="P015")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-video", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--bins", type=int, default=72)
    parser.add_argument("--window-s", type=float, default=20.0)
    parser.add_argument("--step-s", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--color-quantile", type=float, default=0.995)
    args = parser.parse_args()
    metadata = run(
        participant_id=args.participant,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        output_video=args.output_video,
        report_path=args.report,
        bins=args.bins,
        window_s=args.window_s,
        step_s=args.step_s,
        fps=args.fps,
        color_quantile=args.color_quantile,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

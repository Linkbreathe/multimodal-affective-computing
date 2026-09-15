from __future__ import annotations

import argparse
import bisect
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

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
from analysis.supplementary.run_gaze_heatmap_video import _build_formal_timeline  # noqa: E402
from real_time_ml.data.io import iter_csv, normalize_participant_id, parse_float, resolve_video_path, sniff_csv  # noqa: E402


DEFAULT_REPORT = ROOT / "artifacts" / "reports" / "attended_video_preview_zh.md"


def _read_video_frames(video_frames_csv: Path, session_dir: Path) -> pd.DataFrame:
    if not video_frames_csv.exists():
        raise FileNotFoundError(f"Missing video frame CSV: {video_frames_csv}")
    _, decimal, _ = sniff_csv(video_frames_csv)
    rows: list[dict[str, Any]] = []
    for row in iter_csv(video_frames_csv):
        time_s = parse_float(row.get("session_elapsed_seconds"), decimal)
        frame_path = resolve_video_path(session_dir, row)
        if time_s is None or frame_path is None or not frame_path.exists():
            continue
        rows.append(
            {
                "session_elapsed_seconds": float(time_s),
                "frame_index": int(float(row.get("frame_index") or 0)),
                "path": str(frame_path),
                "width": int(float(row.get("width") or 0)),
                "height": int(float(row.get("height") or 0)),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No readable video frames found in {video_frames_csv}")
    return frame.sort_values("session_elapsed_seconds").reset_index(drop=True)


def _nearest_video_frame(video: pd.DataFrame, time_s: float) -> pd.Series:
    values = video["session_elapsed_seconds"].to_numpy(float)
    index = bisect.bisect_left(values, time_s)
    candidates = []
    if index < len(values):
        candidates.append(index)
    if index > 0:
        candidates.append(index - 1)
    best = min(candidates, key=lambda candidate: abs(values[candidate] - time_s))
    return video.iloc[int(best)]


def _current_gaze(gaze: pd.DataFrame, formal_time_s: float) -> pd.Series:
    values = gaze["formal_elapsed_s"].to_numpy(float)
    index = bisect.bisect_right(values, formal_time_s) - 1
    index = max(0, min(index, len(gaze) - 1))
    return gaze.iloc[int(index)]


def _rolling_gaze(gaze: pd.DataFrame, formal_time_s: float, window_s: float) -> pd.DataFrame:
    start = max(0.0, formal_time_s - window_s)
    return gaze.loc[gaze["formal_elapsed_s"].between(start, formal_time_s, inclusive="both")]


def _heatmap_mask(rolling: pd.DataFrame, width: int, height: int, bins: int, vmax: float) -> np.ndarray:
    if rolling.empty:
        return np.zeros((height, width), dtype=np.float32)
    hist, _, _ = np.histogram2d(
        rolling["gaze_y_norm"].to_numpy(float),
        rolling["gaze_x_norm"].to_numpy(float),
        bins=bins,
        range=[[0.0, 1.0], [0.0, 1.0]],
        weights=rolling["time_weight_s"].to_numpy(float),
    )
    normalized = np.clip(hist / max(vmax, 1e-9), 0.0, 1.0)
    mask = cv2.resize(np.flipud(normalized).astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(2.0, width / 180.0), sigmaY=max(2.0, height / 180.0))
    return np.clip(mask, 0.0, 1.0)


def _estimate_vmax(gaze: pd.DataFrame, bins: int, window_s: float, step_s: float, quantile: float) -> float:
    values = []
    max_time = float(gaze["formal_elapsed_s"].max())
    for time_s in np.arange(0.0, max_time + 1e-9, step_s):
        rolling = _rolling_gaze(gaze, float(time_s), window_s)
        if rolling.empty:
            continue
        hist, _, _ = np.histogram2d(
            rolling["gaze_y_norm"].to_numpy(float),
            rolling["gaze_x_norm"].to_numpy(float),
            bins=bins,
            range=[[0.0, 1.0], [0.0, 1.0]],
            weights=rolling["time_weight_s"].to_numpy(float),
        )
        if np.any(hist > 0):
            values.append(hist[hist > 0])
    if not values:
        return 1.0
    return float(np.quantile(np.concatenate(values), quantile))


def _gaze_pixel(row: pd.Series, width: int, height: int) -> tuple[int, int]:
    x = int(round(np.clip(float(row["gaze_x_norm"]), 0.0, 1.0) * (width - 1)))
    y = int(round((1.0 - np.clip(float(row["gaze_y_norm"]), 0.0, 1.0)) * (height - 1)))
    return x, y


def _crop_with_padding(image: np.ndarray, center: tuple[int, int], crop_size: int) -> np.ndarray:
    height, width = image.shape[:2]
    half = crop_size // 2
    x, y = center
    left, right = x - half, x + half
    top, bottom = y - half, y + half
    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - width)
    pad_bottom = max(0, bottom - height)
    padded = cv2.copyMakeBorder(
        image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_REFLECT_101,
    )
    left += pad_left
    right += pad_left
    top += pad_top
    bottom += pad_top
    return padded[top:bottom, left:right]


def _soft_attend(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    weight = 0.18 + 0.82 * np.clip(mask[..., None], 0.0, 1.0)
    return np.clip(frame.astype(np.float32) * weight, 0, 255).astype(np.uint8)


def _overlay_heatmap(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    heat = cv2.applyColorMap((np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    alpha = np.clip(mask[..., None] * 0.78, 0.0, 0.78)
    return np.clip(frame.astype(np.float32) * (1.0 - alpha) + heat.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


def _draw_text(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    scale: float = 0.62,
    color: tuple[int, int, int] = (235, 235, 235),
    thickness: int = 1,
) -> None:
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _panel_label(image: np.ndarray, text: str) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 34), (0, 0, 0), -1)
    _draw_text(image, text, 12, 24, 0.66, (255, 255, 255), 1)


def _make_model_input(frame: np.ndarray, mask: np.ndarray, current: pd.Series, crop_size: int, model_size: int) -> np.ndarray:
    attended = _soft_attend(frame, mask)
    center = _gaze_pixel(current, frame.shape[1], frame.shape[0])
    crop = _crop_with_padding(attended, center, crop_size)
    return cv2.resize(crop, (model_size, model_size), interpolation=cv2.INTER_AREA)


def _make_preview_frame(
    frame: np.ndarray,
    mask: np.ndarray,
    current: pd.Series,
    participant_id: str,
    formal_time_s: float,
    total_s: float,
    condition: str,
    condition_elapsed_s: float,
    crop_size: int,
    model_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = frame.shape[:2]
    overlay = _overlay_heatmap(frame, mask)
    attended = _soft_attend(frame, mask)
    current_pixel = _gaze_pixel(current, width, height)
    cv2.circle(overlay, current_pixel, 12, (255, 255, 255), 2)
    cv2.circle(overlay, current_pixel, 5, (0, 255, 255), -1)

    model_input = _make_model_input(frame, mask, current, crop_size, model_size)
    preview_width, preview_height = 1600, 900
    canvas = np.full((preview_height, preview_width, 3), (14, 15, 18), dtype=np.uint8)

    left = cv2.resize(overlay, (960, 540), interpolation=cv2.INTER_AREA)
    _panel_label(left, "original egocentric frame + rolling gaze heatmap")
    canvas[70:610, 40:1000] = left

    attended_panel = cv2.resize(attended, (480, 270), interpolation=cv2.INTER_AREA)
    _panel_label(attended_panel, "attention-weighted full frame")
    canvas[70:340, 1060:1540] = attended_panel

    crop_panel = cv2.resize(model_input, (360, 360), interpolation=cv2.INTER_NEAREST)
    _panel_label(crop_panel, f"model input crop ({model_size}x{model_size})")
    canvas[430:790, 1120:1480] = crop_panel

    _draw_text(canvas, f"{participant_id} attended egocentric video preview", 40, 38, 0.9, (255, 255, 255), 2)
    _draw_text(canvas, f"Condition {condition} | formal {formal_time_s:.1f}/{total_s:.1f}s | condition {condition_elapsed_s:.1f}s", 40, 650, 0.68)
    _draw_text(canvas, "Mapping: gaze_hit_y/z plane normalized to frame pixels (pilot approximation)", 40, 685, 0.58, (210, 210, 210))
    _draw_text(canvas, "Excluded: gaze_on_painting (constant); included: rolling gaze heatmap + egocentric frame", 40, 715, 0.58, (210, 230, 255))
    _draw_text(canvas, "What replaces global video: the attended crop/weighted frame, not the full unattended stimulus", 40, 745, 0.58, (210, 255, 210))
    return canvas, model_input


def _condition_at_time(segments: list[dict[str, Any]], time_s: float) -> tuple[str, float]:
    for segment in segments:
        if float(segment["formal_start_s"]) <= time_s <= float(segment["formal_end_s"]):
            return str(segment["condition"]), time_s - float(segment["formal_start_s"])
    segment = segments[-1]
    return str(segment["condition"]), time_s - float(segment["formal_start_s"])


def write_attended_video_preview(
    gaze: pd.DataFrame,
    video: pd.DataFrame,
    segments: list[dict[str, Any]],
    participant_id: str,
    output_preview: Path,
    output_model: Path,
    bins: int,
    window_s: float,
    step_s: float,
    fps: float,
    crop_size: int,
    model_size: int,
    color_quantile: float,
) -> dict[str, Any]:
    output_preview.parent.mkdir(parents=True, exist_ok=True)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    total_s = float(gaze["formal_elapsed_s"].max())
    times = np.arange(0.0, total_s + 1e-9, step_s)
    vmax = _estimate_vmax(gaze, bins=bins, window_s=window_s, step_s=step_s, quantile=color_quantile)

    preview_writer = cv2.VideoWriter(
        str(output_preview),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (1600, 900),
    )
    model_writer = cv2.VideoWriter(
        str(output_model),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (model_size, model_size),
    )
    if not preview_writer.isOpened():
        raise RuntimeError(f"Could not open preview writer: {output_preview}")
    if not model_writer.isOpened():
        raise RuntimeError(f"Could not open model writer: {output_model}")

    try:
        for time_s in times:
            current = _current_gaze(gaze, float(time_s))
            video_row = _nearest_video_frame(video, float(current["session_elapsed_seconds"]))
            frame = cv2.imread(str(video_row["path"]))
            if frame is None:
                continue
            rolling = _rolling_gaze(gaze, float(time_s), window_s)
            mask = _heatmap_mask(rolling, frame.shape[1], frame.shape[0], bins=bins, vmax=vmax)
            condition, condition_elapsed = _condition_at_time(segments, float(time_s))
            preview_frame, model_input = _make_preview_frame(
                frame=frame,
                mask=mask,
                current=current,
                participant_id=participant_id,
                formal_time_s=float(time_s),
                total_s=total_s,
                condition=condition,
                condition_elapsed_s=condition_elapsed,
                crop_size=crop_size,
                model_size=model_size,
            )
            preview_writer.write(preview_frame)
            model_writer.write(model_input)
    finally:
        preview_writer.release()
        model_writer.release()

    return {
        "participant_id": participant_id,
        "preview_video": str(output_preview),
        "model_input_video": str(output_model),
        "n_video_frames": int(len(times)),
        "output_fps": float(fps),
        "video_duration_s": float(len(times) / fps),
        "formal_duration_s": total_s,
        "step_s": float(step_s),
        "window_s": float(window_s),
        "bins": int(bins),
        "crop_size_px": int(crop_size),
        "model_size_px": int(model_size),
        "color_quantile": float(color_quantile),
        "vmax_s_per_bin": float(vmax),
        "mapping": "pilot_plane_normalized_gaze_hit_yz_to_frame_pixels",
        "excluded_fields": ["gaze_on_painting"],
        "segments": segments,
    }


def write_report(report_path: Path, metadata: dict[str, Any]) -> None:
    lines = [
        f"# {metadata['participant_id']} attended egocentric video preview",
        "",
        "## Outputs",
        "",
        f"- Preview video: `{Path(metadata['preview_video']).as_posix()}`",
        f"- Model-input crop video: `{Path(metadata['model_input_video']).as_posix()}`",
        f"- Metadata: `{Path(metadata['preview_video']).with_suffix('.json').as_posix()}`",
        "",
        "## Interpretation",
        "",
        "- This preview replaces global egocentric video with the area emphasized by rolling gaze heatmap.",
        "- The right-bottom panel is the 224x224 crop-style stream that can be used as a model-input prototype.",
        "- This is not yet a calibrated Unity camera projection; it maps normalized `gaze_hit_z/y` plane coordinates to frame pixels.",
        "- `gaze_on_painting` is excluded because it is constant in the current data.",
        "",
        "## Parameters",
        "",
        f"- Rolling heatmap window: {metadata['window_s']:.1f}s.",
        f"- Data step per frame: {metadata['step_s']:.1f}s.",
        f"- Playback fps: {metadata['output_fps']:.1f}.",
        f"- Model crop: {metadata['model_size_px']}x{metadata['model_size_px']}, crop source size {metadata['crop_size_px']} px.",
        f"- Formal viewing compressed from {metadata['formal_duration_s']:.1f}s to {metadata['video_duration_s']:.1f}s.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run(
    participant_id: str,
    manifest_path: Path,
    output_dir: Path,
    report_path: Path,
    bins: int,
    window_s: float,
    step_s: float,
    fps: float,
    crop_size: int,
    model_size: int,
    color_quantile: float,
) -> dict[str, Any]:
    participant_id = normalize_participant_id(participant_id)
    manifest = _read_manifest(manifest_path)
    source = _participant_source(manifest, participant_id)
    session_dir = Path(str(source["session_dir"]))
    eye_tracking_csv = Path(str(source["eye_tracking_csv"]))
    video_frames_csv = Path(str(source["video_frames_csv"]))

    gaze = load_viewing_gaze(eye_tracking_csv, participant_id)
    gaze = add_time_weights(gaze)
    bounds = coordinate_bounds(gaze)
    gaze = add_normalized_coordinates(gaze, bounds)
    gaze, segments = _build_formal_timeline(gaze)
    video = _read_video_frames(video_frames_csv, session_dir)

    preview = output_dir / f"{participant_id}_attended_video_preview.mp4"
    model_input = output_dir / f"{participant_id}_attended_model_input.mp4"
    metadata = write_attended_video_preview(
        gaze=gaze,
        video=video,
        segments=segments,
        participant_id=participant_id,
        output_preview=preview,
        output_model=model_input,
        bins=bins,
        window_s=window_s,
        step_s=step_s,
        fps=fps,
        crop_size=crop_size,
        model_size=model_size,
        color_quantile=color_quantile,
    )
    metadata.update(
        {
            "eye_tracking_csv": str(eye_tracking_csv),
            "video_frames_csv": str(video_frames_csv),
            "coordinate_bounds": bounds.__dict__,
        }
    )
    metadata_path = preview.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(report_path, metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Create attended egocentric video preview.")
    parser.add_argument("--participant", default="P007")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--bins", type=int, default=72)
    parser.add_argument("--window-s", type=float, default=20.0)
    parser.add_argument("--step-s", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--crop-size", type=int, default=360)
    parser.add_argument("--model-size", type=int, default=224)
    parser.add_argument("--color-quantile", type=float, default=0.995)
    args = parser.parse_args()
    metadata = run(
        participant_id=args.participant,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        report_path=args.report,
        bins=args.bins,
        window_s=args.window_s,
        step_s=args.step_s,
        fps=args.fps,
        crop_size=args.crop_size,
        model_size=args.model_size,
        color_quantile=args.color_quantile,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

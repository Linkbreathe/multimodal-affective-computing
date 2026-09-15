from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.supplementary.run_attended_video_preview import (  # noqa: E402
    _current_gaze,
    _draw_text,
    _estimate_vmax,
    _gaze_pixel,
    _heatmap_mask,
    _nearest_video_frame,
    _panel_label,
    _read_video_frames,
    _rolling_gaze,
)
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
from real_time_ml.data.io import normalize_participant_id  # noqa: E402


DEFAULT_REPORT = ROOT / "artifacts" / "reports" / "four_channel_attended_video_P007_zh.md"


def _crop_with_padding_2d(values: np.ndarray, center: tuple[int, int], crop_size: int) -> np.ndarray:
    height, width = values.shape[:2]
    half = crop_size // 2
    x, y = center
    left, right = x - half, x + half
    top, bottom = y - half, y + half
    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - width)
    pad_bottom = max(0, bottom - height)
    padded = cv2.copyMakeBorder(
        values,
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


def _expand_attention_mask(mask: np.ndarray, expand_sigma: float, gamma: float) -> np.ndarray:
    expanded = np.asarray(mask, dtype=np.float32)
    if expand_sigma > 0:
        expanded = cv2.GaussianBlur(expanded, (0, 0), sigmaX=expand_sigma, sigmaY=expand_sigma)
    max_value = float(np.max(expanded))
    if max_value > 0:
        expanded = expanded / max_value
    expanded = np.power(np.clip(expanded, 0.0, 1.0), gamma)
    return np.clip(expanded, 0.0, 1.0).astype(np.float32)


def _attention_center(mask: np.ndarray, fallback: tuple[int, int]) -> tuple[int, int]:
    total = float(np.sum(mask))
    if total <= 1e-6:
        return fallback
    height, width = mask.shape
    ys, xs = np.indices((height, width), dtype=np.float32)
    x = int(round(float(np.sum(xs * mask) / total)))
    y = int(round(float(np.sum(ys * mask) / total)))
    return int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1))


def _crop_rectangle(center: tuple[int, int], crop_size: int, width: int, height: int) -> tuple[int, int, int, int]:
    half = crop_size // 2
    x, y = center
    return (
        int(np.clip(x - half, 0, width - 1)),
        int(np.clip(y - half, 0, height - 1)),
        int(np.clip(x + half, 0, width - 1)),
        int(np.clip(y + half, 0, height - 1)),
    )


def _condition_at_time(segments: list[dict[str, Any]], time_s: float) -> tuple[str, float]:
    for segment in segments:
        start = float(segment["formal_start_s"])
        end = float(segment["formal_end_s"])
        if start <= time_s <= end:
            return str(segment["condition"]), time_s - start
    segment = segments[-1]
    return str(segment["condition"]), time_s - float(segment["formal_start_s"])


def _make_four_channel_crop(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    center: tuple[int, int],
    crop_size: int,
    model_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb_crop_bgr = _crop_with_padding_2d(frame_bgr, center, crop_size)
    mask_crop = _crop_with_padding_2d(mask, center, crop_size)
    rgb_model_bgr = cv2.resize(rgb_crop_bgr, (model_size, model_size), interpolation=cv2.INTER_AREA)
    mask_model = cv2.resize(mask_crop, (model_size, model_size), interpolation=cv2.INTER_AREA)
    mask_u8 = np.clip(mask_model * 255.0, 0, 255).astype(np.uint8)
    rgb_model = cv2.cvtColor(rgb_model_bgr, cv2.COLOR_BGR2RGB)
    rgba_like = np.dstack([rgb_model, mask_u8])
    return rgba_like, rgb_model_bgr, mask_u8


def _mask_visual(mask_u8: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(mask_u8, cv2.COLORMAP_MAGMA)


def _overlay_crop(frame: np.ndarray, mask: np.ndarray, center: tuple[int, int], gaze: tuple[int, int], crop_size: int) -> np.ndarray:
    output = frame.copy()
    heat = cv2.applyColorMap(np.clip(mask * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    alpha = np.clip(mask[..., None] * 0.45, 0.0, 0.45)
    output = np.clip(output.astype(np.float32) * (1.0 - alpha) + heat.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    left, top, right, bottom = _crop_rectangle(center, crop_size, frame.shape[1], frame.shape[0])
    cv2.rectangle(output, (left, top), (right, bottom), (80, 255, 120), 3)
    cv2.circle(output, center, 12, (80, 255, 120), 2)
    cv2.circle(output, center, 4, (80, 255, 120), -1)
    cv2.circle(output, gaze, 11, (255, 255, 255), 2)
    cv2.circle(output, gaze, 4, (0, 255, 255), -1)
    return output


def _preview_frame(
    *,
    frame: np.ndarray,
    mask: np.ndarray,
    rgb_model_bgr: np.ndarray,
    mask_u8: np.ndarray,
    center: tuple[int, int],
    gaze: tuple[int, int],
    crop_size: int,
    participant_id: str,
    condition: str,
    formal_time_s: float,
    total_s: float,
    condition_elapsed_s: float,
) -> np.ndarray:
    canvas = np.full((900, 1600, 3), (14, 15, 18), dtype=np.uint8)
    raw = _overlay_crop(frame, mask, center, gaze, crop_size)
    raw_panel = cv2.resize(raw, (960, 540), interpolation=cv2.INTER_AREA)
    _panel_label(raw_panel, "raw egocentric video + crop box + heatmap channel")
    canvas[70:610, 40:1000] = raw_panel

    rgb_panel = cv2.resize(rgb_model_bgr, (360, 360), interpolation=cv2.INTER_NEAREST)
    _panel_label(rgb_panel, "RGB crop, unchanged color")
    canvas[70:430, 1120:1480] = rgb_panel

    heat_panel = cv2.resize(_mask_visual(mask_u8), (360, 360), interpolation=cv2.INTER_NEAREST)
    _panel_label(heat_panel, "4th channel: expanded attention mask")
    canvas[500:860, 1120:1480] = heat_panel

    _draw_text(canvas, f"{participant_id} RGB+attention four-channel preview", 40, 38, 0.86, (255, 255, 255), 2)
    _draw_text(canvas, f"Condition {condition} | formal {formal_time_s:.1f}/{total_s:.1f}s | condition {condition_elapsed_s:.1f}s", 40, 650, 0.68)
    _draw_text(canvas, "Model tensor per frame: 224x224x4 = RGB + heatmap mask; RGB is not darkened", 40, 685, 0.58, (210, 255, 210))
    _draw_text(canvas, f"Crop source: {crop_size}x{crop_size}, centered on rolling attention centroid; white dot = current gaze", 40, 715, 0.58, (220, 220, 220))
    _draw_text(canvas, "Spatial mapping remains pilot gaze_hit_y/z plane normalized to frame pixels", 40, 745, 0.58, (210, 230, 255))
    return canvas


def _export_preview_frames(video_path: Path, frame_index: int) -> Path:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open preview video for frame export: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
    output = video_path.with_name(f"{video_path.stem}_frame_{frame_index}.png")
    cv2.imwrite(str(output), frame)
    return output


def write_four_channel_video(
    *,
    gaze: pd.DataFrame,
    video: pd.DataFrame,
    segments: list[dict[str, Any]],
    participant_id: str,
    output_preview: Path,
    output_rgb_crop: Path,
    output_mask: Path,
    output_npz: Path,
    bins: int,
    window_s: float,
    step_s: float,
    fps: float,
    crop_size: int,
    model_size: int,
    color_quantile: float,
    expand_sigma: float,
    mask_gamma: float,
    export_tensor: bool,
) -> dict[str, Any]:
    output_preview.parent.mkdir(parents=True, exist_ok=True)
    total_s = float(gaze["formal_elapsed_s"].max())
    times = np.arange(0.0, total_s + 1e-9, step_s)
    vmax = _estimate_vmax(gaze, bins=bins, window_s=window_s, step_s=step_s, quantile=color_quantile)

    preview_writer = cv2.VideoWriter(str(output_preview), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1600, 900))
    rgb_writer = cv2.VideoWriter(str(output_rgb_crop), cv2.VideoWriter_fourcc(*"mp4v"), fps, (model_size, model_size))
    mask_writer = cv2.VideoWriter(str(output_mask), cv2.VideoWriter_fourcc(*"mp4v"), fps, (model_size, model_size))
    if not preview_writer.isOpened() or not rgb_writer.isOpened() or not mask_writer.isOpened():
        raise RuntimeError("Could not open one or more four-channel preview writers")

    tensor_frames: list[np.ndarray] = []
    center_rows: list[dict[str, Any]] = []
    try:
        for time_s in times:
            current = _current_gaze(gaze, float(time_s))
            video_row = _nearest_video_frame(video, float(current["session_elapsed_seconds"]))
            frame = cv2.imread(str(video_row["path"]))
            if frame is None:
                continue
            rolling = _rolling_gaze(gaze, float(time_s), window_s)
            base_mask = _heatmap_mask(rolling, frame.shape[1], frame.shape[0], bins=bins, vmax=vmax)
            mask = _expand_attention_mask(base_mask, expand_sigma=expand_sigma, gamma=mask_gamma)
            current_gaze = _gaze_pixel(current, frame.shape[1], frame.shape[0])
            center = _attention_center(mask, fallback=current_gaze)
            four_channel, rgb_model_bgr, mask_u8 = _make_four_channel_crop(
                frame,
                mask,
                center=center,
                crop_size=crop_size,
                model_size=model_size,
            )
            condition, condition_elapsed = _condition_at_time(segments, float(time_s))
            preview = _preview_frame(
                frame=frame,
                mask=mask,
                rgb_model_bgr=rgb_model_bgr,
                mask_u8=mask_u8,
                center=center,
                gaze=current_gaze,
                crop_size=crop_size,
                participant_id=participant_id,
                condition=condition,
                formal_time_s=float(time_s),
                total_s=total_s,
                condition_elapsed_s=condition_elapsed,
            )
            preview_writer.write(preview)
            rgb_writer.write(rgb_model_bgr)
            mask_writer.write(_mask_visual(mask_u8))
            if export_tensor:
                tensor_frames.append(four_channel)
            center_rows.append(
                {
                    "formal_elapsed_s": float(time_s),
                    "condition": condition,
                    "session_elapsed_seconds": float(current["session_elapsed_seconds"]),
                    "crop_center_x_px": int(center[0]),
                    "crop_center_y_px": int(center[1]),
                    "current_gaze_x_px": int(current_gaze[0]),
                    "current_gaze_y_px": int(current_gaze[1]),
                    "mask_mean": float(np.mean(mask)),
                    "mask_max": float(np.max(mask)),
                }
            )
    finally:
        preview_writer.release()
        rgb_writer.release()
        mask_writer.release()

    tensor_shape: tuple[int, ...] | None = None
    if export_tensor:
        tensor = np.stack(tensor_frames, axis=0).astype(np.uint8)
        tensor_shape = tuple(int(value) for value in tensor.shape)
        np.savez_compressed(
            output_npz,
            frames_rgb_heatmap=tensor,
            channel_names=np.asarray(["R", "G", "B", "attention_mask"]),
        )

    center_csv = output_preview.with_name(f"{participant_id}_four_channel_crop_centers.csv")
    pd.DataFrame(center_rows).to_csv(center_csv, index=False)
    preview_frame = _export_preview_frames(output_preview, min(300, len(center_rows) - 1))
    metadata = {
        "participant_id": participant_id,
        "preview_video": str(output_preview),
        "rgb_crop_video": str(output_rgb_crop),
        "attention_mask_video": str(output_mask),
        "four_channel_tensor_npz": str(output_npz) if export_tensor else "",
        "crop_centers_csv": str(center_csv),
        "preview_frame": str(preview_frame),
        "n_video_frames": int(len(center_rows)),
        "output_fps": float(fps),
        "video_duration_s": float(len(center_rows) / fps),
        "formal_duration_s": total_s,
        "step_s": float(step_s),
        "window_s": float(window_s),
        "bins": int(bins),
        "crop_size_px": int(crop_size),
        "model_size_px": int(model_size),
        "color_quantile": float(color_quantile),
        "vmax_s_per_bin": float(vmax),
        "attention_expand_sigma_px": float(expand_sigma),
        "attention_mask_gamma": float(mask_gamma),
        "tensor_shape": tensor_shape,
        "channel_order": ["R", "G", "B", "attention_mask"],
        "rgb_policy": "unchanged RGB crop; attention is stored only in channel 4",
        "center_policy": "rolling expanded attention centroid, fallback current gaze",
        "mapping": "pilot_plane_normalized_gaze_hit_yz_to_frame_pixels",
    }
    return metadata


def write_report(report_path: Path, metadata: dict[str, Any]) -> None:
    lines = [
        f"# {metadata['participant_id']} four-channel attended video preview",
        "",
        "## Outputs",
        "",
        f"- Raw-view preview video: `{Path(metadata['preview_video']).as_posix()}`",
        f"- RGB crop video: `{Path(metadata['rgb_crop_video']).as_posix()}`",
        f"- Attention-mask video: `{Path(metadata['attention_mask_video']).as_posix()}`",
        f"- 4-channel tensor: `{Path(metadata['four_channel_tensor_npz']).as_posix()}`",
        f"- Preview frame: `{Path(metadata['preview_frame']).as_posix()}`",
        f"- Crop centers: `{Path(metadata['crop_centers_csv']).as_posix()}`",
        "",
        "## Interpretation",
        "",
        "- The true model-style tensor is 224x224x4: R, G, B, attention_mask.",
        "- RGB is not darkened or recolored. The heatmap is the fourth channel.",
        "- The crop is centered on the rolling attention centroid and uses a larger source crop for context.",
        "- This is still a pilot mapping from normalized gaze_hit_y/z to frame pixels, not calibrated Unity projection.",
        "",
        "## Parameters",
        "",
        f"- Frames: {metadata['n_video_frames']}.",
        f"- Playback fps: {metadata['output_fps']:.1f}.",
        f"- Formal viewing compressed from {metadata['formal_duration_s']:.1f}s to {metadata['video_duration_s']:.1f}s.",
        f"- Rolling heatmap window: {metadata['window_s']:.1f}s.",
        f"- Crop source size: {metadata['crop_size_px']} px.",
        f"- Model size: {metadata['model_size_px']} px.",
        f"- Attention expansion sigma: {metadata['attention_expand_sigma_px']:.1f} px.",
        f"- Attention gamma: {metadata['attention_mask_gamma']:.2f}.",
        f"- Tensor shape: {metadata['tensor_shape']}.",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run(
    *,
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
    expand_sigma: float,
    mask_gamma: float,
    export_tensor: bool,
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

    output_preview = output_dir / f"{participant_id}_four_channel_rgb_heatmap_preview.mp4"
    output_rgb_crop = output_dir / f"{participant_id}_four_channel_rgb_crop.mp4"
    output_mask = output_dir / f"{participant_id}_four_channel_attention_mask.mp4"
    output_npz = output_dir / f"{participant_id}_four_channel_rgb_heatmap_tensor.npz"
    metadata = write_four_channel_video(
        gaze=gaze,
        video=video,
        segments=segments,
        participant_id=participant_id,
        output_preview=output_preview,
        output_rgb_crop=output_rgb_crop,
        output_mask=output_mask,
        output_npz=output_npz,
        bins=bins,
        window_s=window_s,
        step_s=step_s,
        fps=fps,
        crop_size=crop_size,
        model_size=model_size,
        color_quantile=color_quantile,
        expand_sigma=expand_sigma,
        mask_gamma=mask_gamma,
        export_tensor=export_tensor,
    )
    metadata.update(
        {
            "eye_tracking_csv": str(eye_tracking_csv),
            "video_frames_csv": str(video_frames_csv),
            "coordinate_bounds": bounds.__dict__,
        }
    )
    metadata_path = output_preview.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(report_path, metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Create RGB+heatmap 4-channel attended video preview.")
    parser.add_argument("--participant", default="P007")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
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
    parser.add_argument("--no-tensor", action="store_true")
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
        expand_sigma=args.expand_sigma,
        mask_gamma=args.mask_gamma,
        export_tensor=not args.no_tensor,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

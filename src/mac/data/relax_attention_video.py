"""Derived Relax attention-video clips from eye tracking and egocentric video."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch

from mac.data.relax_foundation import RelaxHardFailure
from mac.data.video_transforms import normalize_clip, preprocess_frame


@dataclass(frozen=True)
class AttentionVideoConfig:
    hfov_deg: float = 90.0
    max_gaze_lag_ms: float = 150.0
    min_valid_projected_frames: int = 12
    sigma_fraction: float = 0.10
    periphery_scale: float = 0.35
    clip_frames: int = 16


def _unit(value: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray | None:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,) or not np.isfinite(arr).all():
        return None
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return None
    return arr / norm


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value) and np.isfinite(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _float_row(row: dict[str, Any] | pd.Series, key: str) -> float:
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def project_gaze_to_frame(
    *,
    gaze_direction: np.ndarray | list[float] | tuple[float, ...],
    camera_forward: np.ndarray | list[float] | tuple[float, ...],
    camera_up: np.ndarray | list[float] | tuple[float, ...],
    width: int | float,
    height: int | float,
    config: AttentionVideoConfig,
) -> tuple[float, float] | None:
    """Project a world-space gaze vector into egocentric frame coordinates."""
    gaze = _unit(gaze_direction)
    forward = _unit(camera_forward)
    up = _unit(camera_up)
    width_f = float(width)
    height_f = float(height)
    if gaze is None or forward is None or up is None or width_f <= 1 or height_f <= 1:
        return None

    right = _unit(np.cross(up, forward))
    if right is None:
        return None
    up = _unit(np.cross(forward, right))
    if up is None:
        return None

    denom = float(np.dot(gaze, forward))
    if denom <= 1e-6:
        return None
    x_tan = float(np.dot(gaze, right) / denom)
    y_tan = float(np.dot(gaze, up) / denom)

    hfov = np.deg2rad(float(config.hfov_deg))
    if not np.isfinite(hfov) or hfov <= 0.0 or hfov >= np.pi:
        raise RelaxHardFailure(f"Invalid attention-video horizontal FOV: {config.hfov_deg}")
    aspect = width_f / height_f
    vfov = 2.0 * np.arctan(np.tan(hfov / 2.0) / aspect)
    x_norm = x_tan / np.tan(hfov / 2.0)
    y_norm = y_tan / np.tan(vfov / 2.0)
    if not np.isfinite([x_norm, y_norm]).all():
        return None

    x = (0.5 + 0.5 * x_norm) * (width_f - 1.0)
    y = (0.5 - 0.5 * y_norm) * (height_f - 1.0)
    x = float(np.clip(x, 0.0, width_f - 1.0))
    y = float(np.clip(y, 0.0, height_f - 1.0))
    return x, y


def apply_gaze_heatmap(
    frame_bgr: np.ndarray,
    *,
    center_xy: tuple[float, float],
    config: AttentionVideoConfig,
) -> np.ndarray:
    """Dim frame periphery while preserving pixels near the gaze projection."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise RelaxHardFailure(f"Expected BGR frame [H,W,3], got shape {frame_bgr.shape}")
    h, w = frame_bgr.shape[:2]
    sigma = max(1.0, float(config.sigma_fraction) * float(min(h, w)))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = float(center_xy[0]), float(center_xy[1])
    heat = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma))
    periphery = float(config.periphery_scale)
    if not np.isfinite(periphery) or periphery < 0.0 or periphery > 1.0:
        raise RelaxHardFailure(f"Invalid attention-video periphery scale: {config.periphery_scale}")
    scale = periphery + (1.0 - periphery) * heat
    out = frame_bgr.astype(np.float32) * scale[..., None]
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _nearest_eye_row(
    eye_rows: pd.DataFrame,
    *,
    timestamp_ms: float,
    config: AttentionVideoConfig,
) -> pd.Series | None:
    if "unix_time_ms" not in eye_rows.columns or eye_rows.empty:
        return None
    times = pd.to_numeric(eye_rows["unix_time_ms"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(times)
    if not finite.any():
        return None
    finite_indices = np.flatnonzero(finite)
    finite_times = times[finite_indices]
    nearest_pos = int(np.argmin(np.abs(finite_times - float(timestamp_ms))))
    lag = abs(float(finite_times[nearest_pos]) - float(timestamp_ms))
    if lag > float(config.max_gaze_lag_ms):
        return None
    return eye_rows.iloc[int(finite_indices[nearest_pos])]


class RelaxAttentionVideoExtractor:
    """Build VideoMAE-compatible attention-video clips from aligned eye/video rows."""

    def __init__(self, config: AttentionVideoConfig) -> None:
        self.config = config

    def build_clip(
        self,
        *,
        frame_rows: list[dict[str, Any]],
        eye_rows: pd.DataFrame,
    ) -> tuple[torch.Tensor, bool, dict[str, Any]]:
        if len(frame_rows) != int(self.config.clip_frames):
            return (
                torch.zeros(3, int(self.config.clip_frames), 224, 224),
                False,
                {"reason": "insufficient_video_frames", "frame_count": len(frame_rows)},
            )

        processed_frames: list[np.ndarray] = []
        valid_projected = 0
        missing_gaze = 0
        unreadable_frames = []

        for frame_row in frame_rows:
            path = Path(str(frame_row.get("path")))
            frame = cv2.imread(str(path))
            if frame is None:
                unreadable_frames.append(str(path))
                continue
            timestamp_ms = _float_row(frame_row, "unix_time_ms")
            eye = _nearest_eye_row(eye_rows, timestamp_ms=timestamp_ms, config=self.config)
            center = None
            if eye is not None and _as_bool(eye.get("gaze_available", True)):
                center = project_gaze_to_frame(
                    gaze_direction=np.array(
                        [
                            _float_row(eye, "gaze_direction_x"),
                            _float_row(eye, "gaze_direction_y"),
                            _float_row(eye, "gaze_direction_z"),
                        ],
                        dtype=float,
                    ),
                    camera_forward=np.array(
                        [
                            _float_row(frame_row, "camera_forward_x"),
                            _float_row(frame_row, "camera_forward_y"),
                            _float_row(frame_row, "camera_forward_z"),
                        ],
                        dtype=float,
                    ),
                    camera_up=np.array(
                        [
                            _float_row(frame_row, "camera_up_x"),
                            _float_row(frame_row, "camera_up_y"),
                            _float_row(frame_row, "camera_up_z"),
                        ],
                        dtype=float,
                    ),
                    width=int(round(_float_row(frame_row, "width"))) if np.isfinite(_float_row(frame_row, "width")) else frame.shape[1],
                    height=int(round(_float_row(frame_row, "height"))) if np.isfinite(_float_row(frame_row, "height")) else frame.shape[0],
                    config=self.config,
                )
            if center is None:
                missing_gaze += 1
                attended = frame
            else:
                valid_projected += 1
                attended = apply_gaze_heatmap(frame, center_xy=center, config=self.config)
            processed_frames.append(preprocess_frame(attended))

        metadata = {
            "valid_projected_frames": valid_projected,
            "missing_or_unprojectable_gaze_frames": missing_gaze,
            "unreadable_frame_count": len(unreadable_frames),
        }
        if unreadable_frames:
            metadata["reason"] = "unreadable_video_frame"
            metadata["unreadable_frames"] = unreadable_frames[:5]
            return torch.zeros(3, int(self.config.clip_frames), 224, 224), False, metadata
        if valid_projected < int(self.config.min_valid_projected_frames):
            metadata["reason"] = "insufficient_projected_gaze_frames"
            return torch.zeros(3, int(self.config.clip_frames), 224, 224), False, metadata
        if len(processed_frames) != int(self.config.clip_frames):
            metadata["reason"] = "insufficient_video_frames"
            return torch.zeros(3, int(self.config.clip_frames), 224, 224), False, metadata
        return normalize_clip(np.stack(processed_frames, axis=0)), True, metadata

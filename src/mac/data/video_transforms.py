"""Shared video preprocessing for VideoMAE v2.

Implements the official preprocessing pipeline from the HuggingFace
preprocessor_config.json: resize shortest edge to 224 (preserve aspect ratio),
center crop to 224x224, ImageNet normalize.

Reference: https://huggingface.co/OpenGVLab/VideoMAEv2-Base
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLIP_LEN = 16  # frames per clip for VideoMAE v2


def resize_short_edge(frame: np.ndarray, size: int = 224) -> np.ndarray:
    """Resize so the shortest edge equals *size*, preserving aspect ratio."""
    h, w = frame.shape[:2]
    if h <= w:
        new_h = size
        new_w = int(round(w * size / h))
    else:
        new_w = size
        new_h = int(round(h * size / w))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def center_crop(frame: np.ndarray, crop_size: int = 224) -> np.ndarray:
    """Center crop to (crop_size, crop_size)."""
    h, w = frame.shape[:2]
    y = (h - crop_size) // 2
    x = (w - crop_size) // 2
    return frame[y : y + crop_size, x : x + crop_size]


def preprocess_frame(frame: np.ndarray, crop_size: int = 224) -> np.ndarray:
    """BGR frame -> RGB, resize shortest edge, center crop.

    Returns uint8 [crop_size, crop_size, 3] RGB array.
    """
    frame = resize_short_edge(frame, crop_size)
    frame = center_crop(frame, crop_size)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def normalize_clip(frames: np.ndarray) -> torch.Tensor:
    """RGB uint8 frames [T, H, W, 3] -> normalized [C, T, H, W] float tensor."""
    clip = frames.astype(np.float32) / 255.0
    clip = (clip - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(clip).permute(3, 0, 1, 2).contiguous()


def make_consecutive_clips(
    frames: np.ndarray, clip_len: int = CLIP_LEN,
) -> list[torch.Tensor]:
    """Non-overlapping consecutive clips [C, T, H, W] from frame stack.

    Drops trailing frames that don't fill a complete clip.
    """
    clips = []
    for i in range(0, len(frames) - clip_len + 1, clip_len):
        clips.append(normalize_clip(frames[i : i + clip_len]))
    return clips


def read_frames(
    cap: cv2.VideoCapture,
    start_sec: float,
    duration_sec: float,
    crop_size: int = 224,
) -> np.ndarray:
    """Read and preprocess frames from a VideoCapture for a time range.

    Uses timestamp-based seeking (CAP_PROP_POS_MSEC) which is more reliable
    than frame-index seeking for compressed video with B-frames.

    Returns [N, crop_size, crop_size, 3] uint8 RGB array.
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    expected_frames = int(duration_sec * fps)

    # Seek by timestamp (milliseconds) -- more reliable than frame index
    cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)

    frames = []
    for _ in range(expected_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(preprocess_frame(frame, crop_size))

    if not frames:
        return np.empty((0, crop_size, crop_size, 3), dtype=np.uint8)
    return np.stack(frames)

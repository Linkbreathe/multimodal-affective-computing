"""Tests for src.data.video_transforms."""
import numpy as np
import pytest
import torch

from mac.data.video_transforms import (
    CLIP_LEN,
    IMAGENET_MEAN,
    IMAGENET_STD,
    center_crop,
    make_consecutive_clips,
    normalize_clip,
    preprocess_frame,
    resize_short_edge,
)


def _make_bgr(h: int = 480, w: int = 640) -> np.ndarray:
    """Dummy BGR frame."""
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


def _make_rgb_frames(n: int = 16, h: int = 224, w: int = 224) -> np.ndarray:
    """Dummy RGB frames [N, H, W, 3]."""
    return np.random.randint(0, 256, (n, h, w, 3), dtype=np.uint8)


# -- resize_short_edge -------------------------------------------------------

def test_resize_short_edge_landscape():
    frame = _make_bgr(480, 640)
    out = resize_short_edge(frame, 224)
    assert out.shape[0] == 224  # height is shortest
    assert out.shape[1] > 224   # width preserves ratio


def test_resize_short_edge_portrait():
    frame = _make_bgr(640, 480)
    out = resize_short_edge(frame, 224)
    assert out.shape[1] == 224
    assert out.shape[0] > 224


def test_resize_short_edge_square():
    frame = _make_bgr(224, 224)
    out = resize_short_edge(frame, 224)
    assert out.shape[:2] == (224, 224)


# -- center_crop --------------------------------------------------------------

def test_center_crop_exact():
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    out = center_crop(frame, 224)
    assert out.shape == (224, 224, 3)


def test_center_crop_preserves_center_pixel():
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    frame[150, 150] = [255, 0, 0]
    out = center_crop(frame, 224)
    cy, cx = 224 // 2, 224 // 2
    # Center pixel of original (150,150) maps to center of crop
    assert out[cy, cx, 0] == 255


# -- preprocess_frame ---------------------------------------------------------

def test_preprocess_frame_output_shape():
    bgr = _make_bgr(1080, 1920)
    out = preprocess_frame(bgr, crop_size=224)
    assert out.shape == (224, 224, 3)
    assert out.dtype == np.uint8


# -- normalize_clip -----------------------------------------------------------

def test_normalize_clip_shape():
    frames = _make_rgb_frames(16, 224, 224)
    clip = normalize_clip(frames)
    assert clip.shape == (3, 16, 224, 224)
    assert clip.dtype == torch.float32


def test_normalize_clip_values():
    # All-zero image should give -(mean/std)
    frames = np.zeros((1, 224, 224, 3), dtype=np.uint8)
    clip = normalize_clip(frames)
    expected_r = -IMAGENET_MEAN[0] / IMAGENET_STD[0]
    assert abs(clip[0, 0, 0, 0].item() - expected_r) < 1e-5


# -- make_consecutive_clips ---------------------------------------------------

def test_make_clips_exact_multiple():
    frames = _make_rgb_frames(32)
    clips = make_consecutive_clips(frames, clip_len=16)
    assert len(clips) == 2
    for c in clips:
        assert c.shape == (3, 16, 224, 224)


def test_make_clips_drops_remainder():
    frames = _make_rgb_frames(50)
    clips = make_consecutive_clips(frames, clip_len=16)
    assert len(clips) == 3  # 48 used, 2 dropped


def test_make_clips_too_few_frames():
    frames = _make_rgb_frames(10)
    clips = make_consecutive_clips(frames, clip_len=16)
    assert len(clips) == 0


def test_clip_len_constant():
    assert CLIP_LEN == 16

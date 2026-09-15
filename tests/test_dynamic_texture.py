from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mac.data.video import VideoFrame, VideoIndex, uniform_clip_frames
from mac.features.dynamic_texture import (
    DYNAMIC_TEXTURE_COLUMNS,
    DynamicTextureSettings,
    dynamic_texture_descriptor,
)


def test_dynamic_texture_descriptor_is_finite_and_grayscale_only() -> None:
    settings = DynamicTextureSettings()
    grid_y, grid_x = np.mgrid[: settings.analysis_size, : settings.analysis_size]
    frames = np.stack(
        [
            np.sin((grid_x + index * 2.0) / 8.0) + np.cos((grid_y - index) / 11.0)
            for index in range(settings.frames_per_window)
        ]
    ).astype(np.float32)
    frames = (frames - frames.min()) / (frames.max() - frames.min())
    timestamps = np.asarray([1_000_000 + index * (590 + index % 3) for index in range(16)])

    descriptor = dynamic_texture_descriptor(frames, timestamps, settings)

    assert tuple(descriptor) == DYNAMIC_TEXTURE_COLUMNS
    assert np.isfinite(list(descriptor.values())).all()
    assert descriptor["video_dynamic_motion_mean_diag_s"] >= 0.0
    assert 0.0 <= descriptor["video_dynamic_direction_resultant"] <= 1.0
    assert 0.0 <= descriptor["video_dynamic_spectral_entropy_3d"] <= 1.0
    assert 0.0 <= descriptor["video_dynamic_motion_energy_stability"] <= 1.0
    with pytest.raises(ValueError, match="RGB tensors are rejected"):
        dynamic_texture_descriptor(np.repeat(frames[..., None], 3, axis=-1), timestamps, settings)


def test_uniform_dynamic_texture_clip_has_distinct_nearest_frames(tmp_path: Path) -> None:
    frames = tuple(
        VideoFrame(10_000 + index * 100, tmp_path / f"frame_{index:06d}.jpg", index)
        for index in range(1, 102)
    )
    index = VideoIndex("PTEST", "unix_time_ms", "ok", frames)

    selected = uniform_clip_frames(index, 10_000.0, 20_000.0, 16)

    assert len(selected) == 16
    assert len({frame.frame_index for frame in selected}) == 16
    assert [frame.unix_time_ms for frame in selected] == sorted(
        frame.unix_time_ms for frame in selected
    )

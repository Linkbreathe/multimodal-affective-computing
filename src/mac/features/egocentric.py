"""Independent handcrafted egocentric-video feature set."""

from __future__ import annotations

from typing import Any

from real_time_ml.config import ProjectConfig
from real_time_ml.data.index import build_index
from real_time_ml.features.extract import extract_features


def extract_handcrafted_egocentric_features(
    config: ProjectConfig, participants: list[str] | None = None
) -> dict[str, Any]:
    """Build a video-enabled table without touching the selected no-video table."""
    selected = participants or config.participants
    # Always restore the complete source manifest before an all-participant visual run.
    build_index(config, selected)
    output = config.path("features") / "video_ml"
    result = extract_features(config, selected, include_video=True, output_dir=output)
    result["feature_dir"] = str(output)
    return result

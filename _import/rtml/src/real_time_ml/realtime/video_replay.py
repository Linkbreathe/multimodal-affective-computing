"""Recorded-only shadow replay entry points for visual model namespaces."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from real_time_ml.config import ProjectConfig
from real_time_ml.realtime.replay import replay


def _visual_config(config: ProjectConfig, model_dir: Path, backend: str) -> ProjectConfig:
    data = deepcopy(config.data)
    data["paths"]["models"] = str(model_dir)
    data["modeling"]["runtime_backend"] = backend
    data["modeling"]["fallback_chain"] = ["full", "no_video"] if backend == "video_dcnn" else ["full", "no_eeg", "behavior_only"]
    return ProjectConfig(source=config.source, data=data)


def _videomae_replay_table(config: ProjectConfig) -> Path:
    import pandas as pd

    root = config.path("video")
    output = root / "features" / "videomae2_dcnn" / "window_features_replay.csv"
    windows = pd.read_csv(config.path("features") / "video_ml" / "window_features.csv")
    embeddings = pd.read_csv(root / "videomae2" / "window_embeddings.csv")
    keys = ["participant_id", "condition", "condition_window_index"]
    visual_columns = [name for name in embeddings.columns if name.startswith("video_embedding_") or name in {"video_available", "video_reason", "video_timestamp_source"}]
    output.parent.mkdir(parents=True, exist_ok=True)
    windows.merge(embeddings[[*keys, *visual_columns]], on=keys, how="left", validate="one_to_one").to_csv(output, index=False)
    return output


def replay_visual_model(
    config: ProjectConfig,
    backend: str,
    participants: list[str] | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    root = config.path("video")
    if backend == "handcrafted":
        source = config.path("features") / "video_ml" / "window_features.csv"
        model_dir = root / "models" / "handcrafted" / "visual"
        shadow_config = _visual_config(config, model_dir, "classical")
        stem = "handcrafted_video_shadow_replay"
    elif backend == "videomae2":
        source = _videomae_replay_table(config)
        model_dir = root / "models" / "videomae2_dcnn"
        shadow_config = _visual_config(config, model_dir, "video_dcnn")
        stem = "videomae2_video_shadow_replay"
    else:
        raise ValueError("backend must be 'handcrafted' or 'videomae2'")
    if not source.exists():
        raise FileNotFoundError(f"Visual replay source is missing: {source}")
    report_dir = root / "reports" / "replay"
    return replay(
        shadow_config, participants, output or root / "realtime" / f"{stem}.jsonl",
        source=source, reports_dir=report_dir, report_stem=stem,
    )

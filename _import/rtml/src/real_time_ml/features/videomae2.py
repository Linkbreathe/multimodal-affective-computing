"""Frozen official VideoMAE V2 embedding extraction for recorded egocentric video."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import types
from typing import Any

import numpy as np

from real_time_ml.config import ProjectConfig
from real_time_ml.data.index import build_index
from real_time_ml.data.tables import read_rows, write_parquet_if_available, write_rows
from real_time_ml.data.video import load_video_index, uniform_clip_frames
from real_time_ml.utils import file_sha256, write_json


def _setting_path(config: ProjectConfig, dotted: str) -> Path:
    value = Path(str(config.get(dotted)))
    return value if value.is_absolute() else (config.source.parent / value).resolve()


def _model_metadata(config: ProjectConfig) -> dict[str, Any]:
    checkpoint = _setting_path(config, "features.video.videomae2.checkpoint")
    return {
        "repository_commit": str(config.get("features.video.videomae2.repo_commit")),
        "model_name": str(config.get("features.video.videomae2.model_name")),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint) if checkpoint.exists() else None,
        "embedding_dimension": int(config.get("features.video.videomae2.embedding_dimension", 384)),
        "frames_per_window": int(config.get("features.video.videomae2.frames_per_window", 16)),
    }


def _install_timm_compat() -> None:
    """Provide the four legacy timm helpers VideoMAE V2 actually imports.

    The pinned 2023 source only imports ``drop_path``, ``to_2tuple``,
    ``trunc_normal_`` and ``register_model``.  Windows CUDA environments can
    have a torchvision/Pillow binary mismatch that makes importing the entire
    modern timm package fail, although the official VideoMAE implementation
    itself does not need torchvision.  The shim preserves the upstream model
    implementation while avoiding that unrelated import path.
    """
    try:
        from timm.models.layers import drop_path, to_2tuple, trunc_normal_  # noqa: F401
        from timm.models.registry import register_model  # noqa: F401
        return
    except Exception:
        pass
    for name in list(sys.modules):
        if name == "timm" or name.startswith("timm."):
            sys.modules.pop(name, None)
    import torch

    timm = types.ModuleType("timm")
    models = types.ModuleType("timm.models")
    layers = types.ModuleType("timm.models.layers")
    registry = types.ModuleType("timm.models.registry")

    def drop_path(value, drop_prob: float | None = None, training: bool = False):
        if not drop_prob or not training:
            return value
        keep = 1.0 - float(drop_prob)
        shape = (value.shape[0],) + (1,) * (value.ndim - 1)
        random = keep + torch.rand(shape, dtype=value.dtype, device=value.device)
        return value.div(keep) * random.floor()

    layers.drop_path = drop_path
    layers.to_2tuple = lambda value: value if isinstance(value, tuple) else (value, value)
    layers.trunc_normal_ = torch.nn.init.trunc_normal_
    registry.register_model = lambda function: function
    sys.modules.update({"timm": timm, "timm.models": models, "timm.models.layers": layers, "timm.models.registry": registry})


def _load_official_model(config: ProjectConfig):
    """Load the pinned OpenGVLab source without substituting an unrelated model."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("VideoMAE2 extraction requires PyTorch in the isolated rtml-videomae2 environment") from error
    repo = _setting_path(config, "features.video.videomae2.repo_dir")
    checkpoint = _setting_path(config, "features.video.videomae2.checkpoint")
    if not repo.exists() or not (repo / "models" / "modeling_finetune.py").exists():
        raise RuntimeError(
            f"Pinned VideoMAE2 source is missing at {repo}. Run scripts/setup_videomae2.ps1 before extraction."
        )
    if not checkpoint.exists():
        raise RuntimeError(
            f"Pinned VideoMAE2 checkpoint is missing at {checkpoint}. Run scripts/setup_videomae2.ps1 before extraction."
        )
    _install_timm_compat()
    sys.path.insert(0, str(repo))
    try:
        module = importlib.import_module("models.modeling_finetune")
        factory = getattr(module, str(config.get("features.video.videomae2.model_name")))
        try:
            # The official constructor initializes ``head.weight`` unconditionally,
            # so retain its 400-class head even though extraction calls only
            # ``forward_features`` and never the classifier.
            model = factory(pretrained=False, num_classes=400, all_frames=16, tubelet_size=2, use_mean_pooling=True)
        except TypeError:
            model = factory(pretrained=False, num_classes=400, all_frames=16, tubelet_size=2)
    finally:
        if sys.path and sys.path[0] == str(repo):
            sys.path.pop(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model, device


def _read_frames_from_mp4(path: Path, frame_indexes: list[int]) -> list[np.ndarray] | None:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required to decode retained MP4 files") from error
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None
    output: list[np.ndarray] = []
    try:
        for index in frame_indexes:
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(index) - 1))
            ok, frame = capture.read()
            if not ok or frame is None:
                return None
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output.append(cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR))
    finally:
        capture.release()
    return output


def _embedding(model, device, frames: list[np.ndarray]) -> np.ndarray:
    import torch

    # VideoMAE V2's official feature example uses [B, C, T, H, W] uint8 scaled to [0, 1].
    tensor = np.stack(frames, axis=0).astype(np.float32) / 255.0
    value = torch.as_tensor(tensor.transpose(3, 0, 1, 2)[None, ...], device=device)
    with torch.no_grad():
        output = model.forward_features(value)
    if isinstance(output, (tuple, list)):
        output = output[0]
    if output.ndim == 3:
        output = output.mean(dim=1)
    vector = output.detach().float().cpu().numpy().reshape(-1)
    return vector.astype(float)


def extract_videomae2_embeddings(
    config: ProjectConfig,
    participants: list[str] | None = None,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Extract exactly one frozen VideoMAE2 embedding for each valid 10-second window."""
    selected = participants or config.participants
    source_windows = config.path("preprocessed") / "windows.csv"
    if not source_windows.exists():
        raise FileNotFoundError("Run 'rtml preprocess' before extracting VideoMAE2 embeddings")
    output_dir = config.path("video") / "videomae2"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "window_embeddings.csv"
    manifest_path = output_dir / "embedding_manifest.json"
    metadata = _model_metadata(config)
    if output.exists() and manifest_path.exists() and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("model") == metadata:
            return {"output": str(output), "reused": True, "rows": len(read_rows(output))}
    source_by_participant = {row["participant_id"]: row for row in build_index(config, selected)}
    indexes = {}
    for participant, source in source_by_participant.items():
        indexes[participant] = load_video_index(
            Path(source["video_frames_csv"]) if source.get("video_frames_csv") else None,
            Path(source["session_dir"]) if source.get("session_dir") else None,
            participant,
        )
    windows = [row for row in read_rows(source_windows) if row["participant_id"] in set(selected)]
    dimensions = int(metadata["embedding_dimension"])
    count = int(metadata["frames_per_window"])
    candidates: list[tuple[dict[str, Any], list[int]]] = []
    rows: list[dict[str, Any]] = []
    for window in sorted(windows, key=lambda item: (item["participant_id"], float(item["window_start_unix_ms"]))):
        participant = window["participant_id"]
        start = float(window["window_start_unix_ms"])
        end = float(window["window_end_unix_ms"])
        clip = uniform_clip_frames(indexes[participant], start, end, count)
        row: dict[str, Any] = {
            "participant_id": participant,
            "condition": window["condition"],
            "condition_window_index": int(float(window["condition_window_index"])),
            "window_start_unix_ms": int(start),
            "window_end_unix_ms": int(end),
            "video_available": 0.0,
            "video_timestamp_source": indexes[participant].timestamp_source,
            "video_reason": "ok" if clip else "insufficient_distinct_video_frames",
            "video_frame_indexes": json.dumps([frame.frame_index for frame in clip]),
        }
        mp4 = config.path("video") / "mp4" / f"{participant}.mp4"
        if clip and mp4.exists():
            row["video_available"] = 1.0
            candidates.append((row, [frame.frame_index for frame in clip]))
        elif clip:
            row["video_reason"] = "retained_mp4_missing"
        rows.append(row)
    if candidates:
        model, device = _load_official_model(config)
        for row, indexes_for_clip in candidates:
            mp4 = config.path("video") / "mp4" / f"{row['participant_id']}.mp4"
            decoded = _read_frames_from_mp4(mp4, indexes_for_clip)
            if decoded is None or len(decoded) != count:
                row["video_available"] = 0.0
                row["video_reason"] = "mp4_decode_failed"
                continue
            vector = _embedding(model, device, decoded)
            if vector.size != dimensions or not np.all(np.isfinite(vector)):
                row["video_available"] = 0.0
                row["video_reason"] = "videomae2_invalid_embedding"
                continue
            row.update({f"video_embedding_{index:03d}": float(value) for index, value in enumerate(vector)})
    for row in rows:
        for index in range(dimensions):
            row.setdefault(f"video_embedding_{index:03d}", float("nan"))
    write_rows(output, rows)
    write_parquet_if_available(output.with_suffix(".parquet"), rows)
    window_hash = file_sha256(source_windows)
    write_json(manifest_path, {
        "model": metadata,
        "source_windows_sha256": window_hash,
        "rows": len(rows),
        "available": sum(float(row["video_available"]) >= 0.5 for row in rows),
        "missing": sum(float(row["video_available"]) < 0.5 for row in rows),
    })
    return {"output": str(output), "rows": len(rows), "available": sum(float(row["video_available"]) >= 0.5 for row in rows)}

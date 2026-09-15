"""Egocentric video indexing and retained-MP4 preparation.

The Unity logs contain the authoritative relationship between an experiment
window and a captured first-person frame.  ``absolute_path`` in older logs can
refer to a different machine, so frame resolution deliberately uses only the
session directory plus ``relative_path``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any, Iterable

import numpy as np

from real_time_ml.config import ProjectConfig
from real_time_ml.data.index import build_index
from real_time_ml.data.io import iter_csv, parse_float, resolve_video_path, sniff_csv
from real_time_ml.data.tables import write_rows
from real_time_ml.utils import file_sha256, write_json


FRAME_NAME = re.compile(r"^frame_(\d+)\.jpg$", re.IGNORECASE)


@dataclass(frozen=True)
class VideoFrame:
    unix_time_ms: int
    path: Path
    frame_index: int


@dataclass(frozen=True)
class VideoIndex:
    participant_id: str
    timestamp_source: str
    reason: str
    frames: tuple[VideoFrame, ...]


def _iso_to_unix_ms(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(round(parsed.timestamp() * 1000.0))


def _plausible_capture_timestamps(values: list[int]) -> bool:
    """Accept a normal 10 fps clock, reject P003's rounded scientific value."""
    if len(values) < 2:
        return bool(values)
    ordered = np.asarray(values, dtype=np.int64)
    deltas = np.diff(ordered)
    positive = deltas[deltas > 0]
    if len(positive) < max(2, int(0.90 * len(deltas))):
        return False
    median = float(np.median(positive))
    # Unity capture is nominally 10 fps but can have normal encoder jitter.
    return 50.0 <= median <= 200.0 and float(np.mean(deltas >= 0)) >= 0.98


def load_video_index(
    path: Path | None,
    session_dir: Path | None,
    participant_id: str,
) -> VideoIndex:
    """Load frames with a safe timestamp-source fallback.

    The whole log uses one timestamp source, avoiding an interleaved sequence
    where a few malformed values silently change ordering.
    """
    if path is None or session_dir is None or not path.exists():
        return VideoIndex(participant_id, "none", "video_log_missing", ())
    _, decimal, _ = sniff_csv(path)
    raw: list[tuple[float | None, int | None, Path | None, int]] = []
    for position, row in enumerate(iter_csv(path), start=1):
        frame_path = resolve_video_path(session_dir, row)
        if frame_path is None:
            continue
        number = parse_float(row.get("unix_time_ms"), decimal)
        timestamp = int(round(number)) if number is not None and number >= 100_000_000_000 else None
        iso_timestamp = _iso_to_unix_ms(row.get("utc_timestamp_iso"))
        frame_number = parse_float(row.get("frame_index"), decimal)
        index = int(round(frame_number)) if frame_number is not None and frame_number > 0 else position
        raw.append((timestamp, iso_timestamp, frame_path, index))
    direct = [item[0] for item in raw if item[0] is not None]
    if len(direct) == len(raw) and _plausible_capture_timestamps([int(value) for value in direct]):
        source = "unix_time_ms"
        selected = [(int(timestamp), frame_path, index) for timestamp, _, frame_path, index in raw]
    else:
        iso = [item[1] for item in raw if item[1] is not None]
        if len(iso) != len(raw) or not _plausible_capture_timestamps([int(value) for value in iso]):
            return VideoIndex(participant_id, "none", "video_timestamps_unusable", ())
        source = "utc_timestamp_iso"
        selected = [(int(timestamp), frame_path, index) for _, timestamp, frame_path, index in raw]
    frames = tuple(
        VideoFrame(timestamp, frame_path, index)
        for timestamp, frame_path, index in sorted(selected, key=lambda item: (item[0], item[2]))
        if frame_path is not None
    )
    if not frames:
        return VideoIndex(participant_id, source, "video_frames_missing", ())
    return VideoIndex(participant_id, source, "ok", frames)


def frames_in_window(index: VideoIndex, start_ms: float, end_ms: float) -> tuple[VideoFrame, ...]:
    if not index.frames:
        return ()
    timestamps = np.asarray([frame.unix_time_ms for frame in index.frames], dtype=float)
    left, right = np.searchsorted(timestamps, [start_ms, end_ms], side="left")
    return index.frames[int(left) : int(right)]


def sample_window_frames(index: VideoIndex, start_ms: float, end_ms: float, fps: float) -> tuple[VideoFrame, ...]:
    """Pick source frames nearest to each 2 fps target without duplication."""
    candidates = frames_in_window(index, start_ms, end_ms)
    if not candidates or fps <= 0:
        return ()
    timestamps = np.asarray([item.unix_time_ms for item in candidates], dtype=float)
    targets = np.arange(start_ms, end_ms, 1000.0 / fps)
    selected: list[VideoFrame] = []
    seen: set[int] = set()
    for target in targets:
        right = int(np.searchsorted(timestamps, target, side="left"))
        choices = [position for position in (right - 1, right) if 0 <= position < len(candidates)]
        if not choices:
            continue
        position = min(choices, key=lambda value: abs(timestamps[value] - target))
        frame = candidates[position]
        if frame.frame_index not in seen:
            selected.append(frame)
            seen.add(frame.frame_index)
    return tuple(selected)


def uniform_clip_frames(index: VideoIndex, start_ms: float, end_ms: float, count: int = 16) -> tuple[VideoFrame, ...]:
    """Return exactly ``count`` distinct frames spanning an entire 10-second window."""
    candidates = frames_in_window(index, start_ms, end_ms)
    if len(candidates) < count:
        return ()
    timestamps = np.asarray([item.unix_time_ms for item in candidates], dtype=float)
    targets = np.linspace(start_ms, end_ms, num=count, endpoint=False, dtype=float)
    targets += (end_ms - start_ms) / (2.0 * count)
    selected: list[VideoFrame] = []
    used: set[int] = set()
    for target in targets:
        order = np.argsort(np.abs(timestamps - target))
        frame = next((candidates[int(position)] for position in order if candidates[int(position)].frame_index not in used), None)
        if frame is None:
            return ()
        selected.append(frame)
        used.add(frame.frame_index)
    return tuple(selected) if len(selected) == count else ()


def _find_executable(name: str) -> str | None:
    candidate = shutil.which(name)
    if candidate:
        return candidate
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        fallback = Path(prefix) / "Library" / "bin" / f"{name}.exe"
        if fallback.exists():
            return str(fallback)
    return None


def _probe_video(ffprobe: str, path: Path, retries: int = 30) -> dict[str, Any]:
    command = [
        ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
        "stream=avg_frame_rate,nb_read_frames,duration,width,height", "-of", "json", str(path),
    ]
    completed = None
    for attempt in range(retries):
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode == 0:
            break
        # On Windows an interrupted/slow FFmpeg process can create the file before
        # its moov atom is written. Wait for the finalized file instead of accepting it.
        if attempt + 1 == retries:
            raise RuntimeError(f"ffprobe failed for {path}: {completed.stderr.strip()}")
        time.sleep(5)
    assert completed is not None
    stream = json.loads(completed.stdout)["streams"][0]
    numerator, denominator = (int(part) for part in str(stream["avg_frame_rate"]).split("/"))
    return {
        "fps": numerator / denominator if denominator else 0.0,
        "frame_count": int(stream.get("nb_read_frames") or 0),
        "duration_seconds": float(stream.get("duration") or 0.0),
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
    }


def _source_pattern(frames: Iterable[VideoFrame]) -> tuple[Path, int, int]:
    ordered = list(frames)
    if not ordered:
        raise ValueError("No video frames available for MP4 conversion")
    parents = {frame.path.parent for frame in ordered}
    numbers = []
    for frame in ordered:
        match = FRAME_NAME.match(frame.path.name)
        if match is None:
            raise ValueError(f"Unsupported frame name for FFmpeg image sequence: {frame.path.name}")
        numbers.append(int(match.group(1)))
    if len(parents) != 1 or numbers != list(range(numbers[0], numbers[0] + len(numbers))):
        raise ValueError("Video frames are not one contiguous frame_%06d.jpg sequence")
    return next(iter(parents)), numbers[0], len(numbers)


def build_video_mp4s(
    config: ProjectConfig,
    participants: list[str] | None = None,
    *,
    force: bool = False,
    ffmpeg: str | None = None,
) -> dict[str, Any]:
    """Encode retained egocentric MP4 cache and write a verification manifest."""
    selected = participants or config.participants
    output_dir = config.path("video") / "mp4"
    output_dir.mkdir(parents=True, exist_ok=True)
    min_free = float(config.get("features.video.mp4.min_free_gb", 30.0))
    free_bytes = shutil.disk_usage(output_dir).free
    if free_bytes < min_free * 1024**3:
        raise RuntimeError(f"MP4 cache requires at least {min_free:.1f} GB free; only {free_bytes / 1024**3:.1f} GB available")
    ffmpeg_path = ffmpeg or _find_executable("ffmpeg")
    ffprobe_path = _find_executable("ffprobe")
    if not ffmpeg_path or not ffprobe_path:
        raise RuntimeError("FFmpeg and ffprobe are required. Activate rtml-p002-p016 or provide --ffmpeg.")
    width = int(config.get("features.video.mp4.width", 512))
    height = int(config.get("features.video.mp4.height", 288))
    fps = float(config.get("features.video.mp4.fps", 10.0))
    crf = int(config.get("features.video.mp4.crf", 18))
    source_by_participant = {row["participant_id"]: row for row in build_index(config, selected)}
    manifest: list[dict[str, Any]] = []
    for participant in selected:
        source = source_by_participant[participant]
        index = load_video_index(
            Path(source["video_frames_csv"]) if source.get("video_frames_csv") else None,
            Path(source["session_dir"]) if source.get("session_dir") else None,
            participant,
        )
        output = output_dir / f"{participant}.mp4"
        record: dict[str, Any] = {
            "participant_id": participant,
            "timestamp_source": index.timestamp_source,
            "index_reason": index.reason,
            "output_path": str(output),
            "source_frame_count": len(index.frames),
            "source_video_log_sha256": file_sha256(Path(source["video_frames_csv"])) if source.get("video_frames_csv") else None,
            "ffmpeg": ffmpeg_path,
            "ffmpeg_fps": fps,
            "ffmpeg_crf": crf,
        }
        if not index.frames:
            record["status"] = "skipped"
            manifest.append(record)
            continue
        parent, start_number, count = _source_pattern(index.frames)
        command = [
            ffmpeg_path, "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(fps),
            "-start_number", str(start_number), "-i", str(parent / "frame_%06d.jpg"), "-frames:v", str(count),
            "-vf", f"scale={width}:{height}:flags=lanczos", "-an", "-c:v", "libx264", "-preset", "medium",
            "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
        ]
        probe: dict[str, Any] | None = None
        if not force and output.exists():
            try:
                probe = _probe_video(ffprobe_path, output)
                if abs(probe["fps"] - fps) > 0.01 or probe["frame_count"] != count:
                    probe = None
            except RuntimeError:
                probe = None
        if force or probe is None:
            subprocess.run(command, check=True)
            record["status"] = "encoded"
        else:
            record["status"] = "reused"
        probe = _probe_video(ffprobe_path, output)
        if abs(probe["fps"] - fps) > 0.01 or probe["frame_count"] != count:
            raise RuntimeError(f"MP4 validation failed for {participant}: expected {count} frames @ {fps}, got {probe}")
        record.update(probe)
        record["mp4_size_bytes"] = output.stat().st_size
        record["mp4_sha256"] = file_sha256(output)
        record["ffmpeg_command"] = json.dumps(command, ensure_ascii=False)
        manifest.append(record)
    manifest_dir = config.path("video")
    write_rows(manifest_dir / "mp4_manifest.csv", manifest)
    write_json(manifest_dir / "mp4_manifest.json", {"retained": True, "rows": manifest})
    return {
        "participants": len(selected),
        "encoded": sum(row["status"] == "encoded" for row in manifest),
        "reused": sum(row["status"] == "reused" for row in manifest),
        "manifest": str(manifest_dir / "mp4_manifest.csv"),
    }

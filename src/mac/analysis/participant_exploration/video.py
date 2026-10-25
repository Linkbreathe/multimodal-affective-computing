"""Fixed classical optical flow. Frames are decoded in memory; no images are written."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from mac.data.video import VideoIndex
from .common import feature_result, weighted_mean_sd
from .specification import SPEC


def optical_flow_features(
    index: VideoIndex, start_ms: float, end_ms: float
) -> tuple[list[dict], list[dict], dict]:
    names = ["video_mean_optical_flow", "video_optical_flow_temporal_sd"]
    s = SPEC["video"]
    if not index.frames:
        return [feature_result(f, reason=index.reason) for f in names], [], {}
    timestamps = np.asarray([f.unix_time_ms for f in index.frames], dtype=float)
    positions = np.flatnonzero((timestamps >= start_ms) & (timestamps < end_ms))
    targets = np.arange(start_ms, end_ms, 1000 / s["sample_fps"])
    selected = []
    seen = set()
    for target in targets:
        if not len(positions):
            selected.append(None)
            continue
        k = int(np.searchsorted(timestamps[positions], target))
        choices = [int(positions[j]) for j in [k - 1, k] if 0 <= j < len(positions)]
        pos = min(choices, key=lambda j: abs(timestamps[j] - target))
        if pos in seen or abs(timestamps[pos] - target) > 1000 * s["target_tolerance_s"]:
            selected.append(None)
        else:
            seen.add(pos)
            selected.append(pos)
    values, weights, files = [], [], []
    previous = None
    missing = 0
    dimensions = set()
    for pos in selected:
        if pos is None:
            previous = None
            missing += 1
            continue
        frame = index.frames[pos]
        record = dict(
            path=str(frame.path),
            frame_index=frame.frame_index,
            unix_time_ms=frame.unix_time_ms,
            timestamp_source=index.timestamp_source,
            sha256="",
            bytes=np.nan,
            status="missing",
        )
        try:
            data = Path(frame.path).read_bytes()
            record.update(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
            image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError("decode_failed")
            dimensions.add((image.shape[1], image.shape[0]))
            small = cv2.resize(image, (s["width"], s["height"]), interpolation=cv2.INTER_AREA)
            record["status"] = "decoded"
        except (OSError, ValueError, cv2.error) as error:
            record["status"] = f"{type(error).__name__}: {error}"
            files.append(record)
            previous = None
            missing += 1
            continue
        files.append(record)
        if previous is not None:
            prev_pos, prev = previous
            dt = (timestamps[pos] - timestamps[prev_pos]) / 1000
            capture_dt = np.diff(timestamps[prev_pos : pos + 1]) / 1000
            if (
                0 < dt <= s["max_pair_gap_s"]
                and (capture_dt > 0).all()
                and (capture_dt <= s["max_capture_gap_s"]).all()
            ):
                flow = cv2.calcOpticalFlowFarneback(
                    prev,
                    small,
                    None,
                    s["pyr_scale"],
                    s["levels"],
                    s["winsize"],
                    s["iterations"],
                    s["poly_n"],
                    s["poly_sigma"],
                    s["flags"],
                )
                value = float(np.linalg.norm(flow, axis=2).mean() / dt)
                if np.isfinite(value):
                    values.append(value)
                    weights.append(dt)
        previous = pos, small
    duration = (end_ms - start_ms) / 1000
    valid_duration = float(sum(weights))
    quality = dict(
        target_duration_s=duration,
        valid_duration_s=valid_duration,
        valid_coverage=valid_duration / duration,
        target_frame_count=len(targets),
        decoded_frame_count=len(files) - sum(r["status"] != "decoded" for r in files),
        missing_target_frame_count=missing,
        valid_pair_count=len(values),
        timestamp_source=index.timestamp_source,
        flow_backend="opencv_cpu",
        source_resolutions=";".join(f"{w}x{h}" for w, h in sorted(dimensions)),
        warning="scene_and_head_motion_mixed",
    )
    good = valid_duration / duration >= SPEC["time"]["coverage_min"] and len(values) >= 2
    mean, sd = weighted_mean_sd(np.asarray(values), np.asarray(weights))
    return (
        [
            feature_result(
                n,
                v if good else np.nan,
                reason="" if good else "insufficient_valid_video_duration",
                **quality,
            )
            for n, v in zip(names, [mean, sd])
        ],
        files,
        quality,
    )

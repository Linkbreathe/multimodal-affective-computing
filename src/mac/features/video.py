from __future__ import annotations

from pathlib import Path

import numpy as np

from real_time_ml.features.common import robust_stats


def video_features(paths: list[Path]) -> tuple[dict[str, float], dict[str, float | bool]]:
    try:
        import cv2
    except ImportError:
        return {}, {"video_coverage": 0.0, "video_usable": False, "video_reason": "opencv_missing"}
    metrics: dict[str, list[float]] = {name: [] for name in ("brightness", "saturation", "colorfulness", "texture", "edge", "sharpness", "optical_flow", "scene_change")}
    previous_gray = None
    loaded = 0
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        loaded += 1
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        metrics["brightness"].append(float(np.mean(hsv[:, :, 2])))
        metrics["saturation"].append(float(np.mean(hsv[:, :, 1])))
        rg = frame[:, :, 2].astype(float) - frame[:, :, 1]
        yb = 0.5 * (frame[:, :, 2].astype(float) + frame[:, :, 1]) - frame[:, :, 0]
        metrics["colorfulness"].append(float(np.sqrt(np.var(rg) + np.var(yb)) + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)))
        metrics["texture"].append(float(np.std(gray)))
        metrics["edge"].append(float(np.mean(cv2.Canny(gray, 80, 160) > 0)))
        metrics["sharpness"].append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        if previous_gray is not None:
            small_prev = cv2.resize(previous_gray, (320, 180))
            small = cv2.resize(gray, (320, 180))
            flow = cv2.calcOpticalFlowFarneback(small_prev, small, None, 0.5, 2, 15, 2, 5, 1.1, 0)
            metrics["optical_flow"].append(float(np.mean(np.linalg.norm(flow, axis=2))))
            metrics["scene_change"].append(float(np.mean(cv2.absdiff(small_prev, small))))
        previous_gray = gray
    features: dict[str, float] = {}
    for name, values in metrics.items():
        features.update(robust_stats(np.asarray(values), f"video_{name}"))
    coverage = loaded / max(1, len(paths))
    usable = bool(loaded >= 5 and coverage >= 0.5)
    return features, {
        "video_coverage": float(coverage),
        "video_usable": usable,
        "video_reason": "ok" if usable else "insufficient_video_frames",
    }

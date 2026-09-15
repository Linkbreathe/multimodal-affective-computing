"""Build segment-to-label mapping for EmbeddingDataset."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.data.segments import SegmentExtractor

log = logging.getLogger(__name__)

EMOTIONS = ["Amused", "Content", "Excited", "Awe", "Neutral", "Fear", "Sad", "Disgust", "Anger"]

# CE manifest label encoding (emotion name -> integer label)
EMOTION_TO_LABEL: dict[str, int] = {
    "Amused": 0,
    "Content": 1,
    "Excited": 2,
    "Awe": 3,
    "Neutral": 4,
    "Fear": 5,
    "Sad": 6,
    "Disgust": 7,
    "Anger": 8,
}

# Session A Video Emotion column uses slightly different names in some cases
SESSION_A_EMOTION_MAP = {
    "Sadness": "Sad",
    "Angry": "Anger",
}

# Session B activity names (lowercase) that appear in task_times
SESSION_B_TASKS = {
    "trynottolaugh", "sadletter", "flappybird", "slenderman",
    "jellybean", "painting", "jenga",
}


def _load_session_label(
    subject_dir: Path, subject_id: str, task_name: str
) -> dict[str, Any] | None:
    """Load labels for a task from Session A or Session B CSV.

    Returns dict with emotion_label (int), soft_label (np.ndarray [9]),
    and vad (np.ndarray [3]), or None if not found.
    """
    task_lower = task_name.lower()

    if task_lower in SESSION_B_TASKS:
        return _lookup_session_b(subject_dir, subject_id, task_lower)
    elif task_name.startswith("video_"):
        emotion_suffix = task_name[len("video_"):]  # e.g. "Neutral", "Angry"
        return _lookup_session_a(subject_dir, subject_id, emotion_suffix)
    else:
        log.debug(f"Unknown task type '{task_name}' for subject {subject_id}")
        return None


def _lookup_session_a(
    subject_dir: Path, subject_id: str, emotion_suffix: str
) -> dict[str, Any] | None:
    csv_path = subject_dir / f"Session_A_{subject_id}.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    # Match row by Video Emotion column (case-insensitive)
    mask = df["Video Emotion"].str.lower() == emotion_suffix.lower()
    if mask.sum() == 0:
        return None
    row = df[mask].iloc[0]
    video_emotion = str(row["Video Emotion"])
    # Normalize to CE emotion name
    ce_emotion = SESSION_A_EMOTION_MAP.get(video_emotion, video_emotion)
    return _build_labels(row, ce_emotion)


def _lookup_session_b(
    subject_dir: Path, subject_id: str, task_lower: str
) -> dict[str, Any] | None:
    csv_path = subject_dir / f"Session_B_{subject_id}.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    # Match row by Activity Name (case-insensitive, ignoring spaces)
    df["_name_lower"] = df["Activity Name"].str.lower().str.replace(" ", "")
    mask = df["_name_lower"] == task_lower.replace(" ", "")
    if mask.sum() == 0:
        return None
    # If multiple rows (e.g. Jenga has 2), average soft labels and VAD
    rows = df[mask]
    soft_label = np.zeros(9, dtype=np.float32)
    for _, r in rows.iterrows():
        for i, e in enumerate(EMOTIONS):
            soft_label[i] += float(r.get(e, 0.0))
    soft_label /= len(rows)
    total = soft_label.sum()
    if total > 0:
        soft_label /= total

    vad = np.array([
        float(rows["Valence"].mean()),
        float(rows["Arousal"].mean()),
        float(rows["Dominance"].mean()),
    ], dtype=np.float32)

    # Determine dominant emotion label from soft_label
    dominant_idx = int(np.argmax(soft_label))
    emotion_name = EMOTIONS[dominant_idx]
    emotion_label = EMOTION_TO_LABEL[emotion_name]

    return {
        "emotion_label": emotion_label,
        "soft_label": soft_label,
        "vad": vad,
    }


def _build_labels(row: pd.Series, ce_emotion: str) -> dict[str, Any]:
    soft_label = np.array(
        [float(row.get(e, 0.0)) for e in EMOTIONS], dtype=np.float32
    )
    total = soft_label.sum()
    if total > 0:
        soft_label /= total

    vad = np.array([
        float(row.get("Valence", 0.0)),
        float(row.get("Arousal", 0.0)),
        float(row.get("Dominance", 0.0)),
    ], dtype=np.float32)

    emotion_label = EMOTION_TO_LABEL.get(ce_emotion, 4)  # default Neutral

    return {
        "emotion_label": emotion_label,
        "soft_label": soft_label,
        "vad": vad,
    }


def build_label_mapping(
    data_dir: str,
    task_times_path: str,
) -> dict[str, dict[str, Any]]:
    """Build a mapping from '{subject_id}_{segment_idx:04d}' to label tensors.

    Returns dict with keys like '005_0000' containing:
        - emotion_label: int tensor
        - soft_label: float tensor [9]
        - vad: float tensor [3]
    """
    data_path = Path(data_dir)
    extractor = SegmentExtractor(data_dir=data_dir, task_times_path=task_times_path)
    mapping: dict[str, dict[str, Any]] = {}

    for subject_id in extractor.get_subject_ids():
        subject_dir = data_path / subject_id
        segments = extractor.get_segments(subject_id)
        for idx, seg in enumerate(segments):
            labels = _load_session_label(subject_dir, subject_id, seg["task_name"])
            if labels is None:
                log.debug(
                    f"No labels for {subject_id}/{seg['task_name']}, skipping"
                )
                continue
            key = f"{subject_id}_{idx:04d}"
            mapping[key] = {
                "emotion_label": torch.tensor(labels["emotion_label"], dtype=torch.long),
                "soft_label": torch.tensor(labels["soft_label"], dtype=torch.float32),
                "vad": torch.tensor(labels["vad"], dtype=torch.float32),
            }

    log.info(f"Built label mapping: {len(mapping)} segments with labels")
    return mapping

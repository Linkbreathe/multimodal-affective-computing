"""Segment extraction from task_times.npy aligned to 90Hz eye-tracker timeline."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


class SegmentExtractor:
    """Extracts temporal segments per subject using task_times.npy."""

    SAMPLE_RATES = {
        "gaze_90fps.npy": 90,
        "pupils_90fps.npy": 90,
        "ppg_ear_125hz.npy": 125,
        "ppg_nose_125hz.npy": 125,
    }

    def __init__(self, data_dir: str, task_times_path: str) -> None:
        self.data_dir = Path(data_dir)
        self.task_times: dict[str, dict[str, list[int]]] = np.load(
            task_times_path, allow_pickle=True
        ).item()
        existing = {
            d.name
            for d in self.data_dir.iterdir()
            if d.is_dir() and d.name.isdigit()
        }
        self.subject_ids = sorted(existing & set(self.task_times.keys()))

    def get_subject_ids(self) -> list[str]:
        return list(self.subject_ids)

    def get_segments(self, subject_id: str) -> list[dict[str, Any]]:
        subject_tasks = self.task_times[subject_id]
        session_a_start = subject_tasks["session_A"][0]
        segments = []
        for task_name, (start, end) in subject_tasks.items():
            if task_name in ("session_A", "session_B"):
                continue
            segments.append(
                {
                    "task_name": task_name,
                    "start_idx": start - session_a_start,
                    "end_idx": end - session_a_start,
                    "subject_id": subject_id,
                }
            )
        return segments

    def load_signal(
        self, subject_id: str, filename: str, start_90hz: int, end_90hz: int
    ) -> np.ndarray:
        filepath = self.data_dir / subject_id / filename
        data = np.load(filepath)
        if data.ndim == 1:
            data = data[:, np.newaxis]

        native_rate = self.SAMPLE_RATES.get(filename, 90)
        if native_rate != 90:
            start = int(start_90hz * native_rate / 90)
            end = int(end_90hz * native_rate / 90)
        else:
            start, end = start_90hz, end_90hz

        return data[start:end]

    def load_ppg_segment(
        self, subject_id: str, start_90hz: int, end_90hz: int
    ) -> np.ndarray:
        return self.load_signal(subject_id, "ppg_ear_125hz.npy", start_90hz, end_90hz)

    def load_eye_tracking_segment(
        self, subject_id: str, start_90hz: int, end_90hz: int
    ) -> np.ndarray:
        gaze = self.load_signal(subject_id, "gaze_90fps.npy", start_90hz, end_90hz)
        pupils = self.load_signal(subject_id, "pupils_90fps.npy", start_90hz, end_90hz)
        min_len = min(len(gaze), len(pupils))
        return np.concatenate([gaze[:min_len], pupils[:min_len]], axis=1)

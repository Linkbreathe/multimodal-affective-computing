"""Segment extraction from task_times.npy aligned to 90Hz eye-tracker timeline."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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

    @staticmethod
    def _canonicalize_encoder_name(encoder_name: str) -> str:
        return "".join(ch for ch in encoder_name.lower() if ch.isalnum())

    def load_eye_tracking_segment(
        self,
        subject_id: str,
        start_90hz: int,
        end_90hz: int,
        encoder_name: str = "patchtst",
    ) -> np.ndarray:
        gaze = self.load_signal(subject_id, "gaze_90fps.npy", start_90hz, end_90hz)
        encoder_key = self._canonicalize_encoder_name(encoder_name)
        if encoder_key == "inceptiontime":
            return gaze[:, :2]
        if encoder_key != "patchtst":
            raise ValueError(
                f"Unsupported eye-tracking encoder '{encoder_name}'. "
                "Expected 'inceptiontime' or 'patchtst'."
            )

        pupils = self.load_signal(subject_id, "pupils_90fps.npy", start_90hz, end_90hz)
        min_len = min(len(gaze), len(pupils))
        return np.concatenate([gaze[:min_len], pupils[:min_len]], axis=1)


class LabelLoader:
    """Loads emotion labels from manifest CSVs."""

    EMOTIONS = [
        "Amused", "Content", "Excited", "Awe", "Neutral",
        "Fear", "Sad", "Disgust", "Anger",
    ]

    def __init__(self, data_dir: str) -> None:
        self.data_dir = Path(data_dir)
        self._ce_cache: pd.DataFrame | None = None
        self._kl_cache: pd.DataFrame | None = None
        self._vad_cache: pd.DataFrame | None = None

    def load_ce_manifest(self) -> pd.DataFrame:
        if self._ce_cache is None:
            path = self.data_dir / "ce_hardlabel_manifests" / "dataset_manifest.csv"
            self._ce_cache = pd.read_csv(path)
        return self._ce_cache

    def load_kl_manifest(self) -> pd.DataFrame:
        if self._kl_cache is None:
            path = self.data_dir / "kl_softlabel_manifests" / "dataset_manifest.csv"
            self._kl_cache = pd.read_csv(path)
        return self._kl_cache

    def load_vad_manifest(self) -> pd.DataFrame:
        if self._vad_cache is None:
            path = self.data_dir / "vad_binary_quadrant_manifests" / "dataset_manifest.csv"
            self._vad_cache = pd.read_csv(path)
        return self._vad_cache

    def get_segment_labels(
        self, subject_id: str, task_name: str
    ) -> dict[str, Any] | None:
        """Get all label types for a segment. Returns None if not found.

        Matches by subject_id AND task_name in segment path.
        Returns emotion_label, soft_label (9-dim), and vad (3-dim).
        """
        ce = self.load_ce_manifest()
        kl = self.load_kl_manifest()
        vad = self.load_vad_manifest()

        subj_str = str(subject_id).zfill(3)
        ce_mask = (
            (ce["subject"].astype(str).str.zfill(3) == subj_str)
            & ce["segment_path"].str.contains(task_name, na=False)
        )
        if ce_mask.sum() == 0:
            return None

        ce_row = ce[ce_mask].iloc[0]

        kl_mask = (
            (kl["subject"].astype(str).str.zfill(3) == subj_str)
            & kl["segment_path"].str.contains(task_name, na=False)
        )
        soft_label = np.zeros(9, dtype=np.float32)
        if kl_mask.sum() > 0:
            kl_row = kl[kl_mask].iloc[0]
            soft_label = np.array(
                [kl_row[e] for e in self.EMOTIONS], dtype=np.float32
            )
            total = soft_label.sum()
            if total > 0:
                soft_label /= total

        vad_scores = np.zeros(3, dtype=np.float32)
        vad_manifest = vad
        vad_mask_filter = vad_manifest["subject"].astype(str).str.zfill(3) == subj_str
        if "segment_path" in vad_manifest.columns:
            vad_mask_filter = vad_mask_filter & vad_manifest["segment_path"].str.contains(task_name, na=False)
        if vad_mask_filter.sum() > 0:
            vad_row = vad_manifest[vad_mask_filter].iloc[0]
            vad_scores = np.array([
                float(vad_row.get("valence_score", 0)),
                float(vad_row.get("arousal_score", 0)),
                float(vad_row.get("dominance_score", 0)),
            ], dtype=np.float32)

        return {
            "emotion_label": int(ce_row["label"]),
            "emotion_name": ce_row["emotion"],
            "soft_label": soft_label,
            "vad": vad_scores,
        }

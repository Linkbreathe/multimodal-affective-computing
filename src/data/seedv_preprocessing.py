"""
Preprocess SEED-V raw EEG (.cnt) files for the EEGPT encoder.

Pipeline (aligned with EEG-FM-Bench):
  1. Parse per-session trial timestamps from the official timestamp file.
  2. Load each .cnt file with MNE, pick 60 EEG channels (10-10 montage,
     excluding CB1/CB2), bandpass 0.1-100 Hz, notch 50 Hz, resample to 256 Hz.
  3. Segment each trial into non-overlapping 10-second windows (60 x 2560).
  4. Save each segment as a .pt dict and write a manifest CSV.

Usage:
    python -m src.data.seedv_preprocessing
"""

from __future__ import annotations

import gc
import logging
import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Emotion label definitions
# ---------------------------------------------------------------------------

EMOTION_NAMES = ["Disgust", "Fear", "Sad", "Neutral", "Happy"]
EMOTION_MAP = {"Disgust": 0, "Fear": 1, "Sad": 2, "Neutral": 3, "Happy": 4}

SESSION_LABELS: dict[int, list[str]] = {
    1: [
        "Happy", "Fear", "Neutral", "Sad", "Disgust",
        "Happy", "Fear", "Neutral", "Sad", "Disgust",
        "Happy", "Fear", "Neutral", "Sad", "Disgust",
    ],
    2: [
        "Sad", "Fear", "Neutral", "Disgust", "Happy",
        "Happy", "Disgust", "Neutral", "Sad", "Fear",
        "Neutral", "Happy", "Fear", "Sad", "Disgust",
    ],
    3: [
        "Sad", "Fear", "Neutral", "Disgust", "Happy",
        "Happy", "Disgust", "Neutral", "Sad", "Fear",
        "Neutral", "Happy", "Fear", "Sad", "Disgust",
    ],
}

# Channels to drop (non-EEG reference / EOG channels)
DROP_CHANNELS = ["M1", "M2", "VEO", "HEO"]

# Canonical SEED-V 60-channel order (10-10 montage, aligned with EEG-FM-Bench).
# CB1 and CB2 are excluded to match EEG-FM-Bench's channel definition.
SEEDV_60_CHANNELS = [
    "FP1", "FPZ", "FP2", "AF3", "AF4",
    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8",
    "O1", "OZ", "O2",
]


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------

def parse_trial_timestamps(timestamp_path: str | Path) -> dict[int, list[tuple[int, int]]]:
    """Parse the SEED-V trial_start_end_timestamp.txt file.

    Returns
    -------
    dict mapping session_id (1-indexed) to a list of (start_sec, end_sec) tuples,
    one per trial.
    """
    timestamp_path = Path(timestamp_path)
    text = timestamp_path.read_text()

    sessions: dict[int, list[tuple[int, int]]] = {}
    current_session: int | None = None
    starts: list[int] | None = None
    ends: list[int] | None = None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Match "Session N:"
        session_match = re.match(r"Session\s+(\d+)\s*:", line)
        if session_match:
            current_session = int(session_match.group(1))
            starts = None
            ends = None
            continue

        if current_session is None:
            continue

        if line.startswith("start_second"):
            nums = re.findall(r"\d+", line)
            starts = [int(n) for n in nums]
        elif line.startswith("end_second"):
            nums = re.findall(r"\d+", line)
            ends = [int(n) for n in nums]

        if starts is not None and ends is not None:
            if len(starts) != len(ends):
                raise ValueError(
                    f"Session {current_session}: mismatched start/end counts "
                    f"({len(starts)} vs {len(ends)})"
                )
            sessions[current_session] = list(zip(starts, ends))
            starts = None
            ends = None

    logger.info("Parsed timestamps for %d sessions", len(sessions))
    return sessions


# ---------------------------------------------------------------------------
# .cnt file enumeration
# ---------------------------------------------------------------------------

def enumerate_cnt_files(data_dir: str | Path) -> list[tuple[Path, int, int]]:
    """Find all .cnt files and parse (subject_id, session_id) from filenames.

    Skips files containing 'repaired' in the name.

    Returns
    -------
    List of (path, subject_id, session_id) sorted by (subject, session).
    """
    eeg_dir = Path(data_dir) / "EEG_raw"
    cnt_files = sorted(eeg_dir.glob("*.cnt"))

    results: list[tuple[Path, int, int]] = []
    # First pass: collect all non-repaired files
    repaired: dict[tuple[int, int], Path] = {}
    regular: dict[tuple[int, int], Path] = {}

    for p in cnt_files:
        parts = p.stem.split("_")
        if len(parts) < 2:
            logger.warning("Unexpected filename format, skipping: %s", p.name)
            continue
        try:
            subject_id = int(parts[0])
            session_id = int(parts[1])
        except ValueError:
            logger.warning("Cannot parse subject/session from: %s", p.name)
            continue

        if "repaired" in p.name.lower():
            repaired[(subject_id, session_id)] = p
        else:
            regular[(subject_id, session_id)] = p

    # Prefer repaired version when available
    for key in sorted(regular.keys() | repaired.keys()):
        if key in repaired:
            logger.info("Using repaired file for subject %d session %d", *key)
            results.append((repaired[key], *key))
        else:
            results.append((regular[key], *key))

    results.sort(key=lambda x: (x[1], x[2]))
    logger.info("Found %d .cnt files to process", len(results))
    return results


# ---------------------------------------------------------------------------
# Core preprocessing
# ---------------------------------------------------------------------------

def process_cnt_file(
    cnt_path: Path,
    subject_id: int,
    session_id: int,
    trial_timestamps: list[tuple[int, int]],
    trial_labels: list[str],
    output_dir: Path,
    target_sr: int = 256,
    window_sec: int = 10,
    global_idx_start: int = 0,
) -> tuple[int, list[dict]]:
    """Process a single .cnt file: load, filter, resample, segment, save.

    Parameters
    ----------
    cnt_path : Path
        Path to the .cnt file.
    subject_id, session_id : int
        Subject and session identifiers.
    trial_timestamps : list of (start_sec, end_sec)
        Trial boundaries in seconds (from timestamp file).
    trial_labels : list of str
        Emotion name per trial.
    output_dir : Path
        Root output directory.
    target_sr : int
        Target sampling rate in Hz.
    window_sec : int
        Window duration in seconds.
    global_idx_start : int
        Starting global segment index for this subject.

    Returns
    -------
    (next_global_idx, manifest_rows) where manifest_rows is a list of dicts.
    """
    samples_per_window = target_sr * window_sec
    subject_dir = output_dir / str(subject_id)
    subject_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Loading %s (subject=%d, session=%d)", cnt_path.name, subject_id, session_id
    )

    # Load and preprocess
    try:
        raw = mne.io.read_raw_cnt(str(cnt_path), preload=True)
    except (RuntimeError, OverflowError) as e:
        logger.warning("Failed to load %s: %s — skipping", cnt_path.name, e)
        return global_idx_start, []

    # Drop non-EEG channels that exist in this recording
    channels_to_drop = [ch for ch in DROP_CHANNELS if ch in raw.ch_names]
    if channels_to_drop:
        raw.drop_channels(channels_to_drop)

    # Reorder to canonical 60-channel order for consistent indexing
    available = set(raw.ch_names)
    missing = [ch for ch in SEEDV_60_CHANNELS if ch not in available]
    if missing:
        logger.warning(
            "Missing EEG channels in %s: %s — skipping this file",
            cnt_path.name, missing,
        )
        del raw
        return global_idx_start, []
    raw.reorder_channels(SEEDV_60_CHANNELS)

    logger.info("Channels after reordering: %d", len(raw.ch_names))

    # Bandpass filter (0.1-100 Hz, aligned with EEG-FM-Bench)
    raw.filter(0.1, 100.0, verbose=False)

    # Notch filter at 50 Hz harmonics (aligned with EEG-FM-Bench)
    orig_fs = raw.info["sfreq"]
    notch_freqs = np.arange(50.0, orig_fs / 2, 50.0).tolist()
    raw.notch_filter(freqs=notch_freqs, verbose=False)

    # Resample
    raw.resample(target_sr, verbose=False)

    # Get numpy data: shape [n_channels, n_samples]
    # MNE returns Volts (SI); convert to microvolts for downstream models
    # EEGPT expects µV (internally scales ×0.001→mV), EEGNet assumes µV-scale
    data = raw.get_data() * 1e6
    n_channels = data.shape[0]
    logger.info(
        "Data shape after resampling: [%d, %d] (sr=%d Hz)",
        n_channels, data.shape[1], target_sr,
    )

    # Free MNE object
    del raw

    global_idx = global_idx_start
    manifest_rows: list[dict] = []

    for trial_idx, ((start_sec, end_sec), emotion_name) in enumerate(
        zip(trial_timestamps, trial_labels)
    ):
        start_sample = start_sec * target_sr
        end_sample = end_sec * target_sr

        # Clamp to data bounds
        end_sample = min(end_sample, data.shape[1])
        if start_sample >= data.shape[1]:
            logger.warning(
                "Trial %d start (%d s) is beyond data length, skipping",
                trial_idx, start_sec,
            )
            continue

        trial_data = data[:, start_sample:end_sample]
        n_trial_samples = trial_data.shape[1]
        n_segments = n_trial_samples // samples_per_window

        if n_segments == 0:
            logger.warning(
                "Trial %d too short for a single window (%d samples), skipping",
                trial_idx, n_trial_samples,
            )
            continue

        emotion_label = EMOTION_MAP[emotion_name]

        for seg_idx in range(n_segments):
            seg_start = seg_idx * samples_per_window
            seg_end = seg_start + samples_per_window
            segment = trial_data[:, seg_start:seg_end]

            tensor = torch.from_numpy(segment.astype(np.float32))  # [60, 2560]

            save_path = subject_dir / f"seg_{global_idx:05d}.pt"
            torch.save(
                {
                    "eeg": tensor,
                    "label": emotion_label,
                    "subject": subject_id,
                    "session": session_id,
                    "trial": trial_idx,
                    "segment_in_trial": seg_idx,
                },
                save_path,
            )

            manifest_rows.append(
                {
                    "subject": subject_id,
                    "session": session_id,
                    "trial": trial_idx,
                    "segment_idx": seg_idx,
                    "emotion_label": emotion_label,
                    "emotion_name": emotion_name,
                    "file_path": str(save_path),
                }
            )
            global_idx += 1

    logger.info(
        "Subject %d session %d: saved %d segments",
        subject_id, session_id, global_idx - global_idx_start,
    )
    return global_idx, manifest_rows


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def preprocess_seedv(
    data_dir: str | Path = "/mnt/c/Users/Public/Data/SEED-V/SEED-V",
    output_dir: str | Path = "data/preprocessed/seedv_preprocessed",
    target_sr: int = 256,
    window_sec: int = 10,
) -> pd.DataFrame:
    """Run the full SEED-V preprocessing pipeline.

    Parameters
    ----------
    data_dir : str or Path
        Path to the SEED-V root directory containing ``EEG_raw/`` and
        ``trial_start_end_timestamp.txt``.
    output_dir : str or Path
        Directory where preprocessed ``.pt`` files and the manifest CSV
        will be saved.
    target_sr : int
        Target sampling rate in Hz (default 256).
    window_sec : int
        Non-overlapping window duration in seconds (default 10).

    Returns
    -------
    pd.DataFrame
        The manifest dataframe with one row per saved segment.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse timestamps
    timestamp_path = data_dir / "trial_start_end_timestamp.txt"
    session_timestamps = parse_trial_timestamps(timestamp_path)

    # 2. Enumerate .cnt files
    cnt_files = enumerate_cnt_files(data_dir)
    if not cnt_files:
        logger.error("No .cnt files found in %s/EEG_raw/", data_dir)
        return pd.DataFrame()

    # 3. Process each file
    all_manifest_rows: list[dict] = []
    # Track per-subject global indices so segment numbering is unique per subject
    subject_global_idx: dict[int, int] = {}

    for cnt_path, subject_id, session_id in tqdm(cnt_files, desc="Processing .cnt files"):
        if session_id not in session_timestamps:
            logger.warning("No timestamps for session %d, skipping %s", session_id, cnt_path.name)
            continue
        if session_id not in SESSION_LABELS:
            logger.warning("No labels for session %d, skipping %s", session_id, cnt_path.name)
            continue

        trial_ts = session_timestamps[session_id]
        trial_labels = SESSION_LABELS[session_id]

        if len(trial_ts) != len(trial_labels):
            logger.error(
                "Timestamp/label count mismatch for session %d: %d vs %d",
                session_id, len(trial_ts), len(trial_labels),
            )
            continue

        start_idx = subject_global_idx.get(subject_id, 0)

        next_idx, rows = process_cnt_file(
            cnt_path=cnt_path,
            subject_id=subject_id,
            session_id=session_id,
            trial_timestamps=trial_ts,
            trial_labels=trial_labels,
            output_dir=output_dir,
            target_sr=target_sr,
            window_sec=window_sec,
            global_idx_start=start_idx,
        )

        subject_global_idx[subject_id] = next_idx
        all_manifest_rows.extend(rows)

        # Free memory
        gc.collect()

    # 4. Save manifest
    manifest = pd.DataFrame(all_manifest_rows)
    manifest_path = output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    logger.info("Manifest saved to %s (%d rows)", manifest_path, len(manifest))

    return manifest


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Suppress verbose MNE output
    mne.set_log_level("WARNING")

    preprocess_seedv()

"""Task-aware 10s segmentation ONLY (no embedding extraction).

Produces manifest.csv and per-subject chunk JSONs for verification.
Reuses the same segmentation logic as segment_and_extract_10s.py.

Usage:
    conda run -n visphy python scripts/segment_only_10s.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CHUNK_LEN_SEC = 10
FS_ET = 90
CHUNK_SAMPLES_90HZ = CHUNK_LEN_SEC * FS_ET  # 900

EMOTIONS = ["Amused", "Content", "Excited", "Awe", "Neutral", "Fear", "Sad", "Disgust", "Anger"]
EMOTION_TO_LABEL = {e: i for i, e in enumerate(EMOTIONS)}

SESSION_B_TASKS = {"trynottolaugh", "sadletter", "flappybird", "slenderman", "jellybean", "painting", "jenga"}

SESSION_A_EMOTION_MAP = {
    "Sadness": "Sad",
    "Angry": "Anger",
}


# ---------------------------------------------------------------------------
# Label loading (copied from segment_and_extract_10s.py for standalone use)
# ---------------------------------------------------------------------------

def load_task_label(subject_dir: Path, subject_id: str, task_name: str) -> dict | None:
    task_lower = task_name.lower()
    if task_lower in SESSION_B_TASKS:
        return _lookup_session_b(subject_dir, subject_id, task_lower)
    elif task_name.startswith("video_"):
        emotion_suffix = task_name[len("video_"):]
        return _lookup_session_a(subject_dir, subject_id, emotion_suffix)
    return None


def _lookup_session_a(subject_dir: Path, subject_id: str, emotion_suffix: str) -> dict | None:
    csv_path = subject_dir / f"Session_A_{subject_id}.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    mask = df["Video Emotion"].str.lower() == emotion_suffix.lower()
    if mask.sum() == 0:
        return None
    row = df[mask].iloc[0]
    video_emotion = str(row["Video Emotion"])
    ce_emotion = SESSION_A_EMOTION_MAP.get(video_emotion, video_emotion)

    soft_label = np.array([float(row.get(e, 0.0)) for e in EMOTIONS], dtype=np.float32)
    total = soft_label.sum()
    if total > 0:
        soft_label /= total

    emotion_label = EMOTION_TO_LABEL.get(ce_emotion, 4)

    return {
        "emotion_label": emotion_label,
        "emotion_name": ce_emotion,
        "soft_label": soft_label,
        "valence": float(row.get("Valence", 0.0)),
        "arousal": float(row.get("Arousal", 0.0)),
        "dominance": float(row.get("Dominance", 0.0)),
    }


def _lookup_session_b(subject_dir: Path, subject_id: str, task_lower: str) -> dict | None:
    csv_path = subject_dir / f"Session_B_{subject_id}.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df["_name_lower"] = df["Activity Name"].str.lower().str.replace(" ", "")
    mask = df["_name_lower"] == task_lower.replace(" ", "")
    if mask.sum() == 0:
        return None
    rows = df[mask]

    soft_label = np.zeros(9, dtype=np.float32)
    for _, r in rows.iterrows():
        for i, e in enumerate(EMOTIONS):
            soft_label[i] += float(r.get(e, 0.0))
    soft_label /= len(rows)
    total = soft_label.sum()
    if total > 0:
        soft_label /= total

    dominant_idx = int(np.argmax(soft_label))
    emotion_name = EMOTIONS[dominant_idx]
    emotion_label = EMOTION_TO_LABEL[emotion_name]

    return {
        "emotion_label": emotion_label,
        "emotion_name": emotion_name,
        "soft_label": soft_label,
        "valence": float(rows["Valence"].mean()),
        "arousal": float(rows["Arousal"].mean()),
        "dominance": float(rows["Dominance"].mean()),
    }


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def get_task_chunks(subject_tasks: dict, subject_dir: Path, subject_id: str) -> list[dict]:
    session_a_start = subject_tasks["session_A"][0]
    chunks = []

    for task_name, (start_raw, end_raw) in subject_tasks.items():
        if task_name in ("session_A", "session_B"):
            continue

        label_info = load_task_label(subject_dir, subject_id, task_name)
        if label_info is None:
            log.debug(f"No label for {subject_id}/{task_name}, skipping")
            continue

        shifted_start = start_raw - session_a_start
        shifted_end = end_raw - session_a_start

        task_samples = shifted_end - shifted_start
        n_chunks = task_samples // CHUNK_SAMPLES_90HZ

        for ci in range(n_chunks):
            chunk_start_90 = shifted_start + ci * CHUNK_SAMPLES_90HZ
            chunk_end_90 = chunk_start_90 + CHUNK_SAMPLES_90HZ
            chunks.append({
                "task_name": task_name,
                "chunk_idx_in_task": ci,
                "start_90hz": int(chunk_start_90),
                "end_90hz": int(chunk_end_90),
                "task_total_samples": int(task_samples),
                "task_total_chunks": int(n_chunks),
                "task_discarded_samples": int(task_samples - n_chunks * CHUNK_SAMPLES_90HZ),
                **label_info,
            })

    return chunks


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_no_overlap(chunks: list[dict]) -> list[str]:
    errors = []
    sorted_chunks = sorted(chunks, key=lambda c: c["start_90hz"])
    for i in range(1, len(sorted_chunks)):
        prev = sorted_chunks[i - 1]
        curr = sorted_chunks[i]
        if curr["start_90hz"] < prev["end_90hz"]:
            errors.append(
                f"Overlap: chunk ending at {prev['end_90hz']} overlaps "
                f"chunk starting at {curr['start_90hz']} "
                f"(tasks: {prev['task_name']} -> {curr['task_name']})"
            )
    return errors


def validate_boundaries(chunks: list[dict], signal_len_90hz: int) -> list[str]:
    errors = []
    for c in chunks:
        if c["start_90hz"] < 0:
            errors.append(f"Negative start: {c['start_90hz']} in {c['task_name']}")
        if c["end_90hz"] > signal_len_90hz:
            errors.append(
                f"Boundary exceeded: end={c['end_90hz']} > signal_len={signal_len_90hz} "
                f"in {c['task_name']}"
            )
        if c["end_90hz"] - c["start_90hz"] != CHUNK_SAMPLES_90HZ:
            errors.append(
                f"Wrong chunk size: {c['end_90hz'] - c['start_90hz']} != {CHUNK_SAMPLES_90HZ} "
                f"in {c['task_name']}"
            )
    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = load_config("configs/base.yaml")
    data_dir = Path(cfg["data_dir"])
    output_dir = Path("data/preprocessed/segmentation_10s_task_aware")

    tt = np.load(data_dir / "task_times.npy", allow_pickle=True).item()

    subject_ids = sorted(
        d.name for d in data_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and d.name in tt
    )
    log.info(f"Found {len(subject_ids)} subjects")

    # Build chunks for all subjects
    chunks_by_subject: dict[str, list[dict]] = {}
    manifest_rows = []
    all_validation_errors = []

    summary_stats = {
        "total_subjects": 0,
        "total_tasks": 0,
        "total_chunks": 0,
        "total_discarded_samples": 0,
        "emotion_distribution": {e: 0 for e in EMOTIONS},
        "chunks_per_subject": {},
        "tasks_skipped_no_label": 0,
    }

    for subj in subject_ids:
        subject_tasks = tt[subj]
        subject_dir = data_dir / subj

        # Count tasks that have no labels
        tasks_with_labels = 0
        tasks_without_labels = 0
        for task_name in subject_tasks:
            if task_name in ("session_A", "session_B"):
                continue
            label = load_task_label(subject_dir, subj, task_name)
            if label is None:
                tasks_without_labels += 1
            else:
                tasks_with_labels += 1

        task_chunks = get_task_chunks(subject_tasks, subject_dir, subj)

        # Assign global_seq
        for seq_idx, c in enumerate(task_chunks):
            c["global_seq"] = seq_idx

        chunks_by_subject[subj] = task_chunks

        # Validate for this subject
        gaze_path = data_dir / subj / "gaze_90fps.npy"
        if gaze_path.exists():
            gaze_len = len(np.load(gaze_path))
            boundary_errors = validate_boundaries(task_chunks, gaze_len)
            if boundary_errors:
                all_validation_errors.extend(
                    [f"[{subj}] {e}" for e in boundary_errors]
                )

        overlap_errors = validate_no_overlap(task_chunks)
        if overlap_errors:
            all_validation_errors.extend(
                [f"[{subj}] {e}" for e in overlap_errors]
            )

        # Build manifest rows
        for c in task_chunks:
            soft_str = ",".join(f"{v:.6f}" for v in c["soft_label"])
            manifest_rows.append({
                "subject": subj,
                "task_name": c["task_name"],
                "chunk_idx_in_task": c["chunk_idx_in_task"],
                "global_seq": c["global_seq"],
                "start_90hz": c["start_90hz"],
                "end_90hz": c["end_90hz"],
                "emotion_label": c["emotion_label"],
                "emotion_name": c["emotion_name"],
                "soft_label": soft_str,
                "valence": c["valence"],
                "arousal": c["arousal"],
                "dominance": c["dominance"],
                "task_total_samples": c["task_total_samples"],
                "task_total_chunks": c["task_total_chunks"],
                "task_discarded_samples": c["task_discarded_samples"],
            })

        # Update summary
        summary_stats["total_subjects"] += 1
        summary_stats["total_tasks"] += tasks_with_labels
        summary_stats["tasks_skipped_no_label"] += tasks_without_labels
        summary_stats["total_chunks"] += len(task_chunks)
        summary_stats["chunks_per_subject"][subj] = len(task_chunks)
        for c in task_chunks:
            summary_stats["emotion_distribution"][c["emotion_name"]] += 1
            summary_stats["total_discarded_samples"] += c["task_discarded_samples"] // c["task_total_chunks"] if c["task_total_chunks"] > 0 else 0

    # Save manifest CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(manifest_path, index=False)
    log.info(f"Manifest saved: {manifest_path} ({len(manifest_df)} rows)")

    # Save per-subject chunk details as JSON for verification
    details_dir = output_dir / "subject_details"
    details_dir.mkdir(parents=True, exist_ok=True)
    for subj, chunks in chunks_by_subject.items():
        detail = []
        for c in chunks:
            detail.append({
                "task_name": c["task_name"],
                "chunk_idx_in_task": c["chunk_idx_in_task"],
                "global_seq": c["global_seq"],
                "start_90hz": c["start_90hz"],
                "end_90hz": c["end_90hz"],
                "duration_sec": (c["end_90hz"] - c["start_90hz"]) / 90.0,
                "emotion_label": c["emotion_label"],
                "emotion_name": c["emotion_name"],
                "task_total_samples": c["task_total_samples"],
                "task_total_chunks": c["task_total_chunks"],
                "task_discarded_samples": c["task_discarded_samples"],
            })
        with open(details_dir / f"{subj}.json", "w") as f:
            json.dump(detail, f, indent=2)

    # Save summary
    summary_stats["total_discarded_sec"] = round(
        summary_stats["total_discarded_samples"] / 90.0, 2
    )
    summary_stats["validation_errors"] = all_validation_errors
    summary_stats["validation_passed"] = len(all_validation_errors) == 0

    with open(output_dir / "segmentation_summary.json", "w") as f:
        json.dump(summary_stats, f, indent=2, default=str)

    # Print report
    log.info("=" * 60)
    log.info("SEGMENTATION REPORT")
    log.info("=" * 60)
    log.info(f"Subjects:        {summary_stats['total_subjects']}")
    log.info(f"Labeled tasks:   {summary_stats['total_tasks']}")
    log.info(f"Skipped (no label): {summary_stats['tasks_skipped_no_label']}")
    log.info(f"Total 10s chunks: {summary_stats['total_chunks']}")
    log.info(f"Discarded time:  {summary_stats['total_discarded_sec']}s")
    log.info("-" * 40)
    log.info("Emotion distribution:")
    for emo, count in sorted(summary_stats["emotion_distribution"].items(), key=lambda x: -x[1]):
        log.info(f"  {emo:12s}: {count:4d} chunks")
    log.info("-" * 40)
    log.info(f"Chunks per subject: min={min(summary_stats['chunks_per_subject'].values())}, "
             f"max={max(summary_stats['chunks_per_subject'].values())}, "
             f"mean={np.mean(list(summary_stats['chunks_per_subject'].values())):.1f}")
    log.info("-" * 40)

    if all_validation_errors:
        log.error(f"VALIDATION FAILED: {len(all_validation_errors)} errors")
        for err in all_validation_errors[:20]:
            log.error(f"  {err}")
    else:
        log.info("VALIDATION PASSED: no overlaps, no boundary violations, all chunks = 900 samples")

    log.info("=" * 60)
    log.info(f"Output directory: {output_dir}")
    log.info(f"  manifest.csv              ({len(manifest_df)} rows)")
    log.info(f"  segmentation_summary.json (stats + validation)")
    log.info(f"  subject_details/*.json    ({len(chunks_by_subject)} files)")


if __name__ == "__main__":
    main()

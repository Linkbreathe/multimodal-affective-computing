#!/usr/bin/env python3
"""Verify task-aware segmentation logic on one real subject from egoEMOTION.

Runs the chunking logic from segment_and_extract_10s.py on a single subject
and prints diagnostic tables to confirm correctness.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from scripts.segment_and_extract_10s import (
    CHUNK_SAMPLES_90HZ,
    EMOTIONS,
    get_task_chunks,
    load_task_label,
)
from src.utils.config import load_config

# ── Resolve data directory ──────────────────────────────────────────────
cfg = load_config("configs/base.yaml")
data_dir = Path(cfg["data_dir"])
if not data_dir.exists():
    data_dir = Path(cfg.get("ref_data_dir", ""))
assert data_dir.exists(), f"Data dir not found: {data_dir}"

tt = np.load(data_dir / "task_times.npy", allow_pickle=True).item()

# ── Pick first subject ──────────────────────────────────────────────────
subject_ids = sorted(d.name for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit() and d.name in tt)
SUBJ = subject_ids[0]
subject_tasks = tt[SUBJ]
subject_dir = data_dir / SUBJ
session_a_start = subject_tasks["session_A"][0]
session_a_end = subject_tasks["session_A"][1]
session_b_start = subject_tasks["session_B"][0]
session_b_end = subject_tasks["session_B"][1]

print(f"{'='*80}")
print(f"SUBJECT: {SUBJ}")
print(f"Session A range: {subject_tasks['session_A']}  (shifted 0 .. {session_a_end - session_a_start})")
print(f"Session B range: {subject_tasks['session_B']}  (shifted {session_b_start - session_a_start} .. {session_b_end - session_a_start})")
print(f"{'='*80}\n")

# ── (a) Task table ──────────────────────────────────────────────────────
print("─── Per-task chunking table ───")
header = f"{'task':<20} {'raw_start':>10} {'raw_end':>10} {'sh_start':>10} {'sh_end':>10} {'dur_sec':>8} {'n_chunks':>8}"
print(header)
print("─" * len(header))

task_rows = []
total_task_samples = 0
total_chunks_manual = 0

for task_name, (start_raw, end_raw) in subject_tasks.items():
    if task_name in ("session_A", "session_B"):
        continue
    shifted_start = start_raw - session_a_start
    shifted_end = end_raw - session_a_start
    task_samples = shifted_end - shifted_start
    n_chunks = task_samples // CHUNK_SAMPLES_90HZ
    dur_sec = task_samples / 90.0

    total_task_samples += task_samples
    total_chunks_manual += n_chunks

    task_rows.append({
        "task": task_name,
        "raw_start": start_raw,
        "raw_end": end_raw,
        "shifted_start": shifted_start,
        "shifted_end": shifted_end,
        "task_samples": task_samples,
        "dur_sec": dur_sec,
        "n_chunks": n_chunks,
    })
    print(f"{task_name:<20} {start_raw:>10} {end_raw:>10} {shifted_start:>10} {shifted_end:>10} {dur_sec:>8.1f} {n_chunks:>8}")

print(f"\nTotal tasks: {len(task_rows)}")
print(f"Total manual chunk count: {total_chunks_manual}")

# ── (b) get_task_chunks output ──────────────────────────────────────────
chunks = get_task_chunks(subject_tasks, subject_dir, SUBJ)
# assign global_seq (the main script does this)
for seq_idx, c in enumerate(chunks):
    c["global_seq"] = seq_idx

print(f"get_task_chunks returned: {len(chunks)} chunks")

# ── (c) Labels per task ─────────────────────────────────────────────────
print("\n─── Labels per task ───")
header2 = f"{'task':<20} {'emotion':<12} {'label':>5} {'soft_label (top 3)':<40} {'V':>5} {'A':>5} {'D':>5}"
print(header2)
print("─" * len(header2))
seen_tasks = set()
for c in chunks:
    if c["task_name"] in seen_tasks:
        continue
    seen_tasks.add(c["task_name"])
    sl = c["soft_label"]
    top3_idx = np.argsort(sl)[::-1][:3]
    top3_str = ", ".join(f"{EMOTIONS[i]}:{sl[i]:.2f}" for i in top3_idx if sl[i] > 0)
    print(f"{c['task_name']:<20} {c['emotion_name']:<12} {c['emotion_label']:>5} {top3_str:<40} {c['valence']:>5.1f} {c['arousal']:>5.1f} {c['dominance']:>5.1f}")

# ── (d) Overlap check ──────────────────────────────────────────────────
print("\n─── Overlap & boundary checks ───")
sorted_chunks = sorted(chunks, key=lambda c: c["start_90hz"])
overlaps = 0
for i in range(len(sorted_chunks) - 1):
    if sorted_chunks[i]["end_90hz"] > sorted_chunks[i + 1]["start_90hz"]:
        overlaps += 1
        print(f"  OVERLAP: chunk {i} end={sorted_chunks[i]['end_90hz']} > chunk {i+1} start={sorted_chunks[i+1]['start_90hz']}")
print(f"Overlapping pairs: {overlaps}")

# Check all chunks fall within their task's range
boundary_violations = 0
for c in chunks:
    task_start_raw, task_end_raw = subject_tasks[c["task_name"]]
    task_shifted_start = task_start_raw - session_a_start
    task_shifted_end = task_end_raw - session_a_start
    if c["start_90hz"] < task_shifted_start or c["end_90hz"] > task_shifted_end:
        boundary_violations += 1
        print(f"  BOUNDARY VIOLATION: {c['task_name']} chunk {c['chunk_idx_in_task']}: "
              f"[{c['start_90hz']}, {c['end_90hz']}) outside [{task_shifted_start}, {task_shifted_end})")
print(f"Boundary violations: {boundary_violations}")

# Chunk count match
print(f"\nTotal chunks (get_task_chunks): {len(chunks)}")
print(f"Total chunks (manual sum):      {total_chunks_manual}")
# Note: manual sum includes tasks without labels; get_task_chunks skips those.
# Count tasks that have labels:
tasks_with_labels = sum(1 for tn in subject_tasks if tn not in ("session_A", "session_B") and load_task_label(subject_dir, SUBJ, tn) is not None)
tasks_without_labels = len(task_rows) - tasks_with_labels
if tasks_without_labels > 0:
    print(f"  ({tasks_without_labels} tasks skipped for missing labels)")
    # Recompute manual sum for tasks with labels only
    manual_with_labels = sum(
        r["n_chunks"] for r in task_rows
        if load_task_label(subject_dir, SUBJ, r["task"]) is not None
    )
    print(f"  Manual sum (label-bearing tasks only): {manual_with_labels}")
    assert len(chunks) == manual_with_labels, "MISMATCH!"
else:
    assert len(chunks) == total_chunks_manual, "MISMATCH!"
print("  MATCH OK")

# ── (e) Excluded time (inter-task gaps) ─────────────────────────────────
print("\n─── Excluded time (inter-task gaps + calibration) ───")
total_recording = (session_b_end - session_a_start)   # total span in samples at 90Hz
total_used_samples = sum(r["task_samples"] for r in task_rows)
excluded_samples = total_recording - total_used_samples
# Also compute how many samples are actually chunked (discards trailing partial chunks)
chunked_samples = len(chunks) * CHUNK_SAMPLES_90HZ

print(f"Total recording span: {total_recording} samples = {total_recording/90:.1f}s = {total_recording/90/60:.1f}min")
print(f"Total task time:      {total_used_samples} samples = {total_used_samples/90:.1f}s = {total_used_samples/90/60:.1f}min")
print(f"Inter-task excluded:  {excluded_samples} samples = {excluded_samples/90:.1f}s = {excluded_samples/90/60:.1f}min")
print(f"Excluded percentage:  {excluded_samples/total_recording*100:.1f}%")
print(f"Chunked (10s only):   {chunked_samples} samples = {chunked_samples/90:.1f}s")
trailing_discard = total_used_samples - chunked_samples
print(f"Trailing partial:     {trailing_discard} samples = {trailing_discard/90:.1f}s (discarded)")

# ── (f) Compare with old manifest ──────────────────────────────────────
print("\n─── Old manifest comparison ───")
old_manifest_candidates = [
    data_dir / "ce_hardlabel_manifests" / "dataset_manifest.csv",
    data_dir.parent / "ce_hardlabel_manifests" / "dataset_manifest.csv",
    Path("data/embeddings") / "manifest.csv",
    Path("data/embeddings_10s_task_aware") / "manifest.csv",
]
old_manifest = None
for p in old_manifest_candidates:
    if p.exists():
        old_manifest = p
        break

if old_manifest is not None:
    print(f"Found old manifest: {old_manifest}")
    old_df = pd.read_csv(old_manifest)
    old_subj = old_df[old_df["subject"].astype(str) == SUBJ]
    print(f"Old manifest entries for {SUBJ}: {len(old_subj)}")
    print(f"New task-aware chunks for {SUBJ}: {len(chunks)}")
    print(f"Difference: {len(chunks) - len(old_subj):+d}")
else:
    print("No old manifest found at any candidate path.")
    print(f"New task-aware chunks for {SUBJ}: {len(chunks)}")
    # Estimate what old naive approach would give:
    # Old approach: session_A + session_B contiguous, chunk entire recording
    naive_chunks_a = (session_a_end - session_a_start) // CHUNK_SAMPLES_90HZ
    naive_chunks_b = (session_b_end - session_b_start) // CHUNK_SAMPLES_90HZ
    print(f"\nNaive whole-session chunking estimate:")
    print(f"  Session A: {naive_chunks_a} chunks (entire session)")
    print(f"  Session B: {naive_chunks_b} chunks (entire session)")
    print(f"  Total naive: {naive_chunks_a + naive_chunks_b}")
    print(f"  Task-aware: {len(chunks)}")
    print(f"  Reduction:  {naive_chunks_a + naive_chunks_b - len(chunks)} fewer chunks "
          f"({(1 - len(chunks)/(naive_chunks_a + naive_chunks_b))*100:.1f}% removed)")

# ── (g) Eye-tracking array bounds check ─────────────────────────────────
print("\n─── Eye-tracking data bounds check ───")
# Check for resampled or raw files
gaze_path = subject_dir / "gaze_90fps.npy"
gaze_raw_path = subject_dir / "gaze.npy"
pupils_path = subject_dir / "pupils_90fps.npy"
pupils_raw_path = subject_dir / "pupils.npy"

if gaze_path.exists():
    gaze = np.load(gaze_path)
    gaze_label = "gaze_90fps.npy"
elif gaze_raw_path.exists():
    gaze = np.load(gaze_raw_path)
    gaze_label = "gaze.npy (raw, not resampled)"
else:
    gaze = None
    gaze_label = "NOT FOUND"

if pupils_path.exists():
    pupils = np.load(pupils_path)
    pupils_label = "pupils_90fps.npy"
elif pupils_raw_path.exists():
    pupils = np.load(pupils_raw_path)
    pupils_label = "pupils.npy (raw, not resampled)"
else:
    pupils = None
    pupils_label = "NOT FOUND"

print(f"Gaze file:   {gaze_label}  shape={gaze.shape if gaze is not None else 'N/A'}")
print(f"Pupils file: {pupils_label}  shape={pupils.shape if pupils is not None else 'N/A'}")

if gaze is not None:
    min_len = len(gaze) if pupils is None else min(len(gaze), len(pupils))
    oob_count = 0
    max_end = 0
    for c in chunks:
        if c["end_90hz"] > min_len:
            oob_count += 1
        max_end = max(max_end, c["end_90hz"])

    in_bounds = len(chunks) - oob_count
    coverage_samples = min(max_end, min_len)
    print(f"\nChunks in bounds:    {in_bounds}/{len(chunks)}")
    print(f"Chunks out of bounds: {oob_count}/{len(chunks)}")
    print(f"Max chunk end index: {max_end}")
    print(f"Array length:        {min_len}")
    print(f"Total samples used by chunks: {len(chunks) * CHUNK_SAMPLES_90HZ}")
    print(f"Total samples available:      {min_len}")
    print(f"Coverage: {len(chunks) * CHUNK_SAMPLES_90HZ / min_len * 100:.1f}% of array used")

    if oob_count > 0:
        print("\n  WARNING: Some chunks exceed array bounds!")
        print("  This likely means the raw data needs resampling to 90fps first.")
        print("  The extraction script expects gaze_90fps.npy / pupils_90fps.npy.")
        # Show which tasks are affected
        for c in chunks:
            if c["end_90hz"] > min_len:
                print(f"    OOB: {c['task_name']} chunk {c['chunk_idx_in_task']} "
                      f"end={c['end_90hz']} > len={min_len}")
                break  # Just show first
        print(f"    ... ({oob_count} total)")
else:
    print("No gaze data found -- cannot verify bounds.")

# ── Summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"Subject:              {SUBJ}")
print(f"Tasks (total):        {len(task_rows)}")
print(f"Tasks (with labels):  {tasks_with_labels}")
print(f"10s chunks:           {len(chunks)}")
print(f"Recording span:       {total_recording/90/60:.1f} min")
print(f"Task time:            {total_used_samples/90/60:.1f} min")
print(f"Excluded (gaps):      {excluded_samples/90/60:.1f} min ({excluded_samples/total_recording*100:.1f}%)")
print(f"Overlaps:             {overlaps}")
print(f"Boundary violations:  {boundary_violations}")
integrity = "PASS" if overlaps == 0 and boundary_violations == 0 else "FAIL"
print(f"Data integrity:       {integrity}")
print(f"{'='*80}")

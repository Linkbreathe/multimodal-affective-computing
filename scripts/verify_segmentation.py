"""
Independent verification of segmentation pipeline output.
Opus 4.6 verification agent -- checks every claim from scratch.

Covers:
  A. Data Copy Verification (gaze, ppg, no pupils, file sizes)
  B. Segmentation Logic (summary stats, manifest, independent recompute x2)
  C. Cross-Reference with Official egoEMOTION (40 subjects, task lists, CSVs)
  D. Label Verification (CSV cross-check, task-level label consistency)
  E. Boundary/Overlap Verification (array bounds, overlap, inter-task gaps)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
PROJECT_ROOT = Path("/home/link/Wei/Models/core/real-time-vis-physio-fusion")
DATA_DIR = PROJECT_ROOT / "data" / "egoemotion_raw"
REF_DATA_DIR = Path("/mnt/c/Users/Public/Data/egoEMOTION/egoEMOTION")
SEG_DIR = PROJECT_ROOT / "data" / "segmentation_10s_task_aware"
MANIFEST_CSV = SEG_DIR / "manifest.csv"
SUMMARY_JSON = SEG_DIR / "segmentation_summary.json"
DETAILS_DIR = SEG_DIR / "subject_details"

# Official 40-subject list from egoEMOTION config
OFFICIAL_40 = [
    "005","006","007","008","009","010","011","012","013","014","015","016","017",
    "018","019","021","022","023","024","025","026","027","028","029","030","031",
    "032","033","034","035","036","037","038","039","040","042","043","044","045","046",
]

CHUNK_SAMPLES = 900  # 10s * 90 Hz
EMOTIONS = ["Amused","Content","Excited","Awe","Neutral","Fear","Sad","Disgust","Anger"]
SESSION_B_TASKS = {"trynottolaugh","sadletter","flappybird","slenderman","jellybean","painting","jenga"}
SESSION_A_EMOTION_MAP = {"Sadness": "Sad", "Angry": "Anger"}

results = {}

def record(check_id, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results[check_id] = {"status": status, "detail": detail}
    flag = "  OK " if passed else "FAIL "
    print(f"[{flag}] {check_id}: {detail}")


# ==================================================================
# A. DATA COPY VERIFICATION
# ==================================================================
print("=" * 70)
print("SECTION A: DATA COPY VERIFICATION")
print("=" * 70)

subject_dirs = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir() and d.name.isdigit()])

# A1: Count subjects with gaze_90fps.npy
gaze_subjects = [s for s in subject_dirs if (DATA_DIR / s / "gaze_90fps.npy").exists()]
record("A1_gaze_count", len(gaze_subjects) == 28,
       f"Subjects with gaze_90fps.npy: {len(gaze_subjects)} (expected 28). List: {gaze_subjects}")

# A2: Count subjects with ppg_ear_125hz.npy
ppg_subjects = [s for s in subject_dirs if (DATA_DIR / s / "ppg_ear_125hz.npy").exists()]
record("A2_ppg_count", len(ppg_subjects) == 28,
       f"Subjects with ppg_ear_125hz.npy: {len(ppg_subjects)} (expected 28). List: {ppg_subjects}")

# A3: Confirm NO pupils_90fps.npy was copied
pupils_subjects = [s for s in subject_dirs if (DATA_DIR / s / "pupils_90fps.npy").exists()]
record("A3_no_pupils", len(pupils_subjects) == 0,
       f"Subjects with pupils_90fps.npy: {len(pupils_subjects)}" +
       (f" -- UNEXPECTED: {pupils_subjects}" if pupils_subjects else " (correct: gaze-only)"))

# A4: Spot-check file sizes (gaze_90fps.npy) for 3 subjects
spot_subjects = ["005", "015", "030"]
size_mismatches = []
size_details = []
for s in spot_subjects:
    local = DATA_DIR / s / "gaze_90fps.npy"
    ref = REF_DATA_DIR / s / "gaze_90fps.npy"
    if local.exists() and ref.exists():
        l_size = local.stat().st_size
        r_size = ref.stat().st_size
        size_details.append(f"{s}: local={l_size} ref={r_size} match={l_size==r_size}")
        if l_size != r_size:
            size_mismatches.append(f"{s}: local={l_size} ref={r_size}")
    elif not ref.exists():
        size_details.append(f"{s}: ref file missing at {ref}")
        size_mismatches.append(f"{s}: ref file missing")
    elif not local.exists():
        size_details.append(f"{s}: local file missing")
        size_mismatches.append(f"{s}: local file missing")

record("A4_size_spotcheck", len(size_mismatches) == 0,
       f"Checked {spot_subjects}. " +
       ("All sizes match." if not size_mismatches else f"MISMATCHES: {size_mismatches}") +
       f" Details: {size_details}")


# ==================================================================
# B. SEGMENTATION LOGIC VERIFICATION
# ==================================================================
print("\n" + "=" * 70)
print("SECTION B: SEGMENTATION LOGIC VERIFICATION")
print("=" * 70)

# B1: Summary JSON top-level stats
with open(SUMMARY_JSON) as f:
    summary = json.load(f)

record("B1a_total_subjects", summary["total_subjects"] == 28,
       f"total_subjects={summary['total_subjects']} (expected 28)")
record("B1b_total_tasks", summary["total_tasks"] == 448,
       f"total_tasks={summary['total_tasks']} (expected 448)")
record("B1c_total_chunks", summary["total_chunks"] == 6663,
       f"total_chunks={summary['total_chunks']} (expected 6663)")
record("B1d_validation_passed", summary["validation_passed"] is True,
       f"validation_passed={summary['validation_passed']}")

# B2: Manifest row count, unique subjects, emotion distribution
manifest = pd.read_csv(MANIFEST_CSV)
record("B2a_manifest_rows", len(manifest) == 6663,
       f"manifest rows={len(manifest)} (expected 6663)")

# Ensure subject column is treated as zero-padded 3-digit strings
manifest["subject_str"] = manifest["subject"].astype(str).str.zfill(3)
unique_subjects = sorted(manifest["subject_str"].unique())
record("B2b_unique_subjects", len(unique_subjects) == 28,
       f"unique subjects in manifest={len(unique_subjects)}")

manifest_emo_dist = manifest["emotion_name"].value_counts().to_dict()
summary_emo_dist = summary["emotion_distribution"]
emo_match = all(manifest_emo_dist.get(e, 0) == summary_emo_dist.get(e, 0) for e in EMOTIONS)
record("B2c_emo_dist_match", emo_match,
       f"Manifest emotion dist matches summary JSON: {emo_match}\n"
       f"    Manifest: {dict(sorted(manifest_emo_dist.items()))}\n"
       f"    Summary:  {dict(sorted(summary_emo_dist.items()))}")

# Verify emotion distribution sums to total chunks
emo_sum = sum(manifest_emo_dist.values())
record("B2d_emo_sum", emo_sum == 6663,
       f"Sum of emotion distribution = {emo_sum} (expected 6663)")

# B3: Independent recomputation for subject 005
print("\n--- B3: Independent recompute for subject 005 ---")
tt = np.load(DATA_DIR / "task_times.npy", allow_pickle=True).item()

def recompute_chunks(subj_id):
    """Independently recompute chunk counts per task for a subject."""
    subj_tasks = tt[subj_id]
    session_a_start = subj_tasks["session_A"][0]
    task_chunks = {}
    total = 0
    for task_name, (start_raw, end_raw) in subj_tasks.items():
        if task_name in ("session_A", "session_B"):
            continue
        shifted_start = start_raw - session_a_start
        shifted_end = end_raw - session_a_start
        task_samples = shifted_end - shifted_start
        n_chunks = task_samples // CHUNK_SAMPLES
        task_chunks[task_name] = {
            "n_chunks": n_chunks,
            "shifted_start": shifted_start,
            "shifted_end": shifted_end,
            "task_samples": task_samples,
        }
        total += n_chunks
    return task_chunks, total

my_005, my_total_005 = recompute_chunks("005")
manifest_005 = manifest[manifest["subject_str"] == "005"]
manifest_005_tasks = manifest_005.groupby("task_name").size().to_dict()

b3_mismatches = []
all_task_names = set(my_005.keys()) | set(manifest_005_tasks.keys())
for tn in sorted(all_task_names):
    my_n = my_005[tn]["n_chunks"] if tn in my_005 else 0
    man_n = manifest_005_tasks.get(tn, 0)
    if man_n > 0 and my_n != man_n:
        b3_mismatches.append(f"  {tn}: recomputed={my_n} manifest={man_n}")

record("B3a_recompute_005_total", my_total_005 >= manifest_005.shape[0],
       f"Subject 005: recomputed total (all tasks incl unlabeled)={my_total_005}, "
       f"manifest={manifest_005.shape[0]}. "
       f"Tasks in task_times: {len(my_005)}, tasks in manifest: {len(manifest_005_tasks)}")
record("B3b_recompute_005_taskwise", len(b3_mismatches) == 0,
       "Task-by-task chunk count: " +
       ("All labeled tasks match." if not b3_mismatches else "\n".join(b3_mismatches)))

print(f"  Subject 005 task-by-task:")
for tn in sorted(all_task_names):
    my_n = my_005[tn]["n_chunks"] if tn in my_005 else 0
    man_n = manifest_005_tasks.get(tn, 0)
    status = "OK" if man_n == 0 or my_n == man_n else "MISMATCH"
    label_note = "(no label in manifest)" if man_n == 0 else ""
    print(f"    {tn:25s}: recomputed={my_n:3d}  manifest={man_n:3d}  {status} {label_note}")

# B4: Independent recomputation for subject 030
print("\n--- B4: Independent recompute for subject 030 ---")
my_030, my_total_030 = recompute_chunks("030")
manifest_030 = manifest[manifest["subject_str"] == "030"]
manifest_030_tasks = manifest_030.groupby("task_name").size().to_dict()

b4_mismatches = []
all_030_tasks = set(my_030.keys()) | set(manifest_030_tasks.keys())
for tn in sorted(all_030_tasks):
    my_n = my_030[tn]["n_chunks"] if tn in my_030 else 0
    man_n = manifest_030_tasks.get(tn, 0)
    if man_n > 0 and my_n != man_n:
        b4_mismatches.append(f"  {tn}: recomputed={my_n} manifest={man_n}")

record("B4a_recompute_030_total", my_total_030 >= manifest_030.shape[0],
       f"Subject 030: recomputed total={my_total_030}, manifest={manifest_030.shape[0]}")
record("B4b_recompute_030_taskwise", len(b4_mismatches) == 0,
       "Task-by-task chunk count: " +
       ("All labeled tasks match." if not b4_mismatches else "\n".join(b4_mismatches)))

print(f"  Subject 030 task-by-task:")
for tn in sorted(all_030_tasks):
    my_n = my_030[tn]["n_chunks"] if tn in my_030 else 0
    man_n = manifest_030_tasks.get(tn, 0)
    status = "OK" if man_n == 0 or my_n == man_n else "MISMATCH"
    label_note = "(no label in manifest)" if man_n == 0 else ""
    print(f"    {tn:25s}: recomputed={my_n:3d}  manifest={man_n:3d}  {status} {label_note}")


# ==================================================================
# C. CROSS-REFERENCE WITH OFFICIAL egoEMOTION
# ==================================================================
print("\n" + "=" * 70)
print("SECTION C: CROSS-REFERENCE WITH OFFICIAL egoEMOTION")
print("=" * 70)

# C1: Verify 28 is a subset of official 40
local_subjects = sorted(subject_dirs)
is_subset = all(s in OFFICIAL_40 for s in local_subjects)
missing_from_official = [s for s in local_subjects if s not in OFFICIAL_40]
not_in_local = sorted(set(OFFICIAL_40) - set(local_subjects))
record("C1_subset_of_40", is_subset and len(local_subjects) == 28,
       f"Local {len(local_subjects)} subjects are subset of official 40: {is_subset}. "
       f"12 subjects not in local: {not_in_local}")
if missing_from_official:
    print(f"  WARNING: subjects NOT in official 40: {missing_from_official}")

# C2: For subject 005, compare task list from task_times vs CSVs
print("\n--- C2: Task list cross-reference for subject 005 ---")
subj = "005"
tt_tasks_005 = [t for t in tt[subj].keys() if t not in ("session_A", "session_B")]

csv_a = DATA_DIR / subj / f"Session_A_{subj}.csv"
csv_b = DATA_DIR / subj / f"Session_B_{subj}.csv"
csv_a_tasks = []
csv_b_tasks = []
if csv_a.exists():
    df_a = pd.read_csv(csv_a)
    csv_a_tasks = [f"video_{row['Video Emotion']}" for _, row in df_a.iterrows()]
    print(f"  Session A CSV video tasks: {csv_a_tasks}")
if csv_b.exists():
    df_b = pd.read_csv(csv_b)
    csv_b_tasks = list(df_b["Activity Name"].str.lower().str.replace(" ", "").unique())
    print(f"  Session B CSV activity tasks: {csv_b_tasks}")

tt_session_a_tasks = [t for t in tt_tasks_005 if t.startswith("video_")]
tt_session_b_tasks = [t for t in tt_tasks_005 if t.lower() in SESSION_B_TASKS]
tt_other_tasks = [t for t in tt_tasks_005 if t not in tt_session_a_tasks and t not in tt_session_b_tasks]

print(f"  task_times Session A tasks: {sorted(tt_session_a_tasks)}")
print(f"  task_times Session B tasks: {sorted(tt_session_b_tasks)}")
if tt_other_tasks:
    print(f"  task_times OTHER tasks (unlabeled): {sorted(tt_other_tasks)}")

# Check Session A overlap
csv_a_overlap = set(csv_a_tasks) & set(tt_session_a_tasks)
csv_a_only = set(csv_a_tasks) - set(tt_session_a_tasks)
tt_a_only = set(tt_session_a_tasks) - set(csv_a_tasks)
record("C2_task_crossref",
       len(csv_a_overlap) > 0,
       f"task_times has {len(tt_tasks_005)} tasks total, "
       f"Session A: {len(tt_session_a_tasks)} in tt / {len(csv_a_tasks)} in CSV, "
       f"overlap={len(csv_a_overlap)}, CSV-only={csv_a_only}, tt-only={tt_a_only}. "
       f"Session B: {len(tt_session_b_tasks)} in tt / {len(csv_b_tasks)} in CSV")


# ==================================================================
# D. LABEL VERIFICATION
# ==================================================================
print("\n" + "=" * 70)
print("SECTION D: LABEL VERIFICATION")
print("=" * 70)

subj = "005"

# D1: Read and display Session A/B CSVs
print(f"\n--- D1: CSV contents for subject {subj} ---")
if csv_a.exists():
    df_a = pd.read_csv(csv_a)
    print(f"  Session A ({len(df_a)} rows):")
    for _, row in df_a.iterrows():
        print(f"    Video Emotion={row['Video Emotion']}, "
              f"Valence={row.get('Valence','?')}, Arousal={row.get('Arousal','?')}")

if csv_b.exists():
    df_b = pd.read_csv(csv_b)
    print(f"  Session B ({len(df_b)} rows):")
    for _, row in df_b.iterrows():
        print(f"    Activity={row['Activity Name']}, "
              f"Valence={row.get('Valence','?')}, Arousal={row.get('Arousal','?')}")

# D2: Check labels in manifest match CSVs
print(f"\n--- D2: Label cross-check for subject {subj} ---")
label_mismatches = []
manifest_subj = manifest[manifest["subject_str"] == subj]

# Session A
if csv_a.exists():
    for _, csv_row in df_a.iterrows():
        video_emotion = str(csv_row["Video Emotion"])
        task_name = f"video_{video_emotion}"
        expected_emotion = SESSION_A_EMOTION_MAP.get(video_emotion, video_emotion)
        manifest_task = manifest_subj[manifest_subj["task_name"] == task_name]
        if len(manifest_task) > 0:
            actual_emotions = manifest_task["emotion_name"].unique()
            if len(actual_emotions) != 1 or actual_emotions[0] != expected_emotion:
                label_mismatches.append(
                    f"Session A {task_name}: expected={expected_emotion} got={list(actual_emotions)}")
                print(f"  MISMATCH: {task_name} expected={expected_emotion} got={list(actual_emotions)}")
            else:
                print(f"  OK: {task_name} -> {expected_emotion}")

# Session B: verify dominant emotion is consistent
if csv_b.exists():
    for task_lower in csv_b_tasks:
        manifest_task = manifest_subj[manifest_subj["task_name"].str.lower() == task_lower]
        if len(manifest_task) > 0:
            actual_emotions = manifest_task["emotion_name"].unique()
            print(f"  Session B '{task_lower}' -> label={list(actual_emotions)}")

record("D2_label_match_csv", len(label_mismatches) == 0,
       "Labels in manifest match Session A CSVs: " +
       ("All match." if not label_mismatches else "\n".join(label_mismatches)))

# D3: Verify all chunks within same task have the SAME label (FULL dataset)
print(f"\n--- D3: Task-level label consistency (full dataset) ---")
task_label_consistency_errors = []
for (subj_id, task_name), group in manifest.groupby(["subject_str", "task_name"]):
    unique_labels = group["emotion_label"].unique()
    unique_names = group["emotion_name"].unique()
    if len(unique_labels) > 1 or len(unique_names) > 1:
        task_label_consistency_errors.append(
            f"  {subj_id}/{task_name}: labels={list(unique_labels)}, names={list(unique_names)}")

record("D3_task_label_consistency", len(task_label_consistency_errors) == 0,
       f"All chunks within same task have same label: "
       f"{'Yes -- checked {0} task groups'.format(manifest.groupby(['subject_str','task_name']).ngroups) if not task_label_consistency_errors else 'NO -- ' + str(len(task_label_consistency_errors)) + ' violations'}")
if task_label_consistency_errors:
    for e in task_label_consistency_errors[:10]:
        print(e)


# ==================================================================
# E. BOUNDARY / OVERLAP VERIFICATION
# ==================================================================
print("\n" + "=" * 70)
print("SECTION E: BOUNDARY / OVERLAP VERIFICATION")
print("=" * 70)

# E1: Boundary check for subject 005
subj = "005"
gaze_005 = np.load(DATA_DIR / subj / "gaze_90fps.npy")
gaze_len_005 = len(gaze_005)
manifest_005 = manifest[manifest["subject_str"] == subj]
boundary_violations_005 = []
for _, row in manifest_005.iterrows():
    if row["end_90hz"] > gaze_len_005:
        boundary_violations_005.append(
            f"  {row['task_name']} chunk {row['chunk_idx_in_task']}: "
            f"end={row['end_90hz']} > gaze_len={gaze_len_005}")
    if row["start_90hz"] < 0:
        boundary_violations_005.append(
            f"  {row['task_name']} chunk {row['chunk_idx_in_task']}: "
            f"start={row['start_90hz']} < 0")
    if row["end_90hz"] - row["start_90hz"] != CHUNK_SAMPLES:
        boundary_violations_005.append(
            f"  {row['task_name']} chunk {row['chunk_idx_in_task']}: "
            f"size={row['end_90hz'] - row['start_90hz']} != {CHUNK_SAMPLES}")

record("E1a_boundary_005", len(boundary_violations_005) == 0,
       f"Subject 005: gaze_len={gaze_len_005}, chunks={len(manifest_005)}. "
       f"Violations: {len(boundary_violations_005)}")

# E1b: Boundary check for subject 030
subj = "030"
gaze_030 = np.load(DATA_DIR / subj / "gaze_90fps.npy")
gaze_len_030 = len(gaze_030)
manifest_030 = manifest[manifest["subject_str"] == subj]
boundary_violations_030 = []
for _, row in manifest_030.iterrows():
    if row["end_90hz"] > gaze_len_030:
        boundary_violations_030.append(
            f"  {row['task_name']} chunk {row['chunk_idx_in_task']}: "
            f"end={row['end_90hz']} > gaze_len={gaze_len_030}")
    if row["start_90hz"] < 0:
        boundary_violations_030.append(
            f"  {row['task_name']} chunk {row['chunk_idx_in_task']}: "
            f"start={row['start_90hz']} < 0")

record("E1b_boundary_030", len(boundary_violations_030) == 0,
       f"Subject 030: gaze_len={gaze_len_030}, chunks={len(manifest_030)}. "
       f"Violations: {len(boundary_violations_030)}")

# E2: No overlap check (ALL 28 subjects)
print("\n--- E2: Overlap check for all subjects ---")
overlap_errors_all = []
for subj_id in local_subjects:
    subj_manifest = manifest[manifest["subject_str"] == subj_id].sort_values("start_90hz")
    starts = subj_manifest["start_90hz"].values
    ends = subj_manifest["end_90hz"].values
    tasks = subj_manifest["task_name"].values
    for i in range(1, len(starts)):
        if starts[i] < ends[i-1]:
            overlap_errors_all.append(
                f"  {subj_id}: overlap at idx {i}: prev_end={ends[i-1]} ({tasks[i-1]}) "
                f"> curr_start={starts[i]} ({tasks[i]})")

record("E2_no_overlap", len(overlap_errors_all) == 0,
       f"Overlap check across all 28 subjects: {len(overlap_errors_all)} overlaps found")
for e in overlap_errors_all[:5]:
    print(e)

# E3: Inter-task gaps (chunks from different tasks should NOT be contiguous)
print("\n--- E3: Inter-task gap verification ---")
gap_issues = 0
gap_ok = 0
gap_details = []
for subj_id in ["005", "030"]:
    subj_manifest = manifest[manifest["subject_str"] == subj_id].sort_values("start_90hz")
    starts = subj_manifest["start_90hz"].values
    ends = subj_manifest["end_90hz"].values
    tasks = subj_manifest["task_name"].values
    for i in range(1, len(starts)):
        if tasks[i] != tasks[i-1]:
            gap = starts[i] - ends[i-1]
            if gap <= 0:
                gap_issues += 1
                detail = (f"  {subj_id}: NO GAP between {tasks[i-1]} -> {tasks[i]} "
                          f"(gap={gap} samples)")
                gap_details.append(detail)
                print(detail)
            else:
                gap_ok += 1

record("E3_inter_task_gaps", gap_issues == 0,
       f"Inter-task transitions: {gap_ok} have positive gaps, "
       f"{gap_issues} have zero/negative gaps")

# E4: Full boundary verification for ALL 28 subjects
print("\n--- E4: Full boundary check all subjects ---")
full_boundary_errors = 0
chunk_size_errors = 0
for subj_id in local_subjects:
    gaze_path = DATA_DIR / subj_id / "gaze_90fps.npy"
    if not gaze_path.exists():
        print(f"  WARNING: no gaze file for {subj_id}")
        full_boundary_errors += 1
        continue
    gaze = np.load(gaze_path)
    gaze_len = len(gaze)
    subj_manifest = manifest[manifest["subject_str"] == subj_id]
    for _, row in subj_manifest.iterrows():
        if row["end_90hz"] > gaze_len or row["start_90hz"] < 0:
            full_boundary_errors += 1
            print(f"  {subj_id}: BOUNDARY VIOLATION "
                  f"start={row['start_90hz']} end={row['end_90hz']} gaze_len={gaze_len}")
        if row["end_90hz"] - row["start_90hz"] != CHUNK_SAMPLES:
            chunk_size_errors += 1
            print(f"  {subj_id}: CHUNK SIZE ERROR "
                  f"size={row['end_90hz'] - row['start_90hz']} != {CHUNK_SAMPLES}")

record("E4_all_boundary", full_boundary_errors == 0,
       f"Boundary check all 28 subjects (6663 chunks): {full_boundary_errors} violations")
record("E4_all_chunk_size", chunk_size_errors == 0,
       f"Chunk size check all 6663 chunks: {chunk_size_errors} != 900 samples")


# ==================================================================
# F. EXTRA: Per-subject chunk count cross-check
# ==================================================================
print("\n" + "=" * 70)
print("SECTION F: PER-SUBJECT CHUNK COUNT CROSS-CHECK")
print("=" * 70)

# Verify chunks_per_subject in summary matches manifest
chunks_per_subj_mismatch = []
for subj_id in local_subjects:
    manifest_count = len(manifest[manifest["subject_str"] == subj_id])
    summary_count = summary["chunks_per_subject"].get(subj_id, -1)
    if manifest_count != summary_count:
        chunks_per_subj_mismatch.append(
            f"  {subj_id}: manifest={manifest_count} summary={summary_count}")

record("F1_chunks_per_subject", len(chunks_per_subj_mismatch) == 0,
       f"chunks_per_subject in summary matches manifest: "
       f"{'All 28 match' if not chunks_per_subj_mismatch else str(len(chunks_per_subj_mismatch)) + ' mismatches'}")

# Verify subject_details JSON files exist and match
detail_mismatches = []
for subj_id in local_subjects:
    detail_path = DETAILS_DIR / f"{subj_id}.json"
    if not detail_path.exists():
        detail_mismatches.append(f"  {subj_id}: JSON file missing")
        continue
    with open(detail_path) as f:
        detail = json.load(f)
    manifest_count = len(manifest[manifest["subject_str"] == subj_id])
    if len(detail) != manifest_count:
        detail_mismatches.append(
            f"  {subj_id}: JSON has {len(detail)} chunks, manifest has {manifest_count}")

record("F2_subject_details_json", len(detail_mismatches) == 0,
       f"subject_details JSONs match manifest: "
       f"{'All 28 match' if not detail_mismatches else str(len(detail_mismatches)) + ' mismatches'}")


# ==================================================================
# FINAL REPORT
# ==================================================================
print("\n" + "=" * 70)
print("FINAL VERIFICATION REPORT")
print("=" * 70)

passes = sum(1 for r in results.values() if r["status"] == "PASS")
fails = sum(1 for r in results.values() if r["status"] == "FAIL")

for check_id, r in results.items():
    print(f"  [{r['status']:4s}] {check_id}")

print(f"\nTotal: {passes} PASS, {fails} FAIL out of {len(results)} checks")

if fails == 0:
    print("\n>>> VERDICT: ALL CHECKS PASSED. Segmentation is CORRECT. <<<")
else:
    print(f"\n>>> VERDICT: {fails} CHECK(S) FAILED. See details above. <<<")
    # Print details of failed checks
    for check_id, r in results.items():
        if r["status"] == "FAIL":
            print(f"\n  FAILED {check_id}: {r['detail']}")

sys.exit(0 if fails == 0 else 1)

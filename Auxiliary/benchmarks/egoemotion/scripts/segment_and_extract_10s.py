"""Task-aware segmentation and embedding extraction (10s non-overlapping chunks).

Follows the official egoEMOTION segmentation approach:
1. Split by task first using task_times.npy
2. Within each task, chunk into 10s non-overlapping windows (900 samples at 90Hz)
3. Discard incomplete trailing chunks
4. Exclude inter-task gaps (calibration, questionnaires)
5. One label per task — all chunks inherit the task's self-reported emotion label

Usage:
    conda run -n visphy python scripts/segment_and_extract_10s.py --encoder papagei_ppg
    conda run -n visphy python scripts/segment_and_extract_10s.py --encoder patchtst_eye
    conda run -n visphy python scripts/segment_and_extract_10s.py --encoder inceptiontime
    conda run -n visphy python scripts/segment_and_extract_10s.py --encoder ecg_founder
    conda run -n visphy python scripts/segment_and_extract_10s.py --encoder video_mae_v2
    conda run -n visphy python scripts/segment_and_extract_10s.py --encoder all
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np
import pandas as pd
import torch
import cv2
from tqdm import tqdm

from mac.data.egoemotion import compute_manifest_hash
from mac.data.video_transforms import make_consecutive_clips, read_frames
from mac.config.simple import load_config
from mac.encoders.registry import ModalityRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

CHUNK_LEN_SEC = 10
FS_ET = 90
CHUNK_SAMPLES_90HZ = CHUNK_LEN_SEC * FS_ET       # 900
CHUNK_SAMPLES_125HZ = CHUNK_LEN_SEC * 125         # 1250
CHUNK_SAMPLES_500HZ = CHUNK_LEN_SEC * 500         # 5000

EMOTIONS = ["Amused", "Content", "Excited", "Awe", "Neutral", "Fear", "Sad", "Disgust", "Anger"]
EMOTION_TO_LABEL = {e: i for i, e in enumerate(EMOTIONS)}

SESSION_B_TASKS = {"trynottolaugh", "sadletter", "flappybird", "slenderman", "jellybean", "painting", "jenga"}

# Session A Video Emotion column sometimes uses different names
SESSION_A_EMOTION_MAP = {
    "Sadness": "Sad",
    "Angry": "Anger",
}


# ---------------------------------------------------------------------------
# Label loading from Session CSVs
# ---------------------------------------------------------------------------

def load_task_label(subject_dir: Path, subject_id: str, task_name: str) -> dict | None:
    """Load emotion labels for a task from Session A or Session B CSV.

    Returns dict with emotion_label (int), emotion_name (str),
    soft_label (np.ndarray [9]), valence, arousal, dominance, or None.
    """
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
# Task enumeration and chunking
# ---------------------------------------------------------------------------

def get_task_chunks(subject_tasks: dict, subject_dir: Path, subject_id: str) -> list[dict]:
    """Enumerate tasks and chunk each into 10s windows.

    Returns list of dicts with: task_name, chunk_idx_in_task, start_90hz, end_90hz,
    and label info.  All coordinates are shifted (relative to session_A start).
    """
    session_a_start = subject_tasks["session_A"][0]
    chunks = []

    for task_name, (start_raw, end_raw) in subject_tasks.items():
        if task_name in ("session_A", "session_B"):
            continue

        # Load label for this task
        label_info = load_task_label(subject_dir, subject_id, task_name)
        if label_info is None:
            log.debug(f"No label for {subject_id}/{task_name}, skipping")
            continue

        # Shifted coordinates (resampled files start from session_A_start)
        shifted_start = start_raw - session_a_start
        shifted_end = end_raw - session_a_start

        # Chunk into 10s windows at 90Hz
        task_samples = shifted_end - shifted_start
        n_chunks = task_samples // CHUNK_SAMPLES_90HZ

        for ci in range(n_chunks):
            chunk_start_90 = shifted_start + ci * CHUNK_SAMPLES_90HZ
            chunk_end_90 = chunk_start_90 + CHUNK_SAMPLES_90HZ
            chunks.append({
                "task_name": task_name,
                "chunk_idx_in_task": ci,
                "start_90hz": chunk_start_90,
                "end_90hz": chunk_end_90,
                **label_info,
            })

    return chunks


# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------

EYE_ENCODER_DIR_TO_KEY = {
    "patchtst_eye": "patchtst",
    "inceptiontime": "inceptiontime",
}


def resolve_encoder_dirs(registry: ModalityRegistry, requested_encoder: str) -> list[str]:
    """Resolve CLI encoder selection to embedding directory names."""
    if requested_encoder != "all":
        return [requested_encoder]
    return [
        registry.get_embedding_dir_name(modality)
        for modality in registry.get_enabled_modalities()
    ]


def extract_ppg(
    chunks_by_subject: dict[str, list[dict]],
    data_dir: Path,
    output_dir: Path,
    device: str,
    ppg_encoder: str = "papagei",
    manifest_hash: str | None = None,
) -> int:
    """Extract PPG embeddings for each 10s chunk.

    Args:
        ppg_encoder: Which PPG encoder to use ('papagei' or 'pulseppg').
    """
    if ppg_encoder == "pulseppg":
        from mac.encoders.pulse_ppg import PulsePPGEncoder
        encoder = PulsePPGEncoder().to(device)
    else:
        from mac.encoders.papagei import PapageiEncoder
        encoder = PapageiEncoder().to(device)
    extracted = 0

    for subj, chunks in tqdm(chunks_by_subject.items(), desc="PPG subjects"):
        ppg_path = data_dir / subj / "ppg_ear_125hz.npy"
        if not ppg_path.exists():
            raw_exists = (data_dir / subj / "ppg_ear.npy").exists()
            hint = " -- run `python scripts/preprocess_ppg.py` first" if raw_exists else ""
            log.warning(f"PPG not found: {ppg_path}{hint}")
            continue
        ppg = np.load(ppg_path)

        for c in chunks:
            save_path = output_dir / subj / f"segment_{c['global_seq']:04d}.pt"
            if save_path.exists():
                extracted += 1
                continue

            start_125 = int(c["start_90hz"] * 125 / 90)
            end_125 = start_125 + CHUNK_SAMPLES_125HZ

            if end_125 > len(ppg):
                log.debug(f"PPG {subj}/seq_{c['global_seq']}: end_125={end_125} > len={len(ppg)}, skipping")
                continue

            chunk = ppg[start_125:end_125]
            assert len(chunk) == CHUNK_SAMPLES_125HZ, (
                f"PPG chunk size mismatch: {len(chunk)} != {CHUNK_SAMPLES_125HZ} "
                f"for {subj}/seq_{c['global_seq']}"
            )

            # Z-score normalize per-segment
            std = chunk.std()
            if std > 0:
                chunk = (chunk - chunk.mean()) / std

            x = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)  # [1, 1, T]
            with torch.no_grad():
                emb = encoder(x)

            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "embedding": emb.cpu(),
                "segment_idx": c["global_seq"],
                "subject": subj,
                "label": c["emotion_label"],
                "emotion": c["emotion_name"],
                "manifest_hash": manifest_hash,
            }, save_path)
            extracted += 1

    return extracted


def extract_ecg(
    chunks_by_subject: dict[str, list[dict]],
    data_dir: Path,
    output_dir: Path,
    device: str,
    manifest_hash: str | None = None,
) -> int:
    """Extract ECGFounder embeddings for each 10s ECG chunk."""
    from mac.preprocessing.ecg import preprocess_ecgfounder_segment
    from mac.encoders.ecgfounder import ECGFounderEncoder

    encoder = ECGFounderEncoder().to(device)
    extracted = 0

    for subj, chunks in tqdm(chunks_by_subject.items(), desc="ECG subjects"):
        ecg_path = data_dir / subj / "ecg_90fps.npy"
        if not ecg_path.exists():
            log.warning(f"ECG not found: {ecg_path}")
            continue
        ecg = np.load(ecg_path)

        for c in chunks:
            save_path = output_dir / subj / f"segment_{c['global_seq']:04d}.pt"
            if save_path.exists():
                extracted += 1
                continue

            start_90 = c["start_90hz"]
            end_90 = start_90 + CHUNK_SAMPLES_90HZ
            if end_90 > len(ecg):
                log.debug(
                    f"ECG {subj}/seq_{c['global_seq']}: end_90={end_90} > len={len(ecg)}, skipping"
                )
                continue

            chunk = ecg[start_90:end_90]
            x_np = preprocess_ecgfounder_segment(chunk, source_fs=90, target_fs=500)
            assert x_np.shape == (1, CHUNK_SAMPLES_500HZ), (
                f"ECG chunk size mismatch: {x_np.shape} for {subj}/seq_{c['global_seq']}"
            )
            x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = encoder(x)

            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "embedding": emb.cpu(),
                "segment_idx": c["global_seq"],
                "subject": subj,
                "label": c["emotion_label"],
                "emotion": c["emotion_name"],
                "manifest_hash": manifest_hash,
            }, save_path)
            extracted += 1

    return extracted


def extract_eye(
    chunks_by_subject: dict[str, list[dict]],
    data_dir: Path,
    output_dir: Path,
    device: str,
    eye_encoder_key: str,
    manifest_hash: str | None = None,
) -> int:
    """Extract eye-tracking embeddings for each 10s chunk."""
    import os
    if eye_encoder_key == "patchtst":
        from mac.encoders.patchtst import PatchTSTEncoder

        encoder = PatchTSTEncoder(
            num_channels=4, patch_len=45, stride=22,
            d_model=128, n_heads=4, n_layers=3, seq_len=900,
        ).to(device)
        pretrained = "checkpoints/patchtst_pretrained.pt"
    elif eye_encoder_key == "inceptiontime":
        from mac.encoders.inceptiontime import InceptionTimeGazeEncoder

        encoder = InceptionTimeGazeEncoder().to(device)
        pretrained = "checkpoints/inceptiontime_gaze_pretrained.pt"
    else:
        raise ValueError(f"Unsupported eye encoder '{eye_encoder_key}'")

    if not os.path.exists(pretrained):
        raise FileNotFoundError(
            f"Pre-trained eye-tracking checkpoint not found at {pretrained}."
        )
    encoder.load_state_dict(torch.load(pretrained, weights_only=True))
    log.info(f"Loaded pre-trained {eye_encoder_key} encoder from {pretrained}")
    encoder.freeze()

    extracted = 0

    for subj, chunks in tqdm(chunks_by_subject.items(), desc="Eye subjects"):
        gaze_path = data_dir / subj / "gaze_90fps.npy"
        pupil_path = data_dir / subj / "pupils_90fps.npy"
        if not gaze_path.exists():
            log.warning(f"Eye data not found for {subj}")
            continue
        gaze = np.load(gaze_path)
        pupils = None
        if eye_encoder_key == "patchtst":
            if not pupil_path.exists():
                log.warning(f"Pupil data not found for {subj}")
                continue
            pupils = np.load(pupil_path)

        for c in chunks:
            save_path = output_dir / subj / f"segment_{c['global_seq']:04d}.pt"
            if save_path.exists():
                extracted += 1
                continue

            start_90 = c["start_90hz"]
            end_90 = c["end_90hz"]

            available_len = len(gaze)
            if pupils is not None:
                available_len = min(available_len, len(pupils))
            if end_90 > available_len:
                log.debug(
                    f"Eye {subj}/seq_{c['global_seq']}: end_90={end_90} > len={available_len}, skipping"
                )
                continue

            gaze_chunk = gaze[start_90:end_90, :2]
            if eye_encoder_key == "patchtst":
                assert pupils is not None
                pupil_chunk = pupils[start_90:end_90]
                eye_chunk = np.concatenate([gaze_chunk, pupil_chunk], axis=1)
                x = torch.tensor(eye_chunk, dtype=torch.float32).unsqueeze(0).to(device)
            else:
                x = torch.tensor(gaze_chunk.T, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                emb = encoder(x)

            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "embedding": emb.cpu(),
                "segment_idx": c["global_seq"],
                "subject": subj,
                "label": c["emotion_label"],
                "emotion": c["emotion_name"],
                "manifest_hash": manifest_hash,
            }, save_path)
            extracted += 1

    return extracted


def extract_video(
    chunks_by_subject: dict[str, list[dict]],
    data_dir: Path,
    output_dir: Path,
    device: str,
    manifest_hash: str | None = None,
) -> int:
    """Extract VideoMAE V2 embeddings for each 10s chunk.

    Preprocessing matches the official HuggingFace preprocessor_config.json:
    resize shortest edge to 224, center crop to 224x224, ImageNet normalize.
    Non-overlapping consecutive 16-frame clips -> [num_clips, 768].
    """
    from mac.encoders.video_mae import VideoMAEV2Encoder

    encoder = VideoMAEV2Encoder().to(device)
    extracted = 0

    for subj, chunks in tqdm(chunks_by_subject.items(), desc="Video subjects"):
        video_path = str(data_dir / subj / "pov.mp4")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            log.warning(f"Cannot open video: {video_path}")
            continue

        for c in chunks:
            save_path = output_dir / subj / f"segment_{c['global_seq']:04d}.pt"
            if save_path.exists():
                extracted += 1
                continue

            start_sec = c["start_90hz"] / 90.0
            frames_arr = read_frames(cap, start_sec, CHUNK_LEN_SEC)

            if len(frames_arr) < 16:
                log.warning(
                    f"Video {subj}/seq_{c['global_seq']}: only {len(frames_arr)} frames, skipping"
                )
                continue

            clips = make_consecutive_clips(frames_arr)
            if not clips:
                log.warning(f"Video {subj}/seq_{c['global_seq']}: no 16-frame clips, skipping")
                continue

            embeddings = []
            with torch.no_grad():
                for clip in clips:
                    emb = encoder(clip.unsqueeze(0).to(device))
                    embeddings.append(emb)
            emb = torch.cat(embeddings, dim=0)

            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "embedding": emb.cpu(),
                "segment_idx": c["global_seq"],
                "subject": subj,
                "label": c["emotion_label"],
                "emotion": c["emotion_name"],
                "manifest_hash": manifest_hash,
            }, save_path)
            extracted += 1

        cap.release()

    return extracted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Task-aware 10s segment extraction")
    parser.add_argument(
        "--config",
        default="Auxiliary/benchmarks/egoemotion/configs/egoemotion.yaml",
    )
    parser.add_argument(
        "--encoder",
        choices=["papagei_ppg", "pulseppg_ppg", "patchtst_eye", "inceptiontime", "ecg_founder", "video_mae_v2", "all"],
        default="all",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="data/embeddings/egoemotion/10s_task_aware")
    args = parser.parse_args()

    cfg = load_config(args.config)
    registry = ModalityRegistry(cfg["modalities"])
    device = args.device if torch.cuda.is_available() else "cpu"
    data_dir = Path(cfg["data_dir"])
    output_dir = Path(args.output_dir)

    # Load task_times
    tt = np.load(data_dir / "task_times.npy", allow_pickle=True).item()

    # Discover subjects (directories that are numeric and in task_times)
    subject_ids = sorted(
        d.name for d in data_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and d.name in tt
    )
    log.info(f"Found {len(subject_ids)} subjects")

    # Build chunks for all subjects and collect manifest rows
    chunks_by_subject: dict[str, list[dict]] = {}
    manifest_rows = []

    for subj in subject_ids:
        subject_tasks = tt[subj]
        subject_dir = data_dir / subj
        task_chunks = get_task_chunks(subject_tasks, subject_dir, subj)

        # Assign global_seq (sequential counter per subject across all tasks)
        for seq_idx, c in enumerate(task_chunks):
            c["global_seq"] = seq_idx

        chunks_by_subject[subj] = task_chunks

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
            })

    total_chunks = sum(len(v) for v in chunks_by_subject.values())
    log.info(f"Total chunks: {total_chunks} across {len(chunks_by_subject)} subjects")

    # Save manifest
    manifest_path = output_dir / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(manifest_path, index=False)
    manifest_hash = compute_manifest_hash(manifest_df)
    log.info(f"Manifest saved: {manifest_path} ({len(manifest_df)} rows, hash={manifest_hash})")

    # Extract embeddings
    encoders = resolve_encoder_dirs(registry, args.encoder)

    PPG_ENCODER_DIR_TO_KEY = {
        "papagei_ppg": "papagei",
        "pulseppg_ppg": "pulseppg",
    }

    for enc_name in encoders:
        log.info(f"\n=== Extracting {enc_name} ===")
        enc_dir = output_dir / enc_name

        if enc_name in PPG_ENCODER_DIR_TO_KEY:
            n = extract_ppg(chunks_by_subject, data_dir, enc_dir, device,
                            ppg_encoder=PPG_ENCODER_DIR_TO_KEY[enc_name],
                            manifest_hash=manifest_hash)
        elif enc_name in EYE_ENCODER_DIR_TO_KEY:
            n = extract_eye(
                chunks_by_subject,
                data_dir,
                enc_dir,
                device,
                eye_encoder_key=EYE_ENCODER_DIR_TO_KEY[enc_name],
                manifest_hash=manifest_hash,
            )
        elif enc_name == "ecg_founder":
            n = extract_ecg(chunks_by_subject, data_dir, enc_dir, device,
                            manifest_hash=manifest_hash)
        elif enc_name == "video_mae_v2":
            n = extract_video(chunks_by_subject, data_dir, enc_dir, device,
                              manifest_hash=manifest_hash)
        else:
            continue

        log.info(f"{enc_name}: {n} segments extracted")

    # Final count verification
    for enc_name in encoders:
        enc_dir = output_dir / enc_name
        count = sum(1 for _ in enc_dir.rglob("segment_*.pt")) if enc_dir.exists() else 0
        log.info(f"{enc_name}: {count} / {total_chunks} segments on disk")


if __name__ == "__main__":
    main()

"""Task-aware segmentation and embedding extraction (20s non-overlapping chunks).

Follows the official egoEMOTION segmentation approach:
1. Split by task first using task_times.npy
2. Within each task, chunk into 20s non-overlapping windows (1800 samples at 90Hz)
3. Discard incomplete trailing chunks
4. Exclude inter-task gaps (calibration, questionnaires)
5. One label per task — all chunks inherit the task's self-reported emotion label

This is a copy of ``segment_and_extract_10s.py`` with CHUNK_LEN_SEC=20 and the
default output directory changed to ``20s_task_aware``. All other logic is
unchanged; the only functional difference is that each segment now covers twice
as many frames and therefore typically yields ~12 consecutive 16-frame video
clips (vs ~6 at 10 s).

Usage:
    conda run -n visphy python scripts/segment_and_extract_20s.py --encoder video_mae_v2
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import torch
import cv2
from tqdm import tqdm

# Workaround for transformers>=5 + torch>=2.10 lazy-init regression:
# OpenGVLab's custom modeling_videomaev2.py does
#     dpr = [x.item() for x in torch.linspace(...)]
# inside __init__, which crashes under the meta-device dispatch that
# AutoModel.from_pretrained enters by default. Forcing low_cpu_mem_usage=False
# is not sufficient in transformers 5.x. Monkey-patch torch.linspace so that
# if it would return a meta tensor, we fall back to a CPU tensor instead.
_orig_linspace = torch.linspace

def _safe_linspace(*args, **kwargs):
    res = _orig_linspace(*args, **kwargs)
    if res.device.type == "meta":
        new_kwargs = {k: v for k, v in kwargs.items() if k != "device"}
        return _orig_linspace(*args, device="cpu", **new_kwargs)
    return res

torch.linspace = _safe_linspace

# Second transformers 5 incompatibility: modeling_utils.mark_tied_weights_as_initialized
# reads model.all_tied_weights_keys (dict), but custom models (OpenGVLab) still
# expose the older _tied_weights_keys list. Shim both shapes onto nn.Module.
from torch import nn as _nn

if not hasattr(_nn.Module, "all_tied_weights_keys"):
    @property
    def _all_tied_weights_keys(self):
        keys = getattr(self, "_tied_weights_keys", None) or []
        return {k: None for k in keys}

    _nn.Module.all_tied_weights_keys = _all_tied_weights_keys

# Third: force eager allocation so pos_embed (and other computed buffers) are
# not left on meta device. Combined with the linspace shim, this keeps
# model init entirely on the requested device. Requires accelerate installed.
import transformers as _tx
_orig_from_pretrained = _tx.AutoModel.from_pretrained

def _patched_from_pretrained(*args, **kwargs):
    kwargs.setdefault("low_cpu_mem_usage", False)
    return _orig_from_pretrained(*args, **kwargs)

_tx.AutoModel.from_pretrained = _patched_from_pretrained

from mac.data.video_transforms import make_consecutive_clips, read_frames
from mac.config.simple import load_config
from mac.encoders.registry import ModalityRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

CHUNK_LEN_SEC = 20
FS_ET = 90
CHUNK_SAMPLES_90HZ = CHUNK_LEN_SEC * FS_ET       # 1800
CHUNK_SAMPLES_125HZ = CHUNK_LEN_SEC * 125         # 2500

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
    """Enumerate tasks and chunk each into 20s windows.

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

        # Chunk into 20s windows at 90Hz
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

def extract_ppg(
    chunks_by_subject: dict[str, list[dict]],
    data_dir: Path,
    output_dir: Path,
    device: str,
    ppg_encoder: str = "papagei",
) -> int:
    """Extract PPG embeddings for each 20s chunk.

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
            }, save_path)
            extracted += 1

    return extracted


def extract_eye(
    chunks_by_subject: dict[str, list[dict]],
    data_dir: Path,
    output_dir: Path,
    device: str,
    eye_encoder_key: str,
) -> int:
    """Extract eye-tracking embeddings for each 20s chunk."""
    import os
    if eye_encoder_key == "patchtst":
        from mac.encoders.patchtst import PatchTSTEncoder

        encoder = PatchTSTEncoder(
            num_channels=4, patch_len=45, stride=22,
            d_model=128, n_heads=4, n_layers=3, seq_len=CHUNK_SAMPLES_90HZ,
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
            }, save_path)
            extracted += 1

    return extracted


def _materialize_meta_pos_embed(model: torch.nn.Module, device: str) -> None:
    """Walk model and replace any meta-device pos_embed with a freshly-computed
    sinusoidal table on the target device.

    Needed because under transformers>=5 lazy init, computed non-state-dict
    buffers (like the VideoMAE sinusoidal positional encoding) can survive the
    from_pretrained pipeline still pointing at meta memory.
    """
    for sub in model.modules():
        pe = getattr(sub, "pos_embed", None)
        if pe is None:
            continue
        if not hasattr(pe, "device"):
            continue
        if pe.device.type != "meta":
            continue
        # Recompute: shape is [1, num_patches, embed_dim]
        n_patches = pe.shape[1]
        d_hid = pe.shape[2]
        pos_i = np.arange(n_patches)[:, None].astype(np.float64)
        hid_j = np.arange(d_hid)[None, :]
        angle = pos_i / np.power(10000, 2 * (hid_j // 2) / d_hid)
        angle[:, 0::2] = np.sin(angle[:, 0::2])
        angle[:, 1::2] = np.cos(angle[:, 1::2])
        new_pe = torch.tensor(angle, dtype=torch.float32, device=device).unsqueeze(0)
        # Preserve Parameter-ness if that's what it was
        if isinstance(pe, torch.nn.Parameter):
            sub.pos_embed = torch.nn.Parameter(new_pe, requires_grad=pe.requires_grad)
        else:
            sub.pos_embed = new_pe


def extract_video(
    chunks_by_subject: dict[str, list[dict]],
    data_dir: Path,
    output_dir: Path,
    device: str,
) -> int:
    """Extract VideoMAE V2 embeddings for each 20s chunk.

    Preprocessing matches the official HuggingFace preprocessor_config.json:
    resize shortest edge to 224, center crop to 224x224, ImageNet normalize.
    Non-overlapping consecutive 16-frame clips -> [num_clips, 768].
    """
    from mac.encoders.video_mae import VideoMAEV2Encoder

    encoder = VideoMAEV2Encoder().to(device)
    _materialize_meta_pos_embed(encoder, device)
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
            }, save_path)
            extracted += 1

        cap.release()

    return extracted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Task-aware 20s segment extraction")
    parser.add_argument("--config", default="configs/egoemotion.yaml")
    parser.add_argument(
        "--encoder",
        choices=["papagei_ppg", "pulseppg_ppg", "patchtst_eye", "inceptiontime", "video_mae_v2", "all"],
        default="video_mae_v2",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="data/embeddings/egoemotion/20s_task_aware")
    parser.add_argument("--data-dir", default=None,
                        help="Override cfg['data_dir'] with an absolute path (useful when CWD differs from repo root)")
    parser.add_argument("--subjects", default=None,
                        help="Comma-separated subject IDs to restrict to (default: all)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    # Resolve config and data_dir relative to the repo root, not the caller's CWD.
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = repo_root / cfg_path
    cfg = load_config(str(cfg_path))
    registry = ModalityRegistry(cfg["modalities"])
    device = args.device if torch.cuda.is_available() else "cpu"

    if args.data_dir:
        data_dir = Path(args.data_dir).resolve()
    else:
        data_dir = Path(cfg["data_dir"])
        if not data_dir.is_absolute():
            data_dir = (repo_root / data_dir).resolve()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (repo_root / output_dir).resolve()

    # Load task_times
    tt = np.load(data_dir / "task_times.npy", allow_pickle=True).item()

    # Discover subjects (directories that are numeric and in task_times)
    if args.subjects:
        requested = {s.strip() for s in args.subjects.split(",") if s.strip()}
        subject_ids = sorted(s for s in requested if s in tt)
        missing = requested - set(subject_ids)
        if missing:
            log.warning(f"Subjects not in task_times, skipping: {sorted(missing)}")
    else:
        subject_ids = sorted(
            d.name for d in data_dir.iterdir()
            if d.is_dir() and d.name.isdigit() and d.name in tt
        )
    log.info(f"Using {len(subject_ids)} subjects: {subject_ids[:10]}{'...' if len(subject_ids) > 10 else ''}")

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
    log.info(f"Manifest saved: {manifest_path} ({len(manifest_df)} rows)")

    # Extract embeddings
    configured_eye_dir = registry.get_embedding_dir_name("eye_tracking")
    configured_ppg_dir = registry.get_embedding_dir_name("ppg")
    encoders = (
        [configured_ppg_dir, configured_eye_dir, "video_mae_v2"]
        if args.encoder == "all"
        else [args.encoder]
    )

    PPG_ENCODER_DIR_TO_KEY = {
        "papagei_ppg": "papagei",
        "pulseppg_ppg": "pulseppg",
    }

    for enc_name in encoders:
        log.info(f"\n=== Extracting {enc_name} ===")
        enc_dir = output_dir / enc_name

        if enc_name in PPG_ENCODER_DIR_TO_KEY:
            n = extract_ppg(chunks_by_subject, data_dir, enc_dir, device,
                            ppg_encoder=PPG_ENCODER_DIR_TO_KEY[enc_name])
        elif enc_name in EYE_ENCODER_DIR_TO_KEY:
            n = extract_eye(
                chunks_by_subject,
                data_dir,
                enc_dir,
                device,
                eye_encoder_key=EYE_ENCODER_DIR_TO_KEY[enc_name],
            )
        elif enc_name == "video_mae_v2":
            n = extract_video(chunks_by_subject, data_dir, enc_dir, device)
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

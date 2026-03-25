"""Segment data into 10s windows matching the paper's manifest and extract embeddings.

The egoEMOTION paper chunks each task into non-overlapping 10-second windows
(at 90Hz = 900 samples per chunk), dropping remainders. The manifest defines
exactly which segments are kept (2,678 across 40 subjects).

IMPORTANT: The resampled data files (gaze_90fps.npy, pupils_90fps.npy,
ppg_ear_125hz.npy) are SHIFTED to start from session_A, NOT from recording
start. Their length = session_B_end - session_A_start (in native Hz).
Segment index N means samples [N*900 : (N+1)*900] in the SHIFTED 90Hz files.
For video (pov.mp4), the file IS absolute, so we need session_A_start offset.

Usage:
    conda run -n visphy python scripts/segment_and_extract_10s.py --encoder papagei_ppg
    conda run -n visphy python scripts/segment_and_extract_10s.py --encoder patchtst_eye
    conda run -n visphy python scripts/segment_and_extract_10s.py --encoder video_mae_v2
    conda run -n visphy python scripts/segment_and_extract_10s.py --encoder all
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import cv2
from tqdm import tqdm

from src.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

CHUNK_SAMPLES_90HZ = 900   # 10s at 90Hz
CHUNK_SAMPLES_125HZ = 1250  # 10s at 125Hz
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def _parse_segment_idx(segment_path: str) -> int:
    """Extract integer segment index from manifest segment_path like '005/ppg_segments/5.p'."""
    return int(segment_path.split("/")[-1].replace(".p", ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract 10s segment embeddings matching paper manifest")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--encoder", choices=["papagei_ppg", "patchtst_eye", "video_mae_v2", "all"], default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="data/embeddings_10s")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = args.device if torch.cuda.is_available() else "cpu"
    data_dir = Path(cfg["data_dir"])
    output_dir = Path(args.output_dir)

    # Load manifest (2,678 segments)
    manifest = pd.read_csv(data_dir / "ce_hardlabel_manifests" / "dataset_manifest.csv")
    log.info(f"Manifest: {len(manifest)} segments across {manifest['subject'].nunique()} subjects")

    # Load task_times for session_A offset
    tt = np.load(data_dir / "task_times.npy", allow_pickle=True).item()

    encoders = ["papagei_ppg", "patchtst_eye", "video_mae_v2"] if args.encoder == "all" else [args.encoder]

    for enc_name in encoders:
        log.info(f"\n=== Extracting {enc_name} ===")
        enc_dir = output_dir / enc_name

        if enc_name == "papagei_ppg":
            extract_ppg(manifest, tt, data_dir, enc_dir, device)
        elif enc_name == "patchtst_eye":
            extract_eye(manifest, tt, data_dir, enc_dir, device)
        elif enc_name == "video_mae_v2":
            extract_video(manifest, tt, data_dir, enc_dir, device)

    # Final count verification
    for enc_name in encoders:
        enc_dir = output_dir / enc_name
        count = sum(1 for _ in enc_dir.rglob("segment_*.pt")) if enc_dir.exists() else 0
        log.info(f"{enc_name}: {count} / {len(manifest)} segments extracted")


def extract_ppg(
    manifest: pd.DataFrame,
    tt: dict,
    data_dir: Path,
    output_dir: Path,
    device: str,
) -> None:
    """Extract Papagei embeddings for each 10s segment."""
    from src.encoders.papagei import PapageiEncoder

    encoder = PapageiEncoder().to(device)
    extracted, skipped = 0, 0

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="PPG"):
        subj = str(row["subject"]).zfill(3)
        seg_idx = _parse_segment_idx(row["segment_path"])
        save_path = output_dir / subj / f"segment_{seg_idx:04d}.pt"
        if save_path.exists():
            extracted += 1
            continue

        # Load full PPG signal
        ppg_path = data_dir / subj / "ppg_ear_125hz.npy"
        if not ppg_path.exists():
            skipped += 1
            continue
        ppg = np.load(ppg_path)

        # Compute SHIFTED position (files start from session_A, not recording start)
        # Segment index N = samples [N*900 : (N+1)*900] in 90Hz shifted space
        start_90 = seg_idx * CHUNK_SAMPLES_90HZ
        end_90 = start_90 + CHUNK_SAMPLES_90HZ
        # Convert shifted 90Hz position to shifted 125Hz position
        start_125 = int(start_90 * 125 / 90)
        end_125 = start_125 + CHUNK_SAMPLES_125HZ  # exact 1250 samples

        if end_125 > len(ppg):
            skipped += 1
            log.debug(f"PPG {subj}/seg_{seg_idx}: end_125={end_125} > len={len(ppg)}, skipping")
            continue

        chunk = ppg[start_125:end_125]
        if len(chunk) < CHUNK_SAMPLES_125HZ:
            # Pad if slightly short due to rate conversion rounding
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES_125HZ - len(chunk)), mode="edge")
        chunk = chunk[:CHUNK_SAMPLES_125HZ]  # exact 10s at 125Hz

        # Z-score normalize per-segment (Papagei expects normalized input)
        std = chunk.std()
        if std > 0:
            chunk = (chunk - chunk.mean()) / std

        x = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)  # [1, T, 1]
        with torch.no_grad():
            emb = encoder(x)  # [1, 512]

        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "embedding": emb.cpu(),
            "segment_idx": seg_idx,
            "subject": subj,
            "label": int(row["label"]),
            "emotion": row["emotion"],
        }, save_path)
        extracted += 1

    log.info(f"PPG: {extracted} extracted, {skipped} skipped")


def extract_eye(
    manifest: pd.DataFrame,
    tt: dict,
    data_dir: Path,
    output_dir: Path,
    device: str,
) -> None:
    """Extract PatchTST embeddings for each 10s segment."""
    import os
    from src.encoders.patchtst import PatchTSTEncoder

    encoder = PatchTSTEncoder(
        num_channels=4, patch_len=45, stride=22,
        d_model=128, n_heads=4, n_layers=3, seq_len=900,
    ).to(device)
    pretrained = "checkpoints/patchtst_pretrained.pt"
    if os.path.exists(pretrained):
        encoder.load_state_dict(torch.load(pretrained, weights_only=True))
        log.info(f"Loaded pre-trained PatchTST from {pretrained}")
    encoder.freeze()

    extracted, skipped = 0, 0

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Eye"):
        subj = str(row["subject"]).zfill(3)
        seg_idx = _parse_segment_idx(row["segment_path"])
        save_path = output_dir / subj / f"segment_{seg_idx:04d}.pt"
        if save_path.exists():
            extracted += 1
            continue

        gaze_path = data_dir / subj / "gaze_90fps.npy"
        pupil_path = data_dir / subj / "pupils_90fps.npy"
        if not gaze_path.exists() or not pupil_path.exists():
            skipped += 1
            continue

        gaze = np.load(gaze_path)    # [N, 2]
        pupils = np.load(pupil_path)  # [N, 2]

        # Compute SHIFTED position (files start from session_A)
        start_90 = seg_idx * CHUNK_SAMPLES_90HZ
        end_90 = start_90 + CHUNK_SAMPLES_90HZ

        min_len = min(len(gaze), len(pupils))
        if end_90 > min_len:
            skipped += 1
            log.debug(f"Eye {subj}/seg_{seg_idx}: end_90={end_90} > len={min_len}, skipping")
            continue

        gaze_chunk = gaze[start_90:end_90]      # [900, 2]
        pupil_chunk = pupils[start_90:end_90]    # [900, 2]
        combined = np.concatenate([gaze_chunk, pupil_chunk], axis=1)  # [900, 4]

        x = torch.tensor(combined, dtype=torch.float32).unsqueeze(0).to(device)  # [1, 900, 4]
        with torch.no_grad():
            emb = encoder(x)       # [1, num_patches, 128]
            emb = emb.mean(dim=1)  # [1, 128]

        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "embedding": emb.cpu(),
            "segment_idx": seg_idx,
            "subject": subj,
            "label": int(row["label"]),
            "emotion": row["emotion"],
        }, save_path)
        extracted += 1

    log.info(f"Eye: {extracted} extracted, {skipped} skipped")


def extract_video(
    manifest: pd.DataFrame,
    tt: dict,
    data_dir: Path,
    output_dir: Path,
    device: str,
) -> None:
    """Extract VideoMAE embeddings for each 10s segment."""
    from src.encoders.video_mae import VideoMAEV2Encoder

    encoder = VideoMAEV2Encoder().to(device)
    extracted, skipped = 0, 0

    # Group by subject to avoid reopening video files
    for subj_id, subj_manifest in tqdm(
        manifest.groupby("subject"), desc="Video subjects"
    ):
        subj = str(subj_id).zfill(3)
        video_path = str(data_dir / subj / "pov.mp4")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            log.warning(f"Cannot open video: {video_path}")
            skipped += len(subj_manifest)
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)

        session_a_start_90 = tt[subj]["session_A"][0]

        for _, row in subj_manifest.iterrows():
            seg_idx = _parse_segment_idx(row["segment_path"])
            save_path = output_dir / subj / f"segment_{seg_idx:04d}.pt"
            if save_path.exists():
                extracted += 1
                continue

            # Video file (pov.mp4) is in ABSOLUTE time, so add session_A offset
            # (unlike gaze/ppg files which are shifted to start from session_A)
            abs_start_90 = session_a_start_90 + seg_idx * CHUNK_SAMPLES_90HZ
            start_sec = abs_start_90 / 90.0
            end_sec = start_sec + 10.0

            start_frame = int(start_sec * fps)
            end_frame = int(end_sec * fps)

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frames = []
            for _ in range(end_frame - start_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

            if len(frames) < 16:
                # Pad with last frame (or zeros if no frames at all)
                while len(frames) < 16:
                    frames.append(
                        frames[-1] if frames
                        else np.zeros((224, 224, 3), dtype=np.uint8)
                    )

            frames_arr = np.stack(frames)

            # Non-overlapping 16-frame clips, then mean-pool embeddings
            clips = []
            for i in range(0, len(frames_arr) - 15, 16):
                clip = frames_arr[i:i + 16].astype(np.float32) / 255.0
                clip = (clip - IMAGENET_MEAN) / IMAGENET_STD
                clip = torch.tensor(clip, dtype=torch.float32).permute(3, 0, 1, 2)  # [3, 16, H, W]
                clips.append(clip)

            if not clips:
                clips.append(torch.zeros(3, 16, 224, 224))

            embeddings = []
            with torch.no_grad():
                for clip in clips:
                    emb = encoder(clip.unsqueeze(0).to(device))  # [1, 768]
                    embeddings.append(emb)
            emb = torch.cat(embeddings, dim=0).mean(dim=0, keepdim=True)  # [1, 768]

            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "embedding": emb.cpu(),
                "segment_idx": seg_idx,
                "subject": subj,
                "label": int(row["label"]),
                "emotion": row["emotion"],
            }, save_path)
            extracted += 1

        cap.release()

    log.info(f"Video: {extracted} extracted, {skipped} skipped")


if __name__ == "__main__":
    main()

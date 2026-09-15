"""CLI for extracting and caching encoder embeddings."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on path
_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np
import torch
import cv2

from mac.data.video_transforms import make_consecutive_clips, preprocess_frame, normalize_clip
from mac.encoders.extract import EmbeddingExtractor
from mac.config.simple import load_config, config_hash

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="Auxiliary/benchmarks/egoemotion/configs/egoemotion.yaml",
    )
    parser.add_argument("--encoder", choices=["video_mae_v2", "patchtst_eye", "papagei_ppg", "all"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = load_config(args.config)
    extractor = EmbeddingExtractor(
        data_dir=cfg["data_dir"],
        output_dir=cfg["embeddings_dir"],
        task_times_path=f"{cfg['data_dir']}/task_times.npy",
    )

    device = args.device if torch.cuda.is_available() else "cpu"
    encoders_to_run = (
        ["video_mae_v2", "patchtst_eye", "papagei_ppg"]
        if args.encoder == "all"
        else [args.encoder]
    )

    for enc_name in encoders_to_run:
        log.info(f"Extracting: {enc_name}")
        if enc_name == "video_mae_v2":
            _extract_video(extractor, device)
        elif enc_name == "patchtst_eye":
            _extract_eye_tracking(extractor, cfg, device)
        elif enc_name == "papagei_ppg":
            _extract_ppg(extractor, device)


def _extract_video(extractor: EmbeddingExtractor, device: str) -> None:
    from mac.encoders.video_mae import VideoMAEV2Encoder

    encoder = VideoMAEV2Encoder().to(device)

    def encode_fn(subject_id: str, segment: dict) -> torch.Tensor:
        video_path = f"{extractor.segment_extractor.data_dir}/{subject_id}/pov.mp4"
        start_sec = segment["start_idx"] / 90.0
        duration_sec = (segment["end_idx"] - segment["start_idx"]) / 90.0
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        expected = int(duration_sec * fps)

        # Timestamp-based seeking (more reliable than frame-index for compressed video)
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)
        frames = []
        for _ in range(expected):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(preprocess_frame(frame))  # center crop + BGR->RGB
        cap.release()

        if len(frames) < 16:
            while len(frames) < 16:
                frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))

        frames_arr = np.stack(frames)
        clips = make_consecutive_clips(frames_arr)

        if not clips:
            clips.append(torch.zeros(3, 16, 224, 224))

        embeddings = []
        with torch.no_grad():
            for clip in clips:
                emb = encoder(clip.unsqueeze(0).to(device))
                embeddings.append(emb)
        return torch.cat(embeddings, dim=0)

    enc_hash = config_hash({"encoder": "VideoMAEV2", "embed_dim": 768})
    extractor.extract_all("video_mae_v2", encode_fn, device, config_hash=enc_hash)


def _extract_eye_tracking(
    extractor: EmbeddingExtractor, cfg: dict, device: str
) -> None:
    from mac.encoders.patchtst import PatchTSTEncoder
    import os

    encoder = PatchTSTEncoder(
        num_channels=4, patch_len=45, stride=22,
        d_model=128, n_heads=4, n_layers=3, seq_len=900,
    ).to(device)

    pretrained_path = "checkpoints/patchtst_pretrained.pt"
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(
            f"Pre-trained PatchTST checkpoint not found at {pretrained_path}. "
            "Run 'python scripts/pretrain_patchtst.py' first."
        )
    encoder.load_state_dict(torch.load(pretrained_path, weights_only=True))
    log.info(f"Loaded pre-trained PatchTST from {pretrained_path}")
    encoder.freeze()

    def encode_fn(subject_id: str, segment: dict) -> torch.Tensor:
        eye_data = extractor.segment_extractor.load_eye_tracking_segment(
            subject_id, segment["start_idx"], segment["end_idx"]
        )
        x = torch.tensor(eye_data, dtype=torch.float32)
        seq_len = 900
        chunks = []
        for i in range(0, len(x) - seq_len + 1, seq_len // 2):
            chunks.append(x[i : i + seq_len])
        if not chunks:
            padded = torch.zeros(seq_len, 4)
            padded[: len(x)] = x
            chunks.append(padded)

        batch = torch.stack(chunks).to(device)
        with torch.no_grad():
            embeddings = encoder(batch)
        return embeddings.reshape(-1, 128)

    enc_hash = config_hash({
        "encoder": "PatchTST", "num_channels": 4, "patch_len": 45,
        "stride": 22, "d_model": 128, "n_heads": 4, "n_layers": 3, "seq_len": 900,
    })
    extractor.extract_all("patchtst_eye", encode_fn, device, config_hash=enc_hash)


def _extract_ppg(extractor: EmbeddingExtractor, device: str) -> None:
    from mac.encoders.papagei import PapageiEncoder

    encoder = PapageiEncoder().to(device)

    def encode_fn(subject_id: str, segment: dict) -> torch.Tensor:
        ppg = extractor.segment_extractor.load_ppg_segment(
            subject_id, segment["start_idx"], segment["end_idx"]
        )
        x = torch.tensor(ppg.squeeze(), dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)  # [1, 1, T]
        with torch.no_grad():
            return encoder(x)

    enc_hash = config_hash({"encoder": "Papagei", "embed_dim": 512})
    extractor.extract_all("papagei_ppg", encode_fn, device, config_hash=enc_hash)


if __name__ == "__main__":
    main()

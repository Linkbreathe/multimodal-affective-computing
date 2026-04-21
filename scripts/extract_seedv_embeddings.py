"""Extract EEGPT embeddings from preprocessed SEED-V segments.

Reads the manifest CSV, loads each segment's EEG tensor, runs it through
the frozen EEGPTEncoder, and saves per-segment embedding .pt files grouped
by subject.  Also writes a new embeddings manifest CSV.

Supports configurable channel selection and window size for ablation studies.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

log = logging.getLogger(__name__)


def _owned_cpu_embedding(embedding: torch.Tensor) -> torch.Tensor:
    """Return a standalone CPU tensor so torch.save writes only this embedding."""
    return embedding.detach().to(device="cpu", copy=True).contiguous()


def _build_encoder(
    device: torch.device,
    channels: list[str] | None = None,
    window_samples: int = 2560,
    weights_path: str = "weights/eegpt/eegpt_mcae_58chs_4s_large4E.ckpt",
):
    """Instantiate frozen EEGPTEncoder and move to *device*."""
    from src.encoders.eegpt import EEGPTEncoder

    encoder = EEGPTEncoder(
        channels=channels,
        window_samples=window_samples,
        weights_path=weights_path,
    )
    encoder.to(device)
    return encoder


def extract_embeddings(
    manifest_path: str | Path,
    output_dir: str | Path,
    batch_size: int = 32,
    device: torch.device | None = None,
    channels: list[str] | None = None,
    window_samples: int = 2560,
    weights_path: str = "weights/eegpt/eegpt_mcae_58chs_4s_large4E.ckpt",
) -> pd.DataFrame:
    """Run the full extraction pipeline.

    Parameters
    ----------
    manifest_path : path to preprocessed segment manifest CSV
    output_dir    : root output directory for embeddings
    batch_size    : segments per forward pass
    device        : compute device
    channels      : EEG channel names to use (None = all 58 pretrained)
    window_samples: temporal input length in samples
    """
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    # ------------------------------------------------------------------
    # 1. Load manifest
    # ------------------------------------------------------------------
    df = pd.read_csv(manifest_path)
    required_cols = {"subject", "session", "trial", "segment_idx", "emotion_label", "emotion_name", "file_path"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")
    log.info("Loaded manifest with %d segments from %s", len(df), manifest_path)

    # ------------------------------------------------------------------
    # 2. Build encoder
    # ------------------------------------------------------------------
    encoder = _build_encoder(device, channels=channels, window_samples=window_samples, weights_path=weights_path)
    log.info("EEGPTEncoder ready on %s", device)

    # ------------------------------------------------------------------
    # 3. Process subjects
    # ------------------------------------------------------------------
    emb_records: list[dict] = []

    for subject_id, group in tqdm(df.groupby("subject"), desc="Subjects"):
        subj_dir = output_dir / "eegpt_eeg" / str(subject_id)
        subj_dir.mkdir(parents=True, exist_ok=True)

        # Collect segment tensors and metadata
        eeg_tensors: list[torch.Tensor] = []
        meta: list[dict] = []
        for _, row in group.iterrows():
            seg_path = Path(row["file_path"])
            item = torch.load(seg_path, map_location="cpu", weights_only=False)
            eeg_tensors.append(item["eeg"])  # [60, window_samples]
            meta.append({
                "subject": int(row["subject"]),
                "session": int(row["session"]),
                "trial": int(row["trial"]),
                "segment_idx": int(row["segment_idx"]),
                "label": int(row["emotion_label"]),
                "emotion_name": row["emotion_name"],
            })

        # Batch inference
        all_embeddings: list[torch.Tensor] = []
        for i in range(0, len(eeg_tensors), batch_size):
            batch = torch.stack(eeg_tensors[i : i + batch_size]).to(device)
            with torch.no_grad():
                emb = encoder(batch)  # [B, 2048]
            all_embeddings.append(emb.cpu())

        all_embeddings = torch.cat(all_embeddings, dim=0)  # [N, 2048]

        # Save per-segment (use sequential idx per subject)
        for idx, (emb_vec, m) in enumerate(zip(all_embeddings, meta)):
            out_name = f"segment_{idx:04d}.pt"
            out_path = subj_dir / out_name
            torch.save(
                {
                    "embedding": _owned_cpu_embedding(emb_vec),  # [2048]
                    "label": m["label"],
                    "subject": m["subject"],
                    "session": m["session"],
                    "trial": m["trial"],
                },
                out_path,
            )
            emb_records.append({
                "subject": m["subject"],
                "session": m["session"],
                "trial": m["trial"],
                "segment_idx": m["segment_idx"],
                "emotion_label": m["label"],
                "emotion_name": m["emotion_name"],
                "file_path": str(out_path),
            })

    # ------------------------------------------------------------------
    # 4. Write embeddings manifest
    # ------------------------------------------------------------------
    emb_df = pd.DataFrame(emb_records)
    manifest_out = output_dir / "manifest.csv"
    emb_df.to_csv(manifest_out, index=False)
    log.info("Saved embeddings manifest (%d rows) to %s", len(emb_df), manifest_out)

    return emb_df


def main():
    parser = argparse.ArgumentParser(
        description="Extract EEGPT embeddings from preprocessed SEED-V segments."
    )
    parser.add_argument(
        "--manifest",
        default="data/preprocessed/seedv_preprocessed/manifest.csv",
        help="Path to the preprocessed segment manifest CSV.",
    )
    parser.add_argument(
        "--output_dir",
        default="data/embeddings/seedv/base",
        help="Root directory for output embeddings.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for GPU inference.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device (cuda / cpu). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--weights_path",
        default="weights/eegpt/eegpt_mcae_58chs_4s_large4E.ckpt",
        help="Path to EEGPT pretrained checkpoint.",
    )
    parser.add_argument(
        "--channels",
        nargs="*",
        default=None,
        help="EEG channels to use (e.g., TP7 TP8). Default: all 58 pretrained.",
    )
    parser.add_argument(
        "--window_sec",
        type=int,
        default=10,
        help="Window duration in seconds (must match preprocessing). Default: 10.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    window_samples = args.window_sec * 256  # 256 Hz sampling rate

    extract_embeddings(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=device,
        channels=args.channels,
        window_samples=window_samples,
        weights_path=args.weights_path,
    )


if __name__ == "__main__":
    main()

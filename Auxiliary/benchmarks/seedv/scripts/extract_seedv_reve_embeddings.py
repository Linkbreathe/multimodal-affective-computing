"""Extract REVE embeddings from preprocessed SEED-V segments.

Reads the manifest CSV, loads each segment's EEG tensor, resamples from
256 Hz to 200 Hz (REVE's native rate), runs through the frozen ReveEncoder,
and saves per-segment embedding .pt files grouped by subject.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
import torchaudio.functional as AF
from tqdm import tqdm

log = logging.getLogger(__name__)

SRC_SR = 256  # SEED-V preprocessing sample rate
TGT_SR = 200  # REVE native sample rate


def _owned_cpu_embedding(embedding: torch.Tensor) -> torch.Tensor:
    """Return a standalone CPU tensor so torch.save writes only this embedding."""
    return embedding.detach().to(device="cpu", copy=True).contiguous()


def _build_encoder(
    device: torch.device,
    weights_path: str = "weights/reve/reve_base.safetensors",
    pos_bank_path: str = "weights/reve/reve_positions.safetensors",
):
    """Instantiate frozen ReveEncoder and move to *device*."""
    from mac.encoders.reve import ReveEncoder

    encoder = ReveEncoder(
        weights_path=weights_path,
        pos_bank_path=pos_bank_path,
    )
    encoder.to(device)
    return encoder


def extract_embeddings(
    manifest_path: str | Path,
    output_dir: str | Path,
    batch_size: int = 16,
    device: torch.device | None = None,
    weights_path: str = "weights/reve/reve_base.safetensors",
    pos_bank_path: str = "weights/reve/reve_positions.safetensors",
) -> pd.DataFrame:
    """Run the full extraction pipeline."""
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    # 1. Load manifest
    df = pd.read_csv(manifest_path)
    required_cols = {"subject", "session", "trial", "segment_idx", "emotion_label", "emotion_name", "file_path"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")
    log.info("Loaded manifest with %d segments from %s", len(df), manifest_path)

    # 2. Build encoder
    encoder = _build_encoder(device, weights_path=weights_path, pos_bank_path=pos_bank_path)
    log.info("ReveEncoder ready on %s", device)

    # 3. Process subjects
    emb_records: list[dict] = []

    for subject_id, group in tqdm(df.groupby("subject"), desc="Subjects"):
        subj_dir = output_dir / "reve_eeg" / str(subject_id)
        subj_dir.mkdir(parents=True, exist_ok=True)

        # Collect segment tensors and metadata
        eeg_tensors: list[torch.Tensor] = []
        meta: list[dict] = []
        for _, row in group.iterrows():
            seg_path = Path(row["file_path"])
            item = torch.load(seg_path, map_location="cpu", weights_only=False)
            eeg_tensors.append(item["eeg"])  # [60, 2560]
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
            batch = torch.stack(eeg_tensors[i : i + batch_size])  # [B, 60, 2560]
            # Resample 256 Hz -> 200 Hz
            batch = AF.resample(batch, SRC_SR, TGT_SR)  # [B, 60, 2000]
            batch = batch.to(device)
            with torch.no_grad():
                emb = encoder(batch)  # [B, 512]
            all_embeddings.append(emb.cpu())

        all_embeddings = torch.cat(all_embeddings, dim=0)  # [N, 512]

        # Save per-segment
        for idx, (emb_vec, m) in enumerate(zip(all_embeddings, meta)):
            out_name = f"segment_{idx:04d}.pt"
            out_path = subj_dir / out_name
            torch.save(
                {
                    "embedding": _owned_cpu_embedding(emb_vec),  # [512]
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

    # 4. Write embeddings manifest
    emb_df = pd.DataFrame(emb_records)
    manifest_out = output_dir / "reve_manifest.csv"
    emb_df.to_csv(manifest_out, index=False)
    log.info("Saved embeddings manifest (%d rows) to %s", len(emb_df), manifest_out)

    return emb_df


def main():
    parser = argparse.ArgumentParser(
        description="Extract REVE embeddings from preprocessed SEED-V segments."
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
        default=16,
        help="Batch size for GPU inference (REVE is large, default 16).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device (cuda / cpu). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--weights_path",
        default="weights/reve/reve_base.safetensors",
        help="Path to REVE pretrained checkpoint.",
    )
    parser.add_argument(
        "--pos_bank_path",
        default="weights/reve/reve_positions.safetensors",
        help="Path to REVE position bank checkpoint.",
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

    extract_embeddings(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=device,
        weights_path=args.weights_path,
        pos_bank_path=args.pos_bank_path,
    )


if __name__ == "__main__":
    main()

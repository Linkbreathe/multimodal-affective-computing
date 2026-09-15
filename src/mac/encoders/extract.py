"""Embedding extraction and caching pipeline."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from mac.data.segments import SegmentExtractor

log = logging.getLogger(__name__)


class EmbeddingExtractor:
    """Extracts and caches embeddings from frozen encoders."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        task_times_path: str,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.segment_extractor = SegmentExtractor(
            data_dir=data_dir,
            task_times_path=task_times_path,
        )

    def get_cache_path(
        self, encoder_name: str, subject_id: str, segment_idx: int
    ) -> Path:
        return self.output_dir / encoder_name / subject_id / f"segment_{segment_idx:04d}.pt"

    def is_cached(
        self, encoder_name: str, subject_id: str, segment_idx: int
    ) -> bool:
        return self.get_cache_path(encoder_name, subject_id, segment_idx).exists()

    def save_embedding(
        self,
        encoder_name: str,
        subject_id: str,
        segment_idx: int,
        embedding: torch.Tensor,
        metadata: dict[str, Any],
        config_hash: str = "",
    ) -> None:
        path = self.get_cache_path(encoder_name, subject_id, segment_idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "embedding": embedding,
            "metadata": metadata,
            "config_hash": config_hash,
        }, path)

    def load_embedding(
        self, encoder_name: str, subject_id: str, segment_idx: int
    ) -> dict[str, Any]:
        path = self.get_cache_path(encoder_name, subject_id, segment_idx)
        return torch.load(path, weights_only=False)

    def validate_cache(self, encoder_name: str, expected_hash: str) -> bool:
        """Check if cached embeddings match the expected config hash.

        Checks ALL cached files to catch mixed old/new caches.
        Returns True only if every file has a matching hash.
        """
        encoder_dir = self.output_dir / encoder_name
        if not encoder_dir.exists():
            return False
        pt_files = list(encoder_dir.rglob("*.pt"))
        if not pt_files:
            return False
        for pt_file in pt_files:
            data = torch.load(pt_file, weights_only=False)
            stored_hash = data.get("config_hash", "")
            if stored_hash != expected_hash:
                log.info(
                    f"Cache hash mismatch in {pt_file.name}: "
                    f"stored={stored_hash!r}, expected={expected_hash!r}"
                )
                return False
        return True

    def extract_all(
        self,
        encoder_name: str,
        encode_fn: callable,
        device: str = "cuda",
        config_hash: str = "",
    ) -> None:
        """Extract embeddings for all subjects and segments."""
        cache_valid = self.validate_cache(encoder_name, config_hash)
        if cache_valid:
            log.info(f"Cache valid for {encoder_name} (hash={config_hash}), skipping existing files")
        else:
            log.info(f"Cache invalid/missing for {encoder_name}, will re-extract all segments")

        subjects = self.segment_extractor.get_subject_ids()
        for subj in tqdm(subjects, desc=f"Extracting {encoder_name}"):
            segments = self.segment_extractor.get_segments(subj)
            for idx, seg in enumerate(segments):
                # Only skip if cache is valid AND file exists.
                # When cache is invalid (hash mismatch), re-extract even if file exists.
                if cache_valid and self.is_cached(encoder_name, subj, idx):
                    continue
                try:
                    embedding = encode_fn(subj, seg)
                    self.save_embedding(
                        encoder_name, subj, idx,
                        embedding.cpu(),
                        metadata=seg,
                        config_hash=config_hash,
                    )
                except Exception as e:
                    log.warning(f"Failed {encoder_name}/{subj}/seg_{idx}: {e}")

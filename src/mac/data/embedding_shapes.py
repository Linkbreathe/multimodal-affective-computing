"""Shape normalization shared by cached-embedding experiment runners."""

import torch


SEQUENCE_ENCODER_DIRS = {"video_mae_v2", "patchtst_eye"}


def normalize_loaded_embedding(
    embedding: torch.Tensor,
    encoder_dir_name: str,
) -> torch.Tensor:
    """Normalize cached embeddings without collapsing singleton sequences.

    Sequence encoders may legitimately emit a single token with shape [1, D].
    Keep that 2D shape intact so batching can still pad/stack sequence inputs.
    Pooled encoders such as PPG should continue to load as [D].
    """
    if embedding.dim() >= 3 and embedding.shape[0] == 1:
        return embedding.squeeze(0)

    if (
        encoder_dir_name not in SEQUENCE_ENCODER_DIRS
        and embedding.dim() == 2
        and embedding.shape[0] == 1
    ):
        return embedding.squeeze(0)

    return embedding

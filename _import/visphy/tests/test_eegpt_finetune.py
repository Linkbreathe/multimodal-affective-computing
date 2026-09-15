from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from src.encoders.eegpt import EEGPTEncoder
from src.encoders.base import BaseEncoder


class _DummyEEGTransformer(torch.nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        patch_stride,
        embed_num,
        embed_dim,
        **kwargs,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.patch_embed = torch.nn.Linear(patch_size, embed_dim)
        self.chan_embed = torch.nn.Embedding(64, embed_dim)
        self.summary_token = torch.nn.Parameter(torch.zeros(1, embed_num, embed_dim))
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(1, 1) for _ in range(8)])
        self.norm = torch.nn.LayerNorm(embed_dim)

    def prepare_chan_ids(self, channels):
        return torch.arange(len(channels)).unsqueeze(0)

    def forward(self, x, chan_ids=None, mask_x=None, mask_t=None):
        stride = self.patch_stride if self.patch_stride is not None else self.patch_size
        n_patches = ((x.shape[-1] - self.patch_size) // stride) + 1
        base = torch.arange(
            n_patches * self.embed_num * self.embed_dim,
            dtype=x.dtype,
            device=x.device,
        )
        return base.view(1, n_patches, self.embed_num, self.embed_dim).repeat(x.shape[0], 1, 1, 1)


class _DummyTokenEncoder(BaseEncoder):
    def __init__(self):
        super().__init__(embed_dim=2048)
        self.proj = torch.nn.Linear(62, 62)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Use forward_tokens for the fine-tune model")

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        return torch.ones(batch, 31, 4, 512, dtype=x.dtype, device=x.device)


def test_eegpt_encoder_requires_checkpoint_unless_random_init(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("src.encoders.eegpt.EEGTransformer", _DummyEEGTransformer)

    missing_ckpt = tmp_path / "missing-eegpt.ckpt"

    with pytest.raises(FileNotFoundError, match="EEGPT weights not found"):
        EEGPTEncoder(
            channels=["FP1", "FP2"],
            window_samples=128,
            weights_path=str(missing_ckpt),
        )


def test_eegpt_encoder_exposes_patch_features_and_random_init_override(monkeypatch):
    monkeypatch.setattr("src.encoders.eegpt.EEGTransformer", _DummyEEGTransformer)

    encoder = EEGPTEncoder(
        channels=["FP1", "FP2"],
        window_samples=128,
        patch_stride=32,
        weights_path=str(Path("/tmp/nonexistent-eegpt.ckpt")),
        random_init=True,
    )

    x = torch.randn(3, 62, 128)
    tokens = encoder.forward_features(x)
    pooled = encoder(x)

    assert encoder.target_encoder.patch_stride == 32
    assert tokens.shape == (3, 3, 2048)
    assert pooled.shape == (3, 2048)


def test_eegpt_encoder_selective_train_keeps_transformer_in_train_mode(monkeypatch):
    monkeypatch.setattr("src.encoders.eegpt.EEGTransformer", _DummyEEGTransformer)

    encoder = EEGPTEncoder(
        channels=["FP1", "FP2"],
        window_samples=128,
        finetune=True,
        weights_path=str(Path("/tmp/nonexistent-eegpt.ckpt")),
        random_init=True,
    )

    encoder.freeze()
    encoder.unfreeze()
    encoder.selective_train()

    assert encoder.training is True
    assert encoder.target_encoder.training is True


def test_finetune_head_forward_shape():
    """AvgPoolClassificationHead produces correct output shape."""
    from src.models.eegpt_finetune_head import AvgPoolClassificationHead

    head = AvgPoolClassificationHead(
        input_dim=2048,
        num_classes=5,
        hidden_dims=[128],
        dropout=0.3,
    )
    # Simulate per-patch features from EEGPT
    x = torch.randn(2, 16, 2048)  # [B, n_patches, D]
    logits = head(x)
    assert logits.shape == (2, 5)


def test_finetune_encoder_head_end_to_end(monkeypatch):
    """Encoder + head produce correct logits shape end-to-end."""
    monkeypatch.setattr("src.encoders.eegpt.EEGTransformer", _DummyEEGTransformer)
    from src.models.eegpt_finetune_head import AvgPoolClassificationHead

    encoder = EEGPTEncoder(
        channels=["FP1", "FP2"],
        window_samples=128,
        finetune=True,
        weights_path=str(Path("/tmp/nonexistent-eegpt.ckpt")),
        random_init=True,
    )
    head = AvgPoolClassificationHead(input_dim=2048, num_classes=5)

    x = torch.randn(2, 62, 128)
    features = encoder.forward_features(x)  # [B, n_patches, 2048]
    logits = head(features)                  # [B, 5]
    assert logits.shape == (2, 5)


def test_finetune_layer_groups_have_correct_structure(monkeypatch):
    """get_layer_groups() returns proper param groups for optimizer."""
    monkeypatch.setattr("src.encoders.eegpt.EEGTransformer", _DummyEEGTransformer)

    encoder = EEGPTEncoder(
        channels=["FP1", "FP2"],
        window_samples=128,
        finetune=True,
        weights_path=str(Path("/tmp/nonexistent-eegpt.ckpt")),
        random_init=True,
    )
    groups = encoder.get_layer_groups()
    assert len(groups) > 0
    for g in groups:
        assert "params" in g
        assert "lr_scale" in g
        assert "name" in g
        assert len(g["params"]) > 0


def test_seedv_manifest_loader_groups_segment_files_by_subject(tmp_path: Path):
    from scripts.run_seedv_finetune import load_manifest

    seg_a = tmp_path / "001_seg_0.pt"
    seg_b = tmp_path / "001_seg_1.pt"
    seg_c = tmp_path / "002_seg_0.pt"
    for idx, path in enumerate([seg_a, seg_b, seg_c]):
        torch.save(
            {
                "eeg": torch.randn(62, 1024),
                "label": idx % 5,
                "subject": 1 if idx < 2 else 2,
            },
            path,
        )

    manifest = pd.DataFrame(
        [
            {
                "subject": 1,
                "session": 1,
                "trial": 0,
                "segment_idx": 0,
                "emotion_label": 0,
                "emotion_name": "Disgust",
                "file_path": str(seg_a),
            },
            {
                "subject": 1,
                "session": 1,
                "trial": 0,
                "segment_idx": 1,
                "emotion_label": 1,
                "emotion_name": "Fear",
                "file_path": str(seg_b),
            },
            {
                "subject": 2,
                "session": 1,
                "trial": 1,
                "segment_idx": 0,
                "emotion_label": 2,
                "emotion_name": "Sad",
                "file_path": str(seg_c),
            },
        ]
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    grouped = load_manifest(manifest_path)

    assert sorted(grouped.keys()) == [1, 2]
    assert len(grouped[1]["files"]) == 2
    assert grouped[1]["files"][0] == seg_a
    assert grouped[2]["labels"][0] == 2

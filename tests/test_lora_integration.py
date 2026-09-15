from __future__ import annotations

from pathlib import Path

import torch

from mac.encoders.eegpt import EEGPTEncoder
from mac.models.eegpt_finetune_head import AvgPoolClassificationHead
from mac.models.lora import LoRALinear


class _DummyAttention(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.qkv = torch.nn.Linear(dim, dim * 3)
        self.proj = torch.nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, _, _ = self.qkv(x).chunk(3, dim=-1)
        return self.proj(q)


class _DummyMLP(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, dim * 2)
        self.fc2 = torch.nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class _DummyBlock(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim)
        self.attn = _DummyAttention(dim)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.mlp = _DummyMLP(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class _DummyEEGTransformer(torch.nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        patch_stride,
        embed_num,
        embed_dim,
        depth,
        **kwargs,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.patch_embed = torch.nn.Linear(patch_size, embed_dim)
        self.chan_embed = torch.nn.Embedding(64, embed_dim)
        self.summary_token = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.token_proj = torch.nn.Linear(patch_size, embed_dim)
        self.blocks = torch.nn.ModuleList([_DummyBlock(embed_dim) for _ in range(depth)])
        self.norm = torch.nn.LayerNorm(embed_dim)

    def prepare_chan_ids(self, channels):
        return torch.arange(len(channels)).unsqueeze(0)

    def forward(self, x: torch.Tensor, chan_ids=None, mask_x=None, mask_t=None) -> torch.Tensor:
        stride = self.patch_stride if self.patch_stride is not None else self.patch_size
        patches = x.mean(dim=1).unfold(-1, self.patch_size, stride)
        tokens = self.token_proj(patches)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return tokens.unsqueeze(2).expand(-1, -1, self.embed_num, -1)


def _build_encoder(monkeypatch) -> EEGPTEncoder:
    monkeypatch.setattr("mac.encoders.eegpt.EEGTransformer", _DummyEEGTransformer)
    return EEGPTEncoder(
        channels=["FP1", "FP2"],
        window_samples=128,
        finetune=True,
        patch_stride=32,
        weights_path=str(Path("/tmp/nonexistent-eegpt.ckpt")),
        random_init=True,
    )


def _lora_config() -> dict:
    return {
        "use_lora": True,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_target_modules": ["default"],
        "lora_lr_scale": 1.0,
    }


def test_inject_lora_wraps_eegpt_target_layers(monkeypatch):
    encoder = _build_encoder(monkeypatch)

    injected = encoder.inject_lora(_lora_config())
    lora_modules = [
        name for name, module in encoder.target_encoder.named_modules()
        if isinstance(module, LoRALinear)
    ]

    assert len(injected) == len(encoder.target_encoder.blocks) * 4
    assert injected == lora_modules


def test_lora_mode_only_lora_and_head_params_are_trainable(monkeypatch):
    encoder = _build_encoder(monkeypatch)
    head = AvgPoolClassificationHead(input_dim=2048, num_classes=5)
    encoder.inject_lora(_lora_config())

    model = torch.nn.ModuleDict({"encoder": encoder, "head": head})
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}

    assert any(name.startswith("head.") for name in trainable)
    assert any(name.startswith("encoder.") and "lora_" in name for name in trainable)
    assert not any(name.startswith("encoder.") and "lora_" not in name for name in trainable)


def test_lora_forward_preserves_output_shapes(monkeypatch):
    encoder = _build_encoder(monkeypatch)
    encoder.inject_lora(_lora_config())

    x = torch.randn(2, 62, 128)
    tokens = encoder.forward_features(x)
    pooled = encoder(x)

    assert tokens.shape == (2, 3, 2048)
    assert pooled.shape == (2, 2048)


def test_lora_state_dict_roundtrip(monkeypatch):
    source = _build_encoder(monkeypatch)
    source.inject_lora(_lora_config())
    for name, param in source.named_parameters():
        if "lora_A" in name:
            torch.nn.init.constant_(param, 0.25)
        if "lora_B" in name:
            torch.nn.init.constant_(param, 0.5)

    state = source.get_lora_state_dict()

    target = _build_encoder(monkeypatch)
    target.inject_lora(_lora_config())
    target.load_lora_state_dict(state)

    for name, param in target.named_parameters():
        if "lora_" in name:
            assert torch.equal(param, source.state_dict()[name])

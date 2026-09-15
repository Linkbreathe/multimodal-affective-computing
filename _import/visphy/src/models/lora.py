"""Self-contained LoRA utilities for EEGPT fine-tuning."""
from __future__ import annotations

import math
from collections import OrderedDict
from typing import Iterable, Sequence

import torch
import torch.nn as nn


EEGPT_LORA_TARGETS: dict[str, list[str]] = {
    "default": ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
    "attention": ["attn.qkv", "attn.proj"],
    "ffn": ["mlp.fc1", "mlp.fc2"],
    "full": ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
}

REVE_LORA_TARGETS: dict[str, list[str]] = {
    "reve_default": ["to_qkv", "to_out"],
    "reve_attention": ["to_qkv", "to_out"],
    "reve_full": ["to_qkv", "to_out"],
}


class LoRALayer(nn.Module):
    """Low-rank update for a linear projection."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {r}")

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling


class LoRALinear(nn.Module):
    """Wrap an existing linear layer with a LoRA residual."""

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.lora = LoRALayer(
            in_features=base_layer.in_features,
            out_features=base_layer.out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        # Move LoRA params to the same device as the base layer
        device = base_layer.weight.device
        self.lora = self.lora.to(device)
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x) + self.lora(x)

    @property
    def weight(self) -> torch.Tensor:
        delta = (self.lora.lora_B @ self.lora.lora_A) * self.lora.scaling
        return self.base_layer.weight + delta

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base_layer.bias


def resolve_lora_targets(target_modules: Sequence[str] | None) -> list[str]:
    """Expand convenience aliases into module suffixes (EEGPT or REVE)."""
    if not target_modules:
        target_modules = ["default"]

    # Merge both registries for lookup
    all_targets = {**EEGPT_LORA_TARGETS, **REVE_LORA_TARGETS}

    resolved: list[str] = []
    for target in target_modules:
        resolved.extend(all_targets.get(target, [target]))
    return list(OrderedDict.fromkeys(resolved))


def find_target_linear_modules(
    model: nn.Module,
    target_modules: Sequence[str] | None,
) -> dict[str, nn.Linear]:
    """Find linear modules whose names match one of the target suffixes."""
    targets = resolve_lora_targets(target_modules)
    matches: dict[str, nn.Linear] = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(name == target or name.endswith(f".{target}") for target in targets):
            matches[name] = module
    return matches


def inject_lora(
    model: nn.Module,
    target_modules: Sequence[str] | None,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
) -> tuple[nn.Module, list[str]]:
    """Replace matched linear layers with LoRA-enabled wrappers."""
    linear_layers = find_target_linear_modules(model, target_modules)
    injected_modules: list[str] = []

    for module_path, base_layer in linear_layers.items():
        if isinstance(base_layer, LoRALinear):
            injected_modules.append(module_path)
            continue

        parent_path, _, child_name = module_path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(
            parent,
            child_name,
            LoRALinear(
                base_layer=base_layer,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            ),
        )
        injected_modules.append(module_path)

    return model, injected_modules


def get_lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return the trainable LoRA parameters in a model."""
    params: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.extend([module.lora.lora_A, module.lora.lora_B])
    return params


def count_lora_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in get_lora_parameters(model))


def get_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Serialize only LoRA tensors."""
    lora_state_dict: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state_dict[f"{name}.lora.lora_A"] = module.lora.lora_A.detach().clone()
            lora_state_dict[f"{name}.lora.lora_B"] = module.lora.lora_B.detach().clone()
    return lora_state_dict


def load_lora_state_dict(
    model: nn.Module,
    lora_state_dict: dict[str, torch.Tensor],
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    """Load LoRA tensors into a model that already has LoRA wrappers."""
    missing_keys: list[str] = []
    unexpected_keys = list(lora_state_dict.keys())

    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue

        a_key = f"{name}.lora.lora_A"
        b_key = f"{name}.lora.lora_B"

        if a_key in lora_state_dict:
            module.lora.lora_A.data.copy_(lora_state_dict[a_key])
            unexpected_keys.remove(a_key)
        elif strict:
            missing_keys.append(a_key)

        if b_key in lora_state_dict:
            module.lora.lora_B.data.copy_(lora_state_dict[b_key])
            unexpected_keys.remove(b_key)
        elif strict:
            missing_keys.append(b_key)

    if strict and (missing_keys or unexpected_keys):
        raise RuntimeError(
            "Error loading LoRA state dict:\n"
            f"  Missing keys: {missing_keys}\n"
            f"  Unexpected keys: {unexpected_keys}"
        )

    return missing_keys, unexpected_keys


def is_lora_parameter(name: str) -> bool:
    return "lora_A" in name or "lora_B" in name


def iter_lora_named_parameters(model: nn.Module) -> Iterable[tuple[str, nn.Parameter]]:
    for name, parameter in model.named_parameters():
        if is_lora_parameter(name):
            yield name, parameter

"""Common contract for pretrained modality encoders.

An encoder can be used in two different ways in this repository: frozen feature
extraction, or selective fine-tuning in a research experiment.  The methods in
this class make that choice explicit and keep it separate from fusion logic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseEncoder(ABC, nn.Module):
    """Base class for all modality encoders.

    Supports both frozen inference and selective fine-tuning via
    :meth:`unfreeze`, :meth:`get_layer_groups`, and :meth:`selective_train`.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self._embed_dim = embed_dim

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def freeze(self) -> None:
        """Freeze all parameters and set to eval mode."""
        # ``eval`` is paired with requires_grad=False so dropout and BatchNorm
        # also behave deterministically during frozen embedding extraction.
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def unfreeze(self, from_layer: int | None = None) -> None:
        """Unfreeze parameters for fine-tuning.

        Args:
            from_layer: If given, only unfreeze layers >= this index.
                Subclasses override to implement encoder-specific logic.
                If ``None``, unfreeze everything.
        """
        if from_layer is not None:
            raise NotImplementedError(
                f"{type(self).__name__} does not implement layer-specific "
                "unfreezing. Override unfreeze() to support from_layer."
            )
        # Subclasses may override this to expose layer-wise unfreezing.  The
        # default deliberately makes the all-or-nothing behavior explicit.
        for param in self.parameters():
            param.requires_grad = True

    def get_layer_groups(self) -> list[dict]:
        """Return named parameter groups for layer-wise LR decay.

        Each dict has keys: ``"name"`` (str), ``"params"`` (list of Parameters),
        ``"lr_scale"`` (float, multiplier relative to base LR).

        Subclasses override to provide encoder-specific grouping.
        """
        return [
            {
                "name": "encoder_all",
                "params": list(self.parameters()),
                "lr_scale": 1.0,
            }
        ]

    def selective_train(self) -> None:
        """Put model in train mode but keep BatchNorm layers in eval mode.

        This is critical for fine-tuning with batch_size=1 (LOSO edge case),
        where BatchNorm running stats would be degenerate. By keeping BN in
        eval mode, we use the pretrained running mean/var instead.
        """
        # Small LOSO folds can have batch_size=1; updating BN statistics from a
        # single sample would corrupt the pretrained representation.
        self.train()
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()

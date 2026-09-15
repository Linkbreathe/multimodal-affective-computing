"""TensorBoard logging helpers."""
from __future__ import annotations

from torch.utils.tensorboard import SummaryWriter


class TBLogger:
    def __init__(self, log_dir: str) -> None:
        self.writer = SummaryWriter(log_dir)

    def log_scalars(self, tag: str, values: dict[str, float], step: int) -> None:
        for k, v in values.items():
            self.writer.add_scalar(f"{tag}/{k}", v, step)

    def log_hparams(self, hparams: dict, metrics: dict) -> None:
        self.writer.add_hparams(hparams, metrics)

    def close(self) -> None:
        self.writer.close()

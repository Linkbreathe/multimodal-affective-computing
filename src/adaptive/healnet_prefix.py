"""Frozen HealNet checkpoint loading and strictly causal prefix inference."""

from __future__ import annotations

from argparse import Namespace
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from scripts.run_relax_foundation_probe import AlignedNativeFusionRegressor


EXPECTED_MODALITIES = ("eeg", "ecg", "eye", "head", "video")
EXPECTED_FUSION = "healnet"


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FrozenHealNetEnsemble:
    """Three independently seeded, fold-local HealNet models in eval mode."""

    def __init__(self, checkpoint_paths: Sequence[str | Path], device: str | torch.device = "cpu") -> None:
        if len(checkpoint_paths) != 3:
            raise ValueError(f"The frozen ensemble requires exactly three checkpoints; received {len(checkpoint_paths)}")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA inference was requested but CUDA is unavailable")
        self.models: list[torch.nn.Module] = []
        self.scalers: list[dict[str, tuple[torch.Tensor, torch.Tensor]]] = []
        self.records: list[dict[str, Any]] = []
        reference: dict[str, Any] | None = None
        for raw_path in checkpoint_paths:
            path = Path(raw_path).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Frozen HealNet checkpoint not found: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self._validate_checkpoint(path, payload)
            comparable = {
                "embed_dims": dict(payload["embed_dims"]),
                "modalities": tuple(payload["modalities"]),
                "fusion_config": dict(payload["fusion_config"]),
                "fold": dict(payload["fold"]),
            }
            if reference is None:
                reference = comparable
            elif comparable != reference:
                raise ValueError(f"Ensemble checkpoint contract differs at {path}")
            args = Namespace(**payload["fusion_config"])
            model = AlignedNativeFusionRegressor(dict(payload["embed_dims"]), args)
            model.load_state_dict(payload["state_dict"], strict=True)
            model.to(self.device).eval()
            scalers = {
                modality: (
                    torch.as_tensor(payload["scalers"][modality]["mean"], dtype=torch.float32, device=self.device),
                    torch.as_tensor(payload["scalers"][modality]["scale"], dtype=torch.float32, device=self.device),
                )
                for modality in EXPECTED_MODALITIES
            }
            self.models.append(model)
            self.scalers.append(scalers)
            self.records.append(
                {
                    "path": str(path),
                    "sha256": file_sha256(path),
                    "seed": int(payload["seed"]),
                    "fold": dict(payload["fold"]),
                    "modalities": list(payload["modalities"]),
                    "embed_dims": {key: int(value) for key, value in payload["embed_dims"].items()},
                    "fusion_config": dict(payload["fusion_config"]),
                }
            )
        seeds = [record["seed"] for record in self.records]
        if len(set(seeds)) != 3:
            raise ValueError(f"Ensemble seeds are not distinct: {seeds}")
        self.modalities = EXPECTED_MODALITIES
        self.embed_dims = dict(reference["embed_dims"]) if reference else {}

    @staticmethod
    def _validate_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
        required = {"state_dict", "embed_dims", "modalities", "fusion_config", "seed", "fold", "scalers"}
        missing = required - set(payload)
        if missing:
            raise ValueError(f"Checkpoint {path} lacks fields: {sorted(missing)}")
        if tuple(payload["modalities"]) != EXPECTED_MODALITIES:
            raise ValueError(f"Checkpoint {path} is not the frozen five-modality model")
        if payload["fusion_config"].get("fusion") != EXPECTED_FUSION:
            raise ValueError(f"Checkpoint {path} is not HealNet")
        fold = payload["fold"]
        if int(fold.get("fold_index", -1)) != 5 or str(fold.get("test_participant")) != "P009":
            raise ValueError(f"Checkpoint {path} is not fold 05 with P009 held out")
        if "P009" in set(str(value) for value in fold.get("train_participants", ())):
            raise ValueError(f"Checkpoint {path} leaks P009 into training")
        if set(payload["scalers"]) != set(EXPECTED_MODALITIES):
            raise ValueError(f"Checkpoint {path} has a non-matching scaler set")

    def predict_prefix(
        self,
        embeddings: Mapping[str, torch.Tensor | np.ndarray],
        masks: Mapping[str, torch.Tensor | np.ndarray],
        prefix_length: int,
    ) -> dict[str, Any]:
        if prefix_length < 1:
            raise ValueError("Causal prefix length must be at least one")
        predictions = []
        with torch.inference_mode():
            for model, scalers in zip(self.models, self.scalers, strict=True):
                model_embeddings: dict[str, torch.Tensor] = {}
                model_masks: dict[str, torch.Tensor] = {}
                for modality in self.modalities:
                    values = torch.as_tensor(embeddings[modality], dtype=torch.float32)
                    mask = torch.as_tensor(masks[modality], dtype=torch.bool)
                    if values.ndim != 2 or mask.ndim != 1 or len(values) != len(mask):
                        raise ValueError(f"Malformed sequence for {modality}: {tuple(values.shape)}, {tuple(mask.shape)}")
                    if prefix_length > len(values):
                        raise ValueError(f"Prefix {prefix_length} exceeds {modality} sequence length {len(values)}")
                    values = values[:prefix_length].to(self.device)
                    mask = mask[:prefix_length].to(self.device)
                    mean, scale = scalers[modality]
                    scaled = (values - mean) / scale
                    model_embeddings[modality] = torch.where(
                        torch.isfinite(scaled), scaled, torch.zeros_like(scaled)
                    ).unsqueeze(0)
                    model_masks[modality] = mask.unsqueeze(0)
                prediction = model(model_embeddings, model_masks).squeeze(0).detach().cpu().numpy().astype(float)
                predictions.append(prediction)
        stacked = np.stack(predictions)
        output: dict[str, Any] = {
            "pred_relaxation": float(stacked[:, 0].mean()),
            "pred_discomfort": float(stacked[:, 1].mean()),
            "pred_relaxation_seed_std": float(stacked[:, 0].std(ddof=0)),
            "pred_discomfort_seed_std": float(stacked[:, 1].std(ddof=0)),
        }
        for record, prediction in zip(self.records, predictions, strict=True):
            seed = int(record["seed"])
            output[f"pred_relaxation_seed_{seed}"] = float(prediction[0])
            output[f"pred_discomfort_seed_{seed}"] = float(prediction[1])
        return output


__all__ = ["EXPECTED_MODALITIES", "FrozenHealNetEnsemble", "file_sha256"]

from __future__ import annotations

# ruff: noqa: E402

from argparse import Namespace
from pathlib import Path
import sys

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.run_relax_foundation_probe import (
    AlignedLateRegressor,
    AlignedNativeFusionRegressor,
    _device,
    _load_split_manifest,
    _modality_variant,
)
from scripts.build_relax_alignment_cache import _local_path, _read_logged_csv
from mac.data.relax_dataset import RelaxConditionEmbeddingDataset


def _manifest(path: Path, participants: list[str]) -> None:
    rows = []
    for fold_index, test in enumerate(participants, start=1):
        validation = participants[fold_index % len(participants)]
        for participant in participants:
            rows.append(
                {
                    "fold_index": fold_index,
                    "test_participant": test,
                    "validation_participant": validation,
                    "participant_id": participant,
                    "role": (
                        "test"
                        if participant == test
                        else "validation"
                        if participant == validation
                        else "train"
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_formal_manifest_is_exactly_13_1_1(tmp_path):
    participants = [f"P{index:03d}" for index in range(2, 17)]
    path = tmp_path / "splits.csv"
    _manifest(path, participants)

    folds = _load_split_manifest(path, participants, strict=True)

    assert len(folds) == 15
    assert all(len(fold.train_participants) == 13 for fold in folds)
    assert sorted(fold.test_participant for fold in folds) == participants
    assert sorted(fold.validation_participant for fold in folds) == participants


def test_formal_nine_participant_manifest_is_exactly_7_1_1(tmp_path):
    participants = ["P003", "P004", "P007", "P008", "P009", "P011", "P012", "P013", "P015"]
    path = tmp_path / "splits.csv"
    _manifest(path, participants)

    folds = _load_split_manifest(path, participants, strict=True)

    assert len(folds) == 9
    assert all(len(fold.train_participants) == 7 for fold in folds)
    assert sorted(fold.test_participant for fold in folds) == participants
    assert sorted(fold.validation_participant for fold in folds) == participants


@pytest.mark.parametrize(
    ("modalities", "expected"),
    [
        (("eeg", "ecg", "eye", "head", "video"), "full"),
        (("ecg", "eye", "head", "video"), "no_eeg"),
        (("eeg", "eye", "head", "video"), "no_ecg"),
        (("eeg", "ecg", "head", "video"), "no_eye"),
        (("eeg", "ecg", "eye", "video"), "no_head"),
        (("eeg", "ecg", "eye", "head"), "no_video"),
    ],
)
def test_modality_variant_names_every_single_modality_ablation(modalities, expected):
    assert _modality_variant(modalities) == expected


def test_dataset_ands_internal_and_external_masks(tmp_path):
    cache_path = tmp_path / "cache.pt"
    torch.save(
        {
            "participant_ids": ["P002", "P003"],
            "conditions": ["C1", "C1"],
            "presentation_positions": [1, 1],
            "targets": torch.tensor([[0.2, 0.8], [0.3, 0.7]]),
            "embeddings": {"eeg": torch.ones(2, 2, 3), "ecg": torch.ones(2, 2, 4)},
            "masks": {
                "eeg": torch.tensor([[True, True], [True, False]]),
                "ecg": torch.tensor([[True, True], [True, True]]),
            },
        },
        cache_path,
    )
    mask_path = tmp_path / "masks.csv"
    pd.DataFrame(
        [
            {"participant_id": "P002", "condition": "C1", "condition_window_index": 0, "eeg_valid": True, "ecg_valid": True},
            {"participant_id": "P002", "condition": "C1", "condition_window_index": 1, "eeg_valid": False, "ecg_valid": True},
            {"participant_id": "P003", "condition": "C1", "condition_window_index": 0, "eeg_valid": True, "ecg_valid": False},
            {"participant_id": "P003", "condition": "C1", "condition_window_index": 1, "eeg_valid": True, "ecg_valid": True},
        ]
    ).to_csv(mask_path, index=False)

    dataset = RelaxConditionEmbeddingDataset(
        cache_path,
        modalities=["eeg", "ecg"],
        mask_manifest=mask_path,
        strict=True,
    )

    assert dataset.masks["eeg"].tolist() == [[True, False], [True, False]]
    assert dataset.masks["ecg"].tolist() == [[True, True], [False, True]]


def test_late_regressor_accepts_samples_with_a_fully_missing_modality():
    model = AlignedLateRegressor({"eeg": 3, "ecg": 4}, d_common=8, dropout=0.0)
    embeddings = {
        "eeg": torch.randn(2, 3, 3),
        "ecg": torch.randn(2, 3, 4),
    }
    masks = {
        "eeg": torch.tensor([[False, False, False], [True, False, True]]),
        "ecg": torch.tensor([[True, True, True], [True, True, True]]),
    }

    output = model(embeddings, masks)

    assert output.shape == (2, 2)
    assert torch.all((0.0 <= output) & (output <= 1.0))


def _native_fusion_args(fusion: str) -> Namespace:
    return Namespace(
        fusion=fusion,
        d_common=8,
        dropout=0.0,
        fusion_depth=1,
        fusion_heads=2,
        cross_attn_freq=1,
        latent_channels=4,
        latent_dim=8,
        dim_head=4,
        lego_mode="merge-sum",
    )


@pytest.mark.parametrize("fusion", ["early", "mid", "qformer", "healnet", "mm_lego"])
def test_aligned_native_fusions_handle_fully_missing_samples(fusion):
    model = AlignedNativeFusionRegressor(
        {"eeg": 3, "ecg": 4},
        _native_fusion_args(fusion),
    )
    embeddings = {
        "eeg": torch.randn(2, 3, 3),
        "ecg": torch.randn(2, 3, 4),
    }
    masks = {
        "eeg": torch.tensor([[False, False, False], [True, False, True]]),
        "ecg": torch.tensor([[True, True, True], [True, True, True]]),
    }

    output = model(embeddings, masks)
    output.sum().backward()

    assert output.shape == (2, 2)
    assert torch.isfinite(output).all()
    assert torch.all((0.0 <= output) & (output <= 1.0))
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_require_cuda_rejects_cpu_before_training():
    args = Namespace(device="cpu", require_cuda=True)
    with pytest.raises(RuntimeError, match="requires --device cuda"):
        _device(args)


def test_windows_source_path_maps_to_wsl_mount():
    path = _local_path(r"C:\Users\linki\dataset\sample.xdf")
    assert path.as_posix() == "/mnt/c/Users/linki/dataset/sample.xdf"


def test_logged_csv_reader_sniffs_semicolon_and_decimal_comma(tmp_path):
    path = tmp_path / "video_frames.csv"
    path.write_text("unix_time_ms;relative_path\n1780570000000,5;frames/a.jpg\n", encoding="utf-8")
    frame = _read_logged_csv(path)
    assert frame.loc[0, "unix_time_ms"] == 1780570000000.5
    assert frame.loc[0, "relative_path"] == "frames/a.jpg"

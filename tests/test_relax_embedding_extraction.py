from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.relax_foundation.extract_relax_foundation_embeddings import (
    _fixed_reve_embedding_dim,
    _merge_window_manifest,
    _resample_last_axis,
    _safe_zscore,
    _select_relax_eeg,
    assemble_condition_cache,
    copy_non_eeg_window_cache,
)
from mac.data.relax_foundation import (
    RELAX_EEG_LEFT_INDICES,
    RELAX_EEG_MONTAGE,
    RELAX_EEG_RIGHT_INDICES,
    RELAX_EEG_XDF_COLUMNS,
    RelaxHardFailure,
)


def test_resample_relax_eeg_500hz_to_reve_200hz_window_length():
    x = torch.randn(2, 4, 5000)

    out = _resample_last_axis(x, source_rate=500.0, target_rate=200.0)

    assert out.shape == (2, 4, 2000)


def test_safe_zscore_keeps_constant_windows_finite():
    x = torch.ones(3, 2, 10)

    out = _safe_zscore(x)

    assert torch.isfinite(out).all()
    assert torch.allclose(out, torch.zeros_like(out))


def test_relax_eeg_column_selection_and_reve_order_are_exact():
    samples = np.arange(30, dtype=float).reshape(5, 6)
    selected = _select_relax_eeg(samples)

    assert RELAX_EEG_XDF_COLUMNS == (1, 2, 3, 4)
    assert RELAX_EEG_MONTAGE == ("M2", "TP9", "TP10", "M1")
    assert RELAX_EEG_LEFT_INDICES == (1, 3)
    assert RELAX_EEG_RIGHT_INDICES == (0, 2)
    assert selected.shape == (4, 5)
    assert np.array_equal(selected, samples[:, [1, 2, 3, 4]].T)


def test_reve_output_is_deterministically_reduced_to_1024_dimensions():
    native = torch.arange(2 * 1216, dtype=torch.float32).reshape(2, 1216)
    reduced = _fixed_reve_embedding_dim(native)

    assert reduced.shape == (2, 1024)
    assert torch.isfinite(reduced).all()
    assert torch.equal(reduced, _fixed_reve_embedding_dim(native))


def test_window_manifest_merges_modalities_without_overwriting_eeg(tmp_path):
    first = _merge_window_manifest(tmp_path, {"eeg": {"reve_montage": list(RELAX_EEG_MONTAGE)}})
    second = _merge_window_manifest(tmp_path, {"attention_video": {"source": "fresh_extraction"}})

    assert first["modalities"]["eeg"]["reve_montage"] == list(RELAX_EEG_MONTAGE)
    assert second["modalities"]["eeg"] == first["modalities"]["eeg"]
    assert "attention_video" in second["modalities"]


def test_copy_non_eeg_cache_is_checksum_verified_and_excludes_eeg(tmp_path):
    source = tmp_path / "legacy"
    destination = tmp_path / "corrected"
    item = {
        "embedding": torch.arange(4, dtype=torch.float32),
        "valid": True,
        "metadata": {"source": "legacy"},
    }
    (source / "ecg" / "P003").mkdir(parents=True)
    torch.save(item, source / "ecg" / "P003" / "C1_w000.pt")

    manifest = copy_non_eeg_window_cache(
        source_root=source,
        destination_root=destination,
        modalities=["ecg"],
    )

    copied = destination / "ecg" / "P003" / "C1_w000.pt"
    assert copied.exists()
    assert manifest["modalities"]["ecg"]["file_count"] == 1
    assert manifest["modalities"]["ecg"]["files"][0]["sha256"]
    with pytest.raises(RelaxHardFailure, match="must not copy legacy EEG"):
        copy_non_eeg_window_cache(
            source_root=source,
            destination_root=destination,
            modalities=["eeg"],
        )


def test_assemble_condition_cache_rejects_ppg(tmp_path):
    with pytest.raises(RelaxHardFailure, match="PPG/Papagei"):
        assemble_condition_cache(
            relax_run_dir=tmp_path,
            window_cache_root=tmp_path,
            modalities=["ecg", "ppg"],
            output_path=tmp_path / "out.pt",
        )

import numpy as np

from auxiliary.benchmarks.seedv.scripts.run_seedv_eeg_eye_fusion import (
    _fit_fold_local_eeg_pca,
    run_loso_pca_concat,
)


def test_fold_local_pca_scaler_fits_training_subjects_only():
    all_eeg = np.array([
        [0.0, 1.0, 2.0],
        [1.0, 2.0, 3.0],
        [100.0, 101.0, 102.0],
        [101.0, 102.0, 103.0],
    ])
    subjects = np.array([1, 1, 2, 2])
    train_mask = subjects != 2
    test_mask = subjects == 2

    train_pca, test_pca, scaler, pca, explained = _fit_fold_local_eeg_pca(
        all_eeg,
        train_mask,
        test_mask,
        n_components=1,
    )

    assert np.allclose(scaler.mean_, all_eeg[train_mask].mean(axis=0))
    assert not np.allclose(scaler.mean_, all_eeg.mean(axis=0))
    assert train_pca.shape == (2, 1)
    assert test_pca.shape == (2, 1)
    assert pca.n_components_ == 1
    assert explained > 0.0


def test_run_loso_pca_concat_refits_per_fold():
    all_eye = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 2.0],
        [1.0, 2.0],
    ])
    all_eeg = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [100.0, 100.0, 100.0],
        [101.0, 100.0, 100.0],
    ])
    labels = np.array([0, 1, 0, 1, 0, 1])
    subjects = np.array([1, 1, 2, 2, 3, 3])

    result = run_loso_pca_concat(
        all_eye,
        all_eeg,
        labels,
        subjects,
        n_components=1,
    )

    assert 0.0 <= result["bal_acc"] <= 1.0
    assert len(result["per_subject_acc"]) == 3
    assert "pca_variance_explained_mean" in result

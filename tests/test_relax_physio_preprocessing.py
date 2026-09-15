from __future__ import annotations

import numpy as np

from src.data.relax_physio_preprocessing import (
    crop_indexes_with_context,
    exact_resample,
    linked_mastoid_reference,
    neurorvq_eeg_filter,
    session_zscore_clip,
    window_zscore,
)


def test_linked_mastoid_reference_precedes_normalization() -> None:
    raw = np.asarray(
        [
            [2.0, 4.0, 6.0],  # M2
            [10.0, 12.0, 14.0],  # TP9
            [20.0, 24.0, 28.0],  # TP10
            [6.0, 8.0, 10.0],  # M1
        ]
    )
    linked = linked_mastoid_reference(raw)
    np.testing.assert_allclose(linked[0], [6.0, 6.0, 6.0])
    np.testing.assert_allclose(linked[1], [16.0, 18.0, 20.0])


def test_exact_resample_and_window_zscore_contract() -> None:
    phase = np.linspace(0.0, 8.0 * np.pi, 5000, endpoint=False)
    values = np.stack((np.sin(phase), 2.0 * np.cos(phase)))
    resampled = exact_resample(values, 2000)
    standardized = window_zscore(resampled)
    assert resampled.shape == (2, 2000)
    assert standardized.dtype == np.float32
    np.testing.assert_allclose(standardized.mean(axis=-1), 0.0, atol=1e-6)
    np.testing.assert_allclose(standardized.std(axis=-1), 1.0, atol=1e-5)


def test_official_neurorvq_filter_is_finite_and_clipped() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(scale=50.0, size=(2, 7000))
    values[:, 3500] = 100000.0
    filtered = neurorvq_eeg_filter(values, 500.0, clip_uv=500.0)
    assert filtered.shape == values.shape
    assert np.isfinite(filtered).all()
    assert float(np.abs(filtered).max()) <= 500.0


def test_session_zscore_uses_channelwise_statistics_and_clips() -> None:
    values = np.stack((np.arange(1000), 10.0 + 2.0 * np.arange(1000))).astype(float)
    values[:, -1] = 1e9
    standardized, mean, scale = session_zscore_clip(values, clip_standard_deviations=3.0)
    assert standardized.shape == values.shape
    assert mean.shape == (2,)
    assert scale.shape == (2,)
    assert float(np.abs(standardized).max()) <= 3.0


def test_crop_indexes_with_context_centers_the_requested_window() -> None:
    timestamps = np.arange(0.0, 20.0, 0.002)
    outer, inner = crop_indexes_with_context(
        timestamps, 5.0, 15.0, context_seconds=2.0
    )
    assert outer.start == 1500
    assert outer.stop == 8500
    assert inner.start == 1000
    assert inner.stop == 6000


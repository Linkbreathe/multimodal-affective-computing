"""Tests for Papagei-style PPG preprocessing."""
import numpy as np
import pytest

from mac.preprocessing.ppg import (
    chebyshev_bandpass_filter,
    compute_segment_quality,
    detect_native_sample_rate,
    preprocess_ppg,
    resample_to_target,
)

FS = 256


@pytest.fixture
def synthetic_ppg():
    """1-Hz sine (heart-rate band) + 0.1-Hz drift + 50-Hz noise, 30 seconds."""
    t = np.linspace(0, 30, 30 * FS, endpoint=False)
    return np.sin(2 * np.pi * 1.0 * t) + 0.3 * np.sin(2 * np.pi * 0.1 * t) + 0.1 * np.sin(2 * np.pi * 50 * t)


def test_filter_preserves_length(synthetic_ppg):
    out = chebyshev_bandpass_filter(synthetic_ppg, FS)
    assert len(out) == len(synthetic_ppg)


def test_filter_removes_out_of_band(synthetic_ppg):
    out = chebyshev_bandpass_filter(synthetic_ppg, FS)
    freqs = np.fft.rfftfreq(len(out), 1 / FS)
    spectrum = np.abs(np.fft.rfft(out))
    # Energy at 0.1 Hz (below 0.5 Hz cutoff) should be attenuated
    idx_01 = np.argmin(np.abs(freqs - 0.1))
    # Energy at 1 Hz (in passband) should be preserved
    idx_1 = np.argmin(np.abs(freqs - 1.0))
    assert spectrum[idx_1] > 10 * spectrum[idx_01], "0.1 Hz should be attenuated vs 1 Hz"
    # Energy at 50 Hz (above 12 Hz cutoff) should be attenuated
    idx_50 = np.argmin(np.abs(freqs - 50))
    assert spectrum[idx_1] > 10 * spectrum[idx_50], "50 Hz should be attenuated vs 1 Hz"


def test_resample_output_length():
    signal = np.random.randn(256 * 10)  # 10s at 256 Hz
    out = resample_to_target(signal, 256, 125)
    expected = int(len(signal) * 125 / 256)
    assert abs(len(out) - expected) <= 1


def test_resample_identity():
    signal = np.random.randn(1250)
    out = resample_to_target(signal, 125, 125)
    np.testing.assert_array_equal(out, signal)


def test_quality_flags_constant():
    signal = np.ones(125 * 30)  # 30s constant signal
    quality = compute_segment_quality(signal, 125, segment_s=10.0)
    assert len(quality) == 3
    assert not quality.any(), "Constant signal should be flagged as bad"


def test_quality_passes_good_signal():
    t = np.linspace(0, 30, 125 * 30, endpoint=False)
    signal = np.sin(2 * np.pi * 1.0 * t)
    quality = compute_segment_quality(signal, 125, segment_s=10.0)
    assert quality.all(), "Sinusoidal signal should pass quality check"


def test_detect_native_sample_rate():
    # Simulate 256 Hz raw and 90 fps versions
    assert detect_native_sample_rate(1126344, 395979) == 256


def test_detect_rate_unknown_raises():
    with pytest.raises(ValueError, match="not in known rates"):
        detect_native_sample_rate(100, 100)  # ratio=1 → 90 Hz, not in known


def test_full_pipeline_dimensions(synthetic_ppg):
    out, meta = preprocess_ppg(synthetic_ppg, fs_native=FS, fs_target=125)
    expected = int(len(synthetic_ppg) * 125 / FS)
    assert abs(len(out) - expected) <= 1
    assert meta["fs_native"] == 256
    assert meta["fs_target"] == 125

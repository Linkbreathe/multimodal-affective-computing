"""Part B - high-quality offline EEG preprocessing primitives.

Provides: zero-phase filtered views of a condition segment (analytic 1-45,
broadband notched 1-100), bipolar temporal derivations (TP9-M1 left,
TP10-M2 right), per-subject IAF, and per-window band-power / spectral features.
Single-window gates (robust range, IMU motion, headset-off) are here.

Montage note: the four raw columns are [M2, TP9, TP10, M1] = [R, L, R, L].
M1/M2 are mastoid REFERENCE electrodes; the cortical signals are TP9 (left)
and TP10 (right). The bipolar view below is the primary, correctly-referenced
montage (ipsilateral mastoid). Per-channel features on the raw m1/m2 columns are
reference channels, not cortical signals, and should be treated as such downstream.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal, stats

from mac.features.common import hjorth, spectral_entropy, safe_divide

from analysis.idiographic import common as C

# eeg raw column order is M2, TP9, TP10, M1  (indices 0,1,2,3)
# Ipsilateral-mastoid bipolar derivations: left = TP9 - M1, right = TP10 - M2.
LEFT = (1, 3)   # TP9 - M1  (left cortical - left mastoid)
RIGHT = (2, 0)  # TP10 - M2 (right cortical - right mastoid)


@dataclass
class CondViews:
    analytic: np.ndarray      # (n,4) zero-phase 1-45
    broadband: np.ndarray     # (n,4) zero-phase 1-100 + notch
    bipolar: np.ndarray       # (n,2) [left=TP9-M1, right=TP10-M2], 1-45
    fs: float


def condition_views(seg: np.ndarray, cfg: dict[str, Any]) -> CondViews:
    fs = float(cfg["sample_rate_hz"])
    fl = cfg["filter"]
    analytic = C.zerophase_bandpass(seg, fl["analytic_band"][0], fl["analytic_band"][1], fs, fl["order"])
    broad = C.zerophase_bandpass(seg, fl["broadband_band"][0], fl["broadband_band"][1], fs, fl["order"])
    broad = C.zerophase_notch(broad, fl["notch_hz"], fs)
    # bipolar from the (detrended) raw seg, then band-limit
    bip_raw = np.column_stack([seg[:, LEFT[0]] - seg[:, LEFT[1]], seg[:, RIGHT[0]] - seg[:, RIGHT[1]]])
    bip = C.zerophase_bandpass(bip_raw, fl["analytic_band"][0], fl["analytic_band"][1], fs, fl["order"])
    return CondViews(analytic=analytic, broadband=broad, bipolar=bip, fs=fs)


# --------------------------------------------------------------------------- #
# per-subject individual alpha frequency
# --------------------------------------------------------------------------- #
def estimate_iaf(analytic_segments: list[np.ndarray], cfg: dict[str, Any]) -> float | None:
    if not analytic_segments:
        return None
    fs = float(cfg["sample_rate_hz"])
    psds = []
    freqs = None
    for seg in analytic_segments:
        f, p = signal.welch(seg, fs=fs, nperseg=int(min(len(seg), fs * 4)), axis=0)
        psds.append(p.mean(axis=1))
        freqs = f
    psd = np.mean(psds, axis=0)
    lo, hi = cfg["iaf_search_hz"]
    fit = cfg["slope_fit_hz"]
    m = (freqs >= fit[0]) & (freqs <= fit[1]) & (psd > 0) & ~((freqs >= 7) & (freqs <= 14))
    if m.sum() < 5:
        return None
    slope, intercept = np.polyfit(np.log10(freqs[m]), np.log10(psd[m]), 1)
    band = (freqs >= lo) & (freqs <= hi) & (psd > 0)
    if not band.any():
        return None
    resid = np.log10(psd[band]) - (slope * np.log10(freqs[band]) + intercept)
    if np.max(resid) <= 0.02:  # no real alpha bump
        return None
    return float(freqs[band][int(np.argmax(resid))])


# --------------------------------------------------------------------------- #
# spectral features for one window/channel
# --------------------------------------------------------------------------- #
def _bandpowers(x: np.ndarray, fs: float, bands: dict[str, list[float]]) -> tuple[dict[str, float], float]:
    f, p = signal.welch(x, fs=fs, nperseg=int(min(len(x), fs * 2)))
    total_mask = (f >= 1.0) & (f <= 45.0)
    total = float(np.trapezoid(p[total_mask], f[total_mask]))
    out = {}
    for name, (lo, hi) in bands.items():
        mask = (f >= lo) & (f < hi)
        out[name] = float(np.trapezoid(p[mask], f[mask])) if mask.any() else float("nan")
    return out, total


def _aperiodic_exponent(x: np.ndarray, fs: float, fit_hz, alpha_excl=(7.0, 14.0)) -> float:
    f, p = signal.welch(x, fs=fs, nperseg=int(min(len(x), fs * 2)))
    m = (f >= fit_hz[0]) & (f <= fit_hz[1]) & (p > 0) & ~((f >= alpha_excl[0]) & (f <= alpha_excl[1]))
    if m.sum() < 5:
        return float("nan")
    slope = np.polyfit(np.log10(f[m]), np.log10(p[m]), 1)[0]
    return float(slope)


def _bandpower_range(x: np.ndarray, fs: float, lo: float, hi: float) -> float:
    f, p = signal.welch(x, fs=fs, nperseg=int(min(len(x), fs * 2)))
    m = (f >= lo) & (f <= min(hi, fs * 0.45))
    return float(np.trapezoid(p[m], f[m])) if m.any() else float("nan")


def window_eeg_features(views: CondViews, l: int, r: int, cfg: dict[str, Any], iaf: float | None) -> dict[str, float]:
    """Raw (un-normalized) per-window EEG features. Normalization vs baseline and
    per-subject z-scoring happens later in features_offline.py."""
    fs = views.fs
    bands = {k: list(v) for k, v in cfg["bands"].items()}
    names = ["m2", "tp9", "tp10", "m1"]
    out: dict[str, float] = {}
    ana = views.analytic[l:r]
    broad = views.broadband[l:r]
    bip = views.bipolar[l:r]
    if len(ana) < int(fs * 2):
        return {}

    alpha_by_ch: list[float] = []
    for ci, nm in enumerate(names):
        bp, total = _bandpowers(ana[:, ci], fs, bands)
        for band_name, val in bp.items():
            out[f"eeg_{nm}_{band_name}_abs"] = val
            out[f"eeg_{nm}_{band_name}_rel"] = safe_divide(val, total)
        alpha_by_ch.append(bp["alpha"])
        act, mob, comp = hjorth(ana[:, ci])
        out[f"eeg_{nm}_hjorth_mobility"] = mob
        out[f"eeg_{nm}_hjorth_complexity"] = comp
        f, psd = signal.welch(ana[:, ci], fs=fs, nperseg=int(min(len(ana), fs * 2)))
        out[f"eeg_{nm}_spectral_entropy"] = spectral_entropy(psd[(f >= 1) & (f <= 45)])
        out[f"eeg_{nm}_aperiodic_slope"] = _aperiodic_exponent(ana[:, ci], fs, cfg["slope_fit_hz"])
        # broadband EMG proxy (temporalis) from the notched 1-100 view
        emg = _bandpower_range(broad[:, ci], fs, cfg["emg_band_hz"][0], cfg["emg_band_hz"][1])
        out[f"eeg_{nm}_emg_proxy"] = emg

    # IAF-relative alpha (per channel mean)
    if iaf is not None:
        hw = cfg["iaf_half_width_hz"]
        iaf_alpha = [_bandpower_range(ana[:, ci], fs, iaf - hw, iaf + hw) for ci in range(4)]
        out["eeg_iaf_alpha_abs"] = float(np.nanmean(iaf_alpha))

    # bipolar temporal features (primary montage)
    for bi, side in enumerate(["templeft", "tempright"]):
        bp, total = _bandpowers(bip[:, bi], fs, bands)
        for band_name, val in bp.items():
            out[f"eeg_{side}_{band_name}_abs"] = val
            out[f"eeg_{side}_{band_name}_rel"] = safe_divide(val, total)
        out[f"eeg_{side}_emg_proxy"] = _bandpower_range(bip[:, bi], fs, cfg["emg_band_hz"][0], cfg["emg_band_hz"][1])

    # global ratios + temporal L-R asymmetry (explicitly temporal, NOT frontal)
    a = np.nanmean([out.get(f"eeg_{n}_alpha_abs", np.nan) for n in names])
    b = np.nanmean([out.get(f"eeg_{n}_beta_abs", np.nan) for n in names])
    t = np.nanmean([out.get(f"eeg_{n}_theta_abs", np.nan) for n in names])
    out["eeg_alpha_beta_ratio"] = safe_divide(a, b)
    out["eeg_theta_beta_ratio"] = safe_divide(t, b)
    # TP9 (left) and TP10 (right) are the cortical signals; M1/M2 are references.
    left_a = out.get("eeg_tp9_alpha_abs", np.nan)
    right_a = out.get("eeg_tp10_alpha_abs", np.nan)
    out["eeg_temporal_alpha_asym_log_rl"] = float(np.log(right_a + 1e-12) - np.log(left_a + 1e-12))
    return out


def window_quality(ana_win: np.ndarray, cfg: dict[str, Any]) -> dict[str, float]:
    """Single-window QC quantities used for artifact flagging."""
    rng = np.percentile(ana_win, 95, axis=0) - np.percentile(ana_win, 5, axis=0)
    var = float(np.mean(np.var(ana_win, axis=0)))
    kurt = float(np.mean(stats.kurtosis(ana_win, axis=0, fisher=True)))
    return {"robust_range_max_uV": float(np.max(rng)), "win_var": var, "win_kurtosis": kurt}

"""Part A - outcome-blind subject selection.

Every metric here is blind to intensity/frequency/subjective labels; it only
measures signal quality. Thresholds are frozen in config.yaml. Output ranks
subjects and freezes the EEG / ECG subsets BEFORE any modeling.

Run:
  PY -m analysis.idiographic.selection            # all subjects -> artifacts/subject_selection.{json,csv}
  PY -m analysis.idiographic.selection --participant P003 --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
from typing import Any

import numpy as np
from scipy import signal

from real_time_ml.features.physio import detect_r_peaks, ecg_features, peak_f1

from analysis.idiographic import common as C


# --------------------------------------------------------------------------- #
# EEG quality primitives
# --------------------------------------------------------------------------- #
def _block_coverage(filt: np.ndarray, fs: float, cfg: dict[str, Any]) -> float:
    sel = cfg["selection"]
    block = max(1, int(round(sel["block_seconds"] * fs)))
    if len(filt) < block:
        return 0.0
    good = []
    for start in range(0, len(filt) - block + 1, block):
        piece = filt[start:start + block]
        std = np.std(piece, axis=0)
        p99 = np.percentile(np.abs(piece), 99, axis=0)
        ok = np.isfinite(piece).all(axis=0) & (std >= sel["eeg_flat_std_uV_min"]) & (p99 <= sel["eeg_abs_uV_max"])
        good.append(float(np.mean(ok) >= sel["block_channel_frac"]))
    return float(np.mean(good)) if good else 0.0


def _mean_psd(segments: list[np.ndarray], fs: float, band=(1.0, 100.0)) -> tuple[np.ndarray, np.ndarray]:
    """Average Welch PSD (channel-mean) across detrended condition segments."""
    psds = []
    freqs = None
    for seg in segments:
        x = signal.detrend(seg, axis=0)
        f, p = signal.welch(x, fs=fs, nperseg=int(min(len(x), fs * 4)), axis=0)
        psds.append(p.mean(axis=1))
        freqs = f
    return freqs, np.mean(psds, axis=0)


def _aperiodic_slope(freqs: np.ndarray, psd: np.ndarray, fit_hz, alpha_excl=(7.0, 14.0)) -> tuple[float, float]:
    m = (freqs >= fit_hz[0]) & (freqs <= fit_hz[1]) & ~((freqs >= alpha_excl[0]) & (freqs <= alpha_excl[1])) & (psd > 0)
    if m.sum() < 5:
        return float("nan"), float("nan")
    lf = np.log10(freqs[m])
    lp = np.log10(psd[m])
    slope, intercept = np.polyfit(lf, lp, 1)
    pred = slope * lf + intercept
    ss_res = np.sum((lp - pred) ** 2)
    ss_tot = np.sum((lp - lp.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(r2)


def _iaf(freqs: np.ndarray, psd: np.ndarray, search=(7.0, 13.0), fit_hz=(2.0, 40.0)) -> tuple[float, float]:
    """Alpha peak frequency + prominence above the aperiodic fit (in log10 units)."""
    m = (freqs >= fit_hz[0]) & (freqs <= fit_hz[1]) & (psd > 0)
    if m.sum() < 5:
        return float("nan"), float("nan")
    excl = ~((freqs >= 7.0) & (freqs <= 14.0))
    fm = m & excl
    slope, intercept = np.polyfit(np.log10(freqs[fm]), np.log10(psd[fm]), 1)
    band = (freqs >= search[0]) & (freqs <= search[1]) & (psd > 0)
    if not band.any():
        return float("nan"), float("nan")
    resid = np.log10(psd[band]) - (slope * np.log10(freqs[band]) + intercept)
    k = int(np.argmax(resid))
    return float(freqs[band][k]), float(resid[k])


def _line_ratio(freqs: np.ndarray, psd: np.ndarray) -> float:
    ln = (freqs >= 49) & (freqs <= 51)
    nb = (freqs >= 40) & (freqs <= 48)
    if not ln.any() or not nb.any():
        return float("nan")
    return float(np.trapezoid(psd[ln], freqs[ln]) / (np.trapezoid(psd[nb], freqs[nb]) + 1e-12))


# --------------------------------------------------------------------------- #
# ECG quality
# --------------------------------------------------------------------------- #
def _ecg_metrics_condition(ecg_seg: np.ndarray, fs: float) -> dict[str, float]:
    feats, qc = ecg_features(ecg_seg, fs)
    hr = feats.get("ecg_hr_bpm", float("nan"))
    quality = qc.get("ecg_quality", float("nan"))
    # cross-check with neurokit2 if available
    f1 = float("nan")
    try:
        import neurokit2 as nk

        peaks_ours, _ = detect_r_peaks(ecg_seg, fs)
        clean = nk.ecg_clean(ecg_seg, sampling_rate=int(fs))
        _, info = nk.ecg_peaks(clean, sampling_rate=int(fs))
        nk_peaks = np.asarray(info["ECG_R_Peaks"], dtype=float)
        if len(peaks_ours) and len(nk_peaks):
            f1 = peak_f1(nk_peaks / fs, np.asarray(peaks_ours, dtype=float) / fs, tolerance_seconds=0.1)
    except Exception:
        f1 = float("nan")
    return {"hr": float(hr), "quality": float(quality), "f1": float(f1)}


# --------------------------------------------------------------------------- #
# per-subject
# --------------------------------------------------------------------------- #
def evaluate_subject(pid: str, cfg: dict[str, Any]) -> dict[str, Any]:
    man = C.load_manifest()
    xdf = man[pid]["xdf_path"]
    fs = float(cfg["sample_rate_hz"])
    sel = cfg["selection"]
    events = C.get_events(xdf, cfg)
    bounds = C.get_condition_boundaries(events, pid)
    st = C.load_streams(xdf, cfg)

    eeg_segments: list[np.ndarray] = []
    filt_segments: list[np.ndarray] = []
    cond_coverage: list[float] = []
    ecg_rows: list[dict[str, float]] = []

    for b in bounds:
        l, r = C.slice_by_xdf(st.t_eeg, b.start_xdf, b.end_xdf)
        seg = st.eeg[l:r]
        if len(seg) < int(fs * 5):
            cond_coverage.append(0.0)
            continue
        filt = C.zerophase_bandpass(seg, cfg["filter"]["analytic_band"][0], cfg["filter"]["analytic_band"][1], fs,
                                    cfg["filter"]["order"])
        cov = _block_coverage(filt, fs, cfg)
        cond_coverage.append(cov)
        eeg_segments.append(seg)
        filt_segments.append(filt)
        ecg_seg = st.ecg[l:r]
        ecg_rows.append(_ecg_metrics_condition(ecg_seg, fs))

    # EEG aggregate metrics
    amp_cov = float(np.mean(cond_coverage)) if cond_coverage else 0.0
    conditions_ok = int(np.sum(np.asarray(cond_coverage) >= sel["min_amplitude_coverage"]))
    cov_std = float(np.std(cond_coverage)) if cond_coverage else float("nan")
    if filt_segments:
        f45, psd45 = _mean_psd(filt_segments, fs)
        slope, slope_r2 = _aperiodic_slope(f45, psd45, cfg["slope_fit_hz"])
        iaf, iaf_prom = _iaf(f45, psd45, tuple(cfg["iaf_search_hz"]), tuple(cfg["slope_fit_hz"]))
        fraw, psdraw = _mean_psd(eeg_segments, fs)  # detrended raw retains 50Hz
        line = _line_ratio(fraw, psdraw)
    else:
        slope = slope_r2 = iaf = iaf_prom = line = float("nan")

    # ECG aggregate
    hrs = np.asarray([r["hr"] for r in ecg_rows], dtype=float)
    quals = np.asarray([r["quality"] for r in ecg_rows], dtype=float)
    f1s = np.asarray([r["f1"] for r in ecg_rows], dtype=float)
    hr_med = float(np.nanmedian(hrs)) if hrs.size else float("nan")
    ecg_sqi = float(np.nanmean(quals)) if quals.size else float("nan")
    ecg_f1 = float(np.nanmean(f1s)) if np.isfinite(f1s).any() else float("nan")
    ecg_cond_ok = int(np.sum((quals >= sel["ecg_min_sqi"]) & (hrs >= sel["ecg_hr_min_bpm"]) & (hrs <= sel["ecg_hr_max_bpm"])))

    eeg_usable = bool(
        amp_cov >= sel["min_amplitude_coverage"]
        and conditions_ok >= sel["min_conditions_ok"]
        and np.isfinite(slope) and sel["slope_min"] <= slope <= sel["slope_max"]
    )
    ecg_usable = bool(
        np.isfinite(hr_med) and sel["ecg_hr_min_bpm"] <= hr_med <= sel["ecg_hr_max_bpm"]
        and ecg_sqi >= sel["ecg_min_sqi"] and ecg_cond_ok >= sel["min_conditions_ok"]
    )

    # composite quality score for EEG ranking (higher = better); blind to outcome
    quality_score = float(
        amp_cov
        + 0.10 * (conditions_ok / 9.0)
        + 0.20 * (1.0 / (1.0 + (line if np.isfinite(line) else 10.0)))
        + 0.20 * (iaf_prom if np.isfinite(iaf_prom) else 0.0)
        - 0.10 * (cov_std if np.isfinite(cov_std) else 0.0)
    )

    return {
        "participant_id": pid,
        "xdf_is_backup": bool(man[pid].get("xdf_is_backup", False)),
        "n_conditions": len(bounds),
        "eeg_amp_coverage": round(amp_cov, 4),
        "eeg_conditions_ok": conditions_ok,
        "eeg_coverage_std": round(cov_std, 4) if np.isfinite(cov_std) else None,
        "eeg_line_ratio": round(line, 4) if np.isfinite(line) else None,
        "eeg_aperiodic_slope": round(slope, 4) if np.isfinite(slope) else None,
        "eeg_slope_fit_r2": round(slope_r2, 4) if np.isfinite(slope_r2) else None,
        "eeg_iaf_hz": round(iaf, 3) if np.isfinite(iaf) else None,
        "eeg_iaf_prominence": round(iaf_prom, 4) if np.isfinite(iaf_prom) else None,
        "eeg_usable": eeg_usable,
        "eeg_quality_score": round(quality_score, 4),
        "ecg_hr_median_bpm": round(hr_med, 1) if np.isfinite(hr_med) else None,
        "ecg_sqi": round(ecg_sqi, 4) if np.isfinite(ecg_sqi) else None,
        "ecg_peak_f1": round(ecg_f1, 4) if np.isfinite(ecg_f1) else None,
        "ecg_conditions_ok": ecg_cond_ok,
        "ecg_usable": ecg_usable,
    }


def run(participants: list[str] | None, dry_run: bool = False) -> dict[str, Any]:
    cfg = C.load_idio_config()
    man = C.load_manifest()
    pids = participants or sorted(man.keys())
    rows = []
    for pid in pids:
        print(f"[selection] {pid} ...", flush=True)
        rows.append(evaluate_subject(pid, cfg))

    eeg_pool = [r for r in rows if r["eeg_usable"]]
    eeg_pool.sort(key=lambda r: r["eeg_quality_score"], reverse=True)
    eeg_subjects = [r["participant_id"] for r in eeg_pool[: cfg["selection"]["top_k_eeg"]]]
    ecg_subjects = [r["participant_id"] for r in rows if r["ecg_usable"]]

    # sanity: the project.yaml eeg_disabled list must not appear as eeg_usable
    disabled = set(cfg["selection"]["known_eeg_disabled"])
    leaked = sorted(set(r["participant_id"] for r in eeg_pool) & disabled)

    payload = {
        "config_seed": cfg["seed"],
        "top_k_eeg": cfg["selection"]["top_k_eeg"],
        "eeg_subjects": eeg_subjects,
        "ecg_subjects": ecg_subjects,
        "eeg_usable_all": [r["participant_id"] for r in eeg_pool],
        "reproduce_disabled_check": {
            "known_disabled": sorted(disabled),
            "leaked_into_usable": leaked,
            "passed": leaked == [],
        },
        "metrics": rows,
    }

    if not dry_run:
        C.ARTIFACTS.mkdir(parents=True, exist_ok=True)
        (C.ARTIFACTS / "subject_selection.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with (C.ARTIFACTS / "subject_selection.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print("\n=== EEG usable (ranked) ===")
    for r in eeg_pool:
        print(f"  {r['participant_id']}  score={r['eeg_quality_score']:.3f}  cov={r['eeg_amp_coverage']:.2f}"
              f"  cond_ok={r['eeg_conditions_ok']}  slope={r['eeg_aperiodic_slope']}  IAF={r['eeg_iaf_hz']}"
              f"  line={r['eeg_line_ratio']}")
    print(f"EEG subjects (top-{cfg['selection']['top_k_eeg']}): {eeg_subjects}")
    print(f"ECG subjects: {ecg_subjects}")
    print(f"reproduce-disabled check passed: {payload['reproduce_disabled_check']['passed']} (leaked={leaked})")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", action="append", help="restrict to one or more participants")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run(args.participant, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

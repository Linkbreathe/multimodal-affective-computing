"""Handcrafted EEG and gap-aware ECG estimates with explicit audit records."""

from __future__ import annotations

from pathlib import Path

import neurokit2 as nk
import numpy as np
import pyxdf
from scipy import signal

from mac.features.physio import EEGMontage, apply_reference, causal_filter
from .common import feature_result, intervals
from .specification import SPEC

MONTAGE = EEGMontage(
    ("M2", "TP9", "TP10", "M1"),
    ("reference", "signal", "signal", "reference"),
    ("right", "left", "right", "left"),
    "linked_mastoid",
)


def continuity(timestamps: np.ndarray, counter: np.ndarray, fs: float) -> np.ndarray:
    dt = np.diff(timestamps)
    return (
        (dt > 0)
        & (dt <= SPEC["time"]["physio_max_gap_samples"] / fs)
        & np.isclose(np.mod(np.diff(counter), 256), 1, atol=1e-6)
    )


def load_physiology(path: Path) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, dict]:
    streams, _ = pyxdf.load_xdf(str(path), select_streams=[{"type": "eeg"}], verbose=False)
    if len(streams) != 1:
        raise ValueError("Expected exactly one physiological stream")
    stream = streams[0]
    fs = float(stream["info"]["nominal_srate"][0])
    samples = np.asarray(stream["time_series"], dtype=float)
    timestamps = np.asarray(stream["time_stamps"], dtype=float)
    channels = stream["info"]["desc"][0]["channels"][0]["channel"]
    labels = [c.get("label", [""])[0].strip() for c in channels]
    units = [c.get("unit", [""])[0].strip() for c in channels]
    if (
        samples.shape[1] != 9
        or not np.isclose(fs, 500)
        or any(units[i] != "uV" for i in [1, 2, 3, 4, 7, 8])
    ):
        raise ValueError(
            f"Unverified physiological channel count/rate/units: {samples.shape}, {fs}, {units}"
        )
    if labels[1:] != [f"ch{i}" for i in range(8)]:
        raise ValueError(f"Unexpected raw channel labels: {labels}")
    edges = continuity(timestamps, samples[:, 0], fs)
    info = dict(
        nominal_srate=fs,
        effective_srate=float(stream["info"].get("effective_srate", fs)),
        samples=len(samples),
        channel_labels=labels,
        channel_units=units,
        physiological_gap_count=int((~edges).sum()),
        reference_mapping_source="configs/project.yaml and current thesis Methods; generic XDF labels do not independently establish electrode placement",
        timestamp_policy="pyxdf clock synchronization and segment dejitter; counter discontinuities retained",
    )
    return samples, timestamps, fs, edges, info


def constant_run(values: np.ndarray, threshold: int) -> bool:
    if len(values) < threshold:
        return False
    for col in values.T:
        equal = np.diff(col) == 0
        for a, b in intervals(equal, np.ones(max(0, len(equal) - 1), dtype=bool)):
            if b - a + 1 >= threshold:
                return True
    return False


def band_powers(filtered: np.ndarray, fs: float) -> dict[str, np.ndarray]:
    n = int(round(2 * fs))
    frequencies, psd = signal.welch(
        filtered,
        fs=fs,
        window=signal.windows.hann(n, sym=False),
        nperseg=n,
        noverlap=n // 2,
        nfft=n,
        detrend="constant",
        scaling="density",
        average="mean",
        axis=0,
    )
    result = {}
    for band, (low, high) in SPEC["eeg"]["bands"].items():
        mask = (frequencies >= low) & (frequencies < high)
        result[band] = np.trapezoid(psd[mask], frequencies[mask], axis=0)
    return result


def eeg_features(
    samples: np.ndarray, times: np.ndarray, fs: float, edges: np.ndarray, start: float, end: float
) -> tuple[list[dict], list[dict]]:
    s = SPEC["eeg"]
    duration = end - start
    totals = {band: np.zeros(2) for band in s["bands"]}
    valid_duration = 0.0
    channel_duration = np.zeros(2)
    audit = []
    for window_start in np.arange(start, end, s["filter_window_s"]):
        window_end = min(end, window_start + s["filter_window_s"])
        left, right = np.searchsorted(times, [window_start, window_end])
        raw = samples[left:right, 1:5]
        ts = times[left:right]
        edge = edges[left : max(left, right - 1)]
        if len(raw) < int(2 * fs):
            continue
        referenced, _, _ = apply_reference(raw, MONTAGE)
        filtered = np.full_like(referenced, np.nan)
        for a, b in intervals(np.isfinite(raw).all(axis=1), edge):
            if b - a >= int(2 * fs):
                filtered[a:b] = causal_filter(referenced[a:b], 1.0, 45.0, fs)
        blocks = []
        for block_start in np.arange(window_start, window_end - 2 + 1e-7, 2.0):
            a, b = np.searchsorted(ts, [block_start, block_start + 2])
            piece = filtered[a:b]
            # A full spectral segment must fit; small endpoint jitter is tolerated by timestamps,
            # but absent samples are never synthesized.
            continuous = len(piece) >= int(2 * fs) and edge[a : max(a, b - 1)].all()
            masks = []
            for channel, name in enumerate(["TP9", "TP10"]):
                finite = bool(len(piece) and np.isfinite(piece[:, channel]).all())
                std = float(np.std(piece[:, channel])) if finite else np.nan
                p99 = float(np.percentile(np.abs(piece[:, channel]), 99)) if finite else np.nan
                raw_indices = [0, 3, channel + 1]
                stuck = constant_run(
                    raw[a:b, raw_indices], int(np.ceil(fs * s["raw_constant_run_max_s"]))
                )
                good = bool(
                    continuous
                    and finite
                    and std >= s["flat_std_min_uv"]
                    and p99 <= s["abs_p99_max_uv"]
                    and not stuck
                )
                masks.append(good)
                channel_duration[channel] += 2 * good
                row = dict(
                    channel=name,
                    block_start_xdf=block_start,
                    block_duration_s=2.0,
                    sample_count=b - a,
                    finite=finite,
                    continuous=continuous,
                    filtered_std_uv=std,
                    filtered_abs_p99_uv=p99,
                    raw_constant_run=stuck,
                    accepted=good,
                )
                if good:
                    power = band_powers(piece[:, channel, None], fs)
                    row.update({band + "_power_uv2": float(v[0]) for band, v in power.items()})
                audit.append(row)
            blocks.append((a, b, all(masks)))
        good = np.array([x[2] for x in blocks], dtype=bool)
        joins = np.array([blocks[i][1] == blocks[i + 1][0] for i in range(max(0, len(blocks) - 1))])
        for lo, hi in intervals(good, joins):
            a, b = blocks[lo][0], blocks[hi - 1][1]
            if not edge[a : max(a, b - 1)].all():
                continue
            powers = band_powers(filtered[a:b], fs)
            seconds = 2.0 * (hi - lo)
            for band in totals:
                totals[band] += powers[band] * seconds
            valid_duration += seconds
    coverage = valid_duration / duration
    quality = dict(
        target_duration_s=duration,
        valid_duration_s=valid_duration,
        valid_coverage=coverage,
        tp9_valid_duration_s=channel_duration[0],
        tp10_valid_duration_s=channel_duration[1],
        channels_used="TP9;TP10",
        channel_aggregation="equal_mean_jointly_valid_spans",
        warning="notch_not_applied_as_in_verified_code;residual_mains_and_beta_muscle_contamination_possible",
    )
    good = coverage >= s["coverage_min"] and valid_duration > 0
    results = []
    for band in ["alpha", "beta", "theta"]:
        value = float(np.mean(totals[band] / valid_duration)) if good else np.nan
        results.append(
            feature_result(
                "eeg_" + band + "_power",
                value,
                reason="" if good else "insufficient_joint_tp9_tp10_coverage",
                **quality,
            )
        )
    return results, audit


def prepare_ecg(
    samples: np.ndarray, times: np.ndarray, fs: float, edges: np.ndarray
) -> tuple[list[dict], list[dict], list[tuple[int, int]]]:
    s = SPEC["ecg"]
    ecg = samples[:, 7] - samples[:, 8]
    rr_rows, peak_rows = [], []
    spans = list(intervals(np.isfinite(ecg), edges))
    for segment, (a, b) in enumerate(spans):
        if b - a < int(5 * fs):
            continue
        cleaned = nk.ecg_clean(
            ecg[a:b], sampling_rate=int(fs), method="neurokit", powerline=s["powerline_hz"]
        )
        _, info = nk.ecg_peaks(
            cleaned,
            sampling_rate=int(fs),
            method="neurokit",
            correct_artifacts=False,
            show=False,
            **{
                k: s[k]
                for k in [
                    "smoothwindow",
                    "avgwindow",
                    "gradthreshweight",
                    "minlenweight",
                    "mindelay",
                ]
            },
        )
        peaks = np.asarray(info["ECG_R_Peaks"], dtype=int) + a
        for i, p in enumerate(peaks):
            peak_rows.append(
                dict(
                    segment_id=segment,
                    peak_index=i,
                    sample_index=int(p),
                    time_xdf=float(times[p]),
                    cleaned_peak_uv=float(cleaned[p - a]),
                )
            )
        rr = np.diff(times[peaks]) * 1000
        plausible = (rr >= s["rr_min_ms"]) & (rr <= s["rr_max_ms"])
        for i, value in enumerate(rr):
            nearby = rr[max(0, i - 2) : min(len(rr), i + 3)]
            local = float(np.median(nearby))
            accepted = bool(
                plausible[i] and abs(value - local) <= s["local_median_fraction"] * local
            )
            rr_rows.append(
                dict(
                    segment_id=segment,
                    rr_index=i,
                    start_xdf=float(times[peaks[i]]),
                    end_xdf=float(times[peaks[i + 1]]),
                    rr_ms=float(value),
                    local_median_ms=local,
                    accepted=accepted,
                    reason=""
                    if accepted
                    else "rr_outside_physiological_bounds"
                    if not plausible[i]
                    else "rr_deviates_from_local_median",
                )
            )
    return rr_rows, peak_rows, spans


def rmssd_from_intervals(rr: list[dict]) -> tuple[float, int]:
    differences = []
    for previous, current in zip(rr[:-1], rr[1:]):
        if (
            previous["accepted"]
            and current["accepted"]
            and previous["segment_id"] == current["segment_id"]
            and current["rr_index"] == previous["rr_index"] + 1
        ):
            differences.append(current["rr_ms"] - previous["rr_ms"])
    return (
        float(np.sqrt(np.mean(np.square(differences)))) if differences else np.nan,
        len(differences),
    )


def ecg_features(
    rr_rows: list[dict],
    times: np.ndarray,
    fs: float,
    spans: list[tuple[int, int]],
    start: float,
    end: float,
) -> tuple[list[dict], list[dict]]:
    s = SPEC["ecg"]
    duration = end - start
    rr = [r for r in rr_rows if r["start_xdf"] >= start and r["end_xdf"] < end]
    accepted = [r for r in rr if r["accepted"]]
    valid_duration = sum(r["rr_ms"] for r in accepted) / 1000
    quality = dict(
        target_duration_s=duration,
        valid_duration_s=valid_duration,
        valid_coverage=valid_duration / duration,
        rr_count=len(rr),
        accepted_rr_count=len(accepted),
        rejected_rr_count=len(rr) - len(accepted),
        accepted_rr_fraction=len(accepted) / len(rr) if rr else np.nan,
        warning="automated_peak_detection_not_manually_validated;no_peak_insertion",
    )
    hr_good = (
        len(accepted) >= s["hr_min_intervals"] and valid_duration / duration >= s["hr_coverage_min"]
    )
    hr = 60 * len(accepted) / valid_duration if hr_good else np.nan
    hr_result = feature_result(
        "ecg_mean_hr", hr, reason="" if hr_good else "insufficient_valid_rr_coverage", **quality
    )
    lo, hi = [start + x for x in s["rmssd_interval_s"]]
    target_rr = [r for r in rr_rows if r["start_xdf"] >= lo and r["end_xdf"] < hi]
    kept = [r for r in target_rr if r["accepted"]]
    continuous = bool(
        hi <= end and any(times[a] <= lo + 1 / fs and times[b - 1] >= hi - 1 / fs for a, b in spans)
    )
    value, n_pairs = rmssd_from_intervals(target_rr)
    fraction = len(kept) / len(target_rr) if target_rr else np.nan
    reason = (
        "fixed_60s_interval_not_continuously_recorded"
        if not continuous
        else "insufficient_accepted_rr_fraction"
        if not np.isfinite(fraction) or fraction < s["rmssd_accepted_rr_fraction_min"]
        else "fewer_than_20_originally_adjacent_valid_rr_pairs"
        if n_pairs < s["rmssd_min_adjacent_pairs"]
        else ""
    )
    rmssd = feature_result(
        "ecg_rmssd",
        value if not reason else np.nan,
        reason=reason,
        target_duration_s=60.0,
        valid_duration_s=sum(r["rr_ms"] for r in kept) / 1000,
        valid_coverage=sum(r["rr_ms"] for r in kept) / 60000,
        interval_start_xdf=lo,
        interval_end_xdf=hi,
        continuous_recording=continuous,
        rr_count=len(target_rr),
        accepted_rr_count=len(kept),
        rejected_rr_count=len(target_rr) - len(kept),
        accepted_rr_fraction=fraction,
        valid_adjacent_rr_pairs=n_pairs,
        warning="60s_ultrashort_rmssd;automated_peak_detection_not_manually_validated;no_peak_insertion",
    )
    audit = [
        dict(r, in_fixed_rmssd_interval=bool(r["start_xdf"] >= lo and r["end_xdf"] < hi))
        for r in rr
    ]
    return [hr_result, rmssd], audit

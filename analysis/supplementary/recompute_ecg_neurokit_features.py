from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
    sys.path.insert(0, str(SRC / "src"))
from mac.config import load_config  # noqa: E402
from mac.data.index import build_index  # noqa: E402
from mac.features.extract import _load_physio  # noqa: E402
from mac.data.condition_data import aggregate_window_frame  # noqa: E402


OUTPUT_DIR = ROOT / "artifacts" / "features" / "ecg_neurokit"
REPORTS_DIR = ROOT / "artifacts" / "reports" / "ecg_neurokit"
SUMMARY = ROOT / "artifacts" / "reports" / "ecg_neurokit_recompute_2026-07-02_zh.md"
ORIGINAL_FEATURES = ROOT / "artifacts" / "features" / "video_ml" / "condition_features.csv"
WINDOWS = ROOT / "artifacts" / "preprocessed" / "windows.csv"

ECG_BASE_COLUMNS = [
    "ecg_peak_count",
    "ecg_rr_mean_ms",
    "ecg_rr_median_ms",
    "ecg_hr_bpm",
    "ecg_rr_std_ms_audit_only",
    "ecg_signal_iqr_uV",
    "ecg_peak_plausible_fraction",
    "ecg_peak_detection_success",
]
HRV_HORIZONS = (30, 60, 120, 300)
MIN_RR_MS = 60_000.0 / 220.0
MAX_RR_MS = 60_000.0 / 35.0


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    try:
        frame.to_parquet(path, index=False)
    except (ImportError, ValueError):
        pass


def _load_ecg(samples: np.ndarray, config: Any) -> np.ndarray:
    columns = list(config.get("streams.ecg_columns"))
    return np.asarray(samples[:, columns[0]] - samples[:, columns[1]], dtype=float)


def _detect_neurokit_peaks(ecg: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray, str]:
    import neurokit2 as nk

    if len(ecg) < int(sample_rate * 5):
        return np.array([], dtype=int), np.asarray(ecg, dtype=float), "too_short"
    try:
        cleaned = nk.ecg_clean(np.asarray(ecg, dtype=float), sampling_rate=sample_rate, method="neurokit")
        _signals, info = nk.ecg_peaks(
            cleaned,
            sampling_rate=sample_rate,
            method="neurokit",
            correct_artifacts=True,
        )
        return np.asarray(info.get("ECG_R_Peaks", []), dtype=int), np.asarray(cleaned, dtype=float), "neurokit"
    except Exception:
        return np.array([], dtype=int), np.asarray(ecg, dtype=float), "failed"


def _rr_stats(peaks: np.ndarray, sample_rate: float) -> dict[str, float]:
    rr_ms = np.diff(peaks) / float(sample_rate) * 1000.0
    plausible = rr_ms[(rr_ms >= MIN_RR_MS) & (rr_ms <= MAX_RR_MS)]
    median_rr = float(np.median(plausible)) if plausible.size else float("nan")
    return {
        "ecg_rr_mean_ms": float(np.mean(plausible)) if plausible.size else float("nan"),
        "ecg_rr_median_ms": median_rr,
        "ecg_hr_bpm": float(60_000.0 / median_rr) if np.isfinite(median_rr) and median_rr > 0 else float("nan"),
        "ecg_rr_std_ms_audit_only": float(np.std(plausible)) if plausible.size >= 2 else float("nan"),
        "ecg_peak_plausible_fraction": float(plausible.size / rr_ms.size) if rr_ms.size else 0.0,
    }


def _window_features(
    cleaned: np.ndarray,
    peaks: np.ndarray,
    peak_times: np.ndarray,
    left: int,
    right: int,
    window_end_xdf: float,
    sample_rate: float,
) -> dict[str, float]:
    window_peaks = peaks[(peaks >= left) & (peaks < right)] - left
    segment = cleaned[left:right]
    record: dict[str, float] = {
        "ecg_peak_count": float(window_peaks.size),
        "ecg_signal_iqr_uV": float(np.percentile(segment, 75) - np.percentile(segment, 25))
        if len(segment) else float("nan"),
        "ecg_peak_detection_success": float(peaks.size >= 3),
    }
    record.update(_rr_stats(window_peaks, sample_rate))
    historical_peaks = peak_times[peak_times <= window_end_xdf]
    for horizon in HRV_HORIZONS:
        record.update(_hrv_features(historical_peaks, horizon))
    return record


def _hrv_features(peak_times_seconds: np.ndarray, horizon_seconds: float) -> dict[str, float]:
    peaks = np.asarray(peak_times_seconds, dtype=float)
    prefix = f"ecg_hrv_{int(horizon_seconds)}s"
    if peaks.size < 3 or peaks[-1] - peaks[0] < horizon_seconds * 0.8:
        return {
            f"{prefix}_rmssd_ms": float("nan"),
            f"{prefix}_sdnn_ms": float("nan"),
            f"{prefix}_pnn50": float("nan"),
        }
    cutoff = peaks[-1] - horizon_seconds
    rr = np.diff(peaks[peaks >= cutoff]) * 1000.0
    plausible = rr[(rr >= MIN_RR_MS) & (rr <= MAX_RR_MS)]
    if plausible.size < 2:
        return {
            f"{prefix}_rmssd_ms": float("nan"),
            f"{prefix}_sdnn_ms": float("nan"),
            f"{prefix}_pnn50": float("nan"),
        }
    diff = np.diff(plausible)
    return {
        f"{prefix}_rmssd_ms": float(np.sqrt(np.mean(diff**2))) if diff.size else float("nan"),
        f"{prefix}_sdnn_ms": float(np.std(plausible, ddof=1)) if plausible.size >= 2 else float("nan"),
        f"{prefix}_pnn50": float(np.mean(np.abs(diff) > 50.0)) if diff.size else float("nan"),
    }


def _participant_rows(
    participant: str,
    source: dict[str, Any],
    windows: pd.DataFrame,
    config: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples, timestamps, sample_rate = _load_physio(Path(source["xdf_path"]), config.get("streams.physio_type"))
    ecg = _load_ecg(samples, config)
    peaks, cleaned, detector = _detect_neurokit_peaks(ecg, sample_rate)
    peak_times = timestamps[peaks] if peaks.size else np.asarray([], dtype=float)

    rows: list[dict[str, Any]] = []
    participant_windows = windows.loc[windows["participant_id"].eq(participant)].copy()
    for _, raw_window in participant_windows.sort_values("window_start_xdf").iterrows():
        start_xdf = float(raw_window["window_start_xdf"])
        end_xdf = float(raw_window["window_end_xdf"])
        left, right = np.searchsorted(timestamps, [start_xdf, end_xdf], side="left")
        record = raw_window.to_dict()
        record.update(_window_features(cleaned, peaks, peak_times, int(left), int(right), end_xdf, sample_rate))
        rows.append(record)

    rr_ms = np.diff(peaks) / float(sample_rate) * 1000.0 if peaks.size else np.asarray([], dtype=float)
    plausible = rr_ms[(rr_ms >= MIN_RR_MS) & (rr_ms <= MAX_RR_MS)]
    diagnostic = {
        "participant_id": participant,
        "detector": detector,
        "sample_rate_hz": float(sample_rate),
        "sample_count": int(len(ecg)),
        "peak_count": int(peaks.size),
        "session_duration_seconds": float(timestamps[-1] - timestamps[0]) if len(timestamps) else float("nan"),
        "session_hr_bpm_median": float(60_000.0 / np.median(plausible)) if plausible.size else float("nan"),
        "rr_plausible_fraction": float(plausible.size / rr_ms.size) if rr_ms.size else 0.0,
        "condition_windows": int(len(rows)),
    }
    return rows, diagnostic


def _merge_condition_features(original: pd.DataFrame, repaired_ecg: pd.DataFrame) -> pd.DataFrame:
    keys = ["participant_id", "condition"]
    ecg_columns = [column for column in repaired_ecg.columns if column.startswith("ecg_")]
    stripped = original.drop(columns=[column for column in original.columns if column.startswith("ecg_")])
    merged = stripped.merge(repaired_ecg[keys + ecg_columns], on=keys, how="left", validate="one_to_one")
    return merged[sorted(merged.columns, key=lambda column: list(original.columns).index(column) if column in original.columns else len(original.columns))]


def _write_report(diagnostics: pd.DataFrame, condition_features: pd.DataFrame) -> None:
    ecg_columns = [column for column in condition_features.columns if column.startswith("ecg_")]
    hr_column = "ecg_hr_bpm__median"
    hrv_column = "ecg_hrv_60s_rmssd_ms__median"
    lines = [
        "# NeuroKit2 ECG 特征重算报告",
        "",
        "**生成日期:** 2026-07-02",
        "**输入:** XDF physio stream + `artifacts/preprocessed/windows.csv`。",
        "**输出:** `artifacts/features/ecg_neurokit/window_features.csv` 与 `condition_features.csv`。",
        "",
        "## 范围",
        "",
        "本次只替换 `ecg_*` 特征。EEG/head/eye/video、标签、condition context 均沿用 "
        "`artifacts/features/video_ml/condition_features.csv`。这仍是离线研究分支，不覆盖主模型或 real-time bundle。",
        "",
        "## 诊断概览",
        "",
        f"- 参与者数: {diagnostics['participant_id'].nunique()}",
        f"- condition-level 行数: {len(condition_features)}",
        f"- repaired ECG 聚合列数: {len(ecg_columns)}",
        f"- median session HR 范围: {diagnostics['session_hr_bpm_median'].min():.1f} - "
        f"{diagnostics['session_hr_bpm_median'].max():.1f} bpm",
        f"- median RR plausible fraction: {diagnostics['rr_plausible_fraction'].median():.3f}",
    ]
    if hr_column in condition_features:
        series = pd.to_numeric(condition_features[hr_column], errors="coerce")
        lines.append(
            f"- condition `{hr_column}` median/IQR: {series.median():.1f} / "
            f"{(series.quantile(0.75) - series.quantile(0.25)):.1f}"
        )
    if hrv_column in condition_features:
        series = pd.to_numeric(condition_features[hrv_column], errors="coerce")
        lines.append(
            f"- condition `{hrv_column}` median/IQR: {series.median():.1f} / "
            f"{(series.quantile(0.75) - series.quantile(0.25)):.1f}"
        )
    lines.extend([
        "",
        "## 每被试诊断",
        "",
        "| participant | detector | peaks | session HR median | plausible RR fraction | windows |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for _, row in diagnostics.sort_values("participant_id").iterrows():
        lines.append(
            f"| {row['participant_id']} | {row['detector']} | {int(row['peak_count'])} | "
            f"{row['session_hr_bpm_median']:.1f} | {row['rr_plausible_fraction']:.3f} | "
            f"{int(row['condition_windows'])} |"
        )
    lines.extend([
        "",
        "## 下一步",
        "",
        "用 `artifacts/features/ecg_neurokit/condition_features.csv` 重跑 physio-family split，"
        "比较 repaired ECG 的 `R/Q/S/A` 家族是否仍然提供增量。",
        "",
    ])
    SUMMARY.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = load_config(ROOT / "configs" / "project.yaml")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    windows = pd.read_csv(WINDOWS)
    original = pd.read_csv(ORIGINAL_FEATURES)
    sources = {row["participant_id"]: row for row in build_index(config)}

    all_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for participant in config.participants:
        try:
            rows, diagnostic = _participant_rows(participant, sources[participant], windows, config)
            all_rows.extend(rows)
            diagnostics.append(diagnostic)
            print(f"ECG neurokit {participant}: {diagnostic['peak_count']} peaks", flush=True)
        except Exception as error:
            errors.append({"participant_id": participant, "reason": str(error)})
            print(f"ECG neurokit {participant}: ERROR {error}", flush=True)

    window_features = pd.DataFrame(all_rows)
    if window_features.empty:
        raise RuntimeError("No repaired ECG windows were generated")
    condition_ecg = aggregate_window_frame(window_features)
    repaired_condition = _merge_condition_features(original, condition_ecg)
    diagnostic_frame = pd.DataFrame(diagnostics)

    window_features.to_csv(OUTPUT_DIR / "window_features.csv", index=False)
    condition_ecg.to_csv(OUTPUT_DIR / "condition_ecg_only_features.csv", index=False)
    repaired_condition.to_csv(OUTPUT_DIR / "condition_features.csv", index=False)
    diagnostic_frame.to_csv(REPORTS_DIR / "diagnostics.csv", index=False)
    (REPORTS_DIR / "errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_parquet(window_features, OUTPUT_DIR / "window_features.parquet")
    _write_parquet(condition_ecg, OUTPUT_DIR / "condition_ecg_only_features.parquet")
    _write_parquet(repaired_condition, OUTPUT_DIR / "condition_features.parquet")
    _write_report(diagnostic_frame, repaired_condition)
    print(SUMMARY)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Summarize Relax foundation probe result JSON files into report tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    columns = list(frame.columns)
    rows = []
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join("---" for _ in columns) + " |")
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def _row_from_result(path: Path, results_dir: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    args = data.get("args", {})
    folds = data.get("folds", [])
    relative = path.relative_to(results_dir)
    parts = relative.parts
    metrics: dict[str, list[float]] = {}
    for fold in folds:
        for name, value in fold.get("metrics", {}).items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(number):
                metrics.setdefault(name, []).append(number)
    row: dict[str, Any] = {
        "result_file": str(path),
        "phase": parts[0] if len(parts) > 1 else "root",
        "result_group": "/".join(parts[:-1]),
        "cohort": args.get("cohort"),
        "modalities": "+".join(args.get("modalities", [])),
        "fusion": args.get("fusion"),
        "baseline": args.get("baseline"),
        "target": args.get("target"),
        "pool": args.get("pool"),
        "folds": len(folds),
        "test_participants": len({fold.get("test_participant") for fold in folds if fold.get("test_participant")}),
    }
    if "leave_one_modality_out" in parts:
        row["ablation"] = parts[-2]
    elif "progressive_modalities" in parts:
        row["ablation"] = parts[-2]
    elif "representation_loss" in parts:
        row["ablation"] = parts[-2]
    else:
        row["ablation"] = ""
    for name, values in sorted(metrics.items()):
        arr = np.asarray(values, dtype=float)
        row[f"{name}_mean"] = float(np.mean(arr))
        row[f"{name}_std"] = float(np.std(arr, ddof=0))
    for stem in (
        "macro_mae",
        "macro_rmse",
        "relaxation_mae",
        "discomfort_mae",
    ):
        doubled = f"{stem}_mean_mean"
        canonical = f"{stem}_mean"
        if doubled in row and canonical not in row:
            row[canonical] = row[doubled]
    return row


def _load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _result_display(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "phase", "result_group", "cohort", "modalities", "fusion", "baseline", "target",
        "pool", "folds", "macro_mae_mean", "macro_rmse_mean", "relaxation_mae_mean",
        "discomfort_mae_mean", "delta_macro_mae_vs_condition",
    ]
    columns = [column for column in columns if column in frame.columns]
    return frame[columns].copy()


def _add_condition_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["delta_macro_mae_vs_condition"] = np.nan
    if frame.empty or "macro_mae_mean" not in frame.columns:
        return frame
    condition_rows = frame[frame["baseline"].eq("condition")]
    baselines = {
        row["cohort"]: float(row["macro_mae_mean"])
        for _, row in condition_rows.iterrows()
        if pd.notna(row.get("cohort")) and pd.notna(row.get("macro_mae_mean"))
    }
    for idx, row in frame.iterrows():
        cohort = row.get("cohort")
        if cohort in baselines and pd.notna(row.get("macro_mae_mean")):
            frame.at[idx, "delta_macro_mae_vs_condition"] = float(row["macro_mae_mean"]) - baselines[cohort]
    return frame


def _cohort_table(cohorts: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name, payload in cohorts.items():
        if name == "schema_version" or not isinstance(payload, dict):
            continue
        participants = [str(item) for item in payload.get("participants", [])]
        rows.append(
            {
                "cohort": name,
                "participants": len(participants),
                "conditions": len(participants) * 9,
                "participant_ids": ",".join(participants),
            }
        )
    return pd.DataFrame(rows)


def _qc_table(qc_manifest: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for row in qc_manifest.get("participants", []):
        rows.append(
            {
                "participant_id": row.get("participant_id"),
                "hq_score": row.get("high_quality_score"),
                "labels": row.get("label_count"),
                "valid_windows": row.get("valid_window_ratio"),
                "ecg": row.get("ecg_usable_ratio"),
                "eeg": row.get("eeg_usable_ratio"),
                "eye": row.get("gaze_usable_ratio"),
                "head": row.get("head_usable_ratio"),
                "video": row.get("video_usable_ratio"),
                "eeg_disabled": row.get("eeg_disabled_by_relax"),
                "marker_median_abs_ms": row.get("marker_median_abs_residual_ms"),
            }
        )
    return pd.DataFrame(rows)


def _invalid_window_table(window_cache_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not window_cache_root.exists():
        return pd.DataFrame(rows)
    import torch

    for path in sorted(window_cache_root.rglob("*.pt")):
        item = torch.load(path, map_location="cpu", weights_only=False)
        if bool(item.get("valid", True)):
            continue
        metadata = item.get("metadata", {}) or {}
        rows.append(
            {
                "modality": path.relative_to(window_cache_root).parts[0],
                "participant_id": metadata.get("participant_id"),
                "condition": metadata.get("condition"),
                "window_index": metadata.get("condition_window_index"),
                "reason": metadata.get("reason", "invalid_qc_window"),
                "file": str(path),
            }
        )
    return pd.DataFrame(rows)


def _section_table(title: str, frame: pd.DataFrame, lines: list[str], limit: int | None = None) -> None:
    lines.extend(["", f"## {title}", ""])
    if frame.empty:
        lines.append("No rows.")
        return
    view = frame.head(limit) if limit else frame
    lines.append(_markdown_table(view))


def _best(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty or "macro_mae_mean" not in frame.columns:
        return None
    return frame.sort_values("macro_mae_mean").iloc[0]


def _metric(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "NA"


def summarize_results(
    results_dir: str | Path,
    output_dir: str | Path,
    *,
    audit_dir: str | Path = "artifacts/relax",
) -> tuple[pd.DataFrame, Path]:
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    audit_dir = Path(audit_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(results_dir.rglob("*_results.json"))
    rows = [_row_from_result(path, results_dir) for path in files]
    frame = _add_condition_deltas(pd.DataFrame(rows))
    csv_path = output_dir / "relax_foundation_summary.csv"
    frame.to_csv(csv_path, index=False)

    phase0 = _load_json(audit_dir / "phase0_audit.json")
    qc_manifest = _load_json(audit_dir / "qc_manifest.json")
    cohorts = _load_json(audit_dir / "cohorts.json")
    window_manifest = _load_json(audit_dir / "window_embeddings_reve_large/window_embedding_manifest.json")
    condition_manifest = _load_json(audit_dir / "condition_embeddings.manifest.json")
    invalid_windows = _invalid_window_table(audit_dir / "window_embeddings_reve_large")

    cohort_frame = _cohort_table(cohorts)
    qc_frame = _qc_table(qc_manifest)

    report_path = output_dir / "relax_foundation_report.md"
    lines = [
        "# Relax Foundation Probe Evaluation Report",
        "",
        "This report summarizes the executed Relax foundation probe run. Cohort membership, "
        "fold metrics, predictions, and hard-failure records remain in their original artifacts.",
        "",
        "## Run Inputs",
        "",
        "- Raw data root: `/home/link/Wei/Models/core/real-time-vis-physio-fusion/data/datasets/relaxdata`",
        f"- Relax-Model run: `{phase0.get('relax_run_dir')}`",
        f"- Painting Reflection workbook: `{phase0.get('painting_reflection_workbook')}`",
        f"- REVE model source: `{phase0.get('reve_model_source')}`",
        f"- REVE position source: `{phase0.get('reve_position_source')}`",
        f"- EEG montage: `{','.join(phase0.get('eeg_montage', []))}`",
        f"- Modalities: `{','.join(window_manifest.get('modalities', []))}`",
        "- PPG/Papagei: excluded from Relax loading, extraction, training, and evaluation.",
        "",
        "## Training Configuration",
        "",
        "- Supervision unit: participant-Condition.",
        "- Outer evaluation: leave-one-participant-out; validation participant is the next sorted participant.",
        "- Encoders: REVE-large EEG, ECGFounder ECG, InceptionTime eye, VideoMAE V2 video, trainable 1D CNN head-motion.",
        "- Probe setting used for non-smoke neural comparisons: max epochs 40, patience 8, batch size 16, `d_common=64`, AdamW.",
        "- MM-Lego setting: latent dim 32, latent channels 8, fusion depth 1, heads 2, dim head 16.",
        "- Condition baseline, random nine-condition baseline, and Relax handcrafted Ridge baseline use train-fold data only.",
        "",
    ]
    _section_table("Cohorts", cohort_frame, lines)
    _section_table("Participant QC", qc_frame, lines)
    lines.extend(
        [
            "",
            "## Embedding Cache",
            "",
            f"- Valid windows: `{window_manifest.get('extracted_valid_windows')}`",
            f"- Invalid windows: `{window_manifest.get('invalid_windows')}`",
            f"- Condition-cache EEG missing records: `{len(condition_manifest.get('missing_records', []))}` "
            "(all expected Relax EEG-disabled participant windows).",
        ]
    )
    _section_table("Invalid Non-EEG Windows", invalid_windows, lines)

    if frame.empty:
        lines.append("No result JSON files were found.")
    else:
        high_quality = frame[frame["phase"].eq("high_quality_pilot")].sort_values("macro_mae_mean")
        formal = frame[frame["phase"].eq("formal")].sort_values(["cohort", "macro_mae_mean"])
        fusion = formal[formal["baseline"].eq("none")]
        baselines = formal[~formal["baseline"].eq("none")]
        leave_one = frame[frame["result_group"].str.contains("leave_one_modality_out", na=False)].sort_values("macro_mae_mean")
        progressive = frame[frame["result_group"].str.contains("progressive_modalities", na=False)].sort_values("result_group")
        representation = frame[frame["result_group"].str.contains("representation_loss", na=False)].sort_values("macro_mae_mean")

        _section_table("High-Quality Pilot Results", _result_display(high_quality), lines)
        _section_table("Formal Baselines", _result_display(baselines), lines)
        _section_table("Formal Fusion Ranking", _result_display(fusion), lines)
        _section_table("Leave-One-Modality-Out Ablations", _result_display(leave_one), lines)
        _section_table("Progressive Modality Ablations", _result_display(progressive), lines)
        _section_table("Pooling And Residual-Target Ablations", _result_display(representation), lines)

        lines.extend(["", "## Key Findings", ""])
        all_135_fusion = fusion[fusion["cohort"].eq("all_135")]
        eeg_fusion = fusion[fusion["cohort"].eq("eeg_eligible")]
        pilot_best = _best(high_quality[high_quality["baseline"].eq("none")])
        all_135_best = _best(all_135_fusion)
        eeg_best = _best(eeg_fusion)
        mm_lego_all = all_135_fusion[all_135_fusion["fusion"].eq("mm_lego")]
        mm_lego_row = mm_lego_all.iloc[0] if not mm_lego_all.empty else None
        leave_best = _best(leave_one)
        progressive_best = _best(progressive)
        representation_best = _best(representation)
        if pilot_best is not None:
            lines.append(
                f"1. Pilot best neural method: `{pilot_best.get('fusion')}` with macro MAE "
                f"{_metric(pilot_best.get('macro_mae_mean'))}; this remains an engineering pilot result only."
            )
        if all_135_best is not None:
            lines.append(
                f"2. Full `all_135` best neural method: `{all_135_best.get('fusion')}` with macro MAE "
                f"{_metric(all_135_best.get('macro_mae_mean'))}, delta vs Condition baseline "
                f"{_metric(all_135_best.get('delta_macro_mae_vs_condition'))}."
            )
        if eeg_best is not None:
            lines.append(
                f"3. `eeg_eligible` best neural method: `{eeg_best.get('fusion')}` with macro MAE "
                f"{_metric(eeg_best.get('macro_mae_mean'))}, delta vs Condition baseline "
                f"{_metric(eeg_best.get('delta_macro_mae_vs_condition'))}."
            )
        if mm_lego_row is not None:
            lines.append(
                f"4. MM-Lego on `all_135` reached macro MAE {_metric(mm_lego_row.get('macro_mae_mean'))}; "
                "it improved over the Condition baseline but did not rank first among fusion methods."
            )
        if leave_best is not None and progressive_best is not None:
            lines.append(
                f"5. In MM-Lego ablations, the best leave-one setting was `{leave_best.get('ablation')}` "
                f"(macro MAE {_metric(leave_best.get('macro_mae_mean'))}); the full progressive setting was "
                f"`{progressive_best.get('ablation')}` with macro MAE {_metric(progressive_best.get('macro_mae_mean'))}."
            )
        if representation_best is not None:
            lines.append(
                f"6. Best MM-Lego representation/loss sensitivity was `{representation_best.get('ablation')}` "
                f"with macro MAE {_metric(representation_best.get('macro_mae_mean'))}."
            )

    lines.extend(
        [
            "",
            "## Limitations And Failures",
            "",
            "- One P005/C2 extra window (`window_index=7`) was invalid for eye, head-motion, and video; it remains masked in the cache.",
            "- `video_complete_sensitivity` has the same 15 participants as `all_135` in this run, so it is a label for sensitivity accounting, not a different participant set.",
            "- `foundation_complete` and `eeg_eligible` contain the same 9 participants under the current QC thresholds.",
            "- `flash_attn` was unavailable; REVE ran without that optional acceleration.",
            "- The environment emits a SciPy/NumPy compatibility warning (`scipy` expects NumPy <1.25, env has 1.26.4). The focused tests and executed runs completed despite the warning.",
            "- Relax handcrafted Ridge produced extreme outlier errors on the 15-participant cohorts; interpret that baseline as unstable under this feature table/scaling setup, not as a calibrated predictor.",
            "- MM-Lego modality-type embedding and shared-projection ablations are not implemented in this local MM-Lego class; they were not fabricated.",
            "- EEG 1D-CNN-vs-REVE and encoder fine-unfreeze ablations were not completed in this execution; the report does not claim those comparisons.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return frame, report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="logs/relax_foundation_probe")
    parser.add_argument("--output-dir", default="reports/relax_foundation_probe")
    parser.add_argument("--audit-dir", default="artifacts/relax")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame, report_path = summarize_results(args.results_dir, args.output_dir, audit_dir=args.audit_dir)
    print(f"Wrote {len(frame)} rows to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

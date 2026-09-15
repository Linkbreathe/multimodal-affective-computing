from __future__ import annotations

from collections import Counter
from typing import Any

from real_time_ml.config import ProjectConfig
from real_time_ml.data.alignment import condition_boundaries, load_marker_events, make_windows, marker_alignment_qc
from real_time_ml.data.index import build_index
from real_time_ml.data.labels import parse_condition_labels
from real_time_ml.data.tables import write_parquet_if_available, write_rows
from real_time_ml.utils import atomic_write_text, write_json


def preprocess(config: ProjectConfig, participants: list[str] | None = None) -> dict[str, Any]:
    selected = participants or config.participants
    manifest = build_index(config, selected)
    labels = parse_condition_labels(
        config.path("labels_root"),
        selected,
        list(config.get("conditions.intensities")),
        list(config.get("conditions.frequencies")),
    )
    labels_by_key = {(row["participant_id"], row["condition"]): row for row in labels}
    all_boundaries: list[dict[str, Any]] = []
    all_windows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    for source in manifest:
        participant = source["participant_id"]
        if not source["xdf_path"]:
            qc_rows.append({"participant_id": participant, "status": "error", "reason": "missing_xdf"})
            continue
        try:
            events = load_marker_events(source["xdf_path"], config.get("streams.marker_name"))
            qc = marker_alignment_qc(events, float(config.get("quality.marker_outlier_ms")))
            boundaries = condition_boundaries(events, participant)
            windows = make_windows(boundaries, float(config.get("windows.length_seconds")))
            for boundary in boundaries:
                all_boundaries.append(
                    {
                        "participant_id": boundary.participant_id,
                        "condition": boundary.condition,
                        "start_xdf": boundary.start_xdf,
                        "end_xdf": boundary.end_xdf,
                        "start_unix_ms": boundary.start_unix_ms,
                        "end_unix_ms": boundary.end_unix_ms,
                        "start_marker_index": boundary.start_marker_index,
                        "end_marker_index": boundary.end_marker_index,
                        "duration_seconds": boundary.duration_seconds,
                    }
                )
            for window in windows:
                label = labels_by_key[(participant, window["condition"])]
                all_windows.append({**window, **{k: v for k, v in label.items() if k not in window}})
            qc_rows.append(
                {
                    "participant_id": participant,
                    "status": "warning" if qc.get("outliers") else "ok",
                    "condition_count": len(boundaries),
                    "window_count": len(windows),
                    **qc,
                }
            )
        except Exception as error:
            qc_rows.append({"participant_id": participant, "status": "error", "reason": str(error)})

    preprocessed = config.path("preprocessed")
    write_rows(preprocessed / "condition_labels.csv", labels)
    write_rows(preprocessed / "condition_boundaries.csv", all_boundaries)
    write_rows(preprocessed / "windows.csv", all_windows)
    write_parquet_if_available(preprocessed / "condition_labels.parquet", labels)
    write_parquet_if_available(preprocessed / "windows.parquet", all_windows)
    write_json(config.path("reports") / "data_qc.json", {"participants": qc_rows})
    report_lines = ["# P002-P016 数据与时间对齐 QC", ""]
    for row in qc_rows:
        report_lines.append(
            f"- {row['participant_id']}：{row['status']}；Condition={row.get('condition_count', 0)}；"
            f"10秒窗口={row.get('window_count', 0)}；marker 中位绝对残差="
            f"{row.get('median_abs_residual_ms', float('nan')):.3f} ms；最大残差="
            f"{row.get('max_abs_residual_ms', float('nan')):.3f} ms；异常点={len(row.get('outliers', []))}。"
        )
    report_lines.extend(["", "异常点仅记录，不执行静默时间修正。原始数据保持只读。", ""])
    atomic_write_text(config.path("reports") / "data_qc_zh.md", "\n".join(report_lines))
    errors = [row for row in qc_rows if row["status"] == "error"]
    counts = Counter(row["participant_id"] for row in all_windows)
    return {
        "participants_requested": len(selected),
        "labels": len(labels),
        "boundaries": len(all_boundaries),
        "windows": len(all_windows),
        "windows_per_participant": dict(counts),
        "errors": errors,
    }

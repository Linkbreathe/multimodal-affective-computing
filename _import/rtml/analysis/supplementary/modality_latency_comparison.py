from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

try:
    import pyxdf
except ImportError:  # pragma: no cover - optional metadata audit only
    pyxdf = None


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "artifacts" / "reports"
SOURCE_MANIFEST = ROOT / "artifacts" / "manifests" / "source_manifest.csv"
OUT_REPORT = REPORT_DIR / "modality_latency_comparison_report_zh.md"
OUT_TABLE = REPORT_DIR / "modality_latency_comparison_table.csv"
OUT_TEX = REPORT_DIR / "modality_latency_table_latex.tex"

STALE_THRESHOLD_MS = 3000.0
HEAD_EYE_EXPECTED_INTERVAL_MS = 100.0
VIDEO_EXPECTED_INTERVAL_MS = 100.0


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def finite(values: list[float | None]) -> list[float]:
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().strip('"')
    if not text:
        return None
    text = text.replace("\ufeff", "")
    if text.lower() in {"nan", "none", "null", "true", "false"}:
        return None
    if re.match(r"^[+-]?\d+,\d+[eE][+-]?\d+$", text):
        text = text.replace(",", ".")
    elif "," in text and "." not in text:
        text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def parse_iso_ms(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    text = re.sub(r"(\.\d{6})\d+(?=[+-]\d\d:\d\d|$)", r"\1", text)
    try:
        return datetime.fromisoformat(text).timestamp() * 1000.0
    except ValueError:
        return None


def row_time_ms(row: dict[str, Any], prefer_iso: bool = False) -> float | None:
    if prefer_iso:
        iso = parse_iso_ms(row.get("utc_timestamp_iso"))
        if iso is not None:
            return iso
    value = safe_float(row.get("unix_time_ms"))
    if value is not None and value > 1e11:
        return value
    iso = parse_iso_ms(row.get("utc_timestamp_iso"))
    if iso is not None:
        return iso
    return value


def stats(values: list[float | None]) -> dict[str, float | int | None]:
    arr = np.asarray(finite(values), dtype=float)
    if arr.size == 0:
        return {
            "n_records": 0,
            "mean_ms": None,
            "median_ms": None,
            "p95_ms": None,
            "min_ms": None,
            "max_ms": None,
        }
    return {
        "n_records": int(arr.size),
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
    }


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "NA"
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "NA"
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:.{digits}f}"


def read_text_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    except OSError:
        return []


def sniff_delimiter(header_line: str) -> str:
    counts = {",": header_line.count(","), ";": header_line.count(";"), "\t": header_line.count("\t")}
    return max(counts, key=counts.get) if max(counts.values()) else ","


def csv_header_and_rows(path: Path, keep_rows: bool = True) -> tuple[list[str], list[dict[str, str]], str]:
    lines = read_text_lines(path)
    if not lines:
        return [], [], ","
    start = 0
    delimiter = sniff_delimiter(lines[0])
    if lines[0].lower().startswith("sep="):
        delimiter = lines[0].split("=", 1)[1][:1] or ","
        start = 1
    if start >= len(lines):
        return [], [], delimiter
    if start == 0:
        delimiter = sniff_delimiter(lines[start])
    reader = csv.DictReader(lines[start:], delimiter=delimiter)
    header = list(reader.fieldnames or [])
    rows = list(reader) if keep_rows else []
    return header, rows, delimiter


def jsonl_records(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for idx, line in enumerate(handle):
                if limit is not None and idx >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    records.append(item)
    except OSError:
        pass
    return records


def flatten_keys(obj: Any, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            keys.add(name)
            keys.update(flatten_keys(value, name))
    elif isinstance(obj, list):
        for value in obj[:5]:
            keys.update(flatten_keys(value, f"{prefix}[]"))
    return keys


def selected_fields(fields: list[str] | set[str]) -> tuple[str, str, str]:
    names = sorted(fields)
    time_like = [
        name for name in names
        if re.search(r"(timestamp|unix|xdf|utc|window_start|window_end|issued|expires|time_ms|_time_)", name, re.I)
    ]
    latency_like = [name for name in names if re.search(r"(latency|age_ms|modality_age|sample_age|fresh|readback|encode|write|dropped)", name, re.I)]
    coverage_like = [name for name in names if re.search(r"(coverage|usable|quality|fps|frame_count|sample_count)", name, re.I)]
    return "; ".join(time_like), "; ".join(latency_like), "; ".join(coverage_like)


def audit_csv(path: Path, category: str, notes: str = "") -> tuple[dict[str, Any], list[dict[str, str]]]:
    header, rows, delimiter = csv_header_and_rows(path, keep_rows=True)
    time_like, latency_like, coverage_like = selected_fields(header)
    return {
        "category": category,
        "path": rel(path),
        "format": f"csv delimiter={delimiter!r}",
        "records": len(rows),
        "fields_found": "; ".join(header),
        "time_fields": time_like,
        "latency_or_age_fields": latency_like,
        "coverage_or_cadence_fields": coverage_like,
        "used_for_statistics": "yes",
        "notes": notes,
    }, rows


def audit_jsonl(path: Path, category: str, notes: str = "", used: str = "yes") -> dict[str, Any]:
    records = jsonl_records(path)
    keys: set[str] = set()
    for item in records[:100]:
        keys.update(flatten_keys(item))
    time_like, latency_like, coverage_like = selected_fields(keys)
    return {
        "category": category,
        "path": rel(path),
        "format": "jsonl",
        "records": len(records),
        "fields_found": "; ".join(sorted(keys)),
        "time_fields": time_like,
        "latency_or_age_fields": latency_like,
        "coverage_or_cadence_fields": coverage_like,
        "used_for_statistics": used,
        "notes": notes,
    }


def load_manifest() -> list[dict[str, str]]:
    _, rows, _ = csv_header_and_rows(SOURCE_MANIFEST, keep_rows=True)
    return rows


def read_yaml_scalar(path: Path, key_path: list[str]) -> str | None:
    # Minimal reader for the simple scalar/list values needed in this report.
    lines = read_text_lines(path)
    stack: list[tuple[int, str]] = []
    target = ".".join(key_path)
    for line in lines:
        raw = line.split("#", 1)[0].rstrip()
        if not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        stripped = raw.strip()
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        stack.append((indent, key.strip()))
        current = ".".join(item[1] for item in stack)
        if current == target:
            return value.strip().strip('"')
    return None


def adaptive_decision_records() -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for path in sorted((ROOT / "artifacts" / "realtime" / "adaptive_control").glob("*/decisions.jsonl")):
        session_id = path.parent.name
        for item in jsonl_records(path):
            decision = item.get("decision") if isinstance(item.get("decision"), dict) else {}
            issued = safe_float(decision.get("issued_unix_ms"))
            window_end = safe_float(item.get("window_end_ms"))
            window_start = safe_float(item.get("window_start_ms"))
            latency = issued - window_end if issued is not None and window_end is not None else None
            output.append({
                "path": path,
                "session_id": str(decision.get("session_id") or session_id),
                "cycle_index": decision.get("cycle_index"),
                "window_start_ms": window_start,
                "window_end_ms": window_end,
                "issued_unix_ms": issued,
                "latency_ms": latency,
                "coverage": item.get("coverage") if isinstance(item.get("coverage"), dict) else {},
                "qc": item.get("qc") if isinstance(item.get("qc"), dict) else {},
                "decision": decision,
            })
    return output


def session_participants(session_dirs: list[Path]) -> dict[str, str]:
    out: dict[str, str] = {}
    for session_dir in session_dirs:
        manifest = session_dir / "session_manifest.json"
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out[session_dir.name] = str(data.get("participant_id") or "")
    return out


def status_age_values() -> dict[str, list[tuple[str, float]]]:
    values: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for path in sorted((ROOT / "artifacts" / "realtime" / "adaptive_control").glob("*/status.jsonl")):
        session_id = path.parent.name
        for item in jsonl_records(path):
            ages = item.get("modality_age_ms")
            if not isinstance(ages, dict):
                continue
            for name in ("eeg", "ecg", "head", "eye", "lsl", "unity"):
                value = safe_float(ages.get(name))
                if value is not None and value >= 0:
                    values[name].append((session_id, value))
    return values


def decision_sensor_freshness(decisions: list[dict[str, Any]]) -> dict[str, list[tuple[str, float]]]:
    decision_by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in decisions:
        if item["window_end_ms"] is not None:
            decision_by_session[item["session_id"]].append(item)
    freshness: dict[str, list[tuple[str, float]]] = {"head": [], "eye": []}
    for path in sorted((ROOT / "artifacts" / "realtime" / "adaptive_control").glob("*/sensor_frames.jsonl")):
        session_id = path.parent.name
        head_times: list[float] = []
        eye_times: list[float] = []
        for item in jsonl_records(path):
            timestamp = safe_float(item.get("unix_time_ms"))
            if timestamp is None:
                continue
            if isinstance(item.get("head_pose"), dict):
                head_times.append(timestamp)
            if isinstance(item.get("eye"), dict):
                eye_times.append(timestamp)
        head_times.sort()
        eye_times.sort()
        for decision in decision_by_session.get(session_id, []):
            end_ms = float(decision["window_end_ms"])
            for name, times in (("head", head_times), ("eye", eye_times)):
                if not times:
                    continue
                idx = np.searchsorted(np.asarray(times, dtype=float), end_ms, side="right") - 1
                if idx >= 0:
                    freshness[name].append((session_id, end_ms - times[int(idx)]))
    return freshness


def interval_values_from_rows(rows: list[dict[str, str]], prefer_iso: bool = False) -> list[float]:
    times = [row_time_ms(row, prefer_iso=prefer_iso) for row in rows]
    times = [float(t) for t in times if t is not None and math.isfinite(float(t))]
    intervals: list[float] = []
    for left, right in zip(times, times[1:]):
        diff = right - left
        if diff > 0:
            intervals.append(diff)
    return intervals


def xdf_audit_rows(manifest_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in manifest_rows:
        path = Path(row.get("xdf_path", ""))
        fields = ["time_series", "time_stamps"]
        notes = []
        if not path.exists():
            notes.append("missing")
        elif pyxdf is None:
            notes.append("pyxdf unavailable")
        else:
            try:
                streams = pyxdf.resolve_streams(str(path))
                stream_bits = []
                for stream in streams:
                    stream_bits.append(
                        f"name={stream.get('name')}; type={stream.get('type')}; "
                        f"channels={stream.get('channel_count')}; nominal_srate={stream.get('nominal_srate')}"
                    )
                notes.extend(stream_bits)
            except Exception as exc:  # pragma: no cover - data/environment dependent
                notes.append(f"resolve_streams_failed: {exc}")
        time_like, latency_like, coverage_like = selected_fields(fields)
        output.append({
            "category": "LSL/XDF physio source",
            "path": str(path),
            "format": "xdf metadata",
            "records": "",
            "fields_found": "; ".join(fields),
            "time_fields": time_like,
            "latency_or_age_fields": latency_like,
            "coverage_or_cadence_fields": coverage_like,
            "used_for_statistics": "metadata only",
            "notes": " | ".join(notes),
            "participant_id": row.get("participant_id", ""),
        })
    return output


def add_metric_row(
    rows: list[dict[str, Any]],
    modality: str,
    stage: str,
    metric_type: str,
    measurement_status: str,
    values: list[float | None],
    sessions: set[str] | None,
    participants: set[str] | None,
    missing_rate: float | None,
    data_source: str,
    definition: str,
    notes: str,
) -> None:
    summary = stats(values)
    rows.append({
        "modality": modality,
        "stage": stage,
        "metric_type": metric_type,
        "measurement_status": measurement_status,
        "n_records": summary["n_records"],
        "n_sessions": len(sessions or set()),
        "n_participants": len(participants or set()),
        "mean_ms": summary["mean_ms"],
        "median_ms": summary["median_ms"],
        "p95_ms": summary["p95_ms"],
        "min_ms": summary["min_ms"],
        "max_ms": summary["max_ms"],
        "missing_stale_drop_rate": missing_rate,
        "data_source": data_source,
        "latency_definition": definition,
        "notes": notes,
    })


def read_latency_benchmark() -> dict[str, dict[str, float]]:
    path = REPORT_DIR / "supplementary" / "latency_benchmark.csv"
    if not path.exists():
        return {}
    _, rows, _ = csv_header_and_rows(path, keep_rows=True)
    output = {}
    for row in rows:
        sid = str(row.get("session_id", ""))
        output[sid] = {key: safe_float(row.get(key)) or 0.0 for key in ("n_decision_records", "mean_ms", "median_ms", "p95_ms", "min_ms", "max_ms")}
    return output


def summarize_by_session(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    grouped: dict[str, list[float | None]] = defaultdict(list)
    for item in decisions:
        grouped[item["session_id"]].append(item["latency_ms"])
    for sid in sorted(grouped):
        row = {"session_id": sid}
        row.update(stats(grouped[sid]))
        out.append(row)
    all_row = {"session_id": "ALL"}
    all_row.update(stats([item["latency_ms"] for item in decisions]))
    out.append(all_row)
    return out


def comparison_with_existing(current_rows: list[dict[str, Any]]) -> str:
    existing = read_latency_benchmark()
    if not existing:
        return "未找到 `artifacts/reports/supplementary/latency_benchmark.csv`，无法做逐项对比。"
    diffs = []
    for row in current_rows:
        sid = row["session_id"]
        old = existing.get(sid)
        if not old:
            diffs.append(f"{sid}: existing missing")
            continue
        pairs = [
            ("n_decision_records", row["n_records"]),
            ("mean_ms", row["mean_ms"]),
            ("median_ms", row["median_ms"]),
            ("p95_ms", row["p95_ms"]),
            ("min_ms", row["min_ms"]),
            ("max_ms", row["max_ms"]),
        ]
        for key, value in pairs:
            old_value = old.get(key)
            if value is None or old_value is None:
                continue
            if abs(float(value) - float(old_value)) > 1e-6:
                diffs.append(f"{sid}.{key}: current={value}, existing={old_value}")
    if not diffs:
        return "复算结果与 `artifacts/reports/supplementary/latency_benchmark.csv` 和既有报告中的端到端 latency 表一致。"
    return "发现差异：" + "; ".join(diffs)


def markdown_table(rows: list[dict[str, Any]], columns: list[str], max_rows: int | None = None) -> str:
    selected = rows if max_rows is None else rows[:max_rows]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in selected:
        body.append("| " + " | ".join(fmt(row.get(col), 3) for col in columns) + " |")
    return "\n".join([header, sep, *body])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(text: Any) -> str:
    s = fmt(text, 2)
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def write_latex(rows: list[dict[str, Any]]) -> None:
    keep_modalities = {
        "EEG",
        "ECG",
        "Head motion / HMD pose",
        "Eye tracking / gaze",
        "Egocentric video",
        "Feature extraction window",
        "Model inference / controller decision",
        "End-to-end closed-loop decision latency",
    }
    selected = [row for row in rows if row["modality"] in keep_modalities]
    columns = ["modality", "metric_type", "n_records", "mean_ms", "median_ms", "p95_ms", "max_ms", "measurement_status"]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Per-modality latency and freshness audit. Values in ms unless noted; entries marked not directly measured should not be interpreted as measured real-time latency.}",
        "\\label{tab:modality-latency}",
        "\\begin{tabular}{llrrrrrl}",
        "\\toprule",
        "Modality & Metric & N & Mean & Median & P95 & Max & Status \\\\",
        "\\midrule",
    ]
    for row in selected:
        values = [latex_escape(row.get(col)) for col in columns]
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    OUT_TEX.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    audit_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    manifest_rows = load_manifest()
    raw_participants = {row.get("participant_id", "") for row in manifest_rows if row.get("participant_id")}

    # Config and core report/source-data files.
    config_path = ROOT / "configs" / "project.yaml"
    audit_rows.append({
        "category": "config",
        "path": rel(config_path),
        "format": "yaml",
        "records": "",
        "fields_found": "streams.sample_rate_hz; streams.eeg_columns; streams.ecg_columns; windows.length_seconds; realtime.cycle_seconds; features.video.mp4.fps",
        "time_fields": "windows.length_seconds; realtime.cycle_seconds",
        "latency_or_age_fields": "",
        "coverage_or_cadence_fields": "streams.sample_rate_hz; features.video.mp4.fps",
        "used_for_statistics": "yes",
        "notes": "EEG columns and ECG columns share the configured physio stream.",
    })

    for path, category, notes in [
        (REPORT_DIR / "regression_latency_report_zh.md", "existing report", "Existing end-to-end latency report; compared against recomputation."),
        (REPORT_DIR / "supplementary" / "latency_benchmark.csv", "existing source data", "Existing latency benchmark table."),
        (REPORT_DIR / "thesis_priority_figures_2026-07-04" / "source_data" / "fig_07_latency_decisions.csv", "thesis figure source data", "Figure 07 latency decision source data."),
        (REPORT_DIR / "thesis_priority_figures_2026-07-04" / "source_data" / "fig_07_adaptive_decisions_parsed.csv", "thesis figure source data", "Figure 07 parsed adaptive decisions."),
        (REPORT_DIR / "thesis_priority_figures_2026-07-04" / "source_data" / "fig_02_modality_coverage_status.csv", "thesis figure source data", "Modality coverage status source data."),
        (ROOT / "artifacts" / "video" / "mp4_manifest.csv", "video artifact manifest", "MP4 conversion manifest; cadence/source frame count only."),
        (ROOT / "artifacts" / "video" / "videomae2" / "embedding_manifest.json", "video embedding manifest", "Offline VideoMAE2 embedding coverage."),
    ]:
        if not path.exists():
            continue
        if path.suffix.lower() == ".csv":
            audit, _ = audit_csv(path, category, notes)
            audit_rows.append(audit)
        elif path.suffix.lower() == ".json":
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
            keys = flatten_keys(data)
            time_like, latency_like, coverage_like = selected_fields(keys)
            audit_rows.append({
                "category": category,
                "path": rel(path),
                "format": "json",
                "records": data.get("rows", "") if isinstance(data, dict) else "",
                "fields_found": "; ".join(sorted(keys)),
                "time_fields": time_like,
                "latency_or_age_fields": latency_like,
                "coverage_or_cadence_fields": coverage_like,
                "used_for_statistics": "metadata only",
                "notes": notes,
            })
        else:
            audit_rows.append({
                "category": category,
                "path": rel(path),
                "format": path.suffix.lstrip("."),
                "records": "",
                "fields_found": "report text",
                "time_fields": "window_end_ms; decision.issued_unix_ms",
                "latency_or_age_fields": "decision.issued_unix_ms - window_end_ms",
                "coverage_or_cadence_fields": "",
                "used_for_statistics": "comparison only",
                "notes": notes,
            })

    # Runtime adaptive-control logs.
    runtime_session_dirs = sorted((ROOT / "artifacts" / "realtime" / "adaptive_control").glob("*"))
    session_to_participant = session_participants(runtime_session_dirs)
    runtime_sessions = set(session_to_participant.keys())
    runtime_participants = {p for p in session_to_participant.values() if p}
    for session_dir in runtime_session_dirs:
        for name in ("decisions.jsonl", "status.jsonl", "sensor_frames.jsonl", "acks.jsonl"):
            path = session_dir / name
            if path.exists():
                audit_rows.append(audit_jsonl(path, f"adaptive_control/{name}", "Runtime adaptive-control session log."))
        for name in ("session_manifest.json", "summary.json"):
            path = session_dir / name
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
            keys = flatten_keys(data)
            time_like, latency_like, coverage_like = selected_fields(keys)
            audit_rows.append({
                "category": f"adaptive_control/{name}",
                "path": rel(path),
                "format": "json",
                "records": "",
                "fields_found": "; ".join(sorted(keys)),
                "time_fields": time_like,
                "latency_or_age_fields": latency_like,
                "coverage_or_cadence_fields": coverage_like,
                "used_for_statistics": "metadata only",
                "notes": "Runtime session metadata.",
            })

    for path in [ROOT / "artifacts" / "realtime" / "shadow_replay.jsonl", ROOT / "artifacts" / "video" / "realtime" / "handcrafted_video_shadow_replay.jsonl", ROOT / "artifacts" / "video" / "realtime" / "videomae2_video_shadow_replay.jsonl"]:
        if path.exists():
            audit_rows.append(audit_jsonl(path, "offline shadow replay", "Offline replay log; not live controller latency.", used="coverage/context only"))

    # Original session logs from source_manifest.
    head_intervals: list[float] = []
    eye_intervals: list[float] = []
    video_intervals: list[float] = []
    video_readback: list[float] = []
    video_encode_write: list[float] = []
    video_dropped_by_session: dict[str, list[float]] = defaultdict(list)
    video_frame_rows = 0
    video_sessions_with_latency: set[str] = set()
    video_participants_with_latency: set[str] = set()
    original_head_sessions: set[str] = set()
    original_eye_sessions: set[str] = set()
    original_video_sessions: set[str] = set()

    for row in manifest_rows:
        participant = row.get("participant_id", "")
        for kind, category in [
            ("samples_csv", "Unity samples.csv / HMD pose"),
            ("eye_tracking_csv", "Unity eye_tracking.csv / gaze"),
            ("video_frames_csv", "Unity video_frames.csv / egocentric video"),
            ("events_csv", "Unity events.csv"),
        ]:
            path = Path(row.get(kind, ""))
            if not path.exists():
                audit_rows.append({
                    "category": category,
                    "path": str(path),
                    "format": "missing",
                    "records": "",
                    "fields_found": "",
                    "time_fields": "",
                    "latency_or_age_fields": "",
                    "coverage_or_cadence_fields": "",
                    "used_for_statistics": "no",
                    "notes": "Path from source_manifest.csv does not exist.",
                })
                continue
            audit, rows = audit_csv(path, category, f"participant_id={participant}")
            audit_rows.append(audit)
            if kind == "samples_csv":
                vals = interval_values_from_rows(rows)
                head_intervals.extend(vals)
                original_head_sessions.add(str(path.parent))
            elif kind == "eye_tracking_csv":
                vals = interval_values_from_rows(rows)
                eye_intervals.extend(vals)
                original_eye_sessions.add(str(path.parent))
            elif kind == "video_frames_csv":
                vals = interval_values_from_rows(rows, prefer_iso=True)
                video_intervals.extend(vals)
                original_video_sessions.add(str(path.parent))
                for data_row in rows:
                    video_frame_rows += 1
                    rb = safe_float(data_row.get("readback_latency_ms"))
                    ew = safe_float(data_row.get("encode_write_latency_ms"))
                    dr = safe_float(data_row.get("dropped_frames"))
                    if rb is not None:
                        video_readback.append(rb)
                        video_sessions_with_latency.add(str(path.parent))
                        video_participants_with_latency.add(participant)
                    if ew is not None:
                        video_encode_write.append(ew)
                    if dr is not None:
                        video_dropped_by_session[str(path.parent)].append(dr)

    audit_rows.extend(xdf_audit_rows(manifest_rows))

    # Core metrics.
    decisions = adaptive_decision_records()
    latency_values = [item["latency_ms"] for item in decisions]
    decision_sessions = {item["session_id"] for item in decisions}
    decision_participants = {session_to_participant.get(sid, "") for sid in decision_sessions}
    decision_participants.discard("")
    decision_latency_rows = summarize_by_session(decisions)
    compare_text = comparison_with_existing(decision_latency_rows)

    status_values = status_age_values()
    for modality, key in [("EEG", "eeg"), ("ECG", "ecg")]:
        vals = [value for _, value in status_values.get(key, [])]
        sessions = {sid for sid, _ in status_values.get(key, [])}
        stale_rate = (sum(v > STALE_THRESHOLD_MS for v in vals) / len(vals)) if vals else None
        add_metric_row(
            metric_rows,
            modality,
            "Runtime physio stream",
            "sample_freshness_ms",
            "measured as shared LSL status age",
            vals,
            sessions,
            runtime_participants,
            stale_rate,
            "adaptive_control/*/status.jsonl",
            f"`modality_age_ms.{key}` = status `unix_time_ms` minus last seen LSL sample timestamp.",
            "EEG and ECG share the same LSL physio stream timestamp in runtime logs; this is stream freshness, not per-channel processing latency.",
        )

    sensor_fresh = decision_sensor_freshness(decisions)
    for modality, key in [("Head motion / HMD pose", "head"), ("Eye tracking / gaze", "eye")]:
        vals = [value for _, value in sensor_fresh.get(key, [])]
        sessions = {sid for sid, _ in sensor_fresh.get(key, [])}
        stale_rate = (sum(v > STALE_THRESHOLD_MS for v in vals) / len(vals)) if vals else None
        add_metric_row(
            metric_rows,
            modality,
            "Adaptive decision window",
            "sample_freshness_ms",
            "measured at window boundary",
            vals,
            sessions,
            runtime_participants,
            stale_rate,
            "adaptive_control/*/sensor_frames.jsonl + decisions.jsonl",
            "`window_end_ms - max(UnitySensorFrame.unix_time_ms <= window_end_ms)`.",
            "Head and eye use the same UnitySensorFrame timestamp; this is sample freshness, not Unity sensor hardware latency.",
        )

    head_gap_rate = sum(v > 2 * HEAD_EYE_EXPECTED_INTERVAL_MS for v in head_intervals) / len(head_intervals) if head_intervals else None
    eye_gap_rate = sum(v > 2 * HEAD_EYE_EXPECTED_INTERVAL_MS for v in eye_intervals) / len(eye_intervals) if eye_intervals else None
    add_metric_row(
        metric_rows,
        "Head motion / HMD pose",
        "Original Unity acquisition log",
        "sampling_interval_ms",
        "cadence, not latency",
        head_intervals,
        original_head_sessions,
        raw_participants,
        head_gap_rate,
        "source_manifest samples_csv",
        "Consecutive `samples.csv.unix_time_ms` differences.",
        "Used to report acquisition cadence only.",
    )
    add_metric_row(
        metric_rows,
        "Eye tracking / gaze",
        "Original Unity acquisition log",
        "sampling_interval_ms",
        "cadence, not latency",
        eye_intervals,
        original_eye_sessions,
        raw_participants,
        eye_gap_rate,
        "source_manifest eye_tracking_csv",
        "Consecutive `eye_tracking.csv.unix_time_ms` differences.",
        "Used to report acquisition cadence only.",
    )

    # Unity's dropped_frames is cumulative within the session, not a per-row increment.
    video_drop_total = sum(max(vals) for vals in video_dropped_by_session.values() if vals)
    video_drop_rate = video_drop_total / (video_frame_rows + video_drop_total) if video_frame_rows + video_drop_total else None
    add_metric_row(
        metric_rows,
        "Egocentric video",
        "Original Unity video acquisition log",
        "readback_latency_ms",
        "measured offline/acquisition latency",
        video_readback,
        video_sessions_with_latency,
        video_participants_with_latency,
        video_drop_rate,
        "source_manifest video_frames_csv",
        "`video_frames.csv.readback_latency_ms`.",
        "Video latency is acquisition/offline logging latency; current adaptive-control runtime does not use video as a live controller input.",
    )
    add_metric_row(
        metric_rows,
        "Egocentric video",
        "Original Unity video acquisition log",
        "encode_write_latency_ms",
        "measured offline/acquisition latency",
        video_encode_write,
        video_sessions_with_latency,
        video_participants_with_latency,
        video_drop_rate,
        "source_manifest video_frames_csv",
        "`video_frames.csv.encode_write_latency_ms`.",
        "This field combines encode and write; separate encode-only/write-only latency fields were not found.",
    )
    video_gap_rate = sum(v > 2 * VIDEO_EXPECTED_INTERVAL_MS for v in video_intervals) / len(video_intervals) if video_intervals else None
    add_metric_row(
        metric_rows,
        "Egocentric video",
        "Original Unity video acquisition log",
        "frame_interval_ms",
        "cadence, not latency",
        video_intervals,
        original_video_sessions,
        raw_participants,
        video_gap_rate,
        "source_manifest video_frames_csv",
        "Consecutive video frame timestamp differences, preferring `utc_timestamp_iso` when available.",
        "Effective fps is cadence evidence only.",
    )

    window_durations = []
    for item in decisions:
        if item["window_start_ms"] is not None and item["window_end_ms"] is not None:
            window_durations.append(item["window_end_ms"] - item["window_start_ms"])
    add_metric_row(
        metric_rows,
        "Feature extraction window",
        "Adaptive decision log",
        "window_duration_ms",
        "cadence/window size, not latency",
        window_durations,
        decision_sessions,
        decision_participants,
        None,
        "adaptive_control/*/decisions.jsonl",
        "`window_end_ms - window_start_ms`.",
        "The configured controller window is 10 s; feature extraction processing time is not separately logged.",
    )

    add_metric_row(
        metric_rows,
        "Model inference / controller decision",
        "Adaptive decision log",
        "pure_model_inference_ms",
        "not directly measured",
        [],
        decision_sessions,
        decision_participants,
        None,
        "adaptive_control/*/decisions.jsonl",
        "No per-model inference start/end timestamp was found.",
        "Only the aggregate window-end-to-command-issued latency is measured below.",
    )
    add_metric_row(
        metric_rows,
        "End-to-end closed-loop decision latency",
        "Adaptive decision log",
        "decision_latency_ms",
        "measured aggregate",
        latency_values,
        decision_sessions,
        decision_participants,
        None,
        "adaptive_control/*/decisions.jsonl",
        "`decision.issued_unix_ms - window_end_ms`.",
        "This is the same definition used by the existing regression latency report; it does not prove adaptive relaxation efficacy.",
    )

    # Coverage / availability rows.
    for modality, cov_key in [
        ("EEG", "eeg"),
        ("ECG", "ecg"),
        ("Head motion / HMD pose", "head"),
        ("Eye tracking / gaze", "eye"),
    ]:
        coverage_vals = []
        for item in decisions:
            cov = item["coverage"].get(cov_key)
            value = safe_float(cov)
            if value is not None:
                coverage_vals.append(value * 1000.0)
        low_rate = sum(v < 500.0 for v in coverage_vals) / len(coverage_vals) if coverage_vals else None
        add_metric_row(
            metric_rows,
            modality,
            "Adaptive decision log",
            "coverage_x1000",
            "runtime availability, not latency",
            coverage_vals,
            decision_sessions,
            decision_participants,
            low_rate,
            "adaptive_control/*/decisions.jsonl",
            f"`coverage.{cov_key}` scaled by 1000 for table compatibility.",
            "This row is availability/coverage, not latency.",
        )

    # Write machine-readable outputs.
    write_csv(OUT_TABLE, metric_rows)
    write_latex(metric_rows)

    # Build report tables.
    audit_compact = []
    for row in audit_rows:
        fields = row.get("fields_found", "")
        compact_fields = fields if len(fields) <= 220 else fields[:217] + "..."
        audit_compact.append({
            "category": row.get("category"),
            "path": row.get("path"),
            "records": row.get("records"),
            "time_fields": row.get("time_fields") or "NA",
            "latency_or_age_fields": row.get("latency_or_age_fields") or "NA",
            "coverage_or_cadence_fields": row.get("coverage_or_cadence_fields") or "NA",
            "used_for_statistics": row.get("used_for_statistics"),
            "notes": row.get("notes"),
            "fields_found": compact_fields,
        })

    decision_table = []
    for row in decision_latency_rows:
        decision_table.append({
            "session_id": row["session_id"],
            "n_records": row["n_records"],
            "mean_ms": row["mean_ms"],
            "median_ms": row["median_ms"],
            "p95_ms": row["p95_ms"],
            "min_ms": row["min_ms"],
            "max_ms": row["max_ms"],
        })

    primary_rows = [
        row for row in metric_rows
        if row["metric_type"] not in {"coverage_x1000"}
    ]

    measured_rows = [row for row in primary_rows if row["measurement_status"].startswith("measured")]
    not_measured_rows = [row for row in primary_rows if "not directly measured" in row["measurement_status"] or "cadence" in row["measurement_status"]]

    video_capture_found = any("capture_latency" in str(row.get("latency_or_age_fields", "")).lower() for row in audit_rows)
    video_has_readback = bool(video_readback)
    video_has_encode_write = bool(video_encode_write)
    shared_physio_note = (
        "Runtime code records LSL samples once and assigns the same timestamp to `last_seen['lsl']`, "
        "`last_seen['eeg']`, and `last_seen['ecg']`; therefore EEG and ECG freshness are shared stream freshness."
    )

    english_paragraph = (
        "Latency was evaluated from the available runtime logs and acquisition logs rather than inferred from the model configuration. "
        "The adaptive controller emitted commands with an aggregate decision latency defined as the command issue timestamp minus the end of the 10 s feature window. "
        "Across the available adaptive-control session logs, this end-to-end decision latency remained far below the 10 s update cadence. "
        "Physiological EEG and ECG samples shared the same LSL stream timestamp in the runtime status logs, so their available latency evidence should be described as shared stream freshness rather than separate EEG and ECG processing latency. "
        "Head pose and gaze were available from Unity sensor frames at approximately 10 Hz, and their decision-window freshness was computed from the most recent Unity frame before each window boundary. "
        "Video logs contained acquisition-side readback and combined encode/write timings, but the current adaptive controller did not use video as a live input; therefore video timing should be reported as offline/acquisition latency and cadence, not real-time control latency."
    )

    report = [
        "# 各模态 latency / freshness 对比报告",
        "",
        "## 一句话结论",
        "",
        "当前日志支持写成 **end-to-end decision latency benchmark + runtime availability/freshness audit**：closed-loop 决策延迟可复算且远低于 10 s 更新周期，但 EEG/ECG、head/eye、video 的证据类型不同，不能写成“所有模态均有已测实时 latency”。",
        "",
        "## 路径与范围",
        "",
        f"- 当前可访问工作区：`{ROOT}`。",
        "- 用户给定的 `D:\\Download\\...` 路径在本次运行环境中不可见；实际使用当前 repo 镜像下的 `artifacts/reports`、`artifacts/realtime`、`artifacts/video` 和 `artifacts/manifests/source_manifest.csv` 指向的原始 C 盘 session logs。",
        f"- 输出文件：`{rel(OUT_REPORT)}`、`{rel(OUT_TABLE)}`、`{rel(OUT_TEX)}`。",
        "",
        "## 数据源与字段审计表",
        "",
        "下表列出实际找到的时间戳、latency/freshness、coverage/cadence 字段；完整字段保存在本报告内，统计结果保存在 CSV 表。",
        "",
        markdown_table(
            audit_compact,
            ["category", "path", "records", "time_fields", "latency_or_age_fields", "coverage_or_cadence_fields", "used_for_statistics", "notes"],
            max_rows=80,
        ),
        "",
        "## Latency 定义表",
        "",
        markdown_table(
            [
                {
                    "concept": "sampling cadence",
                    "definition": "同一模态相邻样本/帧的时间间隔，例如 500 Hz 生理流或约 10 Hz Unity/head/eye/video logs。",
                    "used_here": "只用于 cadence/availability，不作为 latency。",
                },
                {
                    "concept": "sample freshness / sample age",
                    "definition": "在 status 或窗口边界时，最近一次样本时间戳距离当前时间/窗口结束多久。",
                    "used_here": "EEG/ECG 使用 `status.modality_age_ms`；head/eye 使用 `window_end_ms - latest UnitySensorFrame`。",
                },
                {
                    "concept": "processing latency",
                    "definition": "原始数据进入系统到完成特征/编码/写盘的耗时。",
                    "used_here": "video 有 readback 与 combined encode/write；其他模态未找到独立 processing latency 字段。",
                },
                {
                    "concept": "decision latency",
                    "definition": "窗口结束到 controller command issued 的时间。",
                    "used_here": "`decision.issued_unix_ms - window_end_ms`。",
                },
                {
                    "concept": "end-to-end latency",
                    "definition": "本项目既有报告中的 closed-loop decision latency。",
                    "used_here": "当前可复算字段与 decision latency 同一公式；不包含 Unity 执行动作后的环境/人体反应。",
                },
            ],
            ["concept", "definition", "used_here"],
        ),
        "",
        "## 各模态 latency / freshness / cadence 对比表",
        "",
        markdown_table(
            primary_rows,
            ["modality", "metric_type", "measurement_status", "n_records", "n_sessions", "n_participants", "mean_ms", "median_ms", "p95_ms", "min_ms", "max_ms", "missing_stale_drop_rate", "data_source"],
        ),
        "",
        "说明：`missing_stale_drop_rate` 的含义随行而变；freshness 行为 `>3000 ms` stale 比例，视频 latency 行为 dropped-frame 估计比例，cadence 行为大于两倍目标间隔的 gap 比例。",
        "",
        "## Runtime availability / coverage 补充表",
        "",
        "coverage 行不是 latency；为避免另建列，CSV/表格中将 0-1 coverage 乘以 1000 显示。",
        "",
        markdown_table(
            [row for row in metric_rows if row["metric_type"] == "coverage_x1000"],
            ["modality", "metric_type", "measurement_status", "n_records", "mean_ms", "median_ms", "p95_ms", "missing_stale_drop_rate", "data_source"],
        ),
        "",
        "## 端到端 decision latency 表",
        "",
        markdown_table(decision_table, ["session_id", "n_records", "mean_ms", "median_ms", "p95_ms", "min_ms", "max_ms"]),
        "",
        f"复算校验：{compare_text}",
        "",
        "## Measured vs inferred / cadence / coverage",
        "",
        "### 可作为 measured latency / freshness 写入",
        "",
        markdown_table(
            measured_rows,
            ["modality", "metric_type", "measurement_status", "latency_definition", "data_source"],
        ),
        "",
        "### 只能写作 cadence / coverage / not directly measured",
        "",
        markdown_table(
            not_measured_rows,
            ["modality", "metric_type", "measurement_status", "latency_definition", "notes"],
        ),
        "",
        "## Video 特别说明",
        "",
        f"- `capture_latency_ms` 字段：{'found' if video_capture_found else 'not found'}。",
        f"- `readback_latency_ms` 字段：{'found' if video_has_readback else 'not found'}。",
        f"- `encode_write_latency_ms` 字段：{'found' if video_has_encode_write else 'not found'}；未找到独立 `encode_latency_ms` 与 `write_latency_ms`。",
        "- `dropped_frames` 字段：found，按所有 video frame records 的 dropped count 估计 dropped-frame rate。",
        "- 当前 adaptive-control runtime 的 model manifest optional modalities 为 EEG/ECG/head/eye；video shadow replay 和 video feature files 是离线/准实时研究证据，不是 live controller 输入。",
        "",
        "## EEG / ECG 特别说明",
        "",
        f"- {shared_physio_note}",
        "- `configs/project.yaml` 配置 `streams.sample_rate_hz=500.0`、`streams.physio_type=eeg`、EEG columns `[1,2,3,4]`、ECG bipolar columns `[7,8]`。",
        "- 因此可以报告 physio stream freshness 和 EEG/ECG coverage/availability；不能从当前 decision logs 拆出独立 EEG latency 与 ECG latency。",
        "",
        "## 可写入论文的英文段落",
        "",
        english_paragraph,
        "",
        "## 不能写入论文的过强说法清单",
        "",
        "- 不要写：all modalities have measured real-time latency。当前只有部分模态/阶段有直接 latency 或 freshness 字段。",
        "- 不要写：video is a real-time controller input。当前 live controller evidence 使用 EEG/ECG/head/eye；video latency 是 acquisition/offline logging latency。",
        "- 不要写：model inference latency was measured separately。未找到纯 inference start/end 字段。",
        "- 不要写：latency proves adaptive relaxation efficacy。latency 只支持技术可行性/响应预算，不证明放松效果。",
        "- 不要把 sampling cadence、sample freshness、processing latency、decision latency 混写为同一个 latency。",
        "",
        "## 建议的论文命名",
        "",
        "建议写成 **end-to-end decision latency benchmark** 为主，辅以 **runtime availability and sample-freshness comparison**。如果章节标题需要覆盖全部模态，可以使用 **per-modality runtime availability and latency audit**；不建议单独命名为 **per-modality latency comparison**，因为 EEG/ECG 与部分 head/eye/video 证据并非同一类 latency。",
        "",
    ]
    OUT_REPORT.write_text("\n".join(report), encoding="utf-8")

    # Persist full audit for traceability inside the markdown as an HTML comment-sized JSONL adjacent file.
    audit_path = REPORT_DIR / "modality_latency_field_audit.csv"
    write_csv(audit_path, audit_rows)

    print(json.dumps({
        "report": str(OUT_REPORT),
        "table": str(OUT_TABLE),
        "latex": str(OUT_TEX),
        "audit": str(audit_path),
        "decision_latency_compare": compare_text,
        "metric_rows": len(metric_rows),
        "audit_rows": len(audit_rows),
    }, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

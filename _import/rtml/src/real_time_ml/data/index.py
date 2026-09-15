from __future__ import annotations

import csv
from typing import Any

from real_time_ml.config import ProjectConfig
from real_time_ml.data.io import discover_session_dir, select_xdf
from real_time_ml.data.labels import find_painting_workbook
from real_time_ml.utils import file_sha256, write_json


def build_index(config: ProjectConfig, participants: list[str] | None = None) -> list[dict[str, Any]]:
    selected = participants or config.participants
    raw_root = config.path("raw_root")
    rows: list[dict[str, Any]] = []
    for participant in selected:
        participant_dir = raw_root / participant
        session_dir = discover_session_dir(participant_dir) if participant_dir.exists() else None
        xdf, backup = select_xdf(participant_dir) if participant_dir.exists() else (None, False)
        record: dict[str, Any] = {
            "participant_id": participant,
            "participant_dir": str(participant_dir),
            "participant_dir_exists": participant_dir.exists(),
            "session_dir": str(session_dir) if session_dir else None,
            "xdf_path": str(xdf) if xdf else None,
            "xdf_is_backup": backup,
            "xdf_size_bytes": xdf.stat().st_size if xdf else None,
        }
        for filename in ("samples.csv", "eye_tracking.csv", "video_frames.csv", "events.csv"):
            record[filename.replace(".", "_")] = str(session_dir / filename) if session_dir and (session_dir / filename).exists() else None
        rows.append(record)
    labels = find_painting_workbook(config.path("labels_root"))
    payload = {
        "schema_version": config.data["schema_version"],
        "participants": rows,
        "label_workbook": str(labels),
        "label_workbook_sha256": file_sha256(labels),
        "config_sha256": file_sha256(config.source),
    }
    target = config.path("manifests") / "source_manifest.json"
    write_json(target, payload)
    csv_path = config.path("manifests") / "source_manifest.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows

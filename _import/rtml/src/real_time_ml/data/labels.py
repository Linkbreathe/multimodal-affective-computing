from __future__ import annotations

from pathlib import Path
from typing import Any

from real_time_ml.data.io import condition_parameters, normalize_condition, normalize_participant_id
from real_time_ml.data.xlsx import read_first_sheet


PAINTING_WORKBOOK = "4. Painting Reflection (Responses).xlsx"


def find_painting_workbook(labels_root: Path) -> Path:
    matches = list(labels_root.rglob(PAINTING_WORKBOOK))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {PAINTING_WORKBOOK}, found {len(matches)}")
    return matches[0]


def _score(value: Any) -> float:
    number = float(value)
    if not 1 <= number <= 7:
        raise ValueError(f"Expected a 1-7 response, got {value!r}")
    return number


def visual_fit(value: Any) -> float:
    text = str(value or "").lower()
    if "appropriate" in text or "keep it" in text:
        return 1.0
    if "too weak" in text or "too strong" in text:
        return 0.0
    return 0.5


def parse_condition_labels(
    labels_root: Path,
    participants: list[str],
    intensities: list[float],
    frequencies: list[float],
) -> list[dict[str, Any]]:
    rows = read_first_sheet(find_painting_workbook(labels_root))
    wanted = set(participants)
    output: list[dict[str, Any]] = []
    for source_row, row in enumerate(rows[1:], start=2):
        if len(row) < 2 or row[1] in (None, ""):
            continue
        participant = normalize_participant_id(row[1])
        if participant not in wanted:
            continue
        for position in range(9):
            base = 2 + position * 7
            if len(row) <= base + 6:
                raise ValueError(f"Incomplete Condition block at workbook row {source_row}, block {position + 1}")
            condition = normalize_condition(row[base])
            pleasant_raw = _score(row[base + 2])
            arousal_raw = _score(row[base + 3])
            relaxation_raw = _score(row[base + 4])
            monotony_raw = _score(row[base + 5])
            discomfort_raw = _score(row[base + 6])
            record = {
                "participant_id": participant,
                "condition": condition,
                "presentation_position": position + 1,
                "visual_fit": visual_fit(row[base + 1]),
                "pleasantness": (pleasant_raw - 1.0) / 6.0,
                "calm": (7.0 - arousal_raw) / 6.0,
                "relaxation": (relaxation_raw - 1.0) / 6.0,
                "monotony": (monotony_raw - 1.0) / 6.0,
                "discomfort": (discomfort_raw - 1.0) / 6.0,
                "pleasantness_raw": pleasant_raw,
                "arousal_raw": arousal_raw,
                "relaxation_raw": relaxation_raw,
                "monotony_raw": monotony_raw,
                "discomfort_raw": discomfort_raw,
                "label_source_row": source_row,
            }
            record.update(condition_parameters(condition, intensities, frequencies))
            output.append(record)
    keys = [(row["participant_id"], row["condition"]) for row in output]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate participant/Condition label keys detected")
    expected = len(participants) * 9
    if len(output) != expected:
        raise ValueError(f"Expected {expected} labels, found {len(output)}")
    return sorted(output, key=lambda row: (row["participant_id"], row["presentation_position"]))


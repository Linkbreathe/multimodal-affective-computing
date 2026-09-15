from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Iterator


PARTICIPANT_RE = re.compile(r"(?:P|N)?\s*0*(\d{1,3})", re.IGNORECASE)
CONDITION_RE = re.compile(r"C?\s*0*(\d{1,2})", re.IGNORECASE)


def normalize_participant_id(value: Any) -> str:
    match = PARTICIPANT_RE.fullmatch(str(value).strip())
    if not match:
        raise ValueError(f"Unrecognized participant id: {value!r}")
    number = int(match.group(1))
    if number < 1 or number > 999:
        raise ValueError(f"Participant id out of range: {value!r}")
    return f"P{number:03d}"


def normalize_condition(value: Any) -> str:
    match = CONDITION_RE.fullmatch(str(value).strip())
    if not match:
        raise ValueError(f"Unrecognized condition: {value!r}")
    number = int(match.group(1))
    if not 1 <= number <= 9:
        raise ValueError(f"Condition must be C1-C9: {value!r}")
    return f"C{number}"


def condition_parameters(condition: str, intensities: list[float], frequencies: list[float]) -> dict[str, float | int]:
    number = int(normalize_condition(condition)[1:]) - 1
    intensity_index, frequency_index = divmod(number, 3)
    return {
        "condition_index": number + 1,
        "intensity_index": intensity_index,
        "frequency_index": frequency_index,
        "intensity": float(intensities[intensity_index]),
        "frequency": float(frequencies[frequency_index]),
    }


def sniff_csv(path: Path) -> tuple[str, str, int]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        first = handle.readline().rstrip("\r\n")
        second = handle.readline()
    skip = 1 if first.lower().startswith("sep=") else 0
    sample = second if skip else first + "\n" + second
    if skip:
        delimiter = first[4:5] or ","
    else:
        delimiter = ";" if sample.count(";") > sample.count(",") else ","
    decimal = "," if delimiter == ";" and re.search(r"\d,\d", sample) else "."
    return delimiter, decimal, skip


def iter_csv(path: Path) -> Iterator[dict[str, str]]:
    delimiter, _, skip = sniff_csv(path)
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        if skip:
            next(handle, None)
        yield from csv.DictReader(handle, delimiter=delimiter)


def parse_float(value: Any, decimal: str = ".") -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "null", "none"}:
        return None
    if decimal == ",":
        if "," in text:
            text = text.replace(".", "").replace(",", ".")
        elif text.count(".") > 1:
            first, *rest = text.split(".")
            text = first + "." + "".join(rest)
    try:
        return float(text)
    except ValueError:
        return None


def discover_session_dir(participant_dir: Path) -> Path | None:
    candidates = [p.parent for p in participant_dir.rglob("samples.csv")]
    candidates += [p.parent for p in participant_dir.rglob("events.csv") if p.parent not in candidates]
    if not candidates:
        return None
    return max(candidates, key=lambda p: sum(f.stat().st_size for f in p.glob("*.csv")))


def select_xdf(participant_dir: Path) -> tuple[Path | None, bool]:
    files = list(participant_dir.rglob("*.xdf"))
    if not files:
        return None, False
    current = [p for p in files if not any(token in str(p).lower() for token in ("backup", "old", "bak"))]
    pool = current or files
    selected = max(pool, key=lambda p: p.stat().st_size)
    return selected, not bool(current)


def resolve_video_path(session_dir: Path, row: dict[str, str]) -> Path | None:
    relative = (row.get("relative_path") or row.get("frame_relative_path") or "").strip()
    if not relative:
        return None
    candidate = (session_dir / Path(relative.replace("\\", "/"))).resolve()
    try:
        candidate.relative_to(session_dir.resolve())
    except ValueError as error:
        raise ValueError(f"Video path escapes session directory: {relative}") from error
    return candidate

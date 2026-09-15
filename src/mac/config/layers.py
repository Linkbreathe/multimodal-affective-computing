"""Layered YAML loading and protocol validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PATH_KEYS = ("repo_dir", "checkpoint")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings while replacing lists and scalar values."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _as_path(value: str, source: Path) -> str:
    if "://" in value:
        return value
    path = Path(value)
    return str(path if path.is_absolute() else (source.parent / path).resolve())


def _resolve_declared_paths(data: dict[str, Any], source: Path) -> dict[str, Any]:
    """Resolve path-like values while their declaring YAML file is known."""
    resolved = deepcopy(data)
    paths = resolved.get("paths")
    if isinstance(paths, dict):
        for key, value in list(paths.items()):
            if isinstance(value, str) and value:
                paths[key] = _as_path(value, source)
    run = resolved.get("run")
    if isinstance(run, dict) and isinstance(run.get("output_root"), str):
        run["output_root"] = _as_path(run["output_root"], source)
    video = resolved.get("features", {}).get("video", {})
    videomae2 = video.get("videomae2", {}) if isinstance(video, dict) else {}
    if isinstance(videomae2, dict):
        for key in PATH_KEYS:
            if isinstance(videomae2.get(key), str):
                videomae2[key] = _as_path(videomae2[key], source)
    return resolved


def read_yaml_with_extends(path: str | Path, seen: set[Path] | None = None) -> tuple[dict[str, Any], Path]:
    """Read one YAML file and all of its ``extends`` parents in order."""
    source = Path(path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Configuration not found: {source}")
    active = seen or set()
    if source in active:
        chain = " -> ".join(str(item) for item in (*active, source))
        raise ValueError(f"Configuration extends cycle: {chain}")
    with source.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a mapping: {source}")
    extends = data.pop("extends", None)
    inherited: dict[str, Any] = {}
    if extends:
        parent_values = extends if isinstance(extends, list) else [extends]
        for parent in parent_values:
            if not isinstance(parent, str):
                raise ValueError(f"extends values must be paths: {source}")
            parent_data, _ = read_yaml_with_extends(source.parent / parent, active | {source})
            inherited = deep_merge(inherited, parent_data)
    return deep_merge(inherited, _resolve_declared_paths(data, source)), source


def validate_layered_config(data: dict[str, Any]) -> None:
    """Reject protocol changes that would invalidate the established evidence."""
    protocol = data.get("protocol", {})
    modeling = data.get("modeling", {})
    policy = data.get("policy", {})
    safety = data.get("safety", {})
    run = data.get("run", {})

    targets = protocol.get("targets", modeling.get("targets"))
    if list(targets or []) != ["relaxation", "discomfort"]:
        raise ValueError("Layered configurations must retain exactly [relaxation, discomfort] targets")
    outer_cv = protocol.get("outer_cv", modeling.get("outer_cv"))
    if outer_cv != "leave_one_participant_out":
        raise ValueError("Layered configurations must retain leave_one_participant_out")
    values = [
        protocol.get("window_seconds"),
        data.get("windows", {}).get("length_seconds"),
        data.get("realtime", {}).get("cycle_seconds"),
    ]
    if any(value is not None and float(value) != 10.0 for value in values):
        raise ValueError("10-second windows are a locked protocol requirement")
    # Both spellings exist during the migration.  A layer may not use one to
    # silently disable the other.
    if policy.get("shadow") is not True or safety.get("shadow") is not True:
        raise ValueError("Layered configurations must keep Shadow mode enabled")
    run_id = run.get("id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("Layered configurations require an explicit non-empty run.id")
    mode = run.get("mode", "runtime")
    if mode not in {"runtime", "research"}:
        raise ValueError("run.mode must be runtime or research")
    backend = modeling.get("runtime_backend", "classical")
    if mode == "runtime" and (backend != "classical" or data.get("experiment", {}).get("research_only")):
        raise ValueError("A runtime configuration may not auto-deploy a research backend")
    if not data.get("paths", {}).get("artifacts_root"):
        raise ValueError("Layered configurations require paths.artifacts_root")
    if not data.get("paths", {}).get("reports_root"):
        raise ValueError("Layered configurations require paths.reports_root")

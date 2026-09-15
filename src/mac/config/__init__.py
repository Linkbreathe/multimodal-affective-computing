"""Legacy and layered configuration entry points.

``load_config`` deliberately preserves the single-``project.yaml`` behavior.
Use ``load_config_layers`` for new run-scoped experiments.
"""

from __future__ import annotations

import json
from hashlib import sha256
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .layers import deep_merge, read_yaml_with_extends, validate_layered_config
from .output import OutputLayout


DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "project.yaml"
DEFAULT_BASE_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "base.yaml"


@dataclass(frozen=True)
class ProjectConfig:
    source: Path
    data: dict[str, Any]
    layout: OutputLayout | None = None

    @property
    def is_legacy(self) -> bool:
        return self.layout is None

    @property
    def run_id(self) -> str | None:
        return self.layout.run_id if self.layout else None

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def path(self, name: str) -> Path:
        if self.layout:
            if name in {"raw_root", "labels_root", "project_root"}:
                value = self.get(f"paths.{name}")
                if not value:
                    raise ValueError(f"paths.{name} must be provided by local.yaml for this command")
                return Path(str(value))
            return self.layout.path_for(name)
        value = self.data["paths"][name]
        path = Path(value)
        return path if path.is_absolute() else (self.source.parent / path).resolve()

    @property
    def participants(self) -> list[str]:
        return list(self.get("participants.include", []))

    def human_report_path(self) -> Path:
        if self.layout:
            return self.layout.summary_report
        return self.path("reports") / "summary_zh.md"

    def ensure_artifact_dirs(self) -> None:
        if self.layout:
            self.layout.ensure_dirs()
            return
        for key in (
            "artifacts", "cache", "manifests", "preprocessed", "features",
            "models", "reports", "realtime_logs", "video",
        ):
            self.path(key).mkdir(parents=True, exist_ok=True)

    def write_run_manifest(self) -> Path | None:
        """Write immutable run identity metadata once; legacy has no run manifest."""
        if not self.layout:
            return None
        output = self.layout.manifests / "run_manifest.json"
        stable_config = json.loads(json.dumps(self.data, default=str))
        stable_paths = stable_config.get("paths", {})
        stable_paths.pop("raw_root", None)
        stable_paths.pop("labels_root", None)
        stable_config.pop("hardware", None)
        stable_dcnn = stable_config.get("modeling", {}).get("dcnn", {})
        if isinstance(stable_dcnn, dict):
            stable_dcnn.pop("device", None)
        fingerprint = sha256(
            json.dumps(stable_config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        payload = {
            "schema_version": self.get("schema_version"),
            "run_id": self.layout.run_id,
            "mode": self.get("run.mode"),
            "research_only": bool(self.get("experiment.research_only", False)),
            "unit_of_analysis": self.get("protocol.unit_of_analysis"),
            "targets": self.get("protocol.targets", self.get("modeling.targets")),
            "outer_cv": self.get("protocol.outer_cv", self.get("modeling.outer_cv")),
            "shadow_only": bool(self.get("protocol.shadow_only", self.get("policy.shadow"))),
            "source": str(self.source),
            "config_sha256": fingerprint,
        }
        if output.exists():
            existing = json.loads(output.read_text(encoding="utf-8"))
            if existing != payload:
                raise ValueError(f"Run id already belongs to a different configuration: {self.layout.run_id}")
            return output
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return output


def _validate_legacy(data: dict[str, Any]) -> None:
    if float(data["realtime"]["cycle_seconds"]) != 10.0:
        raise ValueError("realtime.cycle_seconds must remain exactly 10.0")
    if float(data["windows"]["length_seconds"]) != 10.0:
        raise ValueError("windows.length_seconds must remain exactly 10.0")
    backend = data.get("modeling", {}).get("runtime_backend", "classical")
    if backend not in {"classical", "dcnn", "video_dcnn"}:
        raise ValueError("modeling.runtime_backend must be 'classical', 'dcnn' or 'video_dcnn'")
    dcnn = data.get("modeling", {}).get("dcnn", {})
    if dcnn and int(dcnn.get("sequence_length", 8)) < 4:
        raise ValueError("modeling.dcnn.sequence_length must support the configured two-pool architecture")


def load_config(path: str | Path | None = None) -> ProjectConfig:
    """Load the frozen single-file configuration used by existing commands."""
    source = Path(path).resolve() if path else DEFAULT_CONFIG.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Configuration not found: {source}")
    with source.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    _validate_legacy(data)
    config = ProjectConfig(source=source, data=data)
    config.ensure_artifact_dirs()
    return config


def load_config_layers(
    base: str | Path | None = None,
    experiment: str | Path | None = None,
    local: str | Path | None = None,
) -> ProjectConfig:
    """Load ``base -> experiment -> local`` with recursive ``extends`` support.

    The experiment normally extends ``../base.yaml``.  Passing ``base`` is
    useful for tests and for a site-specific base; it is merged before the
    experiment even when that experiment has no explicit ``extends``.
    """
    base_source = Path(base).resolve() if base else DEFAULT_BASE_CONFIG.resolve()
    if not base_source.exists():
        raise FileNotFoundError(f"Base configuration not found: {base_source}")
    base_data, _ = read_yaml_with_extends(base_source)
    merged = base_data
    # Keep a stable project-level source for legacy helpers that derive the
    # repository root from ``config.source``.  All individual path values were
    # resolved while reading their declaring YAML file, so this does not alter
    # relative-path semantics.
    source = base_source
    if experiment:
        experiment_data, _experiment_source = read_yaml_with_extends(experiment)
        # If the experiment declares base.yaml itself, its resolved parent is
        # already included.  Merging remains idempotent for the default base.
        merged = deep_merge(merged, experiment_data)
    if local:
        local_data, _local_source = read_yaml_with_extends(local)
        merged = deep_merge(merged, local_data)
    validate_layered_config(merged)
    provisional = ProjectConfig(source=source, data=merged)
    layout = OutputLayout.for_run(provisional)
    config = ProjectConfig(source=source, data=merged, layout=layout)
    config.ensure_artifact_dirs()
    return config


__all__ = [
    "DEFAULT_BASE_CONFIG",
    "DEFAULT_CONFIG",
    "OutputLayout",
    "ProjectConfig",
    "deep_merge",
    "load_config",
    "load_config_layers",
    "validate_layered_config",
]

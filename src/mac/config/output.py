"""Run-scoped artifact layout for the layered configuration entry point."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imported only by type checkers
    from . import ProjectConfig


@dataclass(frozen=True)
class OutputLayout:
    """The only path constructor used by a layered-config run.

    Legacy commands still use the paths declared in ``project.yaml``.  A
    layered run instead receives a private namespace so experimental outputs
    cannot replace those frozen artifacts.
    """

    run_id: str
    root: Path
    reports_root: Path

    @classmethod
    def for_run(cls, config: "ProjectConfig") -> "OutputLayout":
        run_id = str(config.get("run.id") or "").strip()
        if not run_id:
            raise ValueError("run.id is required for a layered configuration")
        output_root = config.get("run.output_root")
        if not output_root:
            artifacts_root = config.get("paths.artifacts_root")
            if not artifacts_root:
                raise ValueError("paths.artifacts_root is required for a layered configuration")
            output_root = Path(str(artifacts_root)) / "runs"
        reports_root = config.get("paths.reports_root")
        if not reports_root:
            raise ValueError("paths.reports_root is required for a layered configuration")
        return cls(run_id=run_id, root=Path(str(output_root)) / run_id, reports_root=Path(str(reports_root)))

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    @property
    def preprocessed(self) -> Path:
        return self.root / "preprocessed"

    @property
    def features(self) -> Path:
        return self.root / "features"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics"

    @property
    def predictions(self) -> Path:
        return self.root / "predictions"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def video(self) -> Path:
        # Video caches remain run-scoped.  They are feature inputs rather than
        # report artifacts, so keep them below the feature namespace.
        return self.features / "video"

    @property
    def summary_report(self) -> Path:
        return self.reports_root / f"{self.run_id}_summary_zh.md"

    @property
    def reports(self) -> Path:
        """Root for the single human-readable report of this run."""
        return self.reports_root

    def ensure_dirs(self) -> None:
        for directory in (
            self.root,
            self.manifests,
            self.preprocessed,
            self.features,
            self.models,
            self.checkpoints,
            self.metrics,
            self.predictions,
            self.logs,
            self.cache,
            self.video,
            self.reports_root,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, name: str) -> Path:
        paths = {
            "artifacts": self.root,
            "cache": self.cache,
            "manifests": self.manifests,
            "preprocessed": self.preprocessed,
            "features": self.features,
            "models": self.models,
            "checkpoints": self.checkpoints,
            # Existing modules call this legacy key for machine-readable
            # metrics.  New human reports use ``summary_report`` instead.
            "reports": self.metrics,
            "metrics": self.metrics,
            "predictions": self.predictions,
            "logs": self.logs,
            "realtime_logs": self.logs / "realtime",
            "video": self.video,
        }
        if name not in paths:
            raise KeyError(f"Unknown run layout path: {name}")
        return paths[name]

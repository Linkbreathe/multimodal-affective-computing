"""Results registry for cross-experiment comparison."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class ResultsRegistry:
    def __init__(self, path: str = "reports/results_registry.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
        else:
            self.data = []

    def add(self, experiment_name: str, fusion_type: str, metrics: dict[str, float], config_hash: str) -> None:
        entry = {
            "experiment_name": experiment_name,
            "fusion_type": fusion_type,
            "timestamp": datetime.now().isoformat(),
            "config_hash": config_hash,
            **metrics,
        }
        self.data.append(entry)
        self.path.write_text(json.dumps(self.data, indent=2))

    def get_all(self) -> list[dict[str, Any]]:
        return self.data

from __future__ import annotations

from pathlib import Path

import pytest

from real_time_ml.config import load_config, load_config_layers
from real_time_ml.cli import main
from real_time_ml.reporting import write_run_summary


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _base(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "base.yaml",
        """
schema_version: "1.0.0"
protocol:
  unit_of_analysis: participant_condition
  targets: [relaxation, discomfort]
  outer_cv: leave_one_participant_out
  window_seconds: 10.0
  shadow_only: true
paths:
  raw_root: null
  labels_root: null
  artifacts_root: artifacts
  reports_root: reports
run:
  id: null
  mode: runtime
  output_root: artifacts/runs
windows: {length_seconds: 10.0}
realtime: {cycle_seconds: 10.0}
modeling:
  targets: [relaxation, discomfort]
  outer_cv: leave_one_participant_out
  runtime_backend: classical
policy: {shadow: true}
safety: {shadow: true}
""".strip()
        + "\n",
    )


def _experiment(tmp_path: Path, body: str = "") -> Path:
    return _write(
        tmp_path / "experiment.yaml",
        (
            """
run:
  id: config-test-v1
  mode: runtime
experiment:
  kind: runtime_classical
  research_only: false
""".strip()
            + "\n"
            + body
        ),
    )


def test_layered_config_resolves_each_declaring_file_and_scopes_output(tmp_path: Path):
    base = _base(tmp_path)
    experiment = _experiment(tmp_path)
    local_dir = tmp_path / "workstation"
    local_dir.mkdir()
    local = _write(
        local_dir / "local.yaml",
        """
paths:
  raw_root: raw
  labels_root: labels
modeling:
  dcnn:
    device: cpu
""".strip()
        + "\n",
    )

    config = load_config_layers(base, experiment, local)

    assert not config.is_legacy
    assert config.path("raw_root") == local_dir / "raw"
    assert config.path("labels_root") == local_dir / "labels"
    assert config.path("artifacts") == tmp_path / "artifacts" / "runs" / "config-test-v1"
    assert config.path("reports") == config.path("metrics")
    assert config.path("predictions").parent == config.path("artifacts")
    assert config.layout is not None
    assert config.layout.summary_report == tmp_path / "reports" / "config-test-v1_summary_zh.md"

    manifest = config.write_run_manifest()
    assert manifest and manifest.exists()
    result = write_run_summary(config)
    assert Path(result["report"]) == config.layout.summary_report
    assert config.layout.summary_report.exists()
    assert not (tmp_path / "artifacts" / "reports").exists()


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("windows: {length_seconds: 5.0}\n", "10-second"),
        ("protocol:\n  targets: [relaxation]\n", "exactly"),
        ("policy: {shadow: false}\n", "Shadow"),
        ("modeling:\n  runtime_backend: dcnn\n", "runtime"),
    ],
)
def test_layered_config_rejects_protocol_or_runtime_contract_changes(tmp_path: Path, body: str, message: str):
    with pytest.raises(ValueError, match=message):
        load_config_layers(_base(tmp_path), _experiment(tmp_path, body))


def test_layered_config_requires_explicit_run_id(tmp_path: Path):
    experiment = _write(tmp_path / "experiment.yaml", "run: {id: null, mode: runtime}\n")
    with pytest.raises(ValueError, match="run.id"):
        load_config_layers(_base(tmp_path), experiment)


def test_legacy_project_yaml_remains_on_legacy_paths():
    config = load_config()
    assert config.is_legacy
    assert config.path("reports").name == "reports"


def test_run_manifest_rejects_reusing_a_run_id_for_different_configuration(tmp_path: Path):
    base = _base(tmp_path)
    experiment = _experiment(tmp_path)
    first = load_config_layers(base, experiment)
    assert first.write_run_manifest()

    _write(
        experiment,
        """
run:
  id: config-test-v1
  mode: runtime
experiment:
  kind: runtime_classical
  research_only: false
modeling:
  candidates: [ridge]
""".strip()
        + "\n",
    )
    reused = load_config_layers(base, experiment)
    with pytest.raises(ValueError, match="different configuration"):
        reused.write_run_manifest()


def test_layered_report_cli_writes_only_the_root_summary(tmp_path: Path):
    base = _base(tmp_path)
    experiment = _experiment(tmp_path).with_name("cli-experiment.yaml")
    _write(
        experiment,
        """
run:
  id: cli-report-v1
  mode: runtime
experiment:
  kind: runtime_classical
  research_only: false
""".strip()
        + "\n",
    )

    assert main(["--base-config", str(base), "--experiment", str(experiment), "report"]) == 0
    assert (tmp_path / "reports" / "cli-report-v1_summary_zh.md").exists()
    assert not list((tmp_path / "artifacts" / "runs" / "cli-report-v1" / "metrics").glob("*.md"))

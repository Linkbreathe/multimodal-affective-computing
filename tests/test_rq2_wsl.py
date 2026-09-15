from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.run_rq2_wsl import (  # noqa: E402
    AlignmentError,
    AuditState,
    _checksum_file,
    _declared_contract_hashes,
    _check_key_bearing_cache,
    preflight,
)


def test_checksum_file_accepts_sha256sum_style(tmp_path: Path) -> None:
    payload = tmp_path / "payload.txt"
    payload.write_text("rq2", encoding="utf-8")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    checksums = tmp_path / "payload.sha256"
    checksums.write_text(f"{digest}  payload.txt\n", encoding="utf-8")

    assert _checksum_file(checksums) == digest


def test_windows_done_hash_walker_finds_nested_contract_hash() -> None:
    digest = "a" * 64
    payload = {"outputs": {"rq2_contract": {"sha256": digest}}}

    assert _declared_contract_hashes(payload) == [digest]


def test_preflight_fails_closed_when_handoff_is_missing(tmp_path: Path) -> None:
    state = AuditState()

    with pytest.raises(AlignmentError, match="Missing required Windows hand-off files"):
        preflight(tmp_path / "missing_shared", tmp_path / "cache", state)

    assert state.checks["required_handoff_files"]["passed"] is False
    assert len(state.checks["required_handoff_files"]["detail"]["missing"]) == 8


def test_cache_audit_rejects_condition_only_cache_without_explicit_window_keys(tmp_path: Path) -> None:
    # This mirrors the current local condition_embeddings.pt situation: a
    # condition-level cache exists, but no participant/condition/window_id
    # representation index exists to make a safe join.
    (tmp_path / "condition_embeddings.pt").write_bytes(b"placeholder")
    state = AuditState()

    with pytest.raises(AlignmentError, match="Cannot load local condition cache"):
        _check_key_bearing_cache(tmp_path, {("P001", "C1", "P001_C1_W00")}, state)

    assert state.checks["wsl_cache_explicit_window_keys"]["passed"] is False


def test_key_bearing_cache_can_be_discovered(tmp_path: Path) -> None:
    index = pd.DataFrame(
        [
            {
                "participant": "P001",
                "condition": "C1",
                "window_id": "P001_C1_W00",
                "modality": "eeg",
            }
        ]
    )
    index.to_csv(tmp_path / "representation_index.csv", index=False)
    state = AuditState()

    observed = _check_key_bearing_cache(
        tmp_path,
        {("P001", "C1", "P001_C1_W00")},
        state,
    )

    assert len(observed) == 1
    assert state.checks["wsl_cache_explicit_window_keys"]["passed"] is True

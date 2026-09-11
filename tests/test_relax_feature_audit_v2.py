from __future__ import annotations

import numpy as np

from scripts.audit_relax_foundation_features_v2 import _weighted_window_spectrum
from scripts.build_relax_compression_fusion_preregistration_v2 import _candidate_outputs


def test_condition_balanced_window_spectrum_is_invariant_to_duplicate_window() -> None:
    base = np.asarray(
        [
            [[-1.0, 0.0], [0.0, 0.0]],
            [[1.0, 1.0], [0.0, 0.0]],
            [[0.0, -1.0], [0.0, 0.0]],
        ]
    )
    base_mask = np.asarray([[True, False], [True, False], [True, False]])
    duplicated = base.copy()
    duplicated[0, 1] = duplicated[0, 0]
    duplicate_mask = base_mask.copy()
    duplicate_mask[0, 1] = True

    base_spectrum, _ = _weighted_window_spectrum(base, base_mask)
    duplicate_spectrum, _ = _weighted_window_spectrum(duplicated, duplicate_mask)

    np.testing.assert_allclose(
        base_spectrum["explained_variance_ratio"],
        duplicate_spectrum["explained_variance_ratio"],
        atol=1e-12,
    )


def test_prereg_outcome_scan_ignores_evidence_and_detects_results(tmp_path) -> None:
    literature = tmp_path / "literature"
    literature.mkdir()
    (literature / "review.md").write_text("evidence", encoding="utf-8")
    assert _candidate_outputs(tmp_path) == []

    runs = tmp_path / "runs"
    runs.mkdir()
    result = runs / "modality_expert_simplex5_s20260705_results.json"
    result.write_text("{}", encoding="utf-8")
    assert _candidate_outputs(tmp_path) == [str(result.relative_to(tmp_path))]

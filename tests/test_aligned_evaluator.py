from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "analysis" / "supplementary" / "evaluate_aligned_comparison.py"
SPEC = importlib.util.spec_from_file_location("evaluate_aligned_comparison", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_exact_sign_flip_is_one_for_symmetric_two_value_delta():
    assert MODULE.exact_sign_flip_pvalue(np.asarray([-1.0, 1.0])) == 1.0


def test_exact_sign_flip_detects_consistent_direction():
    value = MODULE.exact_sign_flip_pvalue(np.ones(15))
    assert value == 2 / (2**15)


def test_holm_adjust_is_monotone_in_sorted_pvalues():
    raw = np.asarray([0.01, 0.04, 0.03])
    adjusted = MODULE.holm_adjust(raw)
    assert np.allclose(adjusted, [0.03, 0.06, 0.06])


def test_new_aligned_fusion_matrix_has_ten_unique_model_names():
    assert len(MODULE.NEW_ALIGNED_MODELS) == 10
    assert len(set(MODULE.NEW_ALIGNED_MODELS)) == 10
    assert set(MODULE.NEW_FUSIONS) == {"early", "mid", "qformer", "healnet", "mm_lego"}
    assert len(MODULE.EXPECTED_MODELS) == 16
    assert len(MODULE.NEURAL_MODELS) == 14

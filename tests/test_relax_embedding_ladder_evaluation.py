from __future__ import annotations

import numpy as np

from scripts.evaluate_relax_embedding_ladder_fusion import (
    _bootstrap_interval,
    _exact_sign_flip,
    _holm,
)


def test_exact_sign_flip_enumerates_all_participant_signs() -> None:
    observed, pvalue = _exact_sign_flip(np.asarray([-1.0, -1.0, -1.0]))
    assert observed == -1.0
    # Two of 2^3 sign patterns have an absolute mean as extreme as one.
    assert pvalue == 0.25


def test_holm_adjustment_is_monotone_in_sorted_order() -> None:
    adjusted = _holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == {"a": 0.03, "c": 0.06, "b": 0.06}


def test_cluster_bootstrap_uses_participant_rows() -> None:
    values = np.asarray([-1.0, 0.0, 1.0])
    draws = np.asarray([[0, 0, 0], [2, 2, 2], [0, 1, 2], [0, 1, 2]])
    low, high = _bootstrap_interval(values, draws)
    assert low < 0.0
    assert high > 0.0


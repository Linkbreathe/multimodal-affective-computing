from __future__ import annotations

import numpy as np
import pandas as pd

from real_time_ml.evaluation.dynamic_texture_five import (
    exact_sign_flip_pvalue,
    holm_adjust,
    paired_holm_families,
)


def test_exact_sign_flip_and_holm_are_deterministic() -> None:
    values = np.ones(9, dtype=float)
    assert exact_sign_flip_pvalue(values) == 2 / (2**9)
    assert np.allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_formal_paired_design_has_three_tests_per_outcome() -> None:
    rows = []
    for model_index, model in enumerate(("a", "b", "c", "d", "e", "f")):
        for participant_index in range(9):
            for outcome_index, outcome in enumerate(("relaxation", "discomfort", "macro")):
                rows.append(
                    {
                        "model_key": model,
                        "participant_id": f"P{participant_index:03d}",
                        "outcome": outcome,
                        "seed_averaged_participant_mae": (
                            model_index * 0.01 + outcome_index * 0.001
                        ),
                    }
                )
    result = paired_holm_families(
        pd.DataFrame(rows),
        [("a", "b", "ab"), ("c", "d", "cd"), ("e", "f", "ef")],
        family_prefix="test",
        delta_definition="right_minus_left",
        bootstrap=1_000,
        seed=7,
    )
    assert len(result) == 9
    assert (result.groupby("holm_family").size() == 3).all()
    assert set(result["outcome"]) == {"relaxation", "discomfort", "macro"}

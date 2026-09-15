from __future__ import annotations

import numpy as np

from real_time_ml.windows_rq2_representations import (
    CONDITIONS,
    MODALITIES,
    PARTICIPANTS,
    SEEDS,
    _build_anchors,
    _build_folds,
    _comparison_contract,
    _write_immutable_npz,
    final_oof_key,
    representation_key,
)


def test_rq2_canonical_keys_are_path_independent() -> None:
    assert representation_key("handcrafted", "EEG", "global", "deterministic", "P003", "C1") == "handcrafted+EEG+global+deterministic+P003+C1"
    assert final_oof_key("frozen_simplex_full5", "01", "20260705", "relaxation", "P003", "C1") == "frozen_simplex_full5+01+20260705+relaxation+P003+C1"


def test_fixed_fold_roles_and_training_only_anchors() -> None:
    folds = _build_folds()
    assert len(folds) == 81
    for fold_index, test_participant in enumerate(PARTICIPANTS, start=1):
        fold = folds[folds["fold_index"] == fold_index]
        assert fold[fold["role"] == "test"]["participant"].tolist() == [test_participant]
        assert fold[fold["role"] == "validation"]["participant"].tolist() == [PARTICIPANTS[fold_index % len(PARTICIPANTS)]]
        assert (fold["role"] == "train").sum() == 7

    labels = []
    for participant_index, participant in enumerate(PARTICIPANTS):
        for condition_index, condition in enumerate(CONDITIONS):
            labels.append({
                "participant_id": participant,
                "condition": condition,
                "relaxation": float(participant_index + condition_index),
                "discomfort": float(100 + participant_index + condition_index),
            })
    import pandas as pd

    anchors = _build_anchors(pd.DataFrame(labels), folds)
    assert len(anchors) == 9 * 9 * 2
    first = anchors[(anchors["fold_index"] == 1) & (anchors["condition"] == "C1") & (anchors["target"] == "relaxation")].iloc[0]
    assert first["train_participants"] == ";".join(PARTICIPANTS[2:])
    assert first["condition_anchor"] == sum(range(2, 9)) / 7


def test_comparison_families_are_two_sided_and_holm_corrected() -> None:
    families = _comparison_contract()
    assert set(families) == {"family_1_representation_comparisons", "family_2_fusion_comparisons"}
    for family in families.values():
        assert family["two_sided"] is True
        assert family["correction"] == "Holm"
        assert family["delta_definition"] == "MAE(left) - MAE(right)"
        assert family["negative_delta_means_left_better"] is True
        assert len(family["comparisons"]) == 4


def test_npz_artifacts_are_byte_deterministic(tmp_path) -> None:
    path = tmp_path / "vectors.npz"
    arrays = {
        "vectors": np.asarray([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32),
        "participants": np.asarray(["P003", "P004"], dtype="U4"),
        "conditions": np.asarray(["C1", "C6"], dtype="U2"),
    }
    _write_immutable_npz(path, **arrays)
    first = path.read_bytes()
    _write_immutable_npz(path, **arrays)
    assert path.read_bytes() == first

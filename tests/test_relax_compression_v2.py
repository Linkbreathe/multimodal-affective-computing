from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from mac.fusion.frozen_compression_v2 import (
    FIVE_MODALITIES,
    JointBlockBalancedWindowPCA,
    LinearGCCASharedPrivate,
    ModalityWindowPCABank,
    TargetwiseModalityPLS1,
    WeightedPCA,
    WeightedStandardizer,
    collect_condition_balanced_windows,
    fit_joint_bank,
    fit_modality_bank,
    fit_shared_private,
    fit_targetwise_pls1,
    hamilton_allocate,
)


def _toy_windows():
    rng = np.random.default_rng(18072026)
    observations, windows = 18, 5
    dimensions = {"eeg": 18, "ecg": 16, "eye": 8, "head": 5, "video": 14}
    latent = rng.normal(size=(observations, windows, 4))
    blocks = {}
    masks = {}
    for modality, dimension in dimensions.items():
        loadings = rng.normal(size=(4, dimension))
        blocks[modality] = latent @ loadings + 0.1 * rng.normal(
            size=(observations, windows, dimension)
        )
        masks[modality] = np.ones((observations, windows), dtype=bool)
        for observation in range(observations):
            masks[modality][observation, 1 + observation % windows :] = False
        masks[modality][15] = False
        blocks[modality][15] = 0.0
    participants = np.asarray([f"P{index // 2:03d}" for index in range(observations)])
    train = np.arange(12)
    validation = np.arange(12, 15)
    test = np.arange(15, 18)
    return blocks, masks, participants, train, validation, test


def test_condition_balanced_collection_gives_every_nonempty_row_total_weight_one():
    blocks, masks, _participants, train, _validation, _test = _toy_windows()
    collection = collect_condition_balanced_windows(blocks["eeg"], masks["eeg"], train)
    totals = {
        int(index): float(collection.weights[collection.observation_indexes == index].sum())
        for index in train
    }
    np.testing.assert_allclose(list(totals.values()), 1.0, atol=1e-12, rtol=0.0)
    assert collection.evidence()["window_count"] != len(train)


def test_weighted_scaler_and_pca_are_deterministic_and_weight_sensitive():
    values = np.asarray([[0.0, 0.0], [1.0, 3.0], [10.0, 1.0]])
    weights = np.asarray([1.0, 1.0, 0.1])
    scaler = WeightedStandardizer().fit(values, weights)
    first = WeightedPCA(2).fit(scaler.transform(values), weights)
    second = WeightedPCA(2).fit(scaler.transform(values), weights)
    np.testing.assert_array_equal(first.components_, second.components_)
    assert scaler.mean_[0] < values.mean(axis=0)[0]


def test_hamilton_allocation_is_exact_capped_and_uses_stable_tie_order():
    allocation = hamilton_allocate(
        {"eeg": 4.0, "ecg": 4.0, "eye": 1.0, "head": 1.0, "video": 2.0},
        total=12,
        caps={"eeg": 2, "ecg": 8, "eye": 4, "head": 4, "video": 8},
    )
    assert sum(allocation.values()) == 12
    assert all(value >= 1 for value in allocation.values())
    assert allocation["eeg"] == 2
    assert allocation == hamilton_allocate(
        {"eeg": 4.0, "ecg": 4.0, "eye": 1.0, "head": 1.0, "video": 2.0},
        total=12,
        caps={"eeg": 2, "ecg": 8, "eye": 4, "head": 4, "video": 8},
    )


def test_modality_bank_fits_once_and_serves_full_and_leave_one_out_prefixes():
    blocks, masks, participants, train, validation, _test = _toy_windows()
    bank = fit_modality_bank(
        blocks,
        masks,
        train,
        participant_ids=participants,
        max_components_per_modality=12,
    )
    full = bank.allocation_for(FIVE_MODALITIES, variant_name="full_repeat")
    without_eeg = tuple(name for name in FIVE_MODALITIES if name != "eeg")
    ablation = bank.allocation_for(without_eeg, variant_name="minus_eeg")
    full_scores = bank.transform_condition_blocks(
        blocks, masks, validation, allocation=full, modalities=FIVE_MODALITIES
    )
    ablation_scores = bank.transform_condition_blocks(
        blocks, masks, validation, allocation=ablation, modalities=without_eeg
    )
    assert sum(full.values()) == 12
    assert sum(ablation.values()) == 12
    assert {name: value.shape[1] for name, value in full_scores.items()} == full
    assert {name: value.shape[1] for name, value in ablation_scores.items()} == ablation
    provenance = bank.provenance()
    assert provenance["fit_indexes"] == train.tolist()
    assert set(provenance["selected_variant_allocations"]) >= {
        "full_five",
        "full_repeat",
        "minus_eeg",
    }


def test_outer_test_windows_cannot_change_joint_or_modality_fitted_state():
    blocks, masks, participants, train, _validation, test = _toy_windows()
    changed = {name: value.copy() for name, value in blocks.items()}
    for modality in FIVE_MODALITIES:
        changed[modality][test] += 1e8
    first_joint = fit_joint_bank(blocks, masks, train, participant_ids=participants)
    second_joint = fit_joint_bank(changed, masks, train, participant_ids=participants)
    assert first_joint.provenance() == second_joint.provenance()
    np.testing.assert_array_equal(
        first_joint.transform_conditions(blocks, masks, train),
        second_joint.transform_conditions(changed, masks, train),
    )
    first_bank = fit_modality_bank(blocks, masks, train, participant_ids=participants)
    second_bank = fit_modality_bank(changed, masks, train, participant_ids=participants)
    assert first_bank.provenance() == second_bank.provenance()


def test_joint_pca12_is_full_five_and_all_missing_condition_maps_to_zero():
    blocks, masks, participants, train, _validation, test = _toy_windows()
    reducer = JointBlockBalancedWindowPCA(FIVE_MODALITIES, 12).fit(
        blocks, masks, train, participant_ids=participants
    )
    transformed = reducer.transform_conditions(blocks, masks, test)
    assert transformed.shape == (len(test), 12)
    np.testing.assert_array_equal(transformed[0], np.zeros(12))
    assert reducer.provenance()["modalities"] == list(FIVE_MODALITIES)
    assert reducer.provenance()["fit_participants"] == sorted(set(participants[train]))


def test_targetwise_pls1_uses_only_supplied_train_target_rows():
    blocks, masks, participants, train, validation, test = _toy_windows()
    bank = ModalityWindowPCABank().fit(
        blocks, masks, train, participant_ids=participants
    )
    allocation = bank.allocation_for(FIVE_MODALITIES)
    scores = bank.transform_condition_blocks(blocks, masks, allocation=allocation)
    availability = {name: masks[name].any(axis=1) for name in FIVE_MODALITIES}
    rng = np.random.default_rng(71)
    targets = rng.normal(size=(18, 2))
    changed = targets.copy()
    changed[test] = 1e9
    first = fit_targetwise_pls1(
        scores,
        targets,
        train,
        availability=availability,
        participant_ids=participants,
    )
    second = fit_targetwise_pls1(
        scores,
        changed,
        train,
        availability=availability,
        participant_ids=participants,
    )
    assert first.provenance() == second.provenance()
    for target in ("relaxation", "discomfort"):
        np.testing.assert_array_equal(
            first.transform_targets(scores, validation, availability)[target],
            second.transform_targets(scores, validation, availability)[target],
        )


def test_regularized_gcca_shared_private_has_two_shared_and_one_private_per_modality():
    blocks, masks, participants, train, validation, test = _toy_windows()
    bank = fit_modality_bank(blocks, masks, train, participant_ids=participants)
    allocation = bank.allocation_for(FIVE_MODALITIES)
    scores = bank.transform_condition_blocks(blocks, masks, allocation=allocation)
    availability = {name: masks[name].any(axis=1) for name in FIVE_MODALITIES}
    reducer = fit_shared_private(
        scores,
        train,
        availability=availability,
        participant_ids=participants,
        shared_components=2,
    )
    transformed = reducer.transform_conditions(
        scores, np.concatenate([validation, test]), availability
    )
    assert transformed.shape == (6, 7)
    np.testing.assert_array_equal(transformed[3], np.zeros(7))
    provenance = reducer.provenance()
    assert provenance["fit_indexes"] == train.tolist()
    assert provenance["output_dimension"] == 7
    assert set(provenance["ledoit_wolf_shrinkage"]) == set(FIVE_MODALITIES)


def test_compact_constructor_types_are_stable():
    blocks, masks, participants, train, _validation, _test = _toy_windows()
    bank = fit_modality_bank(blocks, masks, train, participant_ids=participants)
    assert isinstance(bank, ModalityWindowPCABank)
    assert isinstance(
        fit_joint_bank(blocks, masks, train, participant_ids=participants),
        JointBlockBalancedWindowPCA,
    )
    allocation = bank.allocation_for(FIVE_MODALITIES)
    scores = bank.transform_condition_blocks(blocks, masks, allocation=allocation)
    availability = {name: masks[name].any(axis=1) for name in FIVE_MODALITIES}
    targets = np.arange(36, dtype=float).reshape(18, 2)
    assert isinstance(
        fit_targetwise_pls1(scores, targets, train, availability=availability),
        TargetwiseModalityPLS1,
    )
    assert isinstance(
        fit_shared_private(scores, train, availability=availability),
        LinearGCCASharedPrivate,
    )

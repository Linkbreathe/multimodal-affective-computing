from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "analysis" / "supplementary" / "evaluate_eeg_eligible_ablation.py"
SPEC = importlib.util.spec_from_file_location("evaluate_eeg_eligible_ablation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_frozen_eeg_eligible_contract_is_9_fold_7_1_1_with_common_masks():
    contract_dir = (
        ROOT
        / "artifacts"
        / "cross_project_alignment_2026-07-16"
        / "eeg_eligible_ablation"
        / "contract"
    )
    contract = MODULE._load_contract(contract_dir)
    fold_by_test, train_windows = MODULE._fold_context(contract_dir)

    assert contract["participant_count"] == 9
    assert contract["observation_count"] == 81
    assert contract["window_count"] == 567
    assert contract["common_valid_window_count"] == 545
    assert contract["zero_common_window_observations"] == [
        {"participant_id": "P004", "condition": "C6"}
    ]
    assert len(fold_by_test) == 9
    assert len(train_windows) == 9
    assert all(value > 0 for value in train_windows.values())


def test_ablation_tests_use_six_separate_ten_test_holm_families():
    rows = []
    for architecture_index, architecture in enumerate(MODULE.ARCHITECTURES):
        for configuration_index, configuration in enumerate(MODULE.CONFIGURATIONS):
            model = f"{architecture}_{configuration}"
            for seed in MODULE.SEEDS:
                for participant_index, participant in enumerate(MODULE.PARTICIPANTS):
                    for target_index, target in enumerate(MODULE.TARGETS):
                        rows.append(
                            {
                                "model": model,
                                "seed": seed,
                                "participant_id": participant,
                                "target": target,
                                "mae": (
                                    0.1
                                    + architecture_index * 0.001
                                    + configuration_index * 0.002
                                    + participant_index * 0.0001
                                    + target_index * 0.00001
                                ),
                            }
                        )
    participant = pd.DataFrame(rows)

    tests = MODULE._ablation_tests(participant, bootstrap=1000, bootstrap_seed=20260716)

    assert len(tests) == 60
    assert set(tests["architecture"]) == set(MODULE.ARCHITECTURES)
    assert all(len(group) == 10 for _, group in tests.groupby("holm_family"))
    assert set(tests["n_participants"]) == {9}
    assert set(tests["holm_significant_0_05"]) == {True}

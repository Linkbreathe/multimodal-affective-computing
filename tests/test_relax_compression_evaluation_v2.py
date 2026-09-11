from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts import evaluate_relax_compression_fusion_v2 as evaluation


def _contract(tmp_path: Path) -> tuple[Path, Path]:
    label_rows = []
    for participant_index, participant in enumerate(evaluation.PARTICIPANTS):
        for condition_index in range(9):
            label_rows.append({
                "participant_id": participant,
                "condition": f"C{condition_index + 1}",
                "presentation_position": condition_index + 1,
                "relaxation": 0.25 + 0.025 * condition_index + 0.002 * participant_index,
                "discomfort": 0.0 if condition_index < 7 else 0.1 * (condition_index - 6) + 0.002 * participant_index,
            })
    labels = pd.DataFrame(label_rows)
    labels_path = tmp_path / "condition_labels.csv"
    labels.to_csv(labels_path, index=False)

    split_rows = []
    for fold_index, test in enumerate(evaluation.PARTICIPANTS):
        validation = evaluation.PARTICIPANTS[(fold_index + 1) % 9]
        for participant in evaluation.PARTICIPANTS:
            role = "test" if participant == test else "validation" if participant == validation else "train"
            split_rows.append({
                "fold_index": fold_index,
                "test_participant": test,
                "validation_participant": validation,
                "participant_id": participant,
                "role": role,
            })
    split_path = tmp_path / "split_manifest.csv"
    pd.DataFrame(split_rows).to_csv(split_path, index=False)
    return labels_path, split_path


def _result_folds(folds: list[dict], modalities: tuple[str, ...], primary: bool) -> list[dict]:
    records = []
    weights = {modality: 1.0 / len(modalities) for modality in modalities}
    for fold in folds:
        targets = {}
        for target in evaluation.TARGETS:
            targets[target] = {"alpha": 100.0, "gamma": 0.5}
            if primary:
                targets[target]["expert_weights"] = weights
        records.append({
            **fold,
            "compression": {"modality_components": {modality: 2 for modality in modalities}},
            "targets": targets,
            "eligibility": {"feature_coefficients_nonzero_for_both_targets": primary},
        })
    return records


def _write_matrix(root: Path, canonical: pd.DataFrame, folds: list[dict]) -> Path:
    prereg_dir = root / "preregistration"
    prereg_dir.mkdir(parents=True)
    prereg_path = prereg_dir / "method_preregistration.json"
    prereg_path.write_text(json.dumps({
        "schema_version": "synthetic_test",
        "primary_candidate": evaluation.PRIMARY,
        "protocol": {
            "participants": list(evaluation.PARTICIPANTS),
            "seeds": list(evaluation.SEEDS), "folds": 9, "observations": 81,
        },
        "methods": [
            {"name": name, "compression_dimension": 12, "parameter_count": "synthetic"}
            for name in evaluation.FULL_METHODS
        ],
    }), encoding="utf-8")

    identities = evaluation._expected_run_ids()
    for candidate, variant, seed in identities:
        modalities = evaluation._modalities_for_variant(variant)
        run_dir = root / "runs" / candidate / variant / f"seed_{seed}"
        run_dir.mkdir(parents=True)
        stem = f"{candidate}_{variant}_s{seed}"
        frame = canonical.copy()
        frame["candidate"] = candidate
        frame["variant"] = variant
        frame["seed"] = seed
        strength = 0.8 if candidate == evaluation.PRIMARY and variant == "full" else 0.55
        for target in evaluation.TARGETS:
            condition = frame[f"condition_only_{target}"].to_numpy(float)
            truth = frame[f"{target}_true"].to_numpy(float)
            frame[f"{target}_pred"] = np.clip(condition + strength * (truth - condition), 0.0, 1.0)
            if candidate == evaluation.PRIMARY and variant == "full":
                exception = (frame["participant_id"] == "P004") & (frame["condition"] == "C6")
                frame.loc[exception, f"{target}_pred"] = frame.loc[exception, f"condition_only_{target}"]
        prediction_path = run_dir / f"{stem}_predictions.csv"
        frame.to_csv(prediction_path, index=False)
        payload = {
            "schema_version": "synthetic_test_run",
            "candidate": candidate, "variant": variant, "seed": seed,
            "modalities": list(modalities), "observation_count": 81, "fold_count": 9,
            "prediction_sha256": evaluation.file_sha256(prediction_path),
            "eligibility": {"all_folds_eligible": candidate == evaluation.PRIMARY},
            "folds": _result_folds(folds, modalities, candidate == evaluation.PRIMARY),
        }
        (run_dir / f"{stem}_results.json").write_text(json.dumps(payload), encoding="utf-8")
    return prereg_path


def test_exact_sign_flip_and_holm_contract() -> None:
    pvalue, assignments = evaluation._exact_sign_flip(np.ones(9))
    assert assignments == 512
    assert pvalue == 2 / 512
    adjusted = evaluation._holm([0.01, 0.04, 0.03])
    assert np.allclose(adjusted, [0.03, 0.06, 0.06])


def test_end_to_end_fixed_matrix_evaluation(tmp_path: Path) -> None:
    labels_path, split_path = _contract(tmp_path)
    _, folds, canonical = evaluation._load_contract(labels_path, split_path)
    root = tmp_path / "experiment"
    prereg_path = _write_matrix(root, canonical, folds)
    output = root / "evaluation"
    args = SimpleNamespace(
        root=root, preregistration=prereg_path, output_dir=output,
        labels=labels_path, split_manifest=split_path,
        no_eeg_root=tmp_path / "missing_no_eeg", healnet_root=tmp_path / "missing_healnet",
        july17_root=tmp_path / "missing_july17", skip_historical=True,
    )
    result = evaluation.evaluate(args)
    assert result["formal_run_count"] == 30
    assert result["formal_oof_row_count"] == 2430
    assert result["success_gate"]["descriptive_success"] is True
    assert result["success_gate"]["stronger_support"] is True
    assert (output / "final_report.md").is_file()
    assert len(pd.read_csv(output / "formal_run_audit.csv")) == 30
    paired = pd.read_csv(output / "paired_inference.csv")
    assert set(paired.groupby("family").size()) == {2, 5}
    ablations = pd.read_csv(output / "ablation_paired_inference.csv")
    assert set(ablations.groupby("outcome").size()) == {5}

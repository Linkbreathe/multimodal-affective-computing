"""Validate and evaluate the separate nine-participant modality-ablation matrix."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import pandas as pd

from evaluate_aligned_comparison import (
    TARGETS,
    _markdown_table,
    _normalize_columns,
    _paired_comparisons,
    _participant_metrics,
    _seed_metrics,
)


ARCHITECTURES = ("late", "early", "mid", "qformer", "healnet", "mm_lego")
CONFIGURATIONS = {
    "full": ("eeg", "ecg", "eye", "head", "video"),
    "no_eeg": ("ecg", "eye", "head", "video"),
    "no_ecg": ("eeg", "eye", "head", "video"),
    "no_eye": ("eeg", "ecg", "head", "video"),
    "no_head": ("eeg", "ecg", "eye", "video"),
    "no_video": ("eeg", "ecg", "eye", "head"),
}
SEEDS = (20260705, 20260706, 20260707)
PARTICIPANTS = ("P003", "P004", "P007", "P008", "P009", "P011", "P012", "P013", "P015")
EXPECTED_MODELS = tuple(
    f"{architecture}_{configuration}"
    for architecture in ARCHITECTURES
    for configuration in CONFIGURATIONS
)
MODEL_TO_PARTS = {
    f"{architecture}_{configuration}": (architecture, configuration)
    for architecture in ARCHITECTURES
    for configuration in CONFIGURATIONS
}
KEYS = ["participant_id", "condition"]


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_true(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "1.0"})


def _load_contract(contract_dir: Path) -> dict[str, Any]:
    contract = _json(contract_dir / "contract.json")
    if contract.get("schema_version") != "eeg_eligible_modality_ablation_contract_v1":
        raise ValueError("Wrong nine-participant ablation contract schema")
    if tuple(contract.get("participants", ())) != PARTICIPANTS:
        raise ValueError("Contract does not contain the frozen nine participants")
    if tuple(contract.get("architectures", ())) != ARCHITECTURES:
        raise ValueError("Contract architectures differ from the benchmark")
    if contract.get("modality_configurations") != {
        name: list(values) for name, values in CONFIGURATIONS.items()
    }:
        raise ValueError("Contract modality configurations differ from the benchmark")
    if tuple(contract.get("seeds", ())) != SEEDS:
        raise ValueError("Contract seeds differ from the benchmark")
    if contract.get("mask_policy", {}).get("name") != "fixed_five_way_intersection":
        raise ValueError("Contract does not use the fixed five-way common-window policy")
    for name, record in contract["files"].items():
        path = contract_dir / record["path"]
        if not path.is_file() or file_sha256(path) != record["sha256"]:
            raise ValueError(f"Contract file failed hash validation: {name}")
    return contract


def _fold_context(contract_dir: Path) -> tuple[dict[str, tuple[int, str]], dict[int, int]]:
    splits = pd.read_csv(contract_dir / "split_manifest.csv")
    if len(splits) != 81:
        raise ValueError("Split manifest must contain 81 participant-role rows")
    fold_by_test: dict[str, tuple[int, str]] = {}
    for fold_index, group in splits.groupby("fold_index", sort=True):
        roles = group.groupby("role")["participant_id"].apply(list).to_dict()
        if len(roles.get("train", ())) != 7 or len(roles.get("validation", ())) != 1 or len(roles.get("test", ())) != 1:
            raise ValueError(f"Fold {fold_index} is not 7/1/1")
        fold_by_test[str(roles["test"][0])] = (int(fold_index), str(roles["validation"][0]))
    if set(fold_by_test) != set(PARTICIPANTS):
        raise ValueError("Every frozen participant must be test exactly once")

    masks = pd.read_csv(contract_dir / "common_valid_window_masks.csv")
    masks["common_valid"] = _as_true(masks["common_valid"])
    if len(masks) != 567 or int(masks["common_valid"].sum()) != 545:
        raise ValueError("Common mask must contain 567 rows and 545 valid windows")
    for modality in ("eeg", "ecg", "eye", "head", "video"):
        if not _as_true(masks[f"{modality}_valid"]).equals(masks["common_valid"]):
            raise ValueError(f"{modality} validity differs from the frozen common mask")
    per_participant = masks.groupby("participant_id")["common_valid"].sum().astype(int).to_dict()
    train_windows = {
        int(fold_index): int(
            sum(
                per_participant[str(participant)]
                for participant in group.loc[group["role"] == "train", "participant_id"]
            )
        )
        for fold_index, group in splits.groupby("fold_index", sort=True)
    }
    return fold_by_test, train_windows


def _read_predictions(root: Path) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for path in sorted(root.rglob("*_predictions.csv")):
        frame = _normalize_columns(pd.read_csv(path))
        variants = tuple(sorted(frame["model_variant"].astype(str).unique()))
        if len(variants) != 1 or variants[0] not in MODEL_TO_PARTS:
            raise ValueError(f"Unexpected model variant in {path}: {variants}")
        architecture, configuration = MODEL_TO_PARTS[variants[0]]
        frame["model"] = variants[0]
        frame["architecture"] = architecture
        frame["configuration"] = configuration
        frame["source_file"] = str(path.resolve())
        frames.append(frame)
    return frames


def _validate_prediction(
    frame: pd.DataFrame,
    labels: pd.DataFrame,
    fold_by_test: dict[str, tuple[int, str]],
) -> dict[str, Any]:
    model = str(frame["model"].iloc[0])
    architecture, configuration = MODEL_TO_PARTS[model]
    seeds = tuple(sorted(pd.to_numeric(frame["seed"], errors="raise").astype(int).unique()))
    if len(seeds) != 1 or seeds[0] not in SEEDS:
        raise ValueError(f"{model} prediction file has invalid seeds: {seeds}")
    seed = seeds[0]
    if len(frame) != 81 or frame.duplicated(KEYS).any():
        raise ValueError(f"{model} seed {seed} does not contain 81 unique predictions")
    contract_labels = labels[[*KEYS, *TARGETS]].rename(
        columns={target: f"{target}_contract" for target in TARGETS}
    )
    merged = frame.merge(contract_labels, on=KEYS, how="outer", validate="one_to_one", indicator=True)
    if not (merged["_merge"] == "both").all():
        raise ValueError(f"{model} seed {seed} keys differ from the contract")
    for target in TARGETS:
        if not np.allclose(
            merged[f"{target}_true"], merged[f"{target}_contract"], atol=1e-6, rtol=0.0
        ):
            raise ValueError(f"{model} seed {seed} {target} truths differ from the contract")
        predictions = merged[f"{target}_pred"].to_numpy(dtype=float)
        if not np.isfinite(predictions).all() or np.any((predictions < 0.0) | (predictions > 1.0)):
            raise ValueError(f"{model} seed {seed} has invalid {target} predictions")
    expected_modalities = CONFIGURATIONS[configuration]
    if tuple(frame["modalities"].astype(str).unique()) != ("+".join(expected_modalities),):
        raise ValueError(f"{model} seed {seed} has the wrong modality configuration")
    if set(frame["split_protocol"].astype(str)) != {"shared_7_train_1_validation_1_test"}:
        raise ValueError(f"{model} seed {seed} has the wrong split protocol")
    if not _as_true(frame["cuda_used"]).all():
        raise ValueError(f"{model} seed {seed} has a non-CUDA prediction")
    for participant, group in frame.groupby("participant_id"):
        fold_index, validation = fold_by_test[str(participant)]
        if set(pd.to_numeric(group["fold_index"], errors="raise").astype(int)) != {fold_index}:
            raise ValueError(f"{model} seed {seed} has the wrong fold for {participant}")
        if set(group["validation_participant"].astype(str)) != {validation}:
            raise ValueError(f"{model} seed {seed} has the wrong validation participant")
        if set(group["test_participant"].astype(str)) != {str(participant)}:
            raise ValueError(f"{model} seed {seed} has the wrong test participant")
    return {
        "model": model,
        "architecture": architecture,
        "configuration": configuration,
        "seed": seed,
        "rows": len(frame),
        "participants": int(frame["participant_id"].nunique()),
        "folds": int(frame["fold_index"].nunique()),
        "prediction_coverage_ok": True,
        "fold_mapping_ok": True,
        "modality_configuration_ok": True,
        "cuda_policy_ok": True,
        "source_file": str(frame["source_file"].iloc[0]),
        "source_sha256": file_sha256(Path(str(frame["source_file"].iloc[0]))),
    }


def _validate_results(
    root: Path,
    contract: dict[str, Any],
    train_windows: dict[int, int],
) -> tuple[list[dict[str, Any]], str]:
    expected_hashes = {
        "labels": contract["files"]["labels"]["sha256"],
        "windows": contract["files"]["windows"]["sha256"],
        "split_manifest": contract["files"]["split_manifest"]["sha256"],
        "mask_manifest": contract["files"]["common_masks"]["sha256"],
        "cohorts": contract["files"]["cohorts"]["sha256"],
    }
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    cache_hashes: set[str] = set()
    for path in sorted(root.rglob("*_results.json")):
        payload = _json(path)
        corrected_model = str(payload.get("model", ""))
        model = corrected_model.removeprefix("corrected_")
        if model not in MODEL_TO_PARTS:
            raise ValueError(f"Unexpected result model in {path}: {corrected_model}")
        architecture, configuration = MODEL_TO_PARTS[model]
        seed = int(payload["seed"])
        if (model, seed) in seen:
            raise ValueError(f"Duplicate result for {model} seed {seed}")
        seen.add((model, seed))
        if seed not in SEEDS or payload.get("cohort") != "eeg_eligible":
            raise ValueError(f"Wrong seed/cohort in {path}")
        if payload.get("modalities") != list(CONFIGURATIONS[configuration]):
            raise ValueError(f"Wrong modalities in {path}")
        for name, expected_hash in expected_hashes.items():
            if payload.get("inputs", {}).get(name, {}).get("sha256") != expected_hash:
                raise ValueError(f"Wrong {name} hash in {path}")
        cache_hashes.add(str(payload.get("inputs", {}).get("embedding_cache", {}).get("sha256")))
        folds = payload.get("folds", [])
        if len(folds) != 9 or not payload.get("runtime", {}).get("cuda_used"):
            raise ValueError(f"Wrong fold count or runtime in {path}")
        for fold in folds:
            fold_index = int(fold["fold_index"])
            counts = fold.get("mask_valid_windows_train", {})
            if (
                int(fold.get("n_train_participants", -1)) != 7
                or int(fold.get("n_validation_participants", -1)) != 1
                or int(fold.get("n_test_participants", -1)) != 1
                or not fold.get("cuda_used")
                or set(counts) != set(CONFIGURATIONS[configuration])
                or set(map(int, counts.values())) != {train_windows[fold_index]}
            ):
                raise ValueError(f"Fold {fold_index} violates the common policy in {path}")
        rows.append(
            {
                "model": model,
                "architecture": architecture,
                "configuration": configuration,
                "seed": seed,
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "cuda_used": True,
                "input_hashes_ok": True,
                "common_window_policy_ok": True,
            }
        )
    expected = {(model, seed) for model in EXPECTED_MODELS for seed in SEEDS}
    if seen != expected:
        raise ValueError(
            f"Result coverage mismatch: missing={sorted(expected - seen)}, extra={sorted(seen - expected)}"
        )
    if len(cache_hashes) != 1 or "None" in cache_hashes:
        raise ValueError(f"Runs do not share exactly one corrected embedding cache: {cache_hashes}")
    return rows, next(iter(cache_hashes))


def _summary(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "mae",
        "participant_macro_mae",
        "rmse",
        "spearman",
        "pearson",
        "ccc",
        "participant_macro_spearman",
        "participant_macro_pairwise_accuracy",
    ]
    summary = seed_metrics.groupby(["model", "target"])[metric_columns].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(part for part in column if part) if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    summary.insert(1, "architecture", summary["model"].map(lambda model: MODEL_TO_PARTS[model][0]))
    summary.insert(2, "configuration", summary["model"].map(lambda model: MODEL_TO_PARTS[model][1]))
    architecture_order = {value: index for index, value in enumerate(ARCHITECTURES)}
    configuration_order = {value: index for index, value in enumerate(CONFIGURATIONS)}
    summary["_architecture_order"] = summary["architecture"].map(architecture_order)
    summary["_configuration_order"] = summary["configuration"].map(configuration_order)
    return summary.sort_values(["_architecture_order", "_configuration_order", "target"]).drop(
        columns=["_architecture_order", "_configuration_order"]
    ).reset_index(drop=True)


def _ablation_tests(participant: pd.DataFrame, bootstrap: int, bootstrap_seed: int) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for architecture_index, architecture in enumerate(ARCHITECTURES):
        pairs = [
            (
                f"{architecture}_full",
                f"{architecture}_{configuration}",
                f"{configuration}_vs_full",
            )
            for configuration in CONFIGURATIONS
            if configuration != "full"
        ]
        family = _paired_comparisons(
            participant,
            pairs,
            bootstrap=bootstrap,
            seed=bootstrap_seed + architecture_index,
            expected_participants=9,
        )
        family.insert(0, "architecture", architecture)
        family.insert(
            1,
            "configuration",
            family["comparison"].str.removesuffix("_vs_full"),
        )
        family.insert(2, "holm_family", f"{architecture}_five_removals_two_targets")
        outputs.append(family)
    return pd.concat(outputs, ignore_index=True)


def _largest_degradation(tests: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (architecture, target), group in tests.groupby(["architecture", "target"], sort=False):
        row = group.loc[group["mean_delta"].idxmax()]
        rows.append(
            {
                "architecture": architecture,
                "target": target,
                "configuration": row["configuration"],
                "removed_modality": str(row["configuration"]).removeprefix("no_"),
                "mean_delta": row["mean_delta"],
                "ci95_low": row["ci95_low"],
                "ci95_high": row["ci95_high"],
                "exact_sign_flip_p": row["exact_sign_flip_p"],
                "holm_p": row["holm_p"],
                "is_observed_degradation": bool(row["mean_delta"] > 0),
                "holm_significant_degradation": bool(
                    row["mean_delta"] > 0 and row["holm_significant_0_05"]
                ),
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract = _load_contract(args.contract_dir)
    labels = pd.read_csv(args.contract_dir / contract["files"]["labels"]["path"])
    if len(labels) != 81 or set(labels["participant_id"].astype(str)) != set(PARTICIPANTS):
        raise ValueError("Contract labels do not contain exactly 81 frozen observations")
    fold_by_test, train_windows = _fold_context(args.contract_dir)

    status_path = args.runs_root / "eeg_eligible_ablation_matrix_status.json"
    status = _json(status_path)
    if (
        not status.get("ok")
        or int(status.get("successful_runs", -1)) != 108
        or int(status.get("failed_runs", -1)) != 0
        or len(status.get("runs", [])) != 108
    ):
        raise ValueError("Formal matrix status is not 108 successful / 0 failed")
    checkpoint_count = len(list(args.runs_root.rglob("fold_*.pt")))
    runner_log_count = len(list(args.runs_root.rglob("runner.log")))
    hard_failure_count = len(list(args.runs_root.rglob("hard_failure.json")))
    if (checkpoint_count, runner_log_count, hard_failure_count) != (972, 108, 0):
        raise ValueError(
            "Formal artifact coverage mismatch: expected 972 checkpoints, 108 logs, and no hard failures"
        )

    frames = _read_predictions(args.runs_root)
    observed = [(str(frame["model"].iloc[0]), int(frame["seed"].iloc[0])) for frame in frames]
    expected = [(model, seed) for model in EXPECTED_MODELS for seed in SEEDS]
    if sorted(observed) != sorted(expected):
        raise ValueError(
            f"Prediction coverage mismatch: missing={sorted(set(expected) - set(observed))}, "
            f"extra={sorted(set(observed) - set(expected))}"
        )
    validation = [_validate_prediction(frame, labels, fold_by_test) for frame in frames]
    provenance, cache_hash = _validate_results(args.runs_root, contract, train_windows)
    normalized = pd.concat(frames, ignore_index=True, sort=False).sort_values(
        ["model", "seed", "participant_id", "presentation_position"]
    ).reset_index(drop=True)
    participant = _participant_metrics(normalized)
    seed_metrics = _seed_metrics(normalized, participant)
    summary = _summary(seed_metrics)
    tests = _ablation_tests(participant, args.bootstrap, args.bootstrap_seed)
    largest = _largest_degradation(tests)
    observed_improvements = tests.loc[tests["mean_delta"] < 0].copy()
    corrected_improvements = observed_improvements.loc[
        observed_improvements["holm_significant_0_05"]
    ].copy()
    corrected_degradations = tests.loc[
        (tests["mean_delta"] > 0) & tests["holm_significant_0_05"]
    ].copy()
    aggregate = (
        tests.groupby(["configuration", "target"], as_index=False)
        .agg(
            mean_delta_across_architectures=("mean_delta", "mean"),
            median_delta_across_architectures=("mean_delta", "median"),
            architectures_degraded=("mean_delta", lambda values: int((values > 0).sum())),
            architectures_improved=("mean_delta", lambda values: int((values < 0).sum())),
            holm_significant_tests=("holm_significant_0_05", "sum"),
        )
    )
    aggregate.insert(1, "removed_modality", aggregate["configuration"].str.removeprefix("no_"))
    overall = (
        tests.groupby("configuration", as_index=False)
        .agg(
            mean_delta_across_architectures_and_targets=("mean_delta", "mean"),
            median_delta_across_architectures_and_targets=("mean_delta", "median"),
            positive_architecture_target_pairs=("mean_delta", lambda values: int((values > 0).sum())),
            negative_architecture_target_pairs=("mean_delta", lambda values: int((values < 0).sum())),
            holm_significant_tests=("holm_significant_0_05", "sum"),
        )
        .sort_values("mean_delta_across_architectures_and_targets", ascending=False)
        .reset_index(drop=True)
    )
    overall.insert(1, "removed_modality", overall["configuration"].str.removeprefix("no_"))

    normalized.to_csv(args.output_dir / "normalized_predictions.csv", index=False)
    pd.DataFrame(validation).to_csv(args.output_dir / "run_validation.csv", index=False)
    pd.DataFrame(provenance).to_csv(args.output_dir / "provenance_validation.csv", index=False)
    participant.to_csv(args.output_dir / "participant_metrics.csv", index=False)
    seed_metrics.to_csv(args.output_dir / "seed_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "ablation_summary.csv", index=False)
    tests.to_csv(args.output_dir / "ablation_vs_full_paired_tests.csv", index=False)
    largest.to_csv(args.output_dir / "largest_degradation_by_architecture_target.csv", index=False)
    observed_improvements.to_csv(args.output_dir / "observed_point_improvements.csv", index=False)
    corrected_degradations.to_csv(args.output_dir / "corrected_significant_degradations.csv", index=False)
    aggregate.to_csv(args.output_dir / "aggregate_modality_effects_by_target.csv", index=False)
    overall.to_csv(args.output_dir / "aggregate_modality_effects_overall.csv", index=False)

    report_summary = summary[
        [
            "architecture",
            "configuration",
            "target",
            "participant_macro_mae_mean",
            "participant_macro_mae_std",
        ]
    ]
    report_tests = tests[
        [
            "architecture",
            "configuration",
            "target",
            "mean_delta",
            "ci95_low",
            "ci95_high",
            "exact_sign_flip_p",
            "holm_p",
            "holm_significant_0_05",
        ]
    ]
    report_path = args.output_dir / "modality_ablation_report.md"
    report_lines = [
        "# Nine-participant EEG-eligible modality-ablation benchmark",
        "",
        "## Formal execution status",
        "",
        "- Formal neural runs: 108 successful, 0 failed.",
        "- Cohort: P003, P004, P007, P008, P009, P011, P012, P013, P015.",
        "- Protocol: 9 folds, each 7 train / 1 validation / 1 test; three seeds.",
        "- Every run and fold records CUDA execution.",
        "- These outputs are separate from the existing 15-participant aligned benchmark.",
        "",
        "## Common-window policy",
        "",
        f"The frozen contract contains {contract['observation_count']} participant-condition observations and "
        f"{contract['window_count']} source windows. The five-way intersection marks "
        f"{contract['common_valid_window_count']} windows valid and copies that same mask to every retained "
        "modality in every ablation. P004/C6 remains an observation but has no common-valid window and is "
        "handled by each model's learned missing token.",
        "",
        "## Three-seed participant-macro MAE",
        "",
        _markdown_table(report_summary),
        "",
        "## Paired removal-versus-full tests",
        "",
        "Delta is ablation minus the corresponding full model participant MAE; positive values indicate "
        "degradation. Each architecture is a separate Holm family of five removals x two targets. "
        "Intervals use 10,000 participant bootstraps, and p-values use the exact 2^9 sign-flip distribution.",
        "",
        _markdown_table(report_tests),
        "",
        "## Largest observed degradation",
        "",
        _markdown_table(largest),
        "",
        "## Cross-architecture modality summary",
        "",
        "These aggregate deltas are descriptive averages across architectures, not an additional inferential "
        "test. Removing ECG has the largest average degradation overall and for relaxation; removing head has "
        "the largest average degradation for discomfort.",
        "",
        _markdown_table(aggregate),
        "",
        _markdown_table(overall),
        "",
        "## Removal improvements",
        "",
        f"There are {len(observed_improvements)} negative point-estimate deltas and "
        f"{len(corrected_improvements)} Holm-significant improvements. Negative point estimates without "
        "corrected significance are descriptive only.",
        "",
        _markdown_table(
            observed_improvements[
                [
                    "architecture",
                    "configuration",
                    "target",
                    "mean_delta",
                    "ci95_low",
                    "ci95_high",
                    "exact_sign_flip_p",
                    "holm_p",
                    "holm_significant_0_05",
                ]
            ]
        ),
        "",
        "## Corrected significant degradations",
        "",
        f"There are {len(corrected_degradations)} Holm-significant degradations.",
        "",
        _markdown_table(
            corrected_degradations[
                [
                    "architecture",
                    "configuration",
                    "target",
                    "mean_delta",
                    "ci95_low",
                    "ci95_high",
                    "exact_sign_flip_p",
                    "holm_p",
                ]
            ]
        ),
        "",
        "## Validation and scope",
        "",
        f"The evaluator validated {len(frames)} prediction files, {len(normalized)} model predictions, "
        f"{len(provenance)} result files, {checkpoint_count} fold checkpoints, {runner_log_count} runner logs, "
        f"zero hard failures, one corrected embedding-cache hash (`{cache_hash}`), the contract "
        "hashes, exact fold mapping, complete prediction coverage, frozen modality masks, and CUDA provenance. "
        "Conclusions apply only to this nine-person cohort, fixed common-window policy, aligned settings, and "
        "three seeds. They must not be combined with the separate 15-participant results.",
        "",
        "## Reproduction commands",
        "",
        f"Contract/evaluator working directory: `{ROOT.resolve()}`.",
        "",
        "```powershell",
        "C:\\Users\\linki\\miniconda3\\envs\\rtml-p002-p016\\python.exe analysis/supplementary/build_eeg_eligible_ablation_contract.py --base-contract-dir artifacts/cross_project_alignment_2026-07-16/alignment_contract --output-dir artifacts/cross_project_alignment_2026-07-16/eeg_eligible_ablation/contract",
        "```",
        "",
        "Project B working directory: `/home/link/Wei/Models/core/real-time-vis-physio-fusion`.",
        "",
        "```powershell",
        "wsl.exe -d Ubuntu --cd /home/link/Wei/Models/core/real-time-vis-physio-fusion /home/link/miniconda3/envs/egoEMOTION/bin/python scripts/run_relax_eeg_eligible_ablation_matrix.py --contract-dir /mnt/c/Users/linki/amaster/data_collection_v3/analysis/real_time_inference/artifacts/cross_project_alignment_2026-07-16/eeg_eligible_ablation/contract --embedding-cache /home/link/Wei/Models/core/real-time-vis-physio-fusion/artifacts/relax/aligned_20260716/condition_embeddings.pt --output-root /mnt/c/Users/linki/amaster/data_collection_v3/analysis/real_time_inference/artifacts/cross_project_alignment_2026-07-16/eeg_eligible_ablation/runs --resume",
        "```",
        "",
        f"Evaluator working directory: `{ROOT.resolve()}`.",
        "",
        "```powershell",
        "C:\\Users\\linki\\miniconda3\\envs\\rtml-p002-p016\\python.exe analysis/supplementary/evaluate_eeg_eligible_ablation.py --contract-dir artifacts/cross_project_alignment_2026-07-16/eeg_eligible_ablation/contract --runs-root artifacts/cross_project_alignment_2026-07-16/eeg_eligible_ablation/runs --output-dir artifacts/cross_project_alignment_2026-07-16/eeg_eligible_ablation/evaluation --bootstrap 10000 --bootstrap-seed 20260716",
        "```",
        "",
        "## Artifact paths",
        "",
        f"- Contract: `{args.contract_dir.resolve()}`",
        f"- Formal run root and per-run logs: `{args.runs_root.resolve()}`",
        f"- Matrix status: `{status_path.resolve()}`",
        f"- Evaluation directory: `{args.output_dir.resolve()}`",
        f"- This report: `{report_path.resolve()}`",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    audit = {
        "schema_version": "eeg_eligible_modality_ablation_evaluation_v1",
        "ok": True,
        "contract_sha256": file_sha256(args.contract_dir / "contract.json"),
        "matrix_status_sha256": file_sha256(status_path),
        "embedding_cache_sha256": cache_hash,
        "participants": list(PARTICIPANTS),
        "architectures": list(ARCHITECTURES),
        "configurations": list(CONFIGURATIONS),
        "seeds": list(SEEDS),
        "prediction_runs": len(frames),
        "prediction_rows": len(normalized),
        "result_files": len(provenance),
        "fold_checkpoints": checkpoint_count,
        "runner_logs": runner_log_count,
        "hard_failures": hard_failure_count,
        "fold_count": 9,
        "train_participants_per_fold": 7,
        "bootstrap": args.bootstrap,
        "exact_sign_flip_assignments": 2**9,
        "holm_families": len(ARCHITECTURES),
        "holm_tests_per_family": 10,
        "observed_point_improvements": len(observed_improvements),
        "holm_significant_improvements": len(corrected_improvements),
        "holm_significant_degradations": len(corrected_degradations),
        "largest_average_degradation_overall": {
            "removed_modality": str(overall.iloc[0]["removed_modality"]),
            "mean_delta": float(overall.iloc[0]["mean_delta_across_architectures_and_targets"]),
        },
        "outputs": sorted(
            {path.name for path in args.output_dir.iterdir() if path.is_file()}
            | {"evaluation_audit.json"}
        ),
    }
    (args.output_dir / "evaluation_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return audit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = ROOT / "artifacts" / "cross_project_alignment_2026-07-16" / "eeg_eligible_ablation"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", type=Path, default=root / "contract")
    parser.add_argument("--runs-root", type=Path, default=root / "runs")
    parser.add_argument("--output-dir", type=Path, default=root / "evaluation")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260716)
    args = parser.parse_args(argv)
    if args.bootstrap < 1000:
        parser.error("--bootstrap must be at least 1000")
    return args


def main(argv: list[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

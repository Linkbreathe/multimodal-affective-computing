"""Evaluate the preregistered Relax embedding ladder with paired inference."""

from __future__ import annotations

import argparse
from hashlib import sha256
import itertools
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


STAGES = (
    "s0_legacy",
    "s1_reve_native_raw4",
    "s2_reve_native_linked2",
    "s3_reve_clean_linked2",
    "s4_neurorvq_clean_z",
    "s5_neurorvq_clean_native",
    "s6_neurorvq_eeg_ecg",
    "r3_reve_official_session",
    "ecg_only_neurorvq",
)
SEEDS = (20260705, 20260706, 20260707)
TARGETS = ("relaxation", "discomfort")
OUTCOMES = ("relaxation", "discomfort", "macro")
MODALITIES = ("eeg", "ecg", "eye", "head", "video")
DEFAULT_ROOT = ROOT / "artifacts/relax/neurorvq_embedding_ladder_20260718_corrected"

COMPARISONS: tuple[tuple[str, str, str, str], ...] = (
    ("primary_s5_minus_s3", "s5_neurorvq_clean_native", "s3_reve_clean_linked2", "primary"),
    ("s1_minus_s0", "s1_reve_native_raw4", "s0_legacy", "sequential"),
    ("s2_minus_s1", "s2_reve_native_linked2", "s1_reve_native_raw4", "sequential"),
    ("s3_minus_s2", "s3_reve_clean_linked2", "s2_reve_native_linked2", "sequential"),
    ("s4_minus_s3", "s4_neurorvq_clean_z", "s3_reve_clean_linked2", "sequential"),
    ("s5_minus_s4", "s5_neurorvq_clean_native", "s4_neurorvq_clean_z", "sequential"),
    ("s6_minus_s5", "s6_neurorvq_eeg_ecg", "s5_neurorvq_clean_native", "sequential"),
    ("r3_minus_s3", "r3_reve_official_session", "s3_reve_clean_linked2", "sensitivity"),
    ("ecg_only_minus_s3", "ecg_only_neurorvq", "s3_reve_clean_linked2", "sensitivity"),
)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _run_paths(root: Path, stage: str, seed: int) -> tuple[Path, Path, Path]:
    run_dir = root / "fusion/runs" / stage / f"seed_{seed}"
    stem = f"modality_expert_simplex5_full_s{seed}"
    return (
        run_dir / f"{stem}_results.json",
        run_dir / f"{stem}_predictions.csv",
        run_dir / f"{stem}_folds.json",
    )


def _load_runs(root: Path) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[tuple[str, int], pd.DataFrame]]:
    results: dict[tuple[str, int], dict[str, Any]] = {}
    predictions: dict[tuple[str, int], pd.DataFrame] = {}
    reference_keys: pd.DataFrame | None = None
    reference_truth: np.ndarray | None = None
    for stage in STAGES:
        for seed in SEEDS:
            result_path, prediction_path, _ = _run_paths(root, stage, seed)
            if not result_path.is_file() or not prediction_path.is_file():
                raise FileNotFoundError(f"Incomplete fusion run: stage={stage}, seed={seed}")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            frame = pd.read_csv(prediction_path).sort_values(
                ["participant_id", "condition"]
            ).reset_index(drop=True)
            if len(frame) != 81 or frame.duplicated(["participant_id", "condition"]).any():
                raise ValueError(f"Expected 81 unique predictions for {stage}/{seed}")
            keys = frame[["participant_id", "condition"]]
            truth = frame[[f"{target}_true" for target in TARGETS]].to_numpy(dtype=float)
            if reference_keys is None:
                reference_keys = keys
                reference_truth = truth
            elif not keys.equals(reference_keys) or not np.allclose(
                truth, reference_truth, atol=0.0, rtol=0.0
            ):
                raise ValueError(f"Prediction keys or truths changed for {stage}/{seed}")
            if result["candidate"] != "modality_expert_simplex5" or result["variant"] != "full":
                raise ValueError(f"Unexpected fusion method for {stage}/{seed}")
            if int(result["seed"]) != seed:
                raise ValueError(f"Seed mismatch for {stage}/{seed}")
            results[(stage, seed)] = result
            predictions[(stage, seed)] = frame
    return results, predictions


def _participant_errors(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame[["participant_id"]].copy()
    for target in TARGETS:
        output[target] = np.abs(
            frame[f"{target}_true"].to_numpy(dtype=float)
            - frame[f"{target}_pred"].to_numpy(dtype=float)
        )
    grouped = output.groupby("participant_id", sort=True)[list(TARGETS)].mean()
    grouped["macro"] = grouped[list(TARGETS)].mean(axis=1)
    return grouped


def _condition_errors(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame[["participant_id"]].copy()
    for target in TARGETS:
        output[target] = np.abs(
            frame[f"{target}_true"].to_numpy(dtype=float)
            - frame[f"condition_only_{target}"].to_numpy(dtype=float)
        )
    grouped = output.groupby("participant_id", sort=True)[list(TARGETS)].mean()
    grouped["macro"] = grouped[list(TARGETS)].mean(axis=1)
    return grouped


def _stage_summary(
    results: Mapping[tuple[str, int], Mapping[str, Any]],
    predictions: Mapping[tuple[str, int], pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stage in STAGES:
        seed_values: dict[str, list[float]] = {outcome: [] for outcome in OUTCOMES}
        condition_values: dict[str, list[float]] = {outcome: [] for outcome in OUTCOMES}
        for seed in SEEDS:
            participant = _participant_errors(predictions[(stage, seed)])
            condition = _condition_errors(predictions[(stage, seed)])
            for outcome in OUTCOMES:
                seed_values[outcome].append(float(participant[outcome].mean()))
                condition_values[outcome].append(float(condition[outcome].mean()))
        record: dict[str, Any] = {
            "stage": stage,
            "eeg_dimension": int(results[(stage, SEEDS[0])]["feature_extraction"]["modalities"]["eeg"]["embedding_dimension"]),
            "ecg_dimension": int(results[(stage, SEEDS[0])]["feature_extraction"]["modalities"]["ecg"]["embedding_dimension"]),
        }
        for outcome in OUTCOMES:
            values = np.asarray(seed_values[outcome], dtype=float)
            baseline = np.asarray(condition_values[outcome], dtype=float)
            record[f"{outcome}_mae_mean"] = float(values.mean())
            record[f"{outcome}_mae_sd"] = float(values.std(ddof=1))
            record[f"{outcome}_mae_min"] = float(values.min())
            record[f"{outcome}_mae_max"] = float(values.max())
            record[f"{outcome}_condition_mae_mean"] = float(baseline.mean())
            record[f"{outcome}_delta_vs_condition_mean"] = float((values - baseline).mean())
            for index, seed in enumerate(SEEDS):
                record[f"{outcome}_mae_seed_{seed}"] = float(values[index])
        rows.append(record)
    return pd.DataFrame(rows)


def _exact_sign_flip(values: np.ndarray) -> tuple[float, float]:
    deltas = np.asarray(values, dtype=float)
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(deltas))))
    observed = float(deltas.mean())
    null = np.mean(signs * deltas[None, :], axis=1)
    pvalue = float(np.mean(np.abs(null) >= abs(observed) - 1e-15))
    return observed, pvalue


def _bootstrap_interval(values: np.ndarray, draws: np.ndarray) -> tuple[float, float]:
    deltas = np.asarray(values, dtype=float)
    distribution = deltas[draws].mean(axis=1)
    low, high = np.quantile(distribution, (0.025, 0.975))
    return float(low), float(high)


def _holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=lambda key: (pvalues[key], key))
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, key in enumerate(ordered):
        value = min(1.0, (total - rank) * float(pvalues[key]))
        running = max(running, value)
        adjusted[key] = running
    return adjusted


def _comparison_summary(
    predictions: Mapping[tuple[str, int], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    participants = sorted(
        predictions[(STAGES[0], SEEDS[0])]["participant_id"].astype(str).unique()
    )
    if len(participants) != 9:
        raise ValueError(f"Expected nine inference clusters, got {participants}")
    bootstrap_draws = np.random.default_rng(20260718).integers(
        0, len(participants), size=(10000, len(participants))
    )
    summary_rows: list[dict[str, Any]] = []
    participant_rows: list[dict[str, Any]] = []
    for name, stage, reference, family in COMPARISONS:
        seed_deltas: dict[str, list[float]] = {outcome: [] for outcome in OUTCOMES}
        participant_deltas: dict[str, list[np.ndarray]] = {
            outcome: [] for outcome in OUTCOMES
        }
        for seed in SEEDS:
            current = _participant_errors(predictions[(stage, seed)]).loc[participants]
            baseline = _participant_errors(predictions[(reference, seed)]).loc[participants]
            delta = current - baseline
            for outcome in OUTCOMES:
                values = delta[outcome].to_numpy(dtype=float)
                seed_deltas[outcome].append(float(values.mean()))
                participant_deltas[outcome].append(values)
        for outcome in OUTCOMES:
            by_seed = np.asarray(seed_deltas[outcome], dtype=float)
            by_participant = np.mean(
                np.stack(participant_deltas[outcome], axis=0), axis=0
            )
            observed, sign_p = _exact_sign_flip(by_participant)
            ci_low, ci_high = _bootstrap_interval(by_participant, bootstrap_draws)
            summary_rows.append(
                {
                    "comparison": name,
                    "stage": stage,
                    "reference_stage": reference,
                    "family": family,
                    "outcome": outcome,
                    "delta_mean": observed,
                    "delta_seed_mean": float(by_seed.mean()),
                    "delta_seed_sd": float(by_seed.std(ddof=1)),
                    "delta_seed_min": float(by_seed.min()),
                    "delta_seed_max": float(by_seed.max()),
                    **{
                        f"delta_seed_{seed}": float(by_seed[index])
                        for index, seed in enumerate(SEEDS)
                    },
                    "participant_bootstrap_ci_low": ci_low,
                    "participant_bootstrap_ci_high": ci_high,
                    "exact_sign_flip_p_two_sided": sign_p,
                    "participants_improved": int(np.sum(by_participant < 0)),
                    "participants_tied": int(np.sum(np.isclose(by_participant, 0.0, atol=1e-15))),
                    "participants_worsened": int(np.sum(by_participant > 0)),
                }
            )
            for index, participant in enumerate(participants):
                participant_rows.append(
                    {
                        "comparison": name,
                        "stage": stage,
                        "reference_stage": reference,
                        "family": family,
                        "outcome": outcome,
                        "participant_id": participant,
                        "seed_averaged_mae_delta": float(by_participant[index]),
                    }
                )
    summary = pd.DataFrame(summary_rows)
    participants_frame = pd.DataFrame(participant_rows)

    sequential_macro = summary.loc[
        (summary["family"] == "sequential") & (summary["outcome"] == "macro")
    ]
    adjusted = _holm(
        dict(
            zip(
                sequential_macro["comparison"],
                sequential_macro["exact_sign_flip_p_two_sided"],
                strict=True,
            )
        )
    )
    summary["holm_adjusted_p"] = np.nan
    for comparison, pvalue in adjusted.items():
        selected = (summary["comparison"] == comparison) & (summary["outcome"] == "macro")
        summary.loc[selected, "holm_adjusted_p"] = pvalue

    primary_targets = summary.loc[
        (summary["comparison"] == "primary_s5_minus_s3")
        & summary["outcome"].isin(TARGETS)
    ]
    adjusted_targets = _holm(
        dict(
            zip(
                primary_targets["outcome"],
                primary_targets["exact_sign_flip_p_two_sided"],
                strict=True,
            )
        )
    )
    for outcome, pvalue in adjusted_targets.items():
        selected = (
            (summary["comparison"] == "primary_s5_minus_s3")
            & (summary["outcome"] == outcome)
        )
        summary.loc[selected, "holm_adjusted_p"] = pvalue
    return summary, participants_frame


def _weight_summary(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stage in STAGES:
        for seed in SEEDS:
            _, _, fold_path = _run_paths(root, stage, seed)
            folds = json.loads(fold_path.read_text(encoding="utf-8"))
            for fold in folds:
                for target in TARGETS:
                    weights = fold["targets"][target]["expert_weights"]
                    for modality in MODALITIES:
                        rows.append(
                            {
                                "stage": stage,
                                "seed": seed,
                                "fold_index": int(fold["fold_index"]),
                                "test_participant": str(fold["test_participant"]),
                                "target": target,
                                "modality": modality,
                                "weight": float(weights[modality]),
                            }
                        )
    return pd.DataFrame(rows)


def _discomfort_strata(
    predictions: Mapping[tuple[str, int], pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for stage in STAGES:
        for seed in SEEDS:
            frame = predictions[(stage, seed)]
            truth = frame["discomfort_true"].to_numpy(dtype=float)
            prediction = frame["discomfort_pred"].to_numpy(dtype=float)
            for stratum, selected in (
                ("zero", np.isclose(truth, 0.0)),
                ("nonzero", ~np.isclose(truth, 0.0)),
            ):
                rows.append(
                    {
                        "stage": stage,
                        "seed": seed,
                        "stratum": stratum,
                        "observations": int(selected.sum()),
                        "mae": float(np.mean(np.abs(truth[selected] - prediction[selected]))),
                    }
                )
    return pd.DataFrame(rows)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    execution_path = args.root / "fusion/fusion_execution_manifest.json"
    if not execution_path.is_file():
        raise FileNotFoundError(f"Fusion matrix is not complete: {execution_path}")
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    if int(execution["completed_runs"]) != 27:
        raise ValueError("Fusion execution manifest does not contain all 27 runs")
    results, predictions = _load_runs(args.root)
    stage_summary = _stage_summary(results, predictions)
    comparisons, participant_deltas = _comparison_summary(predictions)
    weights = _weight_summary(args.root)
    discomfort = _discomfort_strata(predictions)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "stage_summary": output_dir / "stage_metrics.csv",
        "comparisons": output_dir / "paired_comparisons.csv",
        "participant_deltas": output_dir / "participant_deltas.csv",
        "expert_weights": output_dir / "expert_weights.csv",
        "discomfort_strata": output_dir / "discomfort_strata.csv",
    }
    stage_summary.to_csv(paths["stage_summary"], index=False)
    comparisons.to_csv(paths["comparisons"], index=False)
    participant_deltas.to_csv(paths["participant_deltas"], index=False)
    weights.to_csv(paths["expert_weights"], index=False)
    discomfort.to_csv(paths["discomfort_strata"], index=False)

    primary = comparisons.loc[
        comparisons["comparison"] == "primary_s5_minus_s3"
    ].set_index("outcome")
    joint = comparisons.loc[
        (comparisons["comparison"] == "s6_minus_s5")
        & (comparisons["outcome"] == "macro")
    ].iloc[0]
    descriptive = bool(
        all(
            primary.loc["macro", f"delta_seed_{seed}"] < 0
            for seed in SEEDS
        )
        and primary.loc["relaxation", "delta_mean"] <= 0
        and primary.loc["discomfort", "delta_mean"] <= 0
    )
    stronger = bool(
        descriptive
        and primary.loc["macro", "participant_bootstrap_ci_high"] < 0
        and primary.loc["macro", "exact_sign_flip_p_two_sided"] < 0.05
    )
    ranked = stage_summary.sort_values("macro_mae_mean")[[
        "stage", "macro_mae_mean", "relaxation_mae_mean", "discomfort_mae_mean"
    ]]
    result = {
        "schema_version": "relax_embedding_ladder_fusion_evaluation_v1",
        "stages": list(STAGES),
        "seeds": list(SEEDS),
        "runs": 27,
        "primary_comparison": primary.reset_index().to_dict(orient="records"),
        "success_rules": {
            "descriptive_pass": descriptive,
            "stronger_support_pass": stronger,
            "joint_s6_minus_s5_macro_improves": bool(joint["delta_mean"] < 0),
        },
        "best_stage_by_macro_mae": ranked.iloc[0].to_dict(),
        "stage_ranking": ranked.to_dict(orient="records"),
        "outputs": {key: str(path.resolve()) for key, path in paths.items()},
        "execution_manifest": {
            "path": str(execution_path.resolve()),
            "sha256": _sha256(execution_path),
        },
        "evaluator_source_sha256": _sha256(Path(__file__)),
    }
    _write_json(output_dir / "evaluation_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir or args.root / "evaluation"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    evaluate(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

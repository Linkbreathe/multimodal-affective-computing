"""Shared registration, resume, and result-integrity checks for Relax suites."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


LOCKED_ARGUMENTS: dict[str, Any] = {
    "batch_size": 16,
    "max_epochs": 40,
    "patience": 8,
    "d_common": 64,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "fusion_depth": 1,
    "fusion_heads": 2,
    "torch_threads": 1,
}


def parse_command(command: list[str]) -> dict[str, Any]:
    """Parse the simple argparse-style command lists emitted by suite scripts."""
    parsed: dict[str, Any] = {}
    index = 2
    while index < len(command):
        token = command[index]
        if not token.startswith("--"):
            index += 1
            continue
        key = token[2:].replace("-", "_")
        if key in {"strict", "smoke"}:
            parsed[key] = True
            index += 1
            continue
        values: list[str] = []
        index += 1
        while index < len(command) and not command[index].startswith("--"):
            values.append(command[index])
            index += 1
        if key == "modalities":
            parsed[key] = values
        elif not values:
            parsed[key] = True
        else:
            parsed[key] = values[0]
    return parsed


def result_paths(command: list[str]) -> tuple[Path, Path]:
    args = parse_command(command)
    output_dir = Path(str(args["output_dir"]))
    cohort = str(args["cohort"])
    baseline = str(args.get("baseline", "none"))
    if baseline != "none":
        run_name = f"{cohort}_{baseline}"
    else:
        run_name = f"{cohort}_{args.get('fusion', 'early')}_{args.get('target', 'original')}_{args.get('pool', 'sequence')}"
    return output_dir / f"{run_name}_results.json", output_dir / f"{run_name}_predictions.csv"


def _cohort_participants(cohorts_path: str | Path, cohort: str) -> list[str]:
    payload = json.loads(Path(cohorts_path).read_text(encoding="utf-8"))
    return [str(value) for value in payload[cohort]["participants"]]


def validate_job_output(command: list[str]) -> tuple[bool, str]:
    """Accept a resumable job only when args, folds, and predictions are complete."""
    expected = parse_command(command)
    result_path, prediction_path = result_paths(command)
    hard_failure = result_path.parent / "hard_failure.json"
    if hard_failure.exists():
        return False, f"hard failure marker exists: {hard_failure}"
    if not result_path.exists() or not prediction_path.exists():
        return False, "result JSON or prediction CSV is missing"
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result_args = result["args"]
        folds = result["folds"]
        predictions = pd.read_csv(prediction_path)
    except Exception as error:
        return False, f"could not read result payload: {error}"

    comparable = {
        "embedding_cache",
        "cohorts",
        "cohort",
        "modalities",
        "fusion",
        "baseline",
        "handcrafted_features",
        "target",
        "condition_control",
        "pool",
        "seed",
        "device",
        "batch_size",
        "max_epochs",
        "patience",
        "lr",
        "weight_decay",
        "d_common",
        "fusion_depth",
        "fusion_heads",
        "dim_head",
        "latent_dim",
        "latent_channels",
        "run_tag",
        "torch_threads",
    }
    numeric_int = {
        "seed", "batch_size", "max_epochs", "patience", "d_common", "fusion_depth",
        "fusion_heads", "dim_head", "latent_dim", "latent_channels", "torch_threads",
    }
    numeric_float = {"lr", "weight_decay"}
    for key in sorted(comparable & set(expected)):
        wanted: Any = expected[key]
        if key in numeric_int:
            wanted = int(wanted)
        elif key in numeric_float:
            wanted = float(wanted)
        actual = result_args.get(key)
        if actual != wanted:
            return False, f"argument mismatch for {key}: {actual!r} != {wanted!r}"

    if not expected.get("smoke"):
        for key, wanted in LOCKED_ARGUMENTS.items():
            if result_args.get(key) != wanted:
                return False, f"locked argument mismatch for {key}: {result_args.get(key)!r} != {wanted!r}"

    try:
        participants = _cohort_participants(expected["cohorts"], str(expected["cohort"]))
    except Exception as error:
        return False, f"could not resolve cohort participants: {error}"
    if len(folds) != len(participants):
        return False, f"fold count {len(folds)} != cohort size {len(participants)}"
    test_participants = [str(fold.get("test_participant")) for fold in folds]
    if sorted(test_participants) != sorted(participants):
        return False, "fold test participants do not match the locked cohort"

    required_columns = {
        "participant_id", "condition", "relaxation_pred", "discomfort_pred",
        "relaxation_true", "discomfort_true", "fold",
    }
    missing_columns = sorted(required_columns - set(predictions.columns))
    if missing_columns:
        return False, f"prediction CSV missing columns {missing_columns}"
    if set(predictions["participant_id"].astype(str)) != set(participants):
        return False, "prediction participants do not match the locked cohort"
    baseline = str(expected.get("baseline", "none"))
    expected_rows = len(participants) * 9 * (1000 if baseline == "random_9_condition" else 1)
    if len(predictions) != expected_rows:
        return False, f"prediction row count {len(predictions)} != {expected_rows}"
    if Path(str(result.get("predictions_csv", ""))) != prediction_path:
        return False, "result JSON points to a different prediction CSV"
    return True, "complete"


def suite_registry(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for job in jobs:
        valid, reason = validate_job_output(job["command"])
        output.append({**job, "complete": valid, "validation": reason})
    return output

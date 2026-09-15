"""Statistics and validation helpers for the dynamic-texture five-modality study."""

from __future__ import annotations

from itertools import product
from typing import Any, Iterable

import numpy as np
import pandas as pd

from real_time_ml.experiments.dynamic_texture_five import project_b_style_metrics


TARGETS = ("relaxation", "discomfort")
OUTCOMES = (*TARGETS, "macro")
KEY_COLUMNS = ("participant_id", "condition")


def exact_sign_flip_pvalue(values: Iterable[float]) -> float:
    """Return the exact two-sided paired sign-flip p-value."""
    array = np.asarray(tuple(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan")
    observed = abs(float(array.mean()))
    signs = np.asarray(tuple(product((-1.0, 1.0), repeat=len(array))), dtype=float)
    statistics = np.abs((signs * array[None, :]).mean(axis=1))
    return float(np.mean(statistics >= observed - 1e-15))


def holm_adjust(pvalues: Iterable[float]) -> np.ndarray:
    """Holm step-down multiplicity adjustment, preserving input order."""
    values = np.asarray(tuple(pvalues), dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Holm adjustment requires finite p-values")
    order = np.argsort(values, kind="stable")
    sorted_adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, original_index in enumerate(order):
        running = max(running, (len(values) - rank) * values[original_index])
        sorted_adjusted[rank] = min(1.0, running)
    adjusted = np.empty(len(values), dtype=float)
    adjusted[order] = sorted_adjusted
    return adjusted


def validate_normalized_oof(
    frame: pd.DataFrame,
    *,
    labels: pd.DataFrame,
    split_manifest: pd.DataFrame,
    expected_seed: int,
    expected_variant: str,
) -> None:
    """Validate truth, folds, validation participants, range, and P004/C6 fallback."""
    required = {
        *KEY_COLUMNS,
        "presentation_position",
        "fold_index",
        "validation_participant",
        "test_participant",
        "seed",
        "variant",
        *(f"true_{target}" for target in TARGETS),
        *(f"pred_{target}" for target in TARGETS),
        *(f"condition_only_{target}" for target in TARGETS),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Normalized OOF frame lacks columns: {missing}")
    if len(frame) != 81 or frame.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("Every formal OOF run must contain 81 unique observations")
    if set(pd.to_numeric(frame["seed"], errors="raise").astype(int)) != {int(expected_seed)}:
        raise ValueError("OOF seed differs from the declared run seed")
    if set(frame["variant"].astype(str)) != {expected_variant}:
        raise ValueError("OOF variant differs from the declared run variant")

    truth_columns = [
        *KEY_COLUMNS,
        "presentation_position",
        *(target for target in TARGETS),
    ]
    expected = labels[truth_columns].copy()
    expected = expected.rename(columns={target: f"expected_{target}" for target in TARGETS})
    checked = frame.merge(
        expected,
        on=list(KEY_COLUMNS),
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if len(checked) != 81 or not checked["_merge"].eq("both").all():
        raise ValueError("OOF keys do not match the frozen 81-label contract")
    if not np.array_equal(
        pd.to_numeric(checked["presentation_position_x"]).to_numpy(dtype=int),
        pd.to_numeric(checked["presentation_position_y"]).to_numpy(dtype=int),
    ):
        raise ValueError("Presentation positions differ from the frozen contract")
    for target in TARGETS:
        if not np.allclose(
            checked[f"true_{target}"].to_numpy(dtype=float),
            checked[f"expected_{target}"].to_numpy(dtype=float),
            rtol=0.0,
            atol=5e-7,
        ):
            raise ValueError(f"{target} truth differs from the frozen contract")
        prediction = pd.to_numeric(frame[f"pred_{target}"], errors="raise")
        if prediction.isna().any() or not prediction.between(0.0, 1.0).all():
            raise ValueError(f"{target} predictions must be finite and within [0, 1]")

    expected_folds = split_manifest.loc[
        split_manifest["role"].astype(str).eq("test"),
        ["fold_index", "test_participant", "validation_participant"],
    ].copy()
    expected_folds["test_participant"] = expected_folds["test_participant"].astype(str)
    fold_map = expected_folds.set_index("test_participant")
    for row in frame.itertuples(index=False):
        test = str(row.participant_id)
        expected_fold = fold_map.loc[test]
        if (
            int(row.fold_index) != int(expected_fold["fold_index"])
            or str(row.test_participant) != test
            or str(row.validation_participant) != str(expected_fold["validation_participant"])
        ):
            raise ValueError(f"Fold metadata mismatch for held-out participant {test}")

    zero = frame["participant_id"].astype(str).eq("P004") & frame["condition"].astype(str).eq("C6")
    if int(zero.sum()) != 1:
        raise ValueError("P004/C6 must be the unique zero-window observation")
    for target in TARGETS:
        if not np.array_equal(
            frame.loc[zero, f"pred_{target}"].to_numpy(dtype=float),
            frame.loc[zero, f"condition_only_{target}"].to_numpy(dtype=float),
        ):
            raise ValueError(f"P004/C6 {target} must use the exact Condition-only fallback")


def assert_saved_metrics(
    frame: pd.DataFrame, saved: dict[str, Any], *, atol: float = 1e-12
) -> dict[str, Any]:
    """Recompute the metric contract and compare it with a saved result payload."""
    recomputed = project_b_style_metrics(frame)
    saved_metrics = saved.get("metrics", saved)
    if int(saved_metrics["n_predictions"]) != 81:
        raise ValueError("Saved metrics do not declare 81 OOF predictions")
    for target in TARGETS:
        for metric in ("participant_macro_mae", "rmse", "spearman", "pearson", "ccc"):
            left = float(recomputed["targets"][target][metric])
            right = float(saved_metrics["targets"][target][metric])
            if not np.isclose(left, right, rtol=0.0, atol=atol, equal_nan=True):
                raise ValueError(
                    f"Saved {target}/{metric} differs from independent recomputation: "
                    f"saved={right}, recomputed={left}"
                )
    if not np.isclose(
        recomputed["macro_mae"],
        float(saved_metrics["macro_mae"]),
        rtol=0.0,
        atol=atol,
    ):
        raise ValueError("Saved Macro MAE differs from independent recomputation")
    return recomputed


def seed_averaged_participant_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    """Create participant-level MAE after averaging the three seed-specific errors."""
    rows: list[dict[str, Any]] = []
    group_columns = ["model_key", "seed", "participant_id"]
    for keys, group in predictions.groupby(group_columns, sort=True):
        model_key, seed, participant = keys
        target_values: dict[str, float] = {}
        for target in TARGETS:
            target_values[target] = float(
                np.abs(
                    group[f"pred_{target}"].to_numpy(dtype=float)
                    - group[f"true_{target}"].to_numpy(dtype=float)
                ).mean()
            )
        target_values["macro"] = float(np.mean(tuple(target_values.values())))
        for outcome, mae in target_values.items():
            rows.append(
                {
                    "model_key": model_key,
                    "seed": int(seed),
                    "participant_id": str(participant),
                    "outcome": outcome,
                    "mae": mae,
                }
            )
    per_seed = pd.DataFrame(rows)
    counts = per_seed.groupby(["model_key", "participant_id", "outcome"])["seed"].nunique()
    if not (counts == 3).all():
        raise ValueError("Every model/participant/outcome must contain exactly three seeds")
    return (
        per_seed.groupby(["model_key", "participant_id", "outcome"], as_index=False)["mae"]
        .mean()
        .rename(columns={"mae": "seed_averaged_participant_mae"})
    )


def paired_holm_families(
    participant_errors: pd.DataFrame,
    pairs: list[tuple[str, str, str]],
    *,
    family_prefix: str,
    delta_definition: str,
    bootstrap: int = 10_000,
    seed: int = 20260719,
) -> pd.DataFrame:
    """Evaluate three paired comparisons in a separate Holm family per outcome."""
    if len(pairs) != 3:
        raise ValueError("The formal design requires exactly three paired comparisons")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for outcome in OUTCOMES:
        outcome_rows: list[dict[str, Any]] = []
        subset = participant_errors.loc[participant_errors["outcome"].eq(outcome)]
        for left, right, comparison in pairs:
            left_frame = subset.loc[
                subset["model_key"].eq(left),
                ["participant_id", "seed_averaged_participant_mae"],
            ].rename(columns={"seed_averaged_participant_mae": "left_mae"})
            right_frame = subset.loc[
                subset["model_key"].eq(right),
                ["participant_id", "seed_averaged_participant_mae"],
            ].rename(columns={"seed_averaged_participant_mae": "right_mae"})
            paired = left_frame.merge(right_frame, on="participant_id", validate="one_to_one")
            if len(paired) != 9:
                raise ValueError(f"{comparison}/{outcome} lacks nine paired participants")
            delta = paired["right_mae"].to_numpy(dtype=float) - paired["left_mae"].to_numpy(
                dtype=float
            )
            draws = rng.choice(delta, size=(bootstrap, len(delta)), replace=True).mean(axis=1)
            outcome_rows.append(
                {
                    "holm_family": f"{family_prefix}_{outcome}",
                    "comparison": comparison,
                    "left_model": left,
                    "right_model": right,
                    "outcome": outcome,
                    "delta_definition": delta_definition,
                    "mean_delta": float(delta.mean()),
                    "median_delta": float(np.median(delta)),
                    "ci95_low": float(np.quantile(draws, 0.025)),
                    "ci95_high": float(np.quantile(draws, 0.975)),
                    "exact_sign_flip_p": exact_sign_flip_pvalue(delta),
                    "n_participants": len(delta),
                    "bootstrap_replicates": int(bootstrap),
                }
            )
        pvalues = [row["exact_sign_flip_p"] for row in outcome_rows]
        adjusted = holm_adjust(pvalues)
        for row, holm_p in zip(outcome_rows, adjusted, strict=True):
            row["holm_p"] = float(holm_p)
            row["holm_significant_0_05"] = bool(holm_p <= 0.05)
        rows.extend(outcome_rows)
    result = pd.DataFrame(rows)
    family_sizes = result.groupby("holm_family").size()
    if len(family_sizes) != 3 or not (family_sizes == 3).all():
        raise ValueError("Every formal outcome must contain one three-test Holm family")
    return result

"""I.5 Sequential / habituation observational modelling  [B level].

Models drift of relaxation / discomfort / monotony against presentation
position plus a first-order carryover of the previous condition's stimulus,
and checks whether the current-stimulus (intensity/frequency) effects survive
after netting out order.  If presentation order were unbalanced every effect
would be downgraded and flagged as confounded; here the design is
counterbalanced (each condition's mean position ~5), so the order terms are
reported as observational nuisance structure.

Output: reports/order_carryover.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from io_common import (
    CONDITIONS,
    INFERENCE_PC,
    add_contract_columns,
    ci,
    condition_sort_key,
    ensure_dirs,
    load_labels,
    write_csv,
)

ORDER_TARGETS = ("relaxation", "discomfort", "monotony")


def _prepare(labels: pd.DataFrame) -> pd.DataFrame:
    out = labels.copy()
    out["presentation_position"] = pd.to_numeric(out["presentation_position"], errors="coerce")
    out["intensity"] = pd.to_numeric(out["intensity"], errors="coerce")
    out["frequency"] = pd.to_numeric(out["frequency"], errors="coerce")
    out = out.sort_values(["participant_id", "presentation_position"])
    # first-order carryover: previous condition's stimulus within the same participant
    out["prev_intensity"] = out.groupby("participant_id")["intensity"].shift(1)
    out["prev_frequency"] = out.groupby("participant_id")["frequency"].shift(1)
    return out


def _within_center(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        out[col + "_c"] = out[col] - out.groupby("participant_id")[col].transform("mean")
    return out


def _ols(frame: pd.DataFrame, target: str, terms: list[str]) -> dict[str, float]:
    data = frame[[target, *terms]].dropna()
    if len(data) < len(terms) + 2:
        return {term: float("nan") for term in terms}
    # participant fixed effects are already absorbed by within-centering of
    # predictors; centre the target within participant too.
    y = data[target].to_numpy(float)
    x = np.column_stack([np.ones(len(data))] + [data[t].to_numpy(float) for t in terms])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return {term: float(beta[i + 1]) for i, term in enumerate(terms)}


def order_balance_metric(labels: pd.DataFrame) -> dict[str, float]:
    ct = pd.crosstab(labels["condition"], labels["presentation_position"])
    expected = ct.values.sum() / ct.size
    chi2 = float(((ct.values - expected) ** 2 / expected).sum())
    mean_pos = labels.groupby("condition")["presentation_position"].mean()
    return {
        "chi2_condition_position": chi2,
        "position_spread_across_conditions": float(mean_pos.max() - mean_pos.min()),
        "balanced": bool((mean_pos.max() - mean_pos.min()) < 1.0),
    }


def run(seed: int, replicates: int) -> dict:
    paths = ensure_dirs()
    labels = _prepare(load_labels())

    balance = order_balance_metric(labels)

    # within-participant centred predictors
    centered = _within_center(
        labels, ["presentation_position", "intensity", "frequency", "prev_intensity", "prev_frequency"]
    )
    centered["int_x_freq_c"] = centered["intensity_c"] * centered["frequency_c"]

    base_terms = ["intensity_c", "frequency_c", "int_x_freq_c"]
    order_terms = base_terms + ["presentation_position_c", "prev_intensity_c", "prev_frequency_c"]

    rows: list[dict] = []
    participants = np.asarray(sorted(centered["participant_id"].unique()))
    grouped = {p: centered[centered["participant_id"] == p] for p in participants}

    for target in ORDER_TARGETS:
        for model_name, terms in (("condition_only", base_terms), ("condition_plus_order", order_terms)):
            point = _ols(centered, target, terms)
            # participant bootstrap of each coefficient
            rng = np.random.default_rng(seed + abs(hash((target, model_name))) % 9973)
            boot = {term: [] for term in terms}
            for _ in range(replicates):
                sampled = rng.choice(participants, size=len(participants), replace=True)
                sample = pd.concat([grouped[p] for p in sampled], ignore_index=True)
                est = _ols(sample, target, terms)
                for term in terms:
                    if np.isfinite(est[term]):
                        boot[term].append(est[term])
            for term in terms:
                low, high = ci(boot[term])
                rows.append({
                    "target": target,
                    "model": model_name,
                    "term": term,
                    "estimate": point[term],
                    "ci_low": low,
                    "ci_high": high,
                    "ci_excludes_zero": bool(np.isfinite(low) and np.isfinite(high) and (low > 0 or high < 0)),
                    "order_balanced": balance["balanced"],
                    "inference_unit": INFERENCE_PC,
                })

    # add a balance summary row
    rows.append({
        "target": "design", "model": "order_balance", "term": "position_spread_across_conditions",
        "estimate": balance["position_spread_across_conditions"], "ci_low": np.nan, "ci_high": np.nan,
        "ci_excludes_zero": np.nan, "order_balanced": balance["balanced"], "inference_unit": INFERENCE_PC,
    })

    table = pd.DataFrame(rows)
    write_csv(paths["reports"] / "order_carryover.csv", table)
    return {"table": add_contract_columns(table, INFERENCE_PC), "balance": balance}


if __name__ == "__main__":
    out = run(seed=20260627, replicates=500)
    print("balance:", out["balance"])
    show = out["table"][out["table"]["term"].isin(
        ["intensity_c", "frequency_c", "presentation_position_c", "prev_intensity_c"])]
    print(show.to_string(index=False))

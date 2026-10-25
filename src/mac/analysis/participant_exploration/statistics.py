"""Descriptive summaries only; windows never become independent rating observations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .common import verify_lock, write_csv, write_json
from .specification import CONDITIONS, FEATURES, OUTCOMES, PARTICIPANTS, SPEC, VARIABLES


def describe(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return dict(
        n_valid=len(v),
        mean=float(np.mean(v)) if len(v) else np.nan,
        std=float(np.std(v, ddof=1)) if len(v) > 1 else np.nan,
        median=float(np.median(v)) if len(v) else np.nan,
        minimum=float(np.min(v)) if len(v) else np.nan,
        maximum=float(np.max(v)) if len(v) else np.nan,
        range=float(np.ptp(v)) if len(v) else np.nan,
    )


def direction(value: float, scale: float = 1.0) -> str:
    tol = SPEC["statistics"]["zero_tolerance"] * max(1.0, abs(scale))
    return "zero" if abs(value) <= tol else "positive" if value > 0 else "negative"


def association(x: np.ndarray, y: np.ndarray, conditions: np.ndarray) -> dict:
    paired = np.isfinite(x) & np.isfinite(y)
    x, y = x[paired], y[paired]
    reason = (
        "fewer_than_six_valid_paired_conditions"
        if len(x) < SPEC["statistics"]["minimum_paired_conditions"]
        else "constant_feature_and_rating"
        if np.unique(x).size < 2 and np.unique(y).size < 2
        else "constant_feature"
        if np.unique(x).size < 2
        else "constant_rating"
        if np.unique(y).size < 2
        else ""
    )
    rho = (
        float(np.corrcoef(rankdata(x, method="average"), rankdata(y, method="average"))[0, 1])
        if not reason
        else np.nan
    )
    if np.isfinite(rho) and abs(rho) < SPEC["statistics"]["zero_tolerance"]:
        rho = 0.0
    return dict(
        spearman_rho=rho,
        n_valid_pairs=len(x),
        included_conditions=";".join(conditions[paired]),
        direction=direction(rho) if np.isfinite(rho) else "",
        estimation_status="estimable" if np.isfinite(rho) else "not_estimable",
        reason=reason,
        feature_unique_values=int(np.unique(x).size),
        rating_unique_values=int(np.unique(y).size),
    )


def analyze(output: Path) -> None:
    verify_lock(output)
    if (output / "COMPLETE.json").exists():
        raise FileExistsError("Completed output is immutable.")
    wide = pd.read_csv(output / "participant_condition_features.csv")
    if len(wide) != 81 or wide.duplicated(["participant_id", "condition"]).any():
        raise ValueError("Expected 81 unique cohort records before analysis")
    centered = []
    summaries = []
    associations = []
    for pid in PARTICIPANTS:
        part = wide.loc[wide.participant_id.eq(pid)].sort_values("condition")
        for variable in VARIABLES:
            raw = part[variable + "_raw"].to_numpy(float)
            analysis = raw if variable in OUTCOMES else part[variable + "_analysis"].to_numpy(float)
            r, a = describe(raw), describe(analysis)
            summaries.append(
                dict(
                    participant_id=pid,
                    variable=variable,
                    **{"raw_" + k: v for k, v in r.items()},
                    **{"analysis_" + k: v for k, v in a.items()},
                )
            )
            for i, row in enumerate(part.to_dict("records")):
                centered.append(
                    dict(
                        participant_id=pid,
                        condition=row["condition"],
                        presentation_order=row["presentation_order"],
                        intensity=row["intensity"],
                        frequency=row["frequency"],
                        variable=variable,
                        raw_value=raw[i],
                        analysis_value=analysis[i],
                        raw_participant_mean=r["mean"],
                        analysis_participant_mean=a["mean"],
                        raw_deviation=raw[i] - r["mean"],
                        analysis_deviation=analysis[i] - a["mean"],
                        n_raw_conditions=r["n_valid"],
                        n_analysis_conditions=a["n_valid"],
                    )
                )
        for feature in FEATURES:
            x = part[feature + "_analysis"].to_numpy(float)
            for outcome in OUTCOMES:
                y = part[outcome + "_raw"].to_numpy(float)
                # Subtract each variable's all-valid-condition mean, not a paired-only mean.
                xc = x - np.mean(x[np.isfinite(x)]) if np.isfinite(x).any() else x
                yc = y - np.mean(y[np.isfinite(y)]) if np.isfinite(y).any() else y
                associations.append(
                    dict(
                        participant_id=pid,
                        feature=feature,
                        outcome=outcome,
                        **association(xc, yc, part.condition.to_numpy()),
                    )
                )
    long = pd.DataFrame(centered)
    write_csv(output / "within_person_centered.csv", long)
    write_csv(output / "participant_descriptive_summary.csv", pd.DataFrame(summaries))
    individual = pd.DataFrame(associations)
    write_csv(output / "individual_associations.csv", individual)
    responses = []
    for condition in CONDITIONS:
        for variable in VARIABLES:
            group = long.loc[long.condition.eq(condition) & long.variable.eq(variable)]
            record = dict(
                condition=condition,
                variable=variable,
                intensity=float(group.intensity.iloc[0]),
                frequency=float(group.frequency.iloc[0]),
            )
            for scale in ["raw", "analysis"]:
                values = group[scale + "_deviation"].to_numpy(float)
                stats = describe(values)
                record.update({scale + "_" + k: v for k, v in stats.items()})
                valid = group.loc[np.isfinite(values)]
                signs = [
                    direction(v, m)
                    for v, m in zip(valid[scale + "_deviation"], valid[scale + "_participant_mean"])
                ]
                record.update(
                    {scale + "_n_" + d: signs.count(d) for d in ["positive", "negative", "zero"]}
                )
                record[scale + "_participants"] = ";".join(valid.participant_id)
            responses.append(record)
    write_csv(output / "condition_response_summary.csv", pd.DataFrame(responses))
    cross = []
    for feature in FEATURES:
        for outcome in OUTCOMES:
            group = individual.loc[individual.feature.eq(feature) & individual.outcome.eq(outcome)]
            values = group.spearman_rho.dropna().to_numpy(float)
            cross.append(
                dict(
                    feature=feature,
                    outcome=outcome,
                    n_participants=9,
                    n_estimable=len(values),
                    n_non_estimable=9 - len(values),
                    n_positive=int(group.direction.eq("positive").sum()),
                    n_negative=int(group.direction.eq("negative").sum()),
                    n_zero=int(group.direction.eq("zero").sum()),
                    median=float(np.median(values)) if len(values) else np.nan,
                    q1=float(np.quantile(values, 0.25, method="linear")) if len(values) else np.nan,
                    q3=float(np.quantile(values, 0.75, method="linear")) if len(values) else np.nan,
                    minimum=float(np.min(values)) if len(values) else np.nan,
                    maximum=float(np.max(values)) if len(values) else np.nan,
                    range=float(np.ptp(values)) if len(values) else np.nan,
                    estimable_participants=";".join(
                        group.loc[group.estimation_status.eq("estimable"), "participant_id"]
                    ),
                )
            )
    write_csv(output / "cross_participant_summary.csv", pd.DataFrame(cross))
    write_json(
        output / "analysis_status.json",
        dict(
            status="complete",
            n_associations=len(individual),
            n_estimable=int(individual.estimation_status.eq("estimable").sum()),
        ),
    )

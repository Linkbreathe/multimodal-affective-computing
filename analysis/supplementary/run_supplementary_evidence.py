from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path("src").resolve()))

from mac.fusion.minimal_fusion import (  # noqa: E402
    HIGH_DISCOMFORT_PREDICTION_THRESHOLD,
    HIGH_DISCOMFORT_TRUTH_THRESHOLD,
    MODALITY_ORDER,
    MODALITY_PREFIXES,
    TARGETS,
    _condition_baseline,
    _history_baseline,
    _modal_columns,
    _point_metrics,
    _rank_features,
)


ROOT = Path(".")
REPORT_DIR = ROOT / "artifacts" / "reports"
TABLE_DIR = REPORT_DIR / "supplementary"
LEVELS = ["Low", "Medium", "High"]
LEVEL_MAP = {
    "C1": ("Low", "Low"),
    "C2": ("Low", "Medium"),
    "C3": ("Low", "High"),
    "C4": ("Medium", "Low"),
    "C5": ("Medium", "Medium"),
    "C6": ("Medium", "High"),
    "C7": ("High", "Low"),
    "C8": ("High", "Medium"),
    "C9": ("High", "High"),
}
EEG_DISABLED = {"P002", "P005", "P006", "P010", "P014", "P016"}
CLASSICAL_COMBINATIONS = ("P", "H", "E", "V", "PHEV")
FEATURE_COUNTS = (10, 20)
RANDOM_SEED = 42


@dataclass(frozen=True)
class DvSpec:
    name: str
    label_zh: str
    columns: tuple[str, ...]
    n_expected: int
    subset: str
    note: str = ""


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def p_fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "NA"
    if number < 0.001:
        return "<0.001"
    return f"{number:.4f}"


def df_to_md(frame: pd.DataFrame, digits: int = 4) -> str:
    if frame.empty:
        return "_无可报告行。_"
    working = frame.copy()
    for column in working.columns:
        if pd.api.types.is_numeric_dtype(working[column]):
            working[column] = working[column].map(lambda value: fmt(value, digits))
        else:
            working[column] = working[column].fillna("").astype(str)
    headers = [str(column) for column in working.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in working.iterrows():
        values = [str(row[column]).replace("\n", " ") for column in working.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def find_cols(frame: pd.DataFrame, pattern: str) -> list[str]:
    return [name for name in frame.columns if re.search(pattern, name, re.IGNORECASE)]


def load_condition_frame() -> pd.DataFrame:
    labels = pd.read_csv(ROOT / "artifacts" / "preprocessed" / "condition_labels.csv")
    feats = pd.read_csv(ROOT / "artifacts" / "features" / "condition_features.csv")
    labels["participant_id"] = labels["participant_id"].astype(str).str.upper().str.strip()
    feats["participant_id"] = feats["participant_id"].astype(str).str.upper().str.strip()
    duplicate_feature_columns = [
        name for name in labels.columns if name not in {"participant_id", "condition"} and name in feats.columns
    ]
    merged = labels.merge(
        feats.drop(columns=duplicate_feature_columns),
        on=["participant_id", "condition"],
        how="inner",
        validate="one_to_one",
    )
    merged = merged[merged["participant_id"].ne("P001")].copy()
    merged["intensity_level"] = pd.Categorical(
        merged["condition"].map(lambda value: LEVEL_MAP[str(value)][0]),
        categories=LEVELS,
        ordered=True,
    )
    merged["frequency_level"] = pd.Categorical(
        merged["condition"].map(lambda value: LEVEL_MAP[str(value)][1]),
        categories=LEVELS,
        ordered=True,
    )
    merged["intensity_value"] = pd.to_numeric(merged["intensity"], errors="coerce")
    merged["frequency_value"] = pd.to_numeric(merged["frequency"], errors="coerce")
    return merged.reset_index(drop=True)


def build_dv_specs(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[DvSpec], dict[str, list[str]]]:
    alpha_cols = find_cols(frame, r"eeg_.*alpha.*power.*__median")
    beta_cols = find_cols(frame, r"eeg_.*beta.*power.*__median")
    required_patterns = {
        "ecg_hr_bpm__mean": r"ecg_hr_bpm__mean",
        "ecg_hrv_60s_rmssd_ms__mean": r"ecg_hrv_60s_rmssd_ms__mean",
        "ecg_hrv_30s_rmssd_ms__mean": r"ecg_hrv_30s_rmssd_ms__mean",
        "ecg_hrv_120s_rmssd_ms__mean": r"ecg_hrv_120s_rmssd_ms__mean",
        "ecg_hrv_300s_rmssd_ms__mean": r"ecg_hrv_300s_rmssd_ms__mean",
        "eye_fixation_fraction_ivt__mean": r"eye_fixation_fraction.*__mean",
        "eye_gaze_on_painting_fraction__mean": r"eye.*gaze_on_painting.*__mean",
        "head_speed_mean__mean": r"head_speed_mean__mean",
        "head_stationary_fraction__mean": r"head_stationary_fraction__mean",
    }
    matched: dict[str, list[str]] = {
        "median_alpha_power_channel_mean": alpha_cols,
        "median_beta_power_channel_mean": beta_cols,
    }
    for key, pattern in required_patterns.items():
        cols = find_cols(frame, pattern)
        if cols:
            matched[key] = cols

    working = frame.copy()
    if alpha_cols:
        working["median_alpha_power_channel_mean"] = working[alpha_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    if beta_cols:
        working["median_beta_power_channel_mean"] = working[beta_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    specs: list[DvSpec] = [
        DvSpec("relaxation", "relaxation", ("relaxation",), 15, "all"),
        DvSpec("discomfort", "discomfort", ("discomfort",), 15, "all"),
        DvSpec("calm", "calm", ("calm",), 15, "all"),
        DvSpec("pleasantness", "pleasantness", ("pleasantness",), 15, "all"),
        DvSpec("monotony_raw", "monotony_raw", ("monotony_raw",), 15, "all", "问卷原始单调度评分"),
    ]
    if alpha_cols:
        specs.append(
            DvSpec(
                "median_alpha_power_channel_mean",
                "median alpha power",
                ("median_alpha_power_channel_mean",),
                9,
                "eeg_available",
                "四个 EEG 通道 __median alpha power 的行内均值；不是事后选取单通道。",
            )
        )
    if beta_cols:
        specs.append(
            DvSpec(
                "median_beta_power_channel_mean",
                "median beta power",
                ("median_beta_power_channel_mean",),
                9,
                "eeg_available",
                "四个 EEG 通道 __median beta power 的行内均值；不是事后选取单通道。",
            )
        )
    feature_specs = [
        ("heart_rate", "heart rate", "ecg_hr_bpm__mean", "all", "ECG heart rate"),
        ("hrv_60s_rmssd", "HRV RMSSD 60s", "ecg_hrv_60s_rmssd_ms__mean", "all", "主 HRV 指标"),
        ("hrv_30s_rmssd_sensitivity", "HRV RMSSD 30s", "ecg_hrv_30s_rmssd_ms__mean", "all", "HRV 敏感性分析"),
        ("hrv_120s_rmssd_sensitivity", "HRV RMSSD 120s", "ecg_hrv_120s_rmssd_ms__mean", "all", "HRV 敏感性分析"),
        ("hrv_300s_rmssd_sensitivity", "HRV RMSSD 300s", "ecg_hrv_300s_rmssd_ms__mean", "all", "HRV 敏感性分析"),
        ("eye_fixation_fraction", "eye fixation fraction", "eye_fixation_fraction_ivt__mean", "all", ""),
        ("eye_gaze_on_painting_fraction", "eye gaze on painting fraction", "eye_gaze_on_painting_fraction__mean", "all", ""),
        ("head_speed_mean", "head speed mean", "head_speed_mean__mean", "all", ""),
        ("head_stationary_fraction", "head stationary fraction", "head_stationary_fraction__mean", "all", ""),
    ]
    for name, label, column, subset, note in feature_specs:
        if column in working.columns:
            specs.append(DvSpec(name, label, (column,), 15, subset, note))
    return working, specs, matched


def dv_frame(frame: pd.DataFrame, spec: DvSpec) -> pd.DataFrame:
    if spec.subset == "eeg_available":
        output = frame[~frame["participant_id"].isin(EEG_DISABLED)].copy()
    else:
        output = frame.copy()
    column = spec.columns[0]
    output["dv_value"] = pd.to_numeric(output[column], errors="coerce")
    return output


def contrast_matrix(k: int) -> np.ndarray:
    matrix = np.zeros((k - 1, k), dtype=float)
    for row in range(k - 1):
        matrix[row, : row + 1] = 1.0 / (row + 1)
        matrix[row, row + 1] = -1.0
    return matrix


def mauchly_and_gg(work: pd.DataFrame, effect: str) -> dict[str, float]:
    columns = pd.MultiIndex.from_product([LEVELS, LEVELS], names=["intensity_level", "frequency_level"])
    pivot = work.pivot_table(
        index="participant_id",
        columns=["intensity_level", "frequency_level"],
        values="dv_value",
        observed=False,
    ).reindex(columns=columns)
    if pivot.isna().any().any() or len(pivot) < 3:
        return {"mauchly_w": np.nan, "mauchly_p": np.nan, "gg_epsilon": np.nan}
    cells = pivot.to_numpy(dtype=float).reshape(len(pivot), 3, 3)
    if effect == "intensity":
        transformed = cells.mean(axis=2) @ contrast_matrix(3).T
    elif effect == "frequency":
        transformed = cells.mean(axis=1) @ contrast_matrix(3).T
    else:
        k = np.kron(contrast_matrix(3), contrast_matrix(3))
        transformed = cells.reshape(len(pivot), 9) @ k.T
    p = transformed.shape[1]
    if p <= 1:
        return {"mauchly_w": 1.0, "mauchly_p": 1.0, "gg_epsilon": 1.0}
    covariance = np.cov(transformed, rowvar=False, ddof=1)
    covariance = np.atleast_2d(covariance)
    trace = float(np.trace(covariance))
    trace_square = float(np.trace(covariance @ covariance))
    if trace <= 0 or trace_square <= 0:
        return {"mauchly_w": np.nan, "mauchly_p": np.nan, "gg_epsilon": np.nan}
    epsilon = (trace * trace) / (p * trace_square)
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        w = 0.0
        p_value = 0.0
    else:
        w = float(math.exp(logdet - p * math.log(trace / p)))
        df = p * (p + 1) / 2 - 1
        correction = (2 * p * p + p + 2) / (6 * p * (len(pivot) - 1))
        chi_square = max(0.0, -(len(pivot) - 1) * (1 - correction) * math.log(max(w, 1e-300)))
        p_value = float(stats.chi2.sf(chi_square, df))
    return {"mauchly_w": w, "mauchly_p": p_value, "gg_epsilon": float(epsilon)}


def run_rm_anova(work: pd.DataFrame, spec: DvSpec) -> list[dict[str, Any]]:
    complete = work.dropna(subset=["dv_value"]).copy()
    cell_counts = complete.groupby(["participant_id", "condition"], observed=False)["dv_value"].size()
    balanced = (
        complete["participant_id"].nunique() == spec.n_expected
        and len(complete) == spec.n_expected * 9
        and (cell_counts == 1).all()
    )
    rows: list[dict[str, Any]] = []
    if not balanced:
        return run_unbalanced_with_mixedlm(complete, spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = AnovaRM(
            complete,
            depvar="dv_value",
            subject="participant_id",
            within=["intensity_level", "frequency_level"],
        ).fit().anova_table.reset_index(names="source_raw")
    source_map = {
        "intensity_level": "intensity",
        "frequency_level": "frequency",
        "intensity_level:frequency_level": "interaction",
    }
    for _, row in table.iterrows():
        raw_source = str(row["source_raw"])
        if raw_source not in source_map:
            continue
        effect = source_map[raw_source]
        f_raw = float(row["F Value"])
        f_value = max(0.0, f_raw)
        df1 = float(row["Num DF"])
        df2 = float(row["Den DF"])
        p_unc = 1.0 if f_raw < 0 else float(row["Pr > F"])
        sph = mauchly_and_gg(complete, effect)
        epsilon = sph["gg_epsilon"]
        p_gg = p_unc
        if np.isfinite(epsilon):
            p_gg = float(stats.f.sf(f_value, df1 * epsilon, df2 * epsilon))
        p_infer = p_gg if np.isfinite(sph["mauchly_p"]) and sph["mauchly_p"] < 0.05 else p_unc
        eta_p = float((f_value * df1) / ((f_value * df1) + df2)) if np.isfinite(f_value) else np.nan
        rows.append(
            {
                "dv": spec.name,
                "label_zh": spec.label_zh,
                "n_participants": complete["participant_id"].nunique(),
                "n_rows": len(complete),
                "method": "rm_anova",
                "source": effect,
                "df1": df1,
                "df2": df2,
                "F": f_value,
                "p_unc": p_unc,
                "mauchly_w": sph["mauchly_w"],
                "mauchly_p": sph["mauchly_p"],
                "gg_epsilon": epsilon,
                "p_gg": p_gg,
                "p_infer": p_infer,
                "eta_p2": eta_p,
            }
        )
    return rows


def run_unbalanced_with_mixedlm(complete: pd.DataFrame, spec: DvSpec) -> list[dict[str, Any]]:
    import statsmodels.formula.api as smf

    if complete.empty or complete["participant_id"].nunique() < 3:
        return []
    effect_terms = {
        "C(intensity_level)": "intensity",
        "C(frequency_level)": "frequency",
        "C(intensity_level):C(frequency_level)": "interaction",
    }
    mixed_p: dict[str, tuple[float, float]] = {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.mixedlm(
                "dv_value ~ C(intensity_level) * C(frequency_level)",
                complete,
                groups=complete["participant_id"],
            )
            result = model.fit(reml=False, method="lbfgs", maxiter=500, disp=False)
            table = result.wald_test_terms(skip_single=False).table
        for raw, effect in effect_terms.items():
            if raw not in table.index:
                continue
            statistic = float(np.asarray(table.loc[raw, "statistic"]).reshape(-1)[0])
            df_constraint = float(table.loc[raw, "df_constraint"])
            p_value = float(table.loc[raw, "pvalue"])
            mixed_p[effect] = (statistic / df_constraint if df_constraint else statistic, p_value)
    except Exception:
        mixed_p = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ols = smf.ols(
            "dv_value ~ C(participant_id) + C(intensity_level) * C(frequency_level)",
            complete,
        ).fit()
        anova = sm.stats.anova_lm(ols, typ=2)
    rows: list[dict[str, Any]] = []
    source_lookup = {
        "C(intensity_level)": "intensity",
        "C(frequency_level)": "frequency",
        "C(intensity_level):C(frequency_level)": "interaction",
    }
    residual_ss = float(anova.loc["Residual", "sum_sq"]) if "Residual" in anova.index else np.nan
    for raw, effect in source_lookup.items():
        if raw not in anova.index:
            continue
        f_value = float(anova.loc[raw, "F"])
        p_ols = float(anova.loc[raw, "PR(>F)"])
        df1 = float(anova.loc[raw, "df"])
        df2 = float(anova.loc["Residual", "df"]) if "Residual" in anova.index else np.nan
        ss_effect = float(anova.loc[raw, "sum_sq"])
        eta_p = ss_effect / (ss_effect + residual_ss) if np.isfinite(residual_ss) and residual_ss >= 0 else np.nan
        mixed_stat, mixed_pvalue = mixed_p.get(effect, (np.nan, np.nan))
        p_unc = mixed_pvalue if np.isfinite(mixed_pvalue) else p_ols
        rows.append(
            {
                "dv": spec.name,
                "label_zh": spec.label_zh,
                "n_participants": complete["participant_id"].nunique(),
                "n_rows": len(complete),
                "method": "mixedlm_wald_unbalanced",
                "source": effect,
                "df1": df1,
                "df2": df2,
                "F": mixed_stat if np.isfinite(mixed_stat) else f_value,
                "p_unc": p_unc,
                "mauchly_w": np.nan,
                "mauchly_p": np.nan,
                "gg_epsilon": np.nan,
                "p_gg": p_unc,
                "p_infer": p_unc,
                "eta_p2": eta_p,
                "ols_fixed_effect_F": f_value,
                "ols_fixed_effect_p": p_ols,
            }
        )
    return rows


def run_linear_trend(work: pd.DataFrame, spec: DvSpec) -> list[dict[str, Any]]:
    import statsmodels.formula.api as smf

    complete = work.dropna(subset=["dv_value", "intensity_value", "frequency_value"]).copy()
    rows: list[dict[str, Any]] = []
    if len(complete) < 10 or complete["participant_id"].nunique() < 3:
        return rows
    terms = ["intensity_value", "frequency_value", "intensity_value:frequency_value"]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.mixedlm(
                "dv_value ~ intensity_value * frequency_value",
                complete,
                groups=complete["participant_id"],
            )
            result = model.fit(reml=False, method="lbfgs", maxiter=500, disp=False)
        params, bse, pvalues = result.params, result.bse, result.pvalues
        method = "mixedlm_random_intercept"
    except Exception:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = smf.ols("dv_value ~ intensity_value * frequency_value", complete).fit(
                cov_type="cluster", cov_kwds={"groups": complete["participant_id"]}
            )
        params, bse, pvalues = result.params, result.bse, result.pvalues
        method = "ols_cluster_fallback"
    for term in terms:
        rows.append(
            {
                "dv": spec.name,
                "label_zh": spec.label_zh,
                "n_participants": complete["participant_id"].nunique(),
                "n_rows": len(complete),
                "method": method,
                "term": term,
                "coef": float(params.get(term, np.nan)),
                "se": float(bse.get(term, np.nan)),
                "p_value": float(pvalues.get(term, np.nan)),
            }
        )
    return rows


def holm(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    return multipletests(p_values, method="holm")[1].tolist()


def posthoc_pairs(work: pd.DataFrame, spec: DvSpec, significant_effects: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pairs = [("Low", "Medium"), ("Medium", "High"), ("Low", "High")]
    for effect in sorted(significant_effects):
        if effect not in {"intensity", "frequency"}:
            continue
        group_cols = ["participant_id", f"{effect}_level"]
        summary = work.dropna(subset=["dv_value"]).groupby(group_cols, observed=False)["dv_value"].mean().reset_index()
        pivot = summary.pivot(index="participant_id", columns=f"{effect}_level", values="dv_value").reindex(columns=LEVELS)
        local_rows: list[dict[str, Any]] = []
        for left, right in pairs:
            pair_frame = pivot[[left, right]].dropna()
            if len(pair_frame) < 2:
                t_value = np.nan
                p_value = np.nan
                mean_diff = np.nan
            else:
                diff = pair_frame[right] - pair_frame[left]
                t_value, p_value = stats.ttest_rel(pair_frame[right], pair_frame[left])
                mean_diff = float(diff.mean())
            local_rows.append(
                {
                    "dv": spec.name,
                    "effect": effect,
                    "contrast": f"{left} vs {right}",
                    "n_participants": int(len(pair_frame)),
                    "mean_diff_right_minus_left": mean_diff,
                    "t": float(t_value) if np.isfinite(t_value) else np.nan,
                    "p_unc": float(p_value) if np.isfinite(p_value) else np.nan,
                }
            )
        corrected = holm([row["p_unc"] for row in local_rows])
        for row, p_holm in zip(local_rows, corrected, strict=True):
            row["p_holm"] = p_holm
            rows.append(row)
    return rows


def run_dose_response() -> dict[str, pd.DataFrame]:
    frame = load_condition_frame()
    frame, specs, matched = build_dv_specs(frame)
    all_anova: list[dict[str, Any]] = []
    all_trends: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    desc_rows: list[dict[str, Any]] = []
    for spec in specs:
        work = dv_frame(frame, spec)
        complete = work.dropna(subset=["dv_value"])
        missing = len(work) - len(complete)
        missing_frame = work.loc[work["dv_value"].isna(), ["participant_id", "condition"]].astype(str)
        missing_cells = [f"{row.participant_id}/{row.condition}" for row in missing_frame.itertuples(index=False)]
        sample_rows.append(
            {
                "dependent_variable": spec.name,
                "label_zh": spec.label_zh,
                "subset": spec.subset,
                "n_participants_expected": spec.n_expected,
                "n_participants_observed": complete["participant_id"].nunique(),
                "n_rows_expected": len(work),
                "n_rows_observed": len(complete),
                "missing_cells": missing,
                "missing_rate": missing / len(work) if len(work) else np.nan,
                "missing_cell_ids": "; ".join(missing_cells[:20]),
                "note": spec.note,
            }
        )
        all_anova.extend(run_rm_anova(work, spec))
        all_trends.extend(run_linear_trend(work, spec))
        grouped = (
            complete.groupby(["intensity_level", "frequency_level"], observed=False)["dv_value"]
            .agg(["count", "mean", "std", "median"])
            .reset_index()
        )
        for _, row in grouped.iterrows():
            desc_rows.append(
                {
                    "dependent_variable": spec.name,
                    "intensity_level": row["intensity_level"],
                    "frequency_level": row["frequency_level"],
                    "n": int(row["count"]),
                    "mean": float(row["mean"]),
                    "sd": float(row["std"]) if np.isfinite(row["std"]) else np.nan,
                    "median": float(row["median"]),
                }
            )
    anova = pd.DataFrame(all_anova)
    if not anova.empty:
        anova["p_fdr"] = multipletests(anova["p_infer"].fillna(1.0), method="fdr_bh")[1]
    trends = pd.DataFrame(all_trends)
    sample = pd.DataFrame(sample_rows)
    desc = pd.DataFrame(desc_rows)
    significant = {
        (row["dv"], row["source"])
        for _, row in anova.iterrows()
        if row["source"] in {"intensity", "frequency"} and np.isfinite(row["p_fdr"]) and row["p_fdr"] <= 0.05
    }
    posthoc_rows: list[dict[str, Any]] = []
    for spec in specs:
        effects = {effect for dv, effect in significant if dv == spec.name}
        if effects:
            posthoc_rows.extend(posthoc_pairs(dv_frame(frame, spec), spec, effects))
    posthoc = pd.DataFrame(posthoc_rows)

    sample.to_csv(TABLE_DIR / "dose_response_sample_summary.csv", index=False)
    anova.to_csv(TABLE_DIR / "dose_response_anova_results.csv", index=False)
    trends.to_csv(TABLE_DIR / "dose_response_linear_trends.csv", index=False)
    posthoc.to_csv(TABLE_DIR / "dose_response_posthoc_pairs.csv", index=False)
    desc.to_csv(TABLE_DIR / "dose_response_descriptives.csv", index=False)
    pd.DataFrame([{"feature": key, "matched_columns": "; ".join(value)} for key, value in matched.items()]).to_csv(
        TABLE_DIR / "dose_response_feature_matches.csv", index=False
    )
    write_dose_report(sample, anova, trends, posthoc, desc, matched)
    return {"sample": sample, "anova": anova, "trends": trends, "posthoc": posthoc, "descriptives": desc}


def effect_summary_table(anova: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dv, group in anova.groupby("dv", sort=False):
        row: dict[str, Any] = {"dependent_variable": dv, "N": int(group["n_participants"].max())}
        for effect in ("intensity", "frequency", "interaction"):
            match = group[group["source"].eq(effect)]
            if match.empty:
                for suffix in ("F", "p_unc", "p_FDR", "eta_p2"):
                    row[f"{effect}_{suffix}"] = np.nan
                continue
            source = match.iloc[0]
            row[f"{effect}_F"] = source["F"]
            row[f"{effect}_p_unc"] = source["p_unc"]
            row[f"{effect}_p_FDR"] = source["p_fdr"]
            row[f"{effect}_eta_p2"] = source["eta_p2"]
        rows.append(row)
    return pd.DataFrame(rows)


def write_dose_report(
    sample: pd.DataFrame,
    anova: pd.DataFrame,
    trends: pd.DataFrame,
    posthoc: pd.DataFrame,
    desc: pd.DataFrame,
    matched: dict[str, list[str]],
) -> None:
    main = effect_summary_table(anova)
    main.to_csv(TABLE_DIR / "dose_response_main_effects_wide.csv", index=False)
    desc_lines: list[str] = []
    for dv, group in desc.groupby("dependent_variable", sort=False):
        table = group.copy()
        table["mean_sd"] = table.apply(lambda row: f"{fmt(row['mean'])} ({fmt(row['sd'])}); N={int(row['n'])}", axis=1)
        wide = table.pivot(index="intensity_level", columns="frequency_level", values="mean_sd").reindex(index=LEVELS, columns=LEVELS)
        wide = wide.reset_index().rename(columns={"intensity_level": "intensity \\ frequency"})
        desc_lines.extend([f"### {dv}", "", df_to_md(wide, digits=4), ""])
    feature_rows = pd.DataFrame(
        [{"derived_or_target": key, "matched_columns": "; ".join(value)} for key, value in matched.items()]
    )
    lines = [
        "# 剂量-反应补充统计报告",
        "",
        "生成日期：2026-07-01。",
        "",
        "本报告使用 `artifacts/preprocessed/condition_labels.csv` 与 `artifacts/features/condition_features.csv`，监督单元为 15 名参与者 x 9 个 Condition。EEG 指标仅使用 P003、P004、P007、P008、P009、P011、P012、P013、P015，共 N=9。10 秒窗口只作为特征切片，不作为独立统计样本。",
        "",
        "## 样本与缺失",
        "",
        df_to_md(sample[[
            "dependent_variable",
            "subset",
            "n_participants_observed",
            "n_rows_observed",
            "missing_cells",
            "missing_rate",
            "note",
        ]]),
        "",
        "## 特征匹配",
        "",
        df_to_md(feature_rows),
        "",
        "Delta-Beta/CFC/coupling 相关字段未在当前 `condition_features.csv` 中找到，因此本报告没有把 5.1.4 作为已实现生理特征处理。",
        "",
        "## 3 x 3 重复测量 ANOVA",
        "",
        "下表的 FDR 采用 Benjamini-Hochberg，校正家族为所有因变量 x intensity、frequency、interaction 三类效应。若 Mauchly 检验提示球形性偏离，则 `p_infer` 使用 Greenhouse-Geisser 校正；宽表中仍同时保留未校正 p 值和 FDR 后 p 值。",
        "",
        df_to_md(main),
        "",
        "详细 ANOVA、Mauchly 与 Greenhouse-Geisser 结果见 `artifacts/reports/supplementary/dose_response_anova_results.csv`。",
        "",
        "## 线性趋势混合模型",
        "",
        "模型形式为 `DV ~ intensity_value * frequency_value + (1 | participant_id)`；在随机截距模型无法稳定拟合时，使用参与者聚类稳健标准误的 OLS 作为回退，并在 `method` 列标出。",
        "",
        df_to_md(trends[["dv", "n_participants", "n_rows", "method", "term", "coef", "se", "p_value"]]),
        "",
        "## 显著主效应的事后比较",
        "",
        df_to_md(posthoc),
        "",
        "## 3 x 3 描述统计",
        "",
        *desc_lines,
        "## 解释约束",
        "",
        "上述结果只能表述为当前 15 名参与者样本中的观察性组间差异或线性关联。它们不构成生理机制、心理机制、个体诊断、临床可用性或自适应控制有效性的证据，也不改变当前 classical / Shadow-only / hold 的部署结论。",
    ]
    write_text(REPORT_DIR / "dose_response_report_zh.md", "\n".join(lines))


def participant_two_target_mae(oof: pd.DataFrame, combination: str, prediction_column: str) -> pd.Series:
    subset = oof[oof["combination"].eq(combination)].copy()
    subset["error"] = (subset["truth"] - subset[prediction_column]).abs()
    return subset.groupby("participant_id")["error"].mean()


def participant_discomfort_recall(oof: pd.DataFrame, combination: str, prediction_column: str) -> pd.Series:
    subset = oof[(oof["combination"].eq(combination)) & (oof["target"].eq("discomfort"))].copy()
    subset["truth_high"] = subset["truth"] >= HIGH_DISCOMFORT_TRUTH_THRESHOLD
    subset["pred_high"] = subset[prediction_column] >= HIGH_DISCOMFORT_PREDICTION_THRESHOLD
    values: dict[str, float] = {}
    for participant, group in subset.groupby("participant_id"):
        high = group[group["truth_high"]]
        if high.empty:
            values[participant] = np.nan
        else:
            values[participant] = float(high["pred_high"].mean())
    return pd.Series(values, name=combination)


def wilcoxon_comparison(name: str, a: pd.Series, b: pd.Series, higher_is_better: bool) -> dict[str, Any]:
    paired = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if higher_is_better:
        improvement = paired["a"] - paired["b"]
    else:
        improvement = paired["b"] - paired["a"]
    nonzero = improvement[np.abs(improvement) > 1e-12]
    if len(nonzero) == 0:
        statistic = 0.0
        p_value = 1.0
        effect = 0.0
    else:
        test = stats.wilcoxon(improvement, zero_method="wilcox", alternative="two-sided", mode="auto")
        statistic = float(test.statistic)
        p_value = float(test.pvalue)
        ranks = stats.rankdata(np.abs(nonzero))
        positive = float(ranks[nonzero.to_numpy() > 0].sum())
        negative = float(ranks[nonzero.to_numpy() < 0].sum())
        effect = (positive - negative) / float(ranks.sum())
    median = float(np.median(improvement)) if len(improvement) else np.nan
    return {
        "comparison": name,
        "N": int(len(paired)),
        "median_difference": median,
        "W": statistic,
        "p_unc": p_value,
        "rank_biserial_effect": effect,
        "direction_note": "positive means model A is better",
    }


def run_model_significance() -> pd.DataFrame:
    ridge = pd.read_csv(ROOT / "artifacts" / "fusion_minimal" / "oof_predictions.csv")
    dcnn = pd.read_csv(ROOT / "artifacts" / "fusion_minimal_dcnn" / "oof_predictions.csv")
    classical = pd.read_csv(
        ROOT
        / "artifacts"
        / "decision_reanalysis"
        / "phase_a"
        / "isolated_condition_lopo"
        / "reports"
        / "condition_level_lopo_predictions.csv"
    )
    single_scores = {
        combo: float(participant_two_target_mae(ridge, combo, "prediction").mean())
        for combo in ("P", "H", "E", "V")
    }
    best_single = min(single_scores, key=single_scores.get)
    rows = [
        wilcoxon_comparison(
            f"Ridge PHEV vs best single modality Ridge {best_single} (two-target participant MAE)",
            participant_two_target_mae(ridge, "PHEV", "prediction"),
            participant_two_target_mae(ridge, best_single, "prediction"),
            higher_is_better=False,
        ),
        wilcoxon_comparison(
            "1DCNN PHEV vs Ridge PHEV (two-target participant MAE)",
            participant_two_target_mae(dcnn, "PHEV", "prediction"),
            participant_two_target_mae(ridge, "PHEV", "prediction"),
            higher_is_better=False,
        ),
        wilcoxon_comparison(
            "Ridge PH vs Condition-only baseline (participant high-discomfort recall)",
            participant_discomfort_recall(ridge, "PH", "prediction"),
            participant_discomfort_recall(ridge, "PH", "condition_only_prediction"),
            higher_is_better=True,
        ),
        wilcoxon_comparison(
            "Classical Condition residual ensemble vs history baseline (participant discomfort MAE)",
            classical.set_index("participant_id").groupby(level=0).apply(
                lambda group: mean_absolute_error(group["discomfort"], group["pred_discomfort"])
            ),
            classical.set_index("participant_id").groupby(level=0).apply(
                lambda group: mean_absolute_error(group["discomfort"], group["history_discomfort"])
            ),
            higher_is_better=False,
        ),
    ]
    result = pd.DataFrame(rows)
    result["p_holm"] = multipletests(result["p_unc"].fillna(1.0), method="holm")[1]
    conclusions = []
    for _, row in result.iterrows():
        if row["p_holm"] <= 0.05 and row["median_difference"] > 0:
            conclusions.append("model A significantly better")
        elif row["p_holm"] <= 0.05 and row["median_difference"] < 0:
            conclusions.append("model B significantly better")
        else:
            conclusions.append("no significant difference")
    result["conclusion"] = conclusions
    result["selection_note"] = ""
    result.loc[0, "selection_note"] = f"Best single modality selected by two-target MAE: {best_single}; scores={single_scores}"
    result.loc[2, "selection_note"] = "Participant-level recall is defined only for participants with at least one high-discomfort label; N can be below 15."
    result.to_csv(TABLE_DIR / "model_significance_tests.csv", index=False)
    lines = [
        "# 模型与模态显著性检验",
        "",
        "Wilcoxon signed-rank 检验基于参与者级配对指标。MAE 比较中 `median_difference` 为 B-A，因此正值表示 A 的误差更低；recall 比较中为 A-B，因此正值表示 A 的召回更高。四个预设比较共同使用 Holm 校正。",
        "",
        df_to_md(result),
        "",
        "这些检验只支持离线模型比较，不改变现有 classical / Shadow-only / hold 部署结论。",
    ]
    write_text(REPORT_DIR / "model_significance_report_zh.md", "\n".join(lines))
    return result


def selected_columns(train: pd.DataFrame, target: str, combination: str, feature_count: int) -> list[str]:
    baseline, _, _ = _condition_baseline(train, train, target)
    residual = train[target].to_numpy(dtype=float) - baseline
    selected: list[str] = []
    for modality in MODALITY_ORDER:
        if modality not in combination:
            continue
        selected.extend(_rank_features(train, _modal_columns(train, modality), residual, feature_count))
    return sorted(selected)


def matrix(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")


def make_model(model_name: str, params: dict[str, Any], seed: int) -> Pipeline:
    if model_name == "ElasticNet":
        estimator = ElasticNet(
            alpha=float(params["alpha"]),
            l1_ratio=float(params["l1_ratio"]),
            max_iter=20000,
            random_state=seed,
        )
    elif model_name == "RandomForest":
        estimator = RandomForestRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=params["max_depth"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            random_state=seed,
            n_jobs=1,
        )
    else:
        raise ValueError(model_name)
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", estimator),
        ]
    )


def candidate_grid(model_name: str) -> list[dict[str, Any]]:
    if model_name == "ElasticNet":
        return [
            {"alpha": alpha, "l1_ratio": l1_ratio}
            for alpha in (0.01, 0.1)
            for l1_ratio in (0.2, 0.8)
        ]
    if model_name == "RandomForest":
        return [
            {"n_estimators": 20, "max_depth": max_depth, "min_samples_leaf": 3}
            for max_depth in (3, None)
        ]
    raise ValueError(model_name)


def inner_select_model(
    train: pd.DataFrame,
    target: str,
    combination: str,
    model_name: str,
    seed: int,
) -> dict[str, Any]:
    groups = train["participant_id"].astype(str).to_numpy()
    n_splits = min(2, len(np.unique(groups)))
    cv = GroupKFold(n_splits=max(2, n_splits))
    trials: list[dict[str, Any]] = []
    for feature_count in FEATURE_COUNTS:
        for params in candidate_grid(model_name):
            fold_mae: list[float] = []
            for fold, (inner_train_index, inner_test_index) in enumerate(cv.split(train, groups=groups), start=1):
                inner_train = train.iloc[inner_train_index].reset_index(drop=True)
                inner_test = train.iloc[inner_test_index].reset_index(drop=True)
                columns = selected_columns(inner_train, target, combination, feature_count)
                baseline_train, baseline_map, fallback = _condition_baseline(inner_train, inner_train, target)
                residual = inner_train[target].to_numpy(dtype=float) - baseline_train
                baseline_test = np.asarray(
                    [float(baseline_map.get(condition, fallback)) for condition in inner_test["condition"]],
                    dtype=float,
                )
                try:
                    model = make_model(model_name, params, seed + fold)
                    model.fit(matrix(inner_train, columns), residual)
                    prediction = np.clip(baseline_test + model.predict(matrix(inner_test, columns)), 0.0, 1.0)
                    fold_mae.append(float(mean_absolute_error(inner_test[target], prediction)))
                except Exception:
                    fold_mae.append(float("inf"))
            trials.append(
                {
                    "feature_count": feature_count,
                    "params": params,
                    "inner_mae": float(np.mean(fold_mae)),
                }
            )
    return min(trials, key=lambda row: (row["inner_mae"], row["feature_count"], json.dumps(row["params"], sort_keys=True)))


def evaluate_classical_model(frame: pd.DataFrame, model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = frame.copy().reset_index(drop=True)
    participants = sorted(frame["participant_id"].unique())
    predictions = {
        combination: {target: np.full(len(frame), np.nan, dtype=float) for target in TARGETS}
        for combination in CLASSICAL_COMBINATIONS
    }
    condition_only = {target: np.full(len(frame), np.nan, dtype=float) for target in TARGETS}
    history = {target: np.full(len(frame), np.nan, dtype=float) for target in TARGETS}
    selection_rows: list[dict[str, Any]] = []
    for fold, participant in enumerate(participants, start=1):
        test_mask = frame["participant_id"].eq(participant).to_numpy()
        test_indexes = np.flatnonzero(test_mask)
        train = frame.loc[~test_mask].reset_index(drop=True)
        test = frame.loc[test_mask].reset_index(drop=True)
        for target in TARGETS:
            baseline, baseline_map, fallback = _condition_baseline(train, test, target)
            condition_only[target][test_indexes] = baseline
            history[target][test_indexes] = _history_baseline(test, fallback, target)
            train_baseline = np.asarray(
                [float(baseline_map.get(condition, fallback)) for condition in train["condition"]],
                dtype=float,
            )
            residual = train[target].to_numpy(dtype=float) - train_baseline
            for combination in CLASSICAL_COMBINATIONS:
                selected = inner_select_model(train, target, combination, model_name, RANDOM_SEED + fold)
                columns = selected_columns(train, target, combination, int(selected["feature_count"]))
                model = make_model(model_name, selected["params"], RANDOM_SEED + fold)
                model.fit(matrix(train, columns), residual)
                predictions[combination][target][test_indexes] = np.clip(
                    baseline + model.predict(matrix(test, columns)), 0.0, 1.0
                )
                selection_rows.append(
                    {
                        "model": model_name,
                        "fold": fold,
                        "held_out_participant": participant,
                        "target": target,
                        "combination": combination,
                        "feature_count_per_modality": int(selected["feature_count"]),
                        "n_features_selected": len(columns),
                        "inner_mae": selected["inner_mae"],
                        "params": json.dumps(selected["params"], sort_keys=True),
                    }
                )
        print(f"{model_name} LOPO fold {fold}/{len(participants)}: {participant}", flush=True)
    oof_rows: list[dict[str, Any]] = []
    for combination in CLASSICAL_COMBINATIONS:
        for row_index, source in frame.iterrows():
            for target in TARGETS:
                truth = float(source[target])
                prediction = float(predictions[combination][target][row_index])
                condition_prediction = float(condition_only[target][row_index])
                history_prediction = float(history[target][row_index])
                is_discomfort = target == "discomfort"
                oof_rows.append(
                    {
                        "model": model_name,
                        "combination": combination,
                        "participant_id": source["participant_id"],
                        "condition": source["condition"],
                        "presentation_position": int(source["presentation_position"]),
                        "target": target,
                        "truth": truth,
                        "prediction": prediction,
                        "absolute_error": abs(truth - prediction),
                        "condition_only_prediction": condition_prediction,
                        "condition_only_absolute_error": abs(truth - condition_prediction),
                        "history_prediction": history_prediction,
                        "history_absolute_error": abs(truth - history_prediction),
                        "high_discomfort_truth": int(truth >= HIGH_DISCOMFORT_TRUTH_THRESHOLD) if is_discomfort else np.nan,
                        "high_discomfort_prediction": int(prediction >= HIGH_DISCOMFORT_PREDICTION_THRESHOLD) if is_discomfort else np.nan,
                    }
                )
    return pd.DataFrame(oof_rows), pd.DataFrame(selection_rows)


def summarize_oof(oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["model", "combination"] if "model" in oof.columns else ["combination"]
    for keys, group in oof.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys, strict=True))
        for target in TARGETS:
            sub = group[group["target"].eq(target)]
            truth = sub["truth"].to_numpy(dtype=float)
            prediction = sub["prediction"].to_numpy(dtype=float)
            point = _point_metrics(truth, prediction, discomfort=target == "discomfort")
            row[f"{target}_mae"] = point["mae"]
            row[f"{target}_spearman"] = point["spearman"]
            row[f"{target}_rmse"] = float(np.sqrt(mean_squared_error(truth, prediction)))
            row[f"{target}_r2"] = float(r2_score(truth, prediction))
            if target == "discomfort":
                row["discomfort_high_recall"] = point["high_recall"]
                row["discomfort_high_precision"] = point["high_precision"]
                row["discomfort_high_false_negatives"] = point["high_false_negatives"]
        rows.append(row)
    return pd.DataFrame(rows)


def run_additional_classical() -> dict[str, pd.DataFrame]:
    source = ROOT / "artifacts" / "features" / "video_ml" / "condition_features.csv"
    frame = pd.read_csv(source)
    frame["participant_id"] = frame["participant_id"].astype(str).str.upper().str.strip()
    frame = frame[frame["participant_id"].ne("P001")].copy().reset_index(drop=True)
    all_oof: list[pd.DataFrame] = []
    all_selection: list[pd.DataFrame] = []
    for model_name in ("ElasticNet", "RandomForest"):
        oof, selection = evaluate_classical_model(frame, model_name)
        all_oof.append(oof)
        all_selection.append(selection)
    oof = pd.concat(all_oof, ignore_index=True)
    selection = pd.concat(all_selection, ignore_index=True)
    metrics = summarize_oof(oof)
    oof.to_csv(TABLE_DIR / "additional_classical_oof_predictions.csv", index=False)
    selection.to_csv(TABLE_DIR / "additional_classical_selection.csv", index=False)
    metrics.to_csv(TABLE_DIR / "additional_classical_metrics.csv", index=False)
    write_classical_report(metrics, selection)
    return {"oof": oof, "selection": selection, "metrics": metrics}


def report_metric_table(metrics: pd.DataFrame, model: str) -> pd.DataFrame:
    cols = [
        "combination",
        "relaxation_mae",
        "relaxation_spearman",
        "relaxation_rmse",
        "relaxation_r2",
        "discomfort_mae",
        "discomfort_spearman",
        "discomfort_rmse",
        "discomfort_r2",
        "discomfort_high_recall",
        "discomfort_high_precision",
        "discomfort_high_false_negatives",
    ]
    return metrics[metrics["model"].eq(model)][cols].reset_index(drop=True)


def write_classical_report(metrics: pd.DataFrame, selection: pd.DataFrame) -> None:
    lines = [
        "# 额外经典模型补充结果",
        "",
        "输入边界与最小融合 Ridge 一致：`artifacts/features/video_ml/condition_features.csv`，P/H/E/V 定义沿用现有报告。每个外层 LOPO 折内用 GroupKFold 选择每模态特征数与超参数，随后在外层训练参与者上重训并预测留出参与者。",
        "",
        "## ElasticNet",
        "",
        df_to_md(report_metric_table(metrics, "ElasticNet")),
        "",
        "## RandomForestRegressor",
        "",
        df_to_md(report_metric_table(metrics, "RandomForest")),
        "",
        "## 折内选择摘要",
        "",
        df_to_md(
            selection.groupby(["model", "combination", "target"], as_index=False).agg(
                n_folds=("fold", "count"),
                median_feature_count_per_modality=("feature_count_per_modality", "median"),
                median_selected_features=("n_features_selected", "median"),
                median_inner_mae=("inner_mae", "median"),
            )
        ),
        "",
        "这些模型是研究补充比较，不改变当前 classical / Shadow-only / hold 部署结论。",
    ]
    write_text(REPORT_DIR / "additional_classical_models_report_zh.md", "\n".join(lines))


def existing_oof_metrics(additional_oof: pd.DataFrame | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, path in [
        ("Ridge minimal fusion", ROOT / "artifacts" / "fusion_minimal" / "oof_predictions.csv"),
        ("1DCNN minimal fusion", ROOT / "artifacts" / "fusion_minimal_dcnn" / "oof_predictions.csv"),
    ]:
        oof = pd.read_csv(path)
        summary = summarize_oof(oof)
        summary.insert(0, "source_model", model_name)
        rows.extend(summary.to_dict("records"))
        for baseline_name, prediction_column in [
            ("Condition-only baseline", "condition_only_prediction"),
            ("history baseline", "history_prediction"),
        ]:
            baseline_oof = oof.copy()
            baseline_oof["prediction"] = baseline_oof[prediction_column]
            summary = summarize_oof(baseline_oof)
            summary.insert(0, "source_model", f"{model_name} {baseline_name}")
            rows.extend(summary.to_dict("records"))
    classical = pd.read_csv(
        ROOT
        / "artifacts"
        / "decision_reanalysis"
        / "phase_a"
        / "isolated_condition_lopo"
        / "reports"
        / "condition_level_lopo_predictions.csv"
    )
    for source_model, rel_col, dis_col in [
        ("Classical Condition residual ensemble", "pred_relaxation", "pred_discomfort"),
        ("Classical Condition-only baseline", "condition_only_relaxation", "condition_only_discomfort"),
        ("Classical history baseline", "history_relaxation", "history_discomfort"),
    ]:
        row: dict[str, Any] = {
            "source_model": source_model,
            "combination": "condition_residual" if "residual" in source_model else "baseline",
        }
        for target, pred_col in [("relaxation", rel_col), ("discomfort", dis_col)]:
            truth = classical[target].to_numpy(dtype=float)
            pred = classical[pred_col].to_numpy(dtype=float)
            point = _point_metrics(truth, pred, discomfort=target == "discomfort")
            row[f"{target}_mae"] = point["mae"]
            row[f"{target}_spearman"] = point["spearman"]
            row[f"{target}_rmse"] = float(np.sqrt(mean_squared_error(truth, pred)))
            row[f"{target}_r2"] = float(r2_score(truth, pred))
            if target == "discomfort":
                row["discomfort_high_recall"] = point["high_recall"]
                row["discomfort_high_precision"] = point["high_precision"]
                row["discomfort_high_false_negatives"] = point["high_false_negatives"]
        rows.append(row)
    if additional_oof is not None and not additional_oof.empty:
        summary = summarize_oof(additional_oof)
        summary.insert(0, "source_model", "Additional classical")
        rows.extend(summary.to_dict("records"))
    output = pd.DataFrame(rows)
    output.to_csv(TABLE_DIR / "regression_rmse_r2_metrics.csv", index=False)
    return output


def latency_metrics() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for decisions_path in sorted((ROOT / "artifacts" / "realtime" / "adaptive_control").glob("*/decisions.jsonl")):
        session_id = decisions_path.parent.name
        latencies: list[float] = []
        with decisions_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                window_end = record.get("window_end_ms") or record.get("window_end")
                decision = record.get("decision") or {}
                issued = decision.get("issued_unix_ms") or record.get("issued_unix_ms")
                if window_end is None or issued is None:
                    continue
                latencies.append(float(issued) - float(window_end))
        source = "adaptive_control_session_log"
        manifest_path = decisions_path.parent / "session_manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                source = str(manifest.get("data_source") or manifest.get("mode") or source)
            except json.JSONDecodeError:
                pass
        if latencies:
            values = np.asarray(latencies, dtype=float)
            rows.append(
                {
                    "session_id": session_id,
                    "n_decision_records": int(len(values)),
                    "mean_ms": float(values.mean()),
                    "median_ms": float(np.median(values)),
                    "p95_ms": float(np.quantile(values, 0.95)),
                    "min_ms": float(values.min()),
                    "max_ms": float(values.max()),
                    "data_source": source,
                }
            )
    output = pd.DataFrame(rows)
    if not output.empty:
        all_latencies: list[float] = []
        for decisions_path in sorted((ROOT / "artifacts" / "realtime" / "adaptive_control").glob("*/decisions.jsonl")):
            with decisions_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    decision = record.get("decision") or {}
                    if "window_end_ms" in record and "issued_unix_ms" in decision:
                        all_latencies.append(float(decision["issued_unix_ms"]) - float(record["window_end_ms"]))
        if all_latencies:
            values = np.asarray(all_latencies, dtype=float)
            output = pd.concat(
                [
                    output,
                    pd.DataFrame(
                        [
                            {
                                "session_id": "ALL",
                                "n_decision_records": int(len(values)),
                                "mean_ms": float(values.mean()),
                                "median_ms": float(np.median(values)),
                                "p95_ms": float(np.quantile(values, 0.95)),
                                "min_ms": float(values.min()),
                                "max_ms": float(values.max()),
                                "data_source": "all adaptive_control_session_log rows",
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
    output.to_csv(TABLE_DIR / "latency_benchmark.csv", index=False)
    return output


def run_regression_latency(additional_oof: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    metrics = existing_oof_metrics(additional_oof)
    latency = latency_metrics()
    standard_note = pd.DataFrame(
        [
            {
                "artifact": "standard Condition 1DCNN OOF",
                "status": "not_found",
                "note": "Only aggregate values/checkpoints are present; RMSE/R2 were not recomputed without saved predicted-vs-actual rows.",
            }
        ]
    )
    standard_note.to_csv(TABLE_DIR / "missing_oof_notes.csv", index=False)
    lines = [
        "# 回归指标与端到端延迟补充报告",
        "",
        "RMSE 与 R2 均从保存的 predicted-vs-actual OOF 行重新计算；未从聚合指标反推。",
        "",
        "## RMSE / R2",
        "",
        df_to_md(metrics.head(80)),
        "",
        "完整表见 `artifacts/reports/supplementary/regression_rmse_r2_metrics.csv`。标准 Condition 1DCNN 当前未保留 OOF predicted-vs-actual 行，因此本次没有为该分支反推 RMSE/R2。",
        "",
        "## 端到端延迟",
        "",
        df_to_md(latency),
        "",
        "延迟定义为 `decision.issued_unix_ms - window_end_ms`。这些日志来自本地 adaptive-control session log，用于技术基准，不代表新的正式受试者实验。",
    ]
    write_text(REPORT_DIR / "regression_latency_report_zh.md", "\n".join(lines))
    return {"metrics": metrics, "latency": latency}


def run_cfc_check() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in [
        ROOT / "artifacts" / "features" / "condition_features.csv",
        ROOT / "artifacts" / "features" / "window_features.csv",
    ]:
        frame = pd.read_csv(path, nrows=10 if path.name == "window_features.csv" else None)
        candidates = [name for name in frame.columns if re.search(r"delta.*beta|cfc|cross.?freq|coupling", name, re.IGNORECASE)]
        rows.append({"file": str(path), "candidate_count": len(candidates), "candidates": "; ".join(candidates)})
    result = pd.DataFrame(rows)
    result.to_csv(TABLE_DIR / "delta_beta_cfc_feature_check.csv", index=False)
    lines = [
        "# Delta-Beta / CFC 特征检查",
        "",
        df_to_md(result),
        "",
        "当前 `condition_features.csv` 和 `window_features.csv` 没有已实现的 delta-beta、CFC、cross-frequency 或 coupling 字段。5.1.4 不应被写成已有统计结果；若论文需要该小节，应作为单独特征工程任务，从原始 EEG 窗口计算 delta/beta 包络相关或相位-振幅耦合指标后，再按 N=9 重新运行剂量-反应统计。",
    ]
    write_text(REPORT_DIR / "cfc_feature_check_report_zh.md", "\n".join(lines))
    return result


def write_consolidated(
    dose: dict[str, pd.DataFrame],
    significance: pd.DataFrame,
    classical: dict[str, pd.DataFrame],
    regression_latency: dict[str, pd.DataFrame],
    cfc: pd.DataFrame,
    skip_deep: bool,
) -> None:
    lines = [
        "# 补充证据综合报告",
        "",
        "生成日期：2026-07-01。",
        "",
        "## 1. Data and QC",
        "",
        "本次补充分析使用 P002-P016，监督单元仍为 participant x Condition，共 135 行。EEG 相关统计仅使用 EEG 可用的 9 名参与者。10 秒窗口不作为独立标签样本。",
        "",
        df_to_md(dose["sample"][["dependent_variable", "subset", "n_participants_observed", "n_rows_observed", "missing_cells"]]),
        "",
        "## 2. Dose-response results",
        "",
        "完整剂量-反应统计见 `dose_response_report_zh.md`。主要 ANOVA 宽表如下：",
        "",
        df_to_md(effect_summary_table(dose["anova"])),
        "",
        "线性趋势混合模型完整表见 `artifacts/reports/supplementary/dose_response_linear_trends.csv`。",
        "",
        "## 3. Model comparison results",
        "",
        "### 预设配对显著性检验",
        "",
        df_to_md(significance),
        "",
        "### 额外经典模型",
        "",
        "ElasticNet 与 RandomForestRegressor 的完整 OOF 与折内选择表已保存。下表包含 P/H/E/V/PHEV 五个预设组合：",
        "",
        df_to_md(classical["metrics"]),
        "",
        "### 额外深度模型",
        "",
        "Task 4 为可选项，本次未训练 LSTM 或 CNN-LSTM。因此论文中不应把深度模型部分写成 LSTM/CNN-LSTM 的系统比较；当前可写的是已有 1DCNN 与本次额外经典模型的补充比较。",
        "" if skip_deep else "",
        "## 4. Supplementary regression metrics and latency",
        "",
        "RMSE/R2 直接从保存的 OOF prediction rows 计算，完整结果见 `artifacts/reports/supplementary/regression_rmse_r2_metrics.csv`。",
        "",
        df_to_md(regression_latency["latency"]),
        "",
        "## 5. Delta-Beta / CFC",
        "",
        df_to_md(cfc),
        "",
        "未发现已实现字段；5.1.4 应暂写为 not yet available，或另立 EEG 特征工程任务。",
        "",
        "## 6. Interpretation constraints",
        "",
        "所有新增结果均为观察性统计关联、组间差异或离线模型比较。它们不证明生理机制，不构成新的正式 human-subject controlled study，也不改变当前 `classical`、Shadow-only、`hold` 的部署结论。",
        "",
        "## 7. Thesis-section mapping",
        "",
        "| 论文位置 | 可插入证据 | 输出文件 |",
        "|---|---|---|",
        "| 3.7(a), 5.1.2-5.1.7, RQ1/H1 | intensity/frequency 的 3 x 3 ANOVA、线性趋势、描述统计 | `dose_response_report_zh.md` |",
        "| 3.7(b), 5.2.9 | 四个预设模型/模态配对 Wilcoxon 检验 | `model_significance_report_zh.md` |",
        "| 3.6.2, 5.2.2 | ElasticNet 与 RandomForestRegressor 的 LOPO 表 | `additional_classical_models_report_zh.md` |",
        "| 3.6.3, 5.2.3 | 本次未新增 LSTM/CNN-LSTM；保留已有 1DCNN 表述 | 本综合报告说明 |",
        "| 3.6.6, 5.2.8 | RMSE/R2 与 adaptive-control session latency | `regression_latency_report_zh.md` |",
        "| 5.1.4 | CFC 字段缺失，需单独工程 | `cfc_feature_check_report_zh.md` |",
    ]
    write_text(REPORT_DIR / "supplementary_evidence_zh.md", "\n".join(lines))


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-classical", action="store_true", help="Skip Task 3 retraining.")
    parser.add_argument("--skip-deep", action="store_true", default=True, help="Task 4 is optional and skipped by default.")
    args = parser.parse_args()
    ensure_dirs()
    dose = run_dose_response()
    significance = run_model_significance()
    if args.skip_classical:
        classical = {
            "oof": pd.DataFrame(),
            "selection": pd.DataFrame(),
            "metrics": pd.DataFrame(),
        }
    else:
        classical = run_additional_classical()
    regression_latency = run_regression_latency(classical["oof"])
    cfc = run_cfc_check()
    write_consolidated(dose, significance, classical, regression_latency, cfc, args.skip_deep)
    print("Supplementary reports written under artifacts/reports", flush=True)


if __name__ == "__main__":
    main()

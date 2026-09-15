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
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path("src").resolve()))

from mac.config import load_config  # noqa: E402
from mac.data.alignment import load_marker_events  # noqa: E402
from mac.data.index import build_index  # noqa: E402
from mac.data.io import normalize_condition  # noqa: E402
from mac.features.extract import extract_features  # noqa: E402
from mac.data.condition_data import aggregate_window_frame  # noqa: E402


ROOT = Path(".")
FEATURE_DIR = ROOT / "artifacts" / "features"
PREPROCESSED_DIR = ROOT / "artifacts" / "preprocessed"
REPORT_DIR = ROOT / "artifacts" / "reports"
ROUND2_DIR = REPORT_DIR / "supplementary_round2"

CONDITION_FEATURES = FEATURE_DIR / "condition_features.csv"
GATED_FEATURES = FEATURE_DIR / "condition_features_eeg_gated.csv"
BASELINE_FEATURES = FEATURE_DIR / "condition_baseline_features.csv"

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
EEG_AVAILABLE = {"P003", "P004", "P007", "P008", "P009", "P011", "P012", "P013", "P015"}

PHYSIO_TARGETS = (
    ("median_alpha_power_channel_mean", "median alpha power", "eeg", 9),
    ("median_beta_power_channel_mean", "median beta power", "eeg", 9),
    ("ecg_hr_bpm__mean", "heart rate", "all", 15),
    ("ecg_hrv_60s_rmssd_ms__mean", "HRV RMSSD 60s", "all", 15),
    ("eye_fixation_fraction_ivt__mean", "eye fixation fraction", "all", 15),
    ("eye_gaze_on_painting_fraction__mean", "eye gaze on painting fraction", "all", 15),
    ("head_speed_mean__mean", "head speed mean", "all", 15),
    ("head_stationary_fraction__mean", "head stationary fraction", "all", 15),
)


@dataclass(frozen=True)
class DvSpec:
    name: str
    label: str
    column: str
    subset: str
    n_expected: int
    note: str = ""


def ensure_dirs() -> None:
    ROUND2_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    try:
        frame.to_parquet(path.with_suffix(".parquet"), index=False)
    except Exception:
        pass


def fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "" if value is None else str(value)
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def p_fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(number):
        return "NA"
    return "<0.001" if number < 0.001 else f"{number:.4f}"


def md_table(frame: pd.DataFrame, digits: int = 4, max_rows: int | None = None) -> str:
    if frame.empty:
        return "_无可报告行。_"
    working = frame.head(max_rows).copy() if max_rows else frame.copy()
    for column in working.columns:
        if pd.api.types.is_numeric_dtype(working[column]):
            working[column] = working[column].map(lambda value: fmt(value, digits))
        else:
            working[column] = working[column].astype(object).where(working[column].notna(), "").astype(str)
    headers = [str(column) for column in working.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in working.iterrows():
        lines.append("| " + " | ".join(str(row[column]).replace("\n", " ") for column in working.columns) + " |")
    if max_rows and len(frame) > max_rows:
        lines.append(f"\n_仅显示前 {max_rows} 行；完整表见 CSV。_")
    return "\n".join(lines)


def find_cols(frame: pd.DataFrame, pattern: str) -> list[str]:
    return [name for name in frame.columns if re.search(pattern, name, re.IGNORECASE)]


def normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["participant_id"] = output["participant_id"].astype(str).str.upper().str.strip()
    output["condition"] = output["condition"].map(normalize_condition)
    return output


def add_design_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = normalize_keys(frame)
    output["intensity_level"] = pd.Categorical(
        output["condition"].map(lambda value: LEVEL_MAP[str(value)][0]),
        categories=LEVELS,
        ordered=True,
    )
    output["frequency_level"] = pd.Categorical(
        output["condition"].map(lambda value: LEVEL_MAP[str(value)][1]),
        categories=LEVELS,
        ordered=True,
    )
    if "intensity" in output.columns:
        output["intensity_value"] = pd.to_numeric(output["intensity"], errors="coerce")
    else:
        output["intensity_value"] = output["condition"].map({"C1": 0.08, "C2": 0.08, "C3": 0.08, "C4": 0.16, "C5": 0.16, "C6": 0.16, "C7": 0.25, "C8": 0.25, "C9": 0.25})
    if "frequency" in output.columns:
        output["frequency_value"] = pd.to_numeric(output["frequency"], errors="coerce")
    else:
        output["frequency_value"] = output["condition"].map({"C1": 0.12, "C2": 0.26, "C3": 0.41, "C4": 0.12, "C5": 0.26, "C6": 0.41, "C7": 0.12, "C8": 0.26, "C9": 0.41})
    return output


def load_analysis_frame(features_path: Path) -> pd.DataFrame:
    labels = normalize_keys(pd.read_csv(PREPROCESSED_DIR / "condition_labels.csv"))
    features = normalize_keys(pd.read_csv(features_path))
    duplicate_feature_columns = [
        name for name in labels.columns if name not in {"participant_id", "condition"} and name in features.columns
    ]
    merged = labels.merge(
        features.drop(columns=duplicate_feature_columns),
        on=["participant_id", "condition"],
        how="inner",
        validate="one_to_one",
    )
    return add_design_columns(merged[merged["participant_id"].ne("P001")].reset_index(drop=True))


def add_eeg_composites(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    alpha_cols = find_cols(output, r"^eeg_.*alpha.*power.*__median$")
    beta_cols = find_cols(output, r"^eeg_.*beta.*power.*__median$")
    if alpha_cols:
        output["median_alpha_power_channel_mean"] = output[alpha_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    if beta_cols:
        output["median_beta_power_channel_mean"] = output[beta_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    return output


def build_all_specs(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[DvSpec]]:
    working = add_eeg_composites(frame)
    specs: list[DvSpec] = []
    for name in ("relaxation", "discomfort", "calm", "pleasantness", "monotony_raw"):
        if name in working.columns:
            specs.append(DvSpec(name, name, name, "all", 15, "questionnaire label; analysis target only"))
    for name, label, subset, n_expected in PHYSIO_TARGETS:
        if name in working.columns:
            specs.append(DvSpec(name, label, name, "eeg_available" if subset == "eeg" else "all", n_expected))
    return working, specs


def build_physio_specs(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[DvSpec]]:
    working = add_eeg_composites(frame)
    specs = [
        DvSpec(name, label, name, "eeg_available" if subset == "eeg" else "all", n_expected)
        for name, label, subset, n_expected in PHYSIO_TARGETS
        if name in working.columns
    ]
    return working, specs


def dv_work(frame: pd.DataFrame, spec: DvSpec) -> pd.DataFrame:
    output = frame.copy()
    if spec.subset == "eeg_available":
        output = output[output["participant_id"].isin(EEG_AVAILABLE)].copy()
    output["dv_value"] = pd.to_numeric(output[spec.column], errors="coerce")
    return output


def zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    std = numeric.std(ddof=1)
    if not np.isfinite(std) or std <= 1e-12:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (numeric - numeric.mean()) / std


def fit_z_trend(work: pd.DataFrame, spec: DvSpec, *, prefix: str = "") -> list[dict[str, Any]]:
    import statsmodels.formula.api as smf

    complete = work.dropna(subset=["dv_value", "intensity_value", "frequency_value"]).copy()
    if len(complete) < 10 or complete["participant_id"].nunique() < 3:
        return []
    complete["dv_value_z"] = zscore(complete["dv_value"])
    complete["intensity_value_z"] = zscore(complete["intensity_value"])
    complete["frequency_value_z"] = zscore(complete["frequency_value"])
    complete = complete.dropna(subset=["dv_value_z", "intensity_value_z", "frequency_value_z"]).copy()
    if complete.empty:
        return []
    formula = "dv_value_z ~ intensity_value_z * frequency_value_z"
    result = None
    method_used = "ols_cluster_fallback"
    converged = False
    for reml in (True, False):
        for method in ("lbfgs", "powell", "cg", "nm"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    candidate = smf.mixedlm(formula, complete, groups=complete["participant_id"]).fit(
                        reml=reml,
                        method=method,
                        maxiter=1000,
                        disp=False,
                    )
                if bool(getattr(candidate, "converged", True)):
                    result = candidate
                    method_used = f"mixedlm_{method}_{'reml' if reml else 'ml'}"
                    converged = True
                    break
            except Exception:
                continue
        if result is not None:
            break
    if result is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = smf.ols(formula, complete).fit(
                cov_type="cluster",
                cov_kwds={"groups": complete["participant_id"]},
            )
    terms = ("intensity_value_z", "frequency_value_z", "intensity_value_z:frequency_value_z")
    rows: list[dict[str, Any]] = []
    for term in terms:
        rows.append(
            {
                "analysis": prefix,
                "dv": spec.name,
                "label": spec.label,
                "n_participants": int(complete["participant_id"].nunique()),
                "n_rows": int(len(complete)),
                "method": method_used,
                "converged": bool(converged),
                "term": term,
                "coef_z": float(result.params.get(term, np.nan)),
                "se_z": float(result.bse.get(term, np.nan)),
                "p_value": float(result.pvalues.get(term, np.nan)),
            }
        )
    return rows


def fit_factorial(work: pd.DataFrame, spec: DvSpec, *, prefix: str = "") -> list[dict[str, Any]]:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    complete = work.dropna(subset=["dv_value"]).copy()
    if complete.empty or complete["participant_id"].nunique() < 3:
        return []
    cell_counts = complete.groupby(["participant_id", "condition"], observed=False)["dv_value"].size()
    balanced = (
        complete["participant_id"].nunique() == spec.n_expected
        and len(complete) == spec.n_expected * 9
        and (cell_counts == 1).all()
    )
    source_map = {
        "intensity_level": "intensity",
        "frequency_level": "frequency",
        "intensity_level:frequency_level": "interaction",
        "C(intensity_level)": "intensity",
        "C(frequency_level)": "frequency",
        "C(intensity_level):C(frequency_level)": "interaction",
    }
    if balanced:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                table = AnovaRM(
                    complete,
                    depvar="dv_value",
                    subject="participant_id",
                    within=["intensity_level", "frequency_level"],
                ).fit().anova_table.reset_index(names="source_raw")
            rows = []
            for _, row in table.iterrows():
                effect = source_map.get(str(row["source_raw"]))
                if not effect:
                    continue
                f_value = max(0.0, float(row["F Value"]))
                df1, df2 = float(row["Num DF"]), float(row["Den DF"])
                eta = (f_value * df1) / ((f_value * df1) + df2) if np.isfinite(f_value) else np.nan
                rows.append(
                    {
                        "analysis": prefix,
                        "dv": spec.name,
                        "label": spec.label,
                        "n_participants": int(complete["participant_id"].nunique()),
                        "n_rows": int(len(complete)),
                        "method": "rm_anova",
                        "source": effect,
                        "df1": df1,
                        "df2": df2,
                        "statistic": f_value,
                        "statistic_type": "F",
                        "p_value": float(row["Pr > F"]),
                        "eta_p2": float(eta),
                    }
                )
            return rows
        except Exception:
            pass

    complete["dv_value_z"] = zscore(complete["dv_value"])
    complete = complete.dropna(subset=["dv_value_z"]).copy()
    formula = "dv_value_z ~ C(intensity_level) * C(frequency_level)"
    result = None
    method_used = "ols_participant_fixed_effect_fallback"
    for reml in (False, True):
        for method in ("lbfgs", "powell", "cg", "nm"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    candidate = smf.mixedlm(formula, complete, groups=complete["participant_id"]).fit(
                        reml=reml,
                        method=method,
                        maxiter=1000,
                        disp=False,
                    )
                    term_table = candidate.wald_test_terms(skip_single=False).table
                if bool(getattr(candidate, "converged", True)):
                    result = term_table
                    method_used = f"mixedlm_wald_{method}_{'reml' if reml else 'ml'}"
                    break
            except Exception:
                continue
        if result is not None:
            break
    rows: list[dict[str, Any]] = []
    if result is not None:
        for raw, effect in source_map.items():
            if raw not in result.index or effect not in {"intensity", "frequency", "interaction"}:
                continue
            statistic = float(np.asarray(result.loc[raw, "statistic"]).reshape(-1)[0])
            df_constraint = float(result.loc[raw, "df_constraint"])
            rows.append(
                {
                    "analysis": prefix,
                    "dv": spec.name,
                    "label": spec.label,
                    "n_participants": int(complete["participant_id"].nunique()),
                    "n_rows": int(len(complete)),
                    "method": method_used,
                    "source": effect,
                    "df1": df_constraint,
                    "df2": np.nan,
                    "statistic": statistic,
                    "statistic_type": "wald_chi2",
                    "p_value": float(result.loc[raw, "pvalue"]),
                    "eta_p2": np.nan,
                }
            )
        return rows

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ols = smf.ols(
            "dv_value_z ~ C(participant_id) + C(intensity_level) * C(frequency_level)",
            complete,
        ).fit()
        anova = sm.stats.anova_lm(ols, typ=2)
    residual_ss = float(anova.loc["Residual", "sum_sq"]) if "Residual" in anova.index else np.nan
    for raw, effect in source_map.items():
        if raw not in anova.index or effect not in {"intensity", "frequency", "interaction"}:
            continue
        ss_effect = float(anova.loc[raw, "sum_sq"])
        eta = ss_effect / (ss_effect + residual_ss) if np.isfinite(residual_ss) and residual_ss >= 0 else np.nan
        rows.append(
            {
                "analysis": prefix,
                "dv": spec.name,
                "label": spec.label,
                "n_participants": int(complete["participant_id"].nunique()),
                "n_rows": int(len(complete)),
                "method": method_used,
                "source": effect,
                "df1": float(anova.loc[raw, "df"]),
                "df2": float(anova.loc["Residual", "df"]) if "Residual" in anova.index else np.nan,
                "statistic": float(anova.loc[raw, "F"]),
                "statistic_type": "F",
                "p_value": float(anova.loc[raw, "PR(>F)"]),
                "eta_p2": eta,
            }
        )
    return rows


def apply_fdr(frame: pd.DataFrame, source_col: str = "p_value") -> pd.DataFrame:
    output = frame.copy()
    if output.empty or source_col not in output.columns:
        output["p_fdr"] = []
        return output
    output["p_fdr"] = multipletests(output[source_col].fillna(1.0), method="fdr_bh")[1]
    return output


def grid_descriptives(frame: pd.DataFrame, dv_column: str, label: str, prefix: str) -> pd.DataFrame:
    work = add_design_columns(frame.copy())
    work["dv_value"] = pd.to_numeric(work[dv_column], errors="coerce")
    desc = (
        work.dropna(subset=["dv_value"])
        .groupby(["intensity_level", "frequency_level"], observed=False)["dv_value"]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
    )
    desc.insert(0, "version", prefix)
    desc.insert(1, "dv", label)
    return desc


def condition_n_table(frame: pd.DataFrame, dv_column: str, label: str) -> pd.DataFrame:
    work = add_design_columns(frame.copy())
    work["dv_value"] = pd.to_numeric(work[dv_column], errors="coerce")
    out = (
        work.groupby(["intensity_level", "frequency_level"], observed=False)["dv_value"]
        .apply(lambda x: int(x.notna().sum()))
        .reset_index(name="n")
    )
    out.insert(0, "dv", label)
    return out


def run_task_a() -> dict[str, pd.DataFrame]:
    feats = normalize_keys(pd.read_csv(CONDITION_FEATURES))
    cov_cols = find_cols(feats, r"^qc_eeg_strict_coverage__mean$")
    if not cov_cols:
        cov_cols = find_cols(feats, r"eeg.*coverage.*__mean")
    if not cov_cols:
        windows = normalize_keys(pd.read_csv(FEATURE_DIR / "window_features.csv"))
        coverage = (
            windows.groupby(["participant_id", "condition"], as_index=False)["qc_eeg_strict_coverage"]
            .mean()
            .rename(columns={"qc_eeg_strict_coverage": "qc_eeg_strict_coverage__mean"})
        )
        feats = feats.merge(coverage, on=["participant_id", "condition"], how="left")
        cov_col = "qc_eeg_strict_coverage__mean"
    else:
        cov_col = cov_cols[0]

    feats = add_eeg_composites(feats)
    eeg_cols = [name for name in feats.columns if name.startswith("eeg_")]
    eeg_rows = feats["participant_id"].isin(EEG_AVAILABLE)
    low_mask = eeg_rows & pd.to_numeric(feats[cov_col], errors="coerce").lt(0.60)
    gated = feats.copy()
    gated.loc[low_mask, eeg_cols] = np.nan
    save_csv(gated.drop(columns=[c for c in ("median_alpha_power_channel_mean", "median_beta_power_channel_mean") if c in gated.columns]), GATED_FEATURES)

    diag_cols = ["participant_id", "condition", cov_col, "median_alpha_power_channel_mean", "median_beta_power_channel_mean"]
    diag = feats.loc[eeg_rows, diag_cols].sort_values([cov_col, "participant_id", "condition"]).reset_index(drop=True)
    low = diag[pd.to_numeric(diag[cov_col], errors="coerce").lt(0.60)].copy()

    safety_rows = []
    for participant, group in diag.dropna(subset=["median_alpha_power_channel_mean", "median_beta_power_channel_mean"]).groupby("participant_id"):
        for _, row in group.iterrows():
            if float(row[cov_col]) < 0.60:
                continue
            peers = group[group["condition"].ne(row["condition"])]
            for dv in ("median_alpha_power_channel_mean", "median_beta_power_channel_mean"):
                peer_median = float(pd.to_numeric(peers[dv], errors="coerce").median())
                value = float(row[dv])
                ratio = value / peer_median if np.isfinite(peer_median) and peer_median > 0 else np.nan
                if np.isfinite(ratio) and ratio >= 10.0:
                    safety_rows.append(
                        {
                            "participant_id": participant,
                            "condition": row["condition"],
                            "dv": dv,
                            "value": value,
                            "participant_other_condition_median": peer_median,
                            "ratio": ratio,
                            "coverage": float(row[cov_col]),
                        }
                    )
    safety = pd.DataFrame(safety_rows)

    before = add_design_columns(add_eeg_composites(feats))
    after = add_design_columns(add_eeg_composites(gated))
    desc = pd.concat(
        [
            grid_descriptives(before, "median_alpha_power_channel_mean", "median_alpha_power_channel_mean", "before_gating"),
            grid_descriptives(after, "median_alpha_power_channel_mean", "median_alpha_power_channel_mean", "after_gating"),
            grid_descriptives(before, "median_beta_power_channel_mean", "median_beta_power_channel_mean", "before_gating"),
            grid_descriptives(after, "median_beta_power_channel_mean", "median_beta_power_channel_mean", "after_gating"),
        ],
        ignore_index=True,
    )
    cell_n = pd.concat(
        [
            condition_n_table(after[after["participant_id"].isin(EEG_AVAILABLE)], "median_alpha_power_channel_mean", "median_alpha_power_channel_mean"),
            condition_n_table(after[after["participant_id"].isin(EEG_AVAILABLE)], "median_beta_power_channel_mean", "median_beta_power_channel_mean"),
        ],
        ignore_index=True,
    )

    analysis = load_analysis_frame(GATED_FEATURES)
    analysis, specs = build_physio_specs(analysis)
    eeg_specs = [spec for spec in specs if spec.subset == "eeg_available"]
    anova = apply_fdr(pd.DataFrame([row for spec in eeg_specs for row in fit_factorial(dv_work(analysis, spec), spec, prefix="eeg_gated_absolute")]))
    trends = apply_fdr(pd.DataFrame([row for spec in eeg_specs for row in fit_z_trend(dv_work(analysis, spec), spec, prefix="eeg_gated_absolute")]))

    save_csv(diag, ROUND2_DIR / "eeg_coverage_diagnostic_all_eeg_available.csv")
    save_csv(low, ROUND2_DIR / "eeg_coverage_low_cells.csv")
    save_csv(safety, ROUND2_DIR / "eeg_physical_safety_flags.csv")
    save_csv(desc, ROUND2_DIR / "eeg_gating_descriptives_before_after.csv")
    save_csv(cell_n, ROUND2_DIR / "eeg_gating_cell_n_after.csv")
    save_csv(anova, ROUND2_DIR / "eeg_gated_factorial_results.csv")
    save_csv(trends, ROUND2_DIR / "eeg_gated_linear_trends_z.csv")

    any_sig = bool((anova.get("p_fdr", pd.Series(dtype=float)) <= 0.05).any() or (trends.get("p_fdr", pd.Series(dtype=float)) <= 0.05).any())
    conclusion = (
        "门控后 EEG 剂量-反应至少有一个 FDR 校正后显著项，需按下表具体解读。"
        if any_sig
        else "EEG 在当前样本与门控下未见 FDR 校正后显著剂量-反应。"
    )
    lines = [
        "# EEG 条件层 coverage 门控报告",
        "",
        "生成日期：2026-07-01。",
        "",
        f"- 条件层 coverage 列：`{cov_col}`。",
        "- 门控阈值：EEG-available participant-condition 中 coverage < 0.60 的 EEG 特征列置为 NaN。",
        "- EEG-available 子集：P003/P004/P007/P008/P009/P011/P012/P013/P015（N=9）。",
        "",
        "## 被门控的条件格",
        "",
        md_table(low[["participant_id", "condition", cov_col, "median_alpha_power_channel_mean", "median_beta_power_channel_mean"]], 4),
        "",
        "## coverage 升序诊断（EEG-available）",
        "",
        md_table(diag[["participant_id", "condition", cov_col, "median_alpha_power_channel_mean", "median_beta_power_channel_mean"]], 4, max_rows=30),
        "",
        "## 物理合理性安全网",
        "",
        "下表列出 coverage 达标但 alpha/beta 高于同参与者其他条件中位数 10 倍以上的格子；空表表示未发现额外达标异常格。",
        "",
        md_table(safety, 4),
        "",
        "## 门控前后 3x3 描述统计",
        "",
        md_table(desc, 4),
        "",
        "## 门控后每格实际 N",
        "",
        md_table(cell_n, 0),
        "",
        "## 门控后 EEG 3x3 结果",
        "",
        md_table(anova[["dv", "source", "method", "n_participants", "n_rows", "statistic_type", "statistic", "p_value", "p_fdr"]], 4),
        "",
        "## 门控后 EEG 线性趋势",
        "",
        md_table(trends[["dv", "term", "method", "converged", "coef_z", "se_z", "p_value", "p_fdr"]], 4),
        "",
        "## 结论",
        "",
        conclusion,
    ]
    write_text(REPORT_DIR / "eeg_coverage_gating_report_zh.md", "\n".join(lines))
    return {"diag": diag, "low": low, "safety": safety, "desc": desc, "anova": anova, "trends": trends}


def run_task_b() -> dict[str, pd.DataFrame]:
    frame = load_analysis_frame(GATED_FEATURES)
    frame, specs = build_all_specs(frame)
    trends = apply_fdr(pd.DataFrame([row for spec in specs for row in fit_z_trend(dv_work(frame, spec), spec, prefix="gated_absolute")]))
    previous_path = REPORT_DIR / "supplementary" / "dose_response_linear_trends.csv"
    comparison_rows = []
    if previous_path.exists():
        previous = pd.read_csv(previous_path)
        term_map = {
            "intensity_value_z": "intensity_value",
            "frequency_value_z": "frequency_value",
            "intensity_value_z:frequency_value_z": "intensity_value:frequency_value",
        }
        for _, row in trends.iterrows():
            prev = previous[(previous["dv"].eq(row["dv"])) & (previous["term"].eq(term_map.get(row["term"], row["term"])))]
            previous_row = prev.iloc[0] if not prev.empty else {}
            comparison_rows.append(
                {
                    "dv": row["dv"],
                    "term": row["term"],
                    "round1_method": previous_row.get("method", ""),
                    "round1_coef": previous_row.get("coef", np.nan),
                    "round1_p": previous_row.get("p_value", np.nan),
                    "round2_method": row["method"],
                    "round2_converged": row["converged"],
                    "round2_coef_z": row["coef_z"],
                    "round2_p": row["p_value"],
                    "round2_p_fdr": row["p_fdr"],
                }
            )
    comparison = pd.DataFrame(comparison_rows)
    method_summary = (
        trends.groupby(["dv", "method", "converged"], dropna=False)
        .size()
        .reset_index(name="n_terms")
        .sort_values(["dv", "method"])
    )
    save_csv(trends, ROUND2_DIR / "dose_response_linear_trends_z.csv")
    save_csv(comparison, ROUND2_DIR / "dose_response_linear_trend_round1_vs_round2.csv")
    save_csv(method_summary, ROUND2_DIR / "dose_response_linear_trend_method_summary.csv")

    relaxation = trends[(trends["dv"].eq("relaxation")) & (trends["term"].isin(["intensity_value_z", "frequency_value_z"]))]
    relaxation_sig = bool((relaxation.get("p_fdr", pd.Series(dtype=float)) <= 0.05).any())
    relaxation_text = (
        "relaxation 的参数线性趋势在标准化 mixedlm/回退链下仍有 FDR 校正后显著项。"
        if relaxation_sig
        else "relaxation 的参数线性趋势在标准化 mixedlm/回退链下未达到 FDR 校正后显著。"
    )
    key_dvs = comparison[comparison["dv"].isin(["relaxation", "calm", "pleasantness", "discomfort"])] if not comparison.empty else trends[trends["dv"].isin(["relaxation", "calm", "pleasantness", "discomfort"])]
    lines = [
        "# 剂量-反应线性趋势 Round-2 更新",
        "",
        "生成日期：2026-07-01。",
        "",
        "本节使用 `artifacts/features/condition_features_eeg_gated.csv`，对 `intensity_value`、`frequency_value` 和 DV 均做 z-score 后拟合随机截距 mixedlm；仅当所有 mixedlm 优化器失败时才回退到按 participant 聚类稳健标准误的 OLS。",
        "",
        "## 拟合方法汇总",
        "",
        md_table(method_summary, 4),
        "",
        "## 第一轮 vs 第二轮关键 DV 对比",
        "",
        md_table(key_dvs, 4, max_rows=36),
        "",
        "## Round-2 完整趋势表",
        "",
        md_table(trends[["dv", "term", "method", "converged", "n_participants", "n_rows", "coef_z", "se_z", "p_value", "p_fdr"]], 4, max_rows=60),
        "",
        "## 结论",
        "",
        relaxation_text,
        "",
        "注意：本轮趋势表只改变统计拟合与 EEG 门控，不改变现有 `classical` / Shadow-only / `hold` 部署结论。",
    ]
    write_text(REPORT_DIR / "dose_response_report_zh.md", "\n".join(lines))
    return {"trends": trends, "comparison": comparison, "method_summary": method_summary}


def parse_analysis_seconds(event: dict[str, Any], default: float = 15.0) -> float:
    notes = str(event.get("notes") or "")
    match = re.search(r"analysis_window_seconds=([0-9.]+)", notes)
    if match:
        return float(match.group(1))
    return default


def build_baseline_windows() -> tuple[pd.DataFrame, pd.DataFrame]:
    config = load_config(ROOT / "configs" / "project.yaml")
    labels = normalize_keys(pd.read_csv(PREPROCESSED_DIR / "condition_labels.csv"))
    label_by_key = {(row.participant_id, row.condition): row._asdict() for row in labels.itertuples(index=False)}
    rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    manifest = build_index(config, config.participants)
    for source in manifest:
        participant = source["participant_id"]
        events = load_marker_events(Path(source["xdf_path"]), config.get("streams.marker_name"))
        starts = [
            event for event in events
            if str(event.get("event_type", "")).lower() == "pre_condition_baseline_start"
        ]
        ends = [
            event for event in events
            if str(event.get("event_type", "")).lower() == "pre_condition_baseline_end"
        ]
        starts = sorted(starts, key=lambda event: float(event["xdf_time"]))
        ends = sorted(ends, key=lambda event: float(event["xdf_time"]))
        if len(starts) != 9 or len(ends) != 9:
            raise ValueError(f"{participant}: expected 9 pre-condition baseline start/end markers")
        used_end_indexes: set[int] = set()
        paired: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for start in starts:
            condition = normalize_condition(start.get("condition_id") or start.get("condition"))
            match_index = next(
                (
                    index for index, end in enumerate(ends)
                    if index not in used_end_indexes
                    and normalize_condition(end.get("condition_id") or end.get("condition")) == condition
                    and float(end["xdf_time"]) > float(start["xdf_time"])
                ),
                None,
            )
            if match_index is None:
                raise ValueError(f"{participant}/{condition}: missing matching pre-condition baseline end")
            used_end_indexes.add(match_index)
            paired.append((start, ends[match_index]))
        participant_freqs = []
        for order_index, (start, end) in enumerate(paired, start=1):
            condition = normalize_condition(start.get("condition_id") or start.get("condition"))
            label = label_by_key[(participant, condition)]
            analysis_seconds = min(parse_analysis_seconds(start), float(end["xdf_time"]) - float(start["xdf_time"]))
            analysis_start_xdf = float(end["xdf_time"]) - analysis_seconds
            analysis_end_xdf = float(end["xdf_time"])
            analysis_end_ms = int(end["unix_time_ms"])
            analysis_start_ms = analysis_end_ms - int(round(analysis_seconds * 1000.0))
            start_freq = float(start.get("applied_frequency_value", np.nan))
            end_freq = float(end.get("applied_frequency_value", np.nan))
            start_intensity = float(start.get("applied_intensity_value", np.nan))
            end_intensity = float(end.get("applied_intensity_value", np.nan))
            if np.isfinite(start_freq):
                participant_freqs.append(start_freq)
            prev_condition = normalize_condition(paired[order_index - 2][0].get("condition_id")) if order_index > 1 else ""
            prev_label = label_by_key.get((participant, prev_condition), {}) if prev_condition else {}
            window = {
                **label,
                "start_xdf": analysis_start_xdf,
                "end_xdf": analysis_end_xdf,
                "start_unix_ms": analysis_start_ms,
                "end_unix_ms": analysis_end_ms,
                "start_marker_index": int(start["marker_index"]),
                "end_marker_index": int(end["marker_index"]),
                "duration_seconds": analysis_seconds,
                "window_id": f"{participant}_{condition}_BASE_W00",
                "condition_window_index": 0,
                "condition_window_count": 1,
                "window_start_xdf": analysis_start_xdf,
                "window_end_xdf": analysis_end_xdf,
                "window_start_unix_ms": analysis_start_ms,
                "window_end_unix_ms": analysis_end_ms,
                "sample_weight": 1.0,
            }
            rows.append(window)
            pair_rows.append(
                {
                    "participant_id": participant,
                    "condition": condition,
                    "baseline_order_index": order_index,
                    "baseline_analysis_seconds": analysis_seconds,
                    "baseline_phase_duration_seconds": float(end["xdf_time"]) - float(start["xdf_time"]),
                    "baseline_full_start_xdf": float(start["xdf_time"]),
                    "baseline_full_end_xdf": float(end["xdf_time"]),
                    "baseline_analysis_start_xdf": analysis_start_xdf,
                    "baseline_analysis_end_xdf": analysis_end_xdf,
                    "baseline_frequency_applied_start": start_freq,
                    "baseline_frequency_applied_end": end_freq,
                    "baseline_intensity_applied_start": start_intensity,
                    "baseline_intensity_applied_end": end_intensity,
                    "prev_condition": prev_condition,
                    "prev_intensity": prev_label.get("intensity", np.nan),
                    "prev_frequency": prev_label.get("frequency", np.nan),
                }
            )
        median_freq = float(np.nanmedian(participant_freqs))
        batch = "early_freq_0.368" if median_freq < 0.42 else "late_freq_0.471"
        for row in pair_rows:
            if row["participant_id"] == participant:
                row["baseline_frequency_participant_median"] = median_freq
                row["baseline_freq_batch"] = batch
    windows = pd.DataFrame(rows).sort_values(["participant_id", "window_start_xdf"]).reset_index(drop=True)
    pairs = pd.DataFrame(pair_rows).sort_values(["participant_id", "baseline_order_index"]).reset_index(drop=True)
    return windows, pairs


def run_task_c() -> dict[str, pd.DataFrame]:
    baseline_windows, pairs = build_baseline_windows()
    condition_windows = pd.read_csv(PREPROCESSED_DIR / "windows.csv")
    combined = pd.concat([condition_windows, baseline_windows], ignore_index=True, sort=False)
    combined = combined.sort_values(["participant_id", "window_start_xdf", "window_id"]).reset_index(drop=True)
    baseline_windows_path = ROUND2_DIR / "pre_condition_baseline_windows.csv"
    combined_windows_path = ROUND2_DIR / "windows_with_precondition_baseline.csv"
    save_csv(baseline_windows, baseline_windows_path)
    save_csv(pairs, ROUND2_DIR / "pre_condition_baseline_pairs.csv")
    save_csv(combined, combined_windows_path)

    config = load_config(ROOT / "configs" / "project.yaml")
    temp_output = FEATURE_DIR / "round2_baseline_tmp"
    result = extract_features(
        config,
        config.participants,
        include_video=False,
        output_dir=temp_output,
        windows_path=combined_windows_path,
    )
    (ROUND2_DIR / "baseline_feature_extraction_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_windows = pd.read_csv(temp_output / "window_features.csv")
    baseline_feature_windows = temp_windows[temp_windows["window_id"].astype(str).str.contains("_BASE_W00", regex=False)].copy()
    baseline_condition = aggregate_window_frame(baseline_feature_windows)
    baseline_condition = normalize_keys(baseline_condition)
    baseline_condition = baseline_condition.merge(
        pairs[
            [
                "participant_id",
                "condition",
                "baseline_order_index",
                "baseline_analysis_seconds",
                "baseline_phase_duration_seconds",
                "baseline_frequency_applied_start",
                "baseline_frequency_applied_end",
                "baseline_intensity_applied_start",
                "baseline_intensity_applied_end",
                "baseline_frequency_participant_median",
                "baseline_freq_batch",
                "prev_condition",
                "prev_intensity",
                "prev_frequency",
            ]
        ],
        on=["participant_id", "condition"],
        how="left",
        validate="one_to_one",
    )
    cov_cols = find_cols(baseline_condition, r"^qc_eeg_strict_coverage__mean$")
    if cov_cols:
        cov_col = cov_cols[0]
        eeg_cols = [name for name in baseline_condition.columns if name.startswith("eeg_")]
        mask = baseline_condition["participant_id"].isin(EEG_AVAILABLE) & pd.to_numeric(baseline_condition[cov_col], errors="coerce").lt(0.60)
        baseline_condition.loc[mask, eeg_cols] = np.nan
    save_csv(baseline_feature_windows, ROUND2_DIR / "condition_baseline_window_features.csv")
    save_csv(baseline_condition, BASELINE_FEATURES)

    baseline_analysis = load_analysis_frame(BASELINE_FEATURES)
    baseline_analysis, specs = build_physio_specs(baseline_analysis)
    desc_rows = []
    for spec in specs:
        work = dv_work(baseline_analysis, spec).dropna(subset=["dv_value"])
        desc_rows.append(
            {
                "dv": spec.name,
                "label": spec.label,
                "n_participants": int(work["participant_id"].nunique()),
                "n_rows": int(len(work)),
                "mean": float(work["dv_value"].mean()) if len(work) else np.nan,
                "sd": float(work["dv_value"].std(ddof=1)) if len(work) > 1 else np.nan,
                "median": float(work["dv_value"].median()) if len(work) else np.nan,
            }
        )
    desc = pd.DataFrame(desc_rows)
    batch = pairs[["participant_id", "baseline_freq_batch", "baseline_frequency_participant_median"]].drop_duplicates().sort_values("participant_id")
    save_csv(desc, ROUND2_DIR / "baseline_feature_descriptives.csv")
    save_csv(batch, ROUND2_DIR / "baseline_frequency_batches.csv")

    lines = [
        "# pre-condition baseline 特征抽取报告",
        "",
        "生成日期：2026-07-01。",
        "",
        "## 分析窗锚定",
        "",
        "Unity 代码在 operator 文案中明确写为 `Analysis window: final ... of this phase`，因此本轮使用每个 20 秒 pre-condition baseline 的最后 15 秒。事件日志中的 start marker 也记录 `analysis_window_seconds=15`。",
        "",
        "## baseline-condition 配对",
        "",
        "每个 baseline 段按 XDF marker 事件流配对到其 `pre_condition_baseline_start/end` 上携带的紧接 condition_id，而不是按 C1-C9 编号排序；`prev_condition` 则来自同一参与者事件流中前一个完成的 condition。",
        "",
        "## 输出与完整性",
        "",
        f"- baseline window rows: {len(baseline_windows)}。",
        f"- baseline condition feature rows: {len(baseline_condition)}。",
        f"- extractor errors: {len(result.get('errors', []))}。",
        "- baseline 段使用 `real_time_ml.features.extract.extract_features` 的同一套 physio/head/eye 特征函数，并用 `aggregate_window_frame` 聚合到 participant-condition 层。",
        "",
        "## baseline DV 描述统计",
        "",
        md_table(desc, 4),
        "",
        "## baseline frequency 批次",
        "",
        md_table(batch, 4),
    ]
    write_text(REPORT_DIR / "baseline_feature_extraction_report_zh.md", "\n".join(lines))
    return {"windows": baseline_windows, "pairs": pairs, "features": baseline_condition, "desc": desc, "batch": batch}


def run_task_d() -> dict[str, pd.DataFrame]:
    baseline = load_analysis_frame(BASELINE_FEATURES)
    baseline, specs = build_physio_specs(baseline)
    anova = apply_fdr(pd.DataFrame([row for spec in specs for row in fit_factorial(dv_work(baseline, spec), spec, prefix="baseline_homogeneity")]))

    carry_rows = []
    import statsmodels.formula.api as smf

    for spec in specs:
        work = dv_work(baseline, spec).dropna(subset=["dv_value", "prev_intensity", "prev_frequency"]).copy()
        if len(work) < 10 or work["participant_id"].nunique() < 3:
            continue
        work["dv_value_z"] = zscore(work["dv_value"])
        work["prev_intensity_z"] = zscore(work["prev_intensity"])
        work["prev_frequency_z"] = zscore(work["prev_frequency"])
        work = work.dropna(subset=["dv_value_z", "prev_intensity_z", "prev_frequency_z"]).copy()
        if len(work) < 10 or work["participant_id"].nunique() < 3:
            continue
        formula = "dv_value_z ~ prev_intensity_z + prev_frequency_z"
        result = None
        method_used = "ols_cluster_fallback"
        converged = False
        for reml in (True, False):
            for method in ("lbfgs", "powell", "cg", "nm"):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        candidate = smf.mixedlm(formula, work, groups=work["participant_id"]).fit(
                            reml=reml,
                            method=method,
                            maxiter=1000,
                            disp=False,
                        )
                    if bool(getattr(candidate, "converged", True)):
                        result = candidate
                        method_used = f"mixedlm_{method}_{'reml' if reml else 'ml'}"
                        converged = True
                        break
                except Exception:
                    continue
            if result is not None:
                break
        if result is None:
            try:
                result = smf.ols(formula, work).fit(cov_type="cluster", cov_kwds={"groups": work["participant_id"]})
            except Exception:
                continue
        for term in ("prev_intensity_z", "prev_frequency_z"):
            carry_rows.append(
                {
                    "dv": spec.name,
                    "label": spec.label,
                    "n_participants": int(work["participant_id"].nunique()),
                    "n_rows": int(len(work)),
                    "method": method_used,
                    "converged": bool(converged),
                    "term": term,
                    "coef_z": float(result.params.get(term, np.nan)),
                    "se_z": float(result.bse.get(term, np.nan)),
                    "p_value": float(result.pvalues.get(term, np.nan)),
                }
            )
    carry = apply_fdr(pd.DataFrame(carry_rows))
    save_csv(anova, ROUND2_DIR / "washstate_baseline_homogeneity_anova.csv")
    save_csv(carry, ROUND2_DIR / "washstate_carryover_mixedlm.csv")
    significant_main = anova[(anova["p_fdr"] <= 0.05)] if not anova.empty else anova
    significant_carry = carry[(carry["p_fdr"] <= 0.05)] if not carry.empty else carry
    if significant_main.empty and significant_carry.empty:
        verdict = "主检验和 carryover 副检验均未发现 FDR 校正后显著项；当前数据支持 baseline 段在 9 个 upcoming conditions 间大体同质，Δ 可作为预先计划的敏感性分析。"
    else:
        verdict = "至少一个 baseline 同质性或 carryover 检验在 FDR 校正后显著；这提示存在条件顺序残留结构，Δ 校正在解释中应视为必要的敏感性证据。"
    lines = [
        "# 洗状态操纵检验报告",
        "",
        "生成日期：2026-07-01。",
        "",
        "baseline 是固定参数 Water Lilies VFX 刺激，不是静息。由于 baseline 参数对同一参与者固定，本检验测的是前序 condition 残留，而不是 baseline 本身的条件依赖。",
        "",
        "## 主检验：upcoming-condition 同质性",
        "",
        md_table(anova[["dv", "source", "method", "n_participants", "n_rows", "statistic_type", "statistic", "p_value", "p_fdr", "eta_p2"]], 4),
        "",
        "## 副检验：前一 condition carryover",
        "",
        md_table(carry[["dv", "term", "method", "converged", "n_participants", "n_rows", "coef_z", "se_z", "p_value", "p_fdr"]], 4),
        "",
        "## 综合判断",
        "",
        verdict,
    ]
    write_text(REPORT_DIR / "washstate_manipulation_check_report_zh.md", "\n".join(lines))
    return {"anova": anova, "carry": carry}


def rank_biserial_from_diff(diff: pd.Series) -> float:
    clean = pd.to_numeric(diff, errors="coerce").dropna()
    clean = clean[clean != 0]
    if clean.empty:
        return 0.0
    ranks = stats.rankdata(np.abs(clean.to_numpy(dtype=float)))
    w_pos = float(ranks[clean.to_numpy(dtype=float) > 0].sum())
    w_neg = float(ranks[clean.to_numpy(dtype=float) < 0].sum())
    denom = len(clean) * (len(clean) + 1) / 2.0
    return (w_pos - w_neg) / denom if denom else np.nan


def run_task_e() -> dict[str, pd.DataFrame]:
    cond = load_analysis_frame(GATED_FEATURES)
    base = load_analysis_frame(BASELINE_FEATURES)
    cond, cond_specs = build_physio_specs(cond)
    base, _ = build_physio_specs(base)
    merged = cond.merge(base, on=["participant_id", "condition"], suffixes=("_cond", "_base"), validate="one_to_one")
    delta_rows = []
    for spec in cond_specs:
        cond_col = f"{spec.column}_cond"
        base_col = f"{spec.column}_base"
        if cond_col not in merged.columns or base_col not in merged.columns:
            continue
        delta_name = f"delta_{spec.name}"
        merged[delta_name] = pd.to_numeric(merged[cond_col], errors="coerce") - pd.to_numeric(merged[base_col], errors="coerce")
        work = merged[
            [
                "participant_id",
                "condition",
                "intensity_level_cond",
                "frequency_level_cond",
                "intensity_value_cond",
                "frequency_value_cond",
                delta_name,
            ]
        ].rename(
            columns={
                "intensity_level_cond": "intensity_level",
                "frequency_level_cond": "frequency_level",
                "intensity_value_cond": "intensity_value",
                "frequency_value_cond": "frequency_value",
                delta_name: spec.column,
            }
        )
        delta_spec = DvSpec(f"delta_{spec.name}", f"Delta {spec.label}", spec.column, spec.subset, spec.n_expected)
        delta_rows.append((delta_spec, work))

    abs_anova = apply_fdr(pd.DataFrame([row for spec in cond_specs for row in fit_factorial(dv_work(cond, spec), spec, prefix="absolute")]))
    abs_trends = apply_fdr(pd.DataFrame([row for spec in cond_specs for row in fit_z_trend(dv_work(cond, spec), spec, prefix="absolute")]))
    delta_anova = apply_fdr(pd.DataFrame([row for spec, work in delta_rows for row in fit_factorial(dv_work(work, spec), spec, prefix="delta")]))
    delta_trends = apply_fdr(pd.DataFrame([row for spec, work in delta_rows for row in fit_z_trend(dv_work(work, spec), spec, prefix="delta")]))

    paired_rows = []
    for spec in cond_specs:
        cond_col = f"{spec.column}_cond"
        base_col = f"{spec.column}_base"
        if cond_col not in merged.columns or base_col not in merged.columns:
            continue
        subset = merged[merged["participant_id"].isin(EEG_AVAILABLE)].copy() if spec.subset == "eeg_available" else merged.copy()
        for condition in sorted(subset["condition"].unique(), key=lambda item: int(str(item)[1:])):
            sub = subset[subset["condition"].eq(condition)].dropna(subset=[cond_col, base_col]).copy()
            if len(sub) < 5:
                continue
            diff = pd.to_numeric(sub[cond_col], errors="coerce") - pd.to_numeric(sub[base_col], errors="coerce")
            try:
                test = stats.wilcoxon(pd.to_numeric(sub[cond_col], errors="coerce"), pd.to_numeric(sub[base_col], errors="coerce"), zero_method="wilcox", alternative="two-sided")
                w_value, p_value = float(test.statistic), float(test.pvalue)
            except ValueError:
                w_value, p_value = np.nan, 1.0
            paired_rows.append(
                {
                    "dv": spec.name,
                    "label": spec.label,
                    "condition": condition,
                    "n": int(len(sub)),
                    "median_diff_condition_minus_baseline": float(diff.median()),
                    "direction": "condition_higher" if diff.median() > 0 else "condition_lower" if diff.median() < 0 else "no_median_difference",
                    "W": w_value,
                    "rank_biserial": rank_biserial_from_diff(diff),
                    "p_unc": p_value,
                }
            )
    paired = pd.DataFrame(paired_rows)
    if not paired.empty:
        paired["p_fdr"] = multipletests(paired["p_unc"].fillna(1.0), method="fdr_bh")[1]

    save_csv(abs_anova, ROUND2_DIR / "baseline_corrected_absolute_factorial.csv")
    save_csv(abs_trends, ROUND2_DIR / "baseline_corrected_absolute_trends_z.csv")
    save_csv(delta_anova, ROUND2_DIR / "baseline_corrected_delta_factorial.csv")
    save_csv(delta_trends, ROUND2_DIR / "baseline_corrected_delta_trends_z.csv")
    save_csv(paired, ROUND2_DIR / "condition_vs_baseline_paired_wilcoxon.csv")
    save_csv(merged, ROUND2_DIR / "condition_baseline_merged_for_delta.csv")

    delta_new_sig = set(delta_anova.loc[delta_anova["p_fdr"] <= 0.05, "dv"]) - set(abs_anova.loc[abs_anova["p_fdr"] <= 0.05, "dv"])
    paired_sig = paired[paired["p_fdr"] <= 0.05] if not paired.empty else paired
    lines = [
        "# baseline 校正敏感性分析报告",
        "",
        "生成日期：2026-07-01。",
        "",
        "本报告预先并列报告绝对值版本与 Δ=condition_feature-baseline_feature 版本；Δ 是敏感性分析，不替代绝对值分析。baseline 是固定参数 Water Lilies VFX 刺激，不是静息或无刺激对照。",
        "",
        "## 绝对值 3x3 剂量-反应",
        "",
        md_table(abs_anova[["dv", "source", "method", "n_participants", "n_rows", "statistic_type", "statistic", "p_value", "p_fdr"]], 4, max_rows=60),
        "",
        "## Δ 3x3 剂量-反应",
        "",
        md_table(delta_anova[["dv", "source", "method", "n_participants", "n_rows", "statistic_type", "statistic", "p_value", "p_fdr"]], 4, max_rows=60),
        "",
        "## 绝对值与 Δ 线性趋势",
        "",
        md_table(pd.concat([abs_trends, delta_trends], ignore_index=True)[["analysis", "dv", "term", "method", "converged", "coef_z", "se_z", "p_value", "p_fdr"]], 4, max_rows=80),
        "",
        "## 逐条件 vs baseline 配对 Wilcoxon",
        "",
        md_table(paired[["dv", "condition", "n", "median_diff_condition_minus_baseline", "direction", "W", "rank_biserial", "p_unc", "p_fdr"]], 4, max_rows=90),
        "",
        "## 结构性解读边界",
        "",
        "1. Δ 可能放大方差：baseline 只有 15 秒，减去带噪声的短基线不一定更稳定。",
        "2. baseline 靠近高刺激角：baseline frequency 高于最高 High 条件，Low/Medium 条件相对 baseline 多数是刺激下降，因此 Δ 的几何结构必须单独说明。",
        "3. 两批 baseline frequency：P002-P008 与 P009-P016 的 baseline frequency 协议不同；被试内 Δ 吸收了批次差异，但 baseline 绝对值的被试间比较存在混淆。",
        "",
        "## 综合判断",
        "",
        f"Δ 中相对绝对值版本新增 FDR 显著的 DV：{', '.join(sorted(delta_new_sig)) if delta_new_sig else '无'}。",
        f"逐条件配对检验 FDR 显著行数：{len(paired_sig)}。",
        "这些结果不改变现有 `classical` / Shadow-only / `hold` 部署结论。",
    ]
    write_text(REPORT_DIR / "baseline_corrected_analysis_report_zh.md", "\n".join(lines))
    return {
        "abs_anova": abs_anova,
        "abs_trends": abs_trends,
        "delta_anova": delta_anova,
        "delta_trends": delta_trends,
        "paired": paired,
    }


def write_round2_combined(task_a: dict[str, pd.DataFrame], task_b: dict[str, pd.DataFrame], task_c: dict[str, pd.DataFrame], task_d: dict[str, pd.DataFrame], task_e: dict[str, pd.DataFrame]) -> None:
    chapters = [
        ("修订后的剂量-反应：EEG 门控", REPORT_DIR / "eeg_coverage_gating_report_zh.md"),
        ("修订后的剂量-反应：线性趋势", REPORT_DIR / "dose_response_report_zh.md"),
        ("baseline 特征抽取", REPORT_DIR / "baseline_feature_extraction_report_zh.md"),
        ("洗状态操纵检验", REPORT_DIR / "washstate_manipulation_check_report_zh.md"),
        ("baseline 校正与逐条件对比", REPORT_DIR / "baseline_corrected_analysis_report_zh.md"),
    ]
    lstm_report = REPORT_DIR / "lstm_model_report_zh.md"
    if lstm_report.exists():
        chapters.append(("深度模型更新：LSTM", lstm_report))
    lines = [
        "# 第二轮补充证据合并报告",
        "",
        "生成日期：2026-07-01。",
        "",
        "## 修订说明",
        "",
        "- 本轮补加 EEG 条件层 coverage<0.60 门控，并将低 coverage 的 EEG 特征置为 NaN。",
        "- 线性趋势改为 z-score 预测变量与 DV 后优先拟合随机截距 mixedlm，仅在全部优化器失败时回退 OLS cluster。",
        "- 新增 pre-condition baseline 最后 15 秒特征抽取、洗状态同质性/carryover 检验、Δ 敏感性分析和逐条件配对检验。",
        "- baseline 是固定参数 Water Lilies VFX 刺激，不是静息或无刺激对照；绝对值与 Δ 两版均为预先计划并列报告。",
        "- 所有新分析均不改变现有 `classical` / Shadow-only / `hold` 部署结论。",
        "",
        "## 证据到论文位置对照",
        "",
        "| 论文位置 | 本轮证据 | 文件 |",
        "| --- | --- | --- |",
        "| 5.1.2-5.1.3, 3.7(a) | EEG coverage 门控后剂量-反应 | `eeg_coverage_gating_report_zh.md` |",
        "| 5.1.7, 3.7(a) | z-score mixedlm 线性趋势 | `dose_response_report_zh.md` |",
        "| 3.4, 4.1 | baseline 最后 15 秒抽取与洗状态检验 | `baseline_feature_extraction_report_zh.md`; `washstate_manipulation_check_report_zh.md` |",
        "| 5.1.2-5.1.6 | 绝对值与 Δ 并列、condition vs baseline 配对 | `baseline_corrected_analysis_report_zh.md` |",
        "| 5.2.3, 3.6.3 | LSTM 与既有 1D-CNN 深度模型对比 | `lstm_model_report_zh.md` |",
        "| 6.1.1 | baseline/EEG/method limitations | 本报告 limitations 段 |",
        "",
        "## limitations 文本草案",
        "",
        "1. **无无刺激/静息对照条件。** baseline 段显示的是固定参数的 Water Lilies VFX（非灰屏静息），因此本研究无法判断《睡莲》刺激相对于无刺激静息是否产生放松，只能判断不同参数设置之间是否产生不同反应。绝对疗效主张需要未来加入 rest/no-stimulus 控制条件；该限制不影响参数化调制与自适应闭环这两个核心贡献。",
        "2. **baseline 参考点偏高刺激。** baseline 参数中 frequency 高于最高的 High 条件，使 baseline 落在 3x3 网格的高刺激角附近，多数 condition 相对 baseline 是一次刺激下降；Δ 分析的几何结构受此影响。",
        "3. **两批 baseline frequency。** P002-P008 的 baseline frequency 约 0.368，P009-P016 约 0.471，是研究中途的协议变更。被试内 Δ 分析吸收了该差异，但 baseline 绝对值的被试间比较存在此混淆。",
        "4. **EEG 条件层 coverage 门控为事后补加。** P004/C2 因 coverage 0.40 产生极端功率值，第一轮统计未拦截；本轮补加条件层门控并重算。EEG 剂量-反应结论基于门控后数据，样本量在 EEG-available 子集下为 N=9。",
        "",
    ]
    for title, path in chapters:
        lines.extend(["", f"## {title}", "", f"来源文件：`{path.as_posix()}`。", ""])
        text = path.read_text(encoding="utf-8")
        body = "\n".join(text.splitlines()[1:]).strip() if text.startswith("# ") else text.strip()
        lines.append(body)
    write_text(REPORT_DIR / "supplementary_evidence_round2_zh.md", "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-baseline-extract", action="store_true", help="reuse existing condition_baseline_features.csv")
    args = parser.parse_args()
    ensure_dirs()
    task_a = run_task_a()
    task_b = run_task_b()
    if args.skip_baseline_extract and BASELINE_FEATURES.exists():
        task_c = {
            "features": pd.read_csv(BASELINE_FEATURES),
            "desc": pd.read_csv(ROUND2_DIR / "baseline_feature_descriptives.csv") if (ROUND2_DIR / "baseline_feature_descriptives.csv").exists() else pd.DataFrame(),
            "batch": pd.read_csv(ROUND2_DIR / "baseline_frequency_batches.csv") if (ROUND2_DIR / "baseline_frequency_batches.csv").exists() else pd.DataFrame(),
        }
    else:
        task_c = run_task_c()
    task_d = run_task_d()
    task_e = run_task_e()
    write_round2_combined(task_a, task_b, task_c, task_d, task_e)
    print(json.dumps(
        {
            "gated_features": str(GATED_FEATURES),
            "baseline_features": str(BASELINE_FEATURES),
            "round2_report": str(REPORT_DIR / "supplementary_evidence_round2_zh.md"),
            "round2_tables": str(ROUND2_DIR),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()

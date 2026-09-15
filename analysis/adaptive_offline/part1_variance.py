"""I.2 Individual differences / variance decomposition  [A level].

Finalises the Phase A heterogeneity question with three robustness routes:

  1. MixedLM ``target ~ intensity_c * frequency_c`` with random ``1 + i + f``.
     For relaxation (which would not converge with a full unstructured random
     covariance in Phase A) we add a *diagonal* random-effect structure
     (independent random intercept + slopes) so the route is estimable.
  2. Floor adaptation for discomfort: because 64% of discomfort labels sit at
     the floor, we additionally model the between-participant SD of the
     floor-crossing probability ``P(discomfort>0)`` and of ``max discomfort``
     to confirm the heterogeneity is not a censoring artifact.
  3. Winner's-curse / argmax null: simulate a *shared* group surface plus
     residual noise and measure how much individual-optimum dispersion noise
     alone produces, then compare the observed modal-optimum share against
     that null.

Outputs:
  - reports/variance_components_final.csv
  - reports/individual_optima.csv
  - reports/personalization_warranted.md
"""

from __future__ import annotations

import warnings
from collections import Counter

import numpy as np
import pandas as pd

from io_common import (
    CONDITIONS,
    INFERENCE_P,
    INFERENCE_PC,
    NEGLIGIBLE_HETEROGENEITY_SD,
    PRIMARY_TARGETS,
    add_contract_columns,
    ci,
    cluster_bootstrap,
    condition_sort_key,
    ensure_dirs,
    format_ci,
    format_number,
    load_labels,
    write_csv,
    write_text,
)


def _center(labels: pd.DataFrame) -> pd.DataFrame:
    out = labels.copy()
    out["intensity_c"] = pd.to_numeric(out["intensity"], errors="coerce")
    out["frequency_c"] = pd.to_numeric(out["frequency"], errors="coerce")
    out["intensity_c"] -= out["intensity_c"].mean()
    out["frequency_c"] -= out["frequency_c"].mean()
    return out


def _grid(labels: pd.DataFrame) -> pd.DataFrame:
    return (
        labels[["condition", "intensity_c", "frequency_c"]]
        .drop_duplicates()
        .sort_values("condition", key=lambda s: s.map(condition_sort_key))
    )


def fit_mixedlm(frame: pd.DataFrame, target: str, group_col: str, grid: pd.DataFrame,
                structure: str) -> dict:
    """structure in {'full','diagonal'}.  'full' = correlated random 1+i+f;
    'diagonal' = random intercept + independent random slopes (vc_formula)."""
    import statsmodels.formula.api as smf

    cols = [target, "intensity_c", "frequency_c", group_col]
    data = frame[cols].dropna().copy()
    if data[group_col].nunique() < 3:
        return {"status": "too_few_groups"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if structure == "full":
                model = smf.mixedlm(
                    f"{target} ~ intensity_c * frequency_c", data=data,
                    groups=data[group_col], re_formula="1 + intensity_c + frequency_c",
                )
            else:
                model = smf.mixedlm(
                    f"{target} ~ intensity_c * frequency_c", data=data,
                    groups=data[group_col], re_formula="1",
                    vc_formula={"i_slope": "0 + intensity_c", "f_slope": "0 + frequency_c"},
                )
            result = model.fit(method="lbfgs", reml=True, maxiter=300, disp=False)
        except Exception:
            return {"status": "fit_failed"}
    if not bool(getattr(result, "converged", False)):
        return {"status": "not_converged"}

    residual = float(result.scale)
    if structure == "full":
        cov = np.asarray(result.cov_re, dtype=float)
        if cov.shape != (3, 3) or not np.isfinite(cov).all():
            return {"status": "invalid_random_effect_covariance"}
        var_int = float(cov[0, 0])
        var_i = float(cov[1, 1])
        var_f = float(cov[2, 2])
        variances = []
        for _, row in grid.iterrows():
            z = np.asarray([1.0, float(row["intensity_c"]), float(row["frequency_c"])])
            variances.append(float(z @ cov @ z.T))
    else:
        cov_re = np.asarray(result.cov_re, dtype=float)
        var_int = float(cov_re[0, 0]) if cov_re.size else float("nan")
        vc = result.vcomp
        var_i = float(vc[0]) if len(vc) > 0 else float("nan")
        var_f = float(vc[1]) if len(vc) > 1 else float("nan")
        variances = []
        for _, row in grid.iterrows():
            variances.append(
                var_int + (float(row["intensity_c"]) ** 2) * var_i
                + (float(row["frequency_c"]) ** 2) * var_f
            )
    between = float(np.mean(variances))
    return {
        "status": "ok",
        "residual_variance": residual,
        "random_intercept_variance": var_int,
        "random_intensity_slope_variance": var_i,
        "random_frequency_slope_variance": var_f,
        "grid_avg_random_effect_variance": between,
        "grid_avg_random_effect_sd": float(np.sqrt(max(between, 0.0))),
        "icc_between_over_total": float(between / (between + residual)) if between + residual > 0 else float("nan"),
    }


METRICS = (
    "residual_variance", "random_intercept_variance", "random_intensity_slope_variance",
    "random_frequency_slope_variance", "grid_avg_random_effect_variance",
    "grid_avg_random_effect_sd", "icc_between_over_total",
)


def mixedlm_route(labels: pd.DataFrame, seed: int, replicates: int) -> tuple[pd.DataFrame, dict]:
    working = _center(labels)
    grid = _grid(working)
    rows: list[dict] = []
    eta_ranges: dict[str, dict] = {}

    structures = {"relaxation": "diagonal", "discomfort": "full"}
    for ti, target in enumerate(PRIMARY_TARGETS):
        structure = structures[target]
        fit = fit_mixedlm(working, target, "participant_id", grid, structure)
        if fit["status"] != "ok":
            # fall back to the alternate structure before giving up
            alt = "full" if structure == "diagonal" else "diagonal"
            fit_alt = fit_mixedlm(working, target, "participant_id", grid, alt)
            if fit_alt["status"] == "ok":
                fit, structure = fit_alt, alt
        if fit["status"] != "ok":
            for metric in METRICS:
                rows.append(_na_row(target, metric, structure, fit["status"]))
            continue

        boot: dict[str, list[float]] = {m: [] for m in METRICS}
        rng = np.random.default_rng(seed + 3000 + ti)
        participants = np.asarray(sorted(working["participant_id"].unique()))
        grouped = {p: working[working["participant_id"] == p] for p in participants}
        failures = 0
        for _ in range(replicates):
            sampled = rng.choice(participants, size=len(participants), replace=True)
            pieces = []
            for di, p in enumerate(sampled):
                piece = grouped[p].copy()
                piece["participant_id"] = f"{p}__b{di:02d}"
                pieces.append(piece)
            sample = pd.concat(pieces, ignore_index=True)
            bfit = fit_mixedlm(sample, target, "participant_id", _grid(sample), structure)
            if bfit["status"] != "ok":
                failures += 1
                continue
            for metric in METRICS:
                boot[metric].append(float(bfit[metric]))
        failure_rate = failures / replicates if replicates else 1.0

        for metric in METRICS:
            low, high = ci(boot[metric])
            rows.append({
                "route": "mixedlm",
                "target": target,
                "structure": structure,
                "metric": metric,
                "estimate": fit[metric],
                "ci_low": low,
                "ci_high": high,
                "bootstrap_failures": failures,
                "bootstrap_failure_rate": failure_rate,
                "status": "estimable" if failure_rate <= 0.5 else "high_failure_rate",
                "inference_unit": INFERENCE_PC,
            })
        sd_low, sd_high = ci(boot["grid_avg_random_effect_sd"])
        eta_ranges[target] = {
            "estimate": fit["grid_avg_random_effect_sd"],
            "ci_low": sd_low, "ci_high": sd_high,
            "structure": structure, "failure_rate": failure_rate,
        }
    return pd.DataFrame(rows), eta_ranges


def _na_row(target, metric, structure, status) -> dict:
    return {
        "route": "mixedlm", "target": target, "structure": structure, "metric": metric,
        "estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan,
        "bootstrap_failures": np.nan, "bootstrap_failure_rate": np.nan,
        "status": f"not_estimable:{status}", "inference_unit": INFERENCE_PC,
    }


def floor_route(labels: pd.DataFrame, seed: int, replicates: int) -> pd.DataFrame:
    """Discomfort floor robustness: between-participant SD of floor-crossing
    probability and of max discomfort, with participant bootstrap CI."""
    rows = []

    def between_sd(sample: pd.DataFrame, fn) -> float:
        per = sample.groupby("participant_id", group_keys=False).apply(fn, include_groups=False)
        return float(np.std(per.to_numpy(dtype=float), ddof=1))

    specs = {
        "between_participant_sd_p_discomfort_gt0":
            lambda g: float((pd.to_numeric(g["discomfort"], errors="coerce") > 0).mean()),
        "between_participant_sd_mean_discomfort":
            lambda g: float(pd.to_numeric(g["discomfort"], errors="coerce").mean()),
        "between_participant_sd_max_discomfort":
            lambda g: float(pd.to_numeric(g["discomfort"], errors="coerce").max()),
        "between_participant_sd_p_high_discomfort":
            lambda g: float(g["high_discomfort"].mean()),
    }
    for name, fn in specs.items():
        boot = cluster_bootstrap(
            labels[["participant_id", "discomfort", "high_discomfort"]].copy(),
            lambda s, fn=fn: between_sd(s, fn),
            seed=seed + abs(hash(name)) % 9973, replicates=replicates,
        )
        rows.append({
            "route": "floor_robustness", "target": "discomfort", "structure": "empirical",
            "metric": name, "estimate": boot["estimate"], "ci_low": boot["ci_low"],
            "ci_high": boot["ci_high"], "bootstrap_failures": boot["bootstrap_failures"],
            "bootstrap_failure_rate": np.nan, "status": "estimable", "inference_unit": INFERENCE_P,
        })
    return pd.DataFrame(rows)


def choose_optimum(group: pd.DataFrame, target: str, direction: str) -> dict:
    opposite = "discomfort" if target == "relaxation" else "relaxation"
    values = pd.to_numeric(group[target], errors="coerce")
    optimum = values.max() if direction == "max" else values.min()
    tied = group[np.isclose(values, optimum)].copy()
    if target == "relaxation":
        tied = tied.sort_values(["discomfort", "condition"],
                                key=lambda s: s.map(condition_sort_key) if s.name == "condition" else s)
    else:
        tied = tied.sort_values(["relaxation", "condition"], ascending=[False, True],
                                key=lambda s: s.map(condition_sort_key) if s.name == "condition" else s)
    primary = tied.iloc[0]
    return {
        "primary_condition": str(primary["condition"]),
        "tied_conditions": ";".join(sorted(tied["condition"].astype(str), key=condition_sort_key)),
        "primary_value": float(primary[target]),
        "opposite_value": float(primary[opposite]),
    }


def individual_optima(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for participant, group in labels.groupby("participant_id"):
        group = group.sort_values("condition", key=lambda s: s.map(condition_sort_key))
        relax = choose_optimum(group, "relaxation", "max")
        disc = choose_optimum(group, "discomfort", "min")
        rows.append({
            "participant_id": participant,
            "relaxation_primary_condition": relax["primary_condition"],
            "relaxation_tied_conditions": relax["tied_conditions"],
            "relaxation_primary_value": relax["primary_value"],
            "discomfort_primary_condition": disc["primary_condition"],
            "discomfort_tied_conditions": disc["tied_conditions"],
            "discomfort_primary_value": disc["primary_value"],
            "inference_unit": INFERENCE_P,
        })
    return pd.DataFrame(rows)


def winners_curse_null(labels: pd.DataFrame, target: str, seed: int, sims: int) -> dict:
    """Compare observed modal-optimum share to a shared-surface + noise null.

    Under the null, every participant shares the group cell means; the only
    source of individual-optimum dispersion is residual noise.  A *lower*
    observed modal share than the null would indicate genuine heterogeneity.
    """
    direction = "max" if target == "relaxation" else "min"
    group_means = labels.groupby("condition")[target].mean().reindex(CONDITIONS)
    resid = labels[target].to_numpy(float) - labels["condition"].map(group_means).to_numpy(float)
    resid_sd = float(np.std(resid, ddof=1))
    n_part = labels["participant_id"].nunique()

    # observed modal share
    observed = individual_optima(labels)
    col = f"{target}_primary_condition"
    obs_counts = Counter(observed[col])
    obs_modal = max(obs_counts.values()) / n_part

    rng = np.random.default_rng(seed)
    null_modal = []
    mean_vec = group_means.to_numpy(float)
    for _ in range(sims):
        shares = []
        for _ in range(n_part):
            draw = mean_vec + rng.normal(0.0, resid_sd, size=len(CONDITIONS))
            idx = int(np.argmax(draw)) if direction == "max" else int(np.argmin(draw))
            shares.append(CONDITIONS[idx])
        counts = Counter(shares)
        null_modal.append(max(counts.values()) / n_part)
    null_modal = np.asarray(null_modal)
    null_low, null_high = ci(null_modal)
    # p = P(null produces modal share <= observed): small p => observed share is
    # unusually concentrated (homogeneous); large => observed dispersion is
    # within what noise alone yields (no evidence for heterogeneity).
    p_more_dispersed = float(np.mean(null_modal <= obs_modal))
    return {
        "target": target,
        "observed_modal_share": obs_modal,
        "null_modal_share_mean": float(np.mean(null_modal)),
        "null_modal_share_ci_low": null_low,
        "null_modal_share_ci_high": null_high,
        "residual_sd": resid_sd,
        "p_observed_at_least_as_dispersed_as_null": p_more_dispersed,
    }


def run(seed: int, replicates: int) -> dict:
    paths = ensure_dirs()
    labels = load_labels()

    mixed, eta_ranges = mixedlm_route(labels, seed, replicates)
    floor = floor_route(labels, seed, replicates)

    null_rows = []
    for target in PRIMARY_TARGETS:
        null_rows.append(winners_curse_null(labels, target, seed=seed + 11, sims=max(2000, replicates)))
    null = pd.DataFrame(null_rows)
    null_long = null.melt(id_vars="target", var_name="metric", value_name="estimate")
    null_long["route"] = "winners_curse_null"
    null_long["structure"] = "shared_surface_plus_noise"
    null_long["ci_low"] = np.nan
    null_long["ci_high"] = np.nan
    null_long["inference_unit"] = INFERENCE_P

    components = pd.concat([mixed, floor, null_long], ignore_index=True)
    write_csv(paths["reports"] / "variance_components_final.csv", components)

    optima = individual_optima(labels)
    write_csv(paths["reports"] / "individual_optima.csv", optima)

    _write_personalization(paths, mixed, floor, null, eta_ranges)

    return {
        "components": add_contract_columns(components, INFERENCE_PC),
        "eta_ranges": eta_ranges,
        "null": null,
        "optima": optima,
    }


def _write_personalization(paths, mixed, floor, null, eta_ranges) -> None:
    sd_rows = mixed[mixed["metric"] == "grid_avg_random_effect_sd"]
    lines = []
    for _, r in sd_rows.iterrows():
        if str(r["status"]).startswith("not_estimable"):
            lines.append(f"- **{r['target']}**（{r['structure']} 协方差）：{r['status']}。")
        else:
            lines.append(
                f"- **{r['target']}**（{r['structure']} 协方差）：grid-averaged random-effect "
                f"SD = {format_number(r['estimate'])}，95% CI = {format_ci(r['ci_low'], r['ci_high'])}"
                f"（bootstrap 失败率 {format_number(r['bootstrap_failure_rate'], 2)}）。"
            )
    floor_lines = [
        f"- {r['metric']}：{format_number(r['estimate'])}，95% CI = {format_ci(r['ci_low'], r['ci_high'])}。"
        for _, r in floor.iterrows()
    ]
    null_lines = []
    for _, r in null.iterrows():
        null_lines.append(
            f"- **{r['target']}**：观测 modal-optimum share = {format_number(r['observed_modal_share'], 3)}；"
            f"共享曲面+噪声零模型 share = {format_number(r['null_modal_share_mean'], 3)} "
            f"(95% CI {format_ci(r['null_modal_share_ci_low'], r['null_modal_share_ci_high'], 3)})；"
            f"P(零模型至少和观测一样分散) = {format_number(r['p_observed_at_least_as_dispersed_as_null'], 3)}。"
        )

    eta_text_parts = []
    for target, info in eta_ranges.items():
        eta_text_parts.append(
            f"{target}: SD≈{format_number(info['estimate'])} "
            f"(CI {format_ci(info['ci_low'], info['ci_high'])})"
        )
    eta_text = "；".join(eta_text_parts) if eta_text_parts else "不可估计"

    text = f"""# I.2 个性化是否值得（personalization warranted）

> 推断单位：参与者级（n=15）/ participant×condition（n=135）。证据等级 **A**（描述性方差分解），关于"个性化能否改善真实结果"的因果声称为 **C 级**，须由未来 MRT 验证。

## 结论（一句话）

与 Phase A 一致：**以 15 名参与者，无法判定个性化是否值得**（`indeterminate_with_15_participants`）。
本节把这一不确定性**量化为一个可信区间**，并直接作为 II.3 异质性敏感性地图中参数 `η` 的可信范围来源。

## 路线 1：MixedLM 异质性（grid-averaged random-effect SD）

预声明的可忽略异质性阈值：SD ≤ {NEGLIGIBLE_HETEROGENEITY_SD:.2f}（0–1 量表）。
relaxation 采用 **diagonal（独立随机斜率）** 协方差以解决 Phase A 中 full 协方差的不收敛。

{chr(10).join(lines)}

解读：若 SD 的 95% CI **下界 > {NEGLIGIBLE_HETEROGENEITY_SD:.2f}**，异质性排除可忽略；若 CI **跨越**该阈值，则证据不足以判定。

## 路线 2：discomfort 地板适配（异质性是否为地板假象）

discomfort 有 ~64% 标签位于地板（=0）。若仅看连续 MixedLM，地板可能伪造异质。下列参与者间 SD 直接刻画"跨越地板的倾向"本身是否异质：

{chr(10).join(floor_lines)}

只要 floor-crossing 概率 `P(discomfort>0)` 的参与者间 SD 明显大于 0，discomfort 的异质性就**不是**纯地板假象，而反映真实的个体风险倾向差异。

## 路线 3：赢家诅咒 / argmax 零模型

即便所有人共享同一群体曲面，**噪声本身**也会让各人的 argmax 最优条件看起来分散。下表把观测 modal-optimum share 与"共享曲面+残差噪声"零模型对比：

{chr(10).join(null_lines)}

解读：若观测 modal share 落在零模型 CI 之内（P 值不极端），则当前观测到的"个体最优分散"**与噪声无法区分**——不能据 argmax 分散声称异质性。这是对赢家诅咒的显式校正。

## 喂给 II.3 的 `η` 可信范围

把 MixedLM 的 grid-averaged random-effect SD 的 95% CI 作为仿真异质参数 `η` 的可信区间：**{eta_text}**。
II.3 的敏感性地图将在此区间上叠加，明确显示"自适应收益 > 0 所需的最小 `η*`"落在该区间的哪一侧——而现有数据无法进一步收紧。

## 可证伪条件

- 若未来更大样本把 relaxation 的 SD CI 下界推到 > {NEGLIGIBLE_HETEROGENEITY_SD:.2f}，则"个性化值得"得到正证据；
- 若 floor-crossing SD 的 CI 收缩到接近 0，则 discomfort 异质性被推翻为地板假象；
- 若观测 modal share 显著低于零模型，则 argmax 分散反映真实异质。
"""
    write_text(paths["reports"] / "personalization_warranted.md", text)


if __name__ == "__main__":
    out = run(seed=20260627, replicates=300)
    print(out["components"][out["components"]["metric"] == "grid_avg_random_effect_sd"].to_string(index=False))
    print(out["null"].to_string(index=False))

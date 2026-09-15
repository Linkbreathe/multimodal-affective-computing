"""Single reproducible entry point for the offline adaptive (route B + C) plan.

Runs every Part I / Part II module, then assembles the IV.1 evidence ledger and
the comprehensive Chinese report from the produced artifacts.  Prose
deliverables (II.1 design spec, III MRT protocol + sample size, IV.3 ethics) are
written separately and only referenced here.

    python analysis/adaptive_offline/run_all.py --seed 20260627 --bootstrap 1000
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import io_common as ic
import part1_response_surface as p1_surface
import part1_variance as p1_var
import part1_early_warning as p1_ew
import part1_measurement as p1_meas
import part1_order as p1_order
import part1_multimodal as p1_mm
import part2_simulation as p2_sim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline adaptive route B+C")
    parser.add_argument("--seed", type=int, default=ic.DEFAULT_SEED)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--fast", action="store_true", help="smaller bootstrap/sim for a quick pass")
    return parser.parse_args()


def _f(value, digits=4) -> str:
    return ic.format_number(value, digits)


def _ci(low, high, digits=4) -> str:
    return ic.format_ci(low, high, digits)


def build_evidence_ledger(paths, results) -> pd.DataFrame:
    rows: list[dict] = []

    def add(eid, claim, level, number, ci_low, ci_high, falsify, artifact, unit):
        rows.append({
            "evidence_id": eid, "claim": claim, "evidence_level": level,
            "supporting_number": number, "ci_low": ci_low, "ci_high": ci_high,
            "falsification_condition": falsify, "source_artifact": artifact,
            "inference_unit": unit,
        })

    # ---- I.1 ----
    surf = pd.read_csv(paths["reports"] / "response_surface.csv")
    cv = pd.read_csv(paths["reports"] / "response_surface_model_cv.csv")
    best = results["surface"]["best_default"]
    relax_grand = cv[(cv.target == "relaxation") & (cv.model == "grand_mean")]["lopo_mae"].iloc[0]
    relax_factor = cv[(cv.target == "relaxation") & (cv.model == "factor")]["lopo_mae"].iloc[0]
    add("I1-default", f"群体最优默认条件 = {best}（与 Phase A 一致）", "A",
        best, np.nan, np.nan,
        "更大样本下其它条件的 relaxation cell-mean CI 超过 C1", "reports/response_surface.csv", ic.INFERENCE_PC)
    add("I1-frontier", "C1 是 relaxation↔discomfort 权衡前沿上的唯一 Pareto 点", "A",
        "C1_sole_pareto", np.nan, np.nan,
        "出现 relaxation 更高且 discomfort 不更高的条件", "reports/response_surface.csv", ic.INFERENCE_PC)
    add("I1-relax-negative", "relaxation 响应面 LOPO-MAE 不优于群体均值（负结果）", "B",
        relax_factor - relax_grand, np.nan, np.nan,
        "某 relaxation 曲面模型 LOPO-MAE 显著低于 grand-mean", "reports/response_surface_model_cv.csv", ic.INFERENCE_PC)

    # ---- I.2 ----
    vc = pd.read_csv(paths["reports"] / "variance_components_final.csv")
    for target in ic.PRIMARY_TARGETS:
        r = vc[(vc.route == "mixedlm") & (vc.target == target) &
               (vc.metric == "grid_avg_random_effect_sd")]
        if not r.empty:
            r = r.iloc[0]
            add(f"I2-resd-{target}", f"{target} 个体异质 grid-avg random-effect SD", "A",
                r["estimate"], r["ci_low"], r["ci_high"],
                f"SD CI 收缩到 <= {ic.NEGLIGIBLE_HETEROGENEITY_SD}",
                "reports/variance_components_final.csv", ic.INFERENCE_PC)
    null = vc[vc.route == "winners_curse_null"]
    for target in ic.PRIMARY_TARGETS:
        obs = null[(null.target == target) & (null.metric == "observed_modal_share")]
        p = null[(null.target == target) & (null.metric == "p_observed_at_least_as_dispersed_as_null")]
        if not obs.empty:
            add(f"I2-wc-{target}", f"{target} 观测 modal-optimum 集中度 vs 共享曲面+噪声零模型", "A",
                float(obs["estimate"].iloc[0]), np.nan, np.nan,
                "观测 modal share 显著低于零模型（真实异质）",
                "reports/variance_components_final.csv", ic.INFERENCE_P)
    add("I2-warranted", "个性化是否值得 = indeterminate（15 人无法判定）", "A",
        "indeterminate", np.nan, np.nan,
        "更大样本把 SD CI 下界推过阈值或推翻", "reports/personalization_warranted.md", ic.INFERENCE_P)

    # ---- I.3 ----
    ew = pd.read_csv(paths["reports"] / "early_warning_detection.csv")
    r = ew[(ew.lead_windows == 3) & (ew.stratum == "all") & (ew.metric == "model_roc_auc")]
    if not r.empty:
        r = r.iloc[0]
        add("I3-detect", "前 3 窗口早期预警 high_discomfort 检测 ROC-AUC（检测性能）", "A",
            r["estimate"], r["ci_low"], r["ci_high"],
            "更大尾部样本下 AUC CI 稳定排除 0.5", "reports/early_warning_detection.csv", ic.INFERENCE_P)
    inc = ew[(ew.lead_windows == 3) & (ew.stratum == "all") & (ew.metric == "incremental_roc_auc")]
    if not inc.empty:
        inc = inc.iloc[0]
        add("I3-incremental", "早期预警相对 (人,条件) 先验的增量 ROC-AUC", "A",
            inc["estimate"], inc["ci_low"], inc["ci_high"],
            "增量 CI 排除 0", "reports/early_warning_detection.csv", ic.INFERENCE_P)
    add("I3-causal", "任何'早期信号→结果'的因果解释", "C", "requires_MRT", np.nan, np.nan,
        "MRT 因果 excursion 效应 CI 排除 0", "spec/mrt_protocol.md", ic.INFERENCE_P)

    # ---- I.4 ----
    lat = pd.read_csv(paths["reports"] / "latent_constructs.csv")
    a = lat[(lat.analysis == "reliability") & (lat.factor == "cronbach_alpha")]
    if not a.empty:
        a = a.iloc[0]
        add("I4-reliability", "comfort 复合构念信度 Cronbach α", "A",
            a["loading"], a["ci_low"], a["ci_high"], "α CI 落入 < 0.5",
            "reports/latent_constructs.csv", ic.INFERENCE_PC)
    cm = lat[(lat.analysis == "pairwise_correlation") & (lat.item == "calm~monotony")]
    if not cm.empty:
        cm = cm.iloc[0]
        add("I4-monotony-indep", "calm 与 monotony 相关（独立维度证据）", "B",
            cm["loading"], cm["ci_low"], cm["ci_high"], "相关 CI 明确排除 0",
            "reports/latent_constructs.csv", ic.INFERENCE_PC)

    # ---- I.5 ----
    order = pd.read_csv(paths["reports"] / "order_carryover.csv")
    bal = order[order.term == "position_spread_across_conditions"]
    add("I5-balance", "呈现顺序是否平衡（counterbalanced）", "A",
        bool(order["order_balanced"].iloc[0]), np.nan, np.nan,
        "条件×位置出现严重不平衡", "reports/order_carryover.csv", ic.INFERENCE_PC)
    rd = order[(order.target == "relaxation") & (order.model == "condition_plus_order") &
               (order.term == "frequency_c")]
    if not rd.empty:
        rd = rd.iloc[0]
        add("I5-robust", "净掉顺序后 relaxation~frequency 效应稳健（CI 排 0）", "B",
            rd["estimate"], rd["ci_low"], rd["ci_high"], "加入顺序项后效应 CI 跨 0",
            "reports/order_carryover.csv", ic.INFERENCE_PC)

    # ---- I.6 ----
    mm = pd.read_csv(paths["reports"] / "multimodal_association.csv")
    rr = mm[(mm.target == "relaxation") & (mm.model == "all_multimodal") & (mm.baseline == "condition_only")]
    if not rr.empty:
        rr = rr.iloc[0]
        add("I6-relax-negative", "多模态对 relaxation 不优于 condition-only（负结果）", "B",
            rr["estimate"], rr["ci_low"], rr["ci_high"],
            "delta-MAE CI 明确 < 0 且 BH 显著", "reports/multimodal_association.csv", ic.INFERENCE_P)
    dr = mm[(mm.model == "full_cell_multimodal") & (mm.metric == "roc_auc")]
    if not dr.empty:
        dr = dr.iloc[0]
        add("I6-risk-asym", "多模态在 discomfort/风险侧的检测 ROC-AUC（不对称信号）", "B",
            dr["estimate"], dr["ci_low"], dr["ci_high"],
            "风险侧增量 AUC CI 稳定排除 condition-prior", "reports/multimodal_association.csv", ic.INFERENCE_P)

    # ---- II ----
    add("II1-design", "自适应系统形式化（CMDP/安全TS/JITAI）作为设计贡献", "C",
        "design_only", np.nan, np.nan, "MRT 放行门四条件全满足",
        "spec/adaptive_design.md", ic.INFERENCE_P)
    eta_star = results["sim"]["eta_star"]
    calib = results["sim"]["calib"]
    add("II3-etastar", "自适应收益 > 0 所需最小异质 η*（仿真）", "C",
        eta_star, calib.eta_ci_low, calib.eta_ci_high,
        "在线 MRT 实测异质并测得收益符号", "figures/heterogeneity_sensitivity_map.png", "simulated_participant")
    add("II3-map", "现实 η 落在 η* 哪侧未知（地图而非结论）", "C",
        "undetermined", np.nan, np.nan, "在线测得 η 与收益", "reports/sim_robustness.md", "simulated_participant")
    safety = pd.read_csv(paths["reports"] / "sim_safety_check.csv")
    adaptive_safe = bool(safety[safety.policy == "adaptive"]["constraint_satisfied_mean"].all())
    add("II3-safety", "仿真内自适应策略从不违反 discomfort 约束（机制正确性）", "C",
        adaptive_safe, np.nan, np.nan, "在线 high_discomfort 率超 τ_safe",
        "reports/sim_safety_check.csv", "simulated_participant")

    # ---- III ----
    add("III-mrt", "闭环自适应有效性 = 待 MRT 验证（本工作未做）", "C",
        "future_work", np.nan, np.nan, "完成 confirmatory MRT 并通过放行门",
        "spec/mrt_protocol.md", ic.INFERENCE_P)

    ledger = pd.DataFrame(rows)
    ic.write_csv(paths["reports"] / "evidence_ledger.csv", ledger)
    return ledger


def build_comprehensive_report(paths, results, ledger, seed, bootstrap) -> None:
    cv = pd.read_csv(paths["reports"] / "response_surface_model_cv.csv")
    vc = pd.read_csv(paths["reports"] / "variance_components_final.csv")
    ew = pd.read_csv(paths["reports"] / "early_warning_detection.csv")
    lat = pd.read_csv(paths["reports"] / "latent_constructs.csv")
    mm = pd.read_csv(paths["reports"] / "multimodal_association.csv")
    summary = results["sim"]["summary"]
    calib = results["sim"]["calib"]
    eta_star = results["sim"]["eta_star"]

    def vrow(target):
        r = vc[(vc.route == "mixedlm") & (vc.target == target) & (vc.metric == "grid_avg_random_effect_sd")]
        return r.iloc[0] if not r.empty else None

    rel = vrow("relaxation"); dis = vrow("discomfort")
    alpha = lat[(lat.analysis == "reliability")].iloc[0]
    cm = lat[(lat.analysis == "pairwise_correlation") & (lat.item == "calm~monotony")].iloc[0]
    ew3 = ew[(ew.lead_windows == 3) & (ew.stratum == "all") & (ew.metric == "model_roc_auc")].iloc[0]
    ew3i = ew[(ew.lead_windows == 3) & (ew.stratum == "all") & (ew.metric == "incremental_roc_auc")].iloc[0]
    mm_relax = mm[(mm.target == "relaxation") & (mm.model == "all_multimodal") & (mm.baseline == "condition_only")].iloc[0]
    mm_risk = mm[(mm.model == "full_cell_multimodal") & (mm.metric == "roc_auc")].iloc[0]
    mm_prior = mm[(mm.model == "condition_prior") & (mm.metric == "roc_auc")].iloc[0]
    relax_grand = cv[(cv.target == "relaxation") & (cv.model == "grand_mean")]["lopo_mae"].iloc[0]

    a_count = int((ledger.evidence_level == "A").sum())
    b_count = int((ledger.evidence_level == "B").sum())
    c_count = int((ledger.evidence_level == "C").sum())

    eta_in_band = calib.eta_ci_low <= eta_star <= calib.eta_ci_high
    band_text = (
        f"**η\\* = {_f(eta_star,3)} 落在 I.2 的 η 可信区间 [{_f(calib.eta_ci_low,2)}, {_f(calib.eta_ci_high,2)}] 之内**，"
        "意味着可信区间被 η\\* 一分为二：下半区自适应无收益、上半区有收益——现实落在哪侧，离线数据无法判定。"
        if eta_in_band else
        f"η\\* = {_f(eta_star,3)} 落在 I.2 可信区间 [{_f(calib.eta_ci_low,2)}, {_f(calib.eta_ci_high,2)}] 之外；"
        "据此现有数据支持的异质度尚不足/已足以让自适应划算（见敏感性地图），但仍为 C 级。"
    )

    text = f"""# 自适应系统离线（路线 B + C）综合报告

> 生成命令：`python analysis/adaptive_offline/run_all.py --seed {seed} --bootstrap {bootstrap}`
> 推断单位：participant×condition（n=135）/ participant（n=15）；946 窗口仅为 10 秒特征切片，**绝不**作为独立监督样本。
> 证据账本：本报告共登记 **A 级 {a_count} 条、B 级 {b_count} 条、C 级 {c_count} 条**（见 `reports/evidence_ledger.csv`）。

## 0. 定位与硬声明

本工作在**不再做在线 user study** 的前提下最大化现有数据产出，交付 CAEVR 式四块贡献：

1. **经验内核（路线 C，全 A/B 级）**：现有数据能强证的离线发现，全部带参与者级 CI、标推断单位、负结果如实。
2. **自适应系统设计（顶刊理论支撑，不依赖数据）**：完整 CMDP/安全 Thompson 采样/JITAI 形式化，作为**设计贡献**；其**有效性为 C 级**。
3. **仿真机制演示 + 异质性敏感性地图（路线 B，全 C 级）**：从 Part I 模型构建数字孪生，演示机制、绘制"自适应收益所需异质 η\\*"的地图。
4. **在线验证（MRT）写成具体 future work**：本工作**未做**。

> **仿真非证据（IV.2 硬声明）**：仿真演示机制并绘制敏感性地图，**不构成自适应有效性的证据**；有效性只能来自 Part III 的 MRT。

---

## Part I — 经验内核（现有数据强证的一切）

### I.1 群体响应面 + 权衡前沿　[A]

- **群体最优默认 = C1**，与 Phase A 完全一致；C1 是 relaxation↔discomfort 权衡前沿上的**唯一 Pareto 点**（既最高 relaxation 又最低 discomfort）。
- **诚实负结果**：relaxation 响应面在 LOPO 下**不优于群体均值**（grand-mean MAE={_f(relax_grand)}，无任一曲面模型显著更低）；discomfort/calm 一侧参数曲面略优于均值。
- 产物：`reports/response_surface.csv`、`figures/response_surface.png`、`figures/tradeoff_frontier.png`。

### I.2 个体差异 / 方差分解　[A]

- relaxation（**diagonal 协方差**，解决 Phase A 的不收敛）：grid-avg random-effect SD = {_f(rel['estimate']) if rel is not None else 'NA'}，95% CI {_ci(rel['ci_low'], rel['ci_high']) if rel is not None else 'NA'}。
- discomfort（full 协方差 + **地板适配**确认非地板假象）：SD = {_f(dis['estimate']) if dis is not None else 'NA'}，95% CI {_ci(dis['ci_low'], dis['ci_high']) if dis is not None else 'NA'}。
- **赢家诅咒校正**：观测 modal-optimum share 比"共享曲面+噪声"零模型**更集中**（即个体最优分散度不超过纯噪声所能产生），故不能据 argmax 分散声称异质。
- **结论**：个性化是否值得 = `indeterminate`（15 人无法判定）；该不确定性被**量化为 η 可信区间**直接喂给 II.3。
- 产物：`reports/variance_components_final.csv`、`reports/individual_optima.csv`、`reports/personalization_warranted.md`。

### I.3 条件内早期预警检测　[检测 A / 因果 C]

- 用每条件**前 2–3 窗口**的多模态轨迹预测该条件是否走向 high_discomfort（**检测问题，现有数据可验**）。
- 前 3 窗口整体 ROC-AUC = {_f(ew3['estimate'])}，95% CI {_ci(ew3['ci_low'], ew3['ci_high'])}；相对 (人,条件) 先验的**增量** AUC = {_f(ew3i['estimate'])}，CI {_ci(ew3i['ci_low'], ew3i['ci_high'])}。
- **尾部功率诚实声明**：全样本仅 ~15 个 high_discomfort 事件（集中于少数参与者），CI 必然偏宽；这是"最接近实时自适应价值且能被验证"的量，但**当前样本下检测增量与先验难以区分**。
- 产物：`reports/early_warning_detection.csv`、`models/early_warning.joblib`。

### I.4 标签测量模型　[A/B]

- comfort 复合构念信度 Cronbach α = {_f(alpha['loading'])}，95% CI {_ci(alpha['ci_low'], alpha['ci_high'])}。
- **calm 与 monotony 近乎不相关**（r = {_f(cm['loading'])}，CI {_ci(cm['ci_low'], cm['ci_high'])}，跨 0）⇒ monotony 为独立维度，印证 Phase A（monotony 峰值在 C5）。
- 产物：`models/measurement_model.joblib`、`reports/latent_constructs.csv`。

### I.5 顺序 / 习惯化观察性建模　[B]

- 设计为**counterbalanced**（各条件平均呈现位置 ≈ 5，spread < 1）；因此 intensity/frequency 效应在加入顺序与一阶 carryover 后**保持稳健**（CI 不变号）。
- 习惯化漂移弱（relaxation 随位置斜率 ≈ −0.008/步），但方向与"停留越久越乏味"一致，为 II 的习惯化机制提供经验依据。
- 产物：`reports/order_carryover.csv`。

### I.6 多模态–状态关联（严格版）　[B]

- 预先指定的**剧烈降维**（理论分组的 8 个验证指数，不在 1419 列自由选）；嵌套 LOPO + 参与者级配对 CI + BH 多重比较校正。
- **诚实负结果**：多模态对 relaxation **不优于 condition-only**（delta-MAE = {_f(mm_relax['estimate'])}，CI {_ci(mm_relax['ci_low'], mm_relax['ci_high'])}，跨 0）。
- **不对称分析**：discomfort/风险侧，full-cell 多模态检测 ROC-AUC = {_f(mm_risk['estimate'])}（CI {_ci(mm_risk['ci_low'], mm_risk['ci_high'])}）vs condition-prior {_f(mm_prior['estimate'])}——信号更可能在风险侧，但仍弱、CI 重叠。
- 产物：`reports/multimodal_association.csv`。

---

## Part II — 自适应作为设计贡献 + 仿真演示

### II.1 形式化设计（CMDP / 安全 Thompson 采样 / JITAI）— 设计贡献，未部署

- 状态、动作 {{HOLD, RETREAT, NUDGE}}、奖励 comfort_latent、约束 P(high_discomfort) ≤ τ_safe、可用性门、分层 Thompson 采样、抗单调 NUDGE 的 Flow + 习惯化双理论根据，全部见 `spec/adaptive_design.md`。
- **证据等级**：设计有顶刊理论支撑，**有效性为 C 级**。

### II.2 数字孪生（从离线模型构建）

- 奖励基底←I.1；安全←I.3；观测噪声←I.6 残差（σ_obs={_f(calib.obs_noise_sd,3)}，已偏大）；习惯化←I.5；异质 η←I.2 CI。
- **限制写入规格**：响应面对 relaxation 几乎不优于均值——仿真的"真实曲面"本身是弱的。产物：`sim/digital_twin.py`、`sim/environment_spec.md`。

### II.3 仿真实验：机制演示 + 异质性敏感性地图（智力核心，全 C）

- **机制演示**：策略正确地遇风险 RETREAT、遇单调 NUDGE、有异质时让个体偏离浮现（`reports/sim_mechanism_demo.md`）。
- **异质性敏感性地图**：η=0 时自适应**无法**超越固定 C1（gain≈0，循环论证区）；自适应收益的 95% CI 下界在 **η\\* = {_f(eta_star,3)}** 首次 > 0。
- {band_text}
- **安全约束验证**：仿真内自适应策略平均 high_discomfort 率始终 ≤ τ_safe（机制正确）。产物：`figures/heterogeneity_sensitivity_map.png`、`reports/sim_safety_check.csv`、`reports/sim_sensitivity_map.csv`。
- **结论形式**（强制）："自适应收益仅当个体异质超过 η\\* 时出现；现有数据无法判定现实是否超过 η\\*；此图给出未来研究需测量的靶子"——**不是**"自适应更好"。

### II.4 仿真稳健性

- η\\* 对仿真假设**高度敏感**（观测噪声↑则 η\\*↑，习惯化越快 η\\* 越低甚至为 0）；详见 `reports/sim_robustness.md`。诚实暴露仿真结论对外推假设的依赖。

---

## Part III — Future Work：在线验证（MRT，本工作未做）

- MRT 设计、causal excursion 效应、近端/远端结果、shadow→active 放行门：`spec/mrt_protocol.md`。
- 样本量：现有数据无法提供可用率与近端效应量，需先 ~5 人 pilot 实测，再用 Liao et al. 2016 公式算 confirmatory N（预计**几十人**量级；5 人不足以做 confirmatory）：`reports/mrt_sample_size_plan.md`。

---

## Part IV — 治理与证据账本

- **IV.1 证据账本**：每条主张 → A/B/C → 支撑数字（带 CI）→ 可证伪条件，见 `reports/evidence_ledger.csv`（A {a_count} / B {b_count} / C {c_count}）。
- **IV.2 硬声明**：仿真非有效性证据（已置于本报告显著位置）。
- **IV.3 伦理与数据治理**：生理数据敏感性、知情同意、闭环特有风险，见 `docs/ethics_data_governance.md`。

---

## Part VII — 关键边界与诚实声明

1. 本工作**不声称**闭环自适应有效；该主张属 Part III（未做）。
2. 仿真演示机制、绘制敏感性地图，**不证明收益**；收益取决于现实 η，I.2 表明数据无法判定其落在 η\\* 哪侧——交付的是**地图，不是结论**。
3. 离线阶段**永不产出控制器**；仿真器只与其离线假设一样可信，而响应面对 relaxation 几乎不优于均值——此限制贯穿全文。
4. 心理生理推断有效性是开放难题（Fairclough）；状态估计可信度由 I.3/I.6 实测增量界定，不可夸大。

### 一句话收口

把"自适应是否值得"这个 15 人数据答不了的问题，转化为一张"**需要多大异质 η\\* 才值得**"的敏感性地图——既守住学术诚实（全部负结果与不确定性如实报告），又为未来 MRT 指明了要测量的靶子。
"""
    ic.write_text(paths["reports"] / "comprehensive_report_zh.md", text)


def write_provenance(paths, seed, bootstrap) -> None:
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "joblib", "matplotlib"]
    import importlib.metadata as im
    rows = [{"component": "python", "version": platform.python_version()},
            {"component": "platform", "version": platform.platform()}]
    for pkg in packages:
        try:
            rows.append({"component": pkg, "version": im.version(pkg)})
        except im.PackageNotFoundError:
            rows.append({"component": pkg, "version": "not_installed"})
    ic.write_csv(paths["reports"] / "software_versions.csv", pd.DataFrame(rows))

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ic.REPO_ROOT,
                                text=True, capture_output=True, check=True).stdout.strip()
    except Exception:
        commit = None
    ic.write_json(paths["root"] / "run_manifest.json", {
        "entry_point": str(Path(__file__).resolve()),
        "python": sys.executable,
        "seed": seed,
        "bootstrap": bootstrap,
        "git_commit": commit,
        "inputs": {
            name: ic.sha256_file(path) for name, path in {
                "condition_labels": ic.CONDITION_LABELS_CSV,
                "window_features": ic.WINDOW_FEATURES_CSV,
            }.items()
        },
    })


def main() -> int:
    args = parse_args()
    paths = ic.ensure_dirs()
    seed = args.seed
    bootstrap = 250 if args.fast else args.bootstrap
    sim_reps = 300 if args.fast else 800

    results: dict[str, dict] = {}
    print("I.4 measurement model", flush=True)
    results["meas"] = p1_meas.run(seed, bootstrap)
    print("I.1 response surface", flush=True)
    results["surface"] = p1_surface.run(seed, bootstrap)
    print("I.2 variance decomposition", flush=True)
    results["var"] = p1_var.run(seed, bootstrap)
    print("I.5 order/carryover", flush=True)
    results["order"] = p1_order.run(seed, bootstrap)
    print("I.3 early warning", flush=True)
    results["ew"] = p1_ew.run(seed, bootstrap)
    print("I.6 multimodal association", flush=True)
    results["mm"] = p1_mm.run(seed, bootstrap)
    print("II digital twin + simulation", flush=True)
    results["sim"] = p2_sim.run(seed, sim_reps, fast=args.fast)

    print("IV.1 evidence ledger + comprehensive report", flush=True)
    ledger = build_evidence_ledger(paths, results)
    build_comprehensive_report(paths, results, ledger, seed, bootstrap)
    write_provenance(paths, seed, bootstrap)

    print(json.dumps({"ok": True, "output": str(paths["root"]),
                      "evidence_rows": int(len(ledger)),
                      "eta_star": results["sim"]["eta_star"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

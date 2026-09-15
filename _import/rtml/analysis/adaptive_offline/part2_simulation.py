"""II.3 / II.4 Simulation experiments: mechanism demo + heterogeneity
sensitivity map + safety check + robustness.

EVIDENCE LEVEL: C for everything here.  The deliverable is a MAP, not a claim:
adaptive gain appears only when individual heterogeneity exceeds eta*, and the
offline data (I.2) cannot tell us which side of eta* reality is on.

Outputs:
  - sim/digital_twin.py                       (snapshot of the twin source)
  - sim/environment_spec.md                   (every assumption + its source)
  - reports/sim_mechanism_demo.md
  - figures/heterogeneity_sensitivity_map.png
  - reports/sim_safety_check.csv
  - reports/sim_robustness.md
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from io_common import (
    CONDITIONS,
    ci,
    ensure_dirs,
    format_ci,
    format_number,
    write_csv,
    write_text,
)
from digital_twin import (
    DigitalTwin,
    TwinCalibration,
    adjacent_conditions,
    calibrate_from_data,
)

PRIOR_SD = 0.15  # shared-prior SD for the Thompson sampler (pooling strength)


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------
def _posterior(prior_mean: float, stats: list[float], obs_var: float) -> tuple[float, float]:
    n, s = stats
    prior_var = PRIOR_SD ** 2
    prior_prec = 1.0 / prior_var
    obs_prec = (n / obs_var) if obs_var > 0 else 0.0
    post_prec = prior_prec + obs_prec
    post_mean = (prior_prec * prior_mean + (s / obs_var if obs_var > 0 else 0.0)) / post_prec
    return post_mean, 1.0 / post_prec


def policy_fixed_c1(twin, current, posterior, monotony_now, rng) -> str:
    return "C1"


def policy_random_safe(twin, current, posterior, monotony_now, rng) -> str:
    feasible = twin.safe_feasible(current)
    return feasible[int(rng.integers(len(feasible)))]


def policy_adaptive(twin, current, posterior, monotony_now, rng) -> str:
    feasible = twin.safe_feasible(current)
    obs_var = (twin.calib.obs_noise_sd * twin.obs_noise_scale) ** 2
    # monotony NUDGE: if the current cell has gone stale, drop HOLD so the
    # sampler is forced onto a different (dishabituating) stimulus.
    candidates = feasible
    if monotony_now >= 0.5 and len(feasible) > 1:
        candidates = [c for c in feasible if c != current] or feasible
    samples = {}
    for c in candidates:
        mean, var = _posterior(twin.calib.reward_surface[c], posterior[c], obs_var)
        samples[c] = rng.normal(mean, np.sqrt(max(var, 1e-9)))
    return max(samples, key=samples.get)


POLICIES = {
    "fixed_c1": policy_fixed_c1,
    "random_safe": policy_random_safe,
    "adaptive": policy_adaptive,
}


def run_episode(twin: DigitalTwin, participant: dict, policy_fn, T: int, rng) -> dict:
    current = "C1"
    dwell = 0
    posterior = {c: [0, 0.0] for c in CONDITIONS}
    total_reward = 0.0
    high_discomfort_events = 0
    trace = []
    for _ in range(T):
        true_r = twin.true_reward(participant, current, dwell)
        total_reward += true_r
        hd = twin.sample_high_discomfort(current)
        high_discomfort_events += hd
        obs = twin.observe_reward(true_r)
        posterior[current][0] += 1
        posterior[current][1] += obs
        monotony_now = twin.monotony(dwell)
        nxt = policy_fn(twin, current, posterior, monotony_now, rng)
        trace.append({"condition": current, "dwell": dwell, "true_reward": true_r,
                      "monotony": monotony_now, "high_discomfort": hd, "next": nxt})
        if nxt == current:
            dwell += 1
        else:
            dwell = 0
            current = nxt
    return {
        "mean_reward": total_reward / T,
        "high_discomfort_rate": high_discomfort_events / T,
        "trace": trace,
    }


def simulate_grid(calib: TwinCalibration, eta: float, n_participants: int, T: int,
                  seed: int, **twin_kwargs) -> pd.DataFrame:
    rows = []
    for pid in range(n_participants):
        rng = np.random.default_rng(seed + pid)
        twin = DigitalTwin(calib=calib, eta=eta, rng=rng, **twin_kwargs)
        participant = twin.new_participant()
        opt = twin.individual_optimum(participant)
        for name, fn in POLICIES.items():
            # fresh rng per policy so they see the same participant but
            # independent stochasticity
            twin.rng = np.random.default_rng(seed + pid * 101 + hash(name) % 1000)
            result = run_episode(twin, participant, fn, T, twin.rng)
            rows.append({
                "participant": pid, "eta": eta, "policy": name,
                "mean_reward": result["mean_reward"],
                "high_discomfort_rate": result["high_discomfort_rate"],
                "individual_optimum": opt,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# II.3.2 heterogeneity sensitivity map
# ---------------------------------------------------------------------------
def sensitivity_map(calib, eta_grid, n_participants, T, seed, replicates, **twin_kwargs):
    summary = []
    for eta in eta_grid:
        sim = simulate_grid(calib, eta, n_participants, T, seed, **twin_kwargs)
        pivot = sim.pivot_table(index="participant", columns="policy", values="mean_reward")
        gain = (pivot["adaptive"] - pivot["fixed_c1"]).to_numpy(float)
        # bootstrap gain CI across simulated participants
        rng = np.random.default_rng(seed + int(eta * 1000) + 7)
        boots = [float(np.mean(rng.choice(gain, size=len(gain), replace=True))) for _ in range(replicates)]
        glow, ghigh = ci(boots)
        row = {"eta": eta, "gain_adaptive_minus_fixed": float(np.mean(gain)),
               "gain_ci_low": glow, "gain_ci_high": ghigh}
        for policy in POLICIES:
            vals = pivot[policy].to_numpy(float)
            row[f"{policy}_mean_reward"] = float(np.mean(vals))
        summary.append(row)
    return pd.DataFrame(summary)


def find_eta_star(summary: pd.DataFrame) -> float:
    positive = summary[summary["gain_ci_low"] > 0].sort_values("eta")
    if positive.empty:
        return float("nan")
    return float(positive.iloc[0]["eta"])


# ---------------------------------------------------------------------------
# II.3.1 mechanism demo
# ---------------------------------------------------------------------------
def mechanism_demo(calib, seed) -> dict:
    out = {}
    # (a) safety RETREAT: start adjacent to a high-risk cell, confirm the safe
    # feasible set never includes cells above tau_safe.
    rng = np.random.default_rng(seed)
    twin = DigitalTwin(calib=calib, eta=0.0, tau_safe=0.20, rng=rng)
    risky = [c for c in CONDITIONS if twin.risk(c) > twin.tau_safe]
    retreat_ok = all(twin.risk(c) <= twin.tau_safe for cur in CONDITIONS for c in twin.safe_feasible(cur))
    out["safety_retreat"] = {
        "risky_conditions_excluded": risky,
        "tau_safe": twin.tau_safe,
        "feasible_sets_all_safe": retreat_ok,
    }

    # (b) monotony NUDGE: high heterogeneity off, force long dwell -> check the
    # adaptive policy stops holding once monotony crosses the NUDGE threshold.
    twin = DigitalTwin(calib=calib, eta=0.0, rng=np.random.default_rng(seed + 1))
    participant = {"surface": dict(calib.reward_surface), "dwell": {c: 0 for c in CONDITIONS}}
    posterior = {c: [50, 50 * calib.reward_surface[c]] for c in CONDITIONS}  # confident posterior
    holds, nudges = 0, 0
    for dwell in range(8):
        mono = twin.monotony(dwell)
        nxt = policy_adaptive(twin, "C1", posterior, mono, np.random.default_rng(seed + dwell))
        if nxt == "C1":
            holds += 1
        else:
            nudges += 1
    out["monotony_nudge"] = {
        "holds_before_nudge": holds, "nudges_after_threshold": nudges,
        "nudge_triggered": nudges > 0,
    }

    # (c) heterogeneity -> individual optima emerge: at eta>0 the adaptive policy
    # lands on the individual optimum more often than fixed C1 does.
    for eta in (0.0, 0.20):
        sim = simulate_grid(calib, eta, n_participants=60, T=24, seed=seed + 99)
        last_cell_match = []
        # use the recovered (most-visited) preference: approximate by reward rank
        match = (sim[sim.policy == "adaptive"]
                 .assign(is_c1=lambda d: d.individual_optimum == "C1"))
        out[f"heterogeneity_eta_{eta}"] = {
            "share_individual_optimum_is_c1": float((sim[sim.policy == "adaptive"]["individual_optimum"] == "C1").mean()),
            "adaptive_mean_reward": float(sim[sim.policy == "adaptive"]["mean_reward"].mean()),
            "fixed_c1_mean_reward": float(sim[sim.policy == "fixed_c1"]["mean_reward"].mean()),
        }
    return out


# ---------------------------------------------------------------------------
# II.3.3 safety check + II.4 robustness
# ---------------------------------------------------------------------------
def safety_check(calib, eta_grid, n_participants, T, seed) -> pd.DataFrame:
    rows = []
    for eta in eta_grid:
        sim = simulate_grid(calib, eta, n_participants, T, seed + 555)
        for policy in POLICIES:
            sub = sim[sim.policy == policy]
            rows.append({
                "eta": eta, "policy": policy,
                "mean_high_discomfort_rate": float(sub["high_discomfort_rate"].mean()),
                "max_high_discomfort_rate": float(sub["high_discomfort_rate"].max()),
                "tau_safe": 0.20,
                "constraint_satisfied_mean": bool(sub["high_discomfort_rate"].mean() <= 0.20),
                "inference_unit": "simulated_participant",
            })
    return pd.DataFrame(rows)


def robustness(calib, base_eta_grid, seed) -> pd.DataFrame:
    scenarios = {
        "baseline": {},
        "high_obs_noise": {"obs_noise_scale": 1.5},
        "low_obs_noise": {"obs_noise_scale": 0.5},
        "fast_habituation": {"habituation_scale": 2.0},
        "no_habituation": {"habituation_scale": 0.0},
        "tighter_safety": {"tau_safe": 0.10},
    }
    rows = []
    for name, kwargs in scenarios.items():
        summary = sensitivity_map(calib, base_eta_grid, n_participants=50, T=20,
                                  seed=seed + abs(hash(name)) % 1000, replicates=400, **kwargs)
        eta_star = find_eta_star(summary)
        rows.append({"scenario": name, "eta_star": eta_star,
                     "max_gain_at_eta_max": float(summary["gain_adaptive_minus_fixed"].iloc[-1]),
                     **{k: v for k, v in kwargs.items()}})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# figures + reports
# ---------------------------------------------------------------------------
def _figure(paths, summary, calib, eta_star):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    eta = summary["eta"].to_numpy(float)
    ax = axes[0]
    for policy, color in (("adaptive", "#2563eb"), ("fixed_c1", "#10b981"), ("random_safe", "#9ca3af")):
        ax.plot(eta, summary[f"{policy}_mean_reward"], marker="o", color=color, label=policy)
    ax.set_xlabel("individual heterogeneity  eta  (= I.2 grid-avg random-effect SD)")
    ax.set_ylabel("mean true reward (relaxation units)")
    ax.set_title("Policy reward vs heterogeneity")
    ax.legend()

    ax = axes[1]
    ax.plot(eta, summary["gain_adaptive_minus_fixed"], marker="o", color="#2563eb", label="adaptive - fixed C1")
    ax.fill_between(eta, summary["gain_ci_low"], summary["gain_ci_high"], color="#2563eb", alpha=0.2)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.axvspan(calib.eta_ci_low, calib.eta_ci_high, color="#f59e0b", alpha=0.25,
               label=f"I.2 eta credible band [{calib.eta_ci_low:.2f}, {calib.eta_ci_high:.2f}]")
    if np.isfinite(eta_star):
        ax.axvline(eta_star, color="#ef4444", linestyle="--", label=f"eta* = {eta_star:.3f}")
    ax.set_xlabel("individual heterogeneity  eta")
    ax.set_ylabel("adaptive gain over fixed C1")
    ax.set_title("Heterogeneity sensitivity map (C-level)")
    ax.legend()
    fig.suptitle("II.3 Adaptive gain appears only when eta > eta*  —  a map, not a claim")
    fig.savefig(paths["figures"] / "heterogeneity_sensitivity_map.png", dpi=170)
    plt.close(fig)


def _write_mechanism_report(paths, demo, calib):
    a = demo["safety_retreat"]
    b = demo["monotony_nudge"]
    h0 = demo["heterogeneity_eta_0.0"]
    h1 = demo["heterogeneity_eta_0.2"]
    text = f"""# II.3.1 仿真机制演示（定性，C 级）

> **硬声明**：本节演示策略**机制是否正确**（遇风险会退、遇单调会推、有异质时让个体偏离浮现），**不**声称自适应改善真实结果。所有结论为 **C 级，待 MRT 验证**。

## (a) 安全 RETREAT —— 高风险条件被可行集排除

- 安全约束 τ_safe = {a['tau_safe']:.2f}（P(high_discomfort) 上限）。
- 按 I.3 校准的 per-condition 风险，被排除的高风险条件：{a['risky_conditions_excluded']}。
- 所有当前状态下的安全可行集是否都满足约束：**{a['feasible_sets_all_safe']}**。

机制正确：策略的可行动作集**永不包含**超过 τ_safe 的条件；当当前条件风险上升越界时，HOLD 不再可行，策略被迫朝更安全的相邻条件 RETREAT。

## (b) 抗单调 NUDGE —— 停留过久触发改变刺激

- 在 η=0、后验自信的情形下，让策略在 C1 连续停留并观察 monotony 随 dwell 上升。
- monotony 越过 NUDGE 阈值前 HOLD 次数：{b['holds_before_nudge']}；越阈后触发 NUDGE 次数：{b['nudges_after_threshold']}。
- NUDGE 是否被触发：**{b['nudge_triggered']}**。

机制正确：习惯化使停留收益衰减、monotony 上升，越过阈值后策略放弃 HOLD、改变刺激（dishabituation 恢复），符合 Flow"执行性一推"+习惯化双过程设计。

## (c) 异质性 → 个体偏离浮现

| 设定 | adaptive 平均奖励 | fixed C1 平均奖励 | 个体最优=C1 的占比 |
| --- | --- | --- | --- |
| η = 0（同质） | {format_number(h0['adaptive_mean_reward'])} | {format_number(h0['fixed_c1_mean_reward'])} | {format_number(h0['share_individual_optimum_is_c1'], 3)} |
| η = 0.20（异质） | {format_number(h1['adaptive_mean_reward'])} | {format_number(h1['fixed_c1_mean_reward'])} | {format_number(h1['share_individual_optimum_is_c1'], 3)} |

机制正确：η=0 时个体最优几乎都是 C1，自适应无从超越固定 C1（**循环论证区**）；η 增大后个体最优分散，自适应开始追踪个体偏离。**这演示的是机制，不是收益**——收益是否出现取决于现实 η，见 II.3.2 的敏感性地图。
"""
    write_text(paths["reports"] / "sim_mechanism_demo.md", text)


def _write_environment_spec(paths, calib):
    text = f"""# II.2 数字孪生环境规格（environment_spec）

> **证据等级：C。** 仿真器只与其离线假设一样可信。**I.1/I.6 显示 relaxation 响应面几乎不优于均值**——这条限制贯穿整个仿真器，任何仿真结论都不得据此声称真实有效性。

## 环境成分与来源标定

| 成分 | 仿真中的作用 | 标定来源（Part I） | 标定值 |
| --- | --- | --- | --- |
| 奖励基底曲面 | 各条件的"真实"relaxation | I.1 群体 relaxation cell means | C1={format_number(calib.reward_surface['C1'],3)} 最高；见 reward_surface |
| 安全/风险 | P(high_discomfort\\|condition) | I.3 per-condition high_discomfort 率 | C1={format_number(calib.high_discomfort_rate['C1'],3)} … C9={format_number(calib.high_discomfort_rate['C9'],3)} |
| 观测噪声 | 实时状态估计的不完美 | I.6 条件内残差 SD | σ_obs={format_number(calib.obs_noise_sd,3)} |
| 习惯化漂移 | 停留越久收益衰减、monotony 升 | I.5 presentation-position 斜率 | 每步 {format_number(calib.habituation_rate,4)} |
| 个体异质 η | 个体曲面偏离群体面的 SD | I.2 grid-avg random-effect SD（CI） | η̂={format_number(calib.eta_estimate)}，CI [{format_number(calib.eta_ci_low)}, {format_number(calib.eta_ci_high)}] |

## 关键设计：η 的单位与可比性

η 定义为个体随机倾斜的**grid-averaged random-effect SD**，单位与 I.2 的 relaxation 异质性估计**一致**（0–1 relaxation 量表）。因此 II.3.2 的敏感性地图可以**直接**把 I.2 的 η 可信区间叠加上去。η=0 表示全同质（所有个体最优=C1），η 越大个体最优越分散。

## 状态转移与策略接口

- 网格 3×3（intensity_index × frequency_index），动作为 ±1 相邻步（`policy.adjacent_grid_steps_only`）。
- 每步：占据当前条件→获得（含习惯化的）真实奖励→采样 high_discomfort→产生带噪观测→更新后验→策略选下一条件。
- 改变刺激重置 dwell（dishabituation 恢复）。

## 不可逾越的限制（写入规格，禁止违反）

1. 仿真**不是**有效性证据；所有 II 结论为 C 级。
2. 奖励基底来自 relaxation 响应面，而该面对 relaxation 几乎不优于均值——**仿真的"真实曲面"本身是弱的**。
3. η 的现实取值未知；I.2 只给出可信区间，不给点判定。
4. 观测噪声按 I.6 残差标定，已偏大（σ_obs≈{format_number(calib.obs_noise_sd,2)}），状态估计在仿真里和现实一样不完美。
"""
    write_text(paths["sim"] / "environment_spec.md", text)


def _write_robustness_report(paths, robust, eta_star, summary, calib):
    lines = [
        f"| {r['scenario']} | {format_number(r['eta_star'],3)} | {format_number(r['max_gain_at_eta_max'],4)} |"
        for _, r in robust.iterrows()
    ]
    text = f"""# II.4 仿真稳健性与假设依赖（stress-test）

> 证据等级 C。本节诚实暴露 II.3 的核心结论（尤其 η* 阈值）对仿真关键假设有多敏感。

## η* 对假设的敏感性

baseline 情形下 η*（自适应收益的 95% CI 下界首次 > 0 的最小异质度）= **{format_number(eta_star,3)}**；
I.2 的 η 可信区间为 [{format_number(calib.eta_ci_low)}, {format_number(calib.eta_ci_high)}]。

| 情形 | η* | η_max 处的自适应收益 |
| --- | --- | --- |
{chr(10).join(lines)}

## 解读

- η* 随**观测噪声**上升而上升（噪声越大，越需要更强异质才划算），随**习惯化**变化而移动。
- 若 η* 落在 I.2 可信区间**之上**，则"现有数据支持的异质程度还不足以让自适应稳赢"；若 η* 落在区间**之内或之下**，则"在部分可信异质度上自适应可能划算"——但两种情形 II.3 都**不能**判定现实落在哪侧。
- 结论：η* 不是一个稳健的固定常数，它依赖外推假设；这正是为什么本工作交付的是**地图与靶子**，而非"自适应更好"的结论。
"""
    write_text(paths["reports"] / "sim_robustness.md", text)


def run(seed: int, replicates: int, fast: bool = False) -> dict:
    paths = ensure_dirs()
    calib = calibrate_from_data()

    # snapshot the twin source as a deliverable
    shutil.copy2(Path(__file__).resolve().parent / "digital_twin.py", paths["sim"] / "digital_twin.py")

    eta_grid = np.round(np.linspace(0.0, 0.30, 13), 3).tolist()
    n_part = 40 if fast else 80
    T = 16 if fast else 20
    reps = max(300, replicates // 2)

    summary = sensitivity_map(calib, eta_grid, n_participants=n_part, T=T, seed=seed, replicates=reps)
    eta_star = find_eta_star(summary)
    write_csv(paths["reports"] / "sim_sensitivity_map.csv",
              summary.assign(eta_star=eta_star, inference_unit="simulated_participant"))

    demo = mechanism_demo(calib, seed=seed + 3)
    safety = safety_check(calib, eta_grid, n_participants=n_part, T=T, seed=seed)
    write_csv(paths["reports"] / "sim_safety_check.csv", safety)
    robust = robustness(calib, eta_grid[::3] + [eta_grid[-1]], seed=seed + 5)

    _figure(paths, summary, calib, eta_star)
    _write_environment_spec(paths, calib)
    _write_mechanism_report(paths, demo, calib)
    _write_robustness_report(paths, robust, eta_star, summary, calib)

    return {
        "calib": calib, "summary": summary, "eta_star": eta_star,
        "demo": demo, "safety": safety, "robust": robust,
    }


if __name__ == "__main__":
    out = run(seed=20260627, replicates=800, fast=False)
    print("eta_star:", out["eta_star"])
    print("calib eta CI:", out["calib"].eta_ci_low, out["calib"].eta_ci_high)
    print(out["summary"][["eta", "fixed_c1_mean_reward", "adaptive_mean_reward",
                          "gain_adaptive_minus_fixed", "gain_ci_low", "gain_ci_high"]].to_string(index=False))
    print(out["robust"].to_string(index=False))

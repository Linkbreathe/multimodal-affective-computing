# 自适应系统 Plan — 无在线研究版本（路线 B + C）

> 本版用于：**不再做在线 user study** 的前提下，最大化现有数据的产出。
> 与原版（Master Plan）的关系：**闭环控制 + MRT + 放行判定整段移到 Future Work**；现有数据能强证的离线分析提升为**经验内核**；完整的自适应形式化**保留为设计贡献**，并配一个**仿真器演示**。
> 交付定位（CAEVR 式四块贡献）：① 经验发现（路线 C，全 A/B 级）；② 自适应系统**设计**（顶刊理论支撑，不依赖数据）；③ **仿真**机制演示 + 敏感性地图（明确非有效性证据）；④ 把在线验证（MRT）写成具体的 future work。

---

## 0. 总则与不可逾越的边界

### 0.1 红线（违反即作废）
1. **推断单位是参与者，不是窗口。** 所有 CI/显著性以 15 名参与者或 135 条 participant×condition 为单位；946 窗口只是切片。
2. **离线数据只产出先验与预测模型，不产出控制器。** 现有数据开环、无 (状态,动作,结果) 元组，任何"动作→结果"的因果效应是 **C 级**，只能由未来 MRT 验证。
3. **仿真不是证据。** 仿真演示的是"在假设的离线环境下策略会怎么动"，**不能**声称自适应改善了真实结果。所有仿真结论统一标 **C 级（待 MRT 验证）**。

### 0.2 证据三级
- **A 级**：135 点被试内数据 + 参与者级 CI 支撑。
- **B 级**：观察性/关联性/预测性假设（无因果声称）。
- **C 级**：需在线干预验证（含全部仿真结论与闭环控制逻辑）。

### 0.3 与已完成工作的衔接
Phase A 已判 `indeterminate_with_15_participants`、C1 群体最优、discomfort 地板主导。本版直接继承：C1 作默认；discomfort 当罕见风险；"个性化是否值得"在离线无法定论——本版把这个**不确定性本身**变成仿真敏感性地图的核心变量（见 II.3）。

---

# Part I — 经验内核（路线 C）：现有数据能强证的一切

**总目标**：用 135 单元 + 946 窗口，产出一组全部 A/B 级、带 CI、经得起审稿的离线发现。**不需要任何新数据。** 全程嵌套 LOPO、参与者级配对 bootstrap、阈值前置、多重比较校正。

## I.1 群体响应面 + 权衡前沿　[A 级]
- **输入**：`condition_features.csv`、`condition_labels.csv`。
- **方法**：
  1. 对 relaxation、discomfort、（及 calm 等）各拟合关于 (intensity, frequency) 的响应面；用 condition 类别因子版交叉验证，避免双线性误设把最优锁在角上。
  2. 输出带 CI 的两/多张热力图 + relaxation↔discomfort 权衡前沿 + 单一最优默认条件。
- **输出**：`reports/response_surface.csv`、`figures/response_surface.png`、`figures/tradeoff_frontier.png`。
- **退出判据**：与 Phase A 的 C1-最优一致；所有估计带 CI、标推断单位。
- **价值**：直接交付"最优默认 = C1"这个 A 级结论。

## I.2 个体差异 / 方差分解（定稿 Phase A，纳入新标签）　[A 级]
- **输入**：`condition_features.csv`、Phase A 的 `variance_components.csv`、六标签全集。
- **方法**：
  1. 分层模型 `target ~ g(i,j) + (1+i+j | participant)`，提取个体偏离的方差成分（带参与者 bootstrap CI）。
  2. 对 relaxation 用更简协方差结构（对角/独立）解决原不收敛；对 discomfort 用地板适配（先建 P(discomfort>0) 或 Tobit）确认"异质"不是地板假象。
  3. 用 I.4 的潜在构念替换单标量重跑，做稳健性。
  4. argmax 零模型校正赢家诅咒（共享曲面+噪声模拟，比观测 modal-share）。
- **输出**：`reports/variance_components_final.csv`、`reports/individual_optima.csv`、`reports/personalization_warranted.md`（结论 + CI；很可能仍 indeterminate，如实报告）。
- **价值**：回答"个性化值不值得"这个比建模更根本的问题；其结果直接喂给 II.3 的敏感性地图。

## I.3 条件内早期预警检测　[检测 A / 任何因果解释 C]
- **输入**：`window_features.csv`（按参与者分层）、discomfort 标签。
- **方法**：
  1. 用每个 Condition **前几个窗口**的多模态轨迹，预测该 Condition 是否走向 high_discomfort。**这是检测问题，不需反事实，现有数据可验。**
  2. 嵌套 LOPO、参与者级配对 CI；EEG-available/disabled 分层。
  3. 报告**增量性**（相对 (人,条件) 先验的增量 recall/AUC，CI）+ **提前量**（前 2–3 窗口能否触发）+ **尾部功率**（high_discomfort 事件够不够支撑可靠检测）。
- **输出**：`reports/early_warning_detection.csv`（含 CI、分层、增量、提前量、功率）、`models/early_warning.joblib`。
- **价值**：这是现有数据里**最接近"实时自适应价值"且能被强证**的东西——检测，而非控制。

## I.4 标签测量模型　[A/B 级]
- **输入**：`condition_labels.csv` 六个 1–7 量表项。
- **方法**：因子分析/IRT 合成潜在构念（comfort 等），报告信度；确认 calm、monotony 是否为独立维度（Phase A 提示 monotony 峰值在 C5，独立）。
- **输出**：`models/measurement_model.joblib`、`reports/latent_constructs.csv`（载荷、信度、维度结构）。
- **价值**：把最弱的"单噪声标量"升级为带信度的构念，强化 I.1/I.2 与仿真奖励。

## I.5 顺序 / 习惯化观察性建模　[B 级]
- **输入**：`condition_labels.csv` 的 `presentation_position`、`windows_elapsed`。
- **方法**：建模 relaxation/discomfort/monotony 随呈现位置的漂移 + 一阶 carryover；报告净掉顺序后条件效应是否稳健。若顺序未平衡，全部降级并标混杂。
- **输出**：`reports/order_carryover.csv`。
- **价值**：数据里最接近"动态"的观察性结构；为 II 的习惯化机制提供经验依据。

## I.6 多模态–状态关联研究（严格版）　[B 级]
- **输入**：`window_features.csv`、潜在构念。
- **方法**：预先指定的剧烈降维（理论分组/少量验证指数，不在 1419 列自由选）；嵌套 LOPO；每个模态/组合的 MAE/recall 带参与者级配对 CI；与 Condition-only、history 基线对照，报告"模型−基线"差值 CI 是否跨 0；多重比较校正。**如实报告 relaxation 几乎不优于均值这一负结果。** 单独评估多模态在 discomfort/风险一侧的增量信息（信号更可能在此）。
- **输出**：`reports/multimodal_association.csv`（含 CI、负结果、不对称分析）。
- **价值**：诚实刻画"已知条件之外，多模态还能解释多少状态"——这决定状态估计器在仿真里有多可信。

---

# Part II — 自适应作为设计贡献 + 仿真演示（路线 B）

**总目标**：保留完整的自适应系统形式化作为**设计贡献**，并用从 Part I 模型构建的**仿真器**演示其机制；所有效果性结论标 C 级。

## II.1 形式化设计（CMDP / 安全 Thompson 采样 / JITAI）—— 设计贡献，非部署系统
- **内容**（沿用原 Master Plan 的 Phase 0–2，此处作为设计呈现）：
  - **CMDP 对象**：状态 `S_t`（实时状态估计 + 上下文 + 因果聚合 + 可用性 + 池化键）、动作 `A_t∈{HOLD, RETREAT, NUDGE}`（±1 网格）、奖励 `R=comfort_latent`、约束 `Cost=P(high_discomfort)≤τ_safe`、可用性门。
  - **决策算法**：可用性门 → 安全过滤可行集 → 分层 Thompson 采样提议动作 → （在真实部署中）MRT 随机化。
  - **抗单调 NUDGE**：flow 的"执行性一推"+ 习惯化"改变刺激带来恢复"。
- **输入**：Part I 的模型 + 设计规范。
- **输出**：`spec/adaptive_design.md`（完整形式化，标注"设计贡献，未部署"）。
- **证据等级**：设计本身有顶刊理论支撑（见 Part VI），但其**有效性**为 C 级。

## II.2 数字孪生 / 仿真器（从离线模型构建环境）
- **输入**：I.1 响应面、I.3/I.6 状态与安全模型、I.5 习惯化漂移、I.4 测量模型、I.2 方差成分。
- **方法**（逐步构建仿真环境）：
  1. **环境状态转移**：以 I.1 响应面为"真实 comfort 曲面"基底；叠加 I.5 的习惯化漂移（同一条件停留越久 comfort 越衰减、monotony 越升）；叠加由 I.6/状态估计残差标定的**观测噪声**。
  2. **个体异质注入（关键）**：用一个**可调参数 `η`** 控制个体响应面偏离群体面的方差（`η=0` 全同质，`η` 大则个体最优分散）。`η` 的可信范围由 I.2 的方差成分 CI 给出。
  3. **安全反应**：高强度/高频条件按 I.3 安全模型抬高 high_discomfort 概率。
  4. **状态估计噪声**：仿真里的"实时状态估计"按 I.3/I.6 实测的不确定性加噪，使仿真状态估计与真实一样不完美。
- **输出**：`sim/digital_twin.py`、`sim/environment_spec.md`（每个环境假设的来源与标定）。
- **证据等级**：C 级。**仿真器只与其离线假设一样可信**，而离线响应面对 relaxation 几乎不优于均值——这条限制必须写进 environment_spec。

## II.3 仿真实验：机制演示 + 异质性敏感性地图（本路线的智力核心）
- **输入**：`digital_twin.py` + II.1 策略。
- **方法**：
  1. **机制演示**（定性，C 级）：在仿真里运行自适应策略，展示它**正确地**：遇 discomfort 风险上升就 RETREAT、单调上升就 NUDGE、有异质时让个体偏离浮现。这演示的是**机制对不对**，不是有没有效。
  2. **异质性敏感性地图**（核心，C 级）：把"自适应策略 vs 固定 C1 vs 随机"的相对收益，**画成异质参数 `η` 的函数**。
     - 因为若仿真最优是 C1 且 `η=0`，自适应不可能赢过固定 C1（循环论证）；收益只在 `η` 足够大时出现。
     - 故产出一条曲线/地图：**"自适应收益 > 0 所需的最小 `η`"**，并把 I.2 给出的 `η` 可信区间叠加上去——明确显示**现实落在哪一侧未知**。
  3. **安全约束验证**（仿真内）：确认策略在仿真里从不违反 discomfort 约束（安全探索的机制正确性）。
- **输出**：`reports/sim_mechanism_demo.md`、`figures/heterogeneity_sensitivity_map.png`、`reports/sim_safety_check.csv`。
- **证据等级**：全部 C 级。**结论形式必须是**："自适应收益仅当个体异质超过 `η*` 时出现；现有数据无法判定现实是否超过 `η*`；此图给出未来研究需测量的目标"——而**不是**"自适应更好"。

## II.4 仿真稳健性与假设依赖（stress-test）
- **输入**：`digital_twin.py`。
- **方法**：对仿真的关键假设做敏感性——响应面形状、噪声水平、习惯化速率、状态估计质量——看 II.3 的结论（尤其 `η*` 阈值）对这些假设有多敏感。
- **输出**：`reports/sim_robustness.md`。
- **价值**：诚实暴露"仿真结论有多依赖外推假设"，防止把脆弱的仿真当强结论。

---

# Part III — Future Work：在线验证（MRT，本工作未做）

> 本段完整保留原 Master Plan 的在线验证设计，但明确标注**为验证路径，非本工作内容**。任何"经过验证的闭环 adaptive"主张都落在这里。

## III.1 MRT 设计
- 每个 available 决策点，当 TS 提议非 HOLD 的 discretionary 动作时以概率 `p_t` 执行、否则 HOLD；硬安全回退确定性执行不随机化。
- 估计量 = causal excursion effect（执行提议 vs 保持 对近端结果）。
- 近端=未来 K 窗口状态变化；远端=条件末问卷。

## III.2 样本量（pilot → 算 N → confirmatory）
- 现有数据无法提供"可用率"与"近端效应量"——先用 ~5 人 **pilot** 实测这两个数 + 跑通安全运行与日志契约（pilot 不估因果效应）。
- 再把实测值喂 Liao et al. 2016 公式算 confirmatory N。
- **现实预期**：因近端效应大概率小、动作空间异质、且参与者级聚类推断与池化先验都需要足够多 cluster，confirmatory N 预计落在**几十人**量级，可能每人多会话；**5 人不足以做 confirmatory**。

## III.3 因果 excursion 效应分析
- 用 Boruvka/Qian 加权-中心化估计因果效应及其调节，参与者级聚类 SE。

## III.4 shadow→active 放行门（预注册，全满足才放行）
1. discretionary 动作近端因果效应 CI 排除 0 且为正；
2. 在线 high_discomfort 率 ≤ 阈值且不劣于固定 C1；
3. 整体 comfort 不劣于固定 C1；
4. 人工批准。

---

# Part IV — 治理与证据账本

## IV.1 证据账本
- 每条主张 → A/B/C → 支撑数字（带 CI）→ 可证伪条件。
- **典型分布**：I.1/I.2/I.4 多为 A；I.3 检测性能 A、其因果解释 C；I.5/I.6 为 B；II 全部 C。
- **输出**：`reports/evidence_ledger.csv`。

## IV.2 硬声明：仿真非证据
- 在论文/报告中显著位置声明：仿真演示机制并绘制敏感性地图，**不构成自适应有效性的证据**；有效性证明只能来自 Part III 的 MRT。

## IV.3 伦理与数据治理
- 生理数据敏感、可能无意暴露医学状况；访问受知情同意约束（Fairclough）。
- **输出**：`docs/ethics_data_governance.md`。

---

# Part V — 阶段闸门与交付物清单

| 部分 | 核心交付 | 闸门 |
| --- | --- | --- |
| Part I 经验内核 | `reports/{response_surface, variance_components_final, early_warning_detection, latent_constructs, order_carryover, multimodal_association}` + 图 + 模型 | 全部带 CI、标推断单位、负结果如实、证据等级标注 |
| Part II 设计+仿真 | `spec/adaptive_design.md`、`sim/digital_twin.py`、`figures/heterogeneity_sensitivity_map.png`、`reports/sim_{mechanism_demo,robustness,safety_check}` | 仿真假设标来源；结论以 `η*` 阈值形式呈现；全标 C 级 |
| Part III Future Work | `spec/mrt_protocol.md`、`reports/mrt_sample_size_plan.md` | 明确标"未做、为验证路径" |
| Part IV 治理 | `reports/evidence_ledger.csv`、伦理文档 | 仿真非证据声明就位 |

---

# Part VI — 文献支撑映射

- **支撑经验内核（Part I）的方法学**：分层/混合效应（部分汇合）；嵌套 CV + 参与者级 bootstrap；测量模型（因子分析/IRT）。
- **支撑自适应设计（Part II.1）**：JITAI（Nahum-Shani et al., *Annals of Behavioral Medicine* 2018；*Health Psychology* 2015）；CMDP（Altman 1999）；安全 RL（García & Fernández, *JMLR* 2015；Achiam et al., *ICML* 2017）；分层 Thompson 采样 + intelligent pooling（Liao et al., *Proc. ACM IMWUT* 2020；Tomkins et al. 2019）；为何离线不够（Kumar et al., *NeurIPS* 2020；Fujimoto et al., *ICML* 2019）；biocybernetic loop（Fairclough, *Interacting with Computers* 2009；Pope et al., *Biological Psychology* 1995）。
- **支撑奖励/约束/习惯化理论**：Flow（Csikszentmihalyi 1975/1990）；习惯化双过程（Groves & Thompson, *Psychological Review* 1970；Rankin et al., *Neurobiology of Learning and Memory* 2009）。
- **支撑未来验证（Part III）**：MRT（Klasnja et al., *Health Psychology* 2015；Qian et al., *Psychological Methods* 2022；Boruvka et al., *JASA* 2018；Qian et al., *Biometrika* 2021；Liao et al., *Statistics in Medicine* 2016；Dempsey et al., *Annals of Applied Statistics* 2020）。
- **领域先例**：Gupta et al. 2024 *IEEE TVCG*（CAEVR——同样以"设计 + 实验框架"为主、预测模型较弱而被顶会接受）。

---

# Part VII — 关键边界与诚实声明

1. **本工作不声称"闭环自适应有效"。** 该主张属 Part III（未做）。本工作交付的是经验发现（A/B）+ 自适应设计（C 级有效性）+ 仿真机制演示与敏感性地图（C）。
2. **仿真演示机制、绘制敏感性地图，不证明收益。** 自适应是否真有收益取决于现实中的个体异质 `η`，而 I.2 表明现有数据无法判定 `η` 落在 `η*` 哪一侧——本工作给出的是**地图，不是结论**。
3. **离线阶段永不产出控制器**；仿真器只与其离线假设一样可信，而该假设（响应面）对 relaxation 几乎不优于均值——这条限制贯穿全文。
4. **心理生理推断有效性是开放难题**（Fairclough）；状态估计的可信度由 I.3/I.6 的实测增量与不确定性界定，不可夸大。

---

### 一句话收口
本版把 plan 重构为**"现有数据强证的经验内核（路线 C，全 A/B）+ 完整自适应设计作为贡献 + 仿真机制演示与异质性敏感性地图（路线 B，全 C）+ MRT 写成 future work"**——它不依赖任何新数据、学术上诚实、且仍是一份完整可交付、可投顶会的工作；其智力核心是把"自适应是否值得"这个数据答不了的问题，转化成一张"需要多大异质才值得"的敏感性地图，既守住诚实，又给未来研究指明了要测量的靶子。

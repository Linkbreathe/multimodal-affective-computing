# 多模态信号 × 建模 联合审阅报告

**生成时间:** 2026-07-02 17:52 WEST
**分析者:** Claude (Fable 5)
**数据:** data_collection_v3,15 名被试(P002–P016),每人 9 个 Condition(3×3 强度×频率),
每个 Condition 结构:`全局Baseline(45s)` → 每条件 `PreConditionBaseline(20s)` → `ConditionViewing(70s)` → `QuestionnaireBreak(评分)`
**方法:** 相位从 marker 流(`WaterLiliesExperimentMarkers`,`unix_time_ms`)切分;ECG 用 neurokit2 重算以规避管线检测器缺陷;
"共通性"= 跨被试同号数 + 跨 9 条件同号数 + Wilcoxon 配对检验。

---

## 0. 一句话结论

> **在这套实验里,唯一稳健、跨人通用的"刺激起始"生理反应是心率下沉(HR↓);行为模态里,eye 只有"注视范围扩大"这一条通用信号,head 在去掉时长混杂后无通用模式;而 egocentric video 是唯一既能通用区分"起始"、又能通用随强度/频率分级的模态——但它主要是在重新捕捉视觉刺激本身。EEG/ECG-HRV 在条件层面几乎无判别力。当前 ML/DL 训练机制是干净的,但模型打不过平凡基线,根因是"条件对生理/行为状态的净驱动太弱、被个体差异淹没",而不是提取坏了。**

---

## 1. 数据质量与可用性(先确定哪些能用)

| 模态 | 可用被试 | 说明 |
|---|---|---|
| **EEG** | **9/15** | P002/P005/P006/P010/P014/P016 被 participant-level 硬禁用;P004(0.71)、P012(0.94)部分可用。coverage 门限饱和(好被试恒 1.0),实为二值"有/无"。 |
| **ECG** | 15/15(信号) | 但**管线 R 波检测器有缺陷**(见 §2),HR/HRV 不可信;用 neurokit2 可救回。 |
| **Eye / Head / Video** | **15/15,全部 coverage=usable=1.0** | 无需剔除任何人。质量非常干净。 |

> 行为三模态质量全绿,和你的预期一致。

---

## 2. ECG 提取缺陷(须修)

- 管线 `detect_r_peaks`(单一全局静态阈值 `median+3·MAD` + 280ms 不应期,无自适应/无 RR 校正)在真实记录上**漏拍(RR 翻倍)+ 不应期边缘重复检**。
- 后果:窗口级 `ecg_hr_bpm` 全队中位 110–125bpm、多人 >150;HRV `RMSSD 130–320ms / SDNN 146–250ms`(正常静息 RMSSD≈20–50ms),**高出 3–5 倍**。
- QC 抓不到:`ecg_usable` 只查 RR∈35–220bpm,漏拍/误检仍落在带内 → quality 恒 1.0。
- **交叉验证(P007,干净 120s):** 管线 162 拍/81bpm/RMSSD174 vs neurokit2 133 拍/66.5bpm/RMSSD103。管线多检 ~22%。
- **建议:** 换 neurokit2 `ecg_peaks`(或加局部阈值 + RR 一致性校正);`ecg_usable` 判据改为"检测器自洽性/F1",而非 RR 带宽。**在此之前,所有 `ecg_hrv_*` 特征应从建模中剔除或降权;HR 仅作"偏高但趋势可用"处理。**

---

## 3. 相位对比:PreConditionBaseline → ConditionViewing 的通用"起始反应"

> 每个 Condition 减去它自己紧邻的 20s PreConditionBaseline。**注意时长混杂:** cond 段 70s vs prebase 20s,
> 所有 `_range`/累积量(`eye_scanpath_deg`、`head_position_range`、video `_range`)会机械性变大 →
> **下表只报"时长稳健"统计量(mean/median/std/iqr/fraction)。**

### 3.1 生理(前几轮,7 个 EEG 可用被试)
- **HR:7/7 被试、8/9 条件下沉**(中位 −1.9bpm,Wilcoxon p=0.001)。**唯一通用的生理起始标记**——注意性/副交感心率减速。
- EEG 各频带/相对功率/比值/谱熵/Hjorth、HRV:跨人符号=抛硬币,**无通用模式**。

### 3.2 行为(本轮,全部 15 被试;时长稳健特征)

| 模态 | 通用起始特征 | 被试同号 | 条件同号 | 中位Δ | 解读 |
|---|---|---|---|---|---|
| **Video** | texture_mean/median ↑ | 14/15 | 9/9 | +1.5~1.8 | 画面更"有细节"(呈现 Water Lilies) |
| | brightness/texture/optical_flow/saturation/colorfulness/scene_change **_std ↑** | 14–15/15 | 9/9 | — | 观看期画面变异更大 |
| | optical_flow **mean/median ↓** | 14/15 | — | −0.6~−0.8 | 平均帧间运动更小(稳定凝视,偶发大扫视) |
| **Eye** | **direction_dispersion_deg ↑** | 13/15 | 9/9 | +3.5° | **注视空间范围扩大**(探索整幅画,唯一稳健眼动通用信号) |
| **Head** | — | ≤8/15 | — | — | **去掉时长混杂后无任何通用模式**;`head_position_range↑` 纯属 70s>20s 的假象 |

**坏特征告警:** `eye_gaze_on_painting_fraction` 全体恒为 0 → 该字段未被记录/未填充,当前是**死特征**,而它本应是最有信息量的眼动特征(是否在看画)。建议排查采集端。

---

## 4. 9 个 Condition 之间:剂量反应(强度/频率)

> 每个特征在 9 个条件上对 intensity / frequency 求 Spearman,统计跨 15 人同号数(偶然 ≥12/15 ≈ 3.5%)。

### 4.1 生理:**无**
- EEG/ECG 特征对强度/频率的相关跨人符号在随机水平;方差分解显示 **40–91% 是被试间方差,强度/频率各自只解释 1–6%(≈噪声底)**。9 个条件在生理层面几乎不可区分。

### 4.2 行为:**video 是唯一强、通用的剂量反应模态**

| 特征 | ~强度 | ~频率 | 同号 |
|---|---|---|---|
| video_edge_mean/median ↑ | ρ +0.65/+0.68 | ρ +0.51/+0.57 | **15/15** |
| video_sharpness_mean/median ↑ | ρ +0.58 | ρ +0.65/+0.69 | **15/15** |
| video_colorfulness_mean ↓ | ρ −0.48 | ρ −0.62 | 13–15/15 |
| video_optical_flow_std ↓ | ρ −0.53 | ρ −0.35 | 14/15 |
| eye/head 各项 | 弱(ρ≈0.2) | 弱 | ≤12/15 |

**关键解读:** 实验的 `natural_modulation`(marker 里的 `applied_intensity_value`/`applied_frequency_value`/`*_depth`)是在**调制视觉刺激本身**——强度/频率越高,渲染画面越"锐/多边缘/少饱和"。头戴相机把这套操纵**原样拍了回来**。
- **好消息:** video 是**唯一**能跨人、跨条件稳健编码实验操纵的模态(EEG/ECG 做不到)。
- **警示:** 这在很大程度上是 video 在**复刻刺激/标签本身**,而非被试的"反应/状态"。用于预测主观 relaxation/discomfort 时,它更像**标签代理**;其价值取决于"视觉操纵是否驱动了评分"。建模时应显式区分 video 的"刺激复刻成分"和"行为反应成分",否则会高估其对主观状态的预测力。

---

## 5. ML / DL 训练与数据/标签分布审阅

### 5.1 机制上是干净的(不用改)
- 无泄漏:所有预处理(缺失过滤/中位插补/方差&相关过滤/StandardScaler/SelectKBest 互信息)封装在 sklearn Pipeline,只在训练折 fit。
- 嵌套 LOPO 正确;残差建模(target = 其他被试的条件均值 + 物理残差)合理。
- DCNN:外层 LOPO + 内层再留出一个被试组做 early-stopping,scaler 只在 core 训练集 fit。规范。

### 5.2 结果:**模型打不过平凡基线,不可部署**(实测)
| 目标 | 模型 MAE | 条件均值基线 | 历史基线(该人上一条件) | 结论 |
|---|---|---|---|---|
| relaxation | 0.186 | **0.183** | 0.211 | 输给条件均值 |
| discomfort | 0.164 | 0.162 | **0.124** | 输给历史基线 |

`deployable=False`,两门全失败。Spearman 仅 0.11/0.13。

### 5.3 需要修改的点(按重要性)
1. **改框定:纯 LOPO → 个性化/在线校准。** 证据:用每人前 3 条件 de-bias,relaxation MAE 0.186→**0.170**,Spearman 0.11→**0.38**。残差里 31%(relax)/54%(discomfort)是**被试 offset**,LOPO 结构上看不到 → 群体模型注定吃力。个性化校准应成为主路径。
2. **discomfort 风险被错误框定为稀有事件。** 高不适(≥0.5)仅 15/135(11%),且集中在 P016=5/P005=3/…,9 人为 0。0.5 阈值在均值 0.12、std 0.21 上是 +1.8SD。LOPO 下近乎不可学(recall 0.8 但 precision 0.13)。→ 下调阈值/改连续+排序目标/只在个性化下报风险,别再当部署硬门。
3. **缺 EEG 的被试污染 `full` 模型。** 55/135 行 EEG 全缺,中位插补成常数。→ EEG/physio 变体应限定 9 个可用被试(但 `expected_labels=135` 硬校验需放开)。
4. **各模态边际价值从未折外评估。** LOPO 指标只对 `full` 算一次;`no_eeg`/`eeg_only`/`behavior_only` 只在最后 fit,不参与折外评估;config 的 `feature_groups`(5 组)与代码变体(3 个)对不上。→ **逐 feature_group 各跑一遍 LOPO**,才能回答"physio/EEG/video 各值多少"。
5. **特征/样本比失衡 + 缺基线归一。** 1419 特征 / 135 行;K 上限到 80(训练仅 126 行)。→ 砍噪声聚合项、K≤10–15。更关键:**特征侧没做被试内基线归一**——physio/behavior 的绝对水平 40–91% 是被试方差,LOPO 下无法泛化。应加"每条件特征 − 其 PreConditionBaseline"(或对全局 Baseline 做 z)。**而这些 baseline 行当前根本没进 `windows.csv`(管线只保留 ConditionViewing)。**

---

## 6. 各模态"能用性"总排名(用于模态取舍)

| 模态 | 起始通用信号 | 条件剂量反应 | 提取质量 | 建模价值判断 |
|---|---|---|---|---|
| **Video(egocentric)** | 强(texture↑/flow↓等) | **强,15/15** | 15/15 干净 | **最高——唯一能通用区分条件**;但需剥离"刺激复刻"成分 |
| **ECG-HR** | HR↓(通用) | 无 | 需换检测器 | 中——好的"状态/起始"标记,非"刺激强度"标记 |
| **Eye** | 注视范围扩大(13/15) | 弱 | 15/15 干净 | 中低;修好 `gaze_on_painting` 后可能提升 |
| **EEG** | 无通用 | 无 | 仅 9/15 | 低(群体);仅适合 per-subject / 个体化 |
| **ECG-HRV** | 无 | 无 | 检测器坏 | 最低,先修再谈 |
| **Head** | 无(去混杂后) | 弱 | 15/15 干净 | 低 |

---

## 7. 建议的下一步(按性价比)

1. **给 LOPO 循环加"逐 feature_group 折外评估"**(context_only / no_eeg / eeg_only / behavior_only / video / full),第一次量化各模态真实边际价值——直接回答你最初的"physio 是不是限制因素"。
2. **把个性化校准提为主评估口径**,并对 EEG/physio 变体放开只用 9 个可用被试。
3. **修 ECG 检测器**(neurokit2),重算 HRV 后再评估。
4. **把三相位 + 基线归一固化进管线**(补 `windows.csv` 缺失的 baseline 行),让"相对 baseline 变化量"成为可用特征。
5. **排查 `eye_gaze_on_painting` 采集**——它可能是眼动里最有价值却当前失效的特征。
6. **建模时显式处理 video 的"刺激复刻"问题**:区分"画面=刺激"与"被试反应",避免把 video 当成主观状态的强预测因子而实为标签代理。

---

*附:所有数字来自本次对 P002–P016 原始 XDF/CSV/帧的重新提取与统计;ECG 指标基于 neurokit2;相位边界来自 marker 流。中间特征表缓存于 scratchpad(`behav_P*.csv`、`phase_features.csv`)。*

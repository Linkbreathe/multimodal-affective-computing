# Foundation Model × Adaptive Controller：单受试者 500 秒回放指南

## 1. 实验动机

本实验希望把多模态 Foundation Model 与自适应控制器连接起来，验证系统是否能够根据同一受试者不断变化的生理状态，持续选择不同的 Intensity 和 Frequency，而不是长时间停留在一个固定 Condition。

Foundation Model 只输出两个实时状态：

- `relaxation`：0–1，越高表示越放松；
- `discomfort`：0–1，越高表示越不适。

控制器需要利用这两个输出、当前参数、历史反应和已经揭示的用户评分，完成以下目标：

1. 展示真正的 adaptive 行为：参数会随状态和历史改变；
2. 保证安全：discomfort 的处理优先于 relaxation；
3. 评估 Foundation Model 对 relaxation/discomfort 的预测准确性；
4. 输出约 500 秒、50 个 10 秒窗口的完整参数变化过程；
5. 区分模型预测能力、控制器行为和真实闭环效果。

本实验不预设“Intensity/Frequency 越高越好”或“越低越好”。每次变化都应被看作一次小幅、可撤销的个人试探。

---

## 2. 必须遵守的实验边界

### 2.1 Foundation Model 输入和输出

- 输入：ECG、EEG、眼动、头动、视频等模型原本需要的多模态数据；
- 输出：只能是 `relaxation` 和 `discomfort`；
- 用户问卷评分不能作为当前窗口的模型输入；
- Controller 可以单独记录当前 Intensity、Frequency 和 Condition；
- 如果 Foundation Model 本身接收了 Intensity、Frequency 或 Condition，必须额外报告这一点，因为模型输出可能直接包含参数信息，而不完全来自用户生理反应。

### 2.2 离线回放不能证明真实闭环效果

历史数据只记录了受试者在实际呈现 Condition 下的生理反应。如果 Controller 推荐了另一个 Condition，但历史下一窗口并没有真的呈现这个 Condition，就不能把下一窗口的生理变化解释为推荐动作的结果。

因此必须分别报告：

1. **Chronological replay**：按真实时间顺序输入50个窗口，Controller只产生“推荐动作”；用于评估模型准确率和adaptive决策是否会变化；
2. **Synthetic policy replay（可选）**：从该受试者已经记录的对应Condition窗口中抽取数据，模拟Controller动作被执行后的路径；必须明确标记为合成回放；
3. **Prospective closed-loop experiment**：只有在XR系统中真实执行Controller动作并重新采集生理信号，才能评价adaptive是否真正改善了用户状态。

离线实验中不能把“推荐动作后的历史下一窗口”直接写成“动作产生的效果”。

---

## 3. 受试者选择

### 3.1 冻结选择原则

必须在查看该受试者的 relaxation/discomfort 评分和模型预测误差之前完成选择，避免挑选模型表现最好的受试者。

只允许使用以下质量信息：

- 9个Condition是否完整；
- ECG是否可用；
- EEG是否通过质量检查；
- 眼动是否完整；
- 头动是否完整；
- 视频是否完整；
- 时间同步和窗口数量是否正常；
- 多模态共同有效窗口比例；
- 问卷标签是否齐全，但不能查看标签高低。

### 3.2 建议受试者

优先使用已经按纯质量规则冻结选择的 `P009`。在现有质量记录中，P009具有9个Condition标签和完整的ECG、EEG、眼动、头动、视频覆盖。

在另一台电脑上重新确认上述质量条件。如果P009缺少必要文件或无法完成50窗口推理，则按照完全相同的纯质量规则选择质量得分最高的下一名受试者，并记录替换原因。禁止根据预测准确率或用户评分选择受试者。

输出一个选择记录：

```text
participant_id
selection_time
selection_rule
modalities_available
n_valid_common_windows
labels_complete
selection_used_outcomes = false
```

---

## 4. 需要准备的数据

### 4.1 每个10秒窗口的模型输入

按照 Foundation Model 原有格式准备：

- participant ID；
- timestamp、window index；
- ECG特征或原始窗口；
- EEG特征、embedding或原始窗口；
- 眼动特征或原始窗口；
- 头动特征或原始窗口；
- 视频特征、embedding或视频窗口；
- 各模态质量和缺失标记；
- 当前真实呈现的Condition、Intensity和Frequency，仅供审计与Controller记录。

所有模态必须使用同一个10秒时间区间。

### 4.2 Foundation Model逐窗口输出

至少保存：

```text
participant_id
window_id
timestamp
pred_relaxation
pred_discomfort
model_version
modalities_used
missing_modalities
signal_valid
```

如果模型能够提供不确定性，可以额外保存上下界；但不能为了本实验修改模型，使其输出新的心理标签。

### 4.3 Ground truth目标

需要收集或连接：

- 每个Condition结束后的用户 `relaxation` 评分；
- 每个Condition结束后的用户 `discomfort` 评分；
- pleasantness；
- calm/activated；
- monotony；
- visual comfort：Too weak / Appropriate / Too strong；
- Condition开始和结束时间。

统一把7分量表转换为0–1：

```text
normalized_score = (raw_score - 1) / 6
```

如果一个Condition只有一个结束评分，而其中有多个10秒窗口，则这些窗口共享同一个Condition级标签。计算窗口指标时，必须令每个窗口权重为：

```text
sample_weight = 1 / 该Condition有效窗口数
```

不能把同一个Condition评分被复制到多个窗口后，当作多个相互独立的ground-truth标签。

---

## 5. Baseline与初始校准

Baseline数据尽量从50窗口正式回放之外获得，避免用测试窗口同时定义阈值和评价结果。

### 5.1 生理baseline

收集约60秒无参数变化的稳定数据，用于：

- 检查各模态是否有效；
- 完成个人生理归一化；
- 记录自然生理波动；
- 检查Foundation Model输出是否异常跳变。

### 5.2 安全Condition baseline

在最低Intensity、最低Frequency的C1下收集约120秒Foundation Model输出。

计算：

```text
R0 = baseline relaxation中位数
D0 = baseline discomfort中位数
epsilon_R = baseline中 |R - R0| 的90百分位数
epsilon_D = baseline中 |D - D0| 的90百分位数
```

`epsilon_R` 和 `epsilon_D` 是这个用户的自然输出波动。后续不使用统一的0.1或0.2作为变化阈值。

如果历史数据不足120秒，应使用全部可用的独立baseline/C1窗口，并明确记录实际窗口数；不能复制窗口补足样本。

### 5.3 初始参数试探

如果数据允许，依次进行：

1. `C1 → C2`：只增加Frequency；
2. `C2 → C5`：只增加Intensity。

每个试探至少观察三个10秒窗口。Controller只使用当时已经产生的Foundation Model输出，不能提前读取该受试者后续评分。

---

## 6. 最终 Adaptive Rules

### 6.1 时间规则

- Foundation Model每10秒输出一次R/D；
- Controller使用最近三个输出的中位数，形成30秒状态；
- 参数变化后至少观察30秒再做普通判断；
- 一个Condition正常情况下最多保持60秒，即6个窗口；
- 恢复状态和信号失效状态可以超过60秒；
- 一次只能改变Intensity或Frequency中的一个；
- 每次只能改变一个等级；
- 禁止Low直接跳到High；
- 禁止同时提高Intensity和Frequency。

### 6.2 信号质量规则

如果当前窗口关键模态无效：

1. 不根据该窗口改变参数；
2. 不将该窗口写入个人偏好历史；
3. 连续三个窗口无效时，回到最近验证安全的Condition；
4. 信号恢复后重新观察三个窗口。

信号异常不能被解释为discomfort。

### 6.3 安全规则

以下任一条件成立时进入恢复状态：

- 最近30秒的平滑 `D >= 0.5`；
- D连续两个有效窗口上升，并且总上升超过 `epsilon_D`；
- 改变参数后，平滑D相对改变前增加超过 `epsilon_D`；
- 已揭示的用户discomfort评分达到4/7或更高；
- 用户报告Too strong。

恢复动作：

1. 优先撤销上一次参数变化；
2. 如果D仍未恢复，每30秒向最近安全Condition再下降一个等级；
3. 每次只改变一个参数；
4. 直到D回到安全范围，并连续两个窗口没有上升；
5. 恢复期间不进行新探索。

### 6.4 上一次动作的评价

定义：

```text
delta_R = 改变后30秒平滑R - 改变前30秒平滑R
delta_D = 改变后30秒平滑D - 改变前30秒平滑D
```

按以下顺序判断：

| 条件 | 结果 | Controller处理 |
|---|---|---|
| `delta_D > epsilon_D` 或 `D >= 0.5` | Unsafe | 撤销动作并标记风险 |
| `delta_R > epsilon_R` 且 `delta_D <= epsilon_D` | Success | 保存为该用户的正响应 |
| `delta_R < -epsilon_R` 且D稳定 | Ineffective | 下次尝试反方向或另一参数 |
| R/D变化均未超过个人阈值 | Neutral | 不认为改善，但可以保留为安全的新鲜感变化 |
| R和D同时明显上升 | Mixed activation | Discomfort优先，撤销或降低负荷 |

### 6.5 下一动作选择

首先生成所有合法相邻Condition：只能有一个参数相差一级。

然后依次应用以下选择顺序，不使用任意加权分数：

1. 删除已导致高discomfort的候选；
2. 删除当前锁定的C9；
3. 优先选择过去产生Success的相邻动作；
4. 没有成功历史时，选择尚未测试的相邻动作；
5. 多个候选相同时，选择最久没有访问的Condition；
6. 冷启动平局时，优先Low/Medium Frequency和Low/Medium Intensity；
7. 尽量与上一次动作交替改变Intensity和Frequency；
8. 禁止连续选择当前相同Condition。

### 6.6 Intensity规则

- 低R不能自动触发Intensity上升；
- Intensity上升只作为一个可撤销试探；
- 用户报告activated或Too strong后，后续优先避免Intensity继续上升；
- 用户报告Too weak、D安全时，允许向上试探一级；
- 任何Intensity上升导致D明显增加时，立即撤销。

### 6.7 Frequency规则

- Low和Medium之间可以正常探索；
- Medium到High只有在D较低且没有上升趋势时允许；
- High Frequency下D一旦明显上升，优先退回Medium；
- High Frequency不能作为解决Monotony的默认动作；
- 如果某名用户多次在Frequency上升后R提高、D稳定，可以将该方向提升为个人优先动作。

### 6.8 C9锁定规则

C9默认不可选。只有同时满足以下条件才解锁：

- High Intensity已单独验证安全；
- High Frequency已单独验证安全；
- 当前D低于0.5且没有上升趋势；
- 最近揭示的用户评分不是Too strong；
- 用户最近的discomfort评分低于4/7。

即使解锁，C9也只能按普通30秒观察规则进行短期试探。

### 6.9 防止一直停留在同一Condition

- 一个安全Condition达到6个窗口后，必须选择合法相邻动作；
- 如果用户Monotony评分达到5/7或更高，可以在完成最少3个观察窗口后提前变化；
- 防无聊优先选择“安全但较久未访问”的Condition；
- 不因为Monotony直接把Frequency提高到High；
- 发生不适或信号失效时，安全和恢复优先于变化展示。

### 6.10 用户评分更新

离线回放中，只能在模拟完成一个Condition之后揭示该Condition评分。禁止在访问Condition之前使用该受试者的评分。

各评分的作用：

| 评分 | Controller用途 |
|---|---|
| Relaxation | 更新该Condition的个人长期收益 |
| Discomfort | 更新安全风险，优先级最高 |
| Calm/Activated | 修正Intensity方向偏好 |
| Too weak/Appropriate/Too strong | 修正Intensity方向并标记风险 |
| Pleasantness | 在多个安全候选间进行排序 |
| Monotony | 决定是否在30秒后提前变化 |

如果用户评分与Foundation Model输出冲突，个人评分优先更新长期偏好；同时记录冲突，供模型校准评估使用。

---

## 7. 50窗口回放步骤

### 步骤1：冻结模型和受试者

- 记录模型版本和checkpoint；
- 冻结P009或按质量规则选择替代者；
- 记录选择发生在查看目标评分和预测误差之前；
- 不在该受试者上重新训练或选择模型。

### 步骤2：选择500秒测试段

- 选择50个连续、按时间排序的10秒窗口；
- 优先从第一个正式观看窗口开始；
- 要求多模态尽可能共同有效；
- baseline/calibration窗口尽量不计入50个测试窗口；
- 如果存在间断，保存真实时间戳并报告间断，禁止把不连续窗口伪装成连续数据。

### 步骤3：逐窗口运行Foundation Model

对每个窗口按时间顺序推理并保存R/D。禁止一次读取全部预测后再为Controller挑选最理想路径。

### 步骤4：运行Chronological replay

每输入一个真实窗口：

1. 更新30秒R/D；
2. 根据Adaptive Rules给出下一Condition建议；
3. 保存建议动作、原因和安全状态；
4. 下一窗口仍然使用历史真实数据；
5. 标记推荐动作是否与历史实际Condition一致。

这一步用于评价模型准确率和Controller是否表现出adaptive，不能评价推荐动作的真实因果效果。

### 步骤5：可选Synthetic policy replay

如果希望展示一个由Controller实际驱动的50窗口Condition轨迹：

1. Controller选择下一个Condition；
2. 只从该受试者该Condition已经记录的有效窗口池中，按预先固定的顺序取下一个窗口；
3. 不允许根据预测好坏挑选窗口；
4. 窗口池耗尽后停止或报告无法继续；
5. 输出必须明确标记 `synthetic_replay = true`；
6. 不把结果描述为真实闭环生理改善。

### 步骤6：揭示评分并更新个人历史

当一个Condition模拟完成后，才允许读取该Condition的用户评分，更新安全记录、Intensity/Frequency偏好和Monotony规则。

### 步骤7：输出轨迹

生成50行的 `adaptive_replay_trace.csv`：

```text
window_id
time_s
actual_condition
controller_condition
intensity_level
frequency_level
pred_relaxation
pred_discomfort
smooth_relaxation_30s
smooth_discomfort_30s
delta_relaxation
delta_discomfort
signal_valid
controller_state
action
action_reason
previous_action_outcome
dwell_windows
c9_locked
rating_available
synthetic_replay
```

同时生成一张轨迹图：

- X轴：0–500秒；
- 第一层：predicted relaxation和discomfort；
- 第二层：Intensity等级；
- 第三层：Frequency等级；
- 标注所有参数变化、撤销、安全回退和评分揭示点。

---

## 8. 需要评估和收集的目标

### 8.1 Foundation Model准确率

分别对relaxation和discomfort报告：

- 加权MAE；
- 加权RMSE；
- Condition级预测均值与真实评分的误差；
- Spearman相关；
- 预测范围和标准差，检查是否只输出接近常数；
- 与训练参与者得到的fold-local participant-mean或condition-only baseline比较。

对于高discomfort安全识别，以真实 `discomfort >= 4/7` 为正例，报告：

- sensitivity/recall；
- specificity；
- balanced accuracy；
- precision；
- AUPRC；
- false negative数量。

如果50窗口只对应少量Condition标签，必须同时报告有效独立Condition数，不能只报告50个窗口的指标。

### 8.2 Adaptive展示程度

报告：

- 50窗口内参数变化次数；
- 访问过的不同Condition数量；
- Intensity变化次数；
- Frequency变化次数；
- 平均和最长dwell窗口数；
- 连续保持同一Condition超过6窗口的次数；
- 相邻一级变化比例，目标为100%；
- 一次只改变一个轴的比例，目标为100%；
- Success、Ineffective、Neutral、Unsafe各出现多少次；
- 撤销动作次数；
- 因Monotony提前变化次数；
- C9是否曾解锁，以及解锁依据；
- Controller是否只在两个Condition之间机械往返。

“能够展现adaptive”至少要求：

- 轨迹中存在多次由不同R/D状态或历史结果触发的参数变化；
- 不只是按照固定时间循环Condition；
- 同样的R/D状态会因为个人历史或安全记录产生不同选择；
- 有成功保留、无效换向或不适撤销中的至少两类行为；
- 没有违反单轴、相邻一级和安全优先规则。

### 8.3 安全性

报告：

- `D >= 0.5` 的窗口数；
- 高D出现后多少个窗口完成回退；
- Controller是否在D上升时继续增加负荷；
- 高风险动作被过滤多少次；
- 信号失效时是否错误适应；
- high-discomfort false negative数量；
- C9未满足条件却被选择的次数，目标为0。

### 8.4 个体化程度

报告：

- 哪些Intensity方向被记录为Success；
- 哪些Frequency方向被记录为Success；
- 哪些动作被标记为高风险；
- 用户评分揭示后，Controller选择是否发生改变；
- 模型输出与用户评分冲突次数；
- 是否出现个人证据覆盖冷启动优先级的情况。

### 8.5 效果指标

Chronological replay只能报告：

- 实际记录窗口中的平均R/D；
- Controller推荐是否随R/D变化；
- 推荐动作与已知Condition评分是否一致；
- 模型准确率和安全检测能力。

Synthetic policy replay可以额外报告：

- 合成路径的平均预测R/D；
- 合成路径访问的Condition及评分；
- 与固定C1、固定C5、固定时间轮换策略的描述性比较。

但必须明确：这些不是推荐动作真实执行后的生理因果效果。

真实闭环实验还需要收集：

- Controller实际执行的每次参数变化；
- 变化前后连续生理窗口；
- 用户实时或阶段性评分；
- 中止、不适和人工接管事件；
- 固定策略对照或随机化对照；
- 多名新受试者，而不是只评价P009。

---

## 9. 最终需要返回的文件

请从另一台电脑返回：

1. `participant_selection.json`：受试者质量选择记录；
2. `foundation_predictions_50windows.csv`：Foundation Model逐窗口R/D；
3. `adaptive_replay_trace.csv`：完整Controller轨迹；
4. `adaptive_replay_metrics.json`：准确率、adaptive、安全和个体化指标；
5. `adaptive_replay_plot.png`：500秒R/D与Intensity/Frequency变化图；
6. `adaptive_replay_report.md`：综合评估；
7. 模型版本、运行命令、随机种子和数据版本；
8. 如果运行Synthetic replay，单独保存结果并明确标记为synthetic。

`adaptive_replay_report.md`必须分别回答：

1. Foundation Model的relaxation/discomfort预测有多准确？
2. 50窗口内是否真正表现出随状态和历史变化的adaptive行为？
3. Intensity和Frequency具体如何变换，为什么变化？
4. 是否发生不安全动作或遗漏高discomfort？
5. 哪些结果属于真实历史数据，哪些属于合成回放？
6. 当前结果能支持什么，不能支持什么？

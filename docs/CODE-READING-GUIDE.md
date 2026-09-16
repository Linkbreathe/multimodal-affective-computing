# 论文主线代码阅读地图

这份文档是源码注释的导航，不替代实现本身。推荐始终带着一个问题阅读：
“当前数据在 pipeline 的哪一个阶段，它会被哪个模型使用？”

## 1. 先确定项目中的三种样本

项目里有三种容易混淆的对象：

1. `window`：一个完整的 10 秒时间窗口，用于信号处理、特征提取和实时历史。
2. `participant-condition`：一个参与者在一个实验条件下的监督样本，包含该条件的 questionnaire label。
3. `embedding token/window`：foundation encoder 对一个窗口产生的向量，formal aligned 流程会用 mask 对它们对齐。

当前论文主线的监督单位是第二种，而不是第一种。窗口是输入组成部分，不能因为数量多就当成独立标签样本。

## 2. 论文主线的调用顺序

```text
configs/project.yaml
    ↓
mac.data.index.build_index
    ↓
mac.preprocessing.pipeline.preprocess
    ├─ marker/alignment.py       对齐 marker、生成 C1-C9 边界和 10 秒窗口
    └─ data/labels.py             读取 condition-level questionnaire labels
    ↓
mac.features.extract.extract_features
    ├─ preprocessing/relax_physio.py + features/physio.py
    ├─ features/eye.py + head.py
    └─ features/video.py（可选）
    ↓
mac.data.condition_data.aggregate_window_frame
    ↓
mac.training.condition_train.train_condition_state
    ├─ models/condition_models.py  缺失值、筛选、缩放、候选模型
    ├─ participant-level LOPO       外层评估
    └─ evaluation/safety.py         是否允许 runtime 使用
    ↓
mac.realtime.engine.InferenceEngine
    └─ realtime/policy/recommender.py  只做 Shadow recommendation
```

最先完整读懂的命令链是：

```text
rtml preprocess
rtml extract-features
rtml train-state
rtml evaluate
rtml replay
```

## 3. 三条训练路线不要混在一起

### A. 当前论文主线：经典 condition-level regression

入口：`mac train-state`

- 目标：`relaxation` 和 `discomfort` 两个归一化回归目标；
- 单位：`participant-condition`；
- 外层：native 流程使用 participant-level LOPO；aligned 流程使用固定 7/1/1 split；
- 模型：condition mean baseline + residual ensemble；
- 输出：`state_model_{variant}.joblib`；
- 使用场景：当前 realtime classical backend。

阅读重点是“训练集如何产生 baseline 和 residual”，而不是先研究所有候选回归器。

### B. 论文中的神经时间模型：DCNN

入口：`rtml train-dcnn-state`

- 每个 temporal feature 有一个独立的 Conv1D stream；
- 各 stream 输出拼接后加入 intensity/frequency context；
- 最后由 MLP 输出 relaxation/discomfort；
- 训练时随机截断历史 prefix，模拟实时过程中不同长度的可用历史；
- validation 用于 early stopping，test participant 只在最终评估使用；
- 输出：`.pt` checkpoint 和对应 scaler/metrics。

阅读顺序：`models/dcnn.py` → `training/dcnn.py` → `realtime/engine.py`。

### C. formal aligned foundation/compression experiments

入口：

- `scripts/run_relax_foundation_probe.py`
- `scripts/run_relax_condition_anchor_probe.py`
- `scripts/run_relax_compression_fusion_v2.py`

这条路线处理的是严格对齐的五模态 embedding cache：EEG、ECG、eye、head、video。
它会验证 cohort、mask、输入 hash、窗口数量和 split manifest，然后在外层训练参与者上拟合 projector、fusion head 或 compression。

需要牢记：

- foundation encoder 通常已经冻结并提前生成 embedding；
- 这不是从原始信号端到端训练；
- `LateFusion` 中的权重是模型内部的组合权重，不是因果意义上的模态重要性；
- cache、mask 和 preregistration 是 formal 复现协议的一部分。

## 4. 模型结构的推荐阅读顺序

```text
ModalityProjector
    ↓
EarlyFusion / MidFusion / LateFusion
    ↓
PerceiverIO / QFormer / HEALNet
    ↓
AlignedLateRegressor / AlignedNativeFusionRegressor
    ↓
formal fold training and masks
```

先理解三种基础 fusion：

- Early：投影后直接拼接，再由共享 MLP 学习交互；
- Mid：每个模态先单独 refinement，再拼接；
- Late：每个模态经过独立 branch，再平均或加权合并。

之后再看 attention/latent-based fusion。否则很容易只记住类名，却不清楚输入是一个向量、一个窗口序列，还是带 mask 的序列。

## 5. 读代码时必须检查的防泄漏位置

每看到一个 `fit`、`mean`、`std`、`select`、`threshold` 或 `baseline`，都要问它使用的是哪一部分数据：

- scaler：只能用 outer train；
- missing/variance/correlation/MI feature selection：只能在当前训练 fold；
- condition baseline：只能使用训练参与者；
- risk threshold：只能由 inner CV 或 validation 选择；
- early stopping：只能看 validation；
- test participant：只用于最终预测和报告。

如果某个新实验绕过了这些边界，就不能直接和当前论文结果比较。

## 6. 暂时可以后读的部分

- `auxiliary/benchmarks/egoemotion/`：理解来源和 benchmark 对照；
- `auxiliary/benchmarks/seedv/`：理解 EEGPT、REVE、LoRA 的替代路径；
- `auxiliary/adaptive_control/`：不属于当前核心训练闭环；
- `auxiliary/integrations/unity/`：只有在研究 UDP/Unity 接口时再读；
- `auxiliary/merge_history/`：用于合并溯源，不是模型主线。

## 7. 建议的逐行学习顺序

1. `configs/project.yaml`
2. `src/mac/data/alignment.py`
3. `src/mac/preprocessing/relax_physio.py`
4. `src/mac/features/extract.py`
5. `src/mac/data/condition_data.py`
6. `src/mac/training/condition_train.py`
7. `src/mac/models/condition_models.py`
8. `src/mac/models/dcnn.py`
9. `scripts/run_relax_foundation_probe.py`
10. `src/mac/realtime/engine.py`
11. `src/mac/realtime/policy/recommender.py`

读完第 1--7 项，就已经能解释论文主线的数据和经典训练过程；读完第 8--11 项，才能把神经模型、formal fusion 和 Shadow runtime 串起来。

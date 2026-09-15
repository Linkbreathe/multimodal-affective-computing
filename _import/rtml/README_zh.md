# P002–P016 多模态实时放松推理

这是一个独立的 Python 3.11 项目。它把 P002–P016 的生理、眼动、头动和第一视角视频统一到 Condition 内不重叠的 10 秒窗口，并以 135 条 Painting Reflection Condition 评分训练 relaxation / discomfort 双目标状态模型。系统只运行 Shadow 推荐：Python 会把建议发给 Unity 并记录，但 Unity 不自动切换 Condition。

## 不可破坏的设计约束

- 原始数据只读；所有派生内容写入 `artifacts/` 或 `project.yaml` 指定的位置。
- 每次 Condition 变化都重置 10 秒窗口原点；每周期恰好输出一条 `StatePrediction` 和一条 `ConditionRecommendation`。
- 训练目标只来自问卷：`relaxation=(x-1)/6`、`discomfort=(x-1)/6`。生理特征绝不反向构造标签；calm、pleasantness 仅保留在源标签表中，不进入模型。
- 模型未优于 Condition-only 与历史状态基线时，推荐器强制 `hold`。
- Condition 只允许在 3×3 网格中保持或上下左右移动一级。任何缺模态、超时、坏质量或高不确定性都触发降级或 `hold`。
- 视频和 Condition 参数属于刺激上下文，不能单独冒充用户状态。

## 安装

在本目录执行：

```powershell
conda env create -f environment.yml
conda activate rtml-p002-p016
rtml --help
```

## 兼容式 run 结构（新实验）

旧 `configs/project.yaml` 与既有 `rtml` 命令保持不变，当前
`artifacts/` 是 frozen legacy 证据，不迁移也不覆盖。新实验使用
`base.yaml + experiments/*.yaml + local.yaml`；复制
[`configs/local.example.yaml`](configs/local.example.yaml) 为不提交的
`configs/local.yaml`，只写本机数据根路径和设备。

```powershell
rtml --experiment configs/experiments/runtime-classical.yaml --local-config configs/local.yaml train-state
rtml --experiment configs/experiments/runtime-classical.yaml --local-config configs/local.yaml evaluate
rtml --experiment configs/experiments/runtime-classical.yaml --local-config configs/local.yaml report
rtml --experiment configs/experiments/research-minimal-ridge.yaml --local-config configs/local.yaml benchmark-minimal-fusion
```

新 run 必须有明确 `run.id`，机器产物只写入
`artifacts/runs/<run_id>/` 下的 `manifests/`、`preprocessed/`、`features/`、
`models/`、`checkpoints/`、`metrics/`、`predictions/` 和 `logs/`。
`rtml report` 只生成 `reports/<run_id>_summary_zh.md`，并且只读取当前
run 的规范化产物。research 配置不能成为 runtime 自动后端；Shadow 和
`hold` 安全契约保持强制。

若要训练或运行 1D-CNN 后端，请先在此环境中按 PyTorch Windows CUDA 安装页选择与本机匹配的稳定 CUDA wheel，再安装项目的 `dcnn` extra。安装后必须确认 `python -c "import torch; print(torch.cuda.is_available())"` 输出 `True`；DCNN 时序最多保留 8 个 10 秒窗口（常规 Condition 为 7 个，P005/C2 有第 8 个完整窗口）。配置默认仍为 `modeling.runtime_backend: classical`，DCNN 通过 LOPO 安全门并人工切换后才会加载。

路径、参与者、通道、QC 阈值、特征开关、模型、UDP/LSL 和策略阈值全部由 [`configs/project.yaml`](configs/project.yaml) 控制。默认原始数据根目录和标签目录已经指向本机当前数据位置。

## 常用工作流

先做轻量真实数据验收：

```powershell
rtml index --participants P003,P015
rtml preprocess --participants P003,P015
rtml extract-features --participants P003,P015 --no-video
```

完整离线构建：

```powershell
rtml run-all
```

断点续跑默认复用已有全量 stage 文件；`--force` 强制重算。调试时可用 `--participants` 筛选；视频快测用 `--no-video`。注意：带参与者筛选的运行会覆盖相应 stage CSV，因此完整训练前应重新执行一次不带筛选的 `rtml run-all`。

分阶段命令：

```powershell
rtml index
rtml preprocess
rtml extract-features
rtml build-video-mp4
rtml extract-handcrafted-video
rtml train-video-ml
rtml extract-videomae2
rtml train-videomae2-dcnn
rtml report-video-fusion
rtml train-video-relaxation-ml
rtml train-videomae2-relaxation
rtml report-video-relaxation
rtml train-videomae2-video-encoder-ablation
rtml report-videomae2-video-encoder-ablation
rtml benchmark-minimal-fusion
rtml benchmark-minimal-fusion-dcnn
rtml analyze-minimal-fusion-dcnn-hp
rtml report-latest-multimodal
rtml replay-video --backend handcrafted
rtml replay-video --backend videomae2
rtml train-state
rtml train-dcnn-state
rtml train-policy
rtml evaluate
rtml replay
rtml serve
rtml adaptive-model list
rtml adaptive-model verify --bundle classical_condition_current
rtml adaptive-control
```

Adaptive Control 本地闭环推荐直接双击项目根目录的 `launch-adaptive-control.cmd`；它会先验证默认模型，再以 `rtml adaptive-control` 启动本机服务。Unity 侧只使用 `Adaptive Control > Configure` 与 `Adaptive Control > Control Panel`。正式开始前先启动 EEG/ECG LSL、戴好头显并确认 head/eye tracking，然后在控制面板点击 `Check Readiness`；只有 Python、EEG/ECG、head tracking、eye tracking 全部通过后，`Start Control` 才会启用。启动后面板会显示当前 10 秒特征窗口的倒计时。

`train-video-relaxation-ml` 与 `train-videomae2-relaxation` 是仅预测 relaxation 的离线研究对照；它们复用现有手工视频特征和冻结 VideoMAE2 嵌入，产物只写入 `artifacts/video/relaxation_only/`。该路径不预测 discomfort、不训练风险分类器，不能用于 `serve`、推荐策略或视频 Shadow replay。

`train-videomae2-video-encoder-ablation` 会在 `artifacts/video/video_encoder_ablation/` 中统一重训无视频、VideoMAE2 直连 MLP 与 VideoMAE2 时间 1DCNN；覆盖双目标和 relaxation-only、135/134 两个 cohort。报告包含 LOPO 配对误差和参与者聚类 bootstrap 区间；全部模型均为 `research_only`，不进入实时或策略路径。

## 数据产物

- `artifacts/manifests/source_manifest.*`：数据文件、当前/备份 XDF 选择和输入哈希。
- `artifacts/preprocessed/condition_labels.*`：每人每 Condition 一行，共应为 135 行。
- `artifacts/preprocessed/condition_boundaries.csv`：Unix/XDF 双时间轴及源 marker 索引。
- `artifacts/preprocessed/windows.*`：Condition 内完整 10 秒窗；同一 Condition 的 `sample_weight` 总和固定为 1。
- `artifacts/features/window_features.*`：10 秒流式特征；只作为汇聚输入，不是独立监督样本。
- `artifacts/features/condition_features.*`：135 行 participant–Condition 汇聚特征，供最终状态/策略训练。
- `artifacts/models/`：状态主模型、full/no-EEG/behavior-only 回退模型和策略模型。
- `artifacts/reports/condition_level_lopo_metrics.json`、`condition_level_lopo_predictions.csv`、`condition_level_lopo_report_zh.md`：135 标签冷启动 LOPO、风险阈值与个体校准结果。
- `artifacts/reports/`：其余数据 QC、模型卡、环境版本、哈希与 Shadow 回放报告。

若安装了 `pyarrow`，CSV 同时写 Parquet；否则 CSV 仍会完整生成并明确保留该可选能力。

## 特征路径

生理流固定解释为 `counter + EEG 1–4 + unused 5–6 + ECG 7–8`，ECG 使用双极 `ch6-ch7`。状态特征通过 `StreamingPhysioProcessor` 生成，离线原始 XDF 回放与在线窗口共用这一个处理器。MNE 连续处理只生成审计/QC，不进入另一套训练特征。

P002、P005、P006、P010、P014、P016 默认禁用 EEG。其他参与者只有严格覆盖率达到 60% 才提取 EEG。10 秒 ECG 只依赖 HR、RR 和质量；30/60/120/300 秒 HRV 是满足历史长度时才出现的慢特征。

头动以 Unity HMD pose 为准；XDF IMU 不进入头动模型。眼动使用 gaze direction 和 I-VT，不训练闭眼。视频默认 OpenCV 2 fps；CLIP 默认关闭，YOLO/MediaPipe 不在第一版。

## 训练与验证

真实问卷标签的训练单位是 **participant–Condition**，即 15×9=135 条，而不是 946 个继承标签的窗口。10 秒窗口只用于生成 Condition 内特征轨迹；最终每个 Condition 汇聚 mean、std、min、max、range、median、first、last、last-first、slope 与 missingness ratio。

每个外层 LOPO 折都先只在训练参与者中试验 20/30/50/80 个保留特征，再在选中的 K 下比较 Ridge、ElasticNet、SVR、RandomForest、ExtraTrees、HistGradientBoosting 的残差回归。最终预测为 `Condition-only 基线 + 残差集成`，relaxation 与 discomfort 分开建模。高 discomfort（≥0.5）另以 LogisticRegression/SVM 概率集成建模，并在 0.20/0.25/0.30/0.35/0.40/0.50 中按召回和假阴性选择阈值。

报告同时给出冷启动 LOPO，以及使用每位参与者前 2 或 3 个 Condition 做残差校准后预测其余 Condition 的结果。只有 relaxation 同时超过两条基线且排序方向为正、并且 discomfort 同时超过两条基线且高风险召回达标，模型才可能通过部署门；否则所有推荐保持 `hold`。

## 实时与 Unity

生理流通过 LSL `type=eeg` 输入。Unity UDP 输入端口为 `127.0.0.1:5055`，Python 输出到 `127.0.0.1:5056`。协议与 C# Shadow 桥接位于 [`integrations/unity/PROTOCOL.md`](integrations/unity/PROTOCOL.md) 和 [`integrations/unity/RtmlShadowUdpBridge.cs`](integrations/unity/RtmlShadowUdpBridge.cs)。

```powershell
rtml serve
```

每个消息含 `schema_version`、`unix_time_ms`、`cycle_index`、`window_start_ms`、`window_end_ms`。源超时、时钟乱序或模态质量不足时仍输出本周期两条消息，但推荐为 `hold`。`shadow=true` 是 schema 硬校验，第一版不能关闭。

## 测试

```powershell
pytest
pytest -m integration
```

单元测试覆盖 ID/Condition 归一化、CSV 方言、量表方向、相对视频路径、通道/counter、流式特征一致性、窗口权重、10 秒周期、乱序 Condition 和消息 schema。真实集成测试固定先检查 P003/P015，并显式断言 P015 的约 90 ms marker 异常被记录。

## 研究边界

Condition 推荐效果仅是探索性离线结果。新的前瞻性实验验证完成前，不得把 C# bridge 改为自动应用推荐，也不得把敏感诊断、Post 问卷或访谈默认加入模型。

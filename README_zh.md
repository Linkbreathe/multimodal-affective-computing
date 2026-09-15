# 多模态情感计算

**一条从录制信号到运行中自适应系统的完整管线**：会话接入 → 特征与表征提取 → 多模态融合 →
被试独立评测 → 录制回放 → 实时推理。

本仓库由两个原本独立的代码库融合而成。融合前它们**已经在互相调用**，只能靠绝对路径和
`sys.path` 注入拼接：

| 原仓库 | 代码中的称呼 | 提供什么 |
| --- | --- | --- |
| `real-time-vis-physio-fusion` | Project B | 可复用的预训练编码器和融合架构；EgoEmotion / SEED-V 比较实验归入辅助目录 |
| `Relax-Model` | Project A | 论文主线的会话索引、手工特征、经典与时序模型、Shadow 推理、Unity 集成 |

现在它们是**同一个可安装包 `mac`**，包内按「模块做什么」而不是「来自哪个项目」组织。
两份 Git 历史完整保留。融合前写下的任何路径或命令，请对照
[合并对照表](docs/MERGE-MAP.md) 翻译。

实现了某个工作流只说明它能跑。它不能证明模型对新被试泛化、某个表征测到了内部状态，
或者自适应控制对人真的有益。

[English overview](README.md) · [合并对照表](docs/MERGE-MAP.md) · [论文代码范围](docs/thesis-scope.md) · [代码地图](docs/code-map.md) · [输入契约](data/contracts/input_tables.md) · [Unity Shadow 协议](integrations/unity/PROTOCOL.md)

## 论文主线

单一传感器很难推断情绪与放松。视频提供视觉上下文，视线和头动描述行为，PPG / ECG / EEG
提供生理测量。这些信号在时序、噪声、可得性和表征需求上都不同。

当前分支优先服务于基于 RELAX 的论文管线：

1. **测量与刻画** —— 生理、视线、头动、视觉上下文如何随条件变化，包括信号质量、缺失和基线效应。
2. **模态贡献** —— 哪些信号携带可用信息，在同一评测协议下多模态是否优于单模态。
3. **表征迁移** —— 冻结的预训练表征迁移到情感目标的效果，以及投影 / 压缩 / 微调 / LoRA 各自何时有用。
4. **被试泛化** —— 在留一被试或显式 split manifest 下，模型能否预测训练中未见的被试。
5. **自适应** —— 状态估计、不确定性、信号可得性如何转化为推荐，在录制回放和非干预式 Shadow 运行时中如何表现。

EgoEmotion 与 SEED-V 仍保留在 `Auxiliary/benchmarks/` 作为辅助基准，不属于论文主线运行路径。
这些是研究目标，不是结论。任何结果都必须在其数据集、目标定义、cohort 和评测协议内解读。

## 包结构

```text
src/mac/
  data/            RELAX 会话索引与 I/O、标签解析、对齐、窗口、
                   condition 数据和缓存协议；保留基准兼容适配器
  preprocessing/   流式预处理管线与质量控制；
                   PPG / ECG / RELAX 生理信号的分模态预处理
  features/        手工生理、眼动、头动、视频特征；动态纹理描述子；冻结 VideoMAE v2 嵌入
  encoders/        可复用的预训练编码器封装：EEGPT、REVE、PaPaGei、Pulse-PPG、
                   ECGFounder、VideoMAE v2、InceptionTime、PatchTST、NeuroRVQ
  fusion/          early / mid / late、bottleneck、HEALNet、Perceiver IO、Q-Former、
                   TMC、CGGM、MM-Lego、蒸馏、冻结压缩，以及最小 Ridge / 1D-CNN 融合基准
  models/          EEG head、LoRA、头动 CNN、经典 condition 模型、时序 1D-CNN、视觉模型
  tasks/           任务 head、损失、relaxation 回归、condition 控制
  training/        LOSO 训练器与早停；condition / state / policy / video 训练入口
  evaluation/      指标、被试折契约、LOPO、安全门
  experiments/     研究专用的实验编排
  adaptive/
    offline/       冻结前缀推理与按时序的录制回放
  realtime/        Shadow 时钟、engine、serve、replay、推荐策略
  reporting/       run 摘要、实验报告、结果注册表
  config/          分层配置（base → experiment → local）与融合 runner 用的扁平 YAML 加载器
  utils/           原子写入、哈希、日志、TensorBoard 助手
  cli.py           `mac` 命令
```

仓库根的配套目录：`scripts/` 与 `analysis/`（论文主线）、`configs/`（运行配置）、
`tests/`（主线测试）、`integrations/unity/`、`data/contracts/`、`docs/`，以及保存历史和
辅助基准的 `Auxiliary/`。具体范围见 [论文代码范围](docs/thesis-scope.md)。

`src/real_time_ml/` 和 `src/src/` 是**自动生成的兼容 shim**，把旧 import 路径别名到 `mac`，
计划在 v2 删除。

## 安装

Python 3.11。

```powershell
conda env create -f environment.yml
conda activate mac
mac --help
```

或安装进已有的 3.11 环境：

```powershell
python -m pip install -e ".[dev]"
```

核心依赖**刻意不含 PyTorch**，这样经典工作流和数据无关的测试不需要 GPU 栈就能装上。
做预训练编码器和融合研究时，先装好适配平台的 PyTorch，再：

```powershell
python -m pip install -e ".[dl,viz,ecg,head]"
```

| Extra | 范围 |
| --- | --- |
| `dl` | torch、torchvision、einops、transformers、huggingface-hub、safetensors、timm、h5py |
| `viz` | matplotlib、seaborn、tqdm、tensorboard |
| `ecg` / `head` | neurokit2 / ahrs |
| `dev` | pytest、pytest-cov、ruff |

`Auxiliary/benchmarks/egoemotion/environment-videomae2.yml` 仍是**独立的基准环境**：
它钉了 `timm` 0.4.12，与编码器栈需要的 `timm` 1.x 无法共存。`requirements*.txt`
保留了融合研究在 Linux 上验证过的钉版。

## 配置

两套配置系统并存 —— 两边各用一套，第一版没有把任何一套归并掉。

```text
configs/fusion/                  共享融合模型模板
configs/project.yaml              RELAX 运行参数                    ─┐  分层：
configs/base.yaml                 RELAX 协议默认值                   │  runtime、
configs/experiments/*.yaml        论文实验设置                       │  classical
                                                                          ─┘  analysis

Auxiliary/benchmarks/egoemotion/configs/  EgoEmotion 基准配置
Auxiliary/benchmarks/seedv/configs/       SEED-V 基准配置
```

`configs/local.yaml` 被 Git 忽略，从 `configs/local.example.yaml` 复制后填
`paths.raw_root`、`paths.labels_root` 和设备设置。`--experiment`、`--local-config`
这类全局选项要写在**子命令之前**。

基准配置有意不占用论文主线的 `configs/` 根目录；RELAX 分层系统使用
`project.yaml → base.yaml → experiments/ → local.yaml`。

## 三条执行路径

| 路径 | 入口 | 范围 |
| --- | --- | --- |
| 离线研究 | `mac` 的提取 / 训练 / 评测子命令；`scripts/`；`analysis/` | 从录制数据产出特征、模型、比较与报告 |
| Shadow 推理 | `mac replay`、`mac serve` | 输出状态预测与推荐供记录或显示，要求 `shadow=true` |

研究专用的视觉与融合 checkpoint 不会自动提升为运行时后端。暂不需要的 Adaptive Control
实验代码已归档到 [`Auxiliary/adaptive_control/`](Auxiliary/adaptive_control/)，不属于论文主线。

### 离线 condition 级工作流

```powershell
$runArgs = @("--experiment","configs/experiments/runtime-classical.yaml","--local-config","configs/local.yaml")
mac @runArgs index
mac @runArgs preprocess
mac @runArgs extract-features --no-video
mac @runArgs train-state
mac @runArgs evaluate
mac @runArgs report
```

EgoEmotion 与 SEED-V 的命令见 [`Auxiliary/benchmarks/README.md`](Auxiliary/benchmarks/README.md)，
它们不再作为论文主线执行路径列出。

### RELAX aligned 协议

```bash
python scripts/run_relax_foundation_probe.py \
  --embedding-cache /path/to/aligned/condition_embeddings.pt \
  --cohorts /path/to/cohorts.json --cohort all_135 \
  --split-manifest /path/to/splits.csv --mask-manifest /path/to/masks.csv \
  --labels /path/to/labels.csv --windows /path/to/windows.csv \
  --modalities eeg ecg eye head video --fusion healnet \
  --seed 20260705 --device cuda --require-cuda --strict \
  --output-dir artifacts/relax/my_aligned_run
```

早期的 foundation 协议（`scripts/relax_foundation/run_relax_foundation_probe.py`）
期望的缓存格式不同，**与 aligned runner 不可互换**，尽管名字相似。
协议边界见[代码地图](docs/code-map.md)。

### Unity

```powershell
mac replay --help
mac serve --help
```

Shadow 默认传输：Unity → Python `127.0.0.1:5055`，Python → Unity `127.0.0.1:5056`。
见 [协议](integrations/unity/PROTOCOL.md) 与 [C# 桥接](integrations/unity/RtmlShadowUdpBridge.cs)。

## 数据与监督信号

**RELAX condition 级研究**：15 被试 × 9 条件 = 135 个被试–条件观测，每个条件内切成完整不重叠的
10 秒窗口。目标来自问卷评分 `(rating - 1) / 6`。把一个条件的评分复制到它的各个窗口
**不会**产生额外的独立观测。

**Windows RQ2 track**：独立锁定的 FMQ-9 契约 —— 9 被试、81 个 condition 标签、567 个源窗口、
545 个 common-valid 窗口，由 [`mac.windows_rq2_representations`](src/mac/windows_rq2_representations.py)
校验。`WINDOWS_DONE.json` 只标记该 track 完成。

**辅助 EgoEmotion 与 SEED-V 基准**各有自己的标签、采样单位、缓存格式和 split，
不能与 RELAX condition 级管线混用。

原始录制、问卷和大部分生成产物是外部输入且被 Git 忽略。全新 clone 支持源码检视和数据无关测试；
复现结果还需要对应的源数据、配置与实验产物。

预训练权重需另行准备：VideoMAE v2（`OpenGVLab/VideoMAEv2-Base`）、PaPaGei / Pulse-PPG
（默认在同级目录 `papagei-foundation-model/` 和 `pulseppg/`）、EEGPT、REVE、ECGFounder、NeuroRVQ。

## 评测与可复现性

每个实验请记录：commit、runner、完整命令与配置；数据集版本、目标定义、cohort、窗口策略；
训练 / 验证 / 测试的被试划分；编码器与 checkpoint 身份、预处理、缓存格式、mask；
随机种子、协议哈希、包版本、硬件；每折输出与聚合方式。

LOSO / LOPO 每折留出一个被试。片段必须跟随被试划分 —— 独立切分相关片段回答的是另一个问题。
融合要和单模态、condition-only 基线在同一评测规则下比较。不同目标、cohort 或协议的分数不可直接比较。

## 测试

```powershell
python -m pytest -m "not external and not integration and not slow"
```

marker 含义：`integration` 读取被试源数据，`slow` 训练模型或做昂贵计算，
`external` 需要预训练权重、外部模型代码或真实被试数据。

融合后套件在开发机（Windows，Python 3.11）上的实测，与融合前两个基线对比：

| | collected | passed | failed | skipped | 采集错误 |
| --- | --- | --- | --- | --- | --- |
| 融合前 `Relax-Model` | 96 | 93 | 3 | 0 | 0 |
| 融合前 `real-time-vis-physio-fusion` | 198 | 185 | 0 | 13 | 7 |
| **合计** | **294** | **278** | **3** | **13** | **7** |
| **融合后，每个测试文件各起一个进程** | **294** | **278** | **3** | **13** | **7** |
| 融合后，全部同一个进程 | 294 | 277 | 4 | 13 | 7 |

每个测试文件各起一个进程时，融合后的套件与融合前基线**逐项相等**。
全部挤在同一个进程里会多出 **1 个失败**，那是进程的性质，不是代码的问题：

- **3 个失败与融合前完全相同** —— 它们读取
  `artifacts/cross_project_alignment_2026-07-16/.../contract.json`，一份从未纳入版本控制的
  生成产物。在原 `Relax-Model` 仓库里同样失败。
- **7 个采集错误与融合前完全相同** —— 该环境缺 `einops` 和 `huggingface_hub`，
  装上 `dl` extra 即消除。
- **1 个新失败，是 Windows 环境脆弱点，不是代码缺陷** ——
  `test_cross_project_alignment.py::test_validation_ranking_uses_dedicated_validation_rows`
  抛的是 `threadpoolctl`（3.6.0）枚举进程已加载 DLL 时的 `OSError: GetModuleFileNameEx failed`，
  不是任何断言失败。单独跑它通过；在本仓库只跑 RTML 那批测试文件时也通过。
  原因是融合后两套依赖栈进入同一个进程，让那次 DLL 枚举更容易撞上竞态。
  让每个测试文件各起一个进程即可消除 —— 上表第一行「融合后」就是这么测的
  （对全部 63 个文件逐个 `pytest <file>`）。省事的做法是
  `pip install pytest-xdist` 后 `pytest -n 4 --dist loadfile`。

不依赖环境的静态验证：

```powershell
# 包内与两棵 shim 树的所有模块都能无错导入
python -c "import pkgutil,importlib,mac; [importlib.import_module(m.name) for m in pkgutil.walk_packages(mac.__path__,'mac.')]"
python -m compileall -q src scripts analysis tests Auxiliary/benchmarks Auxiliary/adaptive_control
```

论文主线的 `scripts/` 与 `analysis/` 入口保持在根目录；数据集专属入口和暂不需要的
Adaptive Control 运行时位于 `Auxiliary/`，只有在具备相应外部数据、模型依赖或 Unity
安装时单独验证。

## 贡献与扩展

- 可复用实现放进 `src/mac/` 中它所属的管线阶段，实验编排放 `scripts/` 或 `analysis/`，参数放 `configs/`。
- 源录制保持只读，生成产物不要放进版本化的源码目录。
- 保持被试–条件的监督关系和声明的 cohort，排除要显式写明。
- 预处理、特征选择、调参只能在允许的训练数据上拟合。
- 新的比较用新的配置和新的 run ID。
- 研究专用表征不要进入运行时模型选择；`adaptive/offline/` 与归档的
  `Auxiliary/adaptive_control/` 保持分开。
- 新增编码器时，写明期望形状、时序采样、通道顺序、单位、权重、缺模态行为。
- 按证据的实际层级描述结论：实现、离线评测、录制回放，还是前瞻性被试研究。

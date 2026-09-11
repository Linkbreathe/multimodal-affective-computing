# Real-time Visual–Physiological Fusion

面向情绪与放松状态研究的多模态实验代码，支持视频、眼动、PPG、EEG、ECG、头部运动的特征提取、融合训练、按被试评估和离线自适应回放。

仓库包含多个研究阶段，各自的标签、缓存和划分协议不同。自适应模块基于历史数据离线回放，不能据此视为已验证的在线控制系统。

## 选择实验入口

| 研究任务 | 入口（相对仓库根目录） | 配置 / 输入 |
| --- | --- | --- |
| EgoEmotion 任务级融合 | `scripts/run_experiment.py` | `configs/base.yaml` + `configs/fusion/` |
| EgoEmotion 10 秒片段融合 | `scripts/run_experiment_10s.py` | 任务内片段 manifest + embeddings |
| PPG 微调 / MM-Lego | `scripts/run_finetune_ppg.py` / `scripts/run_lego_experiment.py` | `configs/finetune/` / Lego 预训练权重 |
| SEED-V EEGPT / REVE / EEG+眼动 | `scripts/run_seedv_experiment.py` / `scripts/run_seedv_reve_experiment.py` / `scripts/run_seedv_eeg_eye_fusion.py` | `configs/seedv*.yaml` |
| RELAX 早期 foundation probe | `scripts/relax_foundation/run_relax_foundation_probe.py` | `samples` 格式缓存 |
| RELAX 跨项目对齐融合 | `scripts/run_relax_foundation_probe.py` | 对齐缓存 + 标签、窗口、mask、split manifests |
| RELAX 冻结特征压缩 | `scripts/run_relax_compression_fusion_v2.py` | 五模态协议、预注册文件、固定 cohort/seeds |
| RQ2 审计与完整流水线 | `scripts/run_rq2_wsl.py` | 外部 hand-off + 已保存的对齐缓存 |
| HEALNet 前缀推理 / 回放 | `scripts/run_healnet_prefix_inference.py` / `scripts/run_healnet_adaptive_replay.py` | P009 历史会话与冻结模型 |

详见 [代码地图与协议边界](docs/code-map.md)、[分支整理记录](docs/branch-history.md)、[验证记录](docs/consolidation-validation.md)。

## 安装

从仓库根目录执行；整理时使用 Linux、Python 3.11 验证。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

也可执行 `conda env create -f environment.yml`，然后 `conda activate visphy`。

依赖按用途拆分：`requirements.txt` 为缓存特征训练/评估的核心依赖，`requirements-extraction.txt` 加入原始信号处理与模型提取依赖，`requirements-dev.txt` 再加入测试依赖。需要 CUDA 时，先安装与机器匹配的 PyTorch 2.10.0 构建；验证机器使用 `2.10.0+cu128`。

版本来自已有环境的运行验证，尚未在全新环境中完成完整安装验证。预训练编码器的外部源码和权重需要单独准备。原系统 pip 清单及完整 Conda 导出保留在 [历史环境目录](docs/environments/)，不作为当前安装入口。

## 目录、数据和权重

```text
configs/                # 基础、融合、消融和数据集配置
src/
  data/                 # 数据加载、切片、预处理、缓存协议
  encoders/             # 编码器包装与注册
  fusion/               # 融合模型、公共工厂与冻结特征压缩
  models/               # EEG 分类头、LoRA、头动 CNN
  tasks/                # 任务 head、loss、条件控制
  trainer/              # LOSO 训练与 early stopping
  adaptive/             # 前缀推理、控制器与离线回放
  utils/                # 配置、指标、日志、报告
scripts/                # 实验入口、调度、审计与报告
  relax_foundation/     # 早期 RELAX 协议，独立于 aligned runner
tests/                  # 标准测试套件
data/
  datasets/             # egoemotion_raw、relaxdata 等原始数据
  preprocessed/         # SEED-V 等预处理输出
  embeddings/           # EgoEmotion / SEED-V 缓存
weights/                # 外部模型权重
checkpoints/            # 本项目训练模型
artifacts/              # RELAX 缓存、协议文件和实验产物
wsl_results/            # RQ2 WSL 结果
combined/               # RQ2 汇总报告
```

数据、缓存和新实验产物保留本地，不随源码提交。历史 Git 已包含部分 PaPaGei 权重、报告及 `code.zip`；新增忽略规则不会移除这些历史文件。

- VideoMAE V2 使用 `OpenGVLab/VideoMAEv2-Base` 的远程模型代码和 safetensors 权重，首次使用需要下载或预先准备 Hugging Face 缓存。
- PaPaGei / Pulse-PPG 默认查找仓库同级的 `papagei-foundation-model/` / `pulseppg/` 源码。
- EEGPT、REVE、ECGFounder、眼动编码器需要对应权重；路径以配置、构造参数和 CLI 的 `--help` 为准。
- NeuroRVQ 适配器接收上游源码目录、checkpoint、模态和通道列表。

部分历史实验保留 Windows/WSL 默认路径、固定参与者、seed 和校验和。迁移机器时需要检查参数并准备相同输入，不能用不同缓存替代后直接比较结果。

## 运行示例

以下命令需要预先准备数据和缓存，均从仓库根目录执行。各入口支持的参数请先查看 `--help`。

### EgoEmotion：10 秒片段融合

通过 `scripts/segment_and_extract_10s.py` 生成任务内片段和缓存，适配 `configs/base.yaml` 中的本机路径后运行：

```bash
python scripts/run_experiment_10s.py \
  --config configs/base.yaml --fusion_config configs/fusion/early.yaml \
  --embeddings_dir data/embeddings/egoemotion/10s_task_aware \
  --name ego_10s_early --device cuda
```

任务级缓存与 10 秒片段缓存不能混用。PPG 微调和 MM-Lego 预训练有各自入口。

### SEED-V

```bash
python scripts/run_seedv_experiment.py --config configs/seedv_base.yaml
```

先检查配置中的预处理数据、embedding 和权重路径，确认通道与信号单位。`extract_seedv_embeddings.py` / `extract_seedv_reve_embeddings.py` 对应不同编码器；LoRA、微调与消融使用相应 `run_seedv_*` 入口。

### RELAX：对齐协议

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

早期 foundation 实验使用独立入口和 `samples` 缓存，不能传入上述对齐缓存：

```bash
python scripts/relax_foundation/run_relax_foundation_probe.py \
  --embedding-cache /path/to/sample_cache.pt \
  --cohorts /path/to/cohorts.json --cohort all_135 \
  --modalities eeg ecg eye head video --fusion early \
  --output-dir artifacts/relax/foundation_run
```

压缩 v1/v2、condition anchor、embedding ladder 是不同实验协议，保留各自运行与评估脚本；详见 [代码地图](docs/code-map.md)。

### RQ2：先审计交接

```bash
python scripts/run_rq2_wsl.py \
  --shared-root /path/to/rq2_shared \
  --cache-root artifacts/relax/aligned_20260716 \
  --output-root wsl_results --audit-only
```

删除 `--audit-only` 才会运行后续流水线。审计检查 contract、标签/窗口、folds、anchors、Windows 完成标记和缓存身份；不通过时退出并写错误报告，不会自动重建输入。完整流水线还会向仓库的 `combined/` 写汇总结果。

## 测试

不依赖真实参与者数据或预训练权重的测试：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  HF_HUB_OFFLINE=1 python -m pytest -q -m "not external"
```

准备好 PaPaGei 源码/权重、ECGFounder 权重和真实 ECG 后，可以运行 `python -m pytest -q`。`external` 标记区分依赖本机资源的检查。测试仅从 `tests/` 收集，产物目录内的临时检查不属于标准套件。

测试和 CLI 检查的具体结果见 [验证记录](docs/consolidation-validation.md)。本轮未重跑完整 LOSO 训练或 GPU 特征提取。

## 维护约定

公共实现放在 `src/`，实验调度与报告放在 `scripts/`，参数放在 `configs/`。普通训练、10 秒训练和 PPG 微调共用 `src/fusion/factory.py` 的工厂与池化逻辑。旧 CLI 路径及主要导入仍兼容。

历史说明保留在 `docs/superpowers/`、`research-wiki/`、`reports/`、`refactor/`；它们对应当时的研究状态，当前运行以本 README、代码地图和实际 CLI 为入口。

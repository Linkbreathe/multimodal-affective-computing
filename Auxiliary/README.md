# Auxiliary material

历史参考材料与非论文基准。这些内容**不参与**论文主线的安装、训练、评测、测试或
Unity 服务运行路径，但为分析、对比和复现提供背景与出处。合并自两个原仓库的
`Auxiliary/`，并在本分支补充了 EgoEmotion / SEED-V 基准归档。

## 本分支的基准归档

| 路径 | 用途 |
| --- | --- |
| `benchmarks/egoemotion/` | EgoEmotion 专属配置、脚本、测试、历史报告与图表。 |
| `benchmarks/seedv/` | SEED-V 专属配置、脚本与测试。 |
| `benchmarks/README.md` | 基准目录说明与边界。 |
| `adaptive_control/` | 暂不参与论文主线的 Unity/UDP Adaptive Control 服务、配置、模型注册表、启动器与专用测试。 |

基准脚本可以从仓库根目录执行，并复用 `src/mac/` 中的公共编码器、融合器和训练
组件；论文主线不得反向依赖这些基准脚本。

## Archived Adaptive Control

`adaptive_control/` 保存实验性的实时控制闭环。它可以复用 `src/mac/` 的公共特征和
模型实现，但不再由 `mac` 主线 CLI 暴露，也不属于论文当前的 Shadow / recorded-replay
证据链。若需要历史复现，从仓库根目录运行 `python -m Auxiliary.adaptive_control.cli`。

## 来自 `real-time-vis-physio-fusion`（Project B）

| 路径 | 用途 |
| --- | --- |
| `.codex` | 空的历史占位文件，不含任何 Codex 配置。 |
| `code.zip` | 更早版本源码树的快照。维护中的代码在 `src/mac/`、`scripts/`、`tests/`。 |
| `refactor/` | 历史代码评审与重构笔记（`fix_opinion.md`）。 |
| `research-wiki/` | 文献笔记、研究想法与生成的主题索引（`index.md`、`gap_map.md`、`query_pack.md`、`ideas/`、`graph/`）。研究背景，非运行时输入。 |

## 来自 `Relax-Model`（Project A）

| 路径 | 用途 |
| --- | --- |
| `plan/` | 研究规划文档（基础模型自适应回放指南）。 |
| `research/` | 历史评审笔记（模态与建模评审、离线自适应方案）。 |

## 注意

这些文档描述的是**写作当时**的仓库状态，其中的路径多为融合前的旧路径
（如 `src/utils/config.py`、`configs/base.yaml`、`real_time_ml.modeling.*`）。
复用其中的命令或路径时，请对照 [合并对照表](../docs/MERGE-MAP.md) 翻译。

活跃文档在 `docs/`，生成与整理过的实验报告在 `reports/`。

# Auxiliary material

历史参考材料。这些内容**不参与**安装、训练、评测、测试或 Unity 服务的任何运行路径，
但为分析提供背景与出处。合并自两个原仓库的 `Auxiliary/`。

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

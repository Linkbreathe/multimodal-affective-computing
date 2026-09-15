# 合并对照表：旧路径 → 新路径

本仓库由两个独立仓库融合而成。历史报告、笔记本、WSL 命令里的路径需要靠本文翻译。

| 原仓库 | 代码中的称呼 | 内容 | 原 GitHub |
| --- | --- | --- | --- |
| `real-time-vis-physio-fusion`（下称 **VisPhy**） | Project B | EgoEmotion / SEED-V / RELAX 离线深度学习研究 | `Linkbreathe/real-time-vis-physio-fusion` |
| `Relax-Model`（下称 **RTML**） | Project A | `rtml` CLI、实时 Shadow 推理、Unity | `Linkbreathe/Relax-Model` |

两份 Git 历史都通过 `git subtree` 完整导入（VisPhy 69 commits、RTML 13 commits）。
源仓库 commit 对象仍在当前 Git 历史中，但 prefix rewrite 之后，
`git log --follow <新路径>` 不一定自动跨回融合前路径；当前 checkout 也只配置了
merged 仓库的 `origin`。需要逐行追溯时，应使用源 HEAD 与融合前路径查询：

```bash
git cat-file -t 8383b3bce7750c1ec220c1134b91a816310edb11
git cat-file -t 371a04eb3f60c036bf7492e6a9f32b8632c8eb48
git log 8383b3b -- src/fusion/healnet.py
```

## 融合原则

包按**管线阶段**组织，不按来源项目。RTML 提供管线前端（会话接入、手工特征）
和 Shadow 后端，VisPhy 提供中段（预训练编码器、融合架构、LOSO 评测）。
融合前两者已经在互相调用，只是必须靠绝对路径和 `sys.path` 注入拼接。

第一版**不删除任何研究内容**：重复实现全部保留，只记录在文末「待去重」。

## 顶层目录

| 旧位置 | 新位置 | 说明 |
| --- | --- | --- |
| VisPhy `src/` | `src/mac/` | 并入统一包 |
| RTML `src/real_time_ml/` | `src/mac/` | 并入统一包（深度不变） |
| VisPhy `scripts/`（80） | `scripts/` | 保持在根、保持扁平，`parents[1]` 仍解析到仓库根 |
| RTML `scripts/`（4） | `scripts/` | 无文件名冲突，直接并入 |
| RTML `analysis/`（56） | `analysis/` | 原样，`parents[2]` 仍解析到仓库根 |
| VisPhy `tests/`（42）+ RTML `tests/`（23） | `tests/`（65） | 无 `test_*.py` 重名 |
| VisPhy `configs/`（52）+ RTML `configs/`（14） | `configs/`（66） | 见下方「改名」 |
| VisPhy `docs/` | `docs/` | 原样 |
| VisPhy `weights/` `figures/` `reports/` | 同名 | 原样 |
| RTML `data/` `integrations/` `artifacts/` | 同名 | 原样 |
| VisPhy `auxiliary/` + RTML `auxiliary/` | `auxiliary/` | 合并 |

## 会影响命令行的改名

| 旧 | 新 | 原因 |
| --- | --- | --- |
| `configs/base.yaml`（VisPhy 的扁平 EgoEmotion 配置） | `configs/egoemotion.yaml` | 与 RTML 的分层 `configs/base.yaml` 撞名；RTML 那份必须留在原位，`mac/config/__init__.py` 靠 `parents[3]` 定位它 |
| `rtml <子命令>` | `mac <子命令>` | `rtml` 保留为别名 |
| `python -m real_time_ml.cli` | `python -m mac.cli` | 旧写法经 shim 仍可用 |

融合基线中的其余配置原本保持不变；在本分支的论文范围整理中，EgoEmotion/SEED-V
专属配置已移到 `auxiliary/benchmarks/egoemotion/configs/` 和
`auxiliary/benchmarks/seedv/configs/`。共享的 `configs/fusion/` 与论文主线的
`configs/project.yaml`、`configs/base.yaml`、`configs/experiments/` 保留在根目录。

## 论文范围整理分支

`thesis-core-auxiliary` 将不属于论文主线的数据集实验材料从根目录移入
`auxiliary/benchmarks/`：

- EgoEmotion：专属 runner、配置、测试、历史报告和 VAD 图表；
- SEED-V：专属 runner、配置和测试；
- `src/mac/` 中仍被 RELAX 使用的通用编码器、融合器和缓存组件不移动；
- 论文主线只依赖 `mac`、RELAX runner、RQ2/分析入口和 Unity Shadow 集成，不依赖
  辅助 benchmark runner 或 Adaptive Control。

暂不需要的 Adaptive Control 实验服务、模型注册表、配置、启动器和专用测试已归档到
`auxiliary/adaptive_control/`，不再属于 `src/mac/` 的活动实现。

具体清单见 [论文代码范围](../../docs/thesis-scope.md)。

## 合并而非移动的文件

这些文件两边各有一份且必须收敛为一份。原件保留在本目录的 `legacy/`，也在 Git 历史中。

| 新文件 | 由什么合并而来 | 保留的原件 |
| --- | --- | --- |
| `README.md` | 两边 README | `legacy/README-visphy.md`、`legacy/README-rtml.md` |
| `README_zh.md` | RTML 中文总览 | `legacy/README_zh-rtml.md` |
| `pyproject.toml` | RTML pyproject + VisPhy `pytest.ini` + 两边依赖 | `legacy/pytest-visphy.ini` |
| `environment.yml` | 两边 conda 环境 | `legacy/environment-visphy.yml`、`legacy/environment-rtml.yml` |
| `.gitignore` | 两边忽略规则 | Git 历史 |
| `tests/conftest.py` | VisPhy 的 `ego_data_dir` fixture + RTML 的源码路径 shim | `legacy/conftest-rtml.py` |
| `auxiliary/README.md` | 两边索引 | `legacy/README-Auxiliary-rtml.md` |

## 兼容层

`src/real_time_ml/` 和 `src/src/` 中保留的模块是**自动生成的 shim**，把仍在活动包
中的旧 import 别名到新位置。Adaptive Control 的专用 shim 与实现已一并移入
`auxiliary/adaptive_control/legacy/`，不再作为安装包兼容层。活动包叶子模块用
`sys.modules` 别名，因此新旧名字拿到的是**同一个对象**：

```python
from real_time_ml.modeling.dcnn import train_dcnn_state as a
from mac.models.dcnn import train_dcnn_state as b
assert a is b          # 已验证通过
```

注意：审计时以为兼容层是反序列化 `artifacts/models/realtime_multimodal_window_full.joblib`
（14.7MB）所必需——**实际不是**。那个 bundle 是纯 sklearn 对象的 dict，不含任何本项目的类路径，
裸环境即可加载。兼容层的实际价值是让历史命令照抄能跑，以及出问题时逐模块对照新旧行为。
计划在 v2 删除。

## 包内模块对照

下表 162 行覆盖两个原仓库 `src/` 里的活动模块，新旧两端都已核对存在于磁盘上。
Adaptive Control 的专用模块不再属于活动 `src/`，已在下方单独标为归档映射；只有
`mac.adaptive` 和 `mac.utils` 两个 `__init__.py` 不在活动表中——它们是融合时新写的，
没有对应的旧 import。

### `mac.data`

| 来源 | 旧 import | 新 import |
| --- | --- | --- |
| RTML | `real_time_ml.data` | `mac.data` |
| VisPhy | `src.data` | `mac.data` |
| RTML | `real_time_ml.data.alignment` | `mac.data.alignment` |
| RTML | `real_time_ml.data.index` | `mac.data.index` |
| RTML | `real_time_ml.data.io` | `mac.data.io` |
| RTML | `real_time_ml.data.labels` | `mac.data.labels` |
| RTML | `real_time_ml.data.tables` | `mac.data.tables` |
| RTML | `real_time_ml.data.video` | `mac.data.video` |
| RTML | `real_time_ml.data.xlsx` | `mac.data.xlsx` |
| RTML | `real_time_ml.modeling.condition_data` | `mac.data.condition_data` |
| VisPhy | `src.data.collate` | `mac.data.collate` |
| VisPhy | `src.data.dataset` | `mac.data.dataset` |
| VisPhy | `src.data.egoemotion` | `mac.data.egoemotion` |
| VisPhy | `src.data.egoemotion.manifest` | `mac.data.egoemotion.manifest` |
| VisPhy | `src.data.embedding_shapes` | `mac.data.embedding_shapes` |
| VisPhy | `src.data.label_builder` | `mac.data.label_builder` |
| VisPhy | `src.data.relax_attention_video` | `mac.data.relax_attention_video` |
| VisPhy | `src.data.relax_dataset` | `mac.data.relax_dataset` |
| VisPhy | `src.data.relax_foundation` | `mac.data.relax_foundation` |
| VisPhy | `src.data.relax_foundation_dataset` | `mac.data.relax_foundation_dataset` |
| VisPhy | `src.data.segments` | `mac.data.segments` |
| VisPhy | `src.data.video_transforms` | `mac.data.video_transforms` |

### `mac.preprocessing`

| 来源 | 旧 import | 新 import |
| --- | --- | --- |
| RTML | `real_time_ml.preprocessing` | `mac.preprocessing` |
| RTML | `real_time_ml.preprocessing.mne_qc` | `mac.preprocessing.mne_qc` |
| RTML | `real_time_ml.preprocessing.pipeline` | `mac.preprocessing.pipeline` |
| VisPhy | `src.data.ecg_preprocessing` | `mac.preprocessing.ecg` |
| VisPhy | `src.data.ppg_preprocessing` | `mac.preprocessing.ppg` |
| VisPhy | `src.data.relax_physio_preprocessing` | `mac.preprocessing.relax_physio` |
| VisPhy | `src.data.seedv_preprocessing` | `mac.preprocessing.seedv` |

### `mac.features`

| 来源 | 旧 import | 新 import |
| --- | --- | --- |
| RTML | `real_time_ml.features` | `mac.features` |
| RTML | `real_time_ml.features.common` | `mac.features.common` |
| RTML | `real_time_ml.features.dynamic_texture` | `mac.features.dynamic_texture` |
| RTML | `real_time_ml.features.egocentric` | `mac.features.egocentric` |
| RTML | `real_time_ml.features.extract` | `mac.features.extract` |
| RTML | `real_time_ml.features.eye` | `mac.features.eye` |
| RTML | `real_time_ml.features.head` | `mac.features.head` |
| RTML | `real_time_ml.features.physio` | `mac.features.physio` |
| RTML | `real_time_ml.features.video` | `mac.features.video` |
| RTML | `real_time_ml.features.videomae2` | `mac.features.videomae2` |

### `mac.encoders`

| 来源 | 旧 import | 新 import |
| --- | --- | --- |
| VisPhy | `src.encoders` | `mac.encoders` |
| VisPhy | `src.encoders.base` | `mac.encoders.base` |
| VisPhy | `src.encoders.ecgfounder` | `mac.encoders.ecgfounder` |
| VisPhy | `src.encoders.ecgfounder_model` | `mac.encoders.ecgfounder_model` |
| VisPhy | `src.encoders.eegpt` | `mac.encoders.eegpt` |
| VisPhy | `src.encoders.eegpt_model` | `mac.encoders.eegpt_model` |
| VisPhy | `src.encoders.extract` | `mac.encoders.extract` |
| VisPhy | `src.encoders.inceptiontime` | `mac.encoders.inceptiontime` |
| VisPhy | `src.encoders.neurorvq` | `mac.encoders.neurorvq` |
| VisPhy | `src.encoders.papagei` | `mac.encoders.papagei` |
| VisPhy | `src.encoders.patchtst` | `mac.encoders.patchtst` |
| VisPhy | `src.encoders.pulse_ppg` | `mac.encoders.pulse_ppg` |
| VisPhy | `src.encoders.registry` | `mac.encoders.registry` |
| VisPhy | `src.encoders.reve` | `mac.encoders.reve` |
| VisPhy | `src.encoders.reve_model` | `mac.encoders.reve_model` |
| VisPhy | `src.encoders.reve_pos_bank` | `mac.encoders.reve_pos_bank` |
| VisPhy | `src.encoders.video_mae` | `mac.encoders.video_mae` |

### `mac.fusion`

| 来源 | 旧 import | 新 import |
| --- | --- | --- |
| RTML | `real_time_ml.modeling.minimal_fusion` | `mac.fusion.minimal_fusion` |
| RTML | `real_time_ml.modeling.minimal_fusion_dcnn` | `mac.fusion.minimal_fusion_dcnn` |
| RTML | `real_time_ml.modeling.minimal_fusion_dcnn_hp` | `mac.fusion.minimal_fusion_dcnn_hp` |
| VisPhy | `src.fusion` | `mac.fusion` |
| VisPhy | `src.fusion.base` | `mac.fusion.base` |
| VisPhy | `src.fusion.bottleneck` | `mac.fusion.bottleneck` |
| VisPhy | `src.fusion.cggm` | `mac.fusion.cggm` |
| VisPhy | `src.fusion.distill_late` | `mac.fusion.distill_late` |
| VisPhy | `src.fusion.early` | `mac.fusion.early` |
| VisPhy | `src.fusion.factory` | `mac.fusion.factory` |
| VisPhy | `src.fusion.frozen_compression` | `mac.fusion.frozen_compression` |
| VisPhy | `src.fusion.frozen_compression_v2` | `mac.fusion.frozen_compression_v2` |
| VisPhy | `src.fusion.healnet` | `mac.fusion.healnet` |
| VisPhy | `src.fusion.late` | `mac.fusion.late` |
| VisPhy | `src.fusion.mid` | `mac.fusion.mid` |
| VisPhy | `src.fusion.multimodal_lego` | `mac.fusion.multimodal_lego` |
| VisPhy | `src.fusion.perceiver_io` | `mac.fusion.perceiver_io` |
| VisPhy | `src.fusion.projector` | `mac.fusion.projector` |
| VisPhy | `src.fusion.qformer` | `mac.fusion.qformer` |
| VisPhy | `src.fusion.tmc` | `mac.fusion.tmc` |

### `mac.models`

| 来源 | 旧 import | 新 import |
| --- | --- | --- |
| RTML | `real_time_ml.modeling` | `mac.models` |
| RTML | `real_time_ml.modeling.condition_models` | `mac.models.condition_models` |
| RTML | `real_time_ml.modeling.dcnn` | `mac.models.dcnn` |
| RTML | `real_time_ml.modeling.groups` | `mac.models.groups` |
| RTML | `real_time_ml.modeling.realtime_multimodal` | `mac.models.realtime_multimodal` |
| RTML | `real_time_ml.modeling.video_dcnn` | `mac.models.video_dcnn` |
| RTML | `real_time_ml.modeling.video_ridge` | `mac.models.video_ridge` |
| VisPhy | `src.models.constrained_layers` | `mac.models.constrained_layers` |
| VisPhy | `src.models.eegpt_finetune_head` | `mac.models.eegpt_finetune_head` |
| VisPhy | `src.models.eegpt_linear_probe` | `mac.models.eegpt_linear_probe` |
| VisPhy | `src.models.head_motion_1dcnn` | `mac.models.head_motion_1dcnn` |
| VisPhy | `src.models.lora` | `mac.models.lora` |
| VisPhy | `src.models.reve_classifier` | `mac.models.reve_classifier` |

### `mac.tasks`

| 来源 | 旧 import | 新 import |
| --- | --- | --- |
| VisPhy | `src.tasks` | `mac.tasks` |
| VisPhy | `src.tasks.heads` | `mac.tasks.heads` |
| VisPhy | `src.tasks.losses` | `mac.tasks.losses` |
| VisPhy | `src.tasks.relax_condition_control` | `mac.tasks.relax_condition_control` |
| VisPhy | `src.tasks.relaxation` | `mac.tasks.relaxation` |

### `mac.training`

| 来源 | 旧 import | 新 import |
| --- | --- | --- |
| RTML | `real_time_ml.modeling.condition_train` | `mac.training.condition_train` |
| RTML | `real_time_ml.modeling.policy_train` | `mac.training.policy_train` |
| RTML | `real_time_ml.modeling.train` | `mac.training.train` |
| RTML | `real_time_ml.modeling.video_train` | `mac.training.video_train` |
| RTML | `real_time_ml.training` | `mac.training` |
| RTML | `real_time_ml.training.condition` | `mac.training.condition` |
| RTML | `real_time_ml.training.dcnn` | `mac.training.dcnn` |
| RTML | `real_time_ml.training.policy` | `mac.training.policy` |
| RTML | `real_time_ml.training.state` | `mac.training.state` |
| VisPhy | `src.trainer.early_stopping` | `mac.training.early_stopping` |
| VisPhy | `src.trainer.fusion_trainer` | `mac.training.fusion_trainer` |

### `mac.evaluation`

| 来源 | 旧 import | 新 import |
| --- | --- | --- |
| RTML | `real_time_ml.evaluation` | `mac.evaluation` |
| RTML | `real_time_ml.evaluation.alignment` | `mac.evaluation.alignment` |
| RTML | `real_time_ml.evaluation.dynamic_texture_five` | `mac.evaluation.dynamic_texture_five` |
| RTML | `real_time_ml.evaluation.lopo` | `mac.evaluation.lopo` |
| RTML | `real_time_ml.modeling.evaluate` | `mac.evaluation.evaluate` |
| RTML | `real_time_ml.modeling.safety` | `mac.evaluation.safety` |
| VisPhy | `src.utils.metrics` | `mac.evaluation.metrics` |

### `mac.experiments`

| 来源 | 旧 import | 新 import |
| --- | --- | --- |
| RTML | `real_time_ml.experiments` | `mac.experiments` |
| RTML | `real_time_ml.experiments.dynamic_texture_five` | `mac.experiments.dynamic_texture_five` |
| RTML | `real_time_ml.experiments.minimal_fusion` | `mac.experiments.minimal_fusion` |
| RTML | `real_time_ml.experiments.minimal_fusion_dcnn` | `mac.experiments.minimal_fusion_dcnn` |
| RTML | `real_time_ml.experiments.minimal_fusion_dcnn_hp` | `mac.experiments.minimal_fusion_dcnn_hp` |

### `mac.adaptive` 与归档 Adaptive Control

| 来源 | 旧 import | 新 import |
| --- | --- | --- |
| RTML | `real_time_ml.adaptive_control` | `auxiliary.adaptive_control`（归档） |
| RTML | `real_time_ml.adaptive_control.contracts` | `auxiliary.adaptive_control.contracts`（归档） |
| RTML | `real_time_ml.adaptive_control.models` | `auxiliary.adaptive_control.models`（归档） |
| RTML | `real_time_ml.adaptive_control.physio_monitor` | `auxiliary.adaptive_control.physio_monitor`（归档） |
| RTML | `real_time_ml.adaptive_control.policy` | `auxiliary.adaptive_control.policy`（归档） |
| RTML | `real_time_ml.adaptive_control.service` | `auxiliary.adaptive_control.service`（归档） |
| RTML | `real_time_ml.adaptive_control.settings` | `auxiliary.adaptive_control.settings`（归档） |
| VisPhy | `src.adaptive` | `mac.adaptive.offline` |
| VisPhy | `src.adaptive.baseline` | `mac.adaptive.offline.baseline` |
| VisPhy | `src.adaptive.condition_grid` | `mac.adaptive.offline.condition_grid` |
| VisPhy | `src.adaptive.controller` | `mac.adaptive.offline.controller` |
| VisPhy | `src.adaptive.healnet_prefix` | `mac.adaptive.offline.healnet_prefix` |
| VisPhy | `src.adaptive.metrics` | `mac.adaptive.offline.metrics` |
| VisPhy | `src.adaptive.replay` | `mac.adaptive.offline.replay` |

### `mac.realtime`

| 来源 | 旧 import | 新 import |
| --- | --- | --- |
| RTML | `real_time_ml.policy` | `mac.realtime.policy` |
| RTML | `real_time_ml.policy.recommender` | `mac.realtime.policy.recommender` |
| RTML | `real_time_ml.realtime` | `mac.realtime` |
| RTML | `real_time_ml.realtime.cycle` | `mac.realtime.cycle` |
| RTML | `real_time_ml.realtime.engine` | `mac.realtime.engine` |
| RTML | `real_time_ml.realtime.replay` | `mac.realtime.replay` |
| RTML | `real_time_ml.realtime.serve` | `mac.realtime.serve` |
| RTML | `real_time_ml.realtime.video_replay` | `mac.realtime.video_replay` |

### `mac.runtime`

| 来源 | 旧 import | 新 import |
| --- | --- | --- |
| RTML | `real_time_ml.runtime` | `mac.runtime` |
| RTML | `real_time_ml.runtime.engine` | `mac.runtime.engine` |
| RTML | `real_time_ml.runtime.replay` | `mac.runtime.replay` |
| RTML | `real_time_ml.runtime.serve` | `mac.runtime.serve` |

### `mac.reporting`

| 来源 | 旧 import | 新 import |
| --- | --- | --- |
| RTML | `real_time_ml.modeling.latest_multimodal_report` | `mac.reporting.latest_multimodal_report` |
| RTML | `real_time_ml.modeling.video_encoder_report` | `mac.reporting.video_encoder_report` |
| RTML | `real_time_ml.modeling.video_report` | `mac.reporting.video_report` |
| RTML | `real_time_ml.reporting` | `mac.reporting` |
| RTML | `real_time_ml.reporting.summary` | `mac.reporting.summary` |
| VisPhy | `src.utils.registry` | `mac.reporting.registry` |
| VisPhy | `src.utils.reporting` | `mac.reporting.experiment` |

### `mac.config`

| 来源 | 旧 import | 新 import |
| --- | --- | --- |
| RTML | `real_time_ml.config` | `mac.config` |
| RTML | `real_time_ml.config.layers` | `mac.config.layers` |
| RTML | `real_time_ml.config.output` | `mac.config.output` |
| VisPhy | `src.utils.config` | `mac.config.simple` |

### `mac.utils`

| 来源 | 旧 import | 新 import |
| --- | --- | --- |
| RTML | `real_time_ml.utils` | `mac.utils.io` |
| VisPhy | `src.utils.logging_setup` | `mac.utils.logging_setup` |
| VisPhy | `src.utils.tb` | `mac.utils.tb` |

### 顶层模块

| 来源 | 旧 import | 新 import |
| --- | --- | --- |
| RTML | `real_time_ml` | `mac` |

### `mac.__main__`

| 来源 | 旧 import | 新 import |
| --- | --- | --- |
| RTML | `real_time_ml.__main__` | `mac.__main__` |

### `mac.cli`

| 来源 | 旧 import | 新 import |
| --- | --- | --- |
| RTML | `real_time_ml.cli` | `mac.cli` |

### `mac.schema`

| 来源 | 旧 import | 新 import |
| --- | --- | --- |
| RTML | `real_time_ml.schema` | `mac.schema` |

### `mac.windows_rq2_representations`

| 来源 | 旧 import | 新 import |
| --- | --- | --- |
| RTML | `real_time_ml.windows_rq2_representations` | `mac.windows_rq2_representations` |

---

## 融合暴露出的问题（v1 未处理）

### 已知断链（融合前就存在）

`scripts/relax_foundation/rebuild_relax_handcrafted_features.py:59` 导入
`mac.eeg_contract`（原 `real_time_ml.eeg_contract`）。**该模块在融合前的两个仓库里都不存在**，
现在也不存在。这不是融合造成的，但融合让它变得可见了。需要定位原实现或重建。
同文件 `:66-68` 依赖它做 EEG 通道契约校验，因此这条路径目前跑不通。

### 融合带来的唯一测试回归：Windows DLL 枚举竞态

`tests/test_cross_project_alignment.py::test_validation_ranking_uses_dedicated_validation_rows`
在整套同进程运行时失败，抛的是：

```
threadpoolctl.py:1099: in _find_libraries_with_enum_process_module_ex
    raise OSError("GetModuleFileNameEx failed")
```

**不是断言失败**，是 `threadpoolctl`（3.6.0）在 Windows 上枚举进程已加载 DLL 时，
`EnumProcessModulesEx` 与 `GetModuleFileNameEx` 之间发生模块卸载导致的竞态。

已确证的归因：

| 场景 | 结果 |
| --- | --- |
| 单独跑这个测试 | 通过 |
| 融合仓库里只跑 RTML 那批测试文件（59 个） | 全部通过 |
| 融合仓库整套同进程 | 失败 |
| 融合仓库每个测试文件各起进程（63 个） | 通过，且总数与基线逐项相等 294/278/3/13/7 |
| 原 `Relax-Model` 整套（两次） | 通过，从不出现 |
| 原 `Relax-Model` 里先 `import torch` 再跑该测试 | 通过（单纯预加载 torch 不足以复现） |
| 加 `OMP/MKL/OPENBLAS_NUM_THREADS=1` | 无效 |

原因是融合后两套依赖栈（torch + sklearn + mne + cv2 + …）进入同一个进程，
加载的 DLL 变多，那次枚举更容易撞上竞态。

**规避**：让每个测试文件各起一个进程，已实测可完全消除。
省事的做法是 `pip install pytest-xdist` 后 `pytest -n 4 --dist loadfile`。
v2 可以考虑把 `pytest-xdist` 加进 `dev` extra 并在 `addopts` 里默认按文件分发。

### 待去重的重复实现

第一版刻意全部保留。它们功能重叠但不一定等价，合并前需逐个核对语义。

| 重复项 | 位置 |
| --- | --- |
| `file_sha256` ×3 | `mac.adaptive.offline.healnet_prefix`、`mac.utils.io`、`mac.evaluation.alignment` |
| `write_json` ×2 | `mac.data.relax_foundation`、`mac.utils.io` |
| `concordance_correlation_coefficient` ×2 | `mac.evaluation.metrics`、`mac.evaluation.dynamic_texture_five` |
| `adjacent_conditions` ×2 | `mac.adaptive.offline.condition_grid`、`mac.realtime.policy.recommender` |
| 配置深合并 ×2 | `mac.config.simple.merge_configs`（VisPhy）、`mac.config.layers.deep_merge`（RTML） |
| 配置哈希 ×2 | `mac.config.simple.config_hash`、`mac.config.ProjectConfig.write_run_manifest` 内联 sha256 |

### 两套并存的机制

| 机制 | VisPhy 侧 | RTML 侧 | 说明 |
| --- | --- | --- | --- |
| 配置加载 | `mac.config.simple.load_config(path) -> dict` | `mac.config.load_config(path=None) -> ProjectConfig` | **同名异义**，已用命名空间隔开，VisPhy 那份刻意不在 `mac.config.__init__` 里 re-export |
| 配置组织 | 扁平 YAML + `--config` | `project.yaml → base.yaml → experiments/ → local.yaml` 分层 | 最终应收敛为一套 |
| 交叉验证 | LOSO（leave-one-subject-out） | LOPO（leave-one-participant-out） | 同一件事的两个名字 |
| 报告生成 | `mac.reporting.experiment` + `registry` | `mac.reporting.summary` | |

### 其他 v2 待办

- 硬编码的 WSL / Windows 绝对路径改为仓库内相对路径，例如：
  - `analysis/supplementary/evaluate_eeg_eligible_ablation.py:570`
  - `analysis/supplementary/audit_cross_project_alignment.py:1074`
  - `analysis/supplementary/evaluate_project_a_dynamic_texture_five.py:51`
  - `scripts/relax_foundation/extract_relax_foundation_embeddings.py` 的 `--relax-model-src`
    （现在可以直接指向包内模块，不再需要注入外部 `src/`）
- `scripts/` 80 个文件按研究线重排（需同步把 `parents[1]` 改成 `parents[2]`）
- PaPaGei 权重（3 个共 63MB）迁 Git LFS 或移出版本控制
- 删除 `src/real_time_ml/` 与 `src/src/` 兼容层，以及 `rtml` CLI 别名
- `docs/`、`reports/`、`auxiliary/` 中的历史文档仍写着旧路径（如 `configs/base.yaml`、
  `src/utils/config.py`）。这些是当时状态的记录，**刻意未改**；复用其中的命令时对照本文翻译。

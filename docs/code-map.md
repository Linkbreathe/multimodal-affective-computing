# 代码地图与协议边界

融合后的统一版本。旧路径请对照 [合并对照表](MERGE-MAP.md)。
论文主线与辅助边界见 [论文代码范围](thesis-scope.md)。

## 公共模块（`src/mac/`）

| 模块 | 职责 / 使用方 |
| --- | --- |
| `data/index.py`、`io.py`、`labels.py`、`tables.py`、`xlsx.py`、`video.py` | RELAX 会话源索引、标签解析、表格与视频 I/O |
| `data/alignment.py` | marker 对齐与窗口构建 |
| `data/condition_data.py` | condition 级标签列定义与窗口聚合 |
| `data/segments.py`、`label_builder.py` | EgoEmotion 兼容数据适配，供辅助基准和旧缓存使用 |
| `data/egoemotion/manifest.py` | EgoEmotion 10 秒片段 manifest、完整性检查与数据加载；辅助基准支持 |
| `data/embedding_shapes.py`、`collate.py`、`dataset.py` | 缓存形状规范、变长批处理、缓存嵌入 Dataset |
| `data/relax_*.py` | RELAX 各协议的 Dataset、cohort 审计与注意力视频派生 |
| `preprocessing/pipeline.py`、`mne_qc.py` | 流式生理处理（`StreamingPhysioProcessor`）与连续 MNE 质控审计 |
| `preprocessing/ppg.py`、`ecg.py`、`relax_physio.py` | RELAX 主线的采样、滤波、通道与单位处理 |
| `preprocessing/seedv.py` | SEED-V 辅助基准的采样、滤波、通道与单位处理 |
| `features/physio.py`、`eye.py`、`head.py`、`video.py`、`egocentric.py` | 手工特征：EEG/ECG、视线、头动、第一人称视频 |
| `features/videomae2.py`、`dynamic_texture.py` | 冻结 VideoMAE v2 嵌入、研究专用动态纹理描述子 |
| `encoders/` | 预训练编码器适配、注册、特征提取及缓存 |
| `fusion/factory.py` | 配置驱动构建、投影与 mask-aware pooling |
| `fusion/` 其余模块 | early/mid/late、Perceiver、Q-Former、HEALNet、MM-Lego、蒸馏、冻结压缩，以及最小 Ridge / 1D-CNN 融合基准 |
| `models/` | EEG/LoRA/头动模型；condition 级经典模型、时序 1D-CNN、视觉模型；特征分组与校验 |
| `tasks/` | 任务 head/loss 与 condition 控制 |
| `training/` | 公共 LOSO 训练与早停；condition / state / policy / video 训练入口 |
| `evaluation/` | 指标、被试折契约、LOPO、动态纹理统计、部署安全门 |
| `adaptive/offline/` | baseline、冻结 HEALNet 前缀、控制器及离线回放指标 |
| `realtime/` | Shadow 时钟、缓冲、engine、replay、serve 与推荐策略 |
| `reporting/`、`config/`、`utils/` | 报告与结果注册表、分层与扁平配置、原子写入与哈希 |

`Auxiliary/benchmarks/egoemotion/scripts/run_experiment.py` 保留 EgoEmotion 历史入口，
直接调用 `mac` 的公共实现；PPG 的 `HybridProjectedFusion` 继承公共包装器，仅增加原始
信号编码。论文主线不依赖该入口。

## 实验入口

### `mac` CLI（原 `rtml`，论文主线子命令）

索引与预处理 `index` `preprocess`；特征 `extract-features` `build-video-mp4`
`extract-handcrafted-video` `extract-videomae2` `extract-dynamic-texture`；
训练 `train-state` `train-dcnn-state` `train-policy` `train-video-ml`
`train-videomae2-dcnn` `train-realtime-multimodal-window` 等；
基准 `benchmark-minimal-fusion` `benchmark-minimal-fusion-dcnn`；
评测与报告 `evaluate` `report` `report-video-fusion` `report-latest-multimodal`；
运行时 `replay` `replay-video` `serve`；
编排 `run-all`。

### 脚本入口

| 研究线 | 准备 / 训练 | 检查 / 评估 |
| --- | --- | --- |
| EgoEmotion（辅助） | `Auxiliary/benchmarks/egoemotion/scripts/segment_and_extract_10s.py`、`extract_embeddings.py`、`pretrain_*`、`run_experiment*.py`、`run_finetune_ppg.py`、`run_lego_experiment.py` | 同目录下的 `verify_*`、`audit_clip_leakage_ego.py`、历史报告 |
| SEED-V（辅助） | `Auxiliary/benchmarks/seedv/scripts/extract_seedv*`、`run_seedv_*`：EEGPT、REVE、EEGNet、DE-SVM、眼动、LoRA、微调 | 同目录下的 `run_seedv_ablation*.py` 与各入口输出 |
| RELAX foundation | `relax_foundation/extract_relax_foundation_embeddings.py`、`run_relax_foundation_probe.py`、`run_relax_ablation_suite.py`、`run_relax_claim_validation.py`、`run_relax_attention_video_experiments.py` | 子目录内 `audit_*`、`analyze_*`、`summarize_*`、`generate_*` |
| RELAX aligned | `build_relax_alignment_cache.py`、`run_relax_foundation_probe.py`、`run_relax_aligned_fusion_matrix.py`、`run_relax_eeg_eligible_ablation_matrix.py` | `audit_relax_foundation_features*.py` |
| RELAX compression | `build_relax_compression_fusion_preregistration_v2.py`、`run_relax_compression_fusion*.py`、`run_relax_compression_fusion_matrix*.py` | `evaluate_relax_compression_fusion*.py`、`plot_relax_compression_fusion.py`、`build_relax_compression_fusion_final_report.py` |
| RELAX condition anchor | `run_relax_condition_anchor_probe.py`、`run_relax_condition_anchor_matrix.py` | `evaluate_relax_condition_anchor.py` |
| RELAX embedding ladder | `build_relax_embedding_ladder.py`、`run_relax_embedding_ladder_fusion.py` | `audit_relax_embedding_ladder.py`、`evaluate_relax_embedding_ladder_fusion.py`、`build_relax_embedding_ladder_report.py` |
| RQ2 | `run_rq2_wsl.py --audit-only` 审计，然后同入口调用 `rq2_pipeline.py`；`run_rq2_modality_ablation.py` | `wsl_results/`、`combined/` |
| HEALNet replay | `run_healnet_prefix_inference.py`、`run_healnet_adaptive_replay.py` | trace、metrics、report、checksums |

### 分析入口（`analysis/`）

`adaptive_offline/`（离线测量与控制模拟）、`decision_reanalysis/`（录制决策复分析）、
`idiographic/`（个体化分析）、`phase_baseline/`（阶段 / 基线）、
`supplementary/`（共享 split 比较、模态消融、图表与报告，38 个脚本）。

### 归档运行时

`Auxiliary/adaptive_control/` 保存暂不参与论文主线的 Unity/UDP Adaptive Control
服务、模型注册表、配置、启动器和专用测试。它不再属于 `src/mac/` 的活动包地图；
需要历史复现时使用 `python -m Auxiliary.adaptive_control.cli`。

## RELAX 协议边界

| 属性 | Foundation sample cache | 跨项目 aligned cache |
| --- | --- | --- |
| Dataset 模块 | `mac.data.relax_foundation_dataset` | `mac.data.relax_dataset` |
| Runner | `scripts/relax_foundation/run_relax_foundation_probe.py` | `scripts/run_relax_foundation_probe.py` |
| 顶层结构 | `samples` 列表、`embedding_dims` | `participant_ids`、`conditions`、`presentation_positions`、`targets`、`embeddings`、`masks` |
| 样本单位 | 每个 sample 内含 participant、condition、labels 和各模态窗口 | 并行数组按 participant-condition 对齐 |
| mask | sample 内各模态 mask | 缓存 mask 与外部共享 mask 取交集 |
| 评估 | 原始目标、condition baseline、residualization 等选项 | 显式 split/label/window manifests，锁定 original/none/sequence |

两个 Dataset 均名为 `RelaxConditionEmbeddingDataset`，但输入不兼容。不能互换 runner 或自动混用缓存。
aligned cache 由 `build_relax_alignment_cache.py` 构建。

压缩 v1/v2、condition anchor、embedding ladder 分别保留；版本涉及不同预注册方法、维度分配或评估约定。
固定 cohort、seed、hash 和外部 manifest 属于实验身份。

## 跨项目交接（融合后的变化）

融合前，RELAX foundation 的特征提取需要把另一个仓库的 `src/` 注入 `sys.path`
（`--relax-model-src /home/link/Wei/Models/core/Relax-Model/src`）。现在
`mac.data.video`、`mac.features.extract`、`mac.features.physio`、`mac.data.condition_data`
都是包内模块，普通 import 即可。相关 CLI 参数仍保留但已非必需，见
[合并对照表](MERGE-MAP.md) 的 v2 待办。

RQ2 的文件系统握手（Windows 侧写 `WINDOWS_DONE.json` → `scripts/run_rq2_wsl.py --shared-root` 读取）
仍按原样工作，两端现在在同一个仓库里。

## 两套并存的机制

| | VisPhy 侧 | RTML 侧 |
| --- | --- | --- |
| 配置加载 | `mac.config.simple.load_config(path) -> dict` | `mac.config.load_config(path=None) -> ProjectConfig` |
| 配置组织 | 扁平 YAML + `--config` | `project.yaml → base.yaml → experiments/ → local.yaml` |
| 交叉验证命名 | LOSO | LOPO |

两个 `load_config` 同名异义，已用命名空间隔开：扁平那份**不在** `mac.config.__init__` 里 re-export。

## 兼容与迁移

顶层历史 CLI 保留，避免破坏已有命令和脚本调度。`src/real_time_ml/` 与 `src/src/`
是自动生成的兼容 shim，叶子模块通过 `sys.modules` 别名到真实模块，新旧名字拿到同一个对象。
`rtml` 保留为 `mac` 的 CLI 别名。三者都计划在 v2 删除。

公共实现优先加入 `src/mac/` 中它所属的管线阶段。涉及 hand-off、预注册及报告的实验函数
仍保留在各自脚本中；本轮整理没有重新定义这些研究方法。
部分历史入口仍含机器相关路径，应先查看 CLI 和对应常量，再准备原实验输入。

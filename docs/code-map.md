# 代码地图与协议边界

## 公共模块

| 模块 | 职责 / 使用方 |
| --- | --- |
| `src/data/segments.py`、`label_builder.py` | EgoEmotion 时间轴切片与标签映射 |
| `src/data/egoemotion/manifest.py` | 10 秒片段 manifest、完整性检查与数据加载 |
| `src/data/embedding_shapes.py`、`collate.py` | 缓存形状规范与变长批处理 |
| `src/data/*preprocessing.py` | 按模态处理信号、采样、通道和单位 |
| `src/encoders/` | 模型适配、注册、特征提取及缓存 |
| `src/fusion/factory.py` | 配置驱动构建、投影与 mask-aware pooling |
| `src/fusion/` 其余模块 | early/mid/late、Perceiver、Q-Former、HEALNet、MM-Lego、蒸馏及冻结压缩 |
| `src/models/`、`src/tasks/` | EEG/LoRA/头动模型、任务 head/loss 与条件控制 |
| `src/trainer/`、`src/utils/` | 公共 LOSO 训练、指标、配置与结果管理 |
| `src/adaptive/` | baseline、冻结 HEALNet 前缀、控制器及离线指标 |

`scripts/run_experiment.py` 重导出迁移后的公共函数/类以兼容历史导入与序列化引用。10 秒入口直接使用公共实现；PPG 的 `HybridProjectedFusion` 继承公共包装器，仅增加原始信号编码。

## 实验入口

以下路径相对 `scripts/`；foundation 行的辅助脚本也位于 `relax_foundation/`。

| 研究线 | 准备 / 训练 | 检查 / 评估 |
| --- | --- | --- |
| EgoEmotion | `segment_and_extract_10s.py`、`extract_embeddings.py`、`pretrain_*`、`run_experiment*.py`、`run_finetune_ppg.py`、`run_lego_experiment.py` | `verify_pipeline.py`、`verify_segmentation.py`、`audit_clip_leakage_ego.py`、`analyze_results.py` |
| SEED-V | `extract_seedv*`、`run_seedv_*`：EEGPT、REVE、EEGNet、DE-SVM、眼动、LoRA、微调 | `run_seedv_ablation*.py` 与各入口输出 |
| RELAX foundation | `relax_foundation/extract_relax_foundation_embeddings.py`、`run_relax_foundation_probe.py`、`run_relax_ablation_suite.py`、`run_relax_claim_validation.py`、`run_relax_attention_video_experiments.py` | 子目录内 `audit_*`、`analyze_*`、`summarize_*`、`generate_*` |
| RELAX aligned | `build_relax_alignment_cache.py`、`run_relax_foundation_probe.py`、`run_relax_aligned_fusion_matrix.py`、`run_relax_eeg_eligible_ablation_matrix.py` | `audit_relax_foundation_features*.py` |
| RELAX compression | `build_relax_compression_fusion_preregistration_v2.py`、`run_relax_compression_fusion*.py`、`run_relax_compression_fusion_matrix*.py` | `evaluate_relax_compression_fusion*.py`、`plot_relax_compression_fusion.py`、`build_relax_compression_fusion_final_report.py` |
| RELAX condition anchor | `run_relax_condition_anchor_probe.py`、`run_relax_condition_anchor_matrix.py` | `evaluate_relax_condition_anchor.py` |
| RELAX embedding ladder | `build_relax_embedding_ladder.py`、`run_relax_embedding_ladder_fusion.py` | `audit_relax_embedding_ladder.py`、`evaluate_relax_embedding_ladder_fusion.py`、`build_relax_embedding_ladder_report.py` |
| RQ2 | `run_rq2_wsl.py --audit-only` 审计，然后同入口调用 `rq2_pipeline.py`；`run_rq2_modality_ablation.py` | `wsl_results/`、`combined/` |
| HEALNet replay | `run_healnet_prefix_inference.py`、`run_healnet_adaptive_replay.py` | trace、metrics、report、checksums |

## RELAX 协议边界

| 属性 | Foundation sample cache | 跨项目 aligned cache |
| --- | --- | --- |
| Dataset 模块 | `src.data.relax_foundation_dataset` | `src.data.relax_dataset` |
| Runner | `scripts/relax_foundation/run_relax_foundation_probe.py` | `scripts/run_relax_foundation_probe.py` |
| 顶层结构 | `samples` 列表、`embedding_dims` | `participant_ids`、`conditions`、`presentation_positions`、`targets`、`embeddings`、`masks` |
| 样本单位 | 每个 sample 内含 participant、condition、labels 和各模态窗口 | 并行数组按 participant-condition 对齐 |
| mask | sample 内各模态 mask | 缓存 mask 与外部共享 mask 取交集 |
| 评估 | 原始目标、condition baseline、residualization 等选项 | 显式 split/label/window manifests，锁定 original/none/sequence |

两个 Dataset 均名为 `RelaxConditionEmbeddingDataset`，但输入不兼容。不能互换 runner 或自动混用缓存。aligned cache 由 `build_relax_alignment_cache.py` 构建。

压缩 v1/v2、condition anchor、embedding ladder 分别保留；版本涉及不同预注册方法、维度分配或评估约定。固定 cohort、seed、hash 和外部 manifest 属于实验身份。

## 兼容与迁移

顶层历史 CLI 保留，避免破坏已有命令和脚本调度。另一份 worktree 的 foundation 脚本移入子目录时，内部 Python 导入、仓库根路径和子进程 runner 路径已同步调整。

公共实现优先加入 `src/`。涉及 hand-off、预注册及报告的实验函数仍保留在各自脚本中；本轮整理没有重新定义这些研究方法。部分历史入口仍含机器相关路径，应先查看 CLI 和对应常量，再准备原实验输入。

# 整理验证记录

日期：2026-09-11。分支：`refactor/consolidate-research-20260911`。

本文记录的是论文范围整理前的合并验证结果；在当前 `thesis-core-auxiliary` 分支中，
EgoEmotion/SEED-V 专属入口、配置和测试已移至 `auxiliary/benchmarks/`，Adaptive Control
实验服务已移至 `auxiliary/adaptive_control/`，因此本文中的
旧路径仅用于追溯历史验证，不代表当前执行路径。

## 环境与结果

验证解释器为本机 `egoEMOTION` 环境的 Python 3.11；主要依赖为 PyTorch `2.10.0+cu128`、Transformers `5.1.0`、NumPy `1.26.4`、pandas `2.2.3`。现有 `visphy` 环境出现 NumPy/pandas 二进制不兼容，因此未用它作为验证环境，也未修改它。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  HF_HUB_OFFLINE=1 python -m pytest -q --tb=short
python -m compileall -q src scripts tests
git diff --check
```

公共模块重构后，标准 `tests/` 套件 **261 passed**，其中包括本机具备资源的 PaPaGei、ECGFounder 检查。没有跳过或失败。3 条 warning 来自 PatchTST 的 nested-tensor 优化提示（2 条）和 pandas 空表拼接的未来行为提示（1 条）。

初次检查曾从整个工作区收集测试，得到 274 passed / 3 failed；3 个失败来自不存在的本机 label manifests。现已使用临时样例验证这些读取器，其他数据加载与缓存测试也改为临时 EgoEmotion 数据。`pytest.ini` 将标准套件限定为 `tests/`，所以最终总数不包含产物目录里的临时检查。

3 个预训练模型/真实数据测试标记为 `external`。新环境可使用 README 中的 `-m "not external"` 命令选择不依赖这些资源的测试。

以下 11 个 CLI 的 `--help` 均以退出码 0 完成：

- `scripts/run_experiment.py`
- `scripts/run_experiment_10s.py`
- `scripts/run_finetune_ppg.py`
- `scripts/run_seedv_experiment.py`
- `scripts/run_relax_foundation_probe.py`
- `scripts/run_relax_compression_fusion_v2.py`
- `scripts/run_rq2_wsl.py`
- `scripts/run_healnet_adaptive_replay.py`
- `scripts/relax_foundation/run_relax_foundation_probe.py`
- `scripts/relax_foundation/extract_relax_foundation_embeddings.py`
- `scripts/relax_foundation/run_relax_ablation_suite.py`

源码编译和 diff 空白检查通过。

## 本轮边界

没有重跑完整 LOSO 训练、VideoMAE/NeuroRVQ 原始特征提取、外部 hand-off 的真实 RQ2 流水线或在线控制实验。CLI 检查证明入口可加载，不能替代输入数据、权重与完整训练验证。依赖清单使用本机已验证版本，尚未验证全新环境的完整安装。

另一份 foundation worktree 的源文件保留原状。其独立协议的源码已纳入新命名空间；重叠修改的取舍见 [分支记录](branch-history.md)。已有实验数据、结果、压缩包和历史分支均保留。

本轮远端交付是整理分支，main 的后续合入方式见 [分支记录](branch-history.md)。

# Git 分支整理记录

检查日期：2026-09-11，已执行 `git fetch origin`。本记录中的 hash 描述整理开始时的阶段节点；分支引用以后可能移动。图省略中间提交和最初 scaffold 阶段的 merge commits。

```text
main                                      4e9f9cd
  └─ ege/baselines                         09c0156
      └─ ege/training-data-pipeline        3fa8e9c
          └─ seedv/eeg-based-fusion        f55aab2
              └─ origin/seedv/quadmodal-fusion  387b246
                  └─ seedv/quadmodal-fusion     38676b1
                      └─ refactor/deep-restructure-20260427  480082e
                          └─ final/eee-hm-ev   1feea70
                              同位置：feature/relax-foundation-probe-v3
                              └─ refactor/consolidate-research-20260911
```

| 分支 | 阶段职责 |
| --- | --- |
| `main` | 最初 README，尚未包含研究实现 |
| `ege/baselines` | 公共数据/编码器/融合/训练基础、EgoEmotion baselines |
| `ege/training-data-pipeline` | 分段、消融、PPG、蒸馏、微调与阶段报告 |
| `seedv/eeg-based-fusion` | EEGPT / REVE、SEED-V 融合及 LoRA |
| `seedv/quadmodal-fusion` | 后续多模态、40 被试运行、数据目录重组；本地比同名远端多一个提交 |
| `refactor/deep-restructure-20260427` | 代码优化阶段 |
| `final/eee-hm-ev` | ECGFounder 集成；本轮代码基点 |
| `feature/relax-foundation-probe-v3` | 提交点相同，但独立 worktree 含未提交的早期 RELAX 实现 |
| `refactor/consolidate-research-20260911` | 汇集源码、隔离协议、公共模块整理、文档与验证 |

这些分支大多是同一条开发线的阶段指针，逐个 merge/cherry-pick 会重复处理已包含的工作。

## 未提交源码的去向

`02dfeaf` 保存当前工作区原有 VideoMAE 修正、RELAX alignment/compression/ladder、RQ2、adaptive replay 及测试。

`0785c01` 纳入另一份 worktree 的 25 个未跟踪源码/测试文件，并增加子包初始化文件：

- `scripts/*.py` → `scripts/relax_foundation/*.py`，调整内部导入和子进程路径。
- `src/data/relax_dataset.py` → `src/data/relax_foundation_dataset.py`，调整调用方。
- 其余无冲突的 `src/`、`tests/` 文件保留路径。

原 worktree 的文件和分支保持不变。其两个已跟踪文件的修改未直接覆盖公共代码：`registry.py` 删除 PPG 注册会影响 EgoEmotion，因此保留原注册；`video_mae.py` 的全局兼容补丁与当前修正重叠，采用当前工作区的显式模型构建与 strict safetensors 加载。原补丁仍留在原 worktree。

实验数据、输出和压缩包留在本地，新增忽略规则。旧分支及已提交历史文件保留。

## Rebase 与交付

fetch 后 `main` 与 `origin/main` 都为 `4e9f9cd`，是整理分支的祖先。`main...final/eee-hm-ev` 左右独有提交数为 `0 / 59`。实际采用 `git rebase --rebase-merges origin/main` 保留最初的合并节点；完成后提交 hash 和文件树保持不变，无冲突。普通 rebase 会摊平早期 merge，因此未采用其改写后的历史作为交付。

本地备份分支 `archive/consolidation-before-rebase-20260911` 指向整理完成、rebase 前的 `ec424a7`，用于恢复核对，不随本轮整理分支推送。

本轮推送整理分支，不移动共享 main 和历史分支。审查后可通过 PR 合入；如果 main 仍未推进，也可快进：

```bash
git fetch origin
git switch main
git merge --ff-only origin/refactor/consolidate-research-20260911
git push origin main
```

上面是后续合入命令，并非本轮已执行操作。main 若出现独有提交，应重新检查关系并先解决整合问题。

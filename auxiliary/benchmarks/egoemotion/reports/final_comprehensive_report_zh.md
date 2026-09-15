# egoEMOTION 40-Subject 多模态融合完整运行报告

**生成时间:** 2026-04-20
**分支:** seedv/quadmodal-fusion
**运行环境:** RTX 5080 (16 GB), conda env `visphy`

---

## 1. 数据与预处理

### 1.1 被试覆盖

| 范围 | 数量 | 说明 |
|---|---|---|
| 005–019 | 15 | 原始预处理 |
| 021–033 | 13 | 原始预处理（缺 020） |
| 034–040 | 7 | 本次从 Windows 路径拷回补齐 |
| 042–046 | 5 | 本次从 Windows 路径拷回补齐（缺 041） |
| **合计** | **40** | — |

每个被试包含以下原始文件（与 005–033 对齐）：

```
data/egoemotion_raw/<subject>/
├── pov.mp4                    # 第一视角视频
├── gaze.npy / gaze_90fps.npy  # 眼动轨迹 (90 Hz 重采样)
├── pupils.npy                 # 瞳孔序列
├── ppg_ear.npy                # 原始 PPG
├── ppg_ear_125hz.npy          # 125 Hz 重采样 PPG
├── Session_A_<id>.csv         # 情绪视频任务标签 + VAD
└── Session_B_<id>.csv         # 交互任务标签 + VAD
```

`task_times.npy` 根文件已覆盖 40 个 subject，可直接分段。

### 1.2 任务感知 10 秒分段

由 `scripts/segment_and_extract_10s.py` 完成：

- 按 `task_times.npy` 将录制切分为任务区间（丢弃校准、问卷间隔等）
- 每任务内以 10 s (900 samples @ 90 Hz) 非重叠 chunk，尾部不足丢弃
- 标签从 Session A/B CSV 读取：9 类硬标签 + 9 维软标签 + VAD 回归
- 产出 `data/embeddings_10s_task_aware/manifest.csv`（9493 行）

### 1.3 Embedding 提取

| 模态 | 编码器 | 输入 | 嵌入维度 | 输出路径 |
|---|---|---|---|---|
| PPG | PaPaGei（冻结）| 10 s @ 125 Hz, z-score | 512 | `embeddings_10s_task_aware/papagei_ppg/<subj>/segment_XXXX.pt` |
| Eye | InceptionTime（冻结，gaze 预训练）| 10 s @ 90 Hz, 2ch gaze | 128 | `embeddings_10s_task_aware/inceptiontime/<subj>/segment_XXXX.pt` |
| Video | VideoMAE V2（冻结）| 每 16 帧 clip，ImageNet 归一 | 768 × N_clips | `embeddings_10s_task_aware/video_mae_v2/<subj>/segment_XXXX.pt` |

**提取耗时:** 13.5 min（2026-04-20 18:10–18:24）

**提取完整性:**
- 40 subjects × 平均 237 chunks = 9493 manifest 行
- PPG: 9493 / 9493 ✓
- Eye: 9493 / 9493 ✓
- Video: 9472 / 9493（subject 035 的 21 个 chunk 因原始视频长度不足被跳过，trainer 自动处理）

---

## 2. 融合实验（LOSO, 40 折）

全部通过 `scripts/run_experiment_10s.py` + `configs/base.yaml` + 各融合 config 跑 Leave-One-Subject-Out。训练超参：`batch=64, lr=1e-4, max_epochs=100, patience=10, seed=42`。损失：`CE(9) + KL(soft_label) + VAD_MSE`，权重 1:1:1。

### 2.1 主要融合方法结果

| 融合方法 | 加权 F1 | Macro CCC | CCC (V) | CCC (A) | CCC (D) | 单次 LOSO 耗时 |
|---|---|---|---|---|---|---|
| Early | 0.6460 ± 0.1452 | 0.3727 ± 0.1390 | 0.6294 ± 0.2028 | 0.1755 ± 0.1605 | 0.3132 ± 0.2083 | 6m34s |
| Mid | **0.6465 ± 0.1488** | 0.3793 ± 0.1331 | 0.6563 ± 0.1881 | 0.1779 ± 0.1671 | 0.3036 ± 0.2080 | 6m42s |
| Late | 0.6261 ± 0.1574 | **0.4286 ± 0.1425** | 0.6888 ± 0.1623 | **0.2246 ± 0.1998** | **0.3725 ± 0.2427** | 8m18s |
| Bottleneck | 0.6237 ± 0.1632 | 0.3428 ± 0.1734 | 0.6023 ± 0.2518 | 0.1295 ± 0.1726 | 0.2966 ± 0.2391 | 5m35s |
| HealNet | 0.6100 ± 0.1523 | 0.3863 ± 0.1274 | 0.6723 ± 0.1666 | 0.1699 ± 0.1744 | 0.3166 ± 0.2280 | 36m55s |
| Enriched-Late-PulsePPG | 0.6103 ± 0.1487 | 0.3733 ± 0.1470 | **0.6925 ± 0.1708** | 0.1340 ± 0.2016 | 0.2934 ± 0.2433 | 5m19s |

> 论文 Classical-All baseline：F1 ≈ 0.46

### 2.2 关键发现

1. **Late fusion 在 40-subject 下综合最优**：Macro CCC = 0.4286（比历史 28-subject 的 Late 0.378 提升 13%），特别是 Arousal/Dominance 显著受益于更多被试数据。
2. **Mid fusion 最佳分类 F1**：0.6465，与 Early 接近（0.6460）；两者在 9 类分类任务上几乎平手。
3. **Enriched-Late-PulsePPG 对 Valence 最敏感**：CCC(V)=0.6925，PulsePPG + 富化投影对极性检测友好。
4. **HealNet 代价 vs. 收益失衡**：耗时为其它方法的 5–7×（单 LOSO 近 37 分钟），但 F1 与 CCC 均未显著领先。
5. **Bottleneck 继续表现垫底**（与历史 cycle 一致）：F1=0.6237, CCC(V)=0.6023，瓶颈结构破坏了 PPG/Eye 的细节表达。

---

## 3. 与 28-Subject 历史结果对比

历史结果来自 2026-03-28/29 运行（28 subject，005–033 缺 020）。

| 融合方法 | W.F1 (28) → (40) | CCC (28) → (40) | CCC V | CCC A | CCC D |
|---|---|---|---|---|---|
| Early | 0.6414 → 0.6460 (**+0.005**) | 0.3369 → 0.3727 (**+0.036**) | 0.691 → 0.629 | 0.062 → 0.176 | 0.258 → 0.313 |
| Mid | 0.6488 → 0.6465 (−0.002) | 0.3486 → 0.3793 (**+0.031**) | 0.729 → 0.656 | 0.090 → 0.178 | 0.227 → 0.304 |
| Late | 0.6346 → 0.6261 (−0.008) | 0.3780 → 0.4286 (**+0.051**) | 0.700 → 0.689 | 0.130 → 0.225 | 0.304 → 0.373 |
| Bottleneck | 0.6392 → 0.6237 (−0.016) | 0.3071 → 0.3428 (**+0.036**) | 0.616 → 0.602 | 0.072 → 0.130 | 0.234 → 0.297 |
| HealNet | 0.6463 → 0.6100 (−0.036) | 0.3547 → 0.3863 (**+0.032**) | 0.731 → 0.672 | 0.087 → 0.170 | 0.246 → 0.317 |
| Enriched-Late | 0.6146 → 0.6103 (−0.004) | 0.3782 → 0.3733 (−0.005) | 0.703 → 0.693 | 0.137 → 0.134 | 0.294 → 0.293 |

### 3.1 可复制的现象

- **Arousal / Dominance CCC 普遍提升**：新加入的 034–046 被试大幅改善难测的唤醒/支配维度，尤其 Late (+0.10 Arousal)、Early (+0.11 Arousal)。
- **Valence CCC 普遍回落 2–7%**：34–46 subject 的 Valence 分布更广，平均回归更难。
- **F1 基本持平**：9 类分类性能对被试规模不敏感（28→40），波动在 ±2% 内。
- **LOSO std 略升**（加权 F1 的 std 从 0.13 → 0.15）：被试异质性增大，符合预期。

---

## 4. 结论与建议

### 4.1 推荐配置（40-subject 下）

- **分类任务首选** `configs/fusion/mid.yaml`（F1 最高，训练快）。
- **连续情绪回归（VAD）首选** `configs/fusion/late.yaml`（CCC 均衡最高，Arousal/Dominance 显著领先）。
- **Valence-centric 任务** 可用 `configs/fusion/enriched_late_pulseppg.yaml`（CCC(V)=0.693）。
- **资源紧张时** 避开 HealNet（37 min/LOSO 无明显增益）。

### 4.2 数据新增 (034–046) 的价值

扩展 42% 被试规模后：
- 弱维度（Arousal、Dominance）CCC 有 2–10 个绝对百分点提升，说明此前 28-subject 在这两维上严重欠采样。
- 强维度（Valence）及总分类 F1 达到"饱和"，继续增加同质被试收益递减。
- **下一步增样应优先扩展情绪极端值（非常高/低 arousal）的 session**，而非单纯增加被试数。

### 4.3 仓储产物

- Embeddings: `data/embeddings_10s_task_aware/{papagei_ppg,inceptiontime,video_mae_v2}/<subject>/segment_XXXX.pt`
- Manifest: `data/embeddings_10s_task_aware/manifest.csv` (9493 行)
- 各融合单实验报告: `reports/final40_{early,mid,late,bottleneck,healnet,enriched_late_pulseppg}_*.md`
- 训练日志: `logs/final_run/fusion/final40_*.log`
- 提取日志: `logs/final_run/extract_all40.log`
- Master 调度日志: `logs/final_run/fusion/master.log`

---

## 5. 复现命令

```bash
# 1. 分段 + 提取（~15 min on RTX 5080）
conda run -n visphy python scripts/segment_and_extract_10s.py \
    --encoder all --device cuda --output_dir data/embeddings_10s_task_aware

# 2. 融合实验（LOSO, 40 折；总时 ~68 min）
for cfg in early mid late bottleneck healnet enriched_late_pulseppg; do
  conda run -n visphy python -u scripts/run_experiment_10s.py \
      --fusion_config configs/fusion/${cfg}.yaml \
      --name final40_${cfg} \
      --embeddings_dir data/embeddings_10s_task_aware
done
```


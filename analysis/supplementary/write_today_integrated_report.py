from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts" / "reports" / "today_integrated_multimodal_modeling_report_2026-07-02_zh.md"

SOURCES = [
    ("原始联合审阅报告", ROOT / "auxiliary" / "research" / "MODALITY_AND_MODELING_REVIEW_2026-07-02.md"),
    ("逐模态 LOPO 后续探索", ROOT / "artifacts" / "reports" / "modality_lopo_followup_2026-07-02_zh.md"),
    ("个性化校准后的逐模态 LOPO 探索", ROOT / "artifacts" / "reports" / "modality_personalization_followup_2026-07-02_zh.md"),
    ("Physio 家族拆分 LOPO 探索 legacy ECG", ROOT / "artifacts" / "reports" / "physio_family_split_2026-07-02_zh.md"),
    ("NeuroKit2 ECG 特征重算", ROOT / "artifacts" / "reports" / "ecg_neurokit_recompute_2026-07-02_zh.md"),
    (
        "Physio 家族拆分 LOPO 探索 NeuroKit2 ECG",
        ROOT / "artifacts" / "reports" / "physio_family_split_ecg_neurokit_2026-07-02_zh.md",
    ),
    ("NeuroKit2 ECG 修复前后建模对照", ROOT / "artifacts" / "reports" / "ecg_neurokit_comparison_2026-07-02_zh.md"),
]

ARTIFACTS = [
    "artifacts/features/video_ml/condition_features.csv",
    "artifacts/features/video_ml/window_features.csv",
    "artifacts/fusion_minimal/metrics.csv",
    "artifacts/fusion_minimal/oof_predictions.csv",
    "artifacts/reports/modality_lopo_followup/single_modality_summary.csv",
    "artifacts/reports/modality_lopo_followup/top_rankings.csv",
    "artifacts/reports/modality_lopo_followup/baseline_pass_counts.csv",
    "artifacts/reports/modality_lopo_followup/modality_presence_summary.csv",
    "artifacts/reports/modality_lopo_followup/modality_addition_summary.csv",
    "artifacts/reports/modality_personalization_followup/personalized_oof_predictions.csv",
    "artifacts/reports/modality_personalization_followup/personalized_metrics.csv",
    "artifacts/reports/modality_personalization_followup/best.csv",
    "artifacts/reports/modality_personalization_followup/baselines.csv",
    "artifacts/reports/modality_personalization_followup/passes.csv",
    "artifacts/reports/modality_personalization_followup/singles.csv",
    "artifacts/reports/modality_personalization_followup/presence.csv",
    "artifacts/reports/physio_family_split/oof_predictions.csv",
    "artifacts/reports/physio_family_split/metrics.csv",
    "artifacts/reports/physio_family_split/personalized_oof_predictions.csv",
    "artifacts/reports/physio_family_split/personalized_metrics.csv",
    "artifacts/reports/physio_family_split/best_raw.csv",
    "artifacts/reports/physio_family_split/best_personalized.csv",
    "artifacts/reports/physio_family_split/pass_counts.csv",
    "artifacts/reports/physio_family_split/cohort_info.csv",
    "artifacts/reports/physio_family_split/column_counts.csv",
    "artifacts/reports/physio_family_split/selection_audit.csv",
    "artifacts/features/ecg_neurokit/window_features.csv",
    "artifacts/features/ecg_neurokit/condition_ecg_only_features.csv",
    "artifacts/features/ecg_neurokit/condition_features.csv",
    "artifacts/reports/ecg_neurokit/diagnostics.csv",
    "artifacts/reports/ecg_neurokit/errors.json",
    "artifacts/reports/physio_family_split_ecg_neurokit/oof_predictions.csv",
    "artifacts/reports/physio_family_split_ecg_neurokit/metrics.csv",
    "artifacts/reports/physio_family_split_ecg_neurokit/personalized_oof_predictions.csv",
    "artifacts/reports/physio_family_split_ecg_neurokit/personalized_metrics.csv",
    "artifacts/reports/physio_family_split_ecg_neurokit/best_raw.csv",
    "artifacts/reports/physio_family_split_ecg_neurokit/best_personalized.csv",
    "artifacts/reports/physio_family_split_ecg_neurokit/pass_counts.csv",
    "artifacts/reports/physio_family_split_ecg_neurokit/cohort_info.csv",
    "artifacts/reports/physio_family_split_ecg_neurokit/column_counts.csv",
    "artifacts/reports/physio_family_split_ecg_neurokit/selection_audit.csv",
    "artifacts/reports/ecg_neurokit_comparison/single_family_comparison.csv",
    "artifacts/reports/ecg_neurokit_comparison/best_raw_comparison.csv",
    "artifacts/reports/ecg_neurokit_comparison/best_personalized_comparison.csv",
    "artifacts/reports/ecg_neurokit_comparison/pass_counts_comparison.csv",
]

SCRIPTS = [
    "analysis/supplementary/summarize_minimal_fusion_followup.py",
    "analysis/supplementary/evaluate_minimal_fusion_personalized.py",
    "analysis/supplementary/evaluate_physio_family_split.py",
    "analysis/supplementary/recompute_ecg_neurokit_features.py",
    "analysis/supplementary/compare_ecg_neurokit_split.py",
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _artifact_line(path: str) -> str:
    absolute = ROOT / path
    if absolute.exists():
        return f"- `{path}`"
    return f"- `{path}` (missing at report generation time)"


def main() -> None:
    for _, path in SOURCES:
        if not path.exists():
            raise FileNotFoundError(path)

    lines: list[str] = [
        "# 2026-07-02 多模态信号与建模综合报告",
        "",
        "**生成日期:** 2026-07-02",
        "**工作目录:** `C:/Users/linki/amaster/data_collection_v3/analysis/real_time_inference`",
        "**数据范围:** P002-P016，15 名被试，每人 9 个 Condition，共 135 条 participant-condition 标签；"
        "10 秒窗口仅作为特征切片，不作为独立监督样本。",
        "",
        "## 0. 阅读方式",
        "",
        "本报告分为两层：",
        "",
        "1. **综合结论层**：把今日所有信号审阅、逐模态 LOPO、个性化校准、physio 家族拆分、"
        "NeuroKit2 ECG 修复与修复前后对照合并成一个统一判断。",
        "2. **完整附录层**：逐份拼接今日使用的源报告原文。为避免遗漏，附录保留每份报告的完整正文；"
        "若综合结论与附录细节需要追溯，以附录中的具体表格和 CSV 产物为准。",
        "",
        "## 1. 今日总判断",
        "",
        "今天的证据链指向同一个结论：**当前训练/验证机制本身不是主要问题；主要瓶颈是标签与生理/行为信号之间的"
        "跨人稳定映射很弱，且个体评分偏移很大。** 纯 LOPO 群体模型仍不能直接部署。更可行的研究路径是"
        "`个性化校准 + 窄模态候选 + 严格基线比较`。",
        "",
        "具体来说：",
        "",
        "- `video` 在信号层面最强，能稳定复刻强度/频率操纵；但在 LOPO 预测主观 relaxation/discomfort 时没有稳定优势。"
        "它更像刺激/标签代理，不应直接写成用户状态传感器。",
        "- `eye` 的稳定信号主要是观看期注视范围扩大；`eye_gaze_on_painting_fraction` 恒为 0，是采集/填充失效特征。",
        "- `head` 去掉时长混杂后无通用起始模式，但在部分 LOPO 表中对 relaxation/discomfort 有弱增量，"
        "更像行为辅助特征而非核心状态信号。",
        "- `ECG-HR` 的起始 HR 下沉是唯一跨人通用的生理起始标记；它标记进入观看/注意状态，不区分 9 个条件强弱。",
        "- `EEG` 群体级无统一剂量反应；但在 EEG 真正可用的 9 人 cohort 中，EEG/head 组合对 discomfort 有局部增量。",
        "- legacy ECG detector 有明显缺陷；NeuroKit2 重算后 HR/HRV 回到合理范围，legacy ECG 结论不能作为部署证据。",
        "",
        "## 2. 数据质量与模态可用性",
        "",
        "| 模态 | 今日结论 | 可用性/限制 |",
        "|---|---|---|",
        "| EEG | 群体级弱；EEG 可用 9 人中对 discomfort 有局部增量 | P003/P004/P007/P008/P009/P011/P012/P013/P015 为 EEG 可用 cohort；禁用者不应用插补强行进入 EEG 模型 |",
        "| ECG | HR 起始下沉稳定；legacy HR/HRV 不可信；NeuroKit2 后可继续研究 | NeuroKit2 session HR 63.7-96.8 bpm，condition HR median 72.3 bpm，60s RMSSD median 30.7 ms |",
        "| Eye | 注视范围扩大是稳定观看/探索信号 | `eye_gaze_on_painting_fraction` 全 0，需排查采集端 |",
        "| Head | 去混杂后无通用起始模式；建模中有弱辅助价值 | 不应单独解释为稳定生理/心理机制 |",
        "| Video | 最能编码实验操纵，但主要复刻刺激本身 | 适合刺激操纵校验和研究对照，不应直接当主观状态传感器 |",
        "",
        "## 3. 原始联合审阅的关键结论",
        "",
        "原始 `auxiliary/research/MODALITY_AND_MODELING_REVIEW_2026-07-02.md` 给出的主结论是：",
        "",
        "- 唯一稳健的跨人通用生理起始反应是 HR 下沉。",
        "- eye 只有注视空间范围扩大这一条通用行为信号。",
        "- head 去掉时长混杂后无通用模式。",
        "- egocentric video 是唯一既能通用区分起始、又能随强度/频率分级的模态，但主要是在捕捉视觉刺激本身。",
        "- 当前 ML/DL 训练机制干净：sklearn Pipeline、嵌套 LOPO、残差建模、DCNN early stopping 均未发现关键泄漏。",
        "- 模型打不过平凡基线：relaxation MAE 0.186 高于 condition-only 0.183；"
        "discomfort MAE 0.164 高于 history baseline 0.124；`deployable=False`。",
        "- 必须改变评估框定：把个性化/在线校准提高为主路径；把 discomfort 从稀有硬分类改成连续风险、排序或个体内预警；"
        "放开 EEG/physio 变体只用可用 cohort；补 baseline 归一。",
        "",
        "## 4. 逐模态 LOPO：video 没有兑现状态预测优势",
        "",
        "使用 `artifacts/features/video_ml/condition_features.csv` 跑 15 个 P/H/E/V Ridge 组合后：",
        "",
        "| 指标 | 最好组合 | 结果 | 对照基线 | 解释 |",
        "|---|---:|---:|---:|---|",
        "| relaxation MAE | H | 0.1805 | condition-only 0.1829 | 只改善 0.0024，幅度很小 |",
        "| discomfort MAE | P | 0.1387 | history 0.1244 | 低于 condition-only，但仍输给 history |",
        "| full PHEV | PHEV | relaxation 0.2059 / discomfort 0.1583 | 非最佳 | 全模态堆叠不是答案 |",
        "| video only | V | relaxation 0.1998 / discomfort 0.1630 | 不优 | video 强剂量信号没有转化成主观状态预测优势 |",
        "",
        "基线通过情况：relaxation 只有 `H` 低于 condition-only；discomfort 没有任何组合低于 history。",
        "",
        "## 5. 个性化校准：有效的是个人 bias，不一定是传感器",
        "",
        "用每个留出被试前 N 个 condition 的真实评分估计 bias，只在剩余 condition 评估：",
        "",
        "| 校准 | relaxation 最低 MAE | 对照 | discomfort 最低 MAE | 对照 |",
        "|---|---:|---|---:|---|",
        "| N=2 | P = 0.2034 | condition-only 0.1850；personalized condition-only 0.1939 | PHV = 0.1192 | personalized condition-only 0.1176；history 0.1190 |",
        "| N=3 | P = 0.1728 | personalized condition-only 0.1595 更强 | PHE = 0.1211 | personalized condition-only 0.1245；history 0.1111 更强 |",
        "",
        "个性化校准确实让 raw model 变好，尤其 N=3；但 personalized condition-only 与 history baseline 也很强。"
        "这说明当前可解释的大头是被试个人评分偏移，而不是模态特征本身。",
        "",
        "## 6. Physio 家族拆分：legacy ECG 下的发现与风险",
        "",
        "将 `P` 拆成 `G=EEG`、`R=ECG rate/timing`、`Q=ECG-HRV`、`S=ECG signal IQR sensitivity`、"
        "`A=audit-only RR std sensitivity`，并保留 `H/E/V` 后：",
        "",
        "| Cohort | raw relaxation 最低 | raw discomfort 最低 | baseline 关系 |",
        "|---|---:|---:|---|",
        "| all15 | R = 0.1704 | G = 0.1387 | discomfort 仍输 history 0.1244 |",
        "| eeg_available9 | RQV = 0.1690 | GQH = 0.0736 | discomfort 优于 condition-only 0.0756 与 history 0.0864 |",
        "",
        "legacy 结果提示：EEG 可用 9 人中，EEG/head/HRV 组合对 discomfort 有局部增量；但 `S/A` 是敏感性或 audit-only，"
        "不能作为部署证据。更重要的是，legacy ECG detector 已知不可靠，所以必须先修 ECG。",
        "",
        "## 7. NeuroKit2 ECG 修复",
        "",
        "NeuroKit2 重算只替换 `ecg_*` 特征，EEG/head/eye/video、标签、condition context 均沿用原 `video_ml` 表。",
        "",
        "| 项 | 结果 |",
        "|---|---:|",
        "| 参与者 | 15 |",
        "| condition-level 行 | 135 |",
        "| repaired ECG 聚合列 | 220 |",
        "| session HR median 范围 | 63.7-96.8 bpm |",
        "| RR plausible fraction median | 1.000 |",
        "| condition HR median/IQR | 72.3 / 11.6 bpm |",
        "| condition 60s RMSSD median/IQR | 30.7 / 25.5 ms |",
        "",
        "这说明 ECG 数值已经回到合理生理范围。legacy ECG/HRV 结论应降级为修复前对照，不应再作为可部署证据。",
        "",
        "## 8. NeuroKit2 ECG 后的建模变化",
        "",
        "用 repaired ECG 重跑 physio-family split 后：",
        "",
        "| Cohort | raw relaxation 最低 | raw discomfort 最低 | 说明 |",
        "|---|---:|---:|---|",
        "| all15 | QH = 0.1690 | G = 0.1387 | HRV/head 取代 legacy R 成为 relaxation 最优；discomfort MAE 最低仍由 EEG 单家族给出，但仍输 history |",
        "| eeg_available9 | H = 0.1757 | H = 0.0741 | MAE 最低不再依赖 ECG；EEG 相关组合仍有 Spearman 优势 |",
        "",
        "NeuroKit2 前后关键对照：",
        "",
        "| 项 | legacy | neurokit | 解释 |",
        "|---|---:|---:|---|",
        "| all15 单独 R relaxation MAE | 0.1704 | 0.1826 | legacy R 优势被削弱，旧 detector 可能混入异常/幅度信息 |",
        "| all15 单独 Q relaxation MAE | 0.1917 | 0.1731 | repaired HRV 变得更合理 |",
        "| eeg_available9 单独 R relaxation MAE | 0.1746 | 0.1865 | ECG-rate 不再是 9 人 cohort 的主要解释 |",
        "| eeg_available9 raw discomfort 最好 | GQH = 0.0736 | H = 0.0741 | 差异很小，MAE 最低不再依赖 ECG |",
        "",
        "NeuroKit2 + N=2 personalized 下，all15 discomfort 出现 4 个组合低于 history：`GQH/GQHV/GRQH/QHV`；"
        "legacy 为 0。这是 repaired HRV 可能对 discomfort 有增量的最有价值信号，但仍属于离线研究结果。",
        "",
        "## 9. 当前可执行判断",
        "",
        "1. **不能部署当前群体 ML/DL 模型。** 训练流程干净，但整体指标未过基线和安全门。",
        "2. **real-time 之前要先固定离线问题。** 当前优先级是 offline narrow benchmark，不是接入实时控制。",
        "3. **ECG 只保留 NeuroKit2 版本继续研究。** legacy ECG 与 audit-only RR std 不得进入部署叙事。",
        "4. **视频不能直接当用户状态特征。** 如果继续用 video，应拆成刺激复刻成分与用户行为残差成分。",
        "5. **个性化校准是主路径。** 未来评估必须同时报告 raw、condition-only、personalized condition-only、history，"
        "并只把超过 personalized condition-only/history 的结果视为传感器增量。",
        "6. **EEG 分析必须分 cohort。** all15 与 EEG-available9 不能混说；EEG 缺失不应用中位数插补掩盖。",
        "",
        "## 10. 推荐下一步",
        "",
        "建议下一步只做一个更窄、更干净的离线候选：",
        "",
        "- 输入：`artifacts/features/ecg_neurokit/condition_features.csv`。",
        "- 模态：`G=EEG`、`Q=NeuroKit2 HRV`、`R=NeuroKit2 HR/RR`、`H=head`。",
        "- Cohort：分别报告 `all15` 与 `eeg_available9`。",
        "- 评估：raw LOPO、personalized N=2、personalized N=3。",
        "- 基线：condition-only、personalized condition-only、history。",
        "- 目标：不要再扩大模态，只判断 `G/Q/R/H + personalization` 是否有稳定增量。",
        "",
        "只有这一步仍能稳定超过 baseline，才值得讨论 real-time 的 shadow-only 监测形态。",
        "",
        "## 11. 今日产物清单",
        "",
        "### 报告源文件",
        "",
    ]

    for title, path in SOURCES:
        lines.append(f"- {title}: `{path.relative_to(ROOT).as_posix()}`")

    lines.extend(["", "### 数据产物", ""])
    lines.extend(_artifact_line(path) for path in ARTIFACTS)
    lines.extend(["", "### 脚本产物", ""])
    lines.extend(_artifact_line(path) for path in SCRIPTS)

    lines.extend([
        "",
        "---",
        "",
        "# 附录 A：完整源报告原文",
        "",
        "以下逐份拼接今日所有源报告原文，保留原报告表格、结论和边界说明。",
        "",
    ])

    for index, (title, path) in enumerate(SOURCES, start=1):
        relative = path.relative_to(ROOT).as_posix()
        content = _read(path)
        lines.extend([
            f"## A{index}. {title}",
            "",
            f"**来源:** `{relative}`",
            "",
            content,
            "",
            "---",
            "",
        ])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()

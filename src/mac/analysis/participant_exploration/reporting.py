"""Text-only Chinese reporting and export documentation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import verify_lock, write_csv
from .specification import FEATURES, OUTCOMES, PARTICIPANTS, VARIABLES

ZH = {
    "relaxation": "放松评分",
    "discomfort": "不适评分",
    "eeg_alpha_power": "EEG α 功率",
    "eeg_beta_power": "EEG β 功率",
    "eeg_theta_power": "EEG θ 功率",
    "ecg_mean_hr": "平均心率",
    "ecg_rmssd": "RMSSD",
    "gaze_mean_angular_velocity": "平均记录视线角速度",
    "gaze_fixation_time_proportion": "低速注视时间比例",
    "head_mean_angular_speed": "平均头部角速度",
    "head_orientation_dispersion": "头部朝向离散度",
    "video_mean_optical_flow": "平均光流速度",
    "video_optical_flow_temporal_sd": "光流速度时间标准差",
}

GUIDE = {
    "participant_condition_features.csv": "81 条参与者×条件记录；原始评分、十一项特征原值及分析尺度值、真实呈现顺序、条件参数。",
    "feature_quality.csv": "891 条参与者×条件×特征质量记录；可用性、有效时长、覆盖率、缺失原因与警告。",
    "within_person_centered.csv": "13 个变量的长表；原值、分析值、参与者均值、偏差及各均值的有效条件数。",
    "participant_descriptive_summary.csv": "参与者×变量描述；原始/分析尺度分别给出数量、均值、样本标准差、中位数、最小、最大和极差。",
    "individual_associations.csv": "全部 198 条个体特征–评分相关，包含不可估计条目、原因及纳入条件。",
    "condition_response_summary.csv": "各条件×变量的个体内中心化响应；有效人数、均值、中位数、极值和正负零人数。",
    "cross_participant_summary.csv": "全部 22 条特征×结果汇总；个体相关方向人数、中位数、四分位数和极差。",
    "variables.csv": "13 个变量的中文名称、模态、原始/分析单位和变换。",
    "data_dictionary.csv": "七个核心结果表逐字段字典；长表数值的具体单位通过 variables.csv 关联。",
    "feature_specification.json": "查看关联前已冻结的完整计算和质量规则；哈希见 resolved_config.json。",
    "feature_definitions_zh.txt": "计算定义、原实现差异、质量规则与解释限制。",
    "input_files.csv": "原始输入和论文/代码来源的路径、字节数及 SHA-256。",
    "recording_sources.csv": "每人选用的 XDF、Unity 日志与备份选择。",
    "condition_manifest.csv": "评分、条件、真实顺序和双时钟边界的对齐主表。",
    "condition_alignment_audit.csv": "不含评分值的边界、身份映射、时钟残差核查。",
    "source_issues.csv": "源文件审计发现的缺失项目；空表表示该阶段未发现缺失，不表示全部特征有效。",
    "eeg_channel_quality.csv": "TP9/TP10 各两秒块质量指标及可用块功率；仅为核查，不增加核心特征或额外相关搜索。",
    "ecg_rr_audit.csv": "RR 时间、原始序号、局部中位数、剔除原因及是否位于固定 RMSSD 时段。",
    "consumed_video_frames.csv": "实际读取的图像输入、时间源、解码状态及哈希；未输出任何图像。",
    "modality_diagnostics.csv": "眼/头采样、视频覆盖率、来源与坐标几何检查。",
    "resolved_config.json": "本机路径、固定规范哈希、冻结时间、Git 信息和运行环境版本。",
    "verification.json": "维度、数学一致性、文件保护和无图形输出检查。",
    "input_integrity_verification.csv": "分析前后源输入哈希比较。",
    "report_zh.txt": "中文结果报告，保留无关联、不一致和不可估计结果。",
    "thesis_supplement_zh.txt": "可自行纳入论文的独立中文段落草稿；未修改论文。",
    "RUNNING.txt": "分阶段及完整执行说明。",
    "extreme_observation_flags.csv": "原始尺度3×IQR极端值提示；没有据此删除数值或重算相关。",
    "existing_prediction_reference.csv": "当前论文已保存的MAE与条件锚点；仅换算为1–7评分单位，不训练模型。",
    "analysis_code_manifest.csv": "实际分析源码与测试文件的路径、SHA-256和字节数。",
    "COMPLETE.json": "最终核验成功的运行完成标记。",
}


def fmt(value, digits=3) -> str:
    return "不可估计" if pd.isna(value) else f"{value:.{digits}f}"


def variable_records() -> list[dict]:
    rows = []
    for variable in VARIABLES:
        modality, unit, transform, description = FEATURES.get(
            variable, ("Rating", "1–7 points", "identity", "Raw questionnaire rating")
        )
        rows.append(
            dict(
                variable=variable,
                name_zh=ZH[variable],
                modality=modality,
                raw_unit=unit,
                analysis_unit=f"ln(value / 1 {unit})" if transform == "ln" else unit,
                transformation=transform,
                description=description,
            )
        )
    return rows


FIELD_DESCRIPTIONS = {
    "participant_id": ("匿名参与者 ID；固定九人队列", "identifier"),
    "condition": ("视觉条件标识 C1–C9，不代表时间顺序", "identifier"),
    "intensity": ("名义视觉强度参数", "renderer units"),
    "frequency": ("名义视觉频率参数", "renderer units; not assumed Hz"),
    "intensity_index": ("强度索引，从 0 开始", "index"),
    "frequency_index": ("频率索引，从 0 开始", "index"),
    "intensity_level": ("强度级别，从 1 开始", "level"),
    "frequency_level": ("频率级别，从 1 开始", "level"),
    "presentation_order": ("事件时间核验的实际呈现位置，1–9", "position"),
    "label_source_row": ("问卷工作簿中的源行号，从 1 开始", "row"),
    "xdf_is_backup": ("是否使用保留的备份 XDF", "boolean"),
    "logged_identity": ("原始日志身份值，不修改原文件", "identifier"),
    "identity_alias_applied": ("是否使用明确记录的 P003/P001 别名映射", "boolean"),
    "alignment_status": ("available 或 unavailable，表示边界是否可用于对齐", "category"),
    "alignment_reason": ("对齐不可用的原因", "text"),
    "alignment_warning": ("是否存在大于 50 ms 的残差或边界差异", "boolean"),
    "start_xdf": ("条件开始的同步 XDF 时间", "seconds"),
    "end_xdf": ("条件结束的同步 XDF 时间", "seconds"),
    "start_unix_ms": ("XDF marker payload 的开始 Unix 时间", "Unix milliseconds"),
    "end_unix_ms": ("XDF marker payload 的结束 Unix 时间", "Unix milliseconds"),
    "unity_start_unix_ms": ("Unity condition_start 日志的 Unix 时间", "Unix milliseconds"),
    "unity_end_unix_ms": ("Unity condition_end 日志的 Unix 时间", "Unix milliseconds"),
    "duration_s": ("XDF 条件结束减开始的时长", "seconds"),
    "marker_median_residual_ms": ("会话 Unix−XDF 中位偏移模型的绝对残差中位数", "ms"),
    "marker_max_residual_ms": ("同一模型的最大绝对残差；不是直接样本错位的估计", "ms"),
    "start_event_marker_difference_ms": ("开始事件与 marker payload 的 Unix 时间绝对差", "ms"),
    "end_event_marker_difference_ms": ("结束事件与 marker payload 的 Unix 时间绝对差", "ms"),
    "feature": ("预定义的十一项核心特征之一", "identifier"),
    "variable": ("两项评分或十一项核心特征之一", "identifier"),
    "outcome": ("relaxation 主结果，discomfort 次结果", "identifier"),
    "raw_value": ("特征/变量原始尺度数值", "variable-specific; variables.csv.raw_unit"),
    "analysis_value": (
        "通过变换规则后的分析数值",
        "variable-specific; variables.csv.analysis_unit",
    ),
    "availability_status": ("available、raw_only 或 unavailable", "category"),
    "raw_available": ("原始数值通过可计算性/质量规则", "boolean"),
    "analysis_available": ("分析尺度数值可用", "boolean"),
    "reason": ("缺失或不可估计的机器可读原因", "text"),
    "warning": ("不自动导致排除的测量解释限制，分号分隔", "text"),
    "target_duration_s": ("该特征的预定测量区间时长；RMSSD 为 60 秒", "seconds"),
    "valid_duration_s": (
        "通过规则的时间；EEG 为共同通过的块，ECG 为有效 RR 总时长，行为/视频为有效相邻区间",
        "seconds",
    ),
    "valid_coverage": ("有效时长除以预定测量时长", "proportion"),
    "tp9_valid_duration_s": ("TP9 单独通过检查的两秒块总时长", "seconds"),
    "tp10_valid_duration_s": ("TP10 单独通过检查的两秒块总时长", "seconds"),
    "channels_used": ("核心 EEG 功率必须共同包含的通道", "text"),
    "channel_aggregation": ("通道聚合规则标识", "text"),
    "rr_count": ("分析时间段内检测的 RR 数", "count"),
    "accepted_rr_count": ("通过范围与局部偏离规则的 RR 数", "count"),
    "rejected_rr_count": ("未通过规则的 RR 数", "count"),
    "accepted_rr_fraction": ("通过的 RR 数除以全部 RR 数", "proportion"),
    "interval_start_xdf": ("固定 RMSSD 时间段开始", "XDF seconds"),
    "interval_end_xdf": ("固定 RMSSD 时间段结束", "XDF seconds"),
    "continuous_recording": ("固定 60 秒是否由同一个连续原始采样段覆盖", "boolean"),
    "valid_adjacent_rr_pairs": ("原始相邻且均有效的 RR 差分数；不跨缺口", "count"),
    "sample_count": ("条件中记录的采样行数", "count"),
    "valid_sample_count": ("通过存在性、有限值和可用标志的采样数", "count"),
    "valid_pair_count": ("参与平均的有效相邻样本/图像对数", "count"),
    "median_sampling_interval_s": ("正采样间隔的中位数", "seconds"),
    "rejected_gap_count": ("超过 250 ms 的行为采样间隔数", "count"),
    "nonpositive_interval_count": ("零或倒退的采样时间间隔数", "count"),
    "gaze_source_identifiable": ("日志是否能区分真实眼动和头前向回退", "boolean"),
    "target_frame_count": ("固定 2 Hz 时间格点个数", "count"),
    "decoded_frame_count": ("实际成功解码的选择图像数", "count"),
    "missing_target_frame_count": ("没有可用图像的目标格点数", "count"),
    "timestamp_source": ("用于该会话视频的时间字段", "text"),
    "flow_backend": ("实际光流运算后端", "text"),
    "source_resolutions": ("实际解码输入宽×高，分号分隔", "pixels"),
    "raw_participant_mean": (
        "该人该变量所有有效条件的原始尺度均值",
        "variable-specific; variables.csv.raw_unit",
    ),
    "analysis_participant_mean": (
        "该人该变量所有有效条件的分析尺度均值",
        "variable-specific; variables.csv.analysis_unit",
    ),
    "raw_deviation": ("原值减该人的原始尺度均值", "variable-specific; variables.csv.raw_unit"),
    "analysis_deviation": (
        "分析值减该人的分析尺度均值；不是静息基线校正",
        "variable-specific; variables.csv.analysis_unit",
    ),
    "n_raw_conditions": ("计算该人原始尺度均值所用的有效条件数", "count"),
    "n_analysis_conditions": ("计算该人分析尺度均值所用的有效条件数", "count"),
    "spearman_rho": ("个体内有效配对条件的 Spearman 相关", "dimensionless"),
    "n_valid_pairs": ("相关使用的有效配对条件数", "count"),
    "included_conditions": ("纳入相关的条件，以分号分隔并按 C1–C9 排序", "identifier list"),
    "direction": ("positive、negative、zero；不可估计时留空", "category"),
    "estimation_status": ("estimable 或 not_estimable", "category"),
    "feature_unique_values": ("有效配对内不同特征值的个数", "count"),
    "rating_unique_values": ("有效配对内不同评分值的个数", "count"),
    "n_participants": ("固定队列参与者总数", "count"),
    "n_estimable": ("相关可估计的参与者数", "count"),
    "n_non_estimable": ("相关不可估计的参与者数", "count"),
    "n_positive": ("正相关参与者数", "count"),
    "n_negative": ("负相关参与者数", "count"),
    "n_zero": ("零相关参与者数", "count"),
    "median": ("个体相关系数中位数", "dimensionless"),
    "q1": ("个体相关系数第 25 百分位，线性插值", "dimensionless"),
    "q3": ("个体相关系数第 75 百分位，线性插值", "dimensionless"),
    "minimum": ("个体相关系数最小值", "dimensionless"),
    "maximum": ("个体相关系数最大值", "dimensionless"),
    "range": ("个体相关系数最大值减最小值", "dimensionless"),
    "estimable_participants": ("相关可估计的参与者 ID，分号分隔", "identifier list"),
}


def field_definition(column: str) -> tuple[str, str]:
    if column in FIELD_DESCRIPTIONS:
        return FIELD_DESCRIPTIONS[column]
    for variable in VARIABLES:
        for scale in ["raw", "analysis"]:
            if column == variable + "_" + scale:
                unit = FEATURES.get(variable, ("", "1–7 points"))[1]
                if scale == "analysis" and variable in FEATURES and FEATURES[variable][2] == "ln":
                    unit = f"ln(value / 1 {unit})"
                return f"{ZH[variable]}，" + ("原始尺度" if scale == "raw" else "分析尺度"), unit
    for scale in ["raw", "analysis"]:
        if column.startswith(scale + "_"):
            suffix = column[len(scale) + 1 :]
            labels = {
                "n_valid": "有效条件数（参与者表）或有效参与者数（条件表）",
                "mean": "均值",
                "std": "样本标准差，ddof=1",
                "median": "中位数",
                "minimum": "最小值",
                "maximum": "最大值",
                "range": "最大值减最小值",
                "n_positive": "高于本人均值的人数",
                "n_negative": "低于本人均值的人数",
                "n_zero": "等于本人均值的人数",
                "participants": "有效参与者 ID，以分号分隔",
            }
            if suffix in labels:
                return ("原始" if scale == "raw" else "分析") + "尺度 " + labels[suffix], (
                    "count"
                    if suffix.startswith("n_")
                    else "identifier list"
                    if suffix == "participants"
                    else f"variable-specific; variables.csv.{scale}_unit"
                )
    raise ValueError(f"Undocumented output field: {column}")


def documentation(output: Path) -> None:
    write_csv(output / "variables.csv", pd.DataFrame(variable_records()))
    records = []
    for filename in list(GUIDE)[:7]:
        frame = pd.read_csv(output / filename)
        for column in frame:
            description, unit = field_definition(column)
            records.append(
                dict(
                    file=filename,
                    field=column,
                    data_type=str(frame[column].dtype),
                    unit=unit,
                    description_zh=description,
                    missing_rule="Blank means missing/not applicable; see reason/status fields.",
                )
            )
    write_csv(output / "data_dictionary.csv", pd.DataFrame(records))
    definitions = """计算定义与质量规则（先于关联计算固定）

本分析在主预测分析之后实施，属于事后探索性补充。十一项核心特征不随关联强弱增删。
完整参数以 feature_specification.json 为准，resolved_config.json 记录冻结时间、SHA-256 和软件版本。

EEG：XDF 第 1–4 列由代码与 Methods 对应 M2、TP9、TP10、M1；XDF 自身仅标 ch0–ch3，因此物理电极身份不是由 XDF 单独验证的。原始单位 µV 经元数据确认。TP9/TP10 分别减去 (M1+M2)/2，不做 CAR、不做编码器标准化。按原手工代码线性去趋势及四阶因果 Butterworth 1–45 Hz；配置与论文提到的 50 Hz notch 在该手工实现中并未执行，本次依照作者选择保持实际实现并披露差异。每十秒窗口内按实际连续段滤波；两秒块核查非有限、缺口、标准差<0.2 µV、绝对值第99百分位>350 µV及原始相关电极至少100 ms同值段。后者是平坦/疑似削顶检测，不是已校准的硬件饱和电压判定。仅对 TP9/TP10 共同有效连续块做 Welch：两秒周期 Hann，50%重叠，段内去常数，单边密度谱；按 [8,13)、[13,30)、[4,8) Hz 的实际频率格点梯形积分。各段原功率按时长加权，双通道等权均值后取 ln。完整有效块覆盖不足60%则核心功率缺失。不能将任何频段直接称为放松或紧张指标；尤其 β 仍可能混有肌电。

ECG：第7减第8列形成 ch6−ch7（Methods 为 LA−RA）。沿用现有 NeuroKit 分支的 ecg_clean(method=neurokit)：五阶0.5 Hz高通及50 Hz电源滤波；这与较早手工代码的5–25 Hz检测分支不同。R峰检测使用冻结的 NeuroKit 梯度方法参数，关闭自动峰插入/修正。不插值、不拼接原始缺口。RR 保留在35–220 bpm对应范围内且与同连续段局部5间期中位数偏离不超过20%的条目。平均心率=60×有效RR数/有效RR总秒数，为瞬时心率按RR时长加权均值；全条件有效RR时长至少60%，至少5个间期。RMSSD只用开始后5–65秒：该60秒必须连续记录、至少80%检测RR通过规则、至少20对原始相邻有效RR；被剔除的RR中断差分邻接。RMSSD=平方差均值的平方根，不是短窗RMSSD均值，也不使用此前条件或基线的心搏。保留ms，分析取ln(ms/1ms)。这是60秒的短时测量，不等同于更长记录。未获得人工R峰真值，不能把通过筛查称为临床验证。

眼动：使用 gaze_available、formal_viewing 与可匹配头显佩戴标志，去除非有限/零方向。单位方向相邻夹角/实际Δt，Δt必须>0且≤250ms，按Δt加权；不跨无效行。100°/s以下的有效时间/总有效时间为I-VT低速比例，不设置额外最小时长。实际约9 Hz的导出不能解析高速眼动细节；不能将其直接当作真实注视事件或注意。方向与hit−origin的反向一致性作为坐标检查记录，不静默反号；固定反号不改变相邻角度。无逐行gaze_source，不能测量回退率；不得把方向与头部相近直接判为回退。平滑追随、头动和回退可能混在该代理指标中；丢失不能当作眨眼。

头动：使用归一化xyzw四元数；相邻最短角度=2 arccos(|q_i·q_(i+1)|)，除实际Δt后按时长加权，避免Euler跨±180°的假跳变。朝向离散度=相对时长加权弦长L2平均旋转的测地角度RMS（度），每个有效间隔的半时长分配到两端。不跨无效或>250ms间隔。眼/头特征均要求至少60%有效时长。

视频：读取原1280×720 JPEG；核查实际分辨率与捕获间隔。P003采用全日志ISO时间字段，其他会话用通过检查的Unix字段，记录相应来源；使用相对图像路径。2 Hz格点最近选择、误差≤150ms、图像不重复；灰度INTER_AREA缩为320×180。Farnebäck(.5,2,15,2,5,1.1,0)，空间均值像素位移/实际秒数；按Δt汇总均值与总体标准差。不得跨>750ms选择间隔或>250ms原捕获缺口；覆盖至少60%、至少2对有效图像。单位是320×180分析图像像素/秒，不是视觉角度。光流同时反映场景和头部运动，不能直接解释为情绪反应。已检查GPU，但安装的OpenCV无CUDA光流，采用CPU并行；不新增绘图依赖或图形文件。

变换与统计：自然对数只用于正的EEG功率与RMSSD，不加epsilon；有意义的原始0保留，对数值留空。按各变量各自有效条件计算参与者均值与偏差，不是静息基线校正。Spearman至少6个有效配对条件；常数不可估计；并列值平均秩；不计算个体p值。标准差ddof=1；相关四分位数线性插值。相关绝对值≤1e−12视数值零；中心化偏差以1e−12×max(1,|本人均值|)容差计等于均值。所有人等权，信号窗口不增加评分样本数。

重用与重提取：重用问卷解析、身份规范、双时钟marker解析、参考电极公式及视频时间回退。旧特征表缺少所需逐块/间期溯源，且旧心率用RR中位数、旧注视比例按采样个数、旧HRV跨历史窗口聚合、旧head缺朝向离散度、已有dynamic-texture表无本次光流定义，因此十一项均从原信号按上述固定规则重新计算；这不修改或重解释历史模型的输入缓存。
"""
    (output / "feature_definitions_zh.txt").write_text(definitions, encoding="utf-8")
    guide = (
        "文件说明\n\n"
        + "\n".join(f"{name}: {description}" for name, description in GUIDE.items())
        + "\n\ncache/Pxxx/: 可恢复的逐人提取表、R峰样本索引、源元数据和错误记录。该目录不含模型或图像。\n"
    )
    (output / "FILE_GUIDE_zh.txt").write_text(guide, encoding="utf-8")
    run = """运行说明

从仓库根目录运行，现有环境：/home/link/miniconda3/envs/rtml-p002-p016/bin/python。
不要求安装绘图包；依赖使用该环境已有的 numpy、scipy、pandas、pyxdf、opencv-python、neurokit2。

完整复现（必须选择新的、尚不存在的输出目录）：
PYTHONPATH=src /home/link/miniconda3/envs/rtml-p002-p016/bin/python -m mac.analysis.participant_exploration run --output-dir artifacts/supplementary/participant_multimodal_NEW

可显式覆盖 --raw-root、--thesis-root、--labels-root、--workers（上限4）。实际源路径见 resolved_config.json。
分阶段：将 run 替换为 audit / extract / analyze / report / verify，每步使用相同 --output-dir。
首次 audit 创建并冻结规则；extract --participants P003 可作真实数据烟雾检查；随后 extract 完成剩余固定九人。
逐人完整缓存可恢复使用，但规则和原输入哈希必须一致；最终分析要求全部九人记录。源码或规则变更应使用新目录。
verify 成功才写 COMPLETE.json；已完成运行拒绝覆盖。没有任何训练或绘图入口。

验证：
PYTHONPATH=src /home/link/miniconda3/envs/rtml-p002-p016/bin/python -m pytest tests/test_participant_exploration.py

fixture只验证数学/边界行为，绝不用于研究表。实际研究输入为 waterlilies 和原问卷/论文来源。
"""
    (output / "RUNNING.txt").write_text(run, encoding="utf-8")


def report(output: Path) -> None:
    config = verify_lock(output)
    if (output / "COMPLETE.json").exists():
        raise FileExistsError("Completed output is immutable.")
    documentation(output)
    wide = pd.read_csv(output / "participant_condition_features.csv")
    quality = pd.read_csv(output / "feature_quality.csv")
    cross = pd.read_csv(output / "cross_participant_summary.csv")
    individual = pd.read_csv(output / "individual_associations.csv")
    response = pd.read_csv(output / "condition_response_summary.csv")
    descriptions = pd.read_csv(output / "participant_descriptive_summary.csv")
    lines = [
        "参与者层面的多模态补充探索性分析",
        "",
        "This analysis was conducted after the main prediction analyses and is post hoc and exploratory.",
        "本分析在主ML/DL预测分析完成后开展，仅提供解释线索，不能倒写成原模型、特征或实验设计的选择依据。",
        "",
        "1. 数据与实际完成范围",
        "固定FMQ-9："
        + ", ".join(PARTICIPANTS)
        + "；每人9条件，共81条评分记录。问卷原始评分与当前论文逐条一致；实际呈现顺序与事件日志一致。",
        "条件持续约70秒。评分按1–7原尺度保留；强度/频率为渲染器参数，不将频率值擅自解释为Hz。",
        "所有十一项特征均按预先冻结规则重新提取。各特征独立判断可用性，没有沿用五模态共同有效mask来删除其他可用观测。",
        "实际路径与所有参数见 resolved_config.json、feature_specification.json；质量阈值是在本次查看关联之前固定，并非原研究预注册。",
        "",
        "特征可用性（有效条件数/81；具有至少一个有效条件的人数/9）：",
    ]
    for feature in FEATURES:
        q = quality.loc[quality.feature.eq(feature)]
        valid = q.analysis_available.eq(True)
        reasons = q.loc[~valid, "reason"].value_counts().to_dict()
        lines.append(
            f"{ZH[feature]}：{int(valid.sum())}/81；{q.loc[valid, 'participant_id'].nunique()}/9；缺失原因 {reasons or '无'}。"
        )
    lines += [
        "",
        "2. 个体内条件变化",
        "每个变量用自身全部有效条件均值中心化；以下极值是描述，不构成条件效应检验。",
    ]
    for pid in PARTICIPANTS:
        part = wide.loc[wide.participant_id.eq(pid)]
        chunks = []
        for outcome in OUTCOMES:
            a = part[outcome + "_raw"]
            high = "、".join(part.loc[a.eq(a.max()), "condition"])
            low = "、".join(part.loc[a.eq(a.min()), "condition"])
            chunks.append(
                f"{ZH[outcome]}均值{a.mean():.2f}，范围{a.min():g}–{a.max():g}，最高{high}，最低{low}"
            )
        assoc = individual.loc[individual.participant_id.eq(pid)]
        chunks.append(
            f"22项特征–结果组合中{int(assoc.estimation_status.eq('estimable').sum())}项可估计"
        )
        lines.append(pid + "：" + "；".join(chunks) + "。")
    lines += [
        "",
        "逐变量的条件方向（分析尺度；均值指各参与者相对本人均值的偏差；正/负/零为人数）：",
    ]
    for variable in VARIABLES:
        group = response.loc[response.variable.eq(variable)].dropna(subset=["analysis_mean"])
        if group.empty:
            lines.append(ZH[variable] + "：无可用条件响应。")
            continue
        hi = group.loc[group.analysis_mean.idxmax()]
        lo = group.loc[group.analysis_mean.idxmin()]

        def text(r):
            return f"{r.condition}偏差均值{fmt(r.analysis_mean)}、中位数{fmt(r.analysis_median)}，正/负/零={int(r.analysis_n_positive)}/{int(r.analysis_n_negative)}/{int(r.analysis_n_zero)}，n={int(r.analysis_n_valid)}"

        lines.append(ZH[variable] + "：最大均值 " + text(hi) + "；最小均值 " + text(lo) + "。")
    lines += [
        "",
        "3. 特征–体验关联及跨人方向",
        "下列为个体Spearman相关的描述分布，不是总体相关；分位数只汇总可估计者，未按p值筛选。每项至少6个有效配对条件。",
        "变量 | 评分 | 可估计人数 | 正/负/零/不可估计 | 中位数 [Q1,Q3] | 最小–最大",
    ]
    for r in cross.itertuples():
        lines.append(
            f"{ZH[r.feature]} | {ZH[r.outcome]} | {r.n_estimable}/9 | {r.n_positive}/{r.n_negative}/{r.n_zero}/{r.n_non_estimable} | {fmt(r.median)} [{fmt(r.q1)}, {fmt(r.q3)}] | {fmt(r.minimum)}–{fmt(r.maximum)}"
        )
    for outcome in OUTCOMES:
        group = cross.loc[cross.outcome.eq(outcome)]
        consistent = group.loc[
            (group.n_estimable >= 6)
            & (group[["n_positive", "n_negative"]].max(axis=1) >= group.n_estimable * 0.75)
        ]
        lines.append("")
        lines.append(
            ZH[outcome]
            + "中，同向人数至少占可估计者四分之三、且至少6人可估计的描述性条目："
            + (
                "；".join(
                    f"{ZH[r.feature]}（正{r.n_positive}/负{r.n_negative}，可估计{r.n_estimable}）"
                    for r in consistent.itertuples()
                )
                if len(consistent)
                else "无"
            )
            + "。该措辞只是阅读辅助，不是显著性阈值或特征选择标准。"
        )
    non = individual.loc[individual.estimation_status.ne("estimable")]
    lines.append(
        "不可估计原因计数（按参与者×特征×结果条目）："
        + str(non.reason.value_counts().to_dict())
        + "。"
    )
    # Flag large raw-scale observations without removing or recomputing any association.
    flags = []
    for pid in PARTICIPANTS:
        part = wide.loc[wide.participant_id.eq(pid)]
        for feature in FEATURES:
            v = part[feature + "_raw"]
            finite = v.dropna()
            if len(finite) < 6:
                continue
            q1, q3 = finite.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                for idx in finite.index[(finite < q1 - 3 * iqr) | (finite > q3 + 3 * iqr)]:
                    flags.append(
                        dict(
                            participant_id=pid,
                            condition=part.loc[idx, "condition"],
                            feature=feature,
                            raw_value=v.loc[idx],
                            q1=q1,
                            q3=q3,
                            rule="outside_Q1_minus_3IQR_or_Q3_plus_3IQR",
                            excluded=False,
                        )
                    )
    write_csv(
        output / "extreme_observation_flags.csv",
        pd.DataFrame(
            flags,
            columns=[
                "participant_id",
                "condition",
                "feature",
                "raw_value",
                "q1",
                "q3",
                "rule",
                "excluded",
            ],
        ),
    )
    lines += [
        "",
        "4. 极端值与测量限制",
        f"对至少6个有效条件、IQR>0的原值做3×IQR提示，共{len(flags)}项；仅用于透明提示，未删除任何值、未重新选择方法、未做逐点删减相关。不能据此证明某个相关由极端值造成。详见 extreme_observation_flags.csv。",
        "EEG只接受共同有效的TP9/TP10片段，不选关联更强的通道。原实现未执行notch，仍需保留工频与肌电污染的解释限制；频段不是情绪标签。",
        "ECG未获得人工R峰真值；60秒RMSSD规则与旧历史短窗聚合不同。丢失、异常间期或不足覆盖导致缺失，不替换为0。",
        "眼日志无来源标志，真实眼动和头前向回退不可分；约9Hz采样下100°/s阈值的低速比例可能接近或等于1，不能解释为持续注意，也不能把丢失当眨眼。",
        "视线与hit−origin在抽查/诊断中可能反向；保持原记录约定，角速度对一致反号不变；没有据此生成假回退率。",
        "视频反映场景与头动的混合；所有光流值属于固定分析分辨率，不能直接解释为情绪。",
        "P003采用有证据的P001→P003身份别名及ISO视频时间回退；P011使用保留备份XDF。P015有会话时钟残差警告，因此使用各条件自身marker/Unity边界，未静默全局校时。",
        "",
        "5. 对既有跨参与者预测结果的解释线索",
    ]
    thesis = Path(config["thesis_root"])
    model = pd.read_csv(thesis / "derived/final_intuitive_revision/source/rq2_group_summary.csv")
    mae = model.loc[model.metric.isin(["model_mae", "condition_only_mae"])].copy()
    if mae.empty:
        raise ValueError("No model_mae/condition_only_mae rows in current thesis source")
    mae["mae_rating_points"] = 6 * mae.participant_macro_mean
    write_csv(output / "existing_prediction_reference.csv", mae)
    lines.append(
        "以下为当前论文所存既有预测汇总的摘录，不是本次重新训练或验证的结果；源表归一化MAE乘6，换算为原1–7评分单位："
    )
    for r in mae.itertuples():
        if r.metric == "model_mae":
            anchor = mae.loc[
                mae.model_id.eq(r.model_id)
                & mae.target.eq(r.target)
                & mae.metric.eq("condition_only_mae"),
                "mae_rating_points",
            ].iloc[0]
            lines.append(
                f"{r.model_id} / {r.target}：模型MAE={fmt(r.mae_rating_points)}分，条件均值锚点={fmt(anchor)}分，模型减锚点={r.mae_rating_points - anchor:+.3f}分，人数={r.participant_count}。"
            )
    floors = int(wide.discomfort_raw.eq(1).sum())
    constant = descriptions.loc[
        descriptions.variable.eq("discomfort") & descriptions.raw_range.eq(0), "participant_id"
    ].tolist()
    lines.append(
        f"本队列不适评分有{floors}/81次为最低1分，且"
        + "、".join(constant)
        + "在全部条件均为1分；因此这三人的不适相关不可估计。这为既有不适预测中评分范围受限的解释提供直接描述证据，但不单独解释模型相对锚点的误差。"
    )
    flow_condition = response.loc[
        response.variable.eq("video_mean_optical_flow") & response.condition.eq("C8")
    ].iloc[0]
    flow_association = cross.loc[
        cross.feature.eq("video_mean_optical_flow") & cross.outcome.eq("relaxation")
    ].iloc[0]
    theta_association = cross.loc[
        cross.feature.eq("eeg_theta_power") & cross.outcome.eq("relaxation")
    ].iloc[0]
    lines.append(
        f"平均光流在C8对{int(flow_condition.analysis_n_positive)}/{int(flow_condition.analysis_n_valid)}人高于本人均值，其放松相关为{int(flow_association.n_positive)}人正、{int(flow_association.n_negative)}人负。这两种描述回答不同问题：条件响应不能替代体验关联。EEG θ与放松有{int(theta_association.n_negative)}人负、{int(theta_association.n_positive)}人正，也不能据此把θ定义为放松反指标。"
    )
    lines += [
        "个体关联方向不一致、评分范围受限或某些特征几乎不变，都可提示跨人映射受到异质性和可辨识信息不足的限制；这些只是可能机制，不能认定为模型表现的既定原因。",
        "同一条件下特征的方向一致，只说明该特征可能随视觉刺激变化；只有与评分的配对关联才涉及体验共变，二者不能替代。",
        "本次使用各特征独立可用观测；既有主预测使用545/567共同有效窗口，P004/C6采用条件锚点。本次的可用性和特征定义不同，不能把本次相关当成对历史模型输入或性能的直接复验。",
        "眼/头/光流的共变也可能来自共同运动来源；本次没有建立因果分解，没有训练个人校准，也没有验证其收益。",
        "",
        "6. 尚未回答的问题与边界",
        "关联不能排除视觉条件、呈现顺序或其他因素；每人最多9个条件，相关可能不稳定。单次会话的跨人方向一致不代表跨天稳定。",
        "相关不建立因果关系，也不能证明跨人预测或个体校准有效。相反方向可能是真实个体差异、噪声或受限评分范围所致。",
        "没有观察到清晰关联不能推广为“生理信号与情绪无关”；观察到关联也不是模型失败的既定解释。",
        "本次不做混合模型、个体显著性检验、池化普通相关、重采样验证、模型训练或绘图。全部弱相关、不一致及不可估计结果均保留。",
        "源数据可追溯性限制：物理电极对应依赖实验配置/Methods，眼动来源标志缺失，R峰缺少人工真值，测量筛查不是完整额外验证项目。",
        "",
        "完整表、字典、参数和执行方式见 FILE_GUIDE_zh.txt、RUNNING.txt。代码成功执行本身不是研究发现。",
    ]
    (output / "report_zh.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    relaxation = cross.loc[cross.outcome.eq("relaxation")]
    summary = "；".join(
        f"{ZH[r.feature]}：正/负/零={r.n_positive}/{r.n_negative}/{r.n_zero}，{r.n_estimable}人可估计，中位ρ={fmt(r.median)}"
        for r in relaxation.itertuples()
    )
    passage = f"""补充分析：参与者层面的多模态条件响应与体验关联

This analysis was conducted after the main prediction analyses and is post hoc and exploratory.
本分析于主要ML/DL预测分析完成后开展，属于事后、探索性补充，旨在为既有跨参与者预测结果提供解释线索，并非原模型、特征或实验设计的选择依据。分析采用与当前五模态主分析一致的九名参与者、每人九个条件，共81条参与者×条件记录。信号窗口只用于计算条件级特征，不增加独立评分数。固定考察三项EEG频带功率、平均心率与RMSSD、两项记录视线指标、两项头动指标及两项光流指标；各特征独立判断有效性。

EEG功率和RMSSD在保留原值的同时采用自然对数分析；其余特征保留原尺度。每个变量减去该参与者有效条件均值，而非减去静息基线。以放松为主要结果、不适为次要结果，计算至少六个有效配对条件的个体Spearman相关；常数或不足配对数记为不可估计，不作个体p值筛选。跨人统计只描述个体相关的方向与分布，不称为总体相关。

放松关联的完整方向概况为：{summary}。不适及全部条件响应见配套导出表。各条目以其实际有效人数解释，尤其应区分评分/特征不变导致的不可估计与数值上的零相关。

这些结果可提示条件响应、个体差异、受限评分范围及测量噪声等因素，但不能确立历史模型表现的原因。随视觉条件变化不等于与主观体验相关。眼日志不能区分真实眼动与头前向回退；低速比例可能混入平滑追随，视频光流同时包含场景与头动，EEG尤其β仍有肌电风险。每人至多九个条件，相关可能不稳定，且不能排除条件、顺序或其他因素；单次会话一致性不意味着跨天稳定。相关既不建立因果，也不证明跨参与者预测或个人校准有效。未观察到清晰关联不意味着生理信号与情绪无关。
"""
    (output / "thesis_supplement_zh.txt").write_text(passage, encoding="utf-8")

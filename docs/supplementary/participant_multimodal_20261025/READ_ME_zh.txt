补充探索性分析：结果阅读入口

This analysis was conducted after the main prediction analyses and is post hoc and exploratory.

一、先读什么
1. participant_profiles_zh.txt：九人逐人报告。每人包含实际呈现顺序、九条件评分、
   全部十一特征的原值、两种评分的相关、有效条件及缺失原因。不是只挑选强相关。
2. P007_P009_comparison_zh.txt：两人详细对照及论文案例草稿。
   此例用于解释异质性，不是预先选定的代表性样本，也不是额外假设检验。
3. ../../../artifacts/supplementary/participant_multimodal_20261025/report_zh.txt：
   原完整中文报告，含条件响应、跨人汇总、既有预测结果和测量限制。
4. 同一 artifacts 目录的 thesis_supplement_zh.txt：独立论文段落草稿；未改动论文。

二、数值与文档路径
仓库相对路径：artifacts/supplementary/participant_multimodal_20261025/
本机绝对路径：/home/link/multimodal-affective-computing/artifacts/supplementary/participant_multimodal_20261025/
本次整理报告：docs/supplementary/participant_multimodal_20261025/

participant_condition_features.csv：81 行；评分、原始/分析尺度特征、条件和顺序。
feature_quality.csv：891 行；每特征独立可用性、覆盖率、有效时长及缺失原因。
within_person_centered.csv：1053 行；原值、分析值、个人均值与偏差。
participant_descriptive_summary.csv：117 行；每人各变量的有效数、均值和分布。
individual_associations.csv：198 行；每人十一特征×两评分，不省略不可估计项。
condition_response_summary.csv：117 行；每条件各变量的中心化响应分布。
cross_participant_summary.csv：22 行；个体相关的方向人数、分位数及范围。
data_dictionary.csv、variables.csv：字段含义、单位、变换。
feature_definitions_zh.txt、feature_specification.json：固定计算定义及质量规则。
existing_prediction_reference.csv：既有模型结果，不是新训练的结果。
extreme_observation_flags.csv：极端原值提示，没有据此删除观测或重算相关。
RUNNING.txt：完整提取/分析执行说明，需要另行配置本地原始数据路径。

三、GitHub 发布边界
本次只明确加入核心结果表、定义、中文报告、整理报告、分析代码、测试和 HISTORY。
原始 waterlilies 数据、问卷、论文源文件、视频帧、逐拍/逐块缓存、绝对输入路径清单
及本机完整审计记录不上传；这些继续保留在本地原输出目录。
结果表含假名参与者 ID 和条件级测量；它们并非原始信号，也不能称为完全匿名数据。
仓库中的表格足以核对报告和描述统计；从头重新提取仍需要原始数据及本地配置。
verification.json 是原完成运行的历史验证记录，不代表新克隆可访问所有原输入。
新增报告不覆盖或改写已完成分析的 CSV、规范、完成标记或代码哈希清单。

四、复现九人报告（仓库根目录，Python 标准库，无绘图依赖）
python scripts/summarize_participant_exploration.py \
  --source artifacts/supplementary/participant_multimodal_20261025 \
  --output docs/supplementary/participant_multimodal_20261025/participant_profiles_zh.txt

该脚本只组织现有结果；两人案例为基于上述表格人工撰写的解释，非新统计分析。
所有结果均为事后探索；条件变化不等于体验关联，关联不等于因果或预测有效。

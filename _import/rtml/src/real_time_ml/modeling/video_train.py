"""Condition-level ML evaluation for handcrafted egocentric visual features."""

from __future__ import annotations

from typing import Any

from real_time_ml.config import ProjectConfig
from real_time_ml.data.tables import write_parquet_if_available
from real_time_ml.modeling.video_ridge import (
    RELAXATION_VIDEO_RIDGE_KIND,
    train_visual_ridge,
)
from real_time_ml.utils import atomic_write_text, write_json


def _without_visual_columns(frame):
    columns = [
        name for name in frame.columns
        if not name.startswith("video_") and not name.startswith("qc_video_")
    ]
    return frame.loc[:, columns].copy()


def _complete_visual_conditions(frame):
    import pandas as pd

    working = frame.copy()
    usable = pd.to_numeric(working.get("qc_video_usable"), errors="coerce").fillna(0.0)
    working["_visual_window_usable"] = usable >= 0.5
    complete = working.groupby(["participant_id", "condition"], sort=False)["_visual_window_usable"].all()
    keys = set(complete[complete].index.tolist())
    return frame[frame.apply(lambda row: (row["participant_id"], row["condition"]) in keys, axis=1)].copy()


def _complete_embedded_video_conditions(frame, embedding_source):
    """Select only Conditions with a frozen embedding for every retained window."""
    import pandas as pd

    embeddings = pd.read_csv(embedding_source)
    keys = ["participant_id", "condition", "condition_window_index"]
    required = {*keys, "video_available"}
    missing = required - set(embeddings.columns)
    if missing:
        raise ValueError(f"VideoMAE2 embedding source is missing {sorted(missing)}")
    availability = embeddings[keys + ["video_available"]].copy()
    availability["video_available"] = pd.to_numeric(availability["video_available"], errors="coerce").fillna(0.0)
    complete = availability.groupby(["participant_id", "condition"], sort=False)["video_available"].apply(
        lambda values: bool((values >= 0.5).all())
    )
    selected = set(complete[complete].index.tolist())
    return frame[frame.apply(lambda row: (row["participant_id"], row["condition"]) in selected, axis=1)].copy(), int(len(selected))


def _line(name: str, result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    return (
        f"| {name} | {result['n_condition_labels']} | "
        f"{metrics['targets']['relaxation']['mae']:.4f} | "
        f"{metrics['targets']['discomfort']['mae']:.4f} | "
        f"{metrics['targets']['discomfort']['risk_at_fold_tuned_threshold']['per_row_threshold_recall']:.1%} | "
        f"{'通过' if metrics['deployable'] else '未通过'} |"
    )


def train_handcrafted_video_ml(config: ProjectConfig) -> dict[str, Any]:
    """Compare visual ML with a same-cohort no-video baseline.

    This function intentionally writes only under ``artifacts/video``.  It
    creates the primary 135-label masked/fallback analysis and a 125-label
    complete-video sensitivity analysis.
    """
    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError("Video ML training requires pandas") from error
    source = config.path("features") / "video_ml" / "window_features.csv"
    if not source.exists():
        raise FileNotFoundError("Run 'rtml extract-handcrafted-video' before 'rtml train-video-ml'")
    root = config.path("video")
    features_dir = root / "features" / "handcrafted"
    models_root = root / "models" / "handcrafted"
    reports_root = root / "reports" / "handcrafted"
    for directory in (features_dir, models_root, reports_root):
        directory.mkdir(parents=True, exist_ok=True)
    visual = pd.read_csv(source)
    no_video = _without_visual_columns(visual)
    no_video_source = features_dir / "window_features_no_video.csv"
    no_video.to_csv(no_video_source, index=False)
    write_parquet_if_available(no_video_source.with_suffix(".parquet"), no_video.to_dict(orient="records"))
    primary_visual = train_visual_ridge(
        config,
        source=source,
        condition_output=features_dir / "condition_features_visual.csv",
        models_dir=models_root / "visual",
        reports_dir=reports_root / "visual",
        include_video=True,
        expected_labels=135,
    )
    primary_no_video = train_visual_ridge(
        config,
        source=no_video_source,
        condition_output=features_dir / "condition_features_no_video.csv",
        models_dir=models_root / "no_video",
        reports_dir=reports_root / "no_video",
        include_video=False,
        expected_labels=135,
    )
    complete = _complete_visual_conditions(visual)
    complete_source = features_dir / "window_features_video_complete.csv"
    complete.to_csv(complete_source, index=False)
    complete_no_video_source = features_dir / "window_features_video_complete_no_video.csv"
    _without_visual_columns(complete).to_csv(complete_no_video_source, index=False)
    complete_labels = int(complete[["participant_id", "condition"]].drop_duplicates().shape[0])
    sensitivity_visual = train_visual_ridge(
        config,
        source=complete_source,
        condition_output=features_dir / "condition_features_video_complete_visual.csv",
        models_dir=models_root / "sensitivity" / "visual",
        reports_dir=reports_root / "sensitivity" / "visual",
        include_video=True,
        expected_labels=complete_labels,
    )
    sensitivity_no_video = train_visual_ridge(
        config,
        source=complete_no_video_source,
        condition_output=features_dir / "condition_features_video_complete_no_video.csv",
        models_dir=models_root / "sensitivity" / "no_video",
        reports_dir=reports_root / "sensitivity" / "no_video",
        include_video=False,
        expected_labels=complete_labels,
    )
    summary = {
        "unit_of_analysis": "participant_condition",
        "primary_masked_fallback": {"visual": primary_visual, "no_video": primary_no_video},
        "video_complete_sensitivity": {"n_labels": complete_labels, "visual": sensitivity_visual, "no_video": sensitivity_no_video},
        "runtime": "offline_or_recorded_shadow_replay_only",
        "search_profile": {"fold_local_feature_selection": "top variance", "base_feature_limit": 120, "video_feature_limit": 120, "residual_model": "Ridge(alpha=10)"},
    }
    write_json(reports_root / "comparison_metrics.json", summary)
    lines = [
        "# 第一视角手工视觉特征 ML 融合报告",
        "",
        "训练单位为 participant–Condition；视频仅作为离线/录制回放上下文。视觉模型不替换默认实时模型，且全部输出保持 Shadow。",
        "本对照为可复现的折内方差筛选 + 残差 Ridge(alpha=10)；同一 cohort 的 visual/no-video 使用完全相同的非视觉特征配置。",
        "",
        "## 主分析：15 名参与者、135 个标签（缺失视频显式 mask）",
        "",
        "| 模型 | 标签数 | Relaxation MAE | Discomfort MAE | 高 discomfort 召回 | 部署门 |",
        "|---|---:|---:|---:|---:|---|",
        _line("无视频基线", primary_no_video),
        _line("手工视觉融合", primary_visual),
        "",
        f"## 视频完整敏感性分析：{complete_labels} 个条件",
        "",
        "| 模型 | 标签数 | Relaxation MAE | Discomfort MAE | 高 discomfort 召回 | 部署门 |",
        "|---|---:|---:|---:|---:|---|",
        _line("无视频基线", sensitivity_no_video),
        _line("手工视觉融合", sensitivity_visual),
        "",
        "不同 cohort 的数值不得直接相互比较；只有同一段表内 visual/no-video 的差异可用于判断视觉特征贡献。",
    ]
    atomic_write_text(reports_root / "comparison_zh.md", "\n".join(lines) + "\n")
    return summary


def train_handcrafted_video_relaxation_ml(config: ProjectConfig) -> dict[str, Any]:
    """Train the isolated relaxation-only handcrafted visual comparison."""
    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError("Video ML training requires pandas") from error
    source = config.path("features") / "video_ml" / "window_features.csv"
    embedding_source = config.path("video") / "videomae2" / "window_embeddings.csv"
    if not source.exists() or not embedding_source.exists():
        raise FileNotFoundError("Run handcrafted video extraction and 'rtml extract-videomae2' first")
    root = config.path("video") / "relaxation_only"
    features_dir = root / "features" / "handcrafted"
    models_root = root / "models" / "handcrafted"
    reports_root = root / "reports" / "handcrafted"
    for directory in (features_dir, models_root, reports_root):
        directory.mkdir(parents=True, exist_ok=True)
    visual = pd.read_csv(source)
    no_video_source = features_dir / "window_features_no_video.csv"
    no_video = _without_visual_columns(visual)
    no_video.to_csv(no_video_source, index=False)
    write_parquet_if_available(no_video_source.with_suffix(".parquet"), no_video.to_dict(orient="records"))
    options = {
        "targets": ("relaxation",),
        "model_kind": RELAXATION_VIDEO_RIDGE_KIND,
        "research_only": True,
    }
    primary_visual = train_visual_ridge(
        config, source=source, condition_output=features_dir / "condition_features_visual.csv",
        models_dir=models_root / "visual", reports_dir=reports_root / "visual",
        include_video=True, expected_labels=135, **options,
    )
    primary_no_video = train_visual_ridge(
        config, source=no_video_source, condition_output=features_dir / "condition_features_no_video.csv",
        models_dir=models_root / "no_video", reports_dir=reports_root / "no_video",
        include_video=False, expected_labels=135, **options,
    )
    complete, complete_labels = _complete_embedded_video_conditions(visual, embedding_source)
    if complete_labels != 134:
        raise ValueError(f"Expected 134 complete-video sensitivity labels; found {complete_labels}")
    complete_source = features_dir / "window_features_video_complete.csv"
    complete.to_csv(complete_source, index=False)
    complete_no_video_source = features_dir / "window_features_video_complete_no_video.csv"
    _without_visual_columns(complete).to_csv(complete_no_video_source, index=False)
    sensitivity_visual = train_visual_ridge(
        config, source=complete_source, condition_output=features_dir / "condition_features_video_complete_visual.csv",
        models_dir=models_root / "sensitivity" / "visual", reports_dir=reports_root / "sensitivity" / "visual",
        include_video=True, expected_labels=complete_labels, **options,
    )
    sensitivity_no_video = train_visual_ridge(
        config, source=complete_no_video_source, condition_output=features_dir / "condition_features_video_complete_no_video.csv",
        models_dir=models_root / "sensitivity" / "no_video", reports_dir=reports_root / "sensitivity" / "no_video",
        include_video=False, expected_labels=complete_labels, **options,
    )
    summary = {
        "targets": ["relaxation"],
        "research_only": True,
        "unit_of_analysis": "participant_condition",
        "primary_masked_fallback": {"visual": primary_visual, "no_video": primary_no_video},
        "video_complete_sensitivity": {
            "n_labels": complete_labels,
            "visual": sensitivity_visual,
            "no_video": sensitivity_no_video,
        },
        "feature_contract": "EEG, ECG, head, eye, optional video; questionnaire raw columns are never features",
    }
    write_json(reports_root / "comparison_metrics.json", summary)
    return summary

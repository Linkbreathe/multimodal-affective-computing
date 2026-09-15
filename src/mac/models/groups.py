from __future__ import annotations


TARGETS = ["relaxation", "discomfort"]
CONTEXT_EXACT = {"intensity", "frequency", "intensity_index", "frequency_index", "condition_index", "presentation_position"}
EXCLUDED_MODEL_FEATURE_PREFIXES = (
    # Legacy ECG R-peak detector inflates HR/HRV 3-5x; the RR-std field is audit-only.
    "ecg_hrv_",
    "ecg_rr_std_ms_audit_only",
    # Dead feature: gaze_on_painting is never populated at collection, so it aggregates
    # to a constant (0 variance across all 946 windows) and carries no signal. The in-fold
    # VarianceThreshold already drops it; excluding it here makes the removal explicit and
    # keeps it out of every model feature set (main + group-ablation paths).
    "eye_gaze_on_painting_fraction",
)


def is_model_feature_allowed(name: str) -> bool:
    return not name.startswith(EXCLUDED_MODEL_FEATURE_PREFIXES)


def columns_for_group(columns: list[str], group: str) -> list[str]:
    eeg = [name for name in columns if name.startswith("eeg_")]
    behavior = [
        name
        for name in columns
        if name.startswith(("ecg_", "head_", "eye_")) and is_model_feature_allowed(name)
    ]
    context = [name for name in columns if name.startswith("video_") or name in CONTEXT_EXACT]
    mapping = {
        "context_only": context,
        "no_eeg": behavior + context,
        "eeg_only": eeg,
        "user_all": eeg + behavior,
        "fused": eeg + behavior + context,
        "full": eeg + behavior + context,
        "behavior_only": [name for name in behavior if name.startswith(("head_", "eye_"))],
    }
    if group not in mapping:
        raise ValueError(f"Unknown feature group: {group}")
    return sorted(set(mapping[group]))

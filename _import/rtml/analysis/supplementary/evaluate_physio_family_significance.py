from __future__ import annotations

from itertools import combinations
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "artifacts" / "reports" / "physio_family_split_ecg_neurokit"
OUTPUT_DIR = ROOT / "artifacts" / "reports" / "physio_family_significance_neurokit_2026-07-03"
REPORT_PATH = ROOT / "artifacts" / "reports" / "physio_family_significance_neurokit_2026-07-03_zh.md"

TARGETS = ("relaxation", "discomfort")
GROUP_ORDER = ("G", "R", "Q", "H")
NARROW_COMBINATIONS = tuple(
    "".join(parts)
    for size in range(1, len(GROUP_ORDER) + 1)
    for parts in combinations(GROUP_ORDER, size)
)
CALIBRATION_COUNTS = (2, 3)
BOOTSTRAP_REPLICATES = 10000
PRIMARY_SEED = 20260703

PRIMARY_CRITERION = {
    "mode": "personalized",
    "calibration_conditions": 2,
    "cohort": "eeg_available9",
    "target": "discomfort",
    "baseline_predictor": "history",
}

REPORTED_CANDIDATE_SIGNAL = {
    "mode": "personalized",
    "calibration_conditions": 2,
    "cohort": "all15",
    "target": "discomfort",
    "baseline_predictor": "history",
    "combinations": ("GQH", "GQHV", "GRQH", "QHV"),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    return (
        float(np.quantile(values, alpha / 2.0)),
        float(np.quantile(values, 1.0 - alpha / 2.0)),
    )


def _bootstrap_mean(values: np.ndarray, *, seed: int, replicates: int) -> dict[str, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "estimate": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "bootstrap_replicates": replicates,
        }
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(replicates, values.size))
    estimates = values[draws].mean(axis=1)
    low, high = _ci(estimates)
    return {
        "estimate": float(values.mean()),
        "ci_low": low,
        "ci_high": high,
        "bootstrap_replicates": replicates,
    }


_SIGN_CACHE: dict[int, np.ndarray] = {}


def _sign_matrix(n_values: int) -> np.ndarray:
    if n_values not in _SIGN_CACHE:
        indexes = np.arange(2**n_values, dtype=np.uint32)[:, None]
        bits = (indexes >> np.arange(n_values, dtype=np.uint32)) & 1
        _SIGN_CACHE[n_values] = np.where(bits == 1, 1.0, -1.0)
    return _SIGN_CACHE[n_values]


def _paired_sign_flip(values: np.ndarray) -> dict[str, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "paired_permutation_p_two_sided": float("nan"),
            "paired_permutation_p_model_better": float("nan"),
        }
    observed = float(values.mean())
    signs = _sign_matrix(int(values.size))
    null_means = signs.dot(values) / float(values.size)
    eps = 1e-12
    return {
        "paired_permutation_p_two_sided": float(np.mean(np.abs(null_means) >= abs(observed) - eps)),
        "paired_permutation_p_model_better": float(np.mean(null_means <= observed + eps)),
    }


def _bh_q(values: pd.Series) -> pd.Series:
    p = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(p)
    q = np.full_like(p, np.nan, dtype=float)
    if not valid.any():
        return pd.Series(q, index=values.index)
    valid_p = p[valid]
    order = np.argsort(valid_p)
    ranked = valid_p[order]
    n = ranked.size
    ranked_q = ranked * n / (np.arange(n) + 1)
    ranked_q = np.minimum.accumulate(ranked_q[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.clip(ranked_q, 0.0, 1.0)
    q[np.flatnonzero(valid)] = out
    return pd.Series(q, index=values.index)


def _coerce_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        output[column] = pd.to_numeric(output[column], errors="raise")
    return output


def _load_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_path = SOURCE_DIR / "oof_predictions.csv"
    personalized_path = SOURCE_DIR / "personalized_oof_predictions.csv"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    if not personalized_path.exists():
        raise FileNotFoundError(personalized_path)

    raw = pd.read_csv(raw_path)
    personalized = pd.read_csv(personalized_path)
    raw_required = {
        "cohort",
        "combination",
        "participant_id",
        "condition",
        "target",
        "truth",
        "prediction",
        "condition_only_prediction",
        "history_prediction",
    }
    personalized_required = {
        "calibration_conditions",
        "cohort",
        "combination",
        "participant_id",
        "condition",
        "target",
        "truth",
        "model_personalized_prediction",
        "condition_only_prediction",
        "condition_only_personalized_prediction",
        "history_prediction",
    }
    for frame_name, frame, required in (
        ("raw", raw, raw_required),
        ("personalized", personalized, personalized_required),
    ):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{frame_name} predictions missing columns: {missing}")

    raw = _coerce_numeric(raw, ["truth", "prediction", "condition_only_prediction", "history_prediction"])
    personalized = _coerce_numeric(
        personalized,
        [
            "calibration_conditions",
            "truth",
            "model_personalized_prediction",
            "condition_only_prediction",
            "condition_only_personalized_prediction",
            "history_prediction",
        ],
    )
    raw["participant_id"] = raw["participant_id"].astype(str)
    personalized["participant_id"] = personalized["participant_id"].astype(str)
    return raw, personalized


def _per_participant_mae(frame: pd.DataFrame, predictor_column: str) -> pd.Series:
    errors = (frame["truth"].to_numpy(dtype=float) - frame[predictor_column].to_numpy(dtype=float))
    work = frame[["participant_id"]].copy()
    work["absolute_error"] = np.abs(errors)
    return work.groupby("participant_id", sort=True)["absolute_error"].mean()


def _metric_row(
    frame: pd.DataFrame,
    *,
    mode: str,
    calibration_conditions: int,
    cohort: str,
    combination: str,
    target: str,
    predictor: str,
    predictor_column: str,
    replicates: int,
) -> dict[str, Any]:
    per_participant = _per_participant_mae(frame, predictor_column)
    boot = _bootstrap_mean(
        per_participant.to_numpy(dtype=float),
        seed=_stable_seed(PRIMARY_SEED, "metric", mode, calibration_conditions, cohort, combination, target, predictor),
        replicates=replicates,
    )
    return {
        "mode": mode,
        "calibration_conditions": calibration_conditions,
        "cohort": cohort,
        "combination": combination,
        "target": target,
        "predictor": predictor,
        "mae": boot["estimate"],
        "mae_ci_low": boot["ci_low"],
        "mae_ci_high": boot["ci_high"],
        "n_participants": int(per_participant.size),
        "n_rows": int(len(frame)),
        "bootstrap_replicates": replicates,
        "inference_unit": "participant",
    }


def _delta_row(
    frame: pd.DataFrame,
    *,
    mode: str,
    calibration_conditions: int,
    cohort: str,
    combination: str,
    target: str,
    model_column: str,
    baseline_predictor: str,
    baseline_column: str,
    replicates: int,
) -> dict[str, Any]:
    model_mae = _per_participant_mae(frame, model_column)
    baseline_mae = _per_participant_mae(frame, baseline_column)
    aligned = pd.concat([model_mae.rename("model"), baseline_mae.rename("baseline")], axis=1).dropna()
    values = aligned["model"].to_numpy(dtype=float) - aligned["baseline"].to_numpy(dtype=float)
    boot = _bootstrap_mean(
        values,
        seed=_stable_seed(
            PRIMARY_SEED,
            "delta",
            mode,
            calibration_conditions,
            cohort,
            combination,
            target,
            baseline_predictor,
        ),
        replicates=replicates,
    )
    permutation = _paired_sign_flip(values)
    return {
        "mode": mode,
        "calibration_conditions": calibration_conditions,
        "cohort": cohort,
        "combination": combination,
        "target": target,
        "model_predictor": "model_raw" if mode == "raw" else "model_personalized",
        "baseline_predictor": baseline_predictor,
        "model_mae": float(aligned["model"].mean()) if len(aligned) else float("nan"),
        "baseline_mae": float(aligned["baseline"].mean()) if len(aligned) else float("nan"),
        "delta_mae_model_minus_baseline": boot["estimate"],
        "delta_ci_low": boot["ci_low"],
        "delta_ci_high": boot["ci_high"],
        "paired_permutation_p_two_sided": permutation["paired_permutation_p_two_sided"],
        "paired_permutation_p_model_better": permutation["paired_permutation_p_model_better"],
        "n_participants": int(len(aligned)),
        "n_rows": int(len(frame)),
        "bootstrap_replicates": replicates,
        "inference_unit": "participant",
        "ci_stable_improvement": bool(np.isfinite(boot["ci_high"]) and boot["ci_high"] < 0.0),
    }


def _append_raw_rows(
    raw: pd.DataFrame,
    metric_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    *,
    replicates: int,
) -> None:
    predictors = {
        "model_raw": "prediction",
        "condition_only": "condition_only_prediction",
        "history": "history_prediction",
    }
    baselines = {
        "condition_only": "condition_only_prediction",
        "history": "history_prediction",
    }
    for cohort in sorted(raw["cohort"].unique()):
        for combination in NARROW_COMBINATIONS:
            for target in TARGETS:
                frame = raw.loc[
                    raw["cohort"].eq(cohort)
                    & raw["combination"].eq(combination)
                    & raw["target"].eq(target)
                ].copy()
                if frame.empty:
                    continue
                for predictor, column in predictors.items():
                    metric_rows.append(
                        _metric_row(
                            frame,
                            mode="raw",
                            calibration_conditions=0,
                            cohort=cohort,
                            combination=combination,
                            target=target,
                            predictor=predictor,
                            predictor_column=column,
                            replicates=replicates,
                        )
                    )
                for baseline, column in baselines.items():
                    delta_rows.append(
                        _delta_row(
                            frame,
                            mode="raw",
                            calibration_conditions=0,
                            cohort=cohort,
                            combination=combination,
                            target=target,
                            model_column="prediction",
                            baseline_predictor=baseline,
                            baseline_column=column,
                            replicates=replicates,
                        )
                    )


def _append_personalized_rows(
    personalized: pd.DataFrame,
    metric_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    *,
    replicates: int,
) -> None:
    predictors = {
        "model_personalized": "model_personalized_prediction",
        "condition_only": "condition_only_prediction",
        "condition_only_personalized": "condition_only_personalized_prediction",
        "history": "history_prediction",
    }
    baselines = {
        "condition_only": "condition_only_prediction",
        "condition_only_personalized": "condition_only_personalized_prediction",
        "history": "history_prediction",
    }
    for count in CALIBRATION_COUNTS:
        count_frame = personalized.loc[personalized["calibration_conditions"].eq(count)]
        for cohort in sorted(count_frame["cohort"].unique()):
            for combination in NARROW_COMBINATIONS:
                for target in TARGETS:
                    frame = count_frame.loc[
                        count_frame["cohort"].eq(cohort)
                        & count_frame["combination"].eq(combination)
                        & count_frame["target"].eq(target)
                    ].copy()
                    if frame.empty:
                        continue
                    for predictor, column in predictors.items():
                        metric_rows.append(
                            _metric_row(
                                frame,
                                mode="personalized",
                                calibration_conditions=count,
                                cohort=cohort,
                                combination=combination,
                                target=target,
                                predictor=predictor,
                                predictor_column=column,
                                replicates=replicates,
                            )
                        )
                    for baseline, column in baselines.items():
                        delta_rows.append(
                            _delta_row(
                                frame,
                                mode="personalized",
                                calibration_conditions=count,
                                cohort=cohort,
                                combination=combination,
                                target=target,
                                model_column="model_personalized_prediction",
                                baseline_predictor=baseline,
                                baseline_column=column,
                                replicates=replicates,
                            )
                        )


def _add_q_values(delta: pd.DataFrame) -> pd.DataFrame:
    output = delta.copy()
    output["bh_q_model_better_all_tests"] = _bh_q(output["paired_permutation_p_model_better"])
    family_columns = ["mode", "calibration_conditions", "cohort", "target", "baseline_predictor"]
    output["bh_q_model_better_family"] = (
        output.groupby(family_columns, dropna=False, group_keys=False)["paired_permutation_p_model_better"]
        .apply(_bh_q)
        .sort_index()
    )
    output["family_q05_stable_improvement"] = (
        output["ci_stable_improvement"]
        & (output["paired_permutation_p_model_better"] <= 0.05)
        & (output["bh_q_model_better_family"] <= 0.05)
    )
    output["global_q05_stable_improvement"] = (
        output["ci_stable_improvement"]
        & (output["paired_permutation_p_model_better"] <= 0.05)
        & (output["bh_q_model_better_all_tests"] <= 0.05)
    )
    return output


def _reported_candidate_signal_tests(personalized: pd.DataFrame, *, replicates: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    candidate = REPORTED_CANDIDATE_SIGNAL
    source = personalized.loc[
        personalized["calibration_conditions"].eq(candidate["calibration_conditions"])
        & personalized["cohort"].eq(candidate["cohort"])
        & personalized["target"].eq(candidate["target"])
    ]
    for combination in candidate["combinations"]:
        frame = source.loc[source["combination"].eq(combination)].copy()
        if frame.empty:
            continue
        rows.append(
            _delta_row(
                frame,
                mode="personalized",
                calibration_conditions=int(candidate["calibration_conditions"]),
                cohort=str(candidate["cohort"]),
                combination=combination,
                target=str(candidate["target"]),
                model_column="model_personalized_prediction",
                baseline_predictor=str(candidate["baseline_predictor"]),
                baseline_column="history_prediction",
                replicates=replicates,
            )
        )
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    output["bh_q_model_better_named_4"] = _bh_q(output["paired_permutation_p_model_better"])
    output["named_4_q05_stable_improvement"] = (
        output["ci_stable_improvement"]
        & (output["paired_permutation_p_model_better"] <= 0.05)
        & (output["bh_q_model_better_named_4"] <= 0.05)
    )
    return output


def _format_number(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _table(frame: pd.DataFrame, columns: list[str], *, max_rows: int | None = None) -> str:
    if frame.empty:
        return "无。"
    data = frame.loc[:, columns].copy()
    if max_rows is not None:
        data = data.head(max_rows)
    for column in data.columns:
        if pd.api.types.is_float_dtype(data[column]):
            data[column] = data[column].map(lambda value: _format_number(value))
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in data.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def _primary_mask(frame: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column, value in PRIMARY_CRITERION.items():
        mask &= frame[column].eq(value)
    return mask


def _write_report(
    metric: pd.DataFrame,
    delta: pd.DataFrame,
    candidate_signal: pd.DataFrame,
    provenance: dict[str, Any],
) -> None:
    primary = delta.loc[_primary_mask(delta)].sort_values("delta_mae_model_minus_baseline")
    ci_pass = delta.loc[delta["ci_stable_improvement"]].sort_values(
        ["mode", "calibration_conditions", "cohort", "target", "baseline_predictor", "delta_mae_model_minus_baseline"]
    )
    family_pass = delta.loc[delta["family_q05_stable_improvement"]].sort_values(
        ["mode", "calibration_conditions", "cohort", "target", "baseline_predictor", "delta_mae_model_minus_baseline"]
    )
    global_pass = delta.loc[delta["global_q05_stable_improvement"]].sort_values(
        ["mode", "calibration_conditions", "cohort", "target", "baseline_predictor", "delta_mae_model_minus_baseline"]
    )
    family_summary = (
        delta.groupby(["mode", "calibration_conditions", "cohort", "target", "baseline_predictor"], sort=True)
        .agg(
            n_tests=("combination", "count"),
            n_ci_pass=("ci_stable_improvement", "sum"),
            n_family_q05_pass=("family_q05_stable_improvement", "sum"),
            n_global_q05_pass=("global_q05_stable_improvement", "sum"),
        )
        .reset_index()
    )

    primary_ci = int(primary["ci_stable_improvement"].sum())
    primary_family = int(primary["family_q05_stable_improvement"].sum())
    primary_global = int(primary["global_q05_stable_improvement"].sum())
    candidate_ci = int(candidate_signal["ci_stable_improvement"].sum()) if not candidate_signal.empty else 0
    candidate_q = (
        int(candidate_signal["named_4_q05_stable_improvement"].sum())
        if not candidate_signal.empty
        else 0
    )

    lines = [
        "# G/Q/R/H + NeuroKit ECG 被试级显著性检查",
        "",
        "## 协议",
        "",
        "- 输入：`artifacts/reports/physio_family_split_ecg_neurokit/oof_predictions.csv` 与 `personalized_oof_predictions.csv`。",
        "- 范围：只保留 `G/Q/R/H` 的 15 个非空组合；不再纳入 `E/V/S/A`。",
        "- 推断单位：participant。MAE 先在每个被试内平均，再对被试均值做 bootstrap 或配对检验。",
        f"- Bootstrap：{BOOTSTRAP_REPLICATES:,} 次 participant resampling with replacement。",
        "- 配对检验：对每个组合计算被试级 `delta MAE = model - baseline`，再做精确 sign-flip permutation；负值代表模型优于 baseline。",
        "- 多重比较：同时报告同一 family 内 15 个组合的 BH q 值，以及全表所有检验的 BH q 值。",
        "",
        "## 主判据",
        "",
        "主判据固定为 `eeg_available9 + discomfort + N=2 personalized + history baseline`，避免在多重比较后再挑最好看的一格。",
        "",
        _table(
            primary,
            [
                "combination",
                "model_mae",
                "baseline_mae",
                "delta_mae_model_minus_baseline",
                "delta_ci_low",
                "delta_ci_high",
                "paired_permutation_p_model_better",
                "bh_q_model_better_family",
                "bh_q_model_better_all_tests",
                "ci_stable_improvement",
                "family_q05_stable_improvement",
            ],
        ),
        "",
        "### 主判据读数",
        "",
        f"- CI 层面稳定优于 history 的组合数：{primary_ci}/15。",
        f"- 加上同 family BH q<=0.05 后仍保留：{primary_family}/15。",
        f"- 加上全表 BH q<=0.05 后仍保留：{primary_global}/15。",
        "",
        "## 原报告点名候选复核",
        "",
        "这里单独复核原报告中 `NeuroKit2 + N=2 personalized + all15 discomfort` 对 history 点估计占优的 4 个组合。它们不并入上面的 `G/Q/R/H` 主判据，因为其中两个组合包含 `V`。",
        "",
        _table(
            candidate_signal.sort_values("delta_mae_model_minus_baseline") if not candidate_signal.empty else candidate_signal,
            [
                "combination",
                "model_mae",
                "baseline_mae",
                "delta_mae_model_minus_baseline",
                "delta_ci_low",
                "delta_ci_high",
                "paired_permutation_p_model_better",
                "bh_q_model_better_named_4",
                "ci_stable_improvement",
                "named_4_q05_stable_improvement",
            ],
        ),
        "",
        f"- 4 个点名候选中，CI 层面稳定优于 history：{candidate_ci}/4。",
        f"- 加上 4 组合内 BH q<=0.05 后仍保留：{candidate_q}/4。",
        "",
        "## Family 汇总",
        "",
        _table(
            family_summary,
            [
                "mode",
                "calibration_conditions",
                "cohort",
                "target",
                "baseline_predictor",
                "n_tests",
                "n_ci_pass",
                "n_family_q05_pass",
                "n_global_q05_pass",
            ],
        ),
        "",
        "## CI 层面稳定超过 baseline 的组合",
        "",
        _table(
            ci_pass,
            [
                "mode",
                "calibration_conditions",
                "cohort",
                "target",
                "combination",
                "baseline_predictor",
                "delta_mae_model_minus_baseline",
                "delta_ci_low",
                "delta_ci_high",
                "paired_permutation_p_model_better",
                "bh_q_model_better_family",
                "bh_q_model_better_all_tests",
            ],
            max_rows=80,
        ),
        "",
        "## Family BH q<=0.05 后仍稳定的组合",
        "",
        _table(
            family_pass,
            [
                "mode",
                "calibration_conditions",
                "cohort",
                "target",
                "combination",
                "baseline_predictor",
                "delta_mae_model_minus_baseline",
                "delta_ci_low",
                "delta_ci_high",
                "paired_permutation_p_model_better",
                "bh_q_model_better_family",
                "bh_q_model_better_all_tests",
            ],
            max_rows=80,
        ),
        "",
        "## 全表 BH q<=0.05 后仍稳定的组合",
        "",
        _table(
            global_pass,
            [
                "mode",
                "calibration_conditions",
                "cohort",
                "target",
                "combination",
                "baseline_predictor",
                "delta_mae_model_minus_baseline",
                "delta_ci_low",
                "delta_ci_high",
                "paired_permutation_p_model_better",
                "bh_q_model_better_all_tests",
            ],
            max_rows=80,
        ),
        "",
        "## 输出文件",
        "",
        "- `artifacts/reports/physio_family_significance_neurokit_2026-07-03/metric_ci.csv`：每个 MAE 的 participant-bootstrap CI。",
        "- `artifacts/reports/physio_family_significance_neurokit_2026-07-03/paired_delta_tests.csv`：模型相对 baseline 的被试级配对差值、CI、置换 p 值、BH q 值。",
        "- `artifacts/reports/physio_family_significance_neurokit_2026-07-03/ci_stable_passes.csv`：CI 上界低于 0 的组合。",
        "- `artifacts/reports/physio_family_significance_neurokit_2026-07-03/family_q05_stable_passes.csv`：同 family BH q<=0.05 后仍稳定的组合。",
        "- `artifacts/reports/physio_family_significance_neurokit_2026-07-03/reported_candidate_signal_tests.csv`：原报告 4 个点名候选的单独复核。",
        "- `artifacts/reports/physio_family_significance_neurokit_2026-07-03/provenance.json`：输入哈希、seed 与协议参数。",
        "",
        "## 结论边界",
        "",
        "这一步只检验既有 OOF 预测的不确定性；没有重训模型，也没有接入 real-time 控制。若某组合只在点估计上更好、但 CI 跨 0 或 q 值不稳定，应从候选叙事里降级。",
        "",
        "```json",
        json.dumps(provenance, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw, personalized = _load_sources()

    metric_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    _append_raw_rows(raw, metric_rows, delta_rows, replicates=BOOTSTRAP_REPLICATES)
    _append_personalized_rows(personalized, metric_rows, delta_rows, replicates=BOOTSTRAP_REPLICATES)

    metric = pd.DataFrame(metric_rows)
    delta = _add_q_values(pd.DataFrame(delta_rows))
    candidate_signal = _reported_candidate_signal_tests(personalized, replicates=BOOTSTRAP_REPLICATES)
    ci_pass = delta.loc[delta["ci_stable_improvement"]].copy()
    family_pass = delta.loc[delta["family_q05_stable_improvement"]].copy()
    global_pass = delta.loc[delta["global_q05_stable_improvement"]].copy()

    metric.to_csv(OUTPUT_DIR / "metric_ci.csv", index=False)
    delta.to_csv(OUTPUT_DIR / "paired_delta_tests.csv", index=False)
    ci_pass.to_csv(OUTPUT_DIR / "ci_stable_passes.csv", index=False)
    family_pass.to_csv(OUTPUT_DIR / "family_q05_stable_passes.csv", index=False)
    global_pass.to_csv(OUTPUT_DIR / "global_q05_stable_passes.csv", index=False)
    candidate_signal.to_csv(OUTPUT_DIR / "reported_candidate_signal_tests.csv", index=False)

    provenance = {
        "source_dir": str(SOURCE_DIR.relative_to(ROOT)),
        "raw_oof_sha256": _sha256_file(SOURCE_DIR / "oof_predictions.csv"),
        "personalized_oof_sha256": _sha256_file(SOURCE_DIR / "personalized_oof_predictions.csv"),
        "output_dir": str(OUTPUT_DIR.relative_to(ROOT)),
        "report": str(REPORT_PATH.relative_to(ROOT)),
        "seed": PRIMARY_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "narrow_combinations": list(NARROW_COMBINATIONS),
        "primary_criterion": PRIMARY_CRITERION,
        "reported_candidate_signal": REPORTED_CANDIDATE_SIGNAL,
        "inference_unit": "participant",
    }
    (OUTPUT_DIR / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    _write_report(metric, delta, candidate_signal, provenance)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()

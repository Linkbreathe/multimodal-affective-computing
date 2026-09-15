"""I.6 Multimodal-state association study (strict)  [B level].

Honest characterisation of how much the multimodal signal explains state
*beyond the already-known condition*.  Dimensionality is reduced up front to
the pre-specified theory-driven indices (NOT a free search over 127 columns).
Every modality model is compared against a condition-only and a participant-
history baseline with participant-paired bootstrap CIs and BH multiple-
comparison correction.  We report the relaxation negative result faithfully and
separately probe the discomfort/risk side where the signal is more plausible.

Output: reports/multimodal_association.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from io_common import (
    INFERENCE_P,
    MULTIMODAL_INDICES,
    add_contract_columns,
    benjamini_hochberg,
    ci,
    ensure_dirs,
    load_labels,
    load_window_features,
    paired_participant_bootstrap,
    write_csv,
)

ASSOC_TARGETS = ("relaxation", "discomfort")
MODALITIES = ("eeg", "ecg", "eye", "head")


def _cell_features(windows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (participant, condition), group in windows.groupby(["participant_id", "condition"]):
        row = {"participant_id": participant, "condition": condition}
        for name, spec in MULTIMODAL_INDICES.items():
            cols = [c for c in spec["columns"] if c in group.columns]
            vals = group[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            row[name] = float(vals.mean())
        row["intensity"] = float(group["intensity"].iloc[0])
        row["frequency"] = float(group["frequency"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def _modality_features(modality: str) -> list[str]:
    return [name for name, spec in MULTIMODAL_INDICES.items() if spec["modality"] == modality]


FEATURE_SETS = {
    "eeg": _modality_features("eeg"),
    "ecg": _modality_features("ecg"),
    "eye": _modality_features("eye"),
    "head": _modality_features("head"),
    "all_multimodal": list(MULTIMODAL_INDICES.keys()),
    "all_multimodal_plus_condition": list(MULTIMODAL_INDICES.keys()) + ["intensity", "frequency"],
}


def _lopo_mae_by_participant(cells: pd.DataFrame, target: str, feats: list[str] | None,
                             seed: int, mode: str) -> dict[str, float]:
    """Return per-participant mean absolute error under LOPO for one model.

    mode: 'ridge' (use feats), 'condition_only' (intensity+frequency ridge),
    'history' (participant grand mean from train... but participant is held out,
    so history = participant's own other-cell mean, available at inference)."""
    from sklearn.linear_model import Ridge

    participants = sorted(cells["participant_id"].unique())
    per_participant_abs: dict[str, list[float]] = {}
    for held in participants:
        train = cells[cells["participant_id"] != held]
        test = cells[cells["participant_id"] == held]
        truth = pd.to_numeric(test[target], errors="coerce").to_numpy(float)

        if mode == "history":
            # leave-one-condition-out participant mean (uses only the held
            # participant's other cells -> available online, no cross-participant leak)
            vals = pd.to_numeric(test[target], errors="coerce").to_numpy(float)
            preds = np.array([(vals.sum() - v) / (len(vals) - 1) for v in vals])
        else:
            use = feats if mode == "ridge" else ["intensity", "frequency"]
            mu = train[use].mean()
            sd = train[use].std(ddof=0).replace(0, 1.0)
            xtr = ((train[use] - mu) / sd).fillna(0.0).to_numpy(float)
            xte = ((test[use] - mu) / sd).fillna(0.0).to_numpy(float)
            model = Ridge(alpha=1.0)
            model.fit(xtr, pd.to_numeric(train[target], errors="coerce").to_numpy(float))
            preds = model.predict(xte)
        per_participant_abs[held] = list(np.abs(truth - preds))
    return {p: float(np.mean(v)) for p, v in per_participant_abs.items()}


def run(seed: int, replicates: int) -> dict:
    paths = ensure_dirs()
    labels = load_labels()
    windows = load_window_features()
    cells = _cell_features(windows).merge(
        labels[["participant_id", "condition", "relaxation", "discomfort", "high_discomfort", "eeg_status"]],
        on=["participant_id", "condition"], how="inner",
    )

    rows: list[dict] = []
    pvals_index: list[int] = []

    for target in ASSOC_TARGETS:
        # per-participant MAE for each model and the two baselines
        baseline_cond = _lopo_mae_by_participant(cells, target, None, seed, "condition_only")
        baseline_hist = _lopo_mae_by_participant(cells, target, None, seed, "history")

        for set_name, feats in FEATURE_SETS.items():
            model_mae = _lopo_mae_by_participant(cells, target, feats, seed, "ridge")
            for baseline_name, baseline_mae in (("condition_only", baseline_cond), ("history", baseline_hist)):
                # paired per-participant difference: model_mae - baseline_mae
                # negative => model better.
                diff = pd.DataFrame({
                    "participant_id": list(model_mae.keys()),
                    "d": [model_mae[p] - baseline_mae[p] for p in model_mae],
                })
                boot = paired_participant_bootstrap(
                    diff, lambda g: float(g["d"].mean()),
                    seed=seed + abs(hash((target, set_name, baseline_name))) % 9973,
                    replicates=replicates,
                )
                # two-sided bootstrap p (mass on the wrong side of 0, doubled)
                p = _bootstrap_p(diff, seed + 1, replicates)
                rows.append({
                    "target": target,
                    "model": set_name,
                    "baseline": baseline_name,
                    "metric": "delta_mae_model_minus_baseline",
                    "estimate": boot["estimate"],
                    "ci_low": boot["ci_low"],
                    "ci_high": boot["ci_high"],
                    "p_value": p,
                    "improves": bool(boot["estimate"] < 0),
                    "ci_excludes_zero": bool(np.isfinite(boot["ci_low"]) and np.isfinite(boot["ci_high"])
                                             and (boot["ci_high"] < 0 or boot["ci_low"] > 0)),
                    "n_participants": boot["n_participants"],
                    "inference_unit": INFERENCE_P,
                })
                pvals_index.append(len(rows) - 1)

    # BH correction across the whole model-vs-baseline family
    pvals = [rows[i]["p_value"] for i in pvals_index]
    qvals = benjamini_hochberg(pvals)
    for slot, q in zip(pvals_index, qvals):
        rows[slot]["bh_q_value"] = float(q)
        rows[slot]["bh_significant"] = bool(q < 0.05 and rows[slot]["improves"])

    # --- discomfort/risk asymmetry: multimodal high_discomfort detection -----
    risk_rows = _risk_side(cells, seed, replicates)
    table = pd.concat([pd.DataFrame(rows), pd.DataFrame(risk_rows)], ignore_index=True)
    write_csv(paths["reports"] / "multimodal_association.csv", table)
    return {"table": add_contract_columns(table, INFERENCE_P)}


def _bootstrap_p(diff: pd.DataFrame, seed: int, replicates: int) -> float:
    rng = np.random.default_rng(seed)
    vals = diff["d"].to_numpy(float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan")
    means = []
    for _ in range(replicates):
        means.append(float(np.mean(rng.choice(vals, size=len(vals), replace=True))))
    means = np.asarray(means)
    frac_wrong = min(np.mean(means >= 0), np.mean(means <= 0))
    return float(min(1.0, 2.0 * frac_wrong))


def _risk_side(cells: pd.DataFrame, seed: int, replicates: int) -> list[dict]:
    """Full-cell multimodal detection of high_discomfort vs condition prior,
    to expose the discomfort-side asymmetry (signal more likely here)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score

    feats = list(MULTIMODAL_INDICES.keys()) + ["intensity", "frequency"]
    participants = sorted(cells["participant_id"].unique())
    preds = []
    for held in participants:
        train = cells[cells["participant_id"] != held]
        test = cells[cells["participant_id"] == held]
        if train["high_discomfort"].nunique() < 2:
            continue
        mu = train[feats].mean()
        sd = train[feats].std(ddof=0).replace(0, 1.0)
        xtr = ((train[feats] - mu) / sd).fillna(0.0).to_numpy(float)
        xte = ((test[feats] - mu) / sd).fillna(0.0).to_numpy(float)
        model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
        model.fit(xtr, train["high_discomfort"].to_numpy(int))
        cond_rate = train.groupby("condition")["high_discomfort"].mean()
        block = test[["participant_id", "high_discomfort", "condition"]].copy()
        block["risk"] = model.predict_proba(xte)[:, 1]
        block["prior"] = test["condition"].map(cond_rate).fillna(float(train["high_discomfort"].mean())).to_numpy(float)
        preds.append(block)
    pred = pd.concat(preds, ignore_index=True)

    def metric(sample: pd.DataFrame, kind: str, col: str) -> float:
        y = sample["high_discomfort"].to_numpy(int)
        if len(np.unique(y)) < 2:
            return float("nan")
        s = sample[col].to_numpy(float)
        return float(roc_auc_score(y, s)) if kind == "roc" else float(average_precision_score(y, s))

    out = []
    for kind in ("roc", "pr"):
        for col, label in (("risk", "full_cell_multimodal"), ("prior", "condition_prior")):
            point = metric(pred, kind, col)
            rng = np.random.default_rng(seed + abs(hash((kind, col))) % 9973)
            grouped = {p: pred[pred["participant_id"] == p] for p in pred["participant_id"].unique()}
            plist = np.asarray(list(grouped.keys()))
            vals = []
            for _ in range(replicates):
                sampled = rng.choice(plist, size=len(plist), replace=True)
                sample = pd.concat([grouped[p] for p in sampled], ignore_index=True)
                v = metric(sample, kind, col)
                if np.isfinite(v):
                    vals.append(v)
            low, high = ci(vals)
            out.append({
                "target": "discomfort",
                "model": label,
                "baseline": "high_discomfort_detection",
                "metric": f"{kind}_auc",
                "estimate": point,
                "ci_low": low,
                "ci_high": high,
                "p_value": np.nan,
                "improves": np.nan,
                "ci_excludes_zero": np.nan,
                "n_participants": int(len(plist)),
                "bh_q_value": np.nan,
                "bh_significant": np.nan,
                "inference_unit": INFERENCE_P,
            })
    return out


if __name__ == "__main__":
    out = run(seed=20260627, replicates=400)
    cols = ["target", "model", "baseline", "estimate", "ci_low", "ci_high", "p_value", "bh_q_value", "improves"]
    print(out["table"][cols].to_string(index=False))

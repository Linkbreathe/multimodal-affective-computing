"""I.3 Within-condition early-warning detection  [detection A; causal C].

Uses the *first K windows* of each condition to predict whether that condition
cell ends in high_discomfort (>=0.5).  This is a detection problem, not a
counterfactual, so the existing open-loop data can validate it.  We report:

  - incremental ROC/PR-AUC and recall over a (person, condition) prior baseline,
  - lead time (does K=2 or K=3 early windows already trigger?),
  - tail power (only ~15 high_discomfort events across 135 cells),
  all with nested LOPO and participant-cluster bootstrap CIs, stratified by
  EEG availability.

Outputs:
  - reports/early_warning_detection.csv
  - models/early_warning.joblib
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib

from io_common import (
    EEG_DISABLED,
    HIGH_DISCOMFORT_THRESHOLD,
    INFERENCE_P,
    MULTIMODAL_INDICES,
    add_contract_columns,
    ci,
    ensure_dirs,
    load_labels,
    load_window_features,
    write_csv,
)

LEAD_WINDOWS = (2, 3)


def _early_cell_features(windows: pd.DataFrame, k: int) -> pd.DataFrame:
    """Average the pre-specified multimodal indices over the first k windows of
    each (participant, condition) cell.  Index = mean of its source columns."""
    source_cols = sorted({c for spec in MULTIMODAL_INDICES.values() for c in spec["columns"]})
    early = (
        windows.sort_values(["participant_id", "condition", "condition_window_index"])
        .groupby(["participant_id", "condition"], group_keys=False)
        .head(k)
    )
    rows = []
    for (participant, condition), group in early.groupby(["participant_id", "condition"]):
        row = {"participant_id": participant, "condition": condition}
        for name, spec in MULTIMODAL_INDICES.items():
            cols = [c for c in spec["columns"] if c in group.columns]
            vals = group[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            row[name] = float(vals.mean())
        # stimulus context is always available
        row["intensity"] = float(group["intensity"].iloc[0])
        row["frequency"] = float(group["frequency"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def _standardize(train: pd.DataFrame, test: pd.DataFrame, feats: list[str]):
    mu = train[feats].mean()
    sd = train[feats].std(ddof=0).replace(0, 1.0)
    tr = (train[feats] - mu) / sd
    te = (test[feats] - mu) / sd
    # impute remaining NaN (e.g. EEG-disabled) with the train mean (=0 post z)
    return tr.fillna(0.0), te.fillna(0.0)


def _lopo_predictions(cells: pd.DataFrame, feats: list[str], seed: int) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression

    out = []
    participants = sorted(cells["participant_id"].unique())
    for held in participants:
        train = cells[cells["participant_id"] != held]
        test = cells[cells["participant_id"] == held]
        if train["high_discomfort"].nunique() < 2:
            continue
        xtr, xte = _standardize(train, test, feats)
        model = LogisticRegression(
            max_iter=2000, class_weight="balanced", C=1.0, random_state=seed
        )
        model.fit(xtr.to_numpy(float), train["high_discomfort"].to_numpy(int))
        proba = model.predict_proba(xte.to_numpy(float))[:, 1]
        # (person, condition) prior baseline computed on TRAIN only:
        cond_rate = train.groupby("condition")["high_discomfort"].mean()
        global_rate = float(train["high_discomfort"].mean())
        prior = test["condition"].map(cond_rate).fillna(global_rate).to_numpy(float)
        block = test[["participant_id", "condition", "high_discomfort", "eeg_status"]].copy()
        block["risk_probability"] = proba
        block["prior_probability"] = prior
        out.append(block)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _auc(y: np.ndarray, score: np.ndarray, kind: str) -> float:
    from sklearn.metrics import average_precision_score, roc_auc_score

    if len(np.unique(y)) < 2:
        return float("nan")
    if kind == "roc":
        return float(roc_auc_score(y, score))
    return float(average_precision_score(y, score))


def _metrics_from_predictions(pred: pd.DataFrame) -> dict[str, float]:
    y = pred["high_discomfort"].to_numpy(int)
    model_roc = _auc(y, pred["risk_probability"].to_numpy(float), "roc")
    prior_roc = _auc(y, pred["prior_probability"].to_numpy(float), "roc")
    model_pr = _auc(y, pred["risk_probability"].to_numpy(float), "pr")
    prior_pr = _auc(y, pred["prior_probability"].to_numpy(float), "pr")
    # recall at a deployment-style operating point: flag top-risk cells equal in
    # number to the prevalence (budget = number of positives).
    n_pos = int(y.sum())
    order = np.argsort(-pred["risk_probability"].to_numpy(float))
    flagged = np.zeros(len(y), dtype=int)
    flagged[order[: max(n_pos, 1)]] = 1
    recall = float(np.sum((y == 1) & (flagged == 1)) / n_pos) if n_pos else float("nan")
    precision = float(np.sum((y == 1) & (flagged == 1)) / max(int(flagged.sum()), 1))
    return {
        "model_roc_auc": model_roc,
        "prior_roc_auc": prior_roc,
        "incremental_roc_auc": model_roc - prior_roc,
        "model_pr_auc": model_pr,
        "prior_pr_auc": prior_pr,
        "incremental_pr_auc": model_pr - prior_pr,
        "recall_at_prevalence_budget": recall,
        "precision_at_prevalence_budget": precision,
        "n_positive": n_pos,
        "n_cells": int(len(y)),
    }


def _bootstrap_metric(pred: pd.DataFrame, metric_key: str, seed: int, replicates: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    participants = np.asarray(sorted(pred["participant_id"].unique()))
    grouped = {p: pred[pred["participant_id"] == p] for p in participants}
    vals = []
    for _ in range(replicates):
        sampled = rng.choice(participants, size=len(participants), replace=True)
        sample = pd.concat([grouped[p] for p in sampled], ignore_index=True)
        m = _metrics_from_predictions(sample)
        if np.isfinite(m[metric_key]):
            vals.append(m[metric_key])
    return ci(vals)


def run(seed: int, replicates: int) -> dict:
    paths = ensure_dirs()
    labels = load_labels()
    windows = load_window_features()
    cell_labels = labels[["participant_id", "condition", "high_discomfort", "eeg_status"]]

    feats = list(MULTIMODAL_INDICES.keys()) + ["intensity", "frequency"]
    rows: list[dict] = []
    saved_predictions: dict[int, pd.DataFrame] = {}

    for k in LEAD_WINDOWS:
        early = _early_cell_features(windows, k)
        cells = early.merge(cell_labels, on=["participant_id", "condition"], how="inner")
        pred = _lopo_predictions(cells, feats, seed)
        saved_predictions[k] = pred
        if pred.empty:
            continue

        for stratum, subset in (("all", pred),
                                ("eeg_available", pred[pred["eeg_status"] == "eeg_available"]),
                                ("eeg_disabled", pred[pred["eeg_status"] == "eeg_disabled"])):
            if subset.empty or subset["high_discomfort"].nunique() < 2:
                metrics = {"model_roc_auc": np.nan, "incremental_roc_auc": np.nan,
                           "model_pr_auc": np.nan, "incremental_pr_auc": np.nan,
                           "recall_at_prevalence_budget": np.nan,
                           "precision_at_prevalence_budget": np.nan,
                           "n_positive": int(subset["high_discomfort"].sum()) if not subset.empty else 0,
                           "n_cells": int(len(subset))}
                report_metrics = ["model_roc_auc", "incremental_roc_auc", "model_pr_auc",
                                  "incremental_pr_auc", "recall_at_prevalence_budget"]
                for mk in report_metrics:
                    rows.append(_row(k, stratum, mk, metrics, np.nan, np.nan,
                                     estimable=False))
                continue
            metrics = _metrics_from_predictions(subset)
            for mk in ["model_roc_auc", "incremental_roc_auc", "model_pr_auc",
                       "incremental_pr_auc", "recall_at_prevalence_budget",
                       "precision_at_prevalence_budget"]:
                low, high = _bootstrap_metric(subset, mk, seed=seed + k * 17 + len(stratum), replicates=replicates)
                rows.append(_row(k, stratum, mk, metrics, low, high, estimable=True))

    table = pd.DataFrame(rows)
    write_csv(paths["reports"] / "early_warning_detection.csv", table)

    # --- fit and persist a final detector on all data (lead K=3) -------------
    early3 = _early_cell_features(windows, 3)
    cells3 = early3.merge(cell_labels, on=["participant_id", "condition"], how="inner")
    from sklearn.linear_model import LogisticRegression

    mu = cells3[feats].mean()
    sd = cells3[feats].std(ddof=0).replace(0, 1.0)
    xfull = ((cells3[feats] - mu) / sd).fillna(0.0)
    final = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    final.fit(xfull.to_numpy(float), cells3["high_discomfort"].to_numpy(int))
    joblib.dump(
        {
            "model": final,
            "features": feats,
            "standardize_mean": mu.to_dict(),
            "standardize_std": sd.to_dict(),
            "lead_windows": 3,
            "label_threshold": HIGH_DISCOMFORT_THRESHOLD,
            "multimodal_indices": {k: v["columns"] for k, v in MULTIMODAL_INDICES.items()},
            "evidence": "detection model; A-level detection performance, C-level for any causal use",
            "inference_unit": INFERENCE_P,
            "note": "open-loop early-warning detector; not a controller and not a causal action model",
        },
        paths["models"] / "early_warning.joblib",
    )

    return {"table": add_contract_columns(table, INFERENCE_P), "predictions": saved_predictions}


def _row(k, stratum, metric, metrics, low, high, estimable) -> dict:
    return {
        "lead_windows": k,
        "stratum": stratum,
        "metric": metric,
        "estimate": metrics.get(metric, np.nan),
        "ci_low": low,
        "ci_high": high,
        "n_positive": metrics["n_positive"],
        "n_cells": metrics["n_cells"],
        "status": "estimable" if estimable else "insufficient_positive_tail",
        "tail_power_note": "rare-event tail (<=15 positives overall); CIs are wide by construction",
        "inference_unit": INFERENCE_P,
    }


if __name__ == "__main__":
    out = run(seed=20260627, replicates=500)
    print(out["table"].to_string(index=False))

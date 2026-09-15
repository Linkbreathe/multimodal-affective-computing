"""Completely-random baseline under the exact benchmark metric definitions.

Metric functions are copied VERBATIM from
analysis/supplementary/run_four_channel_ranking_benchmark.py so the random
numbers are directly comparable to the published condition-only / model rows.
The harness first VALIDATES itself by reproducing the published condition-only
baseline numbers, then reports random baselines with Monte-Carlo bands.

Run:
    python analysis/supplementary/run_random_baseline.py
Writes:
    artifacts/random_baseline/random_baseline_metrics.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/features/ecg_neurokit/condition_features.csv"
OUT_DIR = ROOT / "artifacts/random_baseline"

CONTINUOUS = ("relaxation", "pleasantness", "calm")
MAE_TARGETS = ("relaxation", "discomfort")
DISC_THRESH = 0.50
SEED = 20260704
R = 1000  # Monte-Carlo repeats


# ---- metric functions copied verbatim from the four-channel benchmark ----
def _safe_spearman(truth, prediction):
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    valid = np.isfinite(truth) & np.isfinite(prediction)
    if valid.sum() < 3 or np.nanstd(truth[valid]) <= 0 or np.nanstd(prediction[valid]) <= 0:
        return 0.0
    value = float(spearmanr(truth[valid], prediction[valid]).statistic)
    return value if np.isfinite(value) else 0.0


def _within_participant_spearman(frame, truth, prediction):
    values = []
    for participant in sorted(frame["participant_id"].unique()):
        mask = frame["participant_id"].eq(participant).to_numpy()
        values.append(_safe_spearman(truth[mask], prediction[mask]))
    return float(np.mean(values)), float(np.median(values))


def _pairwise_ranking_accuracy(frame, truth, prediction):
    scores = []
    for participant in sorted(frame["participant_id"].unique()):
        mask = np.flatnonzero(frame["participant_id"].eq(participant).to_numpy())
        correct = 0.0
        total = 0
        for li, left in enumerate(mask):
            for right in mask[li + 1:]:
                td = truth[left] - truth[right]
                pdd = prediction[left] - prediction[right]
                if abs(td) <= 1e-12:
                    continue
                total += 1
                if abs(pdd) <= 1e-12:
                    correct += 0.5
                elif np.sign(td) == np.sign(pdd):
                    correct += 1.0
        if total:
            scores.append(correct / total)
    return float(np.mean(scores)) if scores else float("nan")


def _top_condition_metrics(frame, truth, prediction):
    hits, top3, regrets = [], [], []
    for participant in sorted(frame["participant_id"].unique()):
        mask = np.flatnonzero(frame["participant_id"].eq(participant).to_numpy())
        pt = truth[mask]
        pp = prediction[mask]
        true_best = np.flatnonzero(pt == np.max(pt))
        order = np.argsort(-pp, kind="stable")
        best = int(order[0])
        hits.append(float(best in set(int(v) for v in true_best)))
        top3.append(float(any(int(v) in set(int(i) for i in true_best) for v in order[:3])))
        regrets.append(float(np.max(pt) - pt[best]))
    return float(np.mean(hits)), float(np.mean(top3)), float(np.mean(regrets))


def _continuous_metrics(frame, target, prediction):
    truth = frame[target].to_numpy(dtype=float)
    wm, wmed = _within_participant_spearman(frame, truth, prediction)
    th, t3, reg = _top_condition_metrics(frame, truth, prediction)
    return {
        "spearman": _safe_spearman(truth, prediction),
        "within_participant_spearman_mean": wm,
        "pairwise_ranking_accuracy": _pairwise_ranking_accuracy(frame, truth, prediction),
        "top1_condition_hit": th,
        "top1_regret": reg,
    }


def _binary_metrics(truth, probability, threshold=DISC_THRESH):
    prediction = (probability >= threshold).astype(int)
    if len(np.unique(truth)) == 2:
        roc = float(roc_auc_score(truth, probability))
        pr = float(average_precision_score(truth, probability))
    else:
        roc = pr = float("nan")
    return {
        "roc_auc": roc,
        "pr_auc": pr,
        "recall": float(recall_score(truth, prediction, zero_division=0)),
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "f1": float(f1_score(truth, prediction, zero_division=0)),
    }


def _condition_baseline(train, test, target):
    fallback = float(train[target].mean())
    by_c = train.groupby("condition", sort=True)[target].mean().astype(float).to_dict()
    pred = np.asarray([float(by_c.get(c, fallback)) for c in test["condition"]], dtype=float)
    return pred, fallback


def _history_baseline(test, fallback, target):
    output = np.full(len(test), fallback, dtype=float)
    for _, group in test.groupby("participant_id", sort=False):
        prev = None
        for index in group.sort_values("presentation_position", kind="stable").index:
            pos = test.index.get_loc(index)
            output[pos] = fallback if prev is None else prev
            prev = float(test.loc[index, target])
    return output


def lopo_baselines(frame, target):
    cond = np.full(len(frame), np.nan)
    hist = np.full(len(frame), np.nan)
    for p in sorted(frame["participant_id"].unique()):
        tm = frame["participant_id"].eq(p).to_numpy()
        idx = np.flatnonzero(tm)
        train = frame.loc[~tm]
        test = frame.loc[tm].reset_index(drop=True)
        pred, fallback = _condition_baseline(train, test, target)
        cond[idx] = pred
        hist[idx] = _history_baseline(test, fallback, target)
    return cond, hist


def band(arr):
    a = np.asarray(arr, dtype=float)
    return float(np.mean(a)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(BASE)
    frame["participant_id"] = frame["participant_id"].astype(str)
    frame["condition"] = frame["condition"].astype(str)
    frame = frame.reset_index(drop=True)
    n = len(frame)
    frame["high_discomfort"] = (frame["discomfort"] >= DISC_THRESH).astype(int)
    pos = int(frame["high_discomfort"].sum())
    print(f"n={n}  participants={frame['participant_id'].nunique()}  "
          f"high_discomfort_pos={pos} (base rate {pos / n:.4f})")

    print("\n=== VALIDATION: condition-only baseline (should match published) ===")
    val = {}
    for t in CONTINUOUS:
        cond, _ = lopo_baselines(frame, t)
        m = _continuous_metrics(frame, t, cond)
        val[t] = m
        print(f"  {t:12s} within={m['within_participant_spearman_mean']:.4f}  "
              f"pairwise={m['pairwise_ranking_accuracy']:.4f}  top1={m['top1_condition_hit']:.4f}")
    for t in MAE_TARGETS:
        cond, hist = lopo_baselines(frame, t)
        truth = frame[t].to_numpy(float)
        print(f"  MAE {t:11s} condition_only={np.mean(np.abs(truth - cond)):.5f}  "
              f"history={np.mean(np.abs(truth - hist)):.5f}")
    cond_b, _ = lopo_baselines(frame, "high_discomfort")
    bm = _binary_metrics(frame["high_discomfort"].to_numpy(int), np.clip(cond_b, 0, 1))
    print(f"  high_discomfort condition_only roc_auc={bm['roc_auc']:.4f}  pr_auc={bm['pr_auc']:.4f}")

    print(f"\n=== RANDOM baselines: {R} Monte-Carlo repeats, seed={SEED} ===")
    rng = np.random.default_rng(SEED)
    results = {"n": n, "high_discomfort_positives": pos, "base_rate": pos / n,
               "monte_carlo_repeats": R, "seed": SEED,
               "validation_condition_only": val, "random": {}}

    for t in CONTINUOUS:
        truth = frame[t].to_numpy(float)
        acc = {k: {"within": [], "pairwise": [], "top1": [], "regret": []} for k in ("uniform", "shuffle")}
        for _ in range(R):
            for kind in ("uniform", "shuffle"):
                pred = rng.random(n) if kind == "uniform" else rng.permutation(truth)
                m = _continuous_metrics(frame, t, pred)
                acc[kind]["within"].append(m["within_participant_spearman_mean"])
                acc[kind]["pairwise"].append(m["pairwise_ranking_accuracy"])
                acc[kind]["top1"].append(m["top1_condition_hit"])
                acc[kind]["regret"].append(m["top1_regret"])
        results["random"].setdefault("continuous", {})[t] = {
            kind: {"within_mean_ci": band(acc[kind]["within"]),
                   "pairwise_ci": band(acc[kind]["pairwise"]),
                   "top1_ci": band(acc[kind]["top1"]),
                   "regret_ci": band(acc[kind]["regret"])}
            for kind in ("uniform", "shuffle")}

    for t in MAE_TARGETS:
        truth = frame[t].to_numpy(float)
        mae_u, mae_s = [], []
        for _ in range(R):
            mae_u.append(float(np.mean(np.abs(truth - rng.random(n)))))
            mae_s.append(float(np.mean(np.abs(truth - rng.permutation(truth)))))
        results["random"].setdefault("mae", {})[t] = {
            "uniform_ci": band(mae_u), "shuffle_ci": band(mae_s)}

    truth_b = frame["high_discomfort"].to_numpy(int)
    roc_u, pr_u = [], []
    for _ in range(R):
        prob = rng.random(n)
        roc_u.append(float(roc_auc_score(truth_b, prob)))
        pr_u.append(float(average_precision_score(truth_b, prob)))
    results["random"]["binary_high_discomfort"] = {
        "uniform_roc_auc_ci": band(roc_u), "uniform_pr_auc_ci": band(pr_u)}

    (OUT_DIR / "random_baseline_metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUT_DIR / 'random_baseline_metrics.json'}")


if __name__ == "__main__":
    main()

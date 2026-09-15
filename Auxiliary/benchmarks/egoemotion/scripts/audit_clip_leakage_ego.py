"""Clip-leakage audit for egoEMOTION embeddings.

Answers: is task classification disguised as emotion recognition?

Tests:
  A) Task-ID linear probe under subject-LOSO (how well can the embedding recover
     which stimulus was shown).
  B) Emotion probe under three protocols:
       B1 subject-LOSO (leakage-permissive baseline)
       B2 task-LOSO (stimulus never seen during training)
       B3 subject x task-LOSO (honest)
  C) V/A/D ridge auxiliary under the same three protocols (within-task residual).

Outputs Auxiliary/benchmarks/egoemotion/reports/clip_leakage_audit_ego.csv
(one row per metric).
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

REPO = Path(__file__).resolve().parents[4]
EMB_ROOT = REPO / "data/embeddings/egoemotion/10s_task_aware"
MANIFEST = EMB_ROOT / "manifest.csv"
REPORTS = REPO / "reports"
CSV_OUT = REPORTS / "clip_leakage_audit_ego.csv"

MODALITIES = {
    "papagei_ppg": {"pool": "squeeze"},      # [1,512]  -> [512]
    "pulseppg_ppg": {"pool": "squeeze"},     # [1,512]  -> [512]
    "video_mae_v2": {"pool": "mean"},        # [6,768]  -> [768] mean-pool
    "inceptiontime": {"pool": "squeeze"},    # [1,128]  -> [128]
}


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #

def pool_embedding(arr: np.ndarray, mode: str) -> np.ndarray:
    if mode == "squeeze":
        if arr.ndim == 2 and arr.shape[0] == 1:
            return arr[0]
        return arr.reshape(-1)
    if mode == "mean":
        return arr.mean(axis=0)
    raise ValueError(mode)


def build_cache(modality: str, manifest: pd.DataFrame, force: bool = False) -> Path:
    cache = EMB_ROOT / f"_cache_{modality}.npz"
    if cache.exists() and not force:
        return cache

    pool_mode = MODALITIES[modality]["pool"]
    root = EMB_ROOT / modality

    xs: List[np.ndarray] = []
    keep_rows: List[int] = []
    t0 = time.time()
    for i, row in manifest.iterrows():
        subj = int(row.subject)
        seq = int(row.global_seq)
        fp = root / f"{subj:03d}" / f"segment_{seq:04d}.pt"
        if not fp.is_file():
            continue
        obj = torch.load(fp, map_location="cpu", weights_only=False)
        emb = obj["embedding"].numpy() if isinstance(obj, dict) else obj.numpy()
        xs.append(pool_embedding(emb, pool_mode).astype(np.float32))
        keep_rows.append(i)
        if (len(xs) % 2000) == 0:
            print(f"  {modality}: {len(xs)} loaded ({time.time()-t0:.1f}s)", flush=True)

    X = np.stack(xs, axis=0)
    sub = manifest.loc[keep_rows].reset_index(drop=True)

    np.savez_compressed(
        cache,
        X=X,
        subject=sub.subject.to_numpy().astype(np.int32),
        task=sub.task_name.to_numpy().astype(object),
        emotion=sub.emotion_label.to_numpy().astype(np.int32),
        valence=sub.valence.to_numpy().astype(np.float32),
        arousal=sub.arousal.to_numpy().astype(np.float32),
        dominance=sub.dominance.to_numpy().astype(np.float32),
    )
    print(f"  {modality}: cached {X.shape} -> {cache.name} ({time.time()-t0:.1f}s)")
    return cache


def load_cache(path: Path) -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------- #
# Probes
# --------------------------------------------------------------------------- #

def lr_probe(
    X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray
) -> float:
    scaler = StandardScaler(with_mean=True, with_std=True).fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_te = scaler.transform(X_te)
    clf = LogisticRegression(
        solver="lbfgs",
        C=1.0,
        max_iter=200,
        tol=1e-3,
    )
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    return balanced_accuracy_score(y_te, pred)


def ridge_probe(
    X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray
) -> float:
    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_te = scaler.transform(X_te)
    reg = Ridge(alpha=1.0).fit(X_tr, y_tr)
    pred = reg.predict(X_te)
    if np.std(pred) < 1e-8 or np.std(y_te) < 1e-8 or len(y_te) < 3:
        return np.nan
    r, _ = pearsonr(pred, y_te)
    return float(r)


def subject_loso(data: Dict[str, np.ndarray], target: str, probe_fn) -> List[dict]:
    out: List[dict] = []
    subjects = np.unique(data["subject"])
    y_all = data[target]
    X_all = data["X"]
    t0 = time.time()
    for k, s in enumerate(subjects, 1):
        te = data["subject"] == s
        tr = ~te
        if len(np.unique(y_all[tr])) < 2 or te.sum() == 0:
            continue
        t1 = time.time()
        score = probe_fn(X_all[tr], y_all[tr], X_all[te], y_all[te])
        out.append({"fold_key": int(s), "score": score, "n_test": int(te.sum())})
        print(f"      fold {k}/{len(subjects)} subj={int(s)} score={score:.3f} "
              f"n_te={int(te.sum())} ({time.time()-t1:.1f}s, tot {time.time()-t0:.0f}s)",
              flush=True)
    return out


def task_loso(data: Dict[str, np.ndarray], target: str, probe_fn) -> List[dict]:
    out: List[dict] = []
    tasks = np.unique(data["task"])
    y_all = data[target]
    X_all = data["X"]
    t0 = time.time()
    for k, t in enumerate(tasks, 1):
        te = data["task"] == t
        tr = ~te
        if len(np.unique(y_all[tr])) < 2 or te.sum() == 0:
            continue
        t1 = time.time()
        score = probe_fn(X_all[tr], y_all[tr], X_all[te], y_all[te])
        out.append({"fold_key": str(t), "score": score, "n_test": int(te.sum())})
        print(f"      fold {k}/{len(tasks)} task={t} score={score:.3f} "
              f"n_te={int(te.sum())} ({time.time()-t1:.1f}s, tot {time.time()-t0:.0f}s)",
              flush=True)
    return out


def subject_x_task_loso(data: Dict[str, np.ndarray], target: str, probe_fn) -> List[dict]:
    out: List[dict] = []
    subjects = np.unique(data["subject"])
    tasks = np.unique(data["task"])
    y_all = data[target]
    X_all = data["X"]
    total = len(subjects) * len(tasks)
    done = 0
    t0 = time.time()
    for s in subjects:
        for t in tasks:
            done += 1
            te = (data["subject"] == s) & (data["task"] == t)
            if te.sum() == 0:
                continue
            tr = ~te
            if len(np.unique(y_all[tr])) < 2:
                continue
            score = probe_fn(X_all[tr], y_all[tr], X_all[te], y_all[te])
            out.append(
                {"fold_key": f"s{int(s)}_t{t}", "score": score, "n_test": int(te.sum())}
            )
            if done % 80 == 0:
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done)
                print(
                    f"    B3 progress {done}/{total} ({elapsed:.0f}s, eta {eta:.0f}s)",
                    flush=True,
                )
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def summarize(folds: List[dict]) -> Tuple[float, float, int]:
    if not folds:
        return (float("nan"), float("nan"), 0)
    s = np.array([f["score"] for f in folds], dtype=float)
    s = s[~np.isnan(s)]
    if len(s) == 0:
        return (float("nan"), float("nan"), 0)
    return float(np.mean(s)), float(np.std(s)), len(s)


def run_modality(
    modality: str,
    manifest: pd.DataFrame,
    rng: np.random.Generator,
    skip_b3: bool,
    skip_vad: bool,
) -> List[dict]:
    print(f"\n=== {modality} ===", flush=True)
    cache = build_cache(modality, manifest)
    data = load_cache(cache)
    # ensure task is string dtype (from np object array)
    data["task"] = data["task"].astype(str)
    print(f"  X shape: {data['X'].shape}, subjects: {len(np.unique(data['subject']))}, tasks: {len(np.unique(data['task']))}")

    rows: List[dict] = []

    # Test A: task-ID probe, subject-LOSO
    print("  [A] task-ID probe, subject-LOSO", flush=True)
    t0 = time.time()
    folds = subject_loso(data, "task", lr_probe)
    m, sd, n = summarize(folds)
    rows.append(dict(modality=modality, test="A_task_id", protocol="subject_loso",
                     target="task_name", metric="bal_acc", mean=m, std=sd, n_folds=n,
                     chance=1/len(np.unique(data["task"])), wall_s=time.time()-t0))
    print(f"    bal_acc={m:.3f} +/- {sd:.3f} over {n} folds (chance ~{1/len(np.unique(data['task'])):.3f})  [{time.time()-t0:.0f}s]")

    # Test B1: emotion probe, subject-LOSO
    print("  [B1] emotion probe, subject-LOSO", flush=True)
    t0 = time.time()
    folds = subject_loso(data, "emotion", lr_probe)
    m, sd, n = summarize(folds)
    rows.append(dict(modality=modality, test="B1_emotion", protocol="subject_loso",
                     target="emotion_label", metric="bal_acc", mean=m, std=sd, n_folds=n,
                     chance=1/9, wall_s=time.time()-t0))
    print(f"    bal_acc={m:.3f} +/- {sd:.3f} over {n} folds (chance ~0.111)  [{time.time()-t0:.0f}s]")

    # Chance control: shuffled labels, subject-LOSO, same modality
    print("  [ctrl] emotion probe with SHUFFLED labels (chance cal.)", flush=True)
    t0 = time.time()
    shuf = data.copy()
    shuf["emotion"] = rng.permutation(data["emotion"])
    folds = subject_loso(shuf, "emotion", lr_probe)
    m, sd, n = summarize(folds)
    rows.append(dict(modality=modality, test="B1_emotion_shuffled", protocol="subject_loso",
                     target="emotion_label_shuffled", metric="bal_acc", mean=m, std=sd, n_folds=n,
                     chance=1/9, wall_s=time.time()-t0))
    print(f"    bal_acc={m:.3f} +/- {sd:.3f} over {n} folds (expected ~0.111)  [{time.time()-t0:.0f}s]")

    # Test B2: emotion probe, task-LOSO
    print("  [B2] emotion probe, task-LOSO", flush=True)
    t0 = time.time()
    folds = task_loso(data, "emotion", lr_probe)
    m, sd, n = summarize(folds)
    rows.append(dict(modality=modality, test="B2_emotion", protocol="task_loso",
                     target="emotion_label", metric="bal_acc", mean=m, std=sd, n_folds=n,
                     chance=1/9, wall_s=time.time()-t0))
    print(f"    bal_acc={m:.3f} +/- {sd:.3f} over {n} folds  [{time.time()-t0:.0f}s]")
    # per-task breakdown, stored separately
    for f in folds:
        rows.append(dict(modality=modality, test="B2_emotion_per_task",
                         protocol="task_loso", target="emotion_label", metric="bal_acc",
                         mean=f["score"], std=0.0, n_folds=1,
                         chance=1/9, wall_s=0.0, fold_key=f["fold_key"], n_test=f["n_test"]))

    # Test B3: emotion probe, subject x task-LOSO
    if not skip_b3:
        print("  [B3] emotion probe, subject x task-LOSO", flush=True)
        t0 = time.time()
        folds = subject_x_task_loso(data, "emotion", lr_probe)
        m, sd, n = summarize(folds)
        rows.append(dict(modality=modality, test="B3_emotion", protocol="subject_x_task_loso",
                         target="emotion_label", metric="bal_acc", mean=m, std=sd, n_folds=n,
                         chance=1/9, wall_s=time.time()-t0))
        print(f"    bal_acc={m:.3f} +/- {sd:.3f} over {n} folds  [{time.time()-t0:.0f}s]")

    # V/A/D auxiliary
    if not skip_vad:
        for dim in ["valence", "arousal", "dominance"]:
            for proto_name, fn in [
                ("subject_loso", subject_loso),
                ("task_loso", task_loso),
            ]:
                print(f"  [aux {dim}] {proto_name}", flush=True)
                t0 = time.time()
                folds = fn(data, dim, ridge_probe)
                m, sd, n = summarize(folds)
                rows.append(dict(modality=modality, test=f"aux_{dim}",
                                 protocol=proto_name, target=dim, metric="pearson_r",
                                 mean=m, std=sd, n_folds=n, chance=0.0,
                                 wall_s=time.time()-t0))
                print(f"    r={m:.3f} +/- {sd:.3f} over {n} folds  [{time.time()-t0:.0f}s]")

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modalities", nargs="+",
                        default=list(MODALITIES.keys()),
                        help="Subset of modalities to run.")
    parser.add_argument("--skip-b3", action="store_true",
                        help="Skip subject x task-LOSO (Test B3).")
    parser.add_argument("--skip-vad", action="store_true",
                        help="Skip continuous V/A/D auxiliary.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=str(CSV_OUT))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    REPORTS.mkdir(exist_ok=True)

    manifest = pd.read_csv(MANIFEST)
    print(f"Manifest: {len(manifest)} rows, {manifest.subject.nunique()} subjects, "
          f"{manifest.task_name.nunique()} tasks")

    all_rows: List[dict] = []
    for mod in args.modalities:
        rows = run_modality(mod, manifest, rng,
                            skip_b3=args.skip_b3, skip_vad=args.skip_vad)
        all_rows.extend(rows)

        # incremental save after each modality
        df = pd.DataFrame(all_rows)
        df.to_csv(args.out, index=False)
        print(f"  -> wrote {len(df)} rows so far to {args.out}")

    print("\nDone. Final CSV at", args.out)


if __name__ == "__main__":
    main()

"""SEED-V 5-class emotion recognition: Differential Entropy + SVM baseline.

Classical DE+SVM is the standard reference method in SEED-V literature
(SJTU SEED-V dataset paper, Li et al. 2022).

Feature extraction
------------------
For each EEG segment [C=62, T=1024] at 128 Hz:
  * Five frequency bands:
      delta  1 – 4 Hz
      theta  4 – 8 Hz
      alpha  8 – 14 Hz
      beta  14 – 30 Hz
      gamma 30 – 47 Hz
  * Band power estimated via Welch PSD.
  * DE = 0.5 * log(2 * pi * e * var)  (equivalent to 0.5 * log(2*pi*e*sigma^2))
    which equals 0.5 * log(2*pi*e) + log(std) for Gaussian signals.
    In practice we use the band-limited power from Welch as the variance estimate.
  * Feature vector: [C × 5] = 310-dim (62 channels × 5 bands).

Classification
--------------
* sklearn SVC (RBF kernel).
* Features standardized with StandardScaler (fit on train split only).
* LOSO protocol: 16 subjects, 14 train / 1 val / 1 test.
  Val subject = subjects[(test_idx + 1) % N] (same rotation used in other scripts).
* C tuned on val set from {0.1, 1, 10, 100} using balanced accuracy.

Also computes a majority-class baseline (most frequent training label).

Results saved to logs/seedv_de_svm/.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, sosfilt, welch
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMOTIONS = ["Disgust", "Fear", "Sad", "Neutral", "Happy"]
N_CLASSES = 5

BANDS: list[tuple[str, float, float]] = [
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 14.0),
    ("beta", 14.0, 30.0),
    ("gamma", 30.0, 47.0),
]

# ---------------------------------------------------------------------------
# Differential Entropy feature extraction
# ---------------------------------------------------------------------------


def _bandpass_sos(low_hz: float, high_hz: float, fs: float, order: int = 5):
    """Return second-order-sections Butterworth bandpass filter."""
    nyq = fs / 2.0
    return butter(order, [low_hz / nyq, high_hz / nyq], btype="band", output="sos")


def de_welch(eeg: np.ndarray, fs: float = 128.0) -> np.ndarray:
    """Compute Differential Entropy features via Welch PSD.

    Parameters
    ----------
    eeg : ndarray [C, T]
        Raw EEG signal.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    features : ndarray [C * n_bands]  (flattened, C-major order)
    """
    n_channels, n_samples = eeg.shape
    n_bands = len(BANDS)
    de = np.zeros((n_channels, n_bands), dtype=np.float64)

    # Welch segment length: use nperseg = min(256, n_samples)
    nperseg = min(256, n_samples)

    for b_idx, (_, low, high) in enumerate(BANDS):
        # Welch PSD for all channels at once
        freqs, psd = welch(eeg, fs=fs, nperseg=nperseg, axis=-1)  # psd: [C, F]
        # Mask to band frequencies
        band_mask = (freqs >= low) & (freqs < high)
        if band_mask.sum() == 0:
            # Fallback: bandpass filter variance
            sos = _bandpass_sos(low, high, fs)
            filtered = sosfilt(sos, eeg, axis=-1)  # [C, T]
            band_var = np.var(filtered, axis=-1, ddof=1).clip(min=1e-10)
        else:
            # Band power = mean PSD in band (proportional to variance)
            band_var = psd[:, band_mask].mean(axis=-1).clip(min=1e-10)

        # DE = 0.5 * log(2 * pi * e * sigma^2)
        de[:, b_idx] = 0.5 * np.log(2.0 * np.pi * np.e * band_var)

    return de.flatten()  # [C * n_bands]


def de_bandpass(eeg: np.ndarray, fs: float = 128.0) -> np.ndarray:
    """Compute DE features via bandpass filtering + variance.

    More accurate band isolation than Welch alone.
    Each band: bandpass → variance → DE formula.

    Parameters
    ----------
    eeg : ndarray [C, T]

    Returns
    -------
    features : ndarray [C * n_bands]
    """
    n_channels, _ = eeg.shape
    n_bands = len(BANDS)
    de = np.zeros((n_channels, n_bands), dtype=np.float64)

    for b_idx, (_, low, high) in enumerate(BANDS):
        sos = _bandpass_sos(low, high, fs)
        filtered = sosfilt(sos, eeg, axis=-1)  # [C, T]
        band_var = np.var(filtered, axis=-1, ddof=1).clip(min=1e-10)
        de[:, b_idx] = 0.5 * np.log(2.0 * np.pi * np.e * band_var)

    return de.flatten()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_subject_features(
    df: pd.DataFrame,
    subject: int,
    fs: float,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load all segments for one subject and extract DE features.

    Returns
    -------
    X : ndarray [N, D]
    y : ndarray [N]
    """
    rows = df[df["subject"] == subject]
    feat_fn = de_bandpass if method == "bandpass" else de_welch

    feats: list[np.ndarray] = []
    labels: list[int] = []

    for _, row in rows.iterrows():
        data = torch.load(row["file_path"], map_location="cpu", weights_only=False)
        eeg = data["eeg"].numpy().astype(np.float64)  # [C, T]
        feats.append(feat_fn(eeg, fs=fs))
        labels.append(int(data["label"]))

    return np.stack(feats), np.array(labels, dtype=np.int64)


def load_all_features(
    manifest_path: str | Path,
    fs: float,
    method: str,
    cache_dir: str | Path | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load and cache DE features for all subjects.

    Returns
    -------
    dict : {subject_id: (X [N, D], y [N])}
    """
    manifest_path = Path(manifest_path)
    df = pd.read_csv(manifest_path)
    subjects = sorted(df["subject"].unique().tolist())

    subject_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    for subj in tqdm(subjects, desc="Extracting DE features"):
        cache_file = (
            cache_dir / f"subject_{subj:02d}_{method}.npz" if cache_dir else None
        )
        if cache_file is not None and cache_file.exists():
            npz = np.load(cache_file)
            subject_data[subj] = (npz["X"], npz["y"])
            log.debug("Loaded cached features for subject %d", subj)
        else:
            X, y = load_subject_features(df, subj, fs, method)
            if cache_file is not None:
                np.savez(cache_file, X=X, y=y)
            subject_data[subj] = (X, y)

    return subject_data


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES))),
        "preds": y_pred,
        "true": y_true,
    }


# ---------------------------------------------------------------------------
# LOSO with val-based C selection
# ---------------------------------------------------------------------------


def run_loso(
    subject_data: dict[int, tuple[np.ndarray, np.ndarray]],
    c_grid: list[float],
    output_dir: Path,
    seed: int,
    classifier: str = "svm",
) -> list[dict]:
    """Leave-one-subject-out with inner val loop for C selection.

    Protocol (matches run_seedv_linear_probe.py):
      val_subject  = subjects[(test_idx + 1) % N]
      train_subjects = all others (14 subjects)
    """
    subjects = sorted(subject_data.keys())
    n_subjects = len(subjects)
    results: list[dict] = []

    if n_subjects < 3:
        raise ValueError(
            f"LOSO requires at least 3 subjects (1 test + 1 val + 1 train), "
            f"but only {n_subjects} subjects provided."
        )

    for test_idx, test_subj in enumerate(tqdm(subjects, desc="LOSO folds")):
        val_subj = subjects[(test_idx + 1) % n_subjects]
        train_subjs = [s for s in subjects if s != test_subj and s != val_subj]

        X_train = np.concatenate([subject_data[s][0] for s in train_subjs])
        y_train = np.concatenate([subject_data[s][1] for s in train_subjs])
        X_val, y_val = subject_data[val_subj]
        X_test, y_test = subject_data[test_subj]

        # --- Standardize (fit on train only) ---
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc = scaler.transform(X_val)
        X_test_sc = scaler.transform(X_test)

        # --- Majority class baseline ---
        majority_label = int(np.bincount(y_train).argmax())
        maj_pred_test = np.full(len(y_test), majority_label, dtype=np.int64)
        majority_metrics = compute_metrics(y_test, maj_pred_test)

        # --- Classifier selection ---
        if classifier == "lda":
            # LDA: no hyperparameter search needed
            lda = LinearDiscriminantAnalysis(solver="svd")
            X_trainval = np.concatenate([X_train_sc, X_val_sc])
            y_trainval = np.concatenate([y_train, y_val])
            lda.fit(X_trainval, y_trainval)
            y_pred = lda.predict(X_test_sc)
            best_c = 0.0
            best_val_bacc = balanced_accuracy_score(y_val, lda.predict(X_val_sc))
            val_scores = {}
        else:
            # SVM: C selection on val set
            best_c = c_grid[0]
            best_val_bacc = -1.0
            val_scores: dict[float, float] = {}

            for c in c_grid:
                svm = SVC(
                    kernel="rbf",
                    C=c,
                    class_weight="balanced",
                    random_state=seed,
                    decision_function_shape="ovr",
                )
                svm.fit(X_train_sc, y_train)
                val_pred = svm.predict(X_val_sc)
                val_bacc = balanced_accuracy_score(y_val, val_pred)
                val_scores[c] = val_bacc
                if val_bacc > best_val_bacc:
                    best_val_bacc = val_bacc
                    best_c = c

            # Train final SVM with best C on train+val
            X_trainval = np.concatenate([X_train_sc, X_val_sc])
            y_trainval = np.concatenate([y_train, y_val])
            final_svm = SVC(
                kernel="rbf",
                C=best_c,
                class_weight="balanced",
                random_state=seed,
                decision_function_shape="ovr",
            )
            final_svm.fit(X_trainval, y_trainval)
            y_pred = final_svm.predict(X_test_sc)

        fold_metrics = compute_metrics(y_test, y_pred)
        fold_metrics.update(
            {
                "test_subject": test_subj,
                "val_subject": val_subj,
                "best_C": best_c,
                "best_val_bacc": best_val_bacc,
                "val_scores": val_scores,
                "n_train": len(y_train),
                "n_val": len(y_val),
                "n_test": len(y_test),
                "majority_label": majority_label,
                "majority_balanced_accuracy": majority_metrics["balanced_accuracy"],
                "majority_weighted_f1": majority_metrics["weighted_f1"],
                "majority_macro_f1": majority_metrics["macro_f1"],
                "majority_cohen_kappa": majority_metrics["cohen_kappa"],
            }
        )
        results.append(fold_metrics)

        print(
            f"  Subject {test_subj:2d} | "
            f"bacc={fold_metrics['balanced_accuracy']:.4f} "
            f"wF1={fold_metrics['weighted_f1']:.4f} "
            f"mF1={fold_metrics['macro_f1']:.4f} "
            f"kappa={fold_metrics['cohen_kappa']:.4f} "
            f"C={best_c} val_bacc={best_val_bacc:.4f}"
        )

    return results


# ---------------------------------------------------------------------------
# Summary and saving
# ---------------------------------------------------------------------------


def summarize_and_save(results: list[dict], output_dir: Path, feat_dim: int | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    scalar_keys = [
        "balanced_accuracy",
        "weighted_f1",
        "macro_f1",
        "cohen_kappa",
    ]
    maj_keys = [
        "majority_balanced_accuracy",
        "majority_weighted_f1",
        "majority_macro_f1",
        "majority_cohen_kappa",
    ]

    # --- Per-fold CSV ---
    fold_rows = []
    for r in results:
        row = {
            "subject": r["test_subject"],
            "val_subject": r["val_subject"],
            "best_C": r["best_C"],
            "best_val_bacc": r["best_val_bacc"],
            "n_train": r["n_train"],
            "n_val": r["n_val"],
            "n_test": r["n_test"],
        }
        for k in scalar_keys + maj_keys:
            row[k] = r[k]
        fold_rows.append(row)

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(output_dir / "loso_results.csv", index=False)

    # --- Aggregated confusion matrix ---
    total_cm = sum(r["confusion_matrix"] for r in results)
    np.save(output_dir / "confusion_matrix.npy", total_cm)

    # --- Aggregated predictions (for per-class stats) ---
    all_true = np.concatenate([r["true"] for r in results])
    all_pred = np.concatenate([r["preds"] for r in results])
    agg_metrics = compute_metrics(all_true, all_pred)

    # --- Summary printout ---
    sep = "=" * 65
    print(f"\n{sep}")
    print("  SEED-V DE+SVM Baseline — LOSO Summary")
    print(sep)

    print("\n  DE+SVM (RBF, balanced weight, C tuned on val):")
    for k in scalar_keys:
        vals = [r[k] for r in results]
        print(f"    {k:28s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    print("\n  Majority-class baseline:")
    for k in maj_keys:
        vals = [r[k] for r in results]
        label = k.replace("majority_", "")
        print(f"    {label:28s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    print(f"\n  Aggregated confusion matrix (across all {len(results)} folds):")
    header = f"{'':>10}" + "".join(f"{e:>10}" for e in EMOTIONS)
    print(f"    {header}")
    for i_row, e in enumerate(EMOTIONS):
        row_str = f"    {e:>10}" + "".join(f"{total_cm[i_row, j]:>10}" for j in range(N_CLASSES))
        print(row_str)
    print(sep)

    # --- Save summary text ---
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("SEED-V Differential Entropy + SVM Baseline — LOSO Results\n")
        f.write(sep + "\n\n")
        f.write("DE+SVM (RBF kernel, class_weight=balanced, C tuned on val):\n")
        for k in scalar_keys:
            vals = [r[k] for r in results]
            f.write(f"  {k:28s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}\n")
        f.write("\nMajority-class baseline:\n")
        for k in maj_keys:
            vals = [r[k] for r in results]
            label = k.replace("majority_", "")
            f.write(f"  {label:28s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}\n")
        f.write("\nAggregated confusion matrix:\n")
        f.write(f"{'':>10}" + "".join(f"{e:>10}" for e in EMOTIONS) + "\n")
        for i_row, e in enumerate(EMOTIONS):
            f.write(f"{e:>10}" + "".join(f"{total_cm[i_row, j]:>10}" for j in range(N_CLASSES)) + "\n")
        f.write("\nPer-subject results:\n")
        f.write(
            f"  {'subj':>4}  {'bacc':>6}  {'wF1':>6}  {'mF1':>6}  {'kappa':>6}  {'best_C':>7}\n"
        )
        for r in results:
            f.write(
                f"  {r['test_subject']:>4}  "
                f"{r['balanced_accuracy']:>6.4f}  "
                f"{r['weighted_f1']:>6.4f}  "
                f"{r['macro_f1']:>6.4f}  "
                f"{r['cohen_kappa']:>6.4f}  "
                f"{r['best_C']:>7}\n"
            )

    # --- Save aggregated metrics JSON ---
    agg_save = {
        "svm": {k: float(np.mean([r[k] for r in results])) for k in scalar_keys},
        "svm_std": {k: float(np.std([r[k] for r in results])) for k in scalar_keys},
        "majority": {
            k.replace("majority_", ""): float(np.mean([r[k] for r in results]))
            for k in maj_keys
        },
        "svm_aggregated": {
            k: float(agg_metrics[k]) for k in scalar_keys
        },
        "n_subjects": len(results),
        "feature_dim": feat_dim,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(agg_save, f, indent=2)

    print(f"\nResults saved to {output_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SEED-V 5-class emotion recognition: Differential Entropy + SVM baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        default="data/seedv_preprocessed_uv/manifest.csv",
        help="Path to the preprocessed SEED-V manifest CSV.",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=256.0,
        help=(
            "EEG sampling frequency in Hz. Must match target_sr in "
            "src/data/seedv_preprocessing.py (currently 256 Hz)."
        ),
    )
    parser.add_argument(
        "--de_method",
        choices=["bandpass", "welch"],
        default="bandpass",
        help=(
            "DE extraction method. "
            "'bandpass': Butterworth bandpass + variance (recommended). "
            "'welch': Welch PSD band-mean as variance proxy."
        ),
    )
    parser.add_argument(
        "--c_grid",
        type=float,
        nargs="+",
        default=[0.1, 1.0, 10.0, 100.0],
        help="C values to search for SVM.",
    )
    parser.add_argument(
        "--output_dir",
        default="logs/seedv_de_svm",
        help="Directory to save results.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="Optional directory to cache per-subject DE features (numpy .npz). "
             "Speeds up re-runs significantly.",
    )
    parser.add_argument(
        "--classifier",
        choices=["svm", "lda"],
        default="svm",
        help="Classifier: 'svm' (RBF, slow on large N) or 'lda' (fast).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for SVM.",
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=None,
        help="Subset of subject IDs to run (default: all). "
             "E.g. --subjects 1 2 3 for a quick smoke-test.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"SEED-V DE+SVM Baseline")
    print(f"  Manifest   : {args.manifest}")
    print(f"  FS         : {args.fs} Hz")
    print(f"  DE method  : {args.de_method}")
    print(f"  C grid     : {args.c_grid}")
    print(f"  Output dir : {output_dir}")
    print(f"  Cache dir  : {args.cache_dir}")

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    t0 = time.time()
    subject_data = load_all_features(
        manifest_path=args.manifest,
        fs=args.fs,
        method=args.de_method,
        cache_dir=args.cache_dir,
    )

    if args.subjects is not None:
        subject_data = {s: subject_data[s] for s in args.subjects if s in subject_data}
        if not subject_data:
            raise ValueError(f"None of the requested subjects {args.subjects} found in data.")

    n_subj = len(subject_data)
    n_total = sum(X.shape[0] for X, _ in subject_data.values())
    feat_dim = next(iter(subject_data.values()))[0].shape[1]
    print(
        f"\nFeatures extracted: {n_subj} subjects, {n_total} segments, "
        f"{feat_dim}-dim ({time.time() - t0:.1f}s)"
    )
    print(f"  Bands  : {[b[0] for b in BANDS]}")
    print(f"  Feature: {feat_dim // len(BANDS)} channels x {len(BANDS)} bands = {feat_dim} dims")

    # ------------------------------------------------------------------
    # LOSO classification
    # ------------------------------------------------------------------
    print(f"\nRunning LOSO ({n_subj} folds)...")
    results = run_loso(
        subject_data=subject_data,
        c_grid=args.c_grid,
        output_dir=output_dir,
        seed=args.seed,
        classifier=args.classifier,
    )

    # ------------------------------------------------------------------
    # Summary + save
    # ------------------------------------------------------------------
    summarize_and_save(results, output_dir, feat_dim=feat_dim)


if __name__ == "__main__":
    main()

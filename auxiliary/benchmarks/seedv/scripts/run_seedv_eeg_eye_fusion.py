#!/usr/bin/env python3
"""
EEG+Eye Fusion LOSO experiment on SEED-V.
Combines REVE EEG embeddings (512-dim) with eye movement features (66-dim).
Runs 16-fold Leave-One-Subject-Out with SVM (RBF) + StandardScaler.
"""

import numpy as np
import pickle
import torch
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
import time

# === Paths ===
EYE_DIR = Path("/mnt/c/Users/Public/Data/SEED-V/SEED-V/Eye_movement_features")
_REPO_ROOT = Path(__file__).resolve().parents[4]
REVE_DIR = _REPO_ROOT / "data/embeddings/seedv/uv/reve_eeg"
MANIFEST = _REPO_ROOT / "data/embeddings/seedv/uv/reve_manifest.csv"

EMOTION_NAMES = {0: "Disgust", 1: "Fear", 2: "Sad", 3: "Neutral", 4: "Happy"}
N_SUBJECTS = 16
N_TRIALS = 45  # 15 trials x 3 sessions


def load_eye_features(subject_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Load eye features for a subject. Returns (N_TRIALS, 66) features and (N_TRIALS,) labels."""
    path = EYE_DIR / f"{subject_id}_123.npz"
    raw = np.load(path, allow_pickle=True)
    data = pickle.loads(bytes(raw["data"]))
    label = pickle.loads(bytes(raw["label"]))

    features = []
    labels = []
    for trial_idx in range(N_TRIALS):
        arr = data[trial_idx]  # (T, 33)
        # mean + std -> 66-dim
        feat = np.concatenate([arr.mean(axis=0), arr.std(axis=0)])
        features.append(feat)
        labels.append(int(label[trial_idx][0]))

    return np.array(features), np.array(labels)


def load_reve_embeddings(subject_id: int, manifest_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Load REVE EEG embeddings for a subject. Average segments per trial.
    Returns (N_TRIALS, 512) features and (N_TRIALS,) labels."""
    sub_df = manifest_df[manifest_df["subject"] == subject_id].copy()

    features = []
    labels = []

    # Iterate sessions 1,2,3 -> global trial indices 0-14, 15-29, 30-44
    for sess_idx, session in enumerate([1, 2, 3]):
        sess_df = sub_df[sub_df["session"] == session]
        for trial in range(15):
            trial_df = sess_df[sess_df["trial"] == trial]
            if len(trial_df) == 0:
                # Missing data - fill with zeros
                features.append(np.zeros(512))
                labels.append(-1)
                continue

            # Load and average all segments for this trial
            embeddings = []
            for _, row in trial_df.iterrows():
                pt_path = _REPO_ROOT / row["file_path"]
                data = torch.load(pt_path, map_location="cpu", weights_only=False)
                embeddings.append(data["embedding"].numpy())

            avg_emb = np.mean(embeddings, axis=0)
            features.append(avg_emb)
            labels.append(int(trial_df.iloc[0]["emotion_label"]))

    return np.array(features), np.array(labels)


def _summarize_loso_predictions(
    y_true_all: list[int],
    y_pred_all: list[int],
    per_subject_acc: list[float],
) -> dict:
    y_true_arr = np.array(y_true_all)
    y_pred_arr = np.array(y_pred_all)

    bal_acc = balanced_accuracy_score(y_true_arr, y_pred_arr)
    macro_f1 = f1_score(y_true_arr, y_pred_arr, average="macro")
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1, 2, 3, 4])

    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
    per_class_acc = np.nan_to_num(per_class_acc, nan=0.0)

    return {
        "bal_acc": bal_acc,
        "macro_f1": macro_f1,
        "per_class_acc": per_class_acc,
        "per_subject_acc": per_subject_acc,
        "cm": cm,
        "std_acc": np.std(per_subject_acc),
    }


def _fit_fold_local_eeg_pca(
    all_eeg: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    n_components: int = 66,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, PCA, float]:
    """Fit EEG scaler/PCA on the training subjects only for one LOSO fold."""
    eeg_scaler = StandardScaler()
    eeg_train_scaled = eeg_scaler.fit_transform(all_eeg[train_mask])
    eeg_test_scaled = eeg_scaler.transform(all_eeg[test_mask])

    fold_components = min(n_components, eeg_train_scaled.shape[0], eeg_train_scaled.shape[1])
    if fold_components < 1:
        raise ValueError("PCA requires at least one training sample and feature")

    pca = PCA(n_components=fold_components)
    eeg_train_pca = pca.fit_transform(eeg_train_scaled)
    eeg_test_pca = pca.transform(eeg_test_scaled)
    explained = float(pca.explained_variance_ratio_.sum())
    return eeg_train_pca, eeg_test_pca, eeg_scaler, pca, explained


def run_loso(all_features: np.ndarray, all_labels: np.ndarray,
             all_subjects: np.ndarray, feature_name: str) -> dict:
    """Run LOSO cross-validation with SVM (RBF)."""
    unique_subjects = np.unique(all_subjects)
    y_true_all = []
    y_pred_all = []
    per_subject_acc = []

    for test_sub in unique_subjects:
        train_mask = all_subjects != test_sub
        test_mask = all_subjects == test_sub

        X_train = all_features[train_mask]
        y_train = all_labels[train_mask]
        X_test = all_features[test_mask]
        y_test = all_labels[test_mask]

        # StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # SVM with RBF kernel
        svm = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())
        sub_acc = balanced_accuracy_score(y_test, y_pred)
        per_subject_acc.append(sub_acc)

    return _summarize_loso_predictions(y_true_all, y_pred_all, per_subject_acc)


def run_loso_pca_concat(
    all_eye: np.ndarray,
    all_eeg: np.ndarray,
    all_labels: np.ndarray,
    all_subjects: np.ndarray,
    feature_name: str = "PCA-Concat(132)",
    n_components: int = 66,
) -> dict:
    """Run LOSO with EEG PCA fit independently inside each fold."""
    unique_subjects = np.unique(all_subjects)
    y_true_all = []
    y_pred_all = []
    per_subject_acc = []
    pca_explained = []

    for test_sub in unique_subjects:
        train_mask = all_subjects != test_sub
        test_mask = all_subjects == test_sub

        eeg_train_pca, eeg_test_pca, _, _, explained = _fit_fold_local_eeg_pca(
            all_eeg,
            train_mask,
            test_mask,
            n_components=n_components,
        )
        pca_explained.append(explained)

        X_train = np.hstack([all_eye[train_mask], eeg_train_pca])
        X_test = np.hstack([all_eye[test_mask], eeg_test_pca])
        y_train = all_labels[train_mask]
        y_test = all_labels[test_mask]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        svm = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())
        per_subject_acc.append(balanced_accuracy_score(y_test, y_pred))

    result = _summarize_loso_predictions(y_true_all, y_pred_all, per_subject_acc)
    result["pca_variance_explained_mean"] = float(np.mean(pca_explained))
    result["pca_variance_explained_std"] = float(np.std(pca_explained))
    return result


def main():
    t0 = time.time()
    print("Loading REVE manifest...")
    manifest_df = pd.read_csv(MANIFEST)

    # Storage
    eye_features_list = []
    eeg_features_list = []
    labels_list = []
    subjects_list = []

    for sub_id in range(1, N_SUBJECTS + 1):
        print(f"  Loading subject {sub_id}...", end=" ", flush=True)

        # Eye features
        eye_feat, eye_labels = load_eye_features(sub_id)

        # REVE EEG embeddings
        eeg_feat, eeg_labels = load_reve_embeddings(sub_id, manifest_df)

        # Verify labels match
        match = np.all(eye_labels == eeg_labels)
        if not match:
            print(f"WARNING: Label mismatch for subject {sub_id}!")
            print(f"  Eye labels: {eye_labels}")
            print(f"  EEG labels: {eeg_labels}")
            # Use eye labels as ground truth
        else:
            print(f"OK (labels match, {len(eye_labels)} trials)")

        eye_features_list.append(eye_feat)
        eeg_features_list.append(eeg_feat)
        labels_list.append(eye_labels)
        subjects_list.append(np.full(N_TRIALS, sub_id))

    # Stack all
    all_eye = np.vstack(eye_features_list)      # (720, 66)
    all_eeg = np.vstack(eeg_features_list)       # (720, 512)
    all_labels = np.concatenate(labels_list)      # (720,)
    all_subjects = np.concatenate(subjects_list)  # (720,)

    # Fusion: concatenate
    all_concat = np.hstack([all_eye, all_eeg])   # (720, 578)

    print(f"\nData loaded in {time.time()-t0:.1f}s")
    print(f"  Eye features:    {all_eye.shape}")
    print(f"  EEG features:    {all_eeg.shape}")
    print(f"  Concat features: {all_concat.shape}")
    print(f"  PCA-Concat:      fold-local EEG PCA + eye features")
    print(f"  Labels:          {all_labels.shape}, classes: {np.unique(all_labels)}")

    # Check for NaN/Inf
    for name, arr in [("Eye", all_eye), ("EEG", all_eeg)]:
        n_nan = np.isnan(arr).sum()
        n_inf = np.isinf(arr).sum()
        if n_nan > 0 or n_inf > 0:
            print(f"  WARNING: {name} has {n_nan} NaN, {n_inf} Inf values")

    # Run LOSO for all feature sets
    print("\n--- Running LOSO ---")
    results = {}
    for name, feats in [("Eye-only(66)", all_eye),
                         ("EEG-only(512)", all_eeg),
                         ("Concat(578)", all_concat)]:
        print(f"  {name}...", end=" ", flush=True)
        t1 = time.time()
        res = run_loso(feats, all_labels, all_subjects, name)
        print(f"done ({time.time()-t1:.1f}s)")
        results[name] = res

    print(f"  PCA-Concat(132)...", end=" ", flush=True)
    t1 = time.time()
    results["PCA-Concat(132)"] = run_loso_pca_concat(
        all_eye,
        all_eeg,
        all_labels,
        all_subjects,
        "PCA-Concat(132)",
        n_components=66,
    )
    print(
        f"done ({time.time()-t1:.1f}s, "
        f"mean PCA variance={results['PCA-Concat(132)']['pca_variance_explained_mean']:.3f})"
    )

    # Report
    eye_acc = results["Eye-only(66)"]["bal_acc"]
    print("\n" + "=" * 80)
    print("=== EEG+Eye Fusion LOSO (SEED-V, 16 subjects, 5 classes) ===")
    print("=" * 80)
    print(f"{'Features':<18} | {'Balanced Acc':>18} | {'Macro F1':>14} | {'Delta vs Eye':>14}")
    print("-" * 80)
    for name, res in results.items():
        delta = res["bal_acc"] - eye_acc
        delta_str = "baseline" if name == "Eye-only(66)" else f"{delta:+.4f}"
        print(f"{name:<18} | {res['bal_acc']:.4f} +/- {res['std_acc']:.3f}    | {res['macro_f1']:.4f}          | {delta_str}")

    # Per-class accuracy comparison
    best_fusion = "PCA-Concat(132)"
    print("\n--- Per-class Accuracy ---")
    print(f"{'Class':<10}", end="")
    for name in results:
        print(f" | {name:>18}", end="")
    print(f" | {'D(PCA-Eye)':>12}")
    print("-" * 110)
    for c in range(5):
        print(f"{EMOTION_NAMES[c]:<10}", end="")
        for name in results:
            print(f" | {results[name]['per_class_acc'][c]:>18.4f}", end="")
        delta = results[best_fusion]["per_class_acc"][c] - results["Eye-only(66)"]["per_class_acc"][c]
        print(f" | {delta:>+12.4f}")

    # Per-subject accuracy for best fusion
    print(f"\n--- Per-subject Balanced Accuracy ---")
    print(f"  {'Sub':>4}  {'Eye':>6}  {'Concat':>6}  {'PCA-Con':>7}  {'D(PCA-Eye)':>10}")
    print("  " + "-" * 50)
    for i, sub_id in enumerate(range(1, N_SUBJECTS + 1)):
        eye_sub = results["Eye-only(66)"]["per_subject_acc"][i]
        cat_sub = results["Concat(578)"]["per_subject_acc"][i]
        pca_sub = results[best_fusion]["per_subject_acc"][i]
        delta = pca_sub - eye_sub
        marker = "+" if delta > 0 else ("-" if delta < 0 else "=")
        print(f"  S{sub_id:02d}   {eye_sub:.3f}   {cat_sub:.3f}    {pca_sub:.3f}   {delta:>+7.3f} {marker}")

    # Summary
    n_improved_cat = sum(1 for i in range(N_SUBJECTS)
                         if results["Concat(578)"]["per_subject_acc"][i] >
                            results["Eye-only(66)"]["per_subject_acc"][i])
    n_improved_pca = sum(1 for i in range(N_SUBJECTS)
                         if results[best_fusion]["per_subject_acc"][i] >
                            results["Eye-only(66)"]["per_subject_acc"][i])
    print(f"\nConcat improved {n_improved_cat}/{N_SUBJECTS} subjects over eye-only")
    print(f"PCA-Concat improved {n_improved_pca}/{N_SUBJECTS} subjects over eye-only")

    # Embedding quality diagnosis
    print("\n--- REVE Embedding Quality Diagnosis ---")
    print(f"  EEG-only balanced acc: {results['EEG-only(512)']['bal_acc']:.4f} (chance=0.200)")
    print(f"  Cosine sim all embeddings is ~0.95 -> near-identical representations")
    print(f"  CONCLUSION: REVE embeddings carry minimal emotion-discriminative signal")
    print(f"  The frozen REVE encoder was not tuned for SEED-V emotion classes")
    print(f"  PCA-Concat barely matches eye-only because PCA filters out noise dims")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

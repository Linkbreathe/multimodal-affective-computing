#!/usr/bin/env python3
"""
Eye-Only Emotion Recognition Baseline on SEED-V with LOSO evaluation.
Diagnostic: does eye tracking carry emotion signal?
"""

import numpy as np
import pickle
import warnings
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    accuracy_score,
)

warnings.filterwarnings("ignore")

DATA_DIR = Path("/mnt/c/Users/Public/Data/SEED-V/SEED-V/Eye_movement_features")
NUM_SUBJECTS = 16
CLASSES = ["Disgust", "Fear", "Sad", "Neutral", "Happy"]
NUM_CLASSES = 5


def load_subject(subj_id: int):
    """Load eye features for one subject. Returns (X_mean, X_meanstd, y)."""
    path = DATA_DIR / f"{subj_id}_123.npz"
    f = np.load(str(path), allow_pickle=True)
    data = pickle.loads(bytes(f["data"]))
    label = pickle.loads(bytes(f["label"]))

    X_mean_list = []
    X_meanstd_list = []
    y_list = []

    for trial_idx in sorted(data.keys()):
        trial_data = data[trial_idx]  # (T, 33)
        trial_label = label[trial_idx]  # (T,)

        # Aggregate: mean and std across time
        feat_mean = np.nanmean(trial_data, axis=0)  # (33,)
        feat_std = np.nanstd(trial_data, axis=0)  # (33,)

        X_mean_list.append(feat_mean)
        X_meanstd_list.append(np.concatenate([feat_mean, feat_std]))  # (66,)

        # Label: take majority (should all be same)
        lbl = int(trial_label[0])
        y_list.append(lbl)

    X_mean = np.array(X_mean_list)      # (45, 33)
    X_meanstd = np.array(X_meanstd_list)  # (45, 66)
    y = np.array(y_list)                 # (45,)

    return X_mean, X_meanstd, y


def run_loso(classifier_fn, feature_type="mean"):
    """Run 16-fold LOSO. Returns per-fold metrics and aggregated confusion matrix."""
    all_bal_acc = []
    all_f1 = []
    all_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    all_per_class_acc = np.zeros((NUM_SUBJECTS, NUM_CLASSES))

    for test_subj in range(1, NUM_SUBJECTS + 1):
        # Collect train and test
        X_train_list, y_train_list = [], []
        X_test, y_test = None, None

        for subj in range(1, NUM_SUBJECTS + 1):
            X_mean, X_meanstd, y = subjects[subj]
            X = X_mean if feature_type == "mean" else X_meanstd

            if subj == test_subj:
                X_test = X
                y_test = y
            else:
                X_train_list.append(X)
                y_train_list.append(y)

        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        # Handle NaN: replace with 0
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)

        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train and predict
        clf = classifier_fn()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Metrics
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=list(range(NUM_CLASSES)))

        all_bal_acc.append(bal_acc)
        all_f1.append(f1)
        all_cm += cm

        # Per-class accuracy for this fold
        for c in range(NUM_CLASSES):
            mask = y_test == c
            if mask.sum() > 0:
                all_per_class_acc[test_subj - 1, c] = (y_pred[mask] == c).mean()

    return {
        "bal_acc_mean": np.mean(all_bal_acc),
        "bal_acc_std": np.std(all_bal_acc),
        "f1_mean": np.mean(all_f1),
        "f1_std": np.std(all_f1),
        "cm": all_cm,
        "per_class_acc": all_per_class_acc.mean(axis=0),
    }


# ==============================================================
# Load all subjects
# ==============================================================
print("Loading 16 subjects...")
subjects = {}
for s in range(1, NUM_SUBJECTS + 1):
    X_mean, X_meanstd, y = load_subject(s)
    subjects[s] = (X_mean, X_meanstd, y)
    label_dist = [int((y == c).sum()) for c in range(NUM_CLASSES)]
    print(f"  Subject {s:2d}: {X_mean.shape[0]} trials, label dist={label_dist}")

# Verify total
total = sum(s[2].shape[0] for s in subjects.values())
print(f"Total trials: {total} ({total / NUM_SUBJECTS:.0f} per subject)")

# ==============================================================
# Define classifiers
# ==============================================================
classifiers = {
    "LR": lambda: LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
    "SVM": lambda: SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced"),
    "MLP": lambda: MLPClassifier(hidden_layer_sizes=(64,), activation="relu", max_iter=200,
                                  early_stopping=True, validation_fraction=0.15,
                                  random_state=42),
}

feature_types = ["mean", "meanstd"]
feature_labels = {"mean": "mean(33)", "meanstd": "mean+std(66)"}

# ==============================================================
# Run all configurations
# ==============================================================
print("\nRunning LOSO evaluation...")
results = {}
best_key = None
best_f1 = -1

for feat in feature_types:
    for clf_name, clf_fn in classifiers.items():
        key = (clf_name, feat)
        print(f"  {clf_name} | {feature_labels[feat]}...", end=" ", flush=True)
        res = run_loso(clf_fn, feature_type=feat)
        results[key] = res
        print(f"BalAcc={res['bal_acc_mean']:.3f} F1={res['f1_mean']:.3f}")

        if res["f1_mean"] > best_f1:
            best_f1 = res["f1_mean"]
            best_key = key

# ==============================================================
# Print summary
# ==============================================================
print("\n" + "=" * 70)
print("=== Eye-Only LOSO Baseline (SEED-V, 5 emotions, 16 subjects) ===")
print("=" * 70)
print(f"{'Classifier':<6} | {'Features':<14} | {'Balanced Acc':>12} | {'Macro F1':>10} | Chance=0.200")
print("-" * 70)

for feat in feature_types:
    for clf_name in classifiers:
        key = (clf_name, feat)
        res = results[key]
        marker = " ***" if key == best_key else ""
        print(
            f"{clf_name:<6} | {feature_labels[feat]:<14} | "
            f"{res['bal_acc_mean']:.3f} +/- {res['bal_acc_std']:.3f} | "
            f"{res['f1_mean']:.3f} +/- {res['f1_std']:.3f} |{marker}"
        )
    print("-" * 70)

# ==============================================================
# Best model details
# ==============================================================
best_res = results[best_key]
print(f"\n*** Best model: {best_key[0]} with {feature_labels[best_key[1]]} ***")
print(f"    Balanced Accuracy: {best_res['bal_acc_mean']:.3f} +/- {best_res['bal_acc_std']:.3f}")
print(f"    Macro F1:          {best_res['f1_mean']:.3f} +/- {best_res['f1_std']:.3f}")

print(f"\nPer-class accuracy (averaged across 16 folds):")
for c in range(NUM_CLASSES):
    bar = "#" * int(best_res["per_class_acc"][c] * 40)
    print(f"  {CLASSES[c]:<8}: {best_res['per_class_acc'][c]:.3f}  {bar}")

print(f"\nAveraged Confusion Matrix (rows=true, cols=pred):")
cm = best_res["cm"]
# Normalize by row
cm_norm = cm / cm.sum(axis=1, keepdims=True)
header = "         " + " ".join(f"{CLASSES[c][:4]:>6}" for c in range(NUM_CLASSES))
print(header)
for r in range(NUM_CLASSES):
    row_str = f"{CLASSES[r]:<8} " + " ".join(f"{cm_norm[r, c]:6.3f}" for c in range(NUM_CLASSES))
    print(row_str)

print(f"\nRaw counts confusion matrix:")
print(header)
for r in range(NUM_CLASSES):
    row_str = f"{CLASSES[r]:<8} " + " ".join(f"{int(cm[r, c]):6d}" for c in range(NUM_CLASSES))
    print(row_str)

# ==============================================================
# Statistical significance vs chance
# ==============================================================
print(f"\n--- Significance Check ---")
print(f"Chance level (5 classes): 0.200")
print(f"Best balanced acc: {best_res['bal_acc_mean']:.3f}")
improvement = (best_res["bal_acc_mean"] - 0.2) / 0.2 * 100
print(f"Relative improvement over chance: {improvement:.1f}%")

if best_res["bal_acc_mean"] > 0.25:
    print("VERDICT: Eye features carry meaningful emotion signal (>25% balanced acc)")
elif best_res["bal_acc_mean"] > 0.22:
    print("VERDICT: Weak but present signal (marginal above chance)")
else:
    print("VERDICT: No clear emotion signal from eye features alone")

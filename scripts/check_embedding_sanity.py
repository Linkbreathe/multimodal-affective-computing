"""Sanity-check EEGPT embeddings: detect degeneracy, compute separability metrics,
and produce a t-SNE / UMAP visualisation.

Usage
-----
    python scripts/check_embedding_sanity.py
    python scripts/check_embedding_sanity.py --embeddings-dir data/embeddings_seedv_4s
    python scripts/check_embedding_sanity.py --embeddings-dir data/embeddings_seedv_4s --vis umap

The script saves all artefacts under logs/embedding_sanity/ and prints a
clear verdict at the end:

    DEGENERATE  — cosine_sim_mean > 0.99 OR per-dimension variance < 1e-6
    USABLE      — otherwise
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering — must come before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Reuse the loader from run_seedv_experiment so we stay in sync with any
# future changes to the canonical loading logic.
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR.parent))

from scripts.run_seedv_experiment import load_embeddings  # noqa: E402

log = logging.getLogger(__name__)

EMOTIONS = ["Disgust", "Fear", "Sad", "Neutral", "Happy"]
DEGENERATE_COSINE_THRESHOLD = 0.99
DEGENERATE_VARIANCE_THRESHOLD = 1e-6
COSINE_SAMPLE_PAIRS = 1_000
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten(
    subject_data: dict[int, dict[str, torch.Tensor]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (embeddings, labels, subject_ids) as numpy arrays."""
    emb_list, lbl_list, sid_list = [], [], []
    for sid, d in sorted(subject_data.items()):
        n = d["embeddings"].shape[0]
        emb_list.append(d["embeddings"].numpy())
        lbl_list.append(d["labels"].numpy())
        sid_list.append(np.full(n, sid, dtype=np.int32))
    return (
        np.concatenate(emb_list, axis=0),
        np.concatenate(lbl_list, axis=0),
        np.concatenate(sid_list, axis=0),
    )


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def check_per_dim_variance(embeddings: np.ndarray) -> dict:
    """Compute per-dimension variance statistics."""
    var = embeddings.var(axis=0)              # shape (D,)
    result = {
        "mean": float(var.mean()),
        "min": float(var.min()),
        "max": float(var.max()),
        "n_dead_dims": int((var < DEGENERATE_VARIANCE_THRESHOLD).sum()),
        "pct_dead_dims": float((var < DEGENERATE_VARIANCE_THRESHOLD).mean() * 100),
    }
    log.info(
        "Per-dim variance — mean: %.4e  min: %.4e  max: %.4e  "
        "dead dims: %d / %d (%.1f%%)",
        result["mean"], result["min"], result["max"],
        result["n_dead_dims"], embeddings.shape[1], result["pct_dead_dims"],
    )
    return result


def check_pairwise_cosine(
    embeddings: np.ndarray,
    n_pairs: int = COSINE_SAMPLE_PAIRS,
    rng: np.random.Generator | None = None,
) -> dict:
    """Sample random pairs and compute cosine similarity statistics."""
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    N = embeddings.shape[0]
    n_pairs = min(n_pairs, N * (N - 1) // 2)

    idx_a = rng.integers(0, N, size=n_pairs)
    idx_b = rng.integers(0, N, size=n_pairs)
    # Avoid self-pairs
    same = idx_a == idx_b
    idx_b[same] = (idx_b[same] + 1) % N

    # L2-normalise
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    normed = embeddings / norms

    cos_sims = (normed[idx_a] * normed[idx_b]).sum(axis=1)

    result = {
        "mean": float(cos_sims.mean()),
        "std": float(cos_sims.std()),
        "min": float(cos_sims.min()),
        "max": float(cos_sims.max()),
        "n_pairs_sampled": int(n_pairs),
    }
    log.info(
        "Pairwise cosine similarity (%d pairs) — mean: %.4f  std: %.4f  "
        "min: %.4f  max: %.4f",
        n_pairs, result["mean"], result["std"], result["min"], result["max"],
    )
    return result


def check_centroid_distances(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: list[str] = EMOTIONS,
) -> dict:
    """Compute per-class centroids and inter-centroid distances."""
    unique_labels = sorted(np.unique(labels))
    centroids = {}
    for lbl in unique_labels:
        mask = labels == lbl
        name = class_names[lbl] if lbl < len(class_names) else str(lbl)
        centroids[name] = embeddings[mask].mean(axis=0)

    names = list(centroids.keys())
    n_cls = len(names)
    dist_matrix = np.zeros((n_cls, n_cls))
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i != j:
                dist_matrix[i, j] = float(
                    np.linalg.norm(centroids[ni] - centroids[nj])
                )

    # Flatten upper triangle
    upper = dist_matrix[np.triu_indices(n_cls, k=1)]
    result = {
        "class_names": names,
        "distance_matrix": dist_matrix.tolist(),
        "mean_inter_centroid_dist": float(upper.mean()) if len(upper) else 0.0,
        "min_inter_centroid_dist": float(upper.min()) if len(upper) else 0.0,
        "max_inter_centroid_dist": float(upper.max()) if len(upper) else 0.0,
    }
    log.info(
        "Inter-centroid distances — mean: %.4f  min: %.4f  max: %.4f",
        result["mean_inter_centroid_dist"],
        result["min_inter_centroid_dist"],
        result["max_inter_centroid_dist"],
    )
    return result


def check_knn_loso(
    subject_data: dict[int, dict[str, torch.Tensor]],
    k_values: tuple[int, ...] = (5, 10),
) -> dict:
    """k-NN accuracy under Leave-One-Subject-Out protocol."""
    subjects = sorted(subject_data.keys())
    results: dict[int, list[float]] = {k: [] for k in k_values}

    for test_subj in subjects:
        train_embs = np.concatenate(
            [subject_data[s]["embeddings"].numpy() for s in subjects if s != test_subj]
        )
        train_labels = np.concatenate(
            [subject_data[s]["labels"].numpy() for s in subjects if s != test_subj]
        )
        test_embs = subject_data[test_subj]["embeddings"].numpy()
        test_labels = subject_data[test_subj]["labels"].numpy()

        # Scale to avoid distance being dominated by high-variance dims
        scaler = StandardScaler()
        train_embs_s = scaler.fit_transform(train_embs)
        test_embs_s = scaler.transform(test_embs)

        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", n_jobs=-1)
            knn.fit(train_embs_s, train_labels)
            preds = knn.predict(test_embs_s)
            results[k].append(float(accuracy_score(test_labels, preds)))

    summary = {}
    for k in k_values:
        summary[f"k{k}_mean"] = float(np.mean(results[k]))
        summary[f"k{k}_std"] = float(np.std(results[k]))
        log.info(
            "k-NN (k=%d) LOSO accuracy — mean: %.4f  std: %.4f",
            k, summary[f"k{k}_mean"], summary[f"k{k}_std"],
        )
    return summary


def check_linear_separability_loso(
    subject_data: dict[int, dict[str, torch.Tensor]],
) -> dict:
    """Logistic regression linear probe under LOSO protocol."""
    subjects = sorted(subject_data.keys())
    fold_accs: list[float] = []

    for test_subj in subjects:
        train_embs = np.concatenate(
            [subject_data[s]["embeddings"].numpy() for s in subjects if s != test_subj]
        )
        train_labels = np.concatenate(
            [subject_data[s]["labels"].numpy() for s in subjects if s != test_subj]
        )
        test_embs = subject_data[test_subj]["embeddings"].numpy()
        test_labels = subject_data[test_subj]["labels"].numpy()

        scaler = StandardScaler()
        train_embs_s = scaler.fit_transform(train_embs)
        test_embs_s = scaler.transform(test_embs)

        clf = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            multi_class="multinomial",
            C=1.0,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        clf.fit(train_embs_s, train_labels)
        preds = clf.predict(test_embs_s)
        fold_accs.append(float(accuracy_score(test_labels, preds)))

    result = {
        "mean_accuracy": float(np.mean(fold_accs)),
        "std_accuracy": float(np.std(fold_accs)),
        "per_fold": fold_accs,
    }
    log.info(
        "Linear probe (LR) LOSO accuracy — mean: %.4f  std: %.4f",
        result["mean_accuracy"], result["std_accuracy"],
    )
    return result


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _build_tsne(embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
    from sklearn.manifold import TSNE
    log.info("Computing t-SNE (perplexity=30) …")
    proj = TSNE(
        n_components=2,
        perplexity=30,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    ).fit_transform(embeddings)
    return proj


def _build_umap(embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
    try:
        import umap  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "umap-learn is not installed. Install it with: pip install umap-learn"
        ) from exc
    log.info("Computing UMAP (n_neighbors=15) …")
    reducer = umap.UMAP(n_neighbors=15, random_state=RANDOM_SEED)
    return reducer.fit_transform(embeddings)


def save_visualisation(
    embeddings: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    method: str = "tsne",
    max_samples: int = 5_000,
) -> None:
    """Down-sample if needed, project to 2-D, and save scatter PNG."""
    rng = np.random.default_rng(RANDOM_SEED)
    N = embeddings.shape[0]
    if N > max_samples:
        idx = rng.choice(N, size=max_samples, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]

    if method == "umap":
        proj = _build_umap(embeddings, labels)
    else:
        proj = _build_tsne(embeddings, labels)

    cmap = matplotlib.colormaps.get_cmap("tab10").resampled(len(EMOTIONS))
    fig, ax = plt.subplots(figsize=(8, 6))
    for cls_idx, cls_name in enumerate(EMOTIONS):
        mask = labels == cls_idx
        if mask.any():
            ax.scatter(
                proj[mask, 0],
                proj[mask, 1],
                c=[cmap(cls_idx)],
                label=cls_name,
                s=10,
                alpha=0.6,
                linewidths=0,
            )
    ax.set_title(f"EEGPT embeddings — {method.upper()} projection")
    ax.legend(markerscale=2, fontsize=9)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Visualisation saved to %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _verdict(
    cosine_mean: float,
    variance_mean: float,
    variance_min: float,
) -> str:
    if cosine_mean > DEGENERATE_COSINE_THRESHOLD:
        return "DEGENERATE"
    if variance_min < DEGENERATE_VARIANCE_THRESHOLD:
        return "DEGENERATE"
    return "USABLE"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sanity-check EEGPT embeddings for degeneracy and separability."
    )
    parser.add_argument(
        "--embeddings-dir",
        default="data/embeddings_seedv",
        help=(
            "Root embeddings directory.  The script appends /eegpt_eeg/ internally "
            "(matching run_seedv_experiment.py behaviour).  "
            "Default: data/embeddings_seedv"
        ),
    )
    parser.add_argument(
        "--vis",
        choices=["tsne", "umap", "none"],
        default="tsne",
        help="Visualisation method (default: tsne).",
    )
    parser.add_argument(
        "--output-dir",
        default="logs/embedding_sanity",
        help="Directory for output artefacts (default: logs/embedding_sanity).",
    )
    parser.add_argument(
        "--no-knn",
        action="store_true",
        help="Skip k-NN LOSO (faster run on large datasets).",
    )
    parser.add_argument(
        "--no-linear",
        action="store_true",
        help="Skip linear probe LOSO (faster run on large datasets).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Unused; accepted for script-family consistency.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # ------------------------------------------------------------------
    # Resolve paths relative to the project root (cwd) so the script
    # works when called from any directory.
    # ------------------------------------------------------------------
    project_root = Path(__file__).resolve().parent.parent
    embeddings_dir = (project_root / args.embeddings_dir).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EEGPT Embedding Sanity Check")
    print("=" * 70)
    print(f"  Embeddings root : {embeddings_dir}")
    print(f"  eegpt_eeg path  : {embeddings_dir / 'eegpt_eeg'}")
    print(f"  Output dir      : {output_dir}")
    print()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    log.info("Loading embeddings from %s …", embeddings_dir)
    subject_data = load_embeddings(embeddings_dir)

    if not subject_data:
        print("ERROR: No embeddings found.  Check --embeddings-dir.")
        sys.exit(1)

    subjects = sorted(subject_data.keys())
    total_samples = sum(d["embeddings"].shape[0] for d in subject_data.values())
    emb_dim = next(iter(subject_data.values()))["embeddings"].shape[1]
    print(f"  Subjects loaded  : {len(subjects)}  {subjects}")
    print(f"  Total segments   : {total_samples}")
    print(f"  Embedding dim    : {emb_dim}")
    print()

    embeddings, labels, subject_ids = _flatten(subject_data)

    # ------------------------------------------------------------------
    # 1. Per-dimension variance
    # ------------------------------------------------------------------
    print("-" * 70)
    print("1. Per-dimension variance")
    print("-" * 70)
    variance_stats = check_per_dim_variance(embeddings)
    print(f"   Mean variance : {variance_stats['mean']:.4e}")
    print(f"   Min  variance : {variance_stats['min']:.4e}")
    print(f"   Max  variance : {variance_stats['max']:.4e}")
    print(
        f"   Dead dims (<{DEGENERATE_VARIANCE_THRESHOLD:.0e}) : "
        f"{variance_stats['n_dead_dims']} / {emb_dim} "
        f"({variance_stats['pct_dead_dims']:.1f}%)"
    )
    print()

    # ------------------------------------------------------------------
    # 2. Pairwise cosine similarity
    # ------------------------------------------------------------------
    print("-" * 70)
    print(f"2. Pairwise cosine similarity  (sampling {COSINE_SAMPLE_PAIRS} pairs)")
    print("-" * 70)
    cosine_stats = check_pairwise_cosine(embeddings)
    print(f"   Mean : {cosine_stats['mean']:.4f}")
    print(f"   Std  : {cosine_stats['std']:.4f}")
    print(f"   Min  : {cosine_stats['min']:.4f}")
    print(f"   Max  : {cosine_stats['max']:.4f}")
    print()

    # ------------------------------------------------------------------
    # 3. Per-class centroid distances
    # ------------------------------------------------------------------
    print("-" * 70)
    print("3. Per-class centroid distances")
    print("-" * 70)
    centroid_stats = check_centroid_distances(embeddings, labels)
    print(
        f"   Mean inter-centroid distance : {centroid_stats['mean_inter_centroid_dist']:.4f}"
    )
    print(
        f"   Min  inter-centroid distance : {centroid_stats['min_inter_centroid_dist']:.4f}"
    )
    print(
        f"   Max  inter-centroid distance : {centroid_stats['max_inter_centroid_dist']:.4f}"
    )
    cls_names = centroid_stats["class_names"]
    dist_mat = np.array(centroid_stats["distance_matrix"])
    col_w = max(len(n) for n in cls_names) + 2
    header = f"{'':>{col_w}}" + "".join(f"{n:>{col_w}}" for n in cls_names)
    print(f"\n   {header}")
    for i, ni in enumerate(cls_names):
        row = f"   {ni:>{col_w}}" + "".join(
            f"{dist_mat[i, j]:>{col_w}.2f}" for j in range(len(cls_names))
        )
        print(row)
    print()

    # ------------------------------------------------------------------
    # 4. k-NN accuracy (LOSO)
    # ------------------------------------------------------------------
    knn_stats: dict = {}
    if not args.no_knn:
        print("-" * 70)
        print("4. k-NN accuracy (LOSO, k=5 and k=10)")
        print("-" * 70)
        knn_stats = check_knn_loso(subject_data, k_values=(5, 10))
        print(f"   k=5  : {knn_stats['k5_mean']:.4f} +/- {knn_stats['k5_std']:.4f}")
        print(f"   k=10 : {knn_stats['k10_mean']:.4f} +/- {knn_stats['k10_std']:.4f}")
        print()
    else:
        print("-" * 70)
        print("4. k-NN accuracy (LOSO) — SKIPPED (--no-knn)")
        print("-" * 70)
        print()

    # ------------------------------------------------------------------
    # 5. Visualisation
    # ------------------------------------------------------------------
    vis_path: Path | None = None
    if args.vis != "none":
        print("-" * 70)
        print(f"5. Visualisation ({args.vis.upper()})")
        print("-" * 70)
        vis_path = output_dir / f"embedding_{args.vis}.png"
        save_visualisation(embeddings, labels, vis_path, method=args.vis)
        print(f"   Saved: {vis_path}")
        print()
    else:
        print("-" * 70)
        print("5. Visualisation — SKIPPED (--vis none)")
        print("-" * 70)
        print()

    # ------------------------------------------------------------------
    # 6. Linear separability (Logistic Regression, LOSO)
    # ------------------------------------------------------------------
    linear_stats: dict = {}
    if not args.no_linear:
        print("-" * 70)
        print("6. Linear separability (Logistic Regression, LOSO)")
        print("-" * 70)
        linear_stats = check_linear_separability_loso(subject_data)
        print(
            f"   Accuracy : {linear_stats['mean_accuracy']:.4f} "
            f"+/- {linear_stats['std_accuracy']:.4f}"
        )
        print()
    else:
        print("-" * 70)
        print("6. Linear separability (Logistic Regression) — SKIPPED (--no-linear)")
        print("-" * 70)
        print()

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    verdict = _verdict(
        cosine_mean=cosine_stats["mean"],
        variance_mean=variance_stats["mean"],
        variance_min=variance_stats["min"],
    )

    print("=" * 70)
    print(f"VERDICT: {verdict}")
    print("=" * 70)
    if verdict == "DEGENERATE":
        reasons = []
        if cosine_stats["mean"] > DEGENERATE_COSINE_THRESHOLD:
            reasons.append(
                f"cosine_sim_mean={cosine_stats['mean']:.4f} > {DEGENERATE_COSINE_THRESHOLD}"
            )
        if variance_stats["min"] < DEGENERATE_VARIANCE_THRESHOLD:
            reasons.append(
                f"min_variance={variance_stats['min']:.4e} < {DEGENERATE_VARIANCE_THRESHOLD:.0e}"
            )
        for r in reasons:
            print(f"  Reason: {r}")
    else:
        print("  Embeddings pass basic non-degeneracy checks.")
    print()

    # ------------------------------------------------------------------
    # Persist results
    # ------------------------------------------------------------------
    results = {
        "embeddings_dir": str(embeddings_dir),
        "n_subjects": len(subjects),
        "subjects": subjects,
        "total_samples": total_samples,
        "embedding_dim": emb_dim,
        "variance": variance_stats,
        "cosine_similarity": cosine_stats,
        "centroid_distances": centroid_stats,
        "knn_loso": knn_stats,
        "linear_probe_loso": linear_stats,
        "verdict": verdict,
        "vis_method": args.vis,
        "vis_path": str(vis_path) if vis_path else None,
    }

    results_path = output_dir / "sanity_results.json"
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2)
    log.info("Full results saved to %s", results_path)
    print(f"Full results saved to: {results_path}")
    if vis_path:
        print(f"Visualisation saved to: {vis_path}")

    # Return non-zero exit code so CI pipelines can catch degenerate embeddings
    sys.exit(1 if verdict == "DEGENERATE" else 0)


if __name__ == "__main__":
    main()

"""I.1 Group response surface + trade-off frontier  [A level].

Fit response surfaces for relaxation, discomfort and calm as a function of the
(intensity, frequency) lattice.  We deliberately compare a parametric bilinear/
quadratic surface against a saturated condition-as-factor (cell-mean) model
under leave-one-participant-out CV: if the parametric surface is mis-specified
it would push the optimum to a lattice corner, so the factor model is the
honest reference for "which cell is best".

Outputs:
  - reports/response_surface.csv        (per-cell estimates + CIs, model CV)
  - figures/response_surface.png        (heatmaps with the optimum marked)
  - figures/tradeoff_frontier.png       (relaxation vs discomfort frontier)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from io_common import (
    CONDITIONS,
    INFERENCE_PC,
    add_contract_columns,
    cluster_bootstrap,
    condition_grid,
    condition_sort_key,
    ensure_dirs,
    load_labels,
    write_csv,
)

SURFACE_TARGETS = ("relaxation", "discomfort", "calm")
# direction of "better": maximise relaxation/calm, minimise discomfort
TARGET_DIRECTION = {"relaxation": "max", "calm": "max", "discomfort": "min"}


def _design(intensity: np.ndarray, frequency: np.ndarray, kind: str) -> np.ndarray:
    i = intensity
    f = frequency
    if kind == "bilinear":
        return np.column_stack([np.ones_like(i), i, f, i * f])
    if kind == "quadratic":
        return np.column_stack([np.ones_like(i), i, f, i * f, i ** 2, f ** 2])
    raise ValueError(kind)


def _ols_predict(train: pd.DataFrame, test: pd.DataFrame, target: str, kind: str) -> np.ndarray:
    xtr = _design(train["intensity"].to_numpy(float), train["frequency"].to_numpy(float), kind)
    ytr = pd.to_numeric(train[target], errors="coerce").to_numpy(float)
    beta, *_ = np.linalg.lstsq(xtr, ytr, rcond=None)
    xte = _design(test["intensity"].to_numpy(float), test["frequency"].to_numpy(float), kind)
    return xte @ beta


def _factor_predict(train: pd.DataFrame, test: pd.DataFrame, target: str) -> np.ndarray:
    cell_mean = train.groupby("condition")[target].mean()
    grand = pd.to_numeric(train[target], errors="coerce").mean()
    return test["condition"].map(cell_mean).fillna(grand).to_numpy(float)


def lopo_cv(labels: pd.DataFrame, target: str) -> dict[str, float]:
    """Leave-one-participant-out MAE for each surface model."""
    participants = sorted(labels["participant_id"].unique())
    errs = {"bilinear": [], "quadratic": [], "factor": [], "grand_mean": []}
    for held in participants:
        train = labels[labels["participant_id"] != held]
        test = labels[labels["participant_id"] == held]
        truth = pd.to_numeric(test[target], errors="coerce").to_numpy(float)
        preds = {
            "bilinear": _ols_predict(train, test, target, "bilinear"),
            "quadratic": _ols_predict(train, test, target, "quadratic"),
            "factor": _factor_predict(train, test, target),
            "grand_mean": np.full(len(test), pd.to_numeric(train[target], errors="coerce").mean()),
        }
        for key, pred in preds.items():
            errs[key].append(np.abs(truth - pred))
    return {key: float(np.mean(np.concatenate(vals))) for key, vals in errs.items()}


def run(seed: int, replicates: int) -> dict[str, object]:
    paths = ensure_dirs()
    labels = load_labels()
    grid = condition_grid().set_index("condition")

    rows: list[dict] = []
    cv_rows: list[dict] = []
    cell_estimates: dict[str, dict[str, float]] = {t: {} for t in SURFACE_TARGETS}

    for target in SURFACE_TARGETS:
        # --- per-cell mean with participant-cluster CI -----------------------
        for condition in CONDITIONS:
            cell = labels[labels["condition"] == condition][["participant_id", target]].copy()
            boot = cluster_bootstrap(
                cell, lambda s, t=target: float(pd.to_numeric(s[t], errors="coerce").mean()),
                seed=seed + condition_sort_key(condition) + len(target), replicates=replicates,
            )
            cell_estimates[target][condition] = boot["estimate"]
            rows.append(
                {
                    "target": target,
                    "condition": condition,
                    "intensity": float(grid.loc[condition, "intensity"]),
                    "frequency": float(grid.loc[condition, "frequency"]),
                    "cell_mean": boot["estimate"],
                    "ci_low": boot["ci_low"],
                    "ci_high": boot["ci_high"],
                    "bootstrap_failures": boot["bootstrap_failures"],
                    "n_participants": int(cell["participant_id"].nunique()),
                    "inference_unit": INFERENCE_PC,
                }
            )

        # --- model comparison via LOPO ---------------------------------------
        cv = lopo_cv(labels, target)
        best_model = min(cv, key=cv.get)
        for model_name, mae in cv.items():
            cv_rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "lopo_mae": mae,
                    "is_best": bool(model_name == best_model),
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "inference_unit": INFERENCE_PC,
                }
            )

    surface = pd.DataFrame(rows)
    cv_frame = pd.DataFrame(cv_rows)

    # --- single best default condition (relaxation primary, comfort tiebreak)
    relax = pd.Series(cell_estimates["relaxation"])
    disc = pd.Series(cell_estimates["discomfort"])
    # rank by relaxation desc, then discomfort asc
    ranking = pd.DataFrame({"relaxation": relax, "discomfort": disc})
    ranking = ranking.sort_values(["relaxation", "discomfort"], ascending=[False, True])
    best_default = ranking.index[0]

    # bootstrap the share of bootstrap replicates in which each condition is the
    # group-level relaxation argmax (winner stability).
    argmax_counts = _argmax_stability(labels, "relaxation", seed=seed + 999, replicates=replicates)

    # --- trade-off frontier: Pareto set on (max relaxation, min discomfort) --
    frontier = _pareto_frontier(relax, disc)

    summary_rows = [
        {
            "target": "default_condition",
            "condition": best_default,
            "intensity": float(grid.loc[best_default, "intensity"]),
            "frequency": float(grid.loc[best_default, "frequency"]),
            "cell_mean": float(relax[best_default]),
            "ci_low": np.nan,
            "ci_high": np.nan,
            "bootstrap_failures": 0,
            "n_participants": 15,
            "inference_unit": INFERENCE_PC,
            "note": f"argmax relaxation; bootstrap argmax share={argmax_counts.get(best_default, 0.0):.3f}; "
            f"on_pareto_frontier={best_default in frontier}",
        }
    ]
    surface = pd.concat([surface, pd.DataFrame(summary_rows)], ignore_index=True)
    surface["on_tradeoff_frontier"] = surface["condition"].isin(frontier)
    surface["argmax_relaxation_bootstrap_share"] = surface["condition"].map(argmax_counts)

    write_csv(paths["reports"] / "response_surface.csv", surface)
    write_csv(paths["reports"] / "response_surface_model_cv.csv", cv_frame)

    _figures(paths, cell_estimates, relax, disc, frontier, best_default)

    return {
        "surface": add_contract_columns(surface, INFERENCE_PC),
        "cv": cv_frame,
        "best_default": best_default,
        "frontier": frontier,
        "argmax_share": argmax_counts,
        "cell_estimates": cell_estimates,
    }


def _argmax_stability(labels: pd.DataFrame, target: str, seed: int, replicates: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    participants = np.asarray(sorted(labels["participant_id"].unique()))
    counts = {c: 0 for c in CONDITIONS}
    grouped = {p: labels[labels["participant_id"] == p] for p in participants}
    for _ in range(replicates):
        sampled = rng.choice(participants, size=len(participants), replace=True)
        sample = pd.concat([grouped[p] for p in sampled], ignore_index=True)
        means = sample.groupby("condition")[target].mean()
        winner = means.sort_values(ascending=False).index[0]
        counts[winner] += 1
    return {c: counts[c] / replicates for c in CONDITIONS}


def _pareto_frontier(relax: pd.Series, disc: pd.Series) -> list[str]:
    frontier = []
    for c in CONDITIONS:
        dominated = False
        for other in CONDITIONS:
            if other == c:
                continue
            if (relax[other] >= relax[c] and disc[other] <= disc[c] and
                    (relax[other] > relax[c] or disc[other] < disc[c])):
                dominated = True
                break
        if not dominated:
            frontier.append(c)
    return sorted(frontier, key=condition_sort_key)


def _grid_image(values: dict[str, float]) -> np.ndarray:
    grid = condition_grid()
    image = np.full((3, 3), np.nan)
    for _, row in grid.iterrows():
        image[int(row["intensity_index"]), int(row["frequency_index"])] = values[row["condition"]]
    return image


def _figures(paths, cell_estimates, relax, disc, frontier, best_default) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from io_common import INTENSITIES, FREQUENCIES

    fig, axes = plt.subplots(1, len(SURFACE_TARGETS), figsize=(13, 4), constrained_layout=True)
    for axis, target in zip(axes, SURFACE_TARGETS):
        image = _grid_image(cell_estimates[target])
        cmap = "viridis" if TARGET_DIRECTION[target] == "max" else "magma_r"
        im = axis.imshow(image, origin="lower", cmap=cmap, aspect="auto")
        axis.set_xticks(range(3), [f"{f:.2f}" for f in FREQUENCIES])
        axis.set_yticks(range(3), [f"{i:.2f}" for i in INTENSITIES])
        axis.set_xlabel("frequency")
        axis.set_ylabel("intensity")
        axis.set_title(f"{target}  ({'↑best' if TARGET_DIRECTION[target]=='max' else '↓best'})")
        for ii in range(3):
            for ff in range(3):
                cond = CONDITIONS[ii * 3 + ff]
                axis.text(ff, ii, f"{cond}\n{image[ii, ff]:.2f}", ha="center", va="center",
                          color="white", fontsize=8)
        fig.colorbar(im, ax=axis, shrink=0.8)
    fig.suptitle("I.1 Group response surfaces (participant-condition cell means)")
    fig.savefig(paths["figures"] / "response_surface.png", dpi=170)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
    for c in CONDITIONS:
        on_front = c in frontier
        axis.scatter(disc[c], relax[c], s=120 if on_front else 60,
                     color="#ef4444" if on_front else "#9ca3af",
                     edgecolor="black", zorder=3 if on_front else 2)
        axis.annotate(c, (disc[c], relax[c]), textcoords="offset points", xytext=(6, 4), fontsize=9)
    front_sorted = sorted(frontier, key=lambda c: disc[c])
    axis.plot([disc[c] for c in front_sorted], [relax[c] for c in front_sorted],
              color="#ef4444", linestyle="--", zorder=1, label="Pareto frontier")
    axis.scatter([disc[best_default]], [relax[best_default]], marker="*", s=320,
                 color="#10b981", edgecolor="black", zorder=4, label=f"default={best_default}")
    axis.set_xlabel("discomfort (lower better)")
    axis.set_ylabel("relaxation (higher better)")
    axis.set_title("I.1 Relaxation vs discomfort trade-off frontier")
    axis.legend()
    fig.savefig(paths["figures"] / "tradeoff_frontier.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    out = run(seed=20260627, replicates=500)
    print("best default:", out["best_default"])
    print("frontier:", out["frontier"])
    print(out["cv"].to_string(index=False))

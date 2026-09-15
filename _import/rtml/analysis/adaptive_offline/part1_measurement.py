"""I.4 Measurement model (latent constructs)  [A/B level].

Factor-analyse the six 1-7 self-report items (here on a 0-1 normalized scale)
to synthesise a latent ``comfort`` construct, report its reliability, and test
whether ``calm`` and ``monotony`` behave as independent dimensions (Phase A
flagged monotony peaking at C5).

Outputs:
  - models/measurement_model.joblib   (loadings + comfort transform for the sim)
  - reports/latent_constructs.csv      (loadings, reliability, dimensionality)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib

from io_common import (
    INFERENCE_PC,
    LABELS,
    add_contract_columns,
    cluster_bootstrap,
    ensure_dirs,
    load_labels,
    write_csv,
)


# Positive-comfort orientation for each item (discomfort & monotony reduce comfort).
COMFORT_SIGN = {
    "visual_fit": +1.0,
    "pleasantness": +1.0,
    "calm": +1.0,
    "relaxation": +1.0,
    "monotony": -1.0,
    "discomfort": -1.0,
}
# Items entering the internal-consistency (Cronbach alpha) comfort composite.
COMFORT_COMPOSITE_ITEMS = ("pleasantness", "calm", "relaxation", "monotony", "discomfort")


def _matrix(labels: pd.DataFrame) -> np.ndarray:
    return labels[list(LABELS)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def cronbach_alpha(frame: pd.DataFrame, items: tuple[str, ...]) -> float:
    data = frame[list(items)].apply(pd.to_numeric, errors="coerce")
    # orient every item toward higher comfort before summing
    oriented = pd.DataFrame(
        {item: COMFORT_SIGN[item] * data[item] for item in items}
    ).dropna()
    if len(oriented) < 3:
        return float("nan")
    k = oriented.shape[1]
    item_var = oriented.var(axis=0, ddof=1).sum()
    total_var = oriented.sum(axis=1).var(ddof=1)
    if total_var <= 0:
        return float("nan")
    return float((k / (k - 1.0)) * (1.0 - item_var / total_var))


def fit_factor_analysis(matrix: np.ndarray, n_factors: int, seed: int):
    from sklearn.decomposition import FactorAnalysis

    finite = matrix[np.isfinite(matrix).all(axis=1)]
    model = FactorAnalysis(n_components=n_factors, rotation="varimax", random_state=seed)
    model.fit(finite)
    return model


def variance_explained(matrix: np.ndarray, n_factors: int, seed: int) -> float:
    model = fit_factor_analysis(matrix, n_factors, seed)
    # communalities / total variance after standardising columns
    loadings = model.components_.T  # (n_items, n_factors)
    communalities = np.sum(loadings ** 2, axis=1)
    total = matrix.shape[1]  # each standardised item has unit variance
    return float(np.sum(communalities) / total)


def run(seed: int, replicates: int) -> dict[str, object]:
    paths = ensure_dirs()
    labels = load_labels()

    # standardise items column-wise for factor analysis / loadings
    raw = _matrix(labels)
    col_mean = np.nanmean(raw, axis=0)
    col_std = np.nanstd(raw, axis=0, ddof=1)
    col_std[col_std == 0] = 1.0
    standardized = (raw - col_mean) / col_std

    rows: list[dict] = []

    # --- 1. one- and two-factor loadings -------------------------------------
    for n_factors in (1, 2):
        model = fit_factor_analysis(standardized, n_factors, seed)
        loadings = model.components_.T  # (6, n_factors)
        for item_index, item in enumerate(LABELS):
            for factor_index in range(n_factors):
                rows.append(
                    {
                        "analysis": f"factor_analysis_{n_factors}f",
                        "item": item,
                        "factor": f"F{factor_index + 1}",
                        "loading": float(loadings[item_index, factor_index]),
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "inference_unit": INFERENCE_PC,
                    }
                )

    # --- 2. variance explained (1F, 2F, 3F) with participant CI ---------------
    item_columns = list(LABELS)
    for n_factors in (1, 2, 3):
        def stat(sample: pd.DataFrame, n_factors=n_factors) -> float:
            mat = sample[item_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            mat = mat[np.isfinite(mat).all(axis=1)]
            mu = np.nanmean(mat, axis=0)
            sd = np.nanstd(mat, axis=0, ddof=1)
            sd[sd == 0] = 1.0
            return variance_explained((mat - mu) / sd, n_factors, seed)

        boot = cluster_bootstrap(
            labels[["participant_id", *item_columns]].copy(), stat,
            seed=seed + 10 * n_factors, replicates=replicates,
        )
        rows.append(
            {
                "analysis": "variance_explained",
                "item": "ALL",
                "factor": f"{n_factors}_factor_model",
                "loading": boot["estimate"],
                "ci_low": boot["ci_low"],
                "ci_high": boot["ci_high"],
                "inference_unit": INFERENCE_PC,
            }
        )

    # --- 3. reliability of the comfort composite ------------------------------
    alpha_boot = cluster_bootstrap(
        labels[["participant_id", *COMFORT_COMPOSITE_ITEMS]].copy(),
        lambda s: cronbach_alpha(s, COMFORT_COMPOSITE_ITEMS),
        seed=seed + 7, replicates=replicates,
    )
    rows.append(
        {
            "analysis": "reliability",
            "item": "+".join(COMFORT_COMPOSITE_ITEMS),
            "factor": "cronbach_alpha",
            "loading": alpha_boot["estimate"],
            "ci_low": alpha_boot["ci_low"],
            "ci_high": alpha_boot["ci_high"],
            "inference_unit": INFERENCE_PC,
        }
    )

    # --- 4. dimensionality check: calm vs monotony independence ---------------
    corr = labels[list(LABELS)].apply(pd.to_numeric, errors="coerce").corr()
    for a, b in (("calm", "monotony"), ("calm", "relaxation"), ("monotony", "relaxation"),
                 ("monotony", "discomfort"), ("calm", "discomfort")):
        def corr_stat(sample: pd.DataFrame, a=a, b=b) -> float:
            x = pd.to_numeric(sample[a], errors="coerce")
            y = pd.to_numeric(sample[b], errors="coerce")
            return float(x.corr(y))

        boot = cluster_bootstrap(
            labels[["participant_id", a, b]].copy(), corr_stat,
            seed=seed + hash((a, b)) % 9973, replicates=replicates,
        )
        rows.append(
            {
                "analysis": "pairwise_correlation",
                "item": f"{a}~{b}",
                "factor": "pearson_r",
                "loading": boot["estimate"],
                "ci_low": boot["ci_low"],
                "ci_high": boot["ci_high"],
                "inference_unit": INFERENCE_PC,
            }
        )

    table = pd.DataFrame(rows)
    write_csv(paths["reports"] / "latent_constructs.csv", table)

    # --- 5. persist the comfort transform for the digital twin ---------------
    # comfort_latent = mean over standardized, comfort-oriented items.
    signs = np.asarray([COMFORT_SIGN[item] for item in LABELS], dtype=float)
    comfort_scores = np.nanmean((standardized * signs), axis=1)
    comfort_mean = float(np.nanmean(comfort_scores))
    comfort_std = float(np.nanstd(comfort_scores, ddof=1))

    one_factor = fit_factor_analysis(standardized, 1, seed)
    bundle = {
        "items": list(LABELS),
        "comfort_sign": COMFORT_SIGN,
        "column_mean": dict(zip(LABELS, col_mean.tolist())),
        "column_std": dict(zip(LABELS, col_std.tolist())),
        "comfort_score_mean": comfort_mean,
        "comfort_score_std": comfort_std,
        "one_factor_loadings": dict(zip(LABELS, one_factor.components_.T[:, 0].tolist())),
        "cronbach_alpha_comfort": alpha_boot["estimate"],
        "composite_items": list(COMFORT_COMPOSITE_ITEMS),
        "note": "comfort_latent = mean of comfort-oriented z-scored items; "
        "relaxation/discomfort dominate; not validated as an effectiveness reward.",
        "inference_unit": INFERENCE_PC,
    }
    joblib.dump(bundle, paths["models"] / "measurement_model.joblib")

    return {
        "table": add_contract_columns(table, INFERENCE_PC),
        "alpha": alpha_boot,
        "corr": corr,
        "comfort_loadings": bundle["one_factor_loadings"],
    }


if __name__ == "__main__":
    out = run(seed=20260627, replicates=500)
    print(out["table"].to_string(index=False))

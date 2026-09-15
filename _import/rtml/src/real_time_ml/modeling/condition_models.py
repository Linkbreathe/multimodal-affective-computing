from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class SparseFeatureFilter(BaseEstimator, TransformerMixin):
    def __init__(self, min_non_missing_fraction: float = 0.40):
        self.min_non_missing_fraction = min_non_missing_fraction

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        self.keep_mask_ = np.mean(np.isfinite(values), axis=0) >= self.min_non_missing_fraction
        if not np.any(self.keep_mask_):
            self.keep_mask_[0] = True
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_mask_]


class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        if values.shape[1] <= 1:
            self.keep_mask_ = np.ones(values.shape[1], dtype=bool)
            return self
        correlation = np.abs(np.corrcoef(values, rowvar=False))
        correlation = np.nan_to_num(correlation, nan=0.0)
        np.fill_diagonal(correlation, 0.0)
        keep = np.ones(values.shape[1], dtype=bool)
        for column in range(values.shape[1]):
            if keep[column] and np.any(correlation[:column, column][keep[:column]] >= self.threshold):
                keep[column] = False
        if not np.any(keep):
            keep[0] = True
        self.keep_mask_ = keep
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_mask_]


class AdaptiveSelectKBest(BaseEstimator, TransformerMixin):
    def __init__(self, score_func: Callable, k: int = 20):
        self.score_func = score_func
        self.k = k

    def fit(self, X, y):
        from sklearn.feature_selection import SelectKBest

        actual_k = max(1, min(int(self.k), np.asarray(X).shape[1]))
        self.selector_ = SelectKBest(score_func=self.score_func, k=actual_k)
        self.selector_.fit(X, y)
        return self

    def transform(self, X):
        return self.selector_.transform(X)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    feature_count: int

    def label(self) -> str:
        return f"{self.name}:k={self.feature_count}"


def make_regression_pipeline(spec: ModelSpec, seed: int, min_non_missing: float, correlation_threshold: float):
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR

    if spec.name == "ridge":
        model = Ridge(alpha=10.0)
    elif spec.name == "elastic_net":
        model = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=30_000, random_state=seed)
    elif spec.name == "svr":
        model = SVR(C=1.0, epsilon=0.05, gamma="scale")
    elif spec.name == "random_forest":
        model = RandomForestRegressor(
            n_estimators=120, max_features=0.6, min_samples_leaf=3, random_state=seed, n_jobs=-1
        )
    elif spec.name == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=120, max_features=0.6, min_samples_leaf=3, random_state=seed, n_jobs=-1
        )
    elif spec.name == "hist_gradient_boosting":
        model = HistGradientBoostingRegressor(
            learning_rate=0.06, max_leaf_nodes=7, l2_regularization=1.0, max_iter=80, random_state=seed
        )
    else:
        raise ValueError(f"Unknown regression model: {spec.name}")
    return Pipeline([
        ("sparse", SparseFeatureFilter(min_non_missing)),
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("variance", __import__("sklearn.feature_selection", fromlist=["VarianceThreshold"]).VarianceThreshold()),
        ("correlation", CorrelationFilter(correlation_threshold)),
        ("scale", StandardScaler()),
        ("select", AdaptiveSelectKBest(partial(mutual_info_regression, random_state=seed), spec.feature_count)),
        ("model", model),
    ])


def make_risk_pipeline(spec: ModelSpec, seed: int, min_non_missing: float, correlation_threshold: float):
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if spec.name == "logistic_regression":
        model = LogisticRegression(C=0.5, class_weight="balanced", max_iter=10_000, random_state=seed)
    elif spec.name == "svm_classifier":
        # ``probability=True`` triggers a hidden five-fold calibration for every nested CV fit.
        # We use the signed SVM margin with a stable sigmoid in condition_train instead.
        model = SVC(C=1.0, class_weight="balanced", random_state=seed)
    else:
        raise ValueError(f"Unknown risk classifier: {spec.name}")
    return Pipeline([
        ("sparse", SparseFeatureFilter(min_non_missing)),
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("variance", __import__("sklearn.feature_selection", fromlist=["VarianceThreshold"]).VarianceThreshold()),
        ("correlation", CorrelationFilter(correlation_threshold)),
        ("scale", StandardScaler()),
        ("select", AdaptiveSelectKBest(partial(mutual_info_classif, random_state=seed), spec.feature_count)),
        ("model", model),
    ])


def condition_baseline(train_frame, target: str) -> tuple[dict[str, float], float]:
    means = train_frame.groupby("condition")[target].mean()
    return {str(condition): float(value) for condition, value in means.items()}, float(train_frame[target].mean())


def apply_condition_baseline(conditions, mapping: dict[str, float], fallback: float) -> np.ndarray:
    return np.asarray([mapping.get(str(condition), fallback) for condition in conditions], dtype=float)

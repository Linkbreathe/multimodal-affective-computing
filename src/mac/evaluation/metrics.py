"""Evaluation metrics: weighted F1, CCC, class weights."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import f1_score


def weighted_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def concordance_correlation_coefficient(
    y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    """Lin's Concordance Correlation Coefficient."""
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    if denominator == 0:
        return 0.0
    return float(2 * covariance / denominator)


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Inverse frequency class weights for weighted CE loss."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.maximum(counts, 1.0)
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)

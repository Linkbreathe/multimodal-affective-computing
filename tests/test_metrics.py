import pytest
import numpy as np
import torch
from mac.evaluation.metrics import weighted_f1_score, concordance_correlation_coefficient, compute_class_weights

def test_weighted_f1_perfect():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    assert weighted_f1_score(y_true, y_pred) == 1.0

def test_weighted_f1_random():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([1, 2, 0, 2, 0, 1])
    f1 = weighted_f1_score(y_true, y_pred)
    assert 0.0 <= f1 <= 1.0

def test_ccc_perfect():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ccc = concordance_correlation_coefficient(x, x)
    assert abs(ccc - 1.0) < 1e-6

def test_ccc_anticorrelated():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    ccc = concordance_correlation_coefficient(x, y)
    assert ccc < 0

def test_class_weights():
    labels = np.array([0, 0, 0, 1, 2])
    weights = compute_class_weights(labels, num_classes=3)
    assert isinstance(weights, torch.Tensor)
    assert weights.shape == (3,)
    assert weights[0] < weights[1]  # class 0 more frequent, lower weight
    assert weights[0] < weights[2]

import numpy as np

from amlgnn.metrics import best_threshold, compute_metrics


def test_perfect_scores():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(y, p)
    assert m["f1"] == 1.0 and m["pr_auc"] == 1.0 and m["roc_auc"] == 1.0


def test_best_threshold_separates():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.05, 0.1, 0.3, 0.6, 0.7])
    t = best_threshold(y, p)
    assert 0.3 < t <= 0.6


def test_single_class_is_safe():
    m = compute_metrics(np.zeros(5), np.linspace(0, 1, 5))
    assert np.isnan(m["pr_auc"])
    assert best_threshold(np.zeros(5), np.linspace(0, 1, 5)) == 0.5

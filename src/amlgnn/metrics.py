"""Metrics focused on the minority (illicit) class - accuracy alone is misleading here (~10% illicit)."""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_recall_curve, precision_recall_fscore_support,
                             roc_auc_score)


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    two_classes = len(np.unique(y_true)) == 2
    return {
        "precision": float(p), "recall": float(r), "f1": float(f),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "pr_auc": float(average_precision_score(y_true, y_prob)) if two_classes else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if two_classes else float("nan"),
    }


def best_threshold(y_true, y_prob) -> float:
    """Threshold maximising illicit-class F1 (choose it on VALIDATION data only)."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return 0.5
    p, r, t = precision_recall_curve(y_true, y_prob)
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
    return float(t[int(np.argmax(f1))]) if len(t) else 0.5


def per_timestep_f1(y, prob, time, mask, threshold: float = 0.5) -> dict:
    """Illicit-class F1 for each test time step (reveals concept drift, e.g. the dark-market shutdown ~step 43)."""
    out = {}
    for t in np.unique(time[mask]):
        m = mask & (time == t)
        if (y[m] == 1).sum() == 0:
            out[int(t)] = float("nan")
            continue
        out[int(t)] = float(f1_score(y[m], (prob[m] >= threshold).astype(int), zero_division=0))
    return out


def evaluate_predictions(prob, data) -> dict:
    """prob: illicit probability for ALL nodes (numpy). Threshold is tuned on validation."""
    y = data.y.cpu().numpy()
    time = data.time.cpu().numpy()
    va = data.val_mask.cpu().numpy()
    te = data.test_mask.cpu().numpy()
    thr = best_threshold(y[va], prob[va])
    return {
        "threshold": thr,
        "val": compute_metrics(y[va], prob[va], thr),
        "test": compute_metrics(y[te], prob[te], thr),
        "test_at_0.5": compute_metrics(y[te], prob[te], 0.5),
        "test_per_timestep": per_timestep_f1(y, prob, time, te, thr),
    }

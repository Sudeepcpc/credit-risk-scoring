import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             precision_score, recall_score, f1_score, confusion_matrix)

from .config import COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE


def ks_statistic(y, p) -> float:
    """Kolmogorov-Smirnov: max gap between cumulative distributions of goods and bads (banking standard)."""
    order = np.argsort(p)
    y = np.asarray(y)[order]
    cum_bad = np.cumsum(y) / y.sum()
    cum_good = np.cumsum(1 - y) / (1 - y).sum()
    return float(np.max(np.abs(cum_bad - cum_good)))


def expected_cost(y, p, threshold) -> float:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return (fn * COST_FALSE_NEGATIVE + fp * COST_FALSE_POSITIVE) / len(y)


def best_threshold(y, p) -> float:
    grid = np.linspace(0.05, 0.95, 181)
    costs = [expected_cost(y, p, t) for t in grid]
    return float(grid[int(np.argmin(costs))])


def report(y, p, threshold) -> dict:
    pred = (p >= threshold).astype(int)
    auc = roc_auc_score(y, p)
    return {
        "roc_auc": round(auc, 4),
        "gini": round(2 * auc - 1, 4),
        "ks": round(ks_statistic(y, p), 4),
        "pr_auc": round(average_precision_score(y, p), 4),
        "brier": round(brier_score_loss(y, p), 4),
        "threshold": round(threshold, 3),
        "precision": round(precision_score(y, pred), 4),
        "recall": round(recall_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
        "expected_cost_per_customer": round(expected_cost(y, p, threshold), 4),
    }

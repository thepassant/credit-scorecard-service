"""Model evaluation: ranking, separation, calibration, cutoff choice and stability."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def ks_statistic(y_true: np.ndarray, pd: np.ndarray) -> float:
    """Kolmogorov-Smirnov: max distance between the score CDFs of goods and bads."""
    fpr, tpr, _ = roc_curve(y_true, pd)
    return float(np.max(tpr - fpr))


def classification_metrics(y_true: np.ndarray, pd: np.ndarray) -> dict[str, float]:
    auc = float(roc_auc_score(y_true, pd))
    return {
        "auc": round(auc, 4),
        "gini": round(2 * auc - 1, 4),
        "ks": round(ks_statistic(y_true, pd), 4),
        "mean_pd": round(float(np.mean(pd)), 4),
        "default_rate": round(float(np.mean(y_true)), 4),
    }


def calibration_table(y_true: np.ndarray, pd: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Predicted vs observed default rate by PD decile."""
    edges = np.unique(np.quantile(pd, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, pd, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        mask = idx == b
        if mask.any():
            rows.append(
                {
                    "bin": b,
                    "count": int(mask.sum()),
                    "predicted_pd": round(float(pd[mask].mean()), 4),
                    "observed_rate": round(float(y_true[mask].mean()), 4),
                }
            )
    return rows


def cost_optimal_threshold(
    y_true: np.ndarray, pd: np.ndarray, cost_approve_bad: float, cost_decline_good: float
) -> dict[str, float]:
    """PD threshold (approve when pd < threshold) that minimises total decision cost."""
    order = np.argsort(pd)
    pd_sorted, y_sorted = pd[order], y_true[order]
    # Approving the k lowest-PD applicants: bads approved = cumsum(y), goods declined = rest.
    bads_approved = np.concatenate([[0], np.cumsum(y_sorted)])
    goods_total = len(y_sorted) - y_sorted.sum()
    goods_approved = np.concatenate([[0], np.cumsum(1 - y_sorted)])
    cost = cost_approve_bad * bads_approved + cost_decline_good * (goods_total - goods_approved)
    k = int(np.argmin(cost))
    threshold = float(pd_sorted[k]) if k < len(pd_sorted) else 1.0
    return {
        "pd_threshold": round(threshold, 4),
        "approval_rate": round(k / len(pd_sorted), 4),
        "bad_rate_approved": round(float(bads_approved[k] / k) if k else 0.0, 4),
        "cost_per_applicant": round(float(cost[k] / len(pd_sorted)), 4),
    }


def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10, eps: float = 1e-4) -> float:
    """Population stability index of `actual` against the `expected` distribution.

    Rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 significant shift.
    """
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))

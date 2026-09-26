"""Estimation and ranking metrics."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy import stats


def mae(pred: Sequence[float], true: Sequence[float]) -> float:
    """Mean absolute error in percentage points (inputs in [0, 1])."""
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(true))) * 100)


def mean_ci(values: Sequence[float], level: float = 0.95) -> tuple[float, float]:
    """Mean and half-width of the t-based confidence interval over runs."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return float(v.mean()), 0.0
    return float(v.mean()), float(stats.t.ppf(0.5 + level / 2, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


def pairwise_accuracy(pred: Sequence[float], true: Sequence[float]) -> float:
    """Fraction of model pairs ordered correctly."""
    p, t = np.asarray(pred), np.asarray(true)
    i, j = np.triu_indices(len(p), k=1)
    keep = t[i] != t[j]
    if not keep.any():
        return 1.0
    return float(np.mean(np.sign(p[i] - p[j])[keep] == np.sign(t[i] - t[j])[keep]))


def kendall_tau(pred: Sequence[float], true: Sequence[float]) -> float:
    tau = stats.kendalltau(pred, true).statistic
    return 0.0 if np.isnan(tau) else float(tau)


def topk_accuracy(pred: Sequence[float], true: Sequence[float], k: int) -> float:
    """Overlap between the predicted and true top-k models."""
    k = min(k, len(pred))
    top_pred = set(np.argsort(-np.asarray(pred), kind="stable")[:k])
    top_true = set(np.argsort(-np.asarray(true), kind="stable")[:k])
    return len(top_pred & top_true) / k


def ranking(pred: Sequence[float], true: Sequence[float]) -> Dict[str, float]:
    return {
        "correct_pairs": pairwise_accuracy(pred, true),
        "kendall_tau": kendall_tau(pred, true),
        "top1": topk_accuracy(pred, true, 1),
        "top5": topk_accuracy(pred, true, 5),
    }

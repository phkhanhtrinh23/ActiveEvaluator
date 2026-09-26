"""Label-free baselines: ATC, DoC, GDE, ALine-D and Majority.

Each estimator returns accuracy estimates for a pool of models on one target,
using labeled calibration data from the source distribution and unlabeled
target predictions/confidences.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence

import numpy as np
from scipy.stats import norm

from .records import ModelRecord


def _probit(p: np.ndarray) -> np.ndarray:
    return norm.ppf(np.clip(p, 1e-4, 1 - 1e-4))


def _agreement(preds: Sequence[np.ndarray]) -> np.ndarray:
    p = np.stack(preds)
    return (p[:, None, :] == p[None, :, :]).mean(-1)


def atc(pool: Sequence[ModelRecord], target: str) -> np.ndarray:
    out = []
    for r in pool:
        conf, correct = r.calib_conf[target], r.calib_correct[target]
        t = np.quantile(conf, 1 - correct.mean())
        out.append(np.mean(r.target_conf[target] > t))
    return np.array(out)


def doc(pool: Sequence[ModelRecord], target: str) -> np.ndarray:
    return np.array([
        r.calib_correct[target].mean() - (r.calib_conf[target].mean() - r.target_conf[target].mean()) for r in pool
    ]).clip(0, 1)


def gde(pool: Sequence[ModelRecord], target: str) -> np.ndarray:
    """Agreement with an independent copy of the model, or with its closest pool model."""
    calib = _agreement([r.calib_pred[target] for r in pool])
    np.fill_diagonal(calib, -1)
    out = []
    for i, r in enumerate(pool):
        twin = r.twin_pred.get(target)
        if twin is None:
            twin = pool[int(np.argmax(calib[i]))].target_pred[target]
        out.append(np.mean(r.target_pred[target] == twin))
    return np.array(out)


def aline_d(pool: Sequence[ModelRecord], target: str) -> np.ndarray:
    n = len(pool)
    acc_s = _probit(np.array([r.calib_correct[target].mean() for r in pool]))
    agr_s = _probit(_agreement([r.calib_pred[target] for r in pool]))
    agr_t = _probit(_agreement([r.target_pred[target] for r in pool]))
    i, j = np.triu_indices(n, k=1)
    xs, ys = agr_s[i, j], agr_t[i, j]
    slope = float(np.cov(xs, ys, bias=True)[0, 1] / xs.var()) if xs.var() > 1e-12 else 1.0
    A = np.zeros((len(i), n))
    A[np.arange(len(i)), i] = 0.5
    A[np.arange(len(i)), j] = 0.5
    b = agr_t[i, j] + slope * (0.5 * (acc_s[i] + acc_s[j]) - agr_s[i, j])
    z = np.linalg.lstsq(A, b, rcond=None)[0]
    return norm.cdf(z)


def majority(pool: Sequence[ModelRecord], target: str) -> np.ndarray:
    preds = np.stack([r.target_pred[target] for r in pool])
    vote = []
    for col in preds.T:
        values, counts = np.unique(col, return_counts=True)
        vote.append(values[np.argmax(counts)])
    return (preds == np.array(vote)[None]).mean(1)


BASELINES: Dict[str, Callable[[List[ModelRecord], str], np.ndarray]] = {
    "ATC": atc,
    "DoC": doc,
    "GDE": gde,
    "ALine-D": aline_d,
    "Majority": majority,
}

"""Reference implementations of the supervision-acquisition baselines and
label-free estimators compared against ActiveEval in the paper.

Every acquisition strategy has the same signature::

    select(X, pair_model, pair_sample, target_mask, budget, *, rng, **kw) -> List[int]

where

* ``X``            -- ``(P, d)`` float array of pair shift-descriptors (one row
                     per (reference-model, sample-set) evaluation *action*),
* ``pair_model``   -- ``(P,)`` int array, the reference-model id of each pair,
* ``pair_sample``  -- ``(P,)`` int array, the sample-set id of each pair,
* ``target_mask``  -- ``(P,)`` bool array, pairs whose sample-set is aligned with
                     the unlabeled deployment workload (target-aware coverage),
* ``budget``       -- number of pairs to acquire,

and returns the indices of the acquired pairs. The selected entries are then
labelled and a meta-evaluator is trained on them; see
``experiments/run_acquisition_benchmark.py``.

These are intentionally compact, self-contained NumPy implementations so the
benchmark runs on CPU with no extra dependencies. The production Text2SQL
pipeline uses the same math via :mod:`active_evaluator.active_selection`
(lazy-greedy facility location, GRAD-MATCH OMP, knapsack budget).
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# kernels / helpers
# ---------------------------------------------------------------------------

def _sqdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distance, shape (len(a), len(b))."""
    aa = (a * a).sum(1)[:, None]
    bb = (b * b).sum(1)[None, :]
    return np.clip(aa + bb - 2.0 * a @ b.T, 0.0, None)


def _median_bandwidth(X: np.ndarray, rng: np.random.Generator, max_pairs: int = 2000) -> float:
    n = len(X)
    if n < 2:
        return 1.0
    i = rng.integers(0, n, size=min(max_pairs, n * n))
    j = rng.integers(0, n, size=min(max_pairs, n * n))
    m = i != j
    d = ((X[i[m]] - X[j[m]]) ** 2).sum(1)
    tau = float(np.median(d)) if d.size else 1.0
    return tau if tau > 0 else 1.0


def _rbf(A: np.ndarray, B: np.ndarray, tau: float) -> np.ndarray:
    return np.exp(-_sqdist(A, B) / tau)


# ---------------------------------------------------------------------------
# acquisition strategies
# ---------------------------------------------------------------------------

def select_random(X, pair_model, pair_sample, target_mask, budget, *, rng, **kw) -> List[int]:
    """Uniform random acquisition (naive lower bound)."""
    return list(rng.choice(len(X), size=min(budget, len(X)), replace=False))


def select_kcenter(X, pair_model, pair_sample, target_mask, budget, *, rng, **kw) -> List[int]:
    """Greedy k-center (diversity-only): maximise the min distance to the set."""
    n = len(X)
    budget = min(budget, n)
    start = int(rng.integers(0, n))
    chosen = [start]
    dmin = _sqdist(X, X[[start]]).ravel()
    while len(chosen) < budget:
        nxt = int(np.argmax(dmin))
        chosen.append(nxt)
        dmin = np.minimum(dmin, _sqdist(X, X[[nxt]]).ravel())
    return chosen


def select_facility(X, pair_model, pair_sample, target_mask, budget, *, rng,
                    influence: np.ndarray | None = None, **kw) -> List[int]:
    """Target-aware (optionally influence-weighted) facility location.

    Maximises ``sum_{q in target} w(q) * max_{s in S} k(q, s)`` by greedy. With
    ``influence=None`` this is plain facility location; passing the gradient-norm
    influence weights recovers ActiveEval's sample-coverage term.
    """
    n = len(X)
    budget = min(budget, n)
    tau = _median_bandwidth(X, rng)
    q_idx = np.where(target_mask)[0]
    if q_idx.size == 0:
        q_idx = np.arange(n)
    K = _rbf(X[q_idx], X, tau)            # (|Q|, P): sim of each target point to each candidate
    w = np.ones(q_idx.size) if influence is None else influence[q_idx]
    coverage = np.zeros(q_idx.size)
    chosen: List[int] = []
    avail = np.ones(n, bool)
    for _ in range(budget):
        gain = (w[:, None] * np.clip(K - coverage[:, None], 0, None)).sum(0)
        gain[~avail] = -np.inf
        pick = int(np.argmax(gain))
        if not np.isfinite(gain[pick]) or gain[pick] <= 0:
            break
        chosen.append(pick)
        avail[pick] = False
        coverage = np.maximum(coverage, K[:, pick])
    # top up with random if the surrogate saturated
    if len(chosen) < budget:
        rest = [i for i in np.where(avail)[0]]
        rng.shuffle(rest)
        chosen.extend(rest[: budget - len(chosen)])
    return chosen


def select_logdet(X, pair_model, pair_sample, target_mask, budget, *, rng, sigma=1.0, **kw) -> List[int]:
    """Log-determinant / DPP-style diversity (Submodular benchmark selection,
    Smola 2026; Kulesza & Taskar 2012). Greedy max of log det(I + K_S/sigma^2)."""
    n = len(X)
    budget = min(budget, n)
    tau = _median_bandwidth(X, rng)
    K = _rbf(X, X, tau) / (sigma ** 2)
    diag = np.diag(K).copy()
    chosen: List[int] = []
    # greedy: pick the point with the largest current marginal log-gain
    L = np.zeros((0, 0))
    cov = np.zeros((n, 0))
    avail = np.ones(n, bool)
    cur_diag = 1.0 + diag
    for _ in range(budget):
        gains = cur_diag.copy()
        gains[~avail] = -np.inf
        pick = int(np.argmax(gains))
        if not np.isfinite(gains[pick]) or gains[pick] <= 0:
            break
        chosen.append(pick)
        avail[pick] = False
        # conditional variance update (incremental Cholesky)
        k = K[chosen[:-1], pick] if cov.shape[1] else np.zeros(0)
        if cov.shape[1]:
            u = np.linalg.solve(L, k)
            d = np.sqrt(max(cur_diag[pick], 1e-9))
            newcol = (K[:, pick] - cov @ u) / d
            L = np.block([[L, np.zeros((L.shape[0], 1))], [u[None, :], np.array([[d]])]])
            cov = np.concatenate([cov, newcol[:, None]], 1)
            cur_diag = np.clip(cur_diag - newcol ** 2, 1e-9, None)
        else:
            d = np.sqrt(max(cur_diag[pick], 1e-9))
            L = np.array([[d]])
            cov = (K[:, pick] / d)[:, None]
            cur_diag = np.clip(cur_diag - cov[:, 0] ** 2, 1e-9, None)
    if len(chosen) < budget:
        rest = list(np.where(avail)[0]); rng.shuffle(rest)
        chosen.extend(rest[: budget - len(chosen)])
    return chosen


def select_matrix_completion(X, pair_model, pair_sample, target_mask, budget, *, rng, rank=8, **kw) -> List[int]:
    """Leverage-score sampling for low-rank matrix completion (Candès & Recht
    2009): probe the entries with the highest statistical leverage."""
    n = len(X)
    budget = min(budget, n)
    # rank-r SVD of the pair-descriptor matrix; row leverage = ||U_r[i]||^2
    Xc = X - X.mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    r = min(rank, U.shape[1])
    lev = (U[:, :r] ** 2).sum(1)
    p = lev / lev.sum()
    return list(rng.choice(n, size=budget, replace=False, p=p))


def select_active_testing(X, pair_model, pair_sample, target_mask, budget, *, rng, **kw) -> List[int]:
    """Active-testing-style acquisition (Kossen et al. 2021): a cheap ridge
    surrogate is fit on a small random seed, then pairs with the highest
    predictive variance (disagreement proxy) are acquired."""
    n = len(X)
    budget = min(budget, n)
    seed_n = max(4, n // 20)
    seed = rng.choice(n, size=seed_n, replace=False)
    # bagged ridge fits -> per-pair predictive std as the acquisition score
    preds = []
    for _ in range(8):
        bs = rng.choice(seed, size=seed_n, replace=True)
        A = X[bs]
        w = np.linalg.lstsq(A.T @ A + 1e-2 * np.eye(X.shape[1]), A.T @ (A[:, 0] * 0 + rng.normal(size=seed_n)), rcond=None)[0]
        preds.append(X @ w)
    var = np.var(np.stack(preds), 0)
    order = np.argsort(-var)
    return list(order[:budget])


def select_bayesian_design(X, pair_model, pair_sample, target_mask, budget, *, rng, **kw) -> List[int]:
    """Bayesian / D-optimal experimental design: greedily maximise
    log det(X_S^T X_S + lambda I) (information of the design matrix)."""
    n, d = X.shape
    budget = min(budget, n)
    lam = 1e-2
    A_inv = np.eye(d) / lam
    chosen: List[int] = []
    avail = np.ones(n, bool)
    for _ in range(budget):
        # D-optimal greedy gain = log(1 + x^T A_inv x)
        proj = (X @ A_inv) * X
        score = np.log1p(proj.sum(1))
        score[~avail] = -np.inf
        pick = int(np.argmax(score))
        chosen.append(pick)
        avail[pick] = False
        x = X[pick][:, None]
        Ax = A_inv @ x
        A_inv = A_inv - (Ax @ Ax.T) / (1.0 + float(x.T @ Ax))
    return chosen


def select_gradmatch(X, pair_model, pair_sample, target_mask, budget, *, rng, lam=1e-2, **kw) -> List[int]:
    """GRAD-MATCH OMP (Killamsetty et al. 2021): pick a weighted subset whose
    summed feature approximates the target-region mean feature. This mirrors the
    orthogonal-matching-pursuit selection in
    :func:`active_evaluator.active_selection.gradient_match_omp`."""
    n = len(X)
    budget = min(budget, n)
    q = np.where(target_mask)[0]
    target = X[q].mean(0) if q.size else X.mean(0)
    residual = target.copy()
    chosen: List[int] = []
    avail = np.ones(n, bool)
    for _ in range(budget):
        scores = np.abs(X @ residual)
        scores[~avail] = -np.inf
        pick = int(np.argmax(scores))
        chosen.append(pick)
        avail[pick] = False
        G = X[chosen]
        gram = G @ G.T + lam * np.eye(len(chosen))
        w = np.linalg.solve(gram, G @ target)
        residual = target - (w[:, None] * G).sum(0)
    return chosen


# ---------------------------------------------------------------------------
# ActiveEval (ours): combine target coverage + diversity + the model axis
# ---------------------------------------------------------------------------

def select_activeeval_sample(X, pair_model, pair_sample, target_mask, budget, *, rng,
                             influence: np.ndarray | None = None, **kw) -> List[int]:
    """ActiveEval-S: target-aware, influence-weighted facility location over
    sample-sets only (the model axis is left uniform)."""
    return select_facility(X, pair_model, pair_sample, target_mask, budget,
                           rng=rng, influence=influence)


def select_activeeval_sm(X, pair_model, pair_sample, target_mask, budget, *, rng,
                         influence: np.ndarray | None = None, **kw) -> List[int]:
    """ActiveEval-S+M: two-stage. First pick behaviorally-diverse reference
    models (k-center in model-mean-descriptor space), then target-aware sample
    sets within those models."""
    models = np.unique(pair_model)
    # behavioral coverage of models: mean pair descriptor per model
    centroids = np.stack([X[pair_model == m].mean(0) for m in models])
    n_models = max(1, int(round(0.5 * len(models))))  # keep half the models
    keep_local = select_kcenter(centroids, np.arange(len(models)), np.arange(len(models)),
                                np.ones(len(models), bool), n_models, rng=rng)
    keep_models = set(models[keep_local].tolist())
    mask = np.array([m in keep_models for m in pair_model])
    sub = np.where(mask)[0]
    infl = None if influence is None else influence[sub]
    local = select_facility(X[sub], pair_model[sub], pair_sample[sub], target_mask[sub],
                            budget, rng=rng, influence=infl)
    return list(sub[local])


def select_activeeval_pair(X, pair_model, pair_sample, target_mask, budget, *, rng,
                           influence: np.ndarray | None = None, alpha=0.8, **kw) -> List[int]:
    """ActiveEval-Pair: direct pair selection blending target-coverage facility
    location with a log-determinant diversity term over the *target-aligned* pairs
    (the strongest variant). The diversity term spreads picks within the target
    region rather than wandering into noisy off-target pairs."""
    n = len(X)
    budget = min(budget, n)
    b_cov = int(round(alpha * budget))
    cov = select_facility(X, pair_model, pair_sample, target_mask, b_cov,
                          rng=rng, influence=influence)
    remaining = budget - len(cov)
    if remaining > 0:
        chosen_set = set(cov)
        # diversify only within the target region (stay on clean, relevant pairs)
        cand = np.array([i for i in np.where(target_mask)[0] if i not in chosen_set])
        if cand.size == 0:
            cand = np.array([i for i in range(n) if i not in chosen_set])
        div_local = select_logdet(X[cand], pair_model[cand], pair_sample[cand],
                                  target_mask[cand], remaining, rng=rng)
        cov = cov + list(cand[div_local])
    return cov


# ---------------------------------------------------------------------------
# label-free estimators (no acquisition; per-model point estimates)
# ---------------------------------------------------------------------------

def estimate_atc(conf_source: np.ndarray, conf_target: np.ndarray, acc_source: float) -> float:
    """Average Thresholded Confidence (Garg et al. 2022): calibrate a threshold
    on the source so the source-acc matches, then count target points above it."""
    t = np.quantile(conf_source, max(0.0, 1.0 - acc_source))
    return float((conf_target >= t).mean())


def estimate_doc(conf_source: np.ndarray, conf_target: np.ndarray, acc_source: float) -> float:
    """Difference of Confidences (Guillory et al. 2021): drop source accuracy by
    the mean-confidence gap between source and target."""
    return float(np.clip(acc_source - (conf_source.mean() - conf_target.mean()), 0.0, 1.0))


ACQUISITION_REGISTRY = {
    "random": select_random,
    "kcenter": select_kcenter,
    "facility_location": select_facility,
    "matrix_completion": select_matrix_completion,
    "active_testing": select_active_testing,
    "bayesian_design": select_bayesian_design,
    "submodular_benchmark": select_logdet,
    "gradmatch": select_gradmatch,
    "activeeval_s": select_activeeval_sample,
    "activeeval_sm": select_activeeval_sm,
    "activeeval_pair": select_activeeval_pair,
}

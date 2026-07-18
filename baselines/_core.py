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
                    influence: np.ndarray | None = None,
                    target_ref: np.ndarray | None = None,
                    gain_threshold: float | None = None, **kw) -> List[int]:
    """Target-aware (optionally influence-weighted) facility location.

    Maximises ``sum_{q in target} w(q) * max_{s in S} k(q, s)`` by greedy, i.e.
    it picks *source* candidates ``S`` that best cover the target region ``Q``.

    The target region ``Q`` is supplied one of two ways:

    * ``target_ref`` -- an explicit ``(T, d)`` array of *unlabeled* target-region
      descriptors. Candidates ``X`` are the (disjoint) labelable source pool, so
      the target slices are never selected/labeled themselves. ``influence`` then
      weights the *candidates* (high-leverage source pairs are preferred).
    * ``target_mask`` -- legacy path: the target rows live inside ``X`` and both
      anchor the coverage objective and are themselves selectable. ``influence``
      weights the target anchors. Used when ``target_ref is None``.

    ``gain_threshold``, if set, makes this an *adaptive-budget* selector: the
    greedy loop stops as soon as the marginal coverage gain of the best
    remaining candidate drops below ``gain_threshold`` of the first pick's
    gain, and the result is **not** padded back up to ``budget`` -- fewer than
    ``budget`` pairs may be returned. ``None`` (default) preserves the original
    fixed-budget behaviour (pad with random candidates if the surrogate
    saturates before reaching ``budget``).
    """
    n = len(X)
    budget = min(budget, n)
    tau = _median_bandwidth(X, rng)
    if target_ref is not None and len(target_ref) > 0:
        Q = np.asarray(target_ref, X.dtype)          # unlabeled target-region anchors
        w = np.ones(len(Q))
        cand_w = np.ones(n) if influence is None else influence   # leverage on candidates
    else:
        q_idx = np.where(target_mask)[0]
        if q_idx.size == 0:
            q_idx = np.arange(n)
        Q = X[q_idx]
        w = np.ones(q_idx.size) if influence is None else influence[q_idx]
        cand_w = np.ones(n)
    K = _rbf(Q, X, tau)                   # (|Q|, P): sim of each target anchor to each candidate
    coverage = np.zeros(len(Q))
    chosen: List[int] = []
    avail = np.ones(n, bool)
    first_gain = None
    for _ in range(budget):
        gain = (w[:, None] * np.clip(K - coverage[:, None], 0, None)).sum(0) * cand_w
        gain[~avail] = -np.inf
        pick = int(np.argmax(gain))
        if not np.isfinite(gain[pick]) or gain[pick] <= 0:
            break
        if first_gain is None:
            first_gain = gain[pick]
        elif gain_threshold is not None and gain[pick] < gain_threshold * first_gain:
            break
        chosen.append(pick)
        avail[pick] = False
        coverage = np.maximum(coverage, K[:, pick])
    # top up with random if the surrogate saturated -- but not when the caller
    # deliberately asked to stop early on a low marginal gain (adaptive budget)
    if gain_threshold is None and len(chosen) < budget:
        rest = [i for i in np.where(avail)[0]]
        rng.shuffle(rest)
        chosen.extend(rest[: budget - len(chosen)])
    return chosen


def select_logdet(X, pair_model, pair_sample, target_mask, budget, *, rng, sigma=1.0,
                  gain_threshold: float | None = None, **kw) -> List[int]:
    """Log-determinant / DPP-style diversity (Submodular benchmark selection,
    Smola 2026; Kulesza & Taskar 2012). Greedy max of log det(I + K_S/sigma^2).

    ``gain_threshold`` (see :func:`select_facility`): stop early once the best
    remaining candidate's conditional-variance gain drops below
    ``gain_threshold`` of the first pick's gain, without padding back to
    ``budget``.
    """
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
    first_gain = None
    for _ in range(budget):
        gains = cur_diag.copy()
        gains[~avail] = -np.inf
        pick = int(np.argmax(gains))
        if not np.isfinite(gains[pick]) or gains[pick] <= 0:
            break
        if first_gain is None:
            first_gain = gains[pick]
        elif gain_threshold is not None and gains[pick] < gain_threshold * first_gain:
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
    if gain_threshold is None and len(chosen) < budget:
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
        A_inv = A_inv - (Ax @ Ax.T) / (1.0 + float((x.T @ Ax).item()))
    return chosen


def select_gradmatch(X, pair_model, pair_sample, target_mask, budget, *, rng, lam=1e-2, **kw) -> List[int]:
    """GRAD-MATCH OMP (Killamsetty et al. 2021): pick a weighted subset whose
    summed feature approximates the target-region mean feature. This mirrors the
    orthogonal-matching-pursuit selection in
    :func:`active_evaluator.active_selection.gradient_match_omp`."""
    n = len(X)
    budget = min(budget, n)
    target_ref = kw.get("target_ref")
    if target_ref is not None and len(target_ref) > 0:
        target = np.asarray(target_ref, X.dtype).mean(0)   # unlabeled target-region mean
    else:
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
# Greedy entropy / mutual information (Alg. 1 / Alg. 2 of the benchmark-
# selection paper; see benchmark-selection/code/greedy_select.py). There the
# algorithms greedily pivot an N x N benchmark correlation matrix; here the
# analogous PSD matrix is an RBF kernel over the candidate pair descriptors,
# restricted to the target-aligned pool (same q_idx pattern as select_facility).
# ---------------------------------------------------------------------------

def _target_universe(n: int, target_mask, rng: np.random.Generator, cap: int | None = 200) -> np.ndarray:
    """Target-aligned candidate indices, sub-sampled to `cap` for tractability.

    greedy_mi's per-step complement refactorization is cubic in the universe
    size, so leaving it uncapped against the full pair pool (often 1000s of
    pairs) would make this baseline far slower than every other method.
    Pass ``cap=None`` to use the full target-aligned pool (e.g. for a
    dedicated entropy-vs-MI comparison where budgets stay within the pool).
    """
    q_idx = np.where(target_mask)[0]
    if q_idx.size == 0:
        q_idx = np.arange(n)
    if cap is not None and q_idx.size > cap:
        q_idx = np.sort(rng.choice(q_idx, size=cap, replace=False))
    return q_idx


def _pivoted_cholesky_entropy(Sigma: np.ndarray, k: int,
                              gain_threshold: float | None = None) -> List[int]:
    """Algorithm 1 (greedy entropy): pivot on argmax conditional variance,
    with a rank-1 Cholesky update of the residual diagonal after each pick.

    ``gain_threshold``: stop early (returning fewer than ``k`` indices) once
    the pivot's conditional variance ``d[j_star]`` -- the point's remaining
    "surprise" given what's already selected -- drops below ``gain_threshold``
    of the first pivot's variance.
    """
    N = Sigma.shape[0]
    k = min(k, N)
    d = np.diag(Sigma).copy().astype(np.float64)
    L = np.zeros((N, k), dtype=np.float64)
    selected: List[int] = []
    selected_set = set()
    first_gain = None
    for t in range(k):
        d_masked = d.copy()
        if selected:
            d_masked[list(selected_set)] = -np.inf
        j_star = int(np.argmax(d_masked))
        if first_gain is None:
            first_gain = d[j_star]
        elif gain_threshold is not None and d[j_star] < gain_threshold * first_gain:
            break
        selected.append(j_star)
        selected_set.add(j_star)
        sqrt_d = np.sqrt(max(d[j_star], 1e-300))
        for j in range(N):
            if j in selected_set and j != j_star:
                continue
            L[j, t] = (Sigma[j, j_star] - L[j, :t] @ L[j_star, :t]) / sqrt_d
            if j != j_star:
                d[j] -= L[j, t] ** 2
    return selected


def _complement_precision_diag(Sigma: np.ndarray, comp: np.ndarray) -> np.ndarray:
    """Diagonal of (Sigma[comp, comp])^{-1}, via Cholesky (eigh fallback)."""
    m = len(comp)
    Sigma_sub = Sigma[np.ix_(comp, comp)]
    try:
        L = np.linalg.cholesky(Sigma_sub)
        L_inv = np.linalg.solve(L, np.eye(m))
        return np.sum(L_inv ** 2, axis=0)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(Sigma_sub)
        eigvals = np.maximum(eigvals, 1e-10)
        return np.sum(eigvecs ** 2 / eigvals[None, :], axis=1)


def _pivoted_cholesky_mi(Sigma: np.ndarray, k: int,
                         gain_threshold: float | None = None) -> List[int]:
    """Algorithm 2 (greedy mutual information): pivot on
    argmax_v [log sigma^2_{v|S} + log P_vv], where P_vv is the v-th diagonal
    entry of the precision matrix of the currently-unselected complement.

    ``gain_threshold``: stop early once the pivot's conditional variance
    ``d[j_star]`` (the same "remaining surprise" quantity thresholded in
    :func:`_pivoted_cholesky_entropy`, kept consistent so entropy and MI are
    compared under the same stopping rule) drops below ``gain_threshold`` of
    the first pivot's variance.
    """
    N = Sigma.shape[0]
    k = min(k, N)
    d = np.diag(Sigma).copy().astype(np.float64)
    L = np.zeros((N, k), dtype=np.float64)
    selected: List[int] = []
    selected_set = set()
    first_gain = None
    for t in range(k):
        comp = np.array([j for j in range(N) if j not in selected_set])
        P_diag_comp = _complement_precision_diag(Sigma, comp)
        P_full = np.zeros(N)
        P_full[comp] = P_diag_comp
        scores = np.full(N, -np.inf)
        valid = (d > 1e-300) & (P_full > 1e-300)
        valid[list(selected_set)] = False
        scores[valid] = np.log(d[valid]) + np.log(P_full[valid])
        j_star = int(np.argmax(scores))
        if first_gain is None:
            first_gain = d[j_star]
        elif gain_threshold is not None and d[j_star] < gain_threshold * first_gain:
            break
        selected.append(j_star)
        selected_set.add(j_star)
        sqrt_d = np.sqrt(max(d[j_star], 1e-300))
        for j in range(N):
            if j in selected_set and j != j_star:
                continue
            L[j, t] = (Sigma[j, j_star] - L[j, :t] @ L[j_star, :t]) / sqrt_d
            if j != j_star:
                d[j] -= L[j, t] ** 2
    return selected


def select_greedy_entropy(X, pair_model, pair_sample, target_mask, budget, *, rng,
                          cap: int | None = 200, sigma: float | None = 1.0,
                          gain_threshold: float | None = None, **kw) -> List[int]:
    """Greedy entropy (Alg. 1): pivoted-Cholesky greedy maximization of
    log det(Sigma_S) over an RBF kernel restricted to the target-aligned pool.

    ``sigma`` defaults to 1.0, matching ``select_logdet``'s ridge-regularized
    ``log det(I + K/sigma^2)`` objective -- the two are the same pivoted-Cholesky
    greedy algorithm and should use the same regularization so they are directly
    comparable. Pass ``sigma=None`` for the original near-zero jitter (``+1e-6``,
    i.e. unregularized greedy entropy).

    ``gain_threshold`` (see :func:`select_facility`): stop early -- returning
    fewer than ``budget`` pairs, unpadded -- once the pivot's conditional
    variance drops below this fraction of the first pivot's.
    """
    n = len(X)
    budget = min(budget, n)
    q_idx = _target_universe(n, target_mask, rng, cap=cap)
    tau = _median_bandwidth(X[q_idx], rng)
    K = _rbf(X[q_idx], X[q_idx], tau)
    if sigma is None:
        Sigma = K + 1e-6 * np.eye(len(q_idx))
    else:
        Sigma = K / (sigma ** 2) + np.eye(len(q_idx))
    local = _pivoted_cholesky_entropy(Sigma, budget, gain_threshold=gain_threshold)
    chosen = list(q_idx[local])
    if gain_threshold is None and len(chosen) < budget:
        avail = np.ones(n, bool)
        avail[chosen] = False
        rest = list(np.where(avail)[0])
        rng.shuffle(rest)
        chosen.extend(rest[: budget - len(chosen)])
    return chosen


def select_greedy_mi(X, pair_model, pair_sample, target_mask, budget, *, rng,
                     cap: int | None = 200, sigma: float | None = 1.0,
                     gain_threshold: float | None = None, **kw) -> List[int]:
    """Greedy mutual information (Alg. 2): pivoted-Cholesky greedy
    maximization of I(X_S; X_{V\\S}) over an RBF kernel restricted to the
    target-aligned pool.

    ``sigma`` defaults to 1.0, matching ``select_logdet``'s ridge-regularized
    ``log det(I + K/sigma^2)`` objective, so entropy/MI/logdet are compared under
    the same regularization. Pass ``sigma=None`` for the original near-zero jitter
    (``+1e-6``, i.e. unregularized greedy MI).

    ``gain_threshold`` (see :func:`select_facility`): stop early -- returning
    fewer than ``budget`` pairs, unpadded -- once the pivot's conditional
    variance drops below this fraction of the first pivot's.
    """
    n = len(X)
    budget = min(budget, n)
    q_idx = _target_universe(n, target_mask, rng, cap=cap)
    tau = _median_bandwidth(X[q_idx], rng)
    K = _rbf(X[q_idx], X[q_idx], tau)
    if sigma is None:
        Sigma = K + 1e-6 * np.eye(len(q_idx))
    else:
        Sigma = K / (sigma ** 2) + np.eye(len(q_idx))
    local = _pivoted_cholesky_mi(Sigma, budget, gain_threshold=gain_threshold)
    chosen = list(q_idx[local])
    if gain_threshold is None and len(chosen) < budget:
        avail = np.ones(n, bool)
        avail[chosen] = False
        rest = list(np.where(avail)[0])
        rng.shuffle(rest)
        chosen.extend(rest[: budget - len(chosen)])
    return chosen


# ---------------------------------------------------------------------------
# ActiveEval (ours): combine target coverage + diversity + the model axis
# ---------------------------------------------------------------------------

def select_activeeval_sample(X, pair_model, pair_sample, target_mask, budget, *, rng,
                             influence: np.ndarray | None = None,
                             target_ref: np.ndarray | None = None,
                             gain_threshold: float | None = None, **kw) -> List[int]:
    """ActiveEval-S: target-aware, influence-weighted facility location over
    sample-sets only (the model axis is left uniform).

    ``gain_threshold`` (see :func:`select_facility`): forwarded as-is, so this
    may return fewer than ``budget`` pairs when set.
    """
    return select_facility(X, pair_model, pair_sample, target_mask, budget,
                           rng=rng, influence=influence, target_ref=target_ref,
                           gain_threshold=gain_threshold)


def select_activeeval_sm(X, pair_model, pair_sample, target_mask, budget, *, rng,
                         influence: np.ndarray | None = None,
                         target_ref: np.ndarray | None = None,
                         gain_threshold: float | None = None, **kw) -> List[int]:
    """ActiveEval-S+M: two-stage. First pick behaviorally-diverse reference
    models (k-center in model-mean-descriptor space), then target-aware sample
    sets within those models.

    ``gain_threshold`` (see :func:`select_facility`) is forwarded to the
    within-model facility-location stage only (the k-center model pick has no
    natural gain to threshold), so this may return fewer than ``budget`` pairs.
    """
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
                            budget, rng=rng, influence=infl, target_ref=target_ref,
                            gain_threshold=gain_threshold)
    return list(sub[local])


def select_activeeval_pair(X, pair_model, pair_sample, target_mask, budget, *, rng,
                           influence: np.ndarray | None = None,
                           target_ref: np.ndarray | None = None, alpha=0.8,
                           gain_threshold: float | None = None, **kw) -> List[int]:
    """ActiveEval-Pair: direct pair selection blending target-coverage facility
    location with a log-determinant diversity term (the strongest variant). The
    coverage term pulls picks toward the (unlabeled) target region; the diversity
    term then spreads the remaining budget across the source pool near those picks
    rather than clumping.

    ``gain_threshold`` (see :func:`select_facility`) is applied independently to
    both the coverage and diversity greedy loops, so the pair may under-spend its
    ``budget`` in either phase (whichever saturates first)."""
    n = len(X)
    budget = min(budget, n)
    b_cov = int(round(alpha * budget))
    cov = select_facility(X, pair_model, pair_sample, target_mask, b_cov,
                          rng=rng, influence=influence, target_ref=target_ref,
                          gain_threshold=gain_threshold)
    remaining = budget - len(cov)
    if remaining > 0:
        chosen_set = set(cov)
        # Diversify, but stay on clean, target-relevant pairs. If target pairs are
        # themselves selectable (legacy), diversify within them. Otherwise the
        # target region is unlabeled/held out, so restrict the diversity pool to
        # the near-target *source* pairs (top affinity to the target reference)
        # rather than the whole source pool -- else the log-det term wanders into
        # far, noisy pairs and undoes the coverage term's target focus.
        if target_ref is not None and len(target_ref) > 0 and not target_mask.any():
            tau = _median_bandwidth(X, rng)
            affinity = _rbf(np.asarray(target_ref, X.dtype), X, tau).max(0)  # per-candidate target affinity
            k = max(remaining, budget, int(0.35 * n))
            near = np.argsort(-affinity)[:k]
            cand = np.array([i for i in near if i not in chosen_set])
        else:
            cand = np.array([i for i in np.where(target_mask)[0] if i not in chosen_set])
        if cand.size == 0:
            cand = np.array([i for i in range(n) if i not in chosen_set])
        div_local = select_logdet(X[cand], pair_model[cand], pair_sample[cand],
                                  target_mask[cand], remaining, rng=rng,
                                  gain_threshold=gain_threshold)
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
    "greedy_entropy": select_greedy_entropy,
    "greedy_mi": select_greedy_mi,
    "activeeval_s": select_activeeval_sample,
    "activeeval_sm": select_activeeval_sm,
    "activeeval_pair": select_activeeval_pair,
}

"""Distance formulas between candidate sample-set point clouds, plus a
submodular facility-location coreset selector built on top of them.

This is an *alternative* to the k-means-based Stage-1 warm-start reduction
in docs/kmeans-submodular-warmstart.md. Instead of Lloyd's heuristic
clustering (no approximation guarantee; measured there to systematically
under-represent the target-relevant tail because it allocates resolution by
candidate *density*, not *representativeness*), this selects K
representative subsets via the same monotone-submodular facility-location
greedy proven in docs/submodularity-audit.md Part 1 -- a provable (1-1/e)
guarantee for *coverage of the whole candidate pool* (the "meta-dataset"),
not target-similarity. Target-awareness is deliberately absent here: the
objective is "best represent everything," matching the changed problem
setting.

Four distance formulas between two sample-set point clouds A (n x d), B
(m x d):

- kernel_mean: Maximum Mean Discrepancy (MMD^2) between the clouds' RBF
  kernel-mean embeddings -- "how different do these two clouds look to an
  RBF kernel, on average."
- sliced_wasserstein: reuses shift_descriptor.metrics.sliced_wasserstein_distance
  (the same metric the real Text2SQL pipeline computes from cached LLM
  embeddings, active_evaluator/descriptors.py).
- hausdorff: the two-sided Hausdorff distance -- the worst-covered point in
  either cloud; a robust/worst-case notion of "how far is the least-well-matched
  point," unlike the other two which are average-case.
- sum: all three, standardized (zero mean / unit std over the pool's
  pairwise distances) then summed, so no single metric's raw scale dominates.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from scipy.spatial.distance import directed_hausdorff

from shift_descriptor.metrics import sliced_wasserstein_distance as _swd


# ---------------------------------------------------------------------------
# distance formulas between two point clouds
# ---------------------------------------------------------------------------


def _median_bandwidth(X: np.ndarray, rng: np.random.Generator, max_pairs: int = 500) -> float:
    n = len(X)
    if n < 2:
        return 1.0
    i = rng.integers(0, n, size=min(max_pairs, n * n))
    j = rng.integers(0, n, size=min(max_pairs, n * n))
    m = i != j
    d = ((X[i[m]] - X[j[m]]) ** 2).sum(1)
    t = float(np.median(d)) if d.size else 1.0
    return t if t > 0 else 1.0


def _rbf_gram(A: np.ndarray, B: np.ndarray, tau: float) -> np.ndarray:
    sq = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return np.exp(-sq / tau)


def kernel_mean_distance(A: np.ndarray, B: np.ndarray, *, tau: float | None = None,
                         rng: np.random.Generator | None = None) -> float:
    """MMD (not squared) between the RBF kernel-mean embeddings of A, B --
    D_ij^KME = ||mu_i - mu_j||_H, matching
    docs/representative_meta_dataset_facility_location_full.md Sec 3.1.
    Callers that need S_ij = exp(-D^2/2tau^2) square this themselves
    (select_representative_subsets does), so this must return MMD, not MMD^2 --
    returning MMD^2 here would silently square it twice.
    """
    rng = rng or np.random.default_rng(0)
    if tau is None:
        tau = _median_bandwidth(np.concatenate([A, B], axis=0), rng)
    Kaa, Kbb, Kab = _rbf_gram(A, A, tau), _rbf_gram(B, B, tau), _rbf_gram(A, B, tau)
    mmd2 = max(Kaa.mean() + Kbb.mean() - 2.0 * Kab.mean(), 0.0)
    return float(np.sqrt(mmd2))


def sliced_wasserstein_cloud_distance(A: np.ndarray, B: np.ndarray, *,
                                      num_projections: int = 20, seed: int = 13) -> float:
    return _swd(A, B, num_projections=num_projections, seed=seed)["swd_mean"]


def hausdorff_cloud_distance(A: np.ndarray, B: np.ndarray) -> float:
    return float(max(directed_hausdorff(A, B)[0], directed_hausdorff(B, A)[0]))


DISTANCE_FNS: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "kernel_mean": kernel_mean_distance,
    "sliced_wasserstein": sliced_wasserstein_cloud_distance,
    "hausdorff": hausdorff_cloud_distance,
}


# ---------------------------------------------------------------------------
# pairwise distance matrices + submodular coreset selection
# ---------------------------------------------------------------------------


def pairwise_distance_matrix(clouds: Sequence[np.ndarray], distance_fn: Callable) -> np.ndarray:
    n = len(clouds)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = distance_fn(clouds[i], clouds[j])
            D[i, j] = D[j, i] = d
    return D


def all_distance_matrices(clouds: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
    """Compute all three base matrices once (each used standalone and inside 'sum')."""
    return {name: pairwise_distance_matrix(clouds, fn) for name, fn in DISTANCE_FNS.items()}


def combined_distance_matrix(base: Dict[str, np.ndarray],
                             weights: Dict[str, float] | None = None) -> np.ndarray:
    """Standardize each of the three matrices then take a weighted sum.

    D_sum(b_i,b_j) = sum_m w_m * z(D_m)_ij,  z(D_m) = (D_m - mean(D_m)) / std(D_m)

    weights=None gives the equal-weighted 'sum of all 3' (w_m=1 each, the
    original formula in Sec 3.4/27.2). Passing e.g.
    {"hausdorff": 2.0, "kernel_mean": 1.0, "sliced_wasserstein": 1.0} gives
    hausdorff double weight relative to the other two -- the
    'sum_hausdorff_weighted' formula in Sec 27.7, motivated by hausdorff
    being the standout performer at K=60 (Sec 27.6).
    """
    total = None
    for name, D in base.items():
        w = 1.0 if weights is None else weights.get(name, 1.0)
        iu = np.triu_indices_from(D, k=1)
        vals = D[iu]
        mu, sd = float(vals.mean()), float(vals.std()) + 1e-9
        Dz = (D - mu) / sd
        term = w * Dz
        total = term if total is None else total + term
    return total


def select_representative_subsets(D: np.ndarray, budget_K: int) -> List[int]:
    """Greedy facility-location coreset selection from a precomputed distance
    matrix D (n x n): maximize F(S) = sum_{x=1..n} max_{s in S} sim(x,s),
    sim = RBF(D) -- the same monotone-submodular objective proven in
    docs/submodularity-audit.md Part 1, fed a non-Euclidean distance.
    """
    n = D.shape[0]
    iu = np.triu_indices(n, k=1)
    tau = float(np.median(D[iu] ** 2)) if iu[0].size else 1.0
    tau = tau if tau > 0 else 1.0
    sim = np.exp(-(D ** 2) / tau)
    np.fill_diagonal(sim, 1.0)

    coverage = np.zeros(n)
    selected: List[int] = []
    remaining = set(range(n))
    for _ in range(min(budget_K, n)):
        gains = np.clip(sim[:, list(remaining)] - coverage[:, None], 0, None).sum(axis=0)
        best_local = int(np.argmax(gains))
        best_gain = float(gains[best_local])
        best_idx = list(remaining)[best_local]
        if best_gain <= 0:
            break
        selected.append(best_idx)
        remaining.discard(best_idx)
        coverage = np.maximum(coverage, sim[:, best_idx])
    return selected


# ---------------------------------------------------------------------------
# ProbCover (Yehuda, Bagon, Baskin & Radzyner, "Active Learning Through a
# Covering Lens", NeurIPS 2022; reference implementation:
# github.com/orobix/active-learning, activelearning/queries/representative/
# probcover_query.py)
# ---------------------------------------------------------------------------


def select_probcover(D: np.ndarray, budget_K: int, delta: float | None = None) -> List[int]:
    """ProbCover coreset selection, adapted from a pre-labeled-set active-learning
    query to a from-scratch coreset problem (no pre-existing labeled set here,
    unlike the reference implementation's X_start).

    Hard-threshold coverage: two candidates are "adjacent" iff D_ij <= delta.
    Greedily pick the candidate that covers the most currently-uncovered
    candidates (including itself), mark those covered, repeat. If no
    candidate covers anything new, halve delta and retry (mirrors the
    reference implementation's response to running out of uncovered points);
    if that still finds nothing, fall back to picking an arbitrary remaining
    candidate so the budget is always filled.

    Unlike facility location's smooth RBF coverage (max similarity, i.e.
    diminishing returns from *how close* a match is), ProbCover only cares
    whether a point is within delta at all -- a hard covering-radius
    criterion, not a submodular-guaranteed one (this is a heuristic greedy,
    with its own covering-radius argument from the source paper, not the
    NWF78 (1-1/e) bound proven for facility location in
    docs/submodularity-audit.md Part 1).
    """
    n = D.shape[0]
    if delta is None:
        iu = np.triu_indices(n, k=1)
        delta = float(np.median(D[iu])) if iu[0].size else 1.0
        delta = delta if delta > 0 else 1.0

    adjacency = D <= delta
    np.fill_diagonal(adjacency, True)  # a point always covers itself

    covered = np.zeros(n, dtype=bool)
    avail = np.ones(n, dtype=bool)
    selected: List[int] = []
    for _ in range(min(budget_K, n)):
        uncovered = ~covered
        gains = (adjacency & uncovered[None, :]).sum(axis=1).astype(float)
        gains[~avail] = -1.0
        best = int(np.argmax(gains))
        if gains[best] <= 0:
            delta = delta / 2.0
            adjacency = D <= delta
            np.fill_diagonal(adjacency, True)
            gains = (adjacency & uncovered[None, :]).sum(axis=1).astype(float)
            gains[~avail] = -1.0
            best = int(np.argmax(gains))
            if gains[best] <= 0:
                remaining = np.where(avail)[0]
                if remaining.size == 0:
                    break
                best = int(remaining[0])
        selected.append(best)
        avail[best] = False
        covered = covered | adjacency[best]
    return selected


# ---------------------------------------------------------------------------
# Kernel herding (Chen, Welling & Smola, ICML 2010)
# ---------------------------------------------------------------------------


def compute_subset_kernel_gram(clouds: Sequence[np.ndarray], tau: float | None = None,
                                rng: np.random.Generator | None = None) -> np.ndarray:
    """Gram matrix G[i,j] = <mu_i, mu_j>_H = mean_{a in cloud_i, b in cloud_j} k(a,b)
    under a shared RBF kernel -- the inner product between subset i and j's
    kernel mean embeddings (Sec 3.1). This is exactly the cross term used
    inside kernel_mean_distance, computed once for every pair so kernel
    herding can be run purely from G (no explicit feature map needed).
    """
    n = len(clouds)
    if tau is None:
        rng = rng or np.random.default_rng(0)
        pooled = np.concatenate(clouds, axis=0)
        tau = _median_bandwidth(pooled, rng)
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            g = float(_rbf_gram(clouds[i], clouds[j], tau).mean())
            G[i, j] = G[j, i] = g
    return G


def select_kernel_herding(G: np.ndarray, budget_K: int) -> List[int]:
    """Kernel herding: greedily pick the candidate whose embedding best fills
    the current gap between the full-set kernel mean mu_B and the selected
    subset's kernel mean mu_{A_t}, using only the Gram matrix G.

    a_{t+1} = argmax_i < phi(b_i), mu_B - mu_{A_t} >
            = argmax_i [ mean_j G[i,j]  -  mean_{a in A_t} G[i,a] ]

    Minimizes MMD(A,B) directly -- a *global distribution-matching*
    objective, structurally different from facility location's *coverage*
    objective (every b_j has a good representative in A). -MMD(A,B) is not
    generally monotone or submodular, so this does not carry the (1-1/e)
    guarantee proven for facility location in docs/submodularity-audit.md
    Part 1 -- it has its own convergence theory (empirical kernel mean
    convergence, Chen/Welling/Smola 2010), not a submodular one.
    """
    n = G.shape[0]
    target = G.mean(axis=1)  # <phi(b_i), mu_B> for every i
    running_sum = np.zeros(n)  # sum_{a in A_t} G[:, a]
    avail = np.ones(n, dtype=bool)
    selected: List[int] = []
    for t in range(1, min(budget_K, n) + 1):
        current_mean_proj = running_sum / max(t - 1, 1)  # <phi(b_i), mu_{A_{t-1}}>, 0 when t=1
        score = target - current_mean_proj
        score = np.where(avail, score, -np.inf)
        best = int(np.argmax(score))
        selected.append(best)
        avail[best] = False
        running_sum = running_sum + G[:, best]
    return selected


# ---------------------------------------------------------------------------
# end-to-end: regenerate the exact source pool a given make_problem() call
# will build, then pick K representatives per distance formula
# ---------------------------------------------------------------------------


def regenerate_source_pool(seed: int, n_train_models: int, n_unseen_models: int,
                           n_samplesets_full: int, d_lat: int) -> Tuple[np.ndarray, List[int]]:
    """Mirrors make_problem()'s early RNG draws exactly (U, Uo, V, target_dir,
    target_score) so the returned source indices line up with what
    make_problem(..., fixed_source_sets=...) will later index into.
    """
    rng = np.random.default_rng(seed)
    _U = rng.normal(size=(n_train_models, d_lat))
    _Uo = rng.normal(size=(n_unseen_models, d_lat)) + 0.15
    V = rng.normal(size=(n_samplesets_full, d_lat))
    target_dir = rng.normal(size=d_lat)
    target_score = V @ target_dir
    n_target = max(1, n_samplesets_full // 4)
    target_idx = set(np.argsort(-target_score)[:n_target].tolist())
    source_idx = [j for j in range(n_samplesets_full) if j not in target_idx]
    return V, source_idx


def build_clouds(V: np.ndarray, source_idx: Sequence[int], *,
                 n_points_per_subset: int = 20, spread: float = 0.3,
                 seed: int = 0) -> List[np.ndarray]:
    """Synthesize a point cloud per source sample-set: n_points scattered
    around its latent factor position V[j] -- standing in for "the actual
    raw examples inside this workload," which is what a real pipeline's
    embeddings would look like (active_evaluator/embedding_cache.py), but
    which the scalar-descriptor synthetic generator doesn't otherwise model.
    """
    rng = np.random.default_rng(seed)
    d = V.shape[1]
    return [V[j] + spread * rng.normal(size=(n_points_per_subset, d)) for j in source_idx]


# Named weighted-sum variants: formula name -> per-metric weight dict passed
# to combined_distance_matrix. "sum" (equal weights) stays the unweighted
# default, handled separately below so existing results/callers are untouched.
WEIGHTED_SUM_FORMULAS: Dict[str, Dict[str, float]] = {
    # Hausdorff was the standout performer at K=60 (Sec 27.6); double its
    # weight relative to kernel_mean/sliced_wasserstein in the combined sum.
    # (A more aggressive 0.7/0.2/0.1 convex-combination weighting was also
    # tried and performed worse than this 2:1:1 version -- see git history
    # for that result; removed here to keep this the one supported variant.)
    "sum_hausdorff_weighted": {"hausdorff": 2.0, "kernel_mean": 1.0, "sliced_wasserstein": 1.0},
}


def select_via_distance_formula(seed: int, formula: str, *, n_train_models: int,
                                n_unseen_models: int, n_samplesets_full: int, d_lat: int,
                                budget_K: int, n_points_per_subset: int = 20,
                                spread: float = 0.3) -> Tuple[List[int], Dict[str, float]]:
    """Returns (selected original sample-set indices, timing/diagnostics dict)."""
    t0 = time.time()
    V, source_idx = regenerate_source_pool(seed, n_train_models, n_unseen_models,
                                           n_samplesets_full, d_lat)
    clouds = build_clouds(V, source_idx, n_points_per_subset=n_points_per_subset,
                          spread=spread, seed=seed)
    t_clouds = time.time() - t0

    t1 = time.time()
    if formula == "kernel_herding":
        # Needs the Gram matrix (inner products), not a distance matrix --
        # branches to select_kernel_herding below instead of the shared
        # facility-location/ProbCover distance-matrix path.
        D = None
        G = compute_subset_kernel_gram(clouds)
    elif formula == "probcover":
        # ProbCover is distance-metric-agnostic; defaults to the kernel_mean
        # (MMD) distance matrix as the closest analogue to the reference
        # implementation's plain Euclidean distance.
        D = pairwise_distance_matrix(clouds, DISTANCE_FNS["kernel_mean"])
    elif formula == "sum":
        base = all_distance_matrices(clouds)
        D = combined_distance_matrix(base)
    elif formula in WEIGHTED_SUM_FORMULAS:
        base = all_distance_matrices(clouds)
        D = combined_distance_matrix(base, weights=WEIGHTED_SUM_FORMULAS[formula])
    else:
        D = pairwise_distance_matrix(clouds, DISTANCE_FNS[formula])
    t_distmat = time.time() - t1

    t2 = time.time()
    if formula == "kernel_herding":
        local_selected = select_kernel_herding(G, budget_K)
    elif formula == "probcover":
        local_selected = select_probcover(D, budget_K)
    else:
        local_selected = select_representative_subsets(D, budget_K)
    t_select = time.time() - t2

    selected_original = [source_idx[i] for i in local_selected]
    diag = {
        "n_candidates": len(source_idx),
        "n_selected": len(selected_original),
        "seconds_build_clouds": t_clouds,
        "seconds_distance_matrix": t_distmat,
        "seconds_greedy_select": t_select,
        "seconds_total": t_clouds + t_distmat + t_select,
    }
    return selected_original, diag

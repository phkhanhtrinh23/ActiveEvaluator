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


def combined_distance_matrix(base: Dict[str, np.ndarray]) -> np.ndarray:
    """Standardize each of the three matrices then sum -- 'sum of all 3'."""
    total = None
    for D in base.values():
        iu = np.triu_indices_from(D, k=1)
        vals = D[iu]
        mu, sd = float(vals.mean()), float(vals.std()) + 1e-9
        Dz = (D - mu) / sd
        total = Dz if total is None else total + Dz
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
    if formula == "sum":
        base = all_distance_matrices(clouds)
        D = combined_distance_matrix(base)
    else:
        D = pairwise_distance_matrix(clouds, DISTANCE_FNS[formula])
    t_distmat = time.time() - t1

    t2 = time.time()
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

"""Representative workload selection by greedy facility location (Alg. 1)."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


def median_bandwidth(D: np.ndarray) -> float:
    """tau = median_{u<v} D_uv^2."""
    n = len(D)
    vals = np.concatenate([D[i, i + 1 :] for i in range(n - 1)]).astype(np.float64) ** 2
    tau = float(np.median(vals, overwrite_input=True)) if vals.size else 1.0
    return tau if tau > 0 else 1.0


def similarity(D: np.ndarray, tau: float | None = None) -> tuple[np.ndarray, float]:
    """S_uv = exp(-D_uv^2 / tau)."""
    tau = median_bandwidth(D) if tau is None else tau
    S = np.exp(-(D.astype(np.float32) ** 2) / np.float32(tau))
    np.fill_diagonal(S, 1.0)
    return S, tau


@dataclass
class Selection:
    order: np.ndarray  # selected workload ids in greedy order
    gains: np.ndarray  # marginal gain of each pick
    weights: Dict[int, np.ndarray]  # budget K -> gamma over order[:K]
    tau: float

    def subset(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        return self.order[:k], self.weights[k]

    def save(self, path) -> None:
        np.savez(
            path,
            order=self.order,
            gains=self.gains,
            tau=self.tau,
            budgets=np.array(sorted(self.weights)),
            **{f"gamma_{k}": w for k, w in self.weights.items()},
        )

    @classmethod
    def load(cls, path) -> "Selection":
        z = np.load(path)
        weights = {int(k): z[f"gamma_{k}"] for k in z["budgets"]}
        return cls(order=z["order"], gains=z["gains"], weights=weights, tau=float(z["tau"]))


def facility_location(S: np.ndarray, budgets: Sequence[int]) -> tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
    """Greedy max of F(A) = sum_j max_{u in A} S_uj with lazy evaluation.

    Returns the greedy order, marginal gains, and for each budget K the
    multiplicities gamma_u = |{v : u_A(v) = u}| of the first K picks.
    """
    n = len(S)
    k_max = min(max(budgets), n)
    snapshots = sorted({min(k, n) for k in budgets})
    cover = np.zeros(n, dtype=np.float32)
    owner = np.full(n, -1, dtype=np.int64)  # position in `order` of each workload's representative
    heap = [(-float(g), q) for q, g in enumerate(S.sum(axis=1, dtype=np.float64))]
    heapq.heapify(heap)
    order: List[int] = []
    gains: List[float] = []
    weights: Dict[int, np.ndarray] = {}
    while len(order) < k_max:
        _, q = heapq.heappop(heap)
        gain = float(np.maximum(S[q] - cover, 0).sum(dtype=np.float64))
        if heap and gain < -heap[0][0]:
            heapq.heappush(heap, (-gain, q))
            continue
        better = S[q] > cover
        owner[better] = len(order)
        cover[better] = S[q][better]
        order.append(q)
        gains.append(gain)
        if len(order) in snapshots:
            weights[len(order)] = np.bincount(owner, minlength=len(order)).astype(np.float32)
    return np.array(order), np.array(gains), weights


def select(D: np.ndarray, budgets: Sequence[int] | int) -> Selection:
    budgets = [budgets] if isinstance(budgets, int) else list(budgets)
    S, tau = similarity(D)
    order, gains, weights = facility_location(S, budgets)
    return Selection(order=order, gains=gains, weights=weights, tau=tau)


def objective(S: np.ndarray, subset: Sequence[int]) -> float:
    return float(S[list(subset)].max(axis=0).sum()) if len(subset) else 0.0

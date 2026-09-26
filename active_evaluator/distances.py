"""Pairwise distances between workload embedding sets."""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence

import numpy as np
import torch

Sets = Sequence[np.ndarray]


def _pad(sets: Sets, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    n, d = len(sets), sets[0].shape[1]
    length = max(len(s) for s in sets)
    x = torch.zeros(n, length, d, device=device)
    mask = torch.zeros(n, length, dtype=torch.bool, device=device)
    for i, s in enumerate(sets):
        x[i, : len(s)] = torch.as_tensor(s, dtype=torch.float32, device=device)
        mask[i, : len(s)] = True
    return x, mask


def _nearest(a: torch.Tensor, b: torch.Tensor, chunk: int) -> torch.Tensor:
    """Distance from every point of a to its nearest point of b, in chunks."""
    out = []
    for i in range(0, len(a), chunk):
        best = torch.full((min(chunk, len(a) - i),), float("inf"), device=a.device)
        for j in range(0, len(b), chunk):
            best = torch.minimum(best, torch.cdist(a[i : i + chunk], b[j : j + chunk]).amin(1))
        out.append(best)
    return torch.cat(out)


def _large_pairwise(sets: Sets, reduce: str, device: torch.device, chunk: int = 4096) -> np.ndarray:
    n = len(sets)
    out = np.zeros((n, n), dtype=np.float32)
    for u in range(n):
        a = torch.as_tensor(sets[u], dtype=torch.float32, device=device)
        for v in range(u + 1, n):
            b = torch.as_tensor(sets[v], dtype=torch.float32, device=device)
            d_ab, d_ba = _nearest(a, b, chunk), _nearest(b, a, chunk)
            if reduce == "max":
                d = torch.maximum(d_ab.max(), d_ba.max())
            else:
                d = 0.5 * (d_ab.mean() + d_ba.mean())
            out[u, v] = out[v, u] = float(d)
    return out


def _set_pairwise(sets: Sets, reduce: str, block: int | None, device: torch.device) -> np.ndarray:
    """Hausdorff (reduce='max') or Chamfer (reduce='mean') for every pair of sets."""
    n = len(sets)
    longest = max(len(s) for s in sets)
    if longest**2 > 2**25:
        return _large_pairwise(sets, reduce, device)
    if block is None:
        block = max(1, int(np.sqrt(2**25 / longest**2)))
    out = np.zeros((n, n), dtype=np.float32)
    starts = list(range(0, n, block))
    for bi, i0 in enumerate(starts):
        xa, ma = _pad(sets[i0 : i0 + block], device)
        na = (xa * xa).sum(-1)
        for j0 in starts[bi:]:
            xb, mb = _pad(sets[j0 : j0 + block], device)
            nb = (xb * xb).sum(-1)
            # (A, B, La, Lb) squared distances
            sq = na[:, None, :, None] + nb[None, :, None, :] - 2 * torch.einsum("ald,bmd->ablm", xa, xb)
            sq = sq.clamp_min(0)
            inf = torch.tensor(float("inf"), device=device)
            to_b = torch.where(mb[None, :, None, :], sq, inf).amin(-1)  # (A, B, La)
            to_a = torch.where(ma[:, None, :, None], sq, inf).amin(-2)  # (A, B, Lb)
            va, vb = ma[:, None, :].expand_as(to_b), mb[None, :, :].expand_as(to_a)
            if reduce == "max":
                d_ab = torch.where(va, to_b, -inf).amax(-1)
                d_ba = torch.where(vb, to_a, -inf).amax(-1)
                d = torch.maximum(d_ab, d_ba).sqrt()
            else:
                d_ab = (to_b.sqrt() * va).sum(-1) / va.sum(-1)
                d_ba = (to_a.sqrt() * vb).sum(-1) / vb.sum(-1)
                d = 0.5 * (d_ab + d_ba)
            d = d.cpu().numpy()
            out[i0 : i0 + len(d), j0 : j0 + d.shape[1]] = d
            out[j0 : j0 + d.shape[1], i0 : i0 + len(d)] = d.T
    np.fill_diagonal(out, 0.0)
    return out


def hausdorff(sets: Sets, block: int | None = None, device: str | None = None) -> np.ndarray:
    return _set_pairwise(sets, "max", block, _device(device))


def chamfer(sets: Sets, block: int | None = None, device: str | None = None) -> np.ndarray:
    return _set_pairwise(sets, "mean", block, _device(device))


def _means(sets: Sets) -> np.ndarray:
    return np.stack([s.astype(np.float64).mean(0) for s in sets])


def euclidean(sets: Sets, **_) -> np.ndarray:
    means = torch.as_tensor(_means(sets))
    return torch.cdist(means, means).numpy().astype(np.float32)


def mahalanobis(sets: Sets, **_) -> np.ndarray:
    pooled = np.concatenate(sets).astype(np.float64)
    cov = np.cov(pooled, rowvar=False) + 1e-6 * np.eye(pooled.shape[1])
    white = np.linalg.cholesky(np.linalg.inv(cov))
    means = torch.as_tensor(_means(sets) @ white)
    return torch.cdist(means, means).numpy().astype(np.float32)


def sliced_wasserstein(sets: Sets, n_proj: int = 128, n_quantiles: int = 64, seed: int = 0, **_) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=(sets[0].shape[1], n_proj))
    theta /= np.linalg.norm(theta, axis=0, keepdims=True)
    levels = (np.arange(n_quantiles) + 0.5) / n_quantiles
    q = np.stack([np.quantile(s.astype(np.float64) @ theta, levels, axis=0).ravel() for s in sets])
    q = torch.as_tensor(q)
    return (torch.cdist(q, q) / np.sqrt(n_proj * n_quantiles)).numpy().astype(np.float32)


def frechet(sets: Sets, block: int = 256, device: str | None = None, **_) -> np.ndarray:
    dev = _device(device)
    d = sets[0].shape[1]
    mu = torch.as_tensor(_means(sets), device=dev)
    cov = torch.as_tensor(
        np.stack([np.cov(s.astype(np.float64), rowvar=False) if len(s) > 1 else np.zeros((d, d)) for s in sets]),
        dtype=torch.float64,
        device=dev,
    )
    evals, evecs = torch.linalg.eigh(cov)
    root = evecs @ torch.diag_embed(evals.clamp_min(0).sqrt()) @ evecs.transpose(-1, -2)
    trace = cov.diagonal(dim1=-2, dim2=-1).sum(-1)
    n = len(sets)
    out = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j0 in range(i + 1, n, block):
            j = slice(j0, min(j0 + block, n))
            m = root[i] @ cov[j] @ root[i]
            cross = torch.linalg.eigvalsh(m).clamp_min(0).sqrt().sum(-1)
            fd = ((mu[i] - mu[j]) ** 2).sum(-1) + trace[i] + trace[j] - 2 * cross
            out[i, j] = out[j, i] = fd.clamp_min(0).sqrt().cpu().numpy()
    return out


def _device(device: str | None) -> torch.device:
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


DISTANCES: Dict[str, Callable[..., np.ndarray]] = {
    "hausdorff": hausdorff,
    "sliced_wasserstein": sliced_wasserstein,
    "frechet": frechet,
    "chamfer": chamfer,
    "mahalanobis": mahalanobis,
    "euclidean": euclidean,
}


def pairwise(sets: List[np.ndarray], metric: str = "hausdorff", **kwargs) -> np.ndarray:
    return DISTANCES[metric](sets, **kwargs)

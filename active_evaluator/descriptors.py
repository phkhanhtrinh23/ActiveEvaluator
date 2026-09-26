"""Shift descriptor SD(D_S, b, f) computed from a model's own features and confidences."""

from __future__ import annotations

import numpy as np

FEATURES = (
    "frechet",
    "mean_shift",
    "mahalanobis",
    "swd_mean",
    "swd_std",
    "swd_max",
    "conf_mean",
    "conf_shift",
)

EPS = 1e-6


class ShiftDescriptor:
    """Fitted on the source data D_S of one model; maps a workload to its shift features."""

    def __init__(self, source_emb: np.ndarray, source_conf: np.ndarray, dim: int = 64,
                 n_proj: int = 128, n_quantiles: int = 64, seed: int = 0):
        x = source_emb.astype(np.float64)
        self.center = x.mean(0)
        _, _, vt = np.linalg.svd(x - self.center, full_matrices=False)
        self.basis = vt[:dim].T
        z = self._reduce(source_emb)
        rng = np.random.default_rng(seed)
        theta = rng.normal(size=(z.shape[1], n_proj))
        self.theta = theta / np.linalg.norm(theta, axis=0, keepdims=True)
        self.levels = (np.arange(n_quantiles) + 0.5) / n_quantiles
        self.mean, self.var, self.cov = z.mean(0), z.var(0).clip(EPS), np.cov(z, rowvar=False)
        self.quantiles = np.quantile(z @ self.theta, self.levels, axis=0)
        self.conf = float(np.mean(source_conf))

    def _reduce(self, emb: np.ndarray) -> np.ndarray:
        return (emb.astype(np.float64) - self.center) @ self.basis

    def __call__(self, emb: np.ndarray, conf: np.ndarray) -> np.ndarray:
        z = self._reduce(emb)
        mean, var = z.mean(0), z.var(0).clip(EPS)
        diff = mean - self.mean
        mean_shift = float(diff @ diff)
        frechet = mean_shift + float(((np.sqrt(var / self.var) - 1.0) ** 2).sum())
        cov = np.cov(z, rowvar=False) if len(z) > 1 else np.zeros_like(self.cov)
        pooled = 0.5 * (self.cov + cov) + EPS * np.eye(len(diff))
        mahalanobis = float(np.sqrt(max(diff @ np.linalg.pinv(pooled) @ diff, 0.0)))
        swd = np.abs(np.quantile(z @ self.theta, self.levels, axis=0) - self.quantiles).mean(0)
        conf_mean = float(np.mean(conf))
        return np.array(
            [frechet, np.sqrt(mean_shift), mahalanobis, swd.mean(), swd.std(), swd.max(), conf_mean, conf_mean - self.conf],
            dtype=np.float32,
        )

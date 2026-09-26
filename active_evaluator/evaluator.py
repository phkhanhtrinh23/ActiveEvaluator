"""Meta-learned accuracy evaluator g_theta(SD) with per-model adaptation.

`cavia` is the MetaEvaluator architecture (context-vector adaptation). The
other algorithms are the ablations in Tab. 2 (left).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ALGORITHMS = ("cavia", "maml", "fomaml", "reptile", "metasgd", "anil", "protonet")
ADAPT_STEPS = {"cavia": 5, "maml": 12, "fomaml": 10, "reptile": 9, "metasgd": 9, "anil": 8, "protonet": 0}
INNER_LR = {"cavia": 1.0, "maml": 0.05, "fomaml": 0.05, "reptile": 0.05, "metasgd": 0.05, "anil": 0.05, "protonet": 0.0}


@dataclass
class Task:
    """Supervision of one model: shift descriptors, accuracies and loss weights gamma."""

    x: np.ndarray
    y: np.ndarray
    w: np.ndarray


def _wmse(pred: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (w * (pred - y) ** 2).sum() / w.sum()


class MetaEvaluator:
    def __init__(self, n_features: int, algorithm: str = "cavia", context_dim: int = 32,
                 hidden: Sequence[int] = (128, 64), inner_lr: float | None = None, inner_steps: int | None = None,
                 adapt_steps: int | None = None, outer_lr: float = 1e-3, weight_decay: float = 1e-4, epochs: int = 2000,
                 tasks_per_batch: int = 4, batch_size: int = 256, seed: int = 0, device: str | None = None):
        if algorithm not in ALGORITHMS:
            raise ValueError(f"unknown algorithm {algorithm}")
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.device = torch.device(device or "cpu")
        self.algorithm = algorithm
        self.inner_lr = INNER_LR[algorithm] if inner_lr is None else inner_lr
        self.adapt_steps = ADAPT_STEPS[algorithm] if adapt_steps is None else adapt_steps
        self.inner_steps = self.adapt_steps if inner_steps is None else inner_steps
        self.autocast = torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda")
        self.epochs, self.tasks_per_batch, self.batch_size = epochs, tasks_per_batch, batch_size
        self.context_dim = context_dim if algorithm in ("cavia", "protonet") else 0

        dims = [n_features + self.context_dim, *hidden, 1]
        layers = [nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])]
        self.params = nn.ParameterList([p for layer in layers for p in (layer.weight, layer.bias)]).to(self.device)
        modules = [self.params]
        if algorithm == "metasgd":
            self.alphas = nn.ParameterList([nn.Parameter(torch.full_like(p, self.inner_lr)) for p in self.params])
            modules.append(self.alphas)
        if algorithm == "protonet":
            self.proto = nn.Sequential(nn.Linear(n_features + 1, context_dim), nn.ReLU(),
                                       nn.Linear(context_dim, context_dim)).to(self.device)
            modules.append(self.proto)
        self.optimizer = torch.optim.Adam([p for m in modules for p in m.parameters()], lr=outer_lr,
                                          weight_decay=weight_decay)
        self.mu = torch.zeros(n_features, device=self.device)
        self.sd = torch.ones(n_features, device=self.device)

    # --- functional model -------------------------------------------------
    def _forward(self, params: Sequence[torch.Tensor], x: torch.Tensor, ctx: torch.Tensor | None) -> torch.Tensor:
        if ctx is not None:
            x = torch.cat([x, ctx.expand(len(x), -1)], dim=1)
        n = len(params) // 2
        for i in range(n):
            x = F.linear(x, params[2 * i], params[2 * i + 1])
            if i < n - 1:
                x = F.relu(x)
        return torch.sigmoid(x).squeeze(-1)

    def _adapt(self, x, y, w, steps: int, create_graph: bool):
        params: List[torch.Tensor] = list(self.params)
        if self.algorithm == "protonet":
            h = self.proto(torch.cat([x, y[:, None]], dim=1))
            return params, (w[:, None] * h).sum(0) / w.sum()
        if self.algorithm == "cavia":
            ctx = torch.zeros(self.context_dim, device=self.device, requires_grad=True)
            for _ in range(steps):
                (g,) = torch.autograd.grad(_wmse(self._forward(params, x, ctx), y, w), ctx, create_graph=create_graph)
                ctx = ctx - self.inner_lr * g
            return params, ctx
        idx = [len(params) - 2, len(params) - 1] if self.algorithm == "anil" else list(range(len(params)))
        for _ in range(steps):
            loss = _wmse(self._forward(params, x, None), y, w)
            grads = torch.autograd.grad(loss, [params[i] for i in idx], create_graph=create_graph)
            for i, g in zip(idx, grads):
                lr = self.alphas[i] if self.algorithm == "metasgd" else self.inner_lr
                params[i] = params[i] - lr * (g if create_graph else g.detach())
        return params, None

    # --- training ----------------------------------------------------------
    def _tensor(self, a: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(a, dtype=torch.float32, device=self.device)

    def _normalize(self, x: np.ndarray) -> torch.Tensor:
        return (self._tensor(x) - self.mu) / self.sd

    def fit(self, tasks: Sequence[Task]) -> "MetaEvaluator":
        x = np.concatenate([t.x for t in tasks])
        w = np.concatenate([t.w for t in tasks])[:, None]
        mu = (w * x).sum(0) / w.sum()
        self.mu = self._tensor(mu)
        self.sd = self._tensor(np.sqrt((w * (x - mu) ** 2).sum(0) / w.sum())).clamp_min(1e-6)
        data = [(self._normalize(t.x), self._tensor(t.y), self._tensor(t.w)) for t in tasks]
        second_order = self.algorithm not in ("fomaml", "reptile")

        for _ in range(self.epochs):
            batch = self.rng.choice(len(data), size=min(self.tasks_per_batch, len(data)), replace=False)
            self.optimizer.zero_grad()
            if self.algorithm == "reptile":
                for p in self.params:
                    p.grad = torch.zeros_like(p)
            total = 0.0
            for b in batch:
                x, y, w = data[b]
                with self.autocast:
                    total = self._task_loss(x, y, w, total, len(batch), second_order)
            if self.algorithm != "reptile":
                (total / len(batch)).backward()
            self.optimizer.step()
        return self

    def _task_loss(self, x, y, w, total, n_batch: int, second_order: bool):
        rows = torch.as_tensor(self.rng.permutation(len(x))[: self.batch_size], device=self.device)
        half = max(1, len(rows) // 2)
        s, q = rows[:half], rows[half:] if len(rows) > 1 else rows
        if self.algorithm == "reptile":
            fast, _ = self._adapt(x[rows], y[rows], w[rows], self.inner_steps, create_graph=False)
            for p, f in zip(self.params, fast):
                p.grad += (p - f).detach().float() / (n_batch * self.inner_lr * max(self.inner_steps, 1))
            return total
        params, ctx = self._adapt(x[s], y[s], w[s], self.inner_steps, create_graph=second_order)
        return total + _wmse(self._forward(params, x[q], ctx), y[q], w[q]).float()

    # --- deployment ----------------------------------------------------------
    def adapt(self, task: Task):
        """Adapt to an unseen model from its (SD, accuracy) pairs on the selected workloads."""
        with self.autocast:
            params, ctx = self._adapt(self._normalize(task.x), self._tensor(task.y), self._tensor(task.w),
                                      self.adapt_steps, create_graph=False)
        return [p.detach() for p in params], None if ctx is None else ctx.detach()

    @torch.no_grad()
    def predict(self, state, x: np.ndarray) -> np.ndarray:
        params, ctx = state
        with self.autocast:
            return self._forward(params, self._normalize(np.atleast_2d(x)), ctx).float().cpu().numpy()

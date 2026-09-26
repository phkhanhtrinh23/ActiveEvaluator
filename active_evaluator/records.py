"""Per-model supervision and target outputs, cached on disk."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np

from .evaluator import Task

_DICTS = ("target_sd", "target_acc", "target_conf", "target_pred", "calib_conf", "calib_correct", "calib_pred", "twin_pred")


@dataclass
class ModelRecord:
    name: str
    workload_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    sd: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    acc: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    seconds: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    target_sd: Dict[str, np.ndarray] = field(default_factory=dict)
    target_acc: Dict[str, float] = field(default_factory=dict)
    target_conf: Dict[str, np.ndarray] = field(default_factory=dict)
    target_pred: Dict[str, np.ndarray] = field(default_factory=dict)
    calib_conf: Dict[str, np.ndarray] = field(default_factory=dict)
    calib_correct: Dict[str, np.ndarray] = field(default_factory=dict)
    calib_pred: Dict[str, np.ndarray] = field(default_factory=dict)
    twin_pred: Dict[str, np.ndarray] = field(default_factory=dict)

    def add_workloads(self, ids, sd, acc, seconds) -> None:
        self.workload_ids = np.concatenate([self.workload_ids, np.asarray(ids, dtype=np.int64)])
        self.sd = np.concatenate([self.sd.reshape(-1, np.shape(sd)[1]), np.asarray(sd, dtype=np.float32)])
        self.acc = np.concatenate([self.acc, np.asarray(acc, dtype=np.float32)])
        self.seconds = np.concatenate([self.seconds, np.asarray(seconds, dtype=np.float32)])

    def missing(self, ids) -> np.ndarray:
        return np.setdiff1d(np.asarray(ids), self.workload_ids)

    def rows(self, ids) -> np.ndarray:
        pos = {w: i for i, w in enumerate(self.workload_ids)}
        return np.array([pos[w] for w in ids], dtype=np.int64)

    def task(self, ids, weights) -> Task:
        r = self.rows(ids)
        return Task(self.sd[r], self.acc[r], np.asarray(weights, dtype=np.float32))

    def save(self, path: Path) -> None:
        arrays = {k: getattr(self, k) for k in ("workload_ids", "sd", "acc", "seconds")}
        for name in _DICTS:
            for key, value in getattr(self, name).items():
                arrays[f"{name}::{key}"] = np.asarray(value)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, name=self.name, **arrays)

    @classmethod
    def load(cls, path: Path) -> "ModelRecord":
        z = np.load(path, allow_pickle=False)
        rec = cls(name=str(z["name"]), **{k: z[k] for k in ("workload_ids", "sd", "acc", "seconds")})
        for key in z.files:
            if "::" in key:
                name, target = key.split("::", 1)
                value = z[key]
                getattr(rec, name)[target] = float(value) if name == "target_acc" else value
        return rec

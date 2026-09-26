"""Evaluation protocol: repeated runs over sampled pools of unseen models."""

from __future__ import annotations

import time
from typing import Dict, List, Sequence

import numpy as np

from . import metrics
from .baselines import BASELINES
from .evaluator import MetaEvaluator
from .records import ModelRecord

Records = Dict[str, ModelRecord]


def split_pool(names: Sequence[str], pool_size: int, run: int, seed: int = 0) -> tuple[List[str], List[str]]:
    """Unseen pool for this run; every other model is a reference model."""
    names = sorted(names)
    unseen = set(np.random.default_rng(seed + run).choice(names, size=pool_size, replace=False))
    return [n for n in names if n not in unseen], [n for n in names if n in unseen]


def _score(pred: Dict[str, Dict[str, float]], records: Records, unseen: Sequence[str]) -> Dict[str, Dict[str, float]]:
    out = {}
    for target, by_model in pred.items():
        p = [by_model[m] for m in unseen]
        t = [records[m].target_acc[target] for m in unseen]
        out[target] = {"mae": metrics.mae(p, t), **metrics.ranking(p, t)}
    return out


def run_meta(records: Records, targets: Sequence[str], ids: np.ndarray, weights: np.ndarray, *, pool_size: int,
             runs: int, seed: int = 0, **evaluator_kwargs) -> List[dict]:
    """Train on the reference models' supervision over `ids`, evaluate unseen models."""
    results = []
    for run in range(runs):
        reference, unseen = split_pool(records, pool_size, run, seed)
        n_features = records[reference[0]].sd.shape[1]
        t0 = time.perf_counter()
        ev = MetaEvaluator(n_features, seed=seed + run, **evaluator_kwargs)
        ev.fit([records[m].task(ids, weights) for m in reference])
        fit_seconds = time.perf_counter() - t0

        pred: Dict[str, Dict[str, float]] = {t: {} for t in targets}
        latency = {}
        for m in unseen:
            rec = records[m]
            t0 = time.perf_counter()
            state = ev.adapt(rec.task(ids, weights))
            for t in targets:
                pred[t][m] = float(ev.predict(state, rec.target_sd[t])[0])
            latency[m] = float(rec.seconds[rec.rows(ids)].sum()) + time.perf_counter() - t0
        supervision = sum(float(records[m].seconds[records[m].rows(ids)].sum()) for m in reference)
        results.append({
            "metrics": _score(pred, records, unseen),
            "fit_seconds": fit_seconds,
            "supervision_seconds": supervision,
            "unseen_seconds": latency,
            "predictions": pred,
        })
    return results


def run_baselines(records: Records, targets: Sequence[str], *, pool_size: int, runs: int, seed: int = 0,
                  names: Sequence[str] = tuple(BASELINES)) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {n: [] for n in names}
    for run in range(runs):
        _, unseen = split_pool(records, pool_size, run, seed)
        pool = [records[m] for m in unseen]
        for name in names:
            pred = {t: dict(zip(unseen, BASELINES[name](pool, t).tolist())) for t in targets}
            out[name].append({"metrics": _score(pred, records, unseen), "predictions": pred})
    return out


def summarize(runs: List[dict], targets: Sequence[str]) -> Dict[str, Dict[str, List[float]]]:
    """Mean and 95% CI per target and metric; 'Avg.' averages means and CIs over targets."""
    keys = runs[0]["metrics"][targets[0]].keys()
    table = {t: {k: list(metrics.mean_ci([r["metrics"][t][k] for r in runs])) for k in keys} for t in targets}
    table["Avg."] = {k: [float(np.mean([table[t][k][i] for t in targets])) for i in (0, 1)] for k in keys}
    return table

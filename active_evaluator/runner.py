"""Shared command line for all modalities.

    select      embed every workload, compute pairwise distances, greedy selection
    records     run each model on the needed workloads and targets, cache SD/accuracy
    experiment  train evaluators on A (or B) and report MAE / ranking over unseen pools
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Protocol

import numpy as np

from . import protocol
from .distances import DISTANCES, pairwise
from .records import ModelRecord
from .selection import Selection, select


class Modality(Protocol):
    targets: List[str]

    def num_workloads(self) -> int: ...

    def workload_sets(self, max_points: int) -> List[np.ndarray]: ...

    def model_names(self) -> List[str]: ...

    def update_record(self, record: ModelRecord, workload_ids: np.ndarray) -> ModelRecord: ...


def slug(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "-" for c in name)


def cmd_select(m: Modality, args) -> None:
    t0 = time.perf_counter()
    sets = m.workload_sets(args.max_points)
    t_embed = time.perf_counter() - t0
    t0 = time.perf_counter()
    D = pairwise(sets, args.metric, device=args.device)
    t_dist = time.perf_counter() - t0
    t0 = time.perf_counter()
    sel = select(D, args.budgets)
    t_greedy = time.perf_counter() - t0
    path = args.out / f"selection_{args.metric}.npz"
    sel.save(path)
    timing = {"embed_seconds": t_embed, "distance_seconds": t_dist, "greedy_seconds": t_greedy, "tau": sel.tau}
    (args.out / f"selection_{args.metric}.json").write_text(json.dumps(timing, indent=2))
    print(f"saved {path}: {len(sel.order)} of {len(D)} workloads, {timing}")


def _workloads(m: Modality, args) -> np.ndarray:
    if args.full:
        return np.arange(m.num_workloads())
    return np.unique(np.concatenate([Selection.load(p).order for p in args.selection]))


def cmd_records(m: Modality, args) -> None:
    ids = _workloads(m, args)
    for name in args.models or m.model_names():
        path = args.out / "records" / f"{slug(name)}.npz"
        record = ModelRecord.load(path) if path.exists() else ModelRecord(name=name)
        record = m.update_record(record, ids)
        record.save(path)
        print(f"{name}: {len(record.workload_ids)} workloads, targets {sorted(record.target_acc)}")


def _load_records(m: Modality, args) -> Dict[str, ModelRecord]:
    names = args.models or m.model_names()
    return {n: ModelRecord.load(args.out / "records" / f"{slug(n)}.npz") for n in names}


def cmd_experiment(m: Modality, args) -> None:
    records = _load_records(m, args)
    targets = [t for t in m.targets if all(t in r.target_acc for r in records.values())]
    settings = []
    for path in args.selection:
        sel = Selection.load(path)
        metric = Path(path).stem.removeprefix("selection_")
        for k in args.budgets:
            ids, w = sel.subset(k)
            settings.append((f"ActiveEvaluator[{metric},K={k}]", ids, w))
    if args.full:
        n = m.num_workloads()
        settings.append(("MetaEvaluator[full]", np.arange(n), np.ones(n, dtype=np.float32)))

    kw = dict(epochs=args.epochs, adapt_steps=args.adapt_steps, inner_steps=args.inner_steps, device=args.device)
    results: Dict[str, dict] = {}
    for pool in args.pool_sizes:
        for label, ids, w in settings:
            for algo in args.algorithms:
                key = f"{label}[{algo}][pool={pool}]"
                runs = protocol.run_meta(records, targets, ids, w, pool_size=pool, runs=args.runs,
                                         seed=args.seed, algorithm=algo, **kw)
                results[key] = {"summary": protocol.summarize(runs, targets), "runs": runs}
                _print(key, results[key]["summary"])
        if args.baselines:
            for name, runs in protocol.run_baselines(records, targets, pool_size=pool, runs=args.runs,
                                                     seed=args.seed).items():
                key = f"{name}[pool={pool}]"
                results[key] = {"summary": protocol.summarize(runs, targets), "runs": runs}
                _print(key, results[key]["summary"])
    out = args.out / args.name
    out.write_text(json.dumps(results, indent=2))
    print(f"saved {out}")


def _print(key: str, summary: dict) -> None:
    cells = "  ".join(f"{t}: {v['mae'][0]:.2f}±{v['mae'][1]:.2f}" for t, v in summary.items())
    print(f"{key}\n  MAE  {cells}\n  tau  {summary['Avg.']['kendall_tau'][0]:.3f}")


def main(modality_factory, description: str) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("command", choices=["select", "records", "experiment"])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--metric", default="hausdorff", choices=sorted(DISTANCES))
    parser.add_argument("--max-points", type=int, default=None, help="optional cap on points per workload embedding set")
    parser.add_argument("--budgets", type=int, nargs="+", default=[9000])
    parser.add_argument("--selection", type=Path, nargs="*", default=None)
    parser.add_argument("--full", action="store_true", help="use the full meta-dataset B")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--pool-sizes", type=int, nargs="+", default=[10])
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--algorithms", nargs="+", default=["cavia"])
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--inner-steps", type=int, default=None)
    parser.add_argument("--adapt-steps", type=int, default=None)
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default="results.json")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    args.out = args.out or Path(config.get("out", "outputs"))
    args.out.mkdir(parents=True, exist_ok=True)
    if args.selection is None:
        default = args.out / f"selection_{args.metric}.npz"
        args.selection = [default] if args.command != "select" and default.exists() else []
    if args.command == "records" and not args.full and not args.selection:
        parser.error("records needs --selection or --full")
    modality = modality_factory(config, out=args.out, device=args.device)
    {"select": cmd_select, "records": cmd_records, "experiment": cmd_experiment}[args.command](modality, args)

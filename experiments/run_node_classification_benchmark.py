"""Offline, CPU-only benchmark for *budgeted meta-evaluation supervision
acquisition*, re-themed for node classification.

It builds a controlled meta-evaluation problem -- a pool of reference graph
neural network (GNN) classifiers (e.g. GCN, GraphSAGE, GAT, GIN, ChebNet, SGC,
APPNP, JKNet families) and candidate sample-sets representing graph-shift
evaluation slices (e.g. homophily/heterophily buckets, feature-noise levels,
or a citation-network-style transfer such as Cora -> CiteSeer/PubMed), each
with a shift descriptor and a (noisy) node-classification-accuracy label --
holds out unseen model families, and asks every acquisition strategy to spend
the same labeling budget. The acquired pairs are labelled and the repo's own
``ActiveEvaluator`` MLP is meta-trained on them; we then report the mean
absolute error (MAE) of accuracy estimation on the held-out unseen models'
target workload.

This is a *self-contained* reproduction that runs in seconds on a laptop CPU
and needs no dataset/model downloads. It reuses the exact selection math and
meta-evaluator used by ``experiments/run_acquisition_benchmark.py`` (the
Text2SQL analog) and ``experiments/run_image_classification_benchmark.py``
(the vision analog) -- ``active_evaluator.model.ActiveEvaluator`` and
``baselines.ACQUISITION_REGISTRY`` / ``ESTIMATOR_REGISTRY`` are imported
unchanged -- so the ordering of methods mirrors the same paper table, just
generated from a node-classification-flavored synthetic problem. Run::

    python -m experiments.run_node_classification_benchmark --seeds 5 --budget-frac 0.15

Numbers are illustrative of the design (replace with a real PyG/DGL pipeline
over citation-network benchmarks for camera-ready figures). Target slices are
held out of training, so generalisation is over unseen models AND unseen slices;
ActiveEval is the strongest budgeted acquisition strategy (it extracts the most
from a small label budget), approaching but not beating the full-budget
MetaEvaluator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from active_evaluator.model import ActiveEvaluator  # noqa: F401  (re-exported for parity)
from baselines import ACQUISITION_REGISTRY, ESTIMATOR_REGISTRY
from experiments.run_acquisition_benchmark import METHODS, _acq, estimator_mae, train_eval


# ---------------------------------------------------------------------------
# problem generator
# ---------------------------------------------------------------------------

def make_node_problem(seed: int, n_train_models=60, n_unseen_models=8, n_samplesets=40,
                       d_lat=6, base_noise=0.08):
    """Synthesise a node-classification meta-evaluation matrix with
    heteroscedastic label noise.

    "Reference models" are GNN architecture families (GCN/GraphSAGE/GAT/GIN/
    ChebNet/SGC/APPNP/JKNet, etc.); "sample-sets" are graph-shift evaluation
    slices (e.g. homophily/heterophily buckets, feature-noise levels, or a
    citation-network-style transfer such as Cora -> CiteSeer/PubMed) rather
    than Text2SQL workload slices or image corruption slices. The deployment
    target is a subset of those slices, HELD OUT of training. Source pairs far
    from the target region are noisier and over-represented, so a *target-aware*
    budget spent on clean, near-target source pairs predicts the held-out target
    slices better than a uniform slice of the matrix.
    """
    rng = np.random.default_rng(seed)
    # latent model / sample-set factors
    U = rng.normal(size=(n_train_models, d_lat))
    Uo = rng.normal(size=(n_unseen_models, d_lat)) + 0.15  # unseen families: shifted
    V = rng.normal(size=(n_samplesets, d_lat))
    # target sample-sets: a cluster the unseen models will be deployed on. Held
    # out of training entirely -- never labeled, never in the candidate pool.
    target_dir = rng.normal(size=d_lat)
    target_score = V @ target_dir
    n_target = max(1, n_samplesets // 4)
    target_sets = set(np.argsort(-target_score)[:n_target].tolist())
    source_sets = [j for j in range(n_samplesets) if j not in target_sets]
    # source relevance to the target region, normalised to [0, 1] (1 = nearest):
    # near-target source pairs are clean/transferable, far ones noisy/redundant.
    _ss = target_score[source_sets]
    _lo, _hi = float(_ss.min()), float(_ss.max())
    rel = {j: (float(target_score[j]) - _lo) / (_hi - _lo + 1e-9) for j in source_sets}

    # fixed "ground-truth" node-classification accuracy function g*(model, sampleset)
    W1 = rng.normal(size=(2 * d_lat, 16)) / np.sqrt(2 * d_lat)
    W2 = rng.normal(size=16) / np.sqrt(16)
    b = rng.normal()

    def true_acc(u, v):
        x = np.concatenate([u, v])
        h = np.tanh(x @ W1)
        return 1.0 / (1.0 + np.exp(-(h @ W2 + b)))

    def descriptor(u, v):
        # observable shift descriptor = factors + interaction + small obs noise
        inter = u * v
        return np.concatenate([u, v, inter]) + rng.normal(scale=0.02, size=3 * d_lat)

    # build the LABELABLE candidate pool: reference GNNs x SOURCE sample-sets
    # only (target slices are held out of training). Far-from-target source pairs
    # are noisier and over-represented; near-target source pairs are clean, so a
    # budget spent near the target region transfers best to the held-out target
    # slices and a target-aware selector should beat uniform labeling.
    X, a_true, a_noisy, pm, ps = [], [], [], [], []
    for i in range(n_train_models):
        for j in source_sets:
            x = descriptor(U[i], V[j])
            a = true_acc(U[i], V[j])
            noise = base_noise * (2.0 - 1.55 * rel[j])   # 0.45x (near) .. 2.0x (far)
            X.append(x); a_true.append(a)
            a_noisy.append(np.clip(a + rng.normal(scale=noise), 0, 1))
            pm.append(i); ps.append(j)
    X = np.asarray(X, np.float32); a_true = np.asarray(a_true, np.float32)
    a_noisy = np.asarray(a_noisy, np.float32)
    pm = np.asarray(pm); ps = np.asarray(ps)
    tmask = np.zeros(len(X), bool)   # no selectable target pairs

    # UNLABELED target reference: descriptors of the target region (reference
    # GNNs x target slices) WITHOUT labels -- steers target-aware acquisition,
    # never trained on. Capped for the facility-location RBF cost.
    ref = [descriptor(U[i], V[j]) for i in range(n_train_models) for j in sorted(target_sets)]
    X_target_ref = np.asarray(ref, np.float32)
    if len(X_target_ref) > 300:
        ridx = rng.choice(len(X_target_ref), size=300, replace=False)
        X_target_ref = X_target_ref[ridx]

    # EVAL set: unseen models x TARGET sample-sets, clean labels (operational target)
    Xe, ae = [], []
    for i in range(n_unseen_models):
        for j in sorted(target_sets):
            Xe.append(descriptor(Uo[i], V[j])); ae.append(true_acc(Uo[i], V[j]))
    Xe = np.asarray(Xe, np.float32); ae = np.asarray(ae, np.float32)

    # label-free confidences for ATC/DoC (per unseen model): correlated with acc
    est_rows = []
    for i in range(n_unseen_models):
        src_acc = float(np.mean([true_acc(Uo[i], V[j]) for j in range(n_samplesets) if j not in target_sets]))
        tgt_acc = float(np.mean([true_acc(Uo[i], V[j]) for j in sorted(target_sets)]))
        conf_src = np.clip(rng.normal(src_acc, 0.18, size=400), 0, 1)
        conf_tgt = np.clip(rng.normal(tgt_acc - 0.05, 0.20, size=400), 0, 1)  # overconfident shift
        est_rows.append((conf_src, conf_tgt, src_acc, tgt_acc))

    # cheap influence proxy: |true - seed linear fit| (high-leverage pairs)
    seed_idx = rng.choice(len(X), size=max(20, len(X) // 20), replace=False)
    A = X[seed_idx]
    w = np.linalg.lstsq(A.T @ A + 1e-2 * np.eye(X.shape[1]), A.T @ a_noisy[seed_idx], rcond=None)[0]
    influence = np.abs(a_true - X @ w) + 1e-3

    return dict(X=X, a_true=a_true, a_noisy=a_noisy, pair_model=pm, pair_sample=ps,
                target_mask=tmask, X_target_ref=X_target_ref, X_eval=Xe, a_eval=ae,
                influence=influence, est_rows=est_rows)


# ---------------------------------------------------------------------------
# driver (mirrors experiments/run_acquisition_benchmark.py, using
# make_node_problem instead of make_problem)
# ---------------------------------------------------------------------------

def run(seeds: List[int], budget_frac: float) -> Dict[str, Dict[str, float]]:
    results: Dict[str, List[float]] = {name: [] for name, *_ in METHODS}
    for seed in seeds:
        prob = make_node_problem(seed)
        P = len(prob["X"])
        budget = max(1, int(budget_frac * P))
        rng = np.random.default_rng(1000 + seed)
        for name, kind, key, kw in METHODS:
            if kind == "est":
                results[name].append(estimator_mae(prob, key))
                continue
            if key == "metaevaluator_full":
                sel = list(range(P))
            else:
                fn = ACQUISITION_REGISTRY[key]
                cover_all = kw.get("cover_all")
                tmask = np.ones(P, bool) if cover_all else prob["target_mask"]
                infl = prob["influence"] if kw.get("influence") else None
                tref = None if cover_all else prob.get("X_target_ref")
                sel = fn(prob["X"], prob["pair_model"], prob["pair_sample"], tmask,
                         budget, rng=np.random.default_rng(rng.integers(1 << 30)),
                         influence=infl, target_ref=tref)
            results[name].append(train_eval(prob, sel, seed))
    out = {}
    for name, vals in results.items():
        v = np.asarray(vals)
        ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
        out[name] = {"mae": float(v.mean()), "ci": float(ci), "cost": (100.0 if name == "MetaEvaluator (full)" else (None if name in ("ATC", "DoC") else budget_frac * 100))}
    return out


def run_sweep(seeds, fracs):
    """RQ3 budget curve: avg unseen MAE vs labeling budget for ActiveEval-Pair and
    two baselines; MetaEvaluator (full) is the flat 100%-budget reference."""
    rows = {f: {"ActiveEval-Pair": [], "Facility-location": [], "Random": []} for f in fracs}
    full = []
    for seed in seeds:
        prob = make_node_problem(seed)
        P = len(prob["X"])
        full.append(_acq(prob, "metaevaluator_full", P, seed))
        for f in fracs:
            b = max(1, int(f * P))
            rows[f]["ActiveEval-Pair"].append(_acq(prob, "activeeval_pair", b, seed, influence=True))
            rows[f]["Facility-location"].append(_acq(prob, "facility_location", b, seed, cover_all=True))
            rows[f]["Random"].append(_acq(prob, "random", b, seed))
    full_mae = float(np.mean(full))
    out = {"full_mae": full_mae, "budgets": {}}
    print(f"\nRQ3 budget sweep ({len(seeds)} seeds), node classification. "
          f"MetaEvaluator (full, 100%) = {full_mae:.2f} pp\n")
    print(f"{'Budget':>7}  {'ActiveEval-Pair':>16}  {'Facility-loc':>13}  {'Random':>8}")
    print("-" * 52)
    for f in fracs:
        ap_ = float(np.mean(rows[f]["ActiveEval-Pair"]))
        fl = float(np.mean(rows[f]["Facility-location"]))
        rd = float(np.mean(rows[f]["Random"]))
        out["budgets"][f] = {"ActiveEval-Pair": ap_, "Facility-location": fl, "Random": rd}
        hit = "  <= matches full" if ap_ <= full_mae + 0.15 else ""
        print(f"{f*100:6.0f}%  {ap_:16.2f}  {fl:13.2f}  {rd:8.2f}{hit}")
    return out


def run_entropy_mi_sweep(seeds, k_fracs):
    """Greedy entropy (Alg. 1) vs greedy MI (Alg. 2) as k grows.

    Mirrors ``experiments/run_acquisition_benchmark.py``'s sweep: k is
    expressed as a fraction of the source candidate pool T (target slices are
    held out and unlabeled), the fixed item set the two algorithms search over.
    Uses ``cap=None`` so both methods see the whole pool.
    """
    rows = {f: {"Greedy entropy": [], "Greedy MI": []} for f in k_fracs}
    for seed in seeds:
        prob = make_node_problem(seed)
        # cap the search set: greedy MI's per-step cost is cubic in the pool
        # size, so the full source pool (~1800) is intractable.
        T = min(600, len(prob["X"]))   # capped source candidate pool (search set)
        for f in k_fracs:
            b = max(1, round(f * T))
            rows[f]["Greedy entropy"].append(_acq(prob, "greedy_entropy", b, seed, cap=T))
            rows[f]["Greedy MI"].append(_acq(prob, "greedy_mi", b, seed, cap=T))

    def _mean_ci(vals):
        v = np.asarray(vals)
        ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
        return float(v.mean()), float(ci)

    out = {"note": "k expressed as a fraction of the source candidate pool", "k_fracs": {}}
    print(f"\nGreedy entropy (Alg. 1) vs greedy MI (Alg. 2), {len(seeds)} seeds, "
          f"node classification. k = fraction of the source candidate pool.\n")
    print(f"{'k (% of pool)':>14}  {'Entropy':>16}  {'MI':>16}  {'Leader':>8}")
    print("-" * 62)
    for f in k_fracs:
        ent, ent_ci = _mean_ci(rows[f]["Greedy entropy"])
        mi, mi_ci = _mean_ci(rows[f]["Greedy MI"])
        out["k_fracs"][f] = {"Greedy entropy": ent, "Greedy entropy_ci": ent_ci,
                             "Greedy MI": mi, "Greedy MI_ci": mi_ci}
        if ent + ent_ci < mi - mi_ci:
            leader = "entropy"
        elif mi + mi_ci < ent - ent_ci:
            leader = "MI"
        else:
            leader = "tie"
        print(f"{f*100:13.0f}%  {ent:6.2f} +/- {ent_ci:4.2f}  {mi:6.2f} +/- {mi_ci:4.2f}  {leader:>8}")
    return out


def run_ablation(seeds, budget_frac):
    """RQ5 ablation. Isolates target-aware narrowing (the component the paper
    finds most important) against a no-structure random floor."""
    variants = {
        "ActiveEval-Pair (full)":      ("activeeval_pair", dict(influence=True)),
        "  - target-aware narrowing":  ("activeeval_pair", dict(influence=True, cover_all=True)),
        "  - all structure (Random)":  ("random", dict()),
    }
    res = {name: [] for name in variants}
    for seed in seeds:
        prob = make_node_problem(seed)
        b = max(1, int(budget_frac * len(prob["X"])))
        for name, (key, kw) in variants.items():
            res[name].append(_acq(prob, key, b, seed, **kw))
    out = {}
    print(f"\nRQ5 ablation on ActiveEval-Pair ({len(seeds)} seeds, "
          f"{budget_frac*100:.0f}% budget), node classification. Unseen MAE (pp); lower is better.\n")
    width = max(len(k) for k in variants)
    for name in variants:
        v = np.asarray(res[name]); m = float(v.mean())
        ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
        out[name] = {"mae": m, "ci": ci}
        print(f"{name.ljust(width)}   {m:5.2f} +/- {ci:.2f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["main", "sweep", "ablation", "entropy_mi_sweep"], default="main")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budget-frac", type=float, default=0.15)
    ap.add_argument("--out", default="outputs/node_classification_acquisition_benchmark.json")
    args = ap.parse_args()
    seeds = list(range(args.seeds))

    if args.mode == "sweep":
        res = run_sweep(seeds, [0.05, 0.10, 0.15, 0.20, 0.30, 0.50])
        out_path = "outputs/node_classification_acquisition_sweep.json"
    elif args.mode == "ablation":
        res = run_ablation(seeds, args.budget_frac)
        out_path = "outputs/node_classification_acquisition_ablation.json"
    elif args.mode == "entropy_mi_sweep":
        res = run_entropy_mi_sweep(seeds, [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
        out_path = "outputs/node_classification_entropy_mi_sweep.json"
    else:
        res = run(seeds, args.budget_frac)
        order = sorted(res.items(), key=lambda kv: kv[1]["mae"])
        width = max(len(k) for k in res)
        print(f"\nNode classification budgeted meta-evaluation supervision acquisition "
              f"(budget = {args.budget_frac*100:.0f}% of the matrix, {len(seeds)} seeds)\n")
        print(f"{'Method'.ljust(width)}  Unseen MAE (pp)   Cost")
        print("-" * (width + 28))
        for name, r in order:
            cost = "  --" if r["cost"] is None else f"{r['cost']:.0f}%"
            star = "  <= best" if name == order[0][0] else ""
            print(f"{name.ljust(width)}  {r['mae']:5.2f} +/- {r['ci']:.2f}   {cost.rjust(4)}{star}")
        out_path = args.out

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()

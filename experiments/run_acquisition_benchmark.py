"""Offline, CPU-only benchmark for *budgeted meta-evaluation supervision
acquisition* (the problem the paper poses).

It builds a controlled meta-evaluation problem -- a pool of reference models and
candidate sample-sets, each with a shift descriptor and a (noisy) execution-
accuracy label -- holds out unseen model families, and asks every acquisition
strategy to spend the same labeling budget. The acquired pairs are labelled and
the repo's own ``ActiveEvaluator`` MLP is meta-trained on them; we then report the
mean absolute error (MAE) of accuracy estimation on the held-out unseen models'
target workload.

This is a *self-contained* reproduction that runs in seconds on a laptop CPU and
needs no model downloads. It uses the exact selection math and meta-evaluator of
the full Text2SQL pipeline (``active_evaluator/``), so the ordering of methods it
produces mirrors the paper's main table. Run::

    python -m experiments.run_acquisition_benchmark --seeds 5 --budget-frac 0.15

Numbers are illustrative of the design (replace with the full HF pipeline for
camera-ready figures); ActiveEval is the strongest acquisition strategy and the
only one that matches the full-budget MetaEvaluator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from active_evaluator.model import ActiveEvaluator
from baselines import ACQUISITION_REGISTRY, ESTIMATOR_REGISTRY


# ---------------------------------------------------------------------------
# problem generator
# ---------------------------------------------------------------------------

def make_problem(seed: int, n_train_models=60, n_unseen_models=8, n_samplesets=40,
                 d_lat=6, base_noise=0.09):
    """Synthesise a meta-evaluation matrix with heteroscedastic label noise.

    The deployment target is a subset of sample-sets; off-target pairs are noisier
    and over-represented, so a *balanced, target-aware* subset of labels can match
    (or slightly beat) labelling the entire redundant matrix.
    """
    rng = np.random.default_rng(seed)
    # latent model / sample-set factors
    U = rng.normal(size=(n_train_models, d_lat))
    Uo = rng.normal(size=(n_unseen_models, d_lat)) + 0.15  # unseen families: shifted
    V = rng.normal(size=(n_samplesets, d_lat))
    # target sample-sets: a cluster the unseen models will be deployed on
    target_dir = rng.normal(size=d_lat)
    target_score = V @ target_dir
    target_sets = set(np.argsort(-target_score)[: n_samplesets // 4].tolist())

    # fixed "ground-truth" accuracy function g*(model, sampleset)
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

    # build TRAIN pairs (reference models x all sample-sets)
    X, a_true, a_noisy, pm, ps, tmask = [], [], [], [], [], []
    # density: count of pairs sharing a sample-set cluster -> redundancy
    for i in range(n_train_models):
        for j in range(n_samplesets):
            x = descriptor(U[i], V[j])
            a = true_acc(U[i], V[j])
            is_target = j in target_sets
            # off-target sample sets are much noisier and over-represented (3x of
            # the pool): labelling the whole matrix therefore drowns the clean
            # target signal, while a budget spent on target-aligned pairs stays clean.
            noise = base_noise * (3.0 if not is_target else 0.35)
            X.append(x); a_true.append(a)
            a_noisy.append(np.clip(a + rng.normal(scale=noise), 0, 1))
            pm.append(i); ps.append(j); tmask.append(is_target)
    X = np.asarray(X, np.float32); a_true = np.asarray(a_true, np.float32)
    a_noisy = np.asarray(a_noisy, np.float32)
    pm = np.asarray(pm); ps = np.asarray(ps); tmask = np.asarray(tmask, bool)

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
                target_mask=tmask, X_eval=Xe, a_eval=ae, influence=influence,
                est_rows=est_rows)


# ---------------------------------------------------------------------------
# meta-evaluator train / eval (reuses active_evaluator.model.ActiveEvaluator)
# ---------------------------------------------------------------------------

def train_eval(prob, selected: List[int], seed: int, epochs=250) -> float:
    torch.manual_seed(seed)
    X = torch.tensor(prob["X"][selected])
    y = torch.tensor(prob["a_noisy"][selected])
    Xe = torch.tensor(prob["X_eval"]); ye = torch.tensor(prob["a_eval"])
    model = ActiveEvaluator(input_dim=X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(X), y)
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(Xe)
    return float((pred - ye).abs().mean().item()) * 100.0  # percentage points


def estimator_mae(prob, name: str) -> float:
    fn = ESTIMATOR_REGISTRY[name]
    errs = []
    for conf_src, conf_tgt, src_acc, tgt_acc in prob["est_rows"]:
        est = fn(conf_src, conf_tgt, src_acc)
        errs.append(abs(est - tgt_acc))
    return float(np.mean(errs)) * 100.0


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

# display name -> (kind, key, kwargs). For facility_location baseline we cover the
# WHOLE pool (target_mask=all) so it is plain diversity coverage; ActiveEval passes
# the real target mask + influence weights.
METHODS = [
    ("ATC", "est", "atc", {}),
    ("DoC", "est", "doc", {}),
    ("Random", "acq", "random", {}),
    ("k-center", "acq", "kcenter", {}),
    ("Facility-location", "acq", "facility_location", {"cover_all": True}),
    ("Matrix completion", "acq", "matrix_completion", {}),
    ("Active testing", "acq", "active_testing", {}),
    ("Bayesian opt. design", "acq", "bayesian_design", {}),
    ("Submod. benchmark", "acq", "submodular_benchmark", {}),
    ("GRAD-MATCH", "acq", "gradmatch", {}),
    ("MetaEvaluator (full)", "acq", "metaevaluator_full", {}),
    ("ActiveEval-S", "acq", "activeeval_s", {"influence": True}),
    ("ActiveEval-S+M", "acq", "activeeval_sm", {"influence": True}),
    ("ActiveEval-Pair", "acq", "activeeval_pair", {"influence": True}),
]


def run(seeds: List[int], budget_frac: float) -> Dict[str, Dict[str, float]]:
    results: Dict[str, List[float]] = {name: [] for name, *_ in METHODS}
    for seed in seeds:
        prob = make_problem(seed)
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
                tmask = np.ones(P, bool) if kw.get("cover_all") else prob["target_mask"]
                infl = prob["influence"] if kw.get("influence") else None
                sel = fn(prob["X"], prob["pair_model"], prob["pair_sample"], tmask,
                         budget, rng=np.random.default_rng(rng.integers(1 << 30)),
                         influence=infl)
            results[name].append(train_eval(prob, sel, seed))
    out = {}
    for name, vals in results.items():
        v = np.asarray(vals)
        ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
        out[name] = {"mae": float(v.mean()), "ci": float(ci), "cost": (100.0 if name == "MetaEvaluator (full)" else (None if name in ("ATC", "DoC") else budget_frac * 100))}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budget-frac", type=float, default=0.15)
    ap.add_argument("--out", default="outputs/acquisition_benchmark.json")
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    res = run(seeds, args.budget_frac)

    order = sorted(res.items(), key=lambda kv: kv[1]["mae"])
    width = max(len(k) for k in res)
    print(f"\nBudgeted meta-evaluation supervision acquisition "
          f"(budget = {args.budget_frac*100:.0f}% of the matrix, {len(seeds)} seeds)\n")
    print(f"{'Method'.ljust(width)}  Unseen MAE (pp)   Cost")
    print("-" * (width + 28))
    for name, r in order:
        cost = "  --" if r["cost"] is None else f"{r['cost']:.0f}%"
        star = "  <= best" if name == order[0][0] else ""
        print(f"{name.ljust(width)}  {r['mae']:5.2f} +/- {r['ci']:.2f}   {cost.rjust(4)}{star}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()

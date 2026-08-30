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
camera-ready figures); the target-aware budgeted methods (greedy MI/entropy run
uncapped and the ActiveEval variants) cluster at the top and all match or beat
the full-budget MetaEvaluator.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from active_evaluator.model import ActiveEvaluator
from baselines import ACQUISITION_REGISTRY, ESTIMATOR_REGISTRY
from experiments.cost_models import make_cost_model, greedy_fill
from experiments.subset_distances import select_via_distance_formula


# ---------------------------------------------------------------------------
# problem generator
# ---------------------------------------------------------------------------

def make_problem(seed: int, n_train_models=60, n_unseen_models=8, n_samplesets=40,
                 d_lat=6, base_noise=0.08, off_on_noise_ratio=2.0 / 0.45,
                 n_clusters=None, reps_per_cluster=None, random_subset_size=None,
                 fixed_source_sets=None):
    """Synthesise a meta-evaluation matrix with heteroscedastic label noise.

    The deployment target is a subset of sample-sets, HELD OUT of training. Source
    pairs far from the target region are noisier and over-represented, so a
    *target-aware* budget spent on clean, near-target source pairs predicts the
    held-out target workload better than a uniform slice of the matrix.

    Optional cheap-vs-expensive two-stage reduction of the SOURCE sample-set
    universe (getting a model's accuracy on a sample-set is expensive -- it means
    running every example in it -- while a sample-set's shift descriptor is cheap
    to compute up front for the whole universe). When ``n_clusters`` and
    ``reps_per_cluster`` are both given, the ``n_samplesets`` source sample-sets
    are first k-means clustered (in descriptor space) into ``n_clusters`` groups,
    and only the ``reps_per_cluster`` sample-sets closest to each cluster centroid
    are ever labeled -- i.e. only K*M (not the full source universe) sample-sets
    ever get run against the N reference models, giving K*M*N labelable triples
    instead of |source|*N. ``random_subset_size`` is a control: pick that many
    source sample-sets uniformly at random instead of via k-means, to isolate
    whether clustering (vs. plain subsampling) is what helps. Both are mutually
    exclusive and off by default (exact prior behavior: use every source
    sample-set). See docs/kmeans-submodular-warmstart.md for the theory this
    feeds (``kmeans_diag.quantization_radius_max`` is the measured rho).
    """
    rng = np.random.default_rng(seed)
    # latent model / sample-set factors
    U = rng.normal(size=(n_train_models, d_lat))
    Uo = rng.normal(size=(n_unseen_models, d_lat)) + 0.15  # unseen families: shifted
    V = rng.normal(size=(n_samplesets, d_lat))
    # target sample-sets: a cluster the unseen models will be deployed on. These
    # are HELD OUT of training entirely -- never labeled, never in the candidate
    # pool. Generalisation is over unseen models AND these unseen workloads.
    target_dir = rng.normal(size=d_lat)
    target_score = V @ target_dir
    n_target = max(1, n_samplesets // 4)
    target_sets = set(np.argsort(-target_score)[:n_target].tolist())
    source_sets_all = [j for j in range(n_samplesets) if j not in target_sets]

    kmeans_diag = None
    if n_clusters is not None and reps_per_cluster is not None:
        from sklearn.cluster import KMeans
        Vs = V[source_sets_all]
        m_eff = max(1, min(n_clusters, len(source_sets_all)))
        km = KMeans(n_clusters=m_eff, n_init=10, random_state=seed).fit(Vs)
        reps, radii = [], []
        for c in range(m_eff):
            members = [source_sets_all[i] for i in range(len(Vs)) if km.labels_[i] == c]
            if not members:
                continue
            # distance of EVERY member to its centroid -- this is what bounds rho
            # in the quantization-stability theorem, not just the chosen reps.
            dists = sorted((float(np.linalg.norm(V[j] - km.cluster_centers_[c])), j) for j in members)
            radii.extend(d for d, _ in dists)
            reps.extend(j for _, j in dists[:reps_per_cluster])
        source_sets = reps
        kmeans_diag = {
            "n_clusters": m_eff,
            "reps_per_cluster": reps_per_cluster,
            "n_source_full": len(source_sets_all),
            "n_source_reduced": len(source_sets),
            "quantization_radius_max": float(np.max(radii)) if radii else 0.0,
            "quantization_radius_mean": float(np.mean(radii)) if radii else 0.0,
            "inertia": float(km.inertia_),
        }
    elif random_subset_size is not None:
        k = max(1, min(random_subset_size, len(source_sets_all)))
        source_sets = sorted(rng.choice(source_sets_all, size=k, replace=False).tolist())
    elif fixed_source_sets is not None:
        # Caller (e.g. experiments.subset_distances.select_via_distance_formula)
        # already picked which source sample-sets to keep, via its own
        # reduction rule -- use exactly that list unchanged.
        source_sets = list(fixed_source_sets)
    else:
        source_sets = source_sets_all
    # source relevance: proximity of each source sample-set to the target region,
    # normalised to [0, 1] (1 = nearest). Near-target source pairs carry clean,
    # transferable signal; far ones are noisy and redundant, so a target-aware
    # selector should prefer near-target source pairs.
    _ss = target_score[source_sets]
    _lo, _hi = float(_ss.min()), float(_ss.max())
    rel = {j: (float(target_score[j]) - _lo) / (_hi - _lo + 1e-9) for j in source_sets}

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

    # build the LABELABLE candidate pool: reference models x SOURCE sample-sets
    # only (target sample-sets are held out of training entirely). Far-from-target
    # source pairs are noisier and over-represented; near-target source pairs are
    # clean. A budget spent near the target region transfers best to the held-out
    # target workload, so a target-aware selector should beat uniform labeling.
    # `off_on_noise_ratio` scales the far-source noise (default 2.0/0.45 keeps the
    # original ~4.4x clean/noisy gap between nearest and farthest source pairs).
    X, a_true, a_noisy, pm, ps = [], [], [], [], []
    for i in range(n_train_models):
        for j in source_sets:
            x = descriptor(U[i], V[j])
            a = true_acc(U[i], V[j])
            noise = base_noise * 0.45 * (1.0 + (off_on_noise_ratio - 1.0) * (1.0 - rel[j]))
            X.append(x); a_true.append(a)
            a_noisy.append(np.clip(a + rng.normal(scale=noise), 0, 1))
            pm.append(i); ps.append(j)
    X = np.asarray(X, np.float32); a_true = np.asarray(a_true, np.float32)
    a_noisy = np.asarray(a_noisy, np.float32)
    pm = np.asarray(pm); ps = np.asarray(ps)
    # target sample-sets are not selectable; the target signal reaches acquisition
    # only through the unlabeled X_target_ref below.
    tmask = np.zeros(len(X), bool)

    # UNLABELED target reference: descriptors of the target region (reference
    # models x target sample-sets) WITHOUT labels -- steers target-aware
    # acquisition toward the target region but is never trained on. Capped for
    # the facility-location RBF cost.
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
                influence=influence, est_rows=est_rows, kmeans_diag=kmeans_diag,
                n_labelable_source=len(source_sets))


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
    # "Greedy entropy (Alg. 1)" is deliberately NOT listed here: with sigma=1.0 (the
    # shared default -- see baselines/_core.py) it is the same pivoted-Cholesky
    # greedy log-det algorithm as "Submod. benchmark" (select_logdet), so it would
    # just duplicate that row in every table below. It stays in ACQUISITION_REGISTRY
    # and is compared against Greedy MI directly in the entropy_mi_sweep mode (its
    # original purpose, mirroring benchmark-selection/code/eval_entropy_vs_mi.py).
    # See docs/rq4-cost-currency-mechanism.md for the ablation confirming the two
    # are statistically indistinguishable once regularization/pool-cap are matched.
    ("Greedy MI (Alg. 2)", "acq", "greedy_mi", {"cap": None}),
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
                cover_all = kw.get("cover_all")
                tmask = np.ones(P, bool) if cover_all else prob["target_mask"]
                infl = prob["influence"] if kw.get("influence") else None
                tref = None if cover_all else prob.get("X_target_ref")
                extra = {k: v for k, v in kw.items() if k not in ("cover_all", "influence")}
                sel = fn(prob["X"], prob["pair_model"], prob["pair_sample"], tmask,
                         budget, rng=np.random.default_rng(rng.integers(1 << 30)),
                         influence=infl, target_ref=tref, **extra)
            results[name].append(train_eval(prob, sel, seed))
    out = {}
    for name, vals in results.items():
        v = np.asarray(vals)
        ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
        out[name] = {"mae": float(v.mean()), "ci": float(ci), "cost": (100.0 if name == "MetaEvaluator (full)" else (None if name in ("ATC", "DoC") else budget_frac * 100))}
    return out


def _acq(prob, key, budget, seed, *, cover_all=False, influence=False, **kw):
    """Run one acquisition rule and return the trained-evaluator unseen MAE."""
    P = len(prob["X"])
    if key == "metaevaluator_full":
        sel = list(range(P))
    else:
        fn = ACQUISITION_REGISTRY[key]
        tmask = np.ones(P, bool) if cover_all else prob["target_mask"]
        infl = prob["influence"] if influence else None
        tref = None if cover_all else prob.get("X_target_ref")
        sel = fn(prob["X"], prob["pair_model"], prob["pair_sample"], tmask, budget,
                 rng=np.random.default_rng(seed), influence=infl, target_ref=tref, **kw)
    return train_eval(prob, sel, seed)


def run_sweep(seeds, fracs):
    """RQ3 budget curve: avg unseen MAE vs labeling budget for ActiveEval-Pair and
    two baselines; MetaEvaluator (full) is the flat 100%-budget reference."""
    rows = {f: {"ActiveEval-Pair": [], "Facility-location": [], "Random": []} for f in fracs}
    full = []
    for seed in seeds:
        prob = make_problem(seed)
        P = len(prob["X"])
        full.append(_acq(prob, "metaevaluator_full", P, seed))
        for f in fracs:
            b = max(1, int(f * P))
            rows[f]["ActiveEval-Pair"].append(_acq(prob, "activeeval_pair", b, seed, influence=True))
            rows[f]["Facility-location"].append(_acq(prob, "facility_location", b, seed, cover_all=True))
            rows[f]["Random"].append(_acq(prob, "random", b, seed))
    full_mae = float(np.mean(full))
    out = {"full_mae": full_mae, "budgets": {}}
    print(f"\nRQ3 budget sweep ({len(seeds)} seeds). MetaEvaluator (full, 100%) = {full_mae:.2f} pp\n")
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
    """Greedy entropy (Alg. 1) vs greedy MI (Alg. 2) across labeling budgets.

    Mirrors benchmark-selection/code/eval_entropy_vs_mi.py's k-sweep, but the
    axis is *reported* as the labeling budget in % of the source candidate pool
    so it is directly comparable with every other table. ``k_fracs`` parameterize
    the sweep as fractions of the source pool T (the fixed item set the two
    algorithms search over; target sample-sets are held out and unlabeled). Uses
    ``cap=None`` so both methods see the whole pool -- the registry default
    (``cap=200``) exists only to keep greedy MI's cubic per-step cost bounded in
    benchmarks where T runs into the thousands.
    """
    rows = {f: {"Greedy entropy": [], "Greedy MI": []} for f in k_fracs}
    # bound the entropy/MI search set to POOL items: greedy MI's per-step
    # complement-precision refactorization is cubic in the pool size, so the full
    # source pool (~1800) is intractable. POOL matches the old target-pool scale.
    POOL = 600
    P = T = None
    for seed in seeds:
        prob = make_problem(seed)
        T = min(POOL, len(prob["X"]))   # capped source candidate pool (search set)
        P = len(prob["X"])
        for f in k_fracs:
            b = max(1, round(f * T))
            rows[f]["Greedy entropy"].append(_acq(prob, "greedy_entropy", b, seed, cap=T))
            rows[f]["Greedy MI"].append(_acq(prob, "greedy_mi", b, seed, cap=T))
    def _mean_ci(vals):
        v = np.asarray(vals)
        ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
        return float(v.mean()), float(ci)

    out = {"note": "budget reported as % of the source candidate pool; both "
                   "methods search the whole source pool (target sample-sets are "
                   "held out and unlabeled)", "budgets": {}}
    print(f"\nGreedy entropy (Alg. 1) vs greedy MI (Alg. 2), {len(seeds)} seeds. "
          f"Budget in % of the source candidate pool.\n")
    print(f"{'Budget':>7}  {'Labels':>6}  {'Entropy':>16}  {'MI':>16}  {'Leader':>8}")
    print("-" * 62)
    for f in k_fracs:
        b = max(1, round(f * T))
        budget_pct = 100.0 * b / P
        ent, ent_ci = _mean_ci(rows[f]["Greedy entropy"])
        mi, mi_ci = _mean_ci(rows[f]["Greedy MI"])
        out["budgets"][round(budget_pct, 2)] = {
            "labels": b,
            "Greedy entropy": ent, "Greedy entropy_ci": ent_ci,
            "Greedy MI": mi, "Greedy MI_ci": mi_ci}
        if ent + ent_ci < mi - mi_ci:
            leader = "entropy"
        elif mi + mi_ci < ent - ent_ci:
            leader = "MI"
        else:
            leader = "tie"
        print(f"{budget_pct:6.2f}%  {b:6d}  {ent:6.2f} +/- {ent_ci:4.2f}  "
              f"{mi:6.2f} +/- {mi_ci:4.2f}  {leader:>8}")
    return out


def run_noise_sweep(seeds, budget_frac, ratios):
    """Sensitivity of the less-is-more effect to the off/on-target noise gap.

    Sweeps ``off_on_noise_ratio`` (near-target source noise stays fixed; only the
    far-source noise scales) and compares ActiveEval-Pair at the given budget
    against MetaEvaluator (full, 100% labels) and whole-pool Random. At ratio
    1.0 the noise is homoscedastic, so the full matrix is strictly more
    information and should win; the sweep locates the noise gap at which a
    clean near-target source subset overtakes labelling everything."""
    methods = ["ActiveEval-Pair", "MetaEvaluator (full)", "Random"]
    rows = {r: {m: [] for m in methods} for r in ratios}
    for seed in seeds:
        for r in ratios:
            prob = make_problem(seed, off_on_noise_ratio=r)
            P = len(prob["X"])
            b = max(1, int(budget_frac * P))
            rows[r]["ActiveEval-Pair"].append(_acq(prob, "activeeval_pair", b, seed, influence=True))
            rows[r]["MetaEvaluator (full)"].append(_acq(prob, "metaevaluator_full", P, seed))
            rows[r]["Random"].append(_acq(prob, "random", b, seed))

    def _mean_ci(vals):
        v = np.asarray(vals)
        ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
        return float(v.mean()), float(ci)

    out = {"note": "off_on_noise_ratio scales far-source label noise only; "
                   "near-target source noise fixed at base_noise*0.45", "ratios": {}}
    print(f"\nNoise-sensitivity sweep ({len(seeds)} seeds, budget = "
          f"{budget_frac*100:.0f}%). Unseen MAE (pp); lower is better.\n")
    print(f"{'off/on noise':>13}  {'ActiveEval-Pair':>17}  {'Full (100%)':>15}  {'Random':>15}  {'Leader':>7}")
    print("-" * 78)
    for r in ratios:
        ae, ae_ci = _mean_ci(rows[r]["ActiveEval-Pair"])
        fu, fu_ci = _mean_ci(rows[r]["MetaEvaluator (full)"])
        rd, rd_ci = _mean_ci(rows[r]["Random"])
        out["ratios"][r] = {"ActiveEval-Pair": ae, "ActiveEval-Pair_ci": ae_ci,
                            "MetaEvaluator (full)": fu, "MetaEvaluator (full)_ci": fu_ci,
                            "Random": rd, "Random_ci": rd_ci}
        if ae + ae_ci < fu - fu_ci:
            leader = "AE"
        elif fu + fu_ci < ae - ae_ci:
            leader = "full"
        else:
            leader = "tie"
        print(f"{r:12.1f}x  {ae:7.2f} +/- {ae_ci:4.2f}  {fu:7.2f} +/- {fu_ci:4.2f}"
              f"  {rd:7.2f} +/- {rd_ci:4.2f}  {leader:>7}")
    return out


# cost-budget currencies in display order (count = the original cardinality budget)
COST_CURRENCIES = [
    ("count", "actions"),
    ("input_tok", "in-tok"),
    ("output_tok", "out-tok"),
    ("latency", "sec"),
    ("memory", "GB-mem"),
    ("storage", "GB-store"),
]
# how many actions of a method's greedy order to precompute (currency-independent,
# since selection is cost-agnostic). A budget that stretches past this length just
# under-spends -- the realized cost fraction is reported so that stays visible.
_N_ORDER = 450
# greedy MI's per-step complement-precision refactorization is cubic in its search
# universe, so a cost budget's larger action count (_N_ORDER) makes the full pool
# intractable -- bound it here. greedy_entropy's pivoted-Cholesky step is only O(N)
# (no complement-precision solve), so it is left uncapped like select_logdet; capping
# it too previously reintroduced a random-250-subsample confound that made it look
# far worse than logdet under the `storage` currency despite being the same algorithm
# (see docs/rq4-cost-currency-mechanism.md).
_GREEDY_CAP = {"greedy_mi": 250}


def _method_order(prob, name, kind, key, kw, budget, rng_master):
    """Selection order a method produces for a cardinality ``budget`` (the same
    call the main table uses). Estimators return None (they do not acquire)."""
    if kind == "est":
        return None
    if key == "metaevaluator_full":
        return list(range(len(prob["X"])))
    fn = ACQUISITION_REGISTRY[key]
    P = len(prob["X"])
    cover_all = kw.get("cover_all")
    tmask = np.ones(P, bool) if cover_all else prob["target_mask"]
    infl = prob["influence"] if kw.get("influence") else None
    tref = None if cover_all else prob.get("X_target_ref")
    extra = {k: v for k, v in kw.items() if k not in ("cover_all", "influence")}
    if key in _GREEDY_CAP:                      # bound the cubic pivoted-Cholesky loop
        extra["cap"] = _GREEDY_CAP[key]
    return fn(prob["X"], prob["pair_model"], prob["pair_sample"], tmask, budget,
              rng=np.random.default_rng(rng_master.integers(1 << 30)),
              influence=infl, target_ref=tref, **extra)


def run_cost_budget(seeds, budget_frac):
    """Real-currency budgets: spend ``budget_frac`` of each currency's whole-pool
    cost instead of a fixed *count* of actions.

    Each acquisition method's (cost-agnostic) greedy order is computed once per
    seed; every currency then walks that order taking actions while its marginal
    cost fits the budget (``experiments/cost_models.greedy_fill``). Cheap-per-action
    picks buy more labels under a token/latency budget; the amortized ``storage``
    currency rewards methods that concentrate on few reference models. Reports
    unseen MAE (pp) per currency, plus the median #actions bought and the realized
    cost fraction."""
    acq = [(name, kind, key, kw) for name, kind, key, kw in METHODS
           if kind == "acq" and key != "metaevaluator_full"]
    cur_names = [c for c, _ in COST_CURRENCIES]
    mae = {name: {c: [] for c in cur_names} for name, *_ in acq}
    items = {name: {c: [] for c in cur_names} for name, *_ in acq}
    frac = {name: {c: [] for c in cur_names} for name, *_ in acq}
    full, ests = [], {n: [] for n, k, *_ in METHODS if k == "est"}

    for seed in seeds:
        prob = make_problem(seed)
        cm = make_cost_model(prob, seed)
        pm = np.asarray(prob["pair_model"])
        P = len(prob["X"])
        full.append(train_eval(prob, list(range(P)), seed))
        for n, kind, key, _ in METHODS:
            if kind == "est":
                ests[n].append(estimator_mae(prob, key))
        rng = np.random.default_rng(1000 + seed)
        pool_tot = {c: cm[c].pool_total(pm) for c in cur_names}
        for name, kind, key, kw in acq:
            order = _method_order(prob, name, kind, key, kw, min(P, _N_ORDER), rng)
            for c in cur_names:
                budget = budget_frac * pool_tot[c]
                kept, paid = greedy_fill(order, cm[c], budget, pm)
                mae[name][c].append(train_eval(prob, kept, seed))
                items[name][c].append(len(kept))
                frac[name][c].append(paid / pool_tot[c] if pool_tot[c] > 0 else 0.0)

    def _mci(v):
        v = np.asarray(v, float)
        ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
        return float(v.mean()), float(ci)

    full_mae = _mci(full)
    out = {"budget_frac": budget_frac, "seeds": len(seeds),
           "full_mae": {"mae": full_mae[0], "ci": full_mae[1]},
           "estimators": {n: _mci(v)[0] for n, v in ests.items()},
           "currencies": {}}
    # per-currency method records
    for ci_, (c, _unit) in enumerate(COST_CURRENCIES):
        out["currencies"][c] = {"unit": cm[c].unit, "methods": {}}
        for name, *_ in acq:
            m, mc = _mci(mae[name][c])
            out["currencies"][c]["methods"][name] = {
                "mae": m, "ci": mc,
                "median_actions": float(np.median(items[name][c])),
                "realized_frac": float(np.mean(frac[name][c]))}

    # ---- printed tables ----
    labels = [lbl for _, lbl in COST_CURRENCIES]
    print(f"\nReal-currency budget benchmark ({len(seeds)} seeds, budget = "
          f"{budget_frac*100:.0f}% of each currency's whole-pool cost).")
    print(f"MetaEvaluator (full, 100% of every currency) = {full_mae[0]:.2f} +/- {full_mae[1]:.2f} pp. "
          f"Estimators (budget-free): " +
          ", ".join(f"{n} {v:.2f}" for n, v in out['estimators'].items()) + "\n")
    width = max(len(n) for n, *_ in acq)
    print("Unseen MAE (pp); lower is better")
    print(f"{'Method'.ljust(width)}  " + "  ".join(f"{l:>8}" for l in labels))
    print("-" * (width + 2 + 10 * len(labels)))
    # rank rows by the count-currency MAE for a stable ordering
    order_rows = sorted(acq, key=lambda t: out["currencies"]["count"]["methods"][t[0]]["mae"])
    for name, *_ in order_rows:
        cells = []
        for c, _ in COST_CURRENCIES:
            r = out["currencies"][c]["methods"][name]
            cells.append(f"{r['mae']:8.2f}")
        print(f"{name.ljust(width)}  " + "  ".join(cells))
    print(f"\nMedian #actions bought under each budget (P = {P})")
    print(f"{'Method'.ljust(width)}  " + "  ".join(f"{l:>8}" for l in labels))
    print("-" * (width + 2 + 10 * len(labels)))
    for name, *_ in order_rows:
        cells = [f"{out['currencies'][c]['methods'][name]['median_actions']:8.0f}"
                 for c, _ in COST_CURRENCIES]
        print(f"{name.ljust(width)}  " + "  ".join(cells))
    return out


def run_fixed_model_pool(seeds, budget_frac):
    """Model-set-controlled fairness check on the RQ4 storage finding.

    Freezes the set of reference models that ``Submod. benchmark`` (select_logdet)
    buys under the amortized ``storage`` budget (docs/rq4-cost-currency-mechanism.md),
    then has every other acquisition method reselect its own pairs restricted to
    *only that model set*, spending the same number of actions Submod. benchmark
    bought. Each method is also run at that same action budget on the full,
    unrestricted pool (same paired RNG draw) as the fairness baseline. The
    fixed-vs-free delta isolates whether a method's storage-budget disadvantage
    comes from spreading across more models, or from how it picks pairs once the
    model set is fixed to match Submod. benchmark's.
    """
    acq = [(name, kind, key, kw) for name, kind, key, kw in METHODS
           if kind == "acq" and key not in ("metaevaluator_full", "submodular_benchmark")]
    sub_name, _sub_kind, sub_key, sub_kw = next(
        (n, k, ky, kw) for n, k, ky, kw in METHODS if ky == "submodular_benchmark")

    mae_fixed = {name: [] for name, *_ in acq}
    mae_free = {name: [] for name, *_ in acq}
    mae_source = []
    full = []
    budgets, n_models_sel = [], []

    for seed in seeds:
        prob = make_problem(seed)
        cm = make_cost_model(prob, seed)
        pm = np.asarray(prob["pair_model"])
        ps = np.asarray(prob["pair_sample"])
        P = len(prob["X"])
        full.append(train_eval(prob, list(range(P)), seed))
        rng = np.random.default_rng(1000 + seed)

        order = _method_order(prob, sub_name, "acq", sub_key, sub_kw, min(P, _N_ORDER), rng)
        storage = cm["storage"]
        budget_cost = budget_frac * storage.pool_total(pm)
        kept, _paid = greedy_fill(order, storage, budget_cost, pm)
        model_list = np.unique(pm[kept])
        k = len(kept)
        budgets.append(k)
        n_models_sel.append(len(model_list))
        mae_source.append(train_eval(prob, kept, seed))

        sub_idx = np.where(np.isin(pm, model_list))[0]
        X_sub, pm_sub, ps_sub = prob["X"][sub_idx], pm[sub_idx], ps[sub_idx]
        tmask_sub, infl_sub = prob["target_mask"][sub_idx], prob["influence"][sub_idx]
        k_local = min(k, len(sub_idx))

        for name, kind, key, kw in acq:
            fn = ACQUISITION_REGISTRY[key]
            cover_all = kw.get("cover_all")
            extra = {kk: vv for kk, vv in kw.items() if kk not in ("cover_all", "influence")}
            rseed = int(rng.integers(1 << 30))

            tmask_f = np.ones(len(sub_idx), bool) if cover_all else tmask_sub
            infl_f = infl_sub if kw.get("influence") else None
            tref_f = None if cover_all else prob.get("X_target_ref")
            sel_local = fn(X_sub, pm_sub, ps_sub, tmask_f, k_local,
                           rng=np.random.default_rng(rseed), influence=infl_f, target_ref=tref_f, **extra)
            sel_global = sub_idx[np.asarray(sel_local, int)]
            mae_fixed[name].append(train_eval(prob, sel_global, seed))

            tmask_full = np.ones(P, bool) if cover_all else prob["target_mask"]
            infl_full = prob["influence"] if kw.get("influence") else None
            tref_full = None if cover_all else prob.get("X_target_ref")
            sel_free = fn(prob["X"], pm, ps, tmask_full, k,
                         rng=np.random.default_rng(rseed), influence=infl_full, target_ref=tref_full, **extra)
            mae_free[name].append(train_eval(prob, sel_free, seed))

    def _mci(v):
        v = np.asarray(v, float)
        ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
        return float(v.mean()), float(ci)

    full_mae = _mci(full)
    src_mae = _mci(mae_source)
    out = {"budget_frac": budget_frac, "seeds": len(seeds),
           "median_budget_k": float(np.median(budgets)),
           "median_models_selected": float(np.median(n_models_sel)),
           "full_mae": {"mae": full_mae[0], "ci": full_mae[1]},
           "submodular_source": {"mae": src_mae[0], "ci": src_mae[1]},
           "methods": {}}
    for name, *_ in acq:
        fm, fc = _mci(mae_fixed[name])
        rm, rc = _mci(mae_free[name])
        out["methods"][name] = {
            "fixed": {"mae": fm, "ci": fc}, "free": {"mae": rm, "ci": rc},
            "delta": fm - rm}

    print(f"\nFixed-model-pool fairness check ({len(seeds)} seeds, storage budget = "
          f"{budget_frac*100:.0f}% of pool). Submod. benchmark bought a median of "
          f"{out['median_budget_k']:.0f} actions across {out['median_models_selected']:.0f} models "
          f"(source MAE {src_mae[0]:.2f} +/- {src_mae[1]:.2f} pp).\n")
    width = max(len(n) for n, *_ in acq)
    print(f"{'Method'.ljust(width)}  {'Free (unrestr.)':>16}  {'Fixed (same models)':>20}  {'Delta':>7}")
    print("-" * (width + 2 + 18 + 22 + 9))
    order_rows = sorted(acq, key=lambda t: out["methods"][t[0]]["delta"])
    for name, *_ in order_rows:
        r = out["methods"][name]
        print(f"{name.ljust(width)}  {r['free']['mae']:7.2f} +/- {r['free']['ci']:.2f}   "
              f"{r['fixed']['mae']:7.2f} +/- {r['fixed']['ci']:.2f}   {r['delta']:+6.2f}")
    return out


def run_kmeans_warmstart(seeds, budget_frac, n_samplesets_full=150,
                         cluster_grid=((30, 1), (15, 2), (10, 3), (6, 5))):
    """K-means-reduced subset universe for the meta-training warm start.

    Simulates: getting a model's accuracy on a sample-set is expensive (running
    every example in it), but a sample-set's shift descriptor is cheap (no model
    run needed). So before spending ANY labeling budget, first cheaply reduce the
    universe of ``n_samplesets_full`` candidate source sample-sets down to M
    k-means clusters x K representatives/cluster = K*M sample-sets -- only THOSE
    ever get run against the N reference models (K*M*N labelable triples, not
    |source|*N). Submodular acquisition (ActiveEval-Pair / Facility-location /
    Random) then spends the usual ``budget_frac`` *within* that reduced pool.

    Compares, at matched K*M sizes: (a) the full, unreduced source pool (upper
    reference -- most triples, most expensive), (b) k-means-reduced pools across
    a grid of (M, K), and (c) a random-subset-of-the-same-size control (same K*M
    sample-sets, chosen uniformly at random instead of via k-means) -- isolating
    whether clustering itself helps, versus just using fewer sample-sets. Also
    reports the measured k-means quantization radius (rho) that
    docs/kmeans-submodular-warmstart.md's stability bound is stated in terms of.
    """
    configs = [("Full source pool", dict())]
    for m, k in cluster_grid:
        configs.append((f"K-means M={m} K={k} (KM={m*k})", dict(n_clusters=m, reps_per_cluster=k)))
        configs.append((f"Random subset (n={m*k})", dict(random_subset_size=m * k)))
    methods = ["ActiveEval-Pair", "Facility-location", "Random"]
    rows = {name: {meth: [] for meth in methods} for name, _ in configs}
    triples = {name: [] for name, _ in configs}
    radius = {name: [] for name, _ in configs}

    for seed in seeds:
        for name, kw in configs:
            prob = make_problem(seed, n_samplesets=n_samplesets_full, **kw)
            P = len(prob["X"])
            triples[name].append(P)
            if prob.get("kmeans_diag"):
                radius[name].append(prob["kmeans_diag"]["quantization_radius_max"])
            budget = max(1, int(budget_frac * P))
            rows[name]["ActiveEval-Pair"].append(_acq(prob, "activeeval_pair", budget, seed, influence=True))
            rows[name]["Facility-location"].append(_acq(prob, "facility_location", budget, seed, cover_all=True))
            rows[name]["Random"].append(_acq(prob, "random", budget, seed))

    def _mci(v):
        v = np.asarray(v, float)
        ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
        return float(v.mean()), float(ci)

    out = {"n_samplesets_full": n_samplesets_full, "budget_frac": budget_frac,
           "seeds": len(seeds), "configs": {}}
    print(f"\nK-means subset-universe reduction ({len(seeds)} seeds, source universe = "
          f"{n_samplesets_full - n_samplesets_full // 4} sample-sets, acquisition budget = "
          f"{budget_frac*100:.0f}% of the (reduced) labelable pool). Unseen MAE (pp); lower is better.\n")
    width = max(len(n) for n, _ in configs)
    print(f"{'Config'.ljust(width)}  {'#triples':>9}  {'rho(max)':>9}  "
          + "  ".join(f"{m:>16}" for m in methods))
    print("-" * (width + 13 + 22 + 18 * len(methods)))
    for name, _ in configs:
        entry = {"n_triples_labelable": float(np.mean(triples[name]))}
        if radius[name]:
            entry["quantization_radius_max_mean"] = float(np.mean(radius[name]))
        rho_str = f"{entry['quantization_radius_max_mean']:9.3f}" if radius[name] else "     --  "
        cells = []
        for meth in methods:
            m, ci = _mci(rows[name][meth])
            entry[meth] = {"mae": m, "ci": ci}
            cells.append(f"{m:6.2f}+/-{ci:4.2f}")
        out["configs"][name] = entry
        print(f"{name.ljust(width)}  {entry['n_triples_labelable']:9.0f}  {rho_str}  "
              + "  ".join(f"{c:>16}" for c in cells))
    return out


def run_distance_formula_comparison(seeds, budget_frac, n_samplesets_full=150, budget_K=30,
                                    n_points_per_subset=20,
                                    formulas=("kernel_mean", "sliced_wasserstein", "hausdorff", "sum"),
                                    include_kmeans=False):
    """Alternative Stage-1 warm-start: submodular facility-location coreset
    selection of K representative source sample-sets ("best represent the
    meta-dataset", no target-awareness), under four distance formulas
    between sample-set point clouds (experiments/subset_distances.py):
    kernel_mean (MMD), sliced_wasserstein, hausdorff, and their standardized
    sum. Compared against the k-means-clustering reduction and a matched-size
    random-subset control from run_kmeans_warmstart, at the same K.

    Reports unseen-model MAE plus wall-clock cost of the Stage-1 reduction
    itself (cloud construction, pairwise distance matrix, greedy selection)
    per formula -- this is a pure CPU/NumPy benchmark with no LLM calls, so
    there is no token cost to report; wall-clock seconds is the real,
    reportable resource cost here.
    """
    formulas = list(formulas)
    configs = [(f"Distance: {f}", f) for f in formulas]
    if include_kmeans:
        configs += [("K-means (M=K, reps=1)", "__kmeans__")]
    configs += [("Full source pool", None), ("Random subset", None)]
    methods = ["ActiveEval-Pair", "Facility-location", "Random"]
    rows = {name: {meth: [] for meth in methods} for name, _ in configs}
    triples = {name: [] for name, _ in configs}
    timing = {f: [] for f in formulas}
    # Full end-to-end wall-clock per config: Stage-1 selection (already in
    # `timing` for distance formulas), problem construction, and Stage-2 --
    # each of the 3 acquisition calls (greedy selection over the reduced
    # pool + train_eval's 250-epoch MLP training, which dominates total
    # runtime -- Stage-1 is sub-3-seconds even for the priciest formula,
    # Stage-2 is where the real minutes go).
    full_timing = {name: {"stage1": [], "make_problem": [], "acq": {m: [] for m in methods}, "total": []}
                   for name, _ in configs}

    for seed in seeds:
        for name, formula in configs:
            t_config0 = time.time()
            t_stage1 = 0.0
            if formula == "__kmeans__":
                t0 = time.time()
                prob = make_problem(seed, n_samplesets=n_samplesets_full,
                                    n_clusters=budget_K, reps_per_cluster=1)
                t_stage1 = time.time() - t0  # k-means clustering is part of make_problem here
            elif formula is not None:
                t0 = time.time()
                sel_idx, diag = select_via_distance_formula(
                    seed, formula, n_train_models=60, n_unseen_models=8,
                    n_samplesets_full=n_samplesets_full, d_lat=6, budget_K=budget_K,
                    n_points_per_subset=n_points_per_subset)
                t_stage1 = time.time() - t0
                timing[formula].append(diag)
                prob = make_problem(seed, n_samplesets=n_samplesets_full, fixed_source_sets=sel_idx)
            elif name == "Full source pool":
                prob = make_problem(seed, n_samplesets=n_samplesets_full)
            else:  # matched-size random control
                prob = make_problem(seed, n_samplesets=n_samplesets_full, random_subset_size=budget_K)
            t_after_problem = time.time()
            full_timing[name]["stage1"].append(t_stage1)
            full_timing[name]["make_problem"].append(t_after_problem - t_config0 - t_stage1)

            P = len(prob["X"])
            triples[name].append(P)
            budget = max(1, int(budget_frac * P))

            t_acq0 = time.time()
            rows[name]["ActiveEval-Pair"].append(_acq(prob, "activeeval_pair", budget, seed, influence=True))
            full_timing[name]["acq"]["ActiveEval-Pair"].append(time.time() - t_acq0)

            t_acq0 = time.time()
            rows[name]["Facility-location"].append(_acq(prob, "facility_location", budget, seed, cover_all=True))
            full_timing[name]["acq"]["Facility-location"].append(time.time() - t_acq0)

            t_acq0 = time.time()
            rows[name]["Random"].append(_acq(prob, "random", budget, seed))
            full_timing[name]["acq"]["Random"].append(time.time() - t_acq0)

            full_timing[name]["total"].append(time.time() - t_config0)

    def _mci(v):
        v = np.asarray(v, float)
        ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
        return float(v.mean()), float(ci)

    out = {"n_samplesets_full": n_samplesets_full, "budget_K": budget_K,
           "budget_frac": budget_frac, "seeds": len(seeds), "configs": {}, "timing": {}}
    print(f"\nDistance-formula coreset comparison ({len(seeds)} seeds, source universe "
          f"~{n_samplesets_full - n_samplesets_full//4} sample-sets, K={budget_K} representatives, "
          f"acquisition budget = {budget_frac*100:.0f}% of the reduced pool). Unseen MAE (pp).\n")
    width = max(len(n) for n, _ in configs)
    print(f"{'Config'.ljust(width)}  {'#triples':>9}  " + "  ".join(f"{m:>16}" for m in methods))
    print("-" * (width + 11 + 18 * len(methods)))
    for name, _ in configs:
        entry = {"n_triples_labelable": float(np.mean(triples[name]))}
        cells = []
        for meth in methods:
            m, ci = _mci(rows[name][meth])
            entry[meth] = {"mae": m, "ci": ci}
            cells.append(f"{m:6.2f}+/-{ci:4.2f}")
        out["configs"][name] = entry
        print(f"{name.ljust(width)}  {entry['n_triples_labelable']:9.0f}  "
              + "  ".join(f"{c:>16}" for c in cells))

    print(f"\nStage-1 reduction wall-clock cost (seconds, mean over {len(seeds)} seeds; no LLM calls "
          f"in this synthetic benchmark, so there is no token cost to report):")
    print(f"{'Formula':<20} {'build_clouds':>13} {'dist_matrix':>13} {'greedy_select':>14} {'total':>8}")
    for f in formulas:
        keys = ["seconds_build_clouds", "seconds_distance_matrix", "seconds_greedy_select", "seconds_total"]
        means = {k: float(np.mean([d[k] for d in timing[f]])) for k in keys}
        out["timing"][f] = means
        print(f"{f:<20} {means['seconds_build_clouds']:13.3f} {means['seconds_distance_matrix']:13.3f} "
              f"{means['seconds_greedy_select']:14.3f} {means['seconds_total']:8.3f}")

    print(f"\nFull end-to-end wall-clock cost per config (seconds, mean over {len(seeds)} seeds). "
          f"'acq' columns = one acquisition-method call each, dominated by train_eval's 250-epoch "
          f"MLP training (torch/Adam, CPU) -- this, not Stage-1 selection, is why full runs take "
          f"minutes: Stage-1 tops out under 3s/config (table above) while a single train_eval call "
          f"runs 250 full-batch gradient steps over a fresh model. No LLM calls anywhere in this "
          f"module -- compute tokens: not applicable, wall-clock seconds is the real cost unit.")
    print(f"{'Config':<32} {'stage1':>8} {'mk_prob':>8} {'acq:AEval':>10} {'acq:FacLoc':>11} "
          f"{'acq:Rand':>9} {'total':>8}")
    out["full_timing"] = {}
    for name, _ in configs:
        ft = full_timing[name]
        row_out = {
            "stage1_s": float(np.mean(ft["stage1"])),
            "make_problem_s": float(np.mean(ft["make_problem"])),
            "acq_activeeval_pair_s": float(np.mean(ft["acq"]["ActiveEval-Pair"])),
            "acq_facility_location_s": float(np.mean(ft["acq"]["Facility-location"])),
            "acq_random_s": float(np.mean(ft["acq"]["Random"])),
            "total_s": float(np.mean(ft["total"])),
            "total_s_all_seeds": float(np.sum(ft["total"])),
        }
        out["full_timing"][name] = row_out
        print(f"{name:<32} {row_out['stage1_s']:8.3f} {row_out['make_problem_s']:8.3f} "
              f"{row_out['acq_activeeval_pair_s']:10.2f} {row_out['acq_facility_location_s']:11.2f} "
              f"{row_out['acq_random_s']:9.2f} {row_out['total_s']:8.2f}")
    grand_total = sum(np.sum(full_timing[name]["total"]) for name, _ in configs)
    print(f"\nGrand total wall-clock across all {len(configs)} configs x {len(seeds)} seeds: "
          f"{grand_total:.1f}s ({grand_total/60:.1f} min).")
    out["grand_total_seconds"] = float(grand_total)
    return out


def run_ablation(seeds, budget_frac):
    """RQ5 ablation. This offline benchmark robustly isolates the component the paper
    finds most important---target-aware narrowing---against a no-structure random
    floor; the remaining components (influence weighting, submodular MI, knapsack
    budgeting, the uncertainty head) are ablated in the full Text2SQL pipeline."""
    # (key, kwargs) per variant
    variants = {
        "ActiveEval-Pair (full)":      ("activeeval_pair", dict(influence=True)),
        "  - target-aware narrowing":  ("activeeval_pair", dict(influence=True, cover_all=True)),
        "  - all structure (Random)":  ("random", dict()),
    }
    res = {name: [] for name in variants}
    for seed in seeds:
        prob = make_problem(seed)
        b = max(1, int(budget_frac * len(prob["X"])))
        for name, (key, kw) in variants.items():
            res[name].append(_acq(prob, key, b, seed, **kw))
    out = {}
    print(f"\nRQ5 ablation on ActiveEval-Pair ({len(seeds)} seeds, "
          f"{budget_frac*100:.0f}% budget). Unseen MAE (pp); lower is better.\n")
    width = max(len(k) for k in variants)
    for name in variants:
        v = np.asarray(res[name]); m = float(v.mean())
        ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
        out[name] = {"mae": m, "ci": ci}
        print(f"{name.ljust(width)}   {m:5.2f} +/- {ci:.2f}")
    return out


def run_threshold_budget(seeds, max_budget_frac, thresholds):
    """Adaptive-budget check: given a hard cap (``max_budget_frac``), does
    stopping a greedy acquisition loop early -- once its marginal gain falls
    below ``threshold`` of the first pick's gain -- find a cheaper subset
    (5/10/15% spend) that matches or beats spending the full cap?

    Applies to every method whose greedy loop already tracks a monotonically
    non-increasing per-step gain (facility-location coverage, log-det
    diversity, greedy MI's pivoted-Cholesky conditional variance, and the
    ActiveEval variants built from them). Because each is a greedy *prefix*
    selection, stopping early at a given threshold is exactly equivalent to
    having fixed the budget at the realized spend from the start -- this sweep
    just locates that spend automatically per seed instead of grid-searching
    budget fractions by hand (``threshold=0`` reproduces the full-cap,
    no-early-stop selection, i.e. the method's normal row in the main table).
    """
    variants = [
        ("Facility-location", "facility_location", dict(cover_all=True)),
        ("Submod. benchmark", "submodular_benchmark", dict()),
        ("Greedy MI (Alg. 2)", "greedy_mi", dict(cap=250)),
        ("ActiveEval-S", "activeeval_s", dict(influence=True)),
        ("ActiveEval-S+M", "activeeval_sm", dict(influence=True)),
        ("ActiveEval-Pair", "activeeval_pair", dict(influence=True)),
    ]
    mae = {name: {t: [] for t in thresholds} for name, *_ in variants}
    frac = {name: {t: [] for t in thresholds} for name, *_ in variants}
    full = []
    for seed in seeds:
        prob = make_problem(seed)
        P = len(prob["X"])
        budget = max(1, int(max_budget_frac * P))
        full.append(train_eval(prob, list(range(P)), seed))
        rng = np.random.default_rng(1000 + seed)
        for name, key, kw in variants:
            fn = ACQUISITION_REGISTRY[key]
            cover_all = kw.get("cover_all")
            tmask = np.ones(P, bool) if cover_all else prob["target_mask"]
            infl = prob["influence"] if kw.get("influence") else None
            tref = None if cover_all else prob.get("X_target_ref")
            extra = {k: v for k, v in kw.items() if k not in ("cover_all", "influence")}
            for th in thresholds:
                gt = None if th <= 0 else th
                sel = fn(prob["X"], prob["pair_model"], prob["pair_sample"], tmask, budget,
                         rng=np.random.default_rng(rng.integers(1 << 30)),
                         influence=infl, target_ref=tref, gain_threshold=gt, **extra)
                mae[name][th].append(train_eval(prob, sel, seed))
                frac[name][th].append(len(sel) / P)

    def _mci(v):
        v = np.asarray(v, float)
        ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
        return float(v.mean()), float(ci)

    full_mae = _mci(full)
    out = {"max_budget_frac": max_budget_frac, "seeds": len(seeds),
           "full_mae": {"mae": full_mae[0], "ci": full_mae[1]}, "methods": {}}

    print(f"\nAdaptive (threshold-stopped) budget under a {max_budget_frac*100:.0f}% cap "
          f"({len(seeds)} seeds). MetaEvaluator (full, 100%) = {full_mae[0]:.2f} +/- {full_mae[1]:.2f} pp.\n"
          f"Threshold = stop once the best remaining pick's marginal gain falls below this "
          f"fraction of the first pick's gain (0 = no early stop, spends the full cap).\n")

    for name, *_ in variants:
        base_mae, base_ci = _mci(mae[name][0.0])
        base_spend, _ = _mci(frac[name][0.0])
        out["methods"][name] = {"base_mae": base_mae, "base_ci": base_ci, "thresholds": {}}
        print(f"{name}  (0 threshold baseline: {base_mae:.2f} +/- {base_ci:.2f} pp @ "
              f"{base_spend*100:.1f}% spend)")
        print(f"  {'threshold':>10}  {'spend %':>8}  {'MAE (pp)':>14}  note")
        best = None
        for th in thresholds:
            m, ci = _mci(mae[name][th])
            f_mean, _ = _mci(frac[name][th])
            spend_pct = f_mean * 100
            out["methods"][name]["thresholds"][th] = {"mae": m, "ci": ci, "spend_pct": spend_pct}
            note = ""
            if th > 0 and spend_pct < max_budget_frac * 100 - 0.5 and m <= base_mae + base_ci:
                note = "<= under-spends, matches/beats full cap"
                if best is None or spend_pct < best[0]:
                    best = (spend_pct, th, m)
            print(f"  {th:10.4g}  {spend_pct:8.1f}  {m:6.2f} +/- {ci:4.2f}  {note}")
        if best:
            print(f"  --> cheapest matching threshold: {best[1]:.4g} ({best[0]:.1f}% spend, {best[2]:.2f} pp)")
        else:
            print("  --> no threshold under-spent while matching the full-cap MAE")
        print()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["main", "sweep", "ablation", "entropy_mi_sweep", "noise_sweep", "cost_budget", "fixed_model_pool", "threshold_budget", "kmeans_warmstart", "distance_formula_comparison"], default="main")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budget-frac", type=float, default=0.15)
    ap.add_argument("--out", default="outputs/acquisition_benchmark.json")
    args = ap.parse_args()
    seeds = list(range(args.seeds))

    if args.mode == "sweep":
        res = run_sweep(seeds, [0.05, 0.10, 0.15, 0.20, 0.30, 0.50])
        out_path = "outputs/acquisition_sweep.json"
    elif args.mode == "ablation":
        res = run_ablation(seeds, args.budget_frac)
        out_path = "outputs/acquisition_ablation.json"
    elif args.mode == "noise_sweep":
        res = run_noise_sweep(seeds, args.budget_frac, [1.0, 1.5, 2.0, 3.0, 4.4, 6.0])
        out_path = "outputs/noise_sensitivity.json"
    elif args.mode == "entropy_mi_sweep":
        res = run_entropy_mi_sweep(seeds, [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
        out_path = "outputs/entropy_mi_sweep.json"
    elif args.mode == "cost_budget":
        res = run_cost_budget(seeds, args.budget_frac)
        out_path = "outputs/cost_budget_benchmark.json"
    elif args.mode == "fixed_model_pool":
        res = run_fixed_model_pool(seeds, args.budget_frac)
        out_path = "outputs/fixed_model_pool_benchmark.json"
    elif args.mode == "threshold_budget":
        res = run_threshold_budget(seeds, args.budget_frac,
                                   [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3])
        out_path = "outputs/threshold_budget_benchmark.json"
    elif args.mode == "kmeans_warmstart":
        res = run_kmeans_warmstart(seeds, args.budget_frac)
        out_path = "outputs/kmeans_warmstart_benchmark.json"
    elif args.mode == "distance_formula_comparison":
        res = run_distance_formula_comparison(seeds, args.budget_frac)
        out_path = "outputs/distance_formula_comparison.json"
    else:
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
        out_path = args.out

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()

# RQ4 mechanism analysis — why method rankings shift across cost currencies

Companion to the "Real-currency budgets (RQ4)" table in [README.md](../README.md). The
README states *what* happens (the `storage` budget inverts the ranking); this note
explains *why*, grounded in the selection code and one reproducible measurement.

> **Update.** Two implementation confounds previously distorted the `Greedy entropy
> (Alg. 1)` / `Greedy MI (Alg. 2)` rows specifically: an unmatched regularization
> constant against `select_logdet`, and an unnecessary candidate-pool cap in the
> `cost_budget` driver. Both are now fixed ([baselines/_core.py:364](../baselines/_core.py#L364),
> [:394](../baselines/_core.py#L394), [experiments/run_acquisition_benchmark.py:413](../experiments/run_acquisition_benchmark.py#L413)).
> The corrected numbers are folded into this note below; `README.md`'s RQ4 tables still
> show the pre-fix numbers for these two rows and need a refresh.

## The shared mechanism: selection is cost-agnostic

Every method ranks the candidate pool exactly once, from shift descriptors alone
(`_method_order`, [experiments/run_acquisition_benchmark.py:412-431](../experiments/run_acquisition_benchmark.py#L412-L431)).
The ranking never sees tokens, seconds, or GB. A currency only decides how far its
budget reaches into that fixed order, via `greedy_fill`
([experiments/cost_models.py:78-104](../experiments/cost_models.py#L78-L104)): walk the
order, keep an action if its *marginal* cost still fits, skip it otherwise. So every
difference between the six per-currency tables comes down to which slice of the pool
each method's fixed ranking happens to prefer, and how that slice is priced.

## Group 1 — `input_tok` / `output_tok` / `latency` / `memory`: mostly noise

These four currencies are **additive** — cost accumulates per action, with no reuse
discount. Because the ranking is cost-agnostic, it draws a roughly representative mix
of cheap/expensive actions, so 15% of any additive total buys close to 15% of the
actions (~250-290) regardless of the unit — the ordering barely moves from `count`.

Where the leaderboard does reshuffle (Submod. benchmark topping `output_tok`,
k-center topping `latency`) is **not a structural effect**: the confidence intervals
of the top cluster overlap almost completely (e.g. `latency`: k-center 4.38 ± 0.65 vs.
ActiveEval-Pair 4.55 ± 0.65). Treat these as seed-to-seed noise among a group of
methods that are genuinely close, not evidence that one method is "built for" that
currency.

## Group 2 — `storage`: a real inversion, with a concrete cause

`storage` is qualitatively different: `checkpoint_gb` is paid **once per unique model**
touched, not per action
([experiments/cost_models.py:145](../experiments/cost_models.py#L145),
[:154-155](../experiments/cost_models.py#L154-L155)). A method that revisits the same
few models pays the checkpoint charge fewer times and buys more actions for the same
GB budget; a method that spreads its picks across many distinct models pays it over
and over.

**Measured** (single seed, budget = 15% of pool storage cost): actions kept before the
selection order touches a new model —

```
ActiveEval-Pair       ~7.3 actions/model  (9 models for 66 kept actions)
ActiveEval-S          ~7.7 actions/model  (9 models for 69 kept actions)
Submod. benchmark     ~14.0 actions/model (25 models for 350 kept actions)
Bayesian opt. design  ~17.6 actions/model (15 models for 264 kept actions)
```

The cause is in the selection objective itself:

- `select_activeeval_sample` calls `select_facility` with the explicit design note
  "**the model axis is left uniform**"
  ([baselines/_core.py:411-414](../baselines/_core.py#L411-L414)) — its objective is to
  cover the unlabeled target region `Q` with the cleanest, most target-similar source
  pairs, indifferent to which of the 60 reference models produced them. It therefore
  jumps to a new model almost every 1-2 picks.
- `select_logdet` (Submod. benchmark) and `select_bayesian_design` maximize a
  whole-pool diversity / D-optimal criterion with no target restriction. In this
  problem instance that criterion exhausts several sample-sets of the *same* model
  before moving to the next one, so its checkpoint cost amortizes far better.

Net effect: ActiveEval-Pair pays the checkpoint charge roughly twice as often per kept
action as Submod./Bayesian. The 15% GB budget runs out after only ~91 actions
(5-seed median from the README table) instead of ~251, erasing the information
advantage that target-aware coverage otherwise provides — hence ActiveEval-Pair's drop
to near-worst (6.81 pp) under `storage` despite leading almost every other table.

## Greedy entropy (Alg. 1) / greedy MI (Alg. 2) — two confounds, now removed

The previous version of this note claimed these two "restrict to the same
`target_mask`-aligned universe as ActiveEval" and "stay mid-pack everywhere,
including under `storage`." Neither claim survives closer inspection; both were
artifacts of how the baselines were configured, not properties of the entropy/MI
selection criteria.

### Confound 1 — unmatched regularization vs. `select_logdet`

`select_logdet` and `select_greedy_entropy` are **the same pivoted-Cholesky greedy
algorithm** (greedy log-det / max-entropy maximization; compare the rank-1 Cholesky
update in `select_logdet` at [baselines/_core.py:152-168](../baselines/_core.py#L152-L168)
against `_pivoted_cholesky_entropy` at [:300-313](../baselines/_core.py#L300-L313) — same
recursion). They only differed in how strongly the kernel diagonal is regularized:
`select_logdet` optimizes `log det(I + K/sigma^2)` (offset = 1.0 by default), while
`select_greedy_entropy`/`select_greedy_mi` used a near-zero jitter (`+1e-6`) — a much
weaker prior that made the greedy pivot more sensitive to noise in the RBF-bandwidth
estimate.

**Ablation** (main table, 5 seeds, budget = 15% of the ~1800-pair pool, both already
searching the full pool with `cap=None`):

| Method | sigma (jitter) | MAE (pp) |
| --- | --- | --- |
| Submod. benchmark (logdet) | sigma=1.0 | 4.68 ± 0.29 |
| Greedy entropy | sigma=None (old default, `+1e-6`) | 5.12 ± 0.14 |
| **Greedy entropy** | **sigma=1.0** | **4.68 ± 0.29** |
| Greedy MI | sigma=None (old default, `+1e-6`) | 5.82 ± 0.50 |
| Greedy MI | sigma=1.0 | 5.10 ± 0.64 |

Matching the regularizer makes `Greedy entropy` land on **exactly** `select_logdet`'s
number — direct confirmation that the two are the same algorithm once compared under
the same assumptions. `Greedy MI`'s genuinely different criterion (it scores by
mutual information with the *unselected* complement, via
`_complement_precision_diag`, [baselines/_core.py:317-328](../baselines/_core.py#L317-L328))
closes most but not all of the gap — the residual ~0.4pp is real and attributable to
the MI objective itself, not to regularization. `sigma` now defaults to `1.0` for both
functions ([baselines/_core.py:364](../baselines/_core.py#L364),
[:394](../baselines/_core.py#L394)); pass `sigma=None` to recover the old unregularized
behavior.

### Confound 2 — an unnecessary candidate-pool cap in `cost_budget`

Separately, the `cost_budget` driver's `_GREEDY_CAP` dict capped **both** `greedy_mi`
*and* `greedy_entropy` to a random 250-pair subsample of the ~1800-pair pool before
each greedy run, on the reasoning that "greedy MI/entropy are cubic in their search
universe." That reasoning only holds for MI: `_pivoted_cholesky_mi` recomputes the
unselected complement's precision matrix every step (genuinely cubic in pool size),
but `_pivoted_cholesky_entropy` has no such step — it is `O(N)` per pick, the same
complexity class as `select_logdet`, which was already running uncapped on the full
pool. Capping `greedy_entropy` to 250 candidates reintroduced, *only inside the
`cost_budget` experiment*, exactly the pool-size confound that the main table's
`cap=None` setting had already avoided — and it hit hardest under `storage`, because a
random 250-pair subsample doesn't preserve the "concentrate on few models" structure
that the full-pool greedy order finds.

**Ablation** (storage currency only, 5 seeds, budget = 15% of storage pool cost):

| Method | MAE (pp) | median #models touched |
| --- | --- | --- |
| Submod. benchmark (logdet, no cap) | 5.04 ± 1.01 | 16 |
| Greedy entropy, cap=250 (old `cost_budget` setting) | 5.90 ± 0.92 | 16 |
| **Greedy entropy, cap=None** | **5.02 ± 0.91** | **16** |

Same median model count either way, but the cap=250 run's order is built from an
arbitrary random slice, so it doesn't sequence *which* actions on those models come
first as well as the full-pool greedy order does — worse MAE for the same amount of
"concentration." `_GREEDY_CAP` now only contains `"greedy_mi": 250`
([experiments/run_acquisition_benchmark.py:413](../experiments/run_acquisition_benchmark.py#L413));
`greedy_entropy` runs uncapped like `select_logdet` everywhere.

### Corrected `storage`-currency numbers (5 seeds, sigma=1.0, entropy uncapped)

| Method | MAE (pp) | median #actions | | old (pre-fix) MAE | old #actions |
| --- | --- | --- | --- | --- | --- |
| Submod. benchmark | 4.99 ± 0.79 | 251 | | 4.99 ± 0.79 | 251 |
| **Greedy entropy (Alg. 1)** | **5.07 ± 0.99** | **249** | | 6.20 ± 1.21 | 106 |
| Bayesian opt. design | 5.22 ± 0.68 | 232 | | 5.22 ± 0.68 | 232 |
| Greedy MI (Alg. 2) | 6.02 ± 0.97 | 157 | | 6.54 ± 1.12 | 113 |

Once both confounds are removed, `Greedy entropy` doesn't just "not collapse" under
`storage` — it lands in the *same tier as Submod. benchmark* (4.99 vs. 5.07, 251 vs.
249 actions bought), exactly as the sigma ablation above predicts for two
implementations of the same algorithm under the same regularization and the same pool
access. `Greedy MI` improves too (6.54 → 6.02) but, consistent with the sigma ablation,
still trails — the residual gap is the MI criterion itself, run here with its genuinely
necessary `cap=250` (its cubic complement-precision step makes the full ~1800-pair
pool intractable per seed).

## Why this matters for the rest of the RQ4 table

Neither confound touches `select_logdet`, `select_facility` (ActiveEval-S/S+M/Pair),
or `select_bayesian_design` — their numbers in this note and in the README are
unaffected. `README.md`'s per-currency tables (`count`, `input_tok`, `output_tok`,
`latency`, `memory`, `storage`) still report the pre-fix `Greedy entropy`/`Greedy MI`
rows and the note "greedy MI/entropy capped at 250 via `_GREEDY_CAP`" — both need
updating to match the corrected code and the numbers above.

## Takeaway

Only `storage` reflects a genuine mechanism difference; the other four currencies
mostly preserve the `count` ranking within noise. The `storage` collapse is a concrete
illustration of a limitation already flagged in the README: current ActiveEval
selection is *count*-optimal, not *cost*-optimal — it has no term that trades target
coverage off against the amortized checkpoint cost of the models it touches. The fix is
a cost-aware / knapsack variant that threads the `cost_fn` already available in
[active_evaluator/active_selection.py](../active_evaluator/active_selection.py) through
the greedy coverage objective, rather than ignoring cost during selection and only
discovering it at spend time.

Separately, the `Greedy entropy` correction is a reminder to benchmark every baseline
under matched hyperparameters (regularization, candidate-pool scope) before attributing
a performance gap to "the algorithm" — here, two supposedly distinct baselines
(`submodular_benchmark` and `greedy_entropy`) turned out to be the same algorithm
wearing different regularization defaults and, in one experiment mode, different (and
in one case gratuitous) pool restrictions.

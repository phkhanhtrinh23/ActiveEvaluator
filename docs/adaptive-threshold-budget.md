# Adaptive (threshold-stopped) budget

Motivating question (user-proposed): given a hard budget cap — say 20% — is there
an acquisition method that **doesn't need the whole cap** (spends only 5/10/15%)
and still performs best? Concretely: can a greedy acquisition loop be given a
**stopping rule** ("stop once the marginal entropy/gain/distance drops below X")
instead of always being forced to spend every last unit of the cap?

Run: `python -m experiments.run_acquisition_benchmark --mode threshold_budget --seeds 5 --budget-frac 0.20`
Code: [experiments/run_acquisition_benchmark.py](../experiments/run_acquisition_benchmark.py) (`run_threshold_budget`)

## Mechanism

Every greedy acquisition method already tracks a per-step **marginal gain** (how
much the best remaining candidate still adds): `select_facility`'s coverage gain,
`select_logdet`'s log-det/diversity gain, and the pivoted-Cholesky conditional
variance behind `select_greedy_entropy`/`select_greedy_mi`. A new optional
`gain_threshold` kwarg was added to each ([baselines/_core.py](../baselines/_core.py))
and threaded through the composed `ActiveEval-S`/`-S+M`/`-Pair` selectors. When
set, the loop stops as soon as:

```
gain_at_step_t  <  gain_threshold × gain_at_step_0
```

i.e. once the best remaining pick's gain has decayed below `gain_threshold` of the
*first* pick's gain — and, critically, the selection is **not** padded back up to
the budget cap (the pre-existing behavior for a saturated greedy loop). So a
method can now genuinely return fewer picks than the cap allows.
`gain_threshold=None` (the `0` column below) reproduces the exact prior
behavior — full-cap spend, used as the baseline.

Because every one of these methods is a **greedy prefix selector** (the order at
step *k* is always a prefix of the order at step *k+1*), stopping early at a given
threshold is mathematically identical to having fixed the budget at the realized
spend from the start. The threshold sweep is just a way to *locate* that natural
stopping point automatically, instead of grid-searching budget fractions by hand.

Threshold grid is log-spaced (`0, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3`)
because the marginal gain of these kernels decays over several orders of
magnitude within the first few picks — a linear grid (e.g. `0.01, 0.05, 0.1, ...`)
collapses every method straight to near-zero spend and misses the transition
region entirely.

## Results (5 seeds, 20% budget cap)

Summary — baseline (full 20% spend, `gain_threshold=None`) vs. the cheapest
threshold that still matches the baseline MAE within its CI:

| Method | Baseline (0, full 20% cap) | Cheapest match found | Verdict |
| --- | --- | --- | --- |
| **ActiveEval-S** | 4.02 ± 0.49 pp @ 20% | **4.33 ± 0.61 pp @ 12.3%** (`gain_threshold=1e-4`) | **Genuine free lunch** — 39% less budget, MAE within the full-cap CI. |
| Greedy MI (Alg. 2) | 5.10 ± 0.37 pp @ 20% | 4.74 ± 0.47 pp @ 13.9% (`gain_threshold=1e-4`) | Looks free, but is a search-universe **`cap=250` artifact**, not real gain decay — see caveat below. |
| Submod. benchmark | 4.67 ± 0.38 pp @ 20% | — never under-spends | Log-det diversity gain never decays enough in this threshold range; always spends the full cap. |
| ActiveEval-Pair | 4.79 ± 0.87 pp @ 20% | — never under-spends | Same — its diversity phase always consumes whatever budget the coverage phase leaves it. |
| Facility-location | 4.94 ± 0.42 pp @ 20% | — decays too fast | Collapses to ≤6.5% spend with much worse MAE at the smallest tested threshold; no usable middle ground in this grid. |
| ActiveEval-S+M | 4.61 ± 0.32 pp @ 20% | — decays too fast | Same pattern as Facility-location. |

Full per-threshold sweep (MAE pp @ realized spend %):

#### Facility-location (base 4.94 ± 0.42 @ 20%)

| threshold | 0 | 1e-4 | 3e-4 | 1e-3 | 3e-3 | 0.01 | 0.03 | 0.1 | 0.3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MAE (pp) | 4.94±0.42 | 4.89±0.48 | 4.79±0.31 | 6.52±1.47 | 7.44±1.84 | 9.20±1.71 | 11.21±2.23 | 9.54±2.34 | 9.54±2.34 |
| spend %   | 20.0 | 20.0 | 20.0 | 6.5 | 2.0 | 0.5 | 0.1 | 0.1 | 0.1 |

#### Submod. benchmark (base 4.67 ± 0.38 @ 20%)

| threshold | 0 | 1e-4 | 3e-4 | 1e-3 | 3e-3 | 0.01 | 0.03 | 0.1 | 0.3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MAE (pp) | 4.67±0.38 | 4.70±0.45 | 4.50±0.31 | 4.67±0.29 | 4.76±0.33 | 4.62±0.34 | 4.55±0.41 | 4.60±0.42 | 4.70±0.44 |
| spend %   | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 |

#### Greedy MI (Alg. 2) (base 5.10 ± 0.37 @ 20%) — see cap-artifact caveat

| threshold | 0 | 1e-4 | 3e-4 | 1e-3 | 3e-3 | 0.01 | 0.03 | 0.1 | 0.3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MAE (pp) | 5.10±0.37 | 4.74±0.47 | 4.91±0.61 | 5.62±1.65 | 5.79±0.95 | 5.73±1.64 | 5.62±1.52 | 5.79±0.98 | 5.39±0.79 |
| spend %   | 20.0 | 13.9 | 13.9 | 13.9 | 13.9 | 13.9 | 13.9 | 13.9 | 13.9 |

#### ActiveEval-S (base 4.02 ± 0.49 @ 20%)

| threshold | 0 | 1e-4 | 3e-4 | 1e-3 | 3e-3 | 0.01 | 0.03 | 0.1 | 0.3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MAE (pp) | 4.02±0.49 | **4.33±0.61** | 4.93±0.58 | 6.51±0.65 | 6.46±1.21 | 9.40±1.82 | 9.94±1.31 | 10.67±1.76 | 10.08±2.15 |
| spend %   | 20.0 | **12.3** | 8.5 | 3.7 | 1.5 | 0.5 | 0.2 | 0.1 | 0.1 |

#### ActiveEval-S+M (base 4.61 ± 0.32 @ 20%)

| threshold | 0 | 1e-4 | 3e-4 | 1e-3 | 3e-3 | 0.01 | 0.03 | 0.1 | 0.3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MAE (pp) | 4.61±0.32 | 5.04±0.65 | 5.23±0.72 | 7.04±0.80 | 7.05±0.43 | 9.36±1.91 | 9.31±1.94 | 9.31±2.20 | 10.03±2.28 |
| spend %   | 20.0 | 8.1 | 5.6 | 2.9 | 1.1 | 0.4 | 0.2 | 0.1 | 0.1 |

#### ActiveEval-Pair (base 4.79 ± 0.87 @ 20%)

| threshold | 0 | 1e-4 | 3e-4 | 1e-3 | 3e-3 | 0.01 | 0.03 | 0.1 | 0.3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MAE (pp) | 4.79±0.87 | 4.53±1.01 | 4.92±1.17 | 4.65±0.93 | 4.76±0.73 | 4.67±0.93 | 4.89±1.04 | 4.73±0.87 | 4.75±1.03 |
| spend %   | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 |

Raw output: `outputs/threshold_budget_benchmark.json`.

## Reading

- **ActiveEval-S is the one genuine result.** Its spend % strictly decreases as
  the threshold grows (20% → 12.3% → 8.5% → 3.7% → ...) — a real gain-decay
  curve — and at the smallest tested threshold (`1e-4`) it stops at 12.3% spend
  with MAE (4.33 ± 0.61) inside the full-cap CI (4.02 ± 0.49). That's a genuine
  ~39%-cheaper subset that performs indistinguishably from spending the whole
  20% cap.
- **Greedy MI's "under-spend" is a measurement artifact, not a finding.** Its
  spend % is identical (13.9%) across *every* nonzero threshold tested (`1e-4`
  through `0.3`) — that only happens because `select_greedy_mi`'s search
  universe is already capped at `cap=250` candidates (needed to keep its
  per-step complement-precision refactorization, which is cubic in the pool
  size, tractable — see `_target_universe` in
  [baselines/_core.py](../baselines/_core.py)). Once padding-to-budget is
  skipped, the pivoted-Cholesky loop is capped at `min(budget, 250)` regardless
  of whether the gain threshold ever actually triggers a break — and across all
  5 seeds and 8 thresholds it never did before hitting that cap. So this row
  demonstrates the *cap*, not the *threshold rule*.
- **Facility-location and ActiveEval-S+M decay too fast to be useful here.**
  Their coverage/facility-location gain drops by orders of magnitude within the
  first handful of picks, so even the smallest tested threshold (`1e-4`)
  either does nothing (facility-location: 20% spend, mae ~ baseline) or, one
  step up (`1e-3`), collapses to single-digit spend with 1.4-2x worse MAE.
  There is no usable middle ground in this log-spaced grid for these two.
- **Submod. benchmark and ActiveEval-Pair never trigger early stopping** in
  this range — their log-det diversity term keeps contributing gain all the
  way to the 20% cap. For ActiveEval-Pair specifically, its two-phase
  alpha-split (80% coverage / 20% diversity) means even when the coverage
  phase stops early, `remaining = budget - len(cov)` grows and the diversity
  phase absorbs the difference, so total spend rarely drops below the cap.
- **Caveat.** CIs overlap across most cells at 5 seeds — treat the "genuine
  free lunch" framing for ActiveEval-S as directional, not yet a
  paired-significant result; re-run with more seeds before citing in the
  paper. If pursued further: (1) `_GREEDY_CAP`-style search-universe caps
  should be reported/flagged whenever `gain_threshold` is also in play, since
  the two interact; (2) ActiveEval-Pair may need per-phase `gain_threshold`
  semantics (or a shared budget re-derivation) to ever show a genuine
  under-spend.

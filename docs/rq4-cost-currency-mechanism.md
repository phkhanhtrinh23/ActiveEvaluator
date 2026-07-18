# RQ4 mechanism analysis — why method rankings shift across cost currencies

Companion to the "Real-currency budgets (RQ4)" table in [README.md](../README.md). The
README states *what* happens (the `storage` budget inverts the ranking); this note
explains *why*, grounded in the selection code and one reproducible measurement.

> **Update.** Two implementation confounds previously distorted the `Greedy entropy
> (Alg. 1)` / `Greedy MI (Alg. 2)` rows specifically: an unmatched regularization
> constant against `select_logdet`, and an unnecessary candidate-pool cap in the
> `cost_budget` driver. Both are now fixed ([baselines/_core.py:364](../baselines/_core.py#L364),
> [:394](../baselines/_core.py#L394), [experiments/run_acquisition_benchmark.py:413](../experiments/run_acquisition_benchmark.py#L413)),
> and `README.md`'s RQ4 tables now reflect the corrected numbers (`Greedy entropy` is no
> longer listed as its own row — see the footnote on `Submod. benchmark` there).

## The shared mechanism: selection is cost-agnostic

A recurring point of confusion is that the experiment "prices actions by
`input_tok`, `output_tok`, …" while this note calls the methods "cost-agnostic."
Both are true, because cost enters the pipeline in only one of its **two separate
stages**:

1. **Ranking (cost-agnostic).** Every method ranks the candidate pool exactly once,
   from shift descriptors alone (`_method_order`,
   [experiments/run_acquisition_benchmark.py:412-431](../experiments/run_acquisition_benchmark.py#L412-L431)).
   The ranking never sees tokens, seconds, or GB — for a given method it is
   *identical* across all six currency tables.
2. **Spending (currency-aware).** `greedy_fill`
   ([experiments/cost_models.py:78-104](../experiments/cost_models.py#L78-L104)) walks
   that fixed order and keeps an action if its *marginal* cost still fits the
   remaining budget. The currency only decides **where the budget cuts the list
   off**, never how the list is ordered.

So every difference between the six per-currency tables comes down to which slice of
the pool each method's fixed ranking happens to prefer, and how that slice is priced.
"Additive" below describes the *price structure* of a currency (cost accumulates per
action, no reuse discount — see `Currency.total`,
[experiments/cost_models.py:65-71](../experiments/cost_models.py#L65-L71)), not a
property of any method.

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

### Read gaps, not ranks

The reshuffles look dramatic only if you read **rank positions**. Under `latency`,
ActiveEval-S falls from 1st (`count`) to 5th — but the entire top-5 spans
4.38–4.59 pp, i.e. **0.21 pp**, less than a third of any single method's ±0.5–0.65
CI half-width. When the field is that tight, rank is a lottery over seeds; a
different seed set would reshuffle it again. Contrast with what a *real* mechanism
difference looks like under `storage`: ActiveEval-Pair moves 4.11 → 6.81 pp
(**2.7 pp**, CI-separated) *and* comes with a measurable cause — its purchasable
actions collapse from 270 to 91. Under `latency` there is no such signature:
ActiveEval-S still buys 255 of ~270 actions.

### The small structural effect that does exist

Additivity doesn't make the currencies perfectly neutral. In the README tables,
ActiveEval-S/-Pair consistently buy slightly **fewer** actions than the other
methods under the token/latency currencies (252 vs. ~280 under `input_tok`, 255 vs.
~282 under `latency`): their target-similar picks land on slightly
pricier-than-average actions (e.g. `latency` ∝ tokens × params). This is a real,
repeatable effect — but losing ~25 of 270 labels moves MAE by tenths of a pp, well
inside seed noise. The qualitative line between Group 1 and Group 2 is the *size*
of the purchasing-power loss: ~10% here vs. ~65% under `storage`.

### How to verify rigorously

All methods run on the same seed set, so the clean test is a **paired per-seed
comparison**: for each seed, compute MAE(method A) − MAE(method B) under the
currency in question and check whether the sign is consistent across seeds. If the
sign flips seed-to-seed, "noise" is confirmed directly rather than inferred from
overlapping CIs. (Not yet run for the Group-1 reshuffles; the CI-overlap argument
above is the current basis.)

## Group 2 — `storage`: a real inversion, with a concrete cause

### Summary of the causal chain (details and evidence below)

Under real (random) per-model checkpoint costs, `Submod. benchmark` ends up with
**both** more distinct models touched **and** more actions bought than
`ActiveEval-Pair`. It's tempting to treat both as structural advantages of
`select_logdet`, but only **one** of them survives a clean test — the other is partly
luck.

**Isolating luck from structure**: which specific models get touched has nothing to do
with which method is running (`checkpoint_gb` is drawn from an rng stream fully
decoupled from the descriptors that drive selection — see the "coin flip" evidence
further down), so re-running with checkpoint cost **neutralized** (every model priced
identically) removes that luck entirely. Result (10 seeds, storage budget with a flat
per-model price):

```
Submod. benchmark    median actions=167   median models=9   actions/model=18.6
ActiveEval-Pair      median actions=67    median models=9   actions/model=7.4
```

**With cost-luck removed, both methods afford the exact same number of models (9 = 9).**
So `select_logdet` touching *more* models than `ActiveEval-Pair` under real (random)
costs (16 vs. 12 in the measurement below) is **not** a robust structural fact — in
that specific 5-seed sample, cost-luck happened to tilt in `select_logdet`'s favor on
top of the real effect. What *does* survive with luck fully removed, essentially
unchanged, is **actions per model** (18.6 vs. 7.4, matching the original 15.6 vs. 7.5)
— this is the one number that is genuinely structural, driven by the intrinsic-vs-extrinsic
objective difference below, independent of which models happen to be cheap or expensive
in any given run.

1. Nearly all storage cost is the checkpoint, paid once per unique model
   ([experiments/cost_models.py:145](../experiments/cost_models.py#L145)) — the cache
   fee per pair is negligible. So "cost per action" ≈ "how often a new checkpoint is
   needed."
2. `select_logdet`'s objective is *intrinsic* (only about the selected set itself) and
   — for this problem's descriptors — genuinely indifferent between "new sample-set,
   same model" and "new model" (verified: both move the descriptor by the same amount
   in expectation). It therefore has no structural pressure to change models often.
   `select_facility`'s objective is *extrinsic* (covering a fixed target region built
   from 8 independent unseen models) with diminishing returns per covered anchor, which
   *does* structurally require sampling many different models (verified: at equal
   action count with no budget constraint, it touches 60 of 60 models vs. `select_logdet`'s
   35).
3. Under a **fixed GB budget**, "extracts fewer actions per model" directly means
   "burns the budget on checkpoints faster relative to actions gained," which directly
   means "buys fewer total actions" — `ActiveEval-Pair` affords only ~90 actions before
   running out of money (real costs) / 67 (neutralized costs); `select_logdet` affords
   ~250 (real costs) / 167 (neutralized costs) in the same budget, because most of its
   actions reuse an already-paid-for model.
4. Whether the *number of models touched* also ends up higher for `select_logdet`
   depends on cost-luck (see above) — with real random costs it usually does (the
   larger action count, spread over even a similar or larger number of models, still
   needs *some* new-model spending along the way), but it is not guaranteed the way the
   actions-per-model gap is.

So the one number to trust as a genuine, luck-independent mechanism is **actions per
model touched**: `select_logdet` extracts roughly 2-2.5× more actions per checkpoint
than `ActiveEval-Pair`, and that alone is what makes it buy far more total actions
under the same GB ceiling — regardless of which specific models happen to be cheap or
expensive in a given run.

### The cost formula, and why it cares about *models*, not *pairs*

```
storage_cost(kept actions)  =  (# distinct models touched) × checkpoint_gb
                              + (# kept actions)            × small_cache_gb
```

([experiments/cost_models.py:145](../experiments/cost_models.py#L145),
[:154-155](../experiments/cost_models.py#L154-L155)): `checkpoint_gb` is paid **once
per unique model**, the first time any of its pairs is kept; every *subsequent* pair
from a model already touched costs only the small per-action cache fee.
`checkpoint_gb ≫ small_cache_gb`, so the first term dominates. That means the number
of actions a fixed GB budget can buy is governed almost entirely by **how many new
models the selection order forces you to pay for**, not by how many pairs it selects
or how feature-diverse those pairs are.

**Worked example** (checkpoint = 10 GB/model, cache ≈ 0.01 GB/action, budget = 25 GB).
An order that stays on one model before moving to the next:

| step | pick | model already touched? | marginal cost | running total |
| --- | --- | --- | --- | --- |
| 1 | (A, 1) | no → pay checkpoint | 10 + 0.01 | 10.01 |
| 2 | (A, 2) | yes | 0.01 | 10.02 |
| 3 | (A, 3) | yes | 0.01 | 10.03 |
| 4 | (A, 4) | yes | 0.01 | 10.04 |
| 5 | (A, 5) | yes | 0.01 | 10.05 |
| 6 | (B, 1) | no → pay checkpoint | 10 + 0.01 | 20.06 |
| 7 | (B, 2) | yes | 0.01 | 20.07 |

→ 25 GB buys **7 actions**, paying the checkpoint only **twice**. Now an order that
jumps to a new model on every pick:

| step | pick | model already touched? | marginal cost | running total |
| --- | --- | --- | --- | --- |
| 1 | (A, 1) | no → pay checkpoint | 10 + 0.01 | 10.01 |
| 2 | (B, 1) | no → pay checkpoint | 10 + 0.01 | 20.02 |
| 3 | (C, 1) | no → pay checkpoint | 10 + 0.01 | 30.03 — **over budget, stop** |

→ The same 25 GB buys only **2 actions**, because every pick demands a fresh
checkpoint. This is what "a method extracts N actions per model touched" means
concretely: it's how many *additional, nearly-free* pairs it manages to squeeze out of
each model before it has to pay for a new one.

### Two different notions of "diverse" — only one of them costs storage

It's tempting to read the result as "`Submod. benchmark` picks fewer models but more
(diverse) pairs, `ActiveEval` picks more models but fewer pairs." That's roughly the
right shape but the causal story is the opposite of "choosing diversity": neither
method is choosing *how many models* to touch as a goal. Model-count is a **side
effect** of what each method's objective actually optimizes, and the two objectives
live on different axes:

- **Pair/feature diversity** (what `select_logdet`'s log-det objective directly
  maximizes) is about spreading picks across the *descriptor space* `(u, v, u⊙v)`.
  Crucially, changing the sample-set `v` alone — even for the *same* reference model
  `u` — already moves the descriptor a lot (`make_problem`'s `descriptor(u, v)`,
  [experiments/run_acquisition_benchmark.py:84-87](../experiments/run_acquisition_benchmark.py#L84-L87)).
  So `select_logdet` can satisfy "pick a feature-diverse set" by taking many
  *different sample-sets from the same model* — it never needs to change models to
  keep scoring well, and in this problem instance it mostly doesn't.
- **Model-identity spread** (what `storage` cost actually taxes) is a completely
  separate axis. `select_facility` (ActiveEval's coverage term) is explicitly
  indifferent to it — "**the model axis is left uniform**"
  ([baselines/_core.py:411-414](../baselines/_core.py#L411-L414)) — its only goal is
  covering the *target* region with the most target-similar pairs, full stop. In this
  problem the handful of pairs closest to the target happen to be scattered thinly
  across many different models (at most 1-2 near-target pairs per model), so chasing
  target-similarity *incidentally* forces the selector to keep jumping to new models —
  not because it values model variety, but because that's where the good pairs are.

So the correct causal statement is: **`select_logdet` stays cheap under `storage`
because pair-diversity doesn't require model-diversity here; `ActiveEval-Pair`
becomes expensive because target-similarity does.** Neither method is "trying" to
manage its model footprint at all — that's exactly the gap the Takeaway below points
at.

### Why, in algorithm-theoretic terms: two submodular objectives with different structure

The two functions don't just happen to behave differently here — they are
**structurally different classes of submodular objective**, and that difference
predicts the model-touching behavior without appealing to luck:

- **`select_logdet`/`select_greedy_entropy` maximize `log det(Σ_S)`**, a **k-DPP MAP /
  D-optimal design** objective. It is entirely *intrinsic*: the marginal gain of a
  candidate `x` given the already-selected set `S` is its **conditional variance**
  `Var(x | S)`, a purely geometric quantity in kernel space with no reference to
  anything outside `S`. Model identity never enters this computation — the function
  literally cannot see `pair_model`.
- **`select_facility` (ActiveEval's coverage term) maximizes
  `Σ_{q∈Q} w(q)·max_{s∈S} k(q,s)`**, a **weighted facility-location / set-cover**
  objective over the *external*, fixed target region `Q`. This class of function has a
  well-known **diminishing-returns-per-anchor** property (Nemhauser, Wolsey & Fisher
  1978): once some anchor `q ∈ Q` is well covered by an existing pick, *any* further
  candidate similar to `q` — regardless of which model it came from — contributes
  almost no additional gain for that anchor. The objective is therefore forced to keep
  searching for candidates that cover *different, still-uncovered* anchors.

**The model/workload axes are empirically interchangeable for the first objective, but
not for the second.** `descriptor(u, v) = concat(u, v, u⊙v)` treats the model factor
`u` and the workload factor `v` symmetrically. Measured directly on the data (seed 0,
20,000 random pairs, mean squared descriptor distance):

```
same model,  different sample-set : 21.83
different model,  same sample-set : 22.21   <- statistically the same as above
different model AND sample-set    : 34.65   <- noticeably farther
```

This matches the closed-form prediction for `d_lat = 6`: switching *either single*
axis moves the descriptor by `≈ 4 × d_lat = 24` in expectation; switching *both* moves
it by `≈ 6 × d_lat = 36` (both u, v ~ N(0, I) independent, so the interaction term
`u⊙v` contributes symmetric cross-variance regardless of which factor changed).
Concretely: for `select_logdet`, taking another workload from an *already-touched*
model is, in expectation, exactly as informative as jumping to a brand-new model —
there is no structural incentive to prefer one over the other, so which axis its
greedy path happens to exploit is essentially arbitrary (consistent with the 11/20
"coin flip" on model cost above). It settles on reusing touched models substantially
(not switching every step) simply because within-model diminishing returns are gentle:
each already-selected workload from model `i` slightly reduces the residual variance
of *other* workloads sharing that same `u_i`, but not by much, so a model's ~30
available workloads keep paying off for many picks before a fresh model becomes
clearly better.

**`select_facility` has no such symmetry to exploit**, because `Q` is not just "the
descriptor space" — it is specifically built from **8 independently-drawn unseen
models** (`Uo`, `make_problem`). Covering those 8 structurally distinct directions
well requires source pairs whose own `u` vectors happen to sit close to *each* of
those 8 independent draws — and because training-model factors are themselves i.i.d.
Gaussian, whichever training model is closest to unseen model #1 is essentially
unrelated to whichever is closest to unseen model #2, ..., #8. Combined with
diminishing returns once an anchor is covered, this is a **provable** requirement to
sample from multiple different training models, not an accident of this particular
run.

### Rate vs. absolute count: why the theory (`select_facility` spreads more) and the storage table (`select_logdet` touches *more* models) don't conflict

The theory above predicts `select_facility` has a *higher per-action rate* of
touching new models than `select_logdet`. The storage table below seems to say the
opposite — `select_logdet` touches **more** models in absolute terms (16 vs. 12). Both
are true, and they don't contradict each other; they answer different questions.

**Isolate the rate directly, on its own terms.** `_method_order` produces a single
cost-agnostic ranking of up to 450 candidates per method
([experiments/run_acquisition_benchmark.py:412-431](../experiments/run_acquisition_benchmark.py#L412-L431)).
Take the first 270 entries of that ranking — the same cardinality budget the main
table uses (`budget_frac=0.15` × `P≈1800` ≈ 270) — with **no GB budget or cost
involved at all**, and count distinct models among them (10 seeds):

```
Submod. benchmark    median 35 models touched  (rate ≈ 35/270 = 0.130 new-model / action)
ActiveEval-Pair      median 60 models touched  (rate ≈ 60/270 = 0.222 new-model / action)
```

This confirms the theory exactly: at equal action count, `ActiveEval-Pair` spreads
across nearly *all 60* training models, while `Submod.` stays within about a third of
them. The rate story is correct on its own.

**The storage table is a *different* measurement, not the same rate re-expressed
under a budget.** `greedy_fill` doesn't take a prefix of the order — it walks the
*entire* 450-long ranking front-to-back and, for each candidate, either keeps it (if
its marginal cost still fits the remaining GB) or **skips it and keeps scanning**
([experiments/cost_models.py:88-104](../experiments/cost_models.py#L88-L104)). A
skipped candidate costs nothing and does not mark its model as "touched" — only a
*kept* candidate pays the checkpoint (once per model) and adds that model to the
`used` set. So the set of kept actions under a GB budget can be a scattered subset of
the 450-long order, not its first 270 entries — there is no clean formula turning the
prefix-based rate above into the GB-budget outcome; the two have to be measured
separately, which is exactly why both tables are reported as direct simulations
rather than one being derived from the other.

**Qualitatively, though, the mechanism is exactly what the rate predicts**:
`ActiveEval-Pair` needs a fresh checkpoint roughly every ~7-8 kept actions (from the
storage-budget measurement above) — consistent with its high per-action rate of
finding new-model candidates near the front of its order — which burns a fixed GB
ceiling far faster than `Submod.`'s roughly every ~15-16 kept actions. So
`ActiveEval-Pair` only affords **90 actions total** before its GB budget is exhausted
— it never gets to *keep* enough of its order to express its full spreading tendency,
because it runs out of money first. `Submod.`, needing a fresh checkpoint far less
often, affords **250 actions** in the same GB envelope, and simply by taking that many
more kept actions, ends up touching more distinct models in total — even though, action
for action, it explores new models at a *lower* rate than `ActiveEval-Pair` does.
In short: **`select_facility`
is *not* less inclined to explore models than `select_logdet` — it is far *more*
inclined to. But under an amortized storage budget, that same inclination is what
throttles its total spending power, so it ends up touching fewer models in absolute
terms, not more.** The rate is a property of the algorithm; the absolute count is a
property of the algorithm *combined with* how a GB budget interacts with that rate.

**Measured** (5 seeds, budget = 15% of pool storage cost; median actions bought,
median distinct models touched among those actions, and the ratio):

| Method | median #actions | median #models touched | actions per model |
| --- | --- | --- | --- |
| Submod. benchmark | 250 | 16 | 15.6 |
| Bayesian opt. design | 232 | 12 | 19.3 |
| ActiveEval-S | 99 | 12 | 8.2 |
| ActiveEval-Pair | 90 | 12 | 7.5 |
| Random | 113 | 15 | 7.5 |

Submod./Bayesian buy roughly **twice as many actions per new model** as
ActiveEval-S/-Pair — each new model they touch "unlocks" ~16–19 further actions at
near-zero marginal storage cost, while ActiveEval's target-chasing unlocks only ~7-8
before it needs to pay for another model. (Random sits at the same ~7.5 ratio as
ActiveEval here — not because it targets model spread either, but because
`greedy_fill`'s skip-and-continue rule under a tight amortized budget already biases
*any* order toward re-using already-paid-for models over brand-new ones; Random's
absolute numbers are still worse because its order isn't optimizing anything, so it
buys fewer total actions for the models it does touch.)

Net effect: ActiveEval-Pair pays the checkpoint charge roughly twice as often per kept
action as Submod./Bayesian. The 15% GB budget runs out after only ~90 actions instead
of ~250, erasing the information advantage that target-aware coverage otherwise
provides — hence ActiveEval-Pair's drop to near-worst (6.81 pp) under `storage`
despite leading almost every other table.

### "Same 15% cost" is the budget rule, not a coincidence to explain

It can look paradoxical that `Submod. benchmark` buys *both* more models (16) *and*
more actions (250) than `ActiveEval-Pair` (12 models, 90 actions) for "the same 15%
cost" — if cost were additive in a simple `models × fixed_price + actions ×
fixed_price` sense, more of both should mean more total cost, not equal. The resolution
is that `budget = 0.15 × pool_total` is a **fixed GB ceiling**, identical for every
method by definition, and `greedy_fill` just keeps buying until it can't afford the
next item ([experiments/cost_models.py:78-104](../experiments/cost_models.py#L78-L104))
— so every method's *realized spend* converges to ≈ that same ceiling; there is nothing
to explain there. The real question is what a method gets *for* that fixed spend, and
the answer is that `checkpoint_gb` is **not a fixed per-model price** — it's drawn
log-uniformly from each model's parameter count, 0.5B–70B params → 1–140 GB checkpoint
([experiments/cost_models.py:127](../experiments/cost_models.py#L127),
[:145](../experiments/cost_models.py#L145)), a **140×** spread, independent of which
model the selection criterion happens to prefer.

**Measured breakdown** (5 seeds; `checkpoint_sum` = total GB spent on model
checkpoints, `cache_sum` = total GB spent on the small per-action fee — note it is
3-4 orders of magnitude smaller and can be ignored):

```
seed 0  budget=237.8GB   Submod: 25 models @ 9.5 GB/model avg  (checkpoint_sum=237.7, cache_sum=0.07)
                         AE-Pair: 9 models @ 26.4 GB/model avg (checkpoint_sum=237.6, cache_sum=0.02)
seed 1  budget=224.0GB   Submod: 14 models @ 16.0 GB/model avg (checkpoint_sum=223.7, cache_sum=0.08)
                         AE-Pair: 16 models @ 14.0 GB/model avg (checkpoint_sum=223.8, cache_sum=0.03)
seed 2  budget=269.8GB   Submod: 10 models @ 26.9 GB/model avg (checkpoint_sum=269.3, cache_sum=0.04)
                         AE-Pair: 12 models @ 22.4 GB/model avg (checkpoint_sum=268.9, cache_sum=0.01)
seed 3  budget=242.3GB   Submod: 16 models @ 15.1 GB/model avg (checkpoint_sum=242.2, cache_sum=0.04)
                         AE-Pair: 11 models @ 22.0 GB/model avg (checkpoint_sum=241.5, cache_sum=0.02)
seed 4  budget=323.5GB   Submod: 18 models @ 18.0 GB/model avg (checkpoint_sum=323.4, cache_sum=0.07)
                         AE-Pair: 12 models @ 26.9 GB/model avg (checkpoint_sum=323.4, cache_sum=0.02)
```

It's tempting to read the 5-seed table above as two compounding effects — "Submod.
touches cheaper models *and* mines each one harder" — but a 20-seed check
(`is Submod.'s avg per-model checkpoint cheaper than ActiveEval-Pair's?` vs. `is
Submod.'s actions/model ratio higher?`) shows only one of those is a real mechanism:

```
Submod. cheaper avg checkpoint-per-model:  11 / 20 seeds   (≈ coin flip)
Submod. higher actions-per-model:          20 / 20 seeds   (never flips)
```

**"Which models are cheap" is noise, not a mechanism.** `params_b` (a model's
checkpoint cost) is drawn from a *separate* rng stream in `make_cost_model`
(`np.random.default_rng(seed + 7919)`,
[experiments/cost_models.py:122](../experiments/cost_models.py#L122)) than the one
that drives each model's behavioral descriptor `U`/`V` in `make_problem`
(`np.random.default_rng(seed)`,
[experiments/run_acquisition_benchmark.py:53](../experiments/run_acquisition_benchmark.py#L53)).
Neither `select_logdet` nor `select_facility` ever sees `params_b` — they only see
`U`/`V`-derived descriptors — so there is no channel through which either selection
criterion could systematically prefer cheap or expensive models. The 5-seed table
above just happened to sample a run where Submod.'s touched models leaned cheaper in
3 of 5 draws; over 20 seeds that settles to 11/20, i.e. statistically indistinguishable
from a coin flip.

**"Actions per model" is the real, 100%-consistent mechanism**, and it's exactly the
structural story from the "two axes" section above: `select_logdet` can keep scoring
well by re-using an already-paid-for model across several sample-sets (13-20
actions/model touched, every seed), while `select_facility`'s target-similarity
objective only finds 1-2 good matches per model before it has to pay for a new one
(7-8.5 actions/model touched, every seed) — a ratio that never inverts across 20
independent draws. This is the one lever that actually explains the storage gap; the
per-model cost variation is real (140× spread) but contributes only noise around it,
not a directional bias in either method's favor.

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
unaffected. `README.md`'s per-currency tables have been updated to match: `Greedy
entropy (Alg. 1)` is no longer listed as a separate row (it duplicates `Submod.
benchmark` once `sigma` is matched — see the footnote on that row), and `Greedy MI`'s
rows now reflect `sigma=1.0`.

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

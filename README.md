# ActiveEvaluator

> **Budgeted meta-evaluation supervision acquisition.**
>
> Given pools of candidate reference models and labeled support workloads, which
> *model–workload evaluation actions* should we pay to run so that a meta-evaluator
> can assess **future, unseen** models on **unlabeled** targets — under a hard
> labeling budget? ActiveEvaluator selects (i) target-aware sample sets, (ii)
> behavior-aware reference models, and (iii) the model–workload pairs that most
> reduce predictor uncertainty, then meta-trains an accuracy predictor on only the
> acquired entries. Selection maximizes monotone-submodular coverage objectives — an
> **influence-weighted facility-location** term (with `(1-1/e)`-approximation lazy
> greedy), a **submodular-mutual-information** model term, and a **log-determinant**
> pair term — with a non-submodular **direct validation-loss reduction** oracle as an
> empirical upper bound. This is distinct from model selection (the deployed model is
> given), benchmark compression (we generalize to *future* models), and active
> testing (the target stays unlabeled).

## Quick start — reproduce the acquisition benchmark (CPU, no downloads)

```bash
pip install torch numpy
python -m experiments.run_acquisition_benchmark --seeds 5 --budget-frac 0.15
```

This self-contained benchmark builds a controlled meta-evaluation matrix (reference
models × sample-sets with shift descriptors and noisy execution-accuracy labels),
holds out unseen model families, gives every acquisition strategy the same 15%
budget, meta-trains the repo's own `ActiveEvaluator` MLP on the acquired pairs, and
reports unseen-model MAE. It uses the same selection math and predictor as the full
Text2SQL pipeline, so the method ordering mirrors the paper's main table.

**Setup.** Target sample-sets are **held out of training entirely** — they are
never labeled and never enter the candidate pool. The labelable pool is the
*source* region (reference models × off-target sample-sets); the target region is
available only as *unlabeled* descriptors (`X_target_ref`) that steer target-aware
acquisition. Generalisation is therefore over **unseen models AND unseen
workloads**, and no acquisition method can train on a target pair.

**Measured results** (5 seeds, 15% labeling budget, unseen-model MAE in percentage
points; lower is better). Under this held-out-target setup the full-budget
MetaEvaluator is the overall best; among **budgeted (15%)** methods, ActiveEval-S
and ActiveEval-Pair are the strongest — they beat whole-pool Random and every other
acquisition baseline and label-free estimator — but they do **not** close the gap to
full labeling (the earlier "less-is-more" reversal was an artifact of the leaky
setup where target pairs were labelable).


| Method                  | Family      | Unseen MAE (pp) | Cost |
| ----------------------- | ----------- | --------------- | ---- |
| MetaEvaluator (full)    | reference   | **3.60 ± 0.47** | 100% |
| **ActiveEval-S**        | ours        | 4.08 ± 0.68     | 15%  |
| **ActiveEval-Pair**     | ours        | 4.11 ± 0.39     | 15%  |
| DoC                     | estimator   | 4.39 ± 0.50     | —    |
| Random                  | acquisition | 4.47 ± 0.90     | 15%  |
| Bayesian opt. design    | acquisition | 4.61 ± 0.67     | 15%  |
| Submod. benchmark [^1]  | acquisition | 4.64 ± 0.22     | 15%  |
| k-center                | acquisition | 4.96 ± 0.51     | 15%  |
| Facility-location       | acquisition | 5.30 ± 0.48     | 15%  |
| Matrix completion       | acquisition | 5.38 ± 0.90     | 15%  |
| **ActiveEval-S+M**      | ours        | 5.66 ± 1.11     | 15%  |
| Greedy MI (Alg. 2)      | acquisition | 5.81 ± 0.71     | 15%  |
| GRAD-MATCH              | acquisition | 6.00 ± 1.48     | 15%  |
| Active testing          | acquisition | 6.10 ± 0.98     | 15%  |
| ATC                     | estimator   | 6.84 ± 3.37     | —    |


At a 15% budget ActiveEval-S/-Pair extract the most signal — concentrating labels on
clean, near-target *source* pairs whose feature region transfers best to the held-out
target — but full labeling retains an information edge. Regenerate with the command
above (`outputs/acquisition_benchmark.json`). Note: `ActiveEval-S+M`'s model-axis
narrowing hurts in this regime, and on the noisier image variant (fewer reference
models) budgeted methods are harder to separate — see the per-benchmark JSONs.

[^1]: `Submod. benchmark` (`select_logdet`) is the same pivoted-Cholesky greedy
    log-det/max-entropy algorithm as `Greedy entropy (Alg. 1)` once both use the same
    regularization (`sigma=1.0`, the shared default — [baselines/_core.py](baselines/_core.py)).
    The previous `greedy_entropy`/`greedy_mi` default (`sigma=None`, i.e. a near-zero
    `+1e-6` jitter on the kernel diagonal) is the value that degraded their results:
    it left the greedy pivot over-sensitive to noise in the RBF-bandwidth estimate,
    costing Greedy entropy 5.12 ± 0.14 vs. 4.68 ± 0.29 pp with `sigma=1.0` (exactly
    `select_logdet`'s number) and Greedy MI 5.82 ± 0.50 vs. 5.10 ± 0.64 pp (main
    table, 5 seeds). `Greedy entropy (Alg. 1)` is therefore omitted from this and the
    other method-comparison tables to avoid listing the same method twice; it remains
    available in `ACQUISITION_REGISTRY` and is compared directly against
    `Greedy MI (Alg. 2)` in the [dedicated sweep below](#greedy-entropy-vs-greedy-mutual-information-exploratory).
    See [docs/rq4-cost-currency-mechanism.md](docs/rq4-cost-currency-mechanism.md) for
    the full ablation isolating this from the pool-cap confound.

### Budget efficiency (RQ3)

```bash
python -m experiments.run_acquisition_benchmark --mode sweep --seeds 5
```

Average unseen MAE (pp) vs. labeling budget (full-budget MetaEvaluator = **3.60**).
ActiveEval-Pair leads Facility-location and whole-pool Random at every budget — the
gap is largest at small budgets, where target-aware coverage matters most — and all
budgeted methods converge toward, but stay above, the full-labeling line.


| Budget | ActiveEval-Pair | Facility-loc | Random   |
| ------ | --------------- | ------------ | -------- |
| 5%     | **5.52**        | 6.59         | 6.83     |
| 10%    | **4.87**        | 5.85         | 5.43     |
| 15%    | **4.19**        | 5.08         | 4.99     |
| 20%    | **4.82**        | 4.94         | 5.33     |
| 30%    | **4.49**        | 4.96         | 4.74     |
| 50%    | 4.21            | 4.24         | **4.04** |


### Ablation (RQ5)

```bash
python -m experiments.run_acquisition_benchmark --mode ablation --seeds 5
```

With target sample-sets held out of training, the ablation tells a different story
than the leaky setup did: removing target-aware narrowing costs little here
(~0.06 pp), while removing *all* selection structure (random) costs the most. In this
regime ActiveEval's budgeted advantage comes mainly from the general
coverage/diversity structure, not from target narrowing per se — the source pool no
longer contains an easy "clean target" cluster to snap onto. (Influence weighting,
submodular MI, knapsack budgeting, and the uncertainty head are ablated in the full
Text2SQL pipeline.)


| Configuration            | Unseen MAE (pp) |
| ------------------------ | --------------- |
| ActiveEval-Pair (full)   | **4.19 ± 0.39** |
| − target-aware narrowing | 4.26 ± 0.18     |
| − all structure (Random) | 4.99 ± 0.56     |


### Mechanism analysis — why the budgeted methods differ

The synthetic problem has three distinct populations; target sample-sets are held
out of training and appear only as an unlabeled reference and as the test set:

```
                 10 target sets           |   30 source sets
               ------------------------   | ---------------------------
60 reference   NOT labelable.             |  LABELABLE POOL: 1800 pairs,
   models      Descriptors only, used     |  graded noise (clean near the
               as unlabeled X_target_ref  |  target region -> noisy far)
               ------------------------   | ---------------------------
 8 unseen      TEST SET: 80 pairs,        |  (never evaluated)
   models      clean labels               |
```

`target_sets` (10 of 40 sample-sets, a coherent latent cluster) is the deployment
workload. It is **held out of training**: the meta-evaluator never sees a labeled
target pair. The candidate pool is the 1800 *source* pairs (60 reference models × 30
source sample-sets); a source pair's label noise scales with its distance from the
target region (near ≈ 0.45×, far ≈ 2.0× base). Target-aware acquisition uses the
unlabeled `X_target_ref` descriptors to bias selection toward clean, near-target
source pairs whose feature region transfers best to the held-out target.

What the numbers say in this corrected (no-leak) regime:

1. **Full labeling wins; there is no less-is-more reversal.** With target pairs
  unlabelable, the earlier crossover (a budgeted subset *beating* the full matrix)
   disappears — it depended on labeling clean target pairs directly. The full
   MetaEvaluator (3.60 pp) is the overall best; budgeted methods approach it from
   above.
2. **Among budgeted methods, target-aware source coverage is what helps — modestly.**
  ActiveEval-S/-Pair (≈4.1 pp) beat whole-pool Random (4.47) and every other
   baseline by concentrating labels on near-target source pairs. But the ablation
   shows the *target-narrowing component itself* now contributes only ~0.06 pp; most
   of the budgeted advantage comes from the general coverage/diversity structure.
   (Diagnostic: facility/ActiveEval-S selections sit at mean distance ≈3.6 from the
   target-reference centroid vs ≈4.3 for Random, with ~35% lower label noise.)
3. **These conclusions are by-construction and need real-pipeline validation.**
  The generator makes far-source labels noisy and over-represented
   (`make_problem`, `experiments/run_acquisition_benchmark.py`) and evaluates only on
   the held-out target workload. Whether real Text2SQL execution-accuracy labels
   exhibit such target-proximity structure is an empirical claim the full pipeline
   must support; components that show little effect here (target narrowing, influence
   weights, submodular MI over models, knapsack budgeting, the uncertainty head) must
   likewise earn their place there.

### Noise sensitivity — when does a budgeted subset beat labelling everything?

```bash
python -m experiments.run_acquisition_benchmark --mode noise_sweep --seeds 15
```

This sweep varies only the far/near-source noise ratio (near-target source noise
fixed; the default benchmark sits at 4.4×). With target sample-sets held out of
training, **full labeling wins at every ratio** — the earlier less-is-more crossover
was a product of labeling clean target pairs directly, which is no longer allowed.
What survives is a clean ordering among the *budgeted* methods: ActiveEval-Pair beats
whole-pool Random throughout, and the ActiveEval–full gap shrinks as far-source noise
grows (0.77 pp at 1× → 0.22 pp at 6×) because heavier far-source noise erodes the full
matrix's information advantage (5 seeds).


| far/near noise | ActiveEval-Pair @15% | Full (100%)     | Random @15% |
| -------------- | -------------------- | --------------- | ----------- |
| 1.0×           | 3.67 ± 0.21          | **2.90 ± 0.39** | 3.88 ± 0.34 |
| 1.5×           | 3.73 ± 0.20          | **2.99 ± 0.36** | 4.05 ± 0.36 |
| 2.0×           | 3.79 ± 0.35          | **3.05 ± 0.38** | 4.14 ± 0.41 |
| 3.0×           | 3.94 ± 0.50          | **3.26 ± 0.43** | 4.52 ± 0.44 |
| 4.4×           | 4.20 ± 0.39          | **3.64 ± 0.48** | 4.96 ± 0.49 |
| 6.0×           | 4.35 ± 0.49          | **4.13 ± 0.51** | 5.50 ± 0.84 |


ActiveEval-Pair and the full matrix both degrade as far-source noise grows, but
whole-pool Random degrades fastest (most of its picks land in the noisy far-source
region), so the target-aware budgeted advantage over Random widens with the noise gap.

### Greedy entropy vs. greedy mutual information (exploratory)

```bash
python -m experiments.run_acquisition_benchmark --mode entropy_mi_sweep --seeds 10
```

Ports Algorithm 1 (greedy entropy) and Algorithm 2 (greedy mutual information) from
`benchmark-selection/code/greedy_select.py` — compared over a benchmark correlation
matrix in `eval_entropy_vs_mi.py` — into this pair-acquisition setting, by building
the same PSD kernel (`Sigma`, an RBF kernel over pair shift-descriptors) over the
*source* candidate pool (target sample-sets are held out and unlabeled), then running
the identical pivoted-Cholesky selection loops. The search set is capped at 600 pairs
because greedy MI's per-step complement-precision refactorization is cubic in the pool
size; the axis is the labeling budget in % of that capped pool (5 seeds).


| Budget | Labels | Entropy (pp) | MI (pp)     |
| ------ | ------ | ------------ | ----------- |
| 0.67%  | 12     | 9.28 ± 2.16  | 8.31 ± 1.10 |
| 1.67%  | 30     | 7.03 ± 1.64  | 6.57 ± 1.20 |
| 3.33%  | 60     | 6.07 ± 0.83  | 7.81 ± 2.22 |
| 6.67%  | 120    | 5.22 ± 1.37  | 5.69 ± 1.00 |
| 10%    | 180    | 4.82 ± 0.88  | 5.39 ± 0.72 |
| 16.67% | 300    | 5.47 ± 1.02  | 5.15 ± 0.79 |
| 25%    | 450    | 4.84 ± 0.29  | 5.05 ± 0.41 |
| 33.33% | 600    | 4.70 ± 0.38  | 4.74 ± 0.45 |


MAE improves with budget (more clean source labels help), but neither pivoted-Cholesky
rule stands out from the simpler baselines in this held-out regime — over the source
pool they act as diversity/coverage selectors without the target-region signal the
leaky setup handed them, and both land well above the full MetaEvaluator (3.60 pp).

### Real-currency budgets (RQ4) — tokens, latency, memory, storage

```bash
python -m experiments.run_acquisition_benchmark --mode cost_budget --seeds 5
```

The other tables set the budget as a **count** of evaluation actions (every
model–workload pair costs 1). In practice, running a model on a workload slice
consumes different *currencies*, and the same 15% budget buys a different number of
labels depending on which one you meter. This mode re-spends the budget as 15% of
each currency's **whole-pool cost** instead of 15% of the action *count*:


| Currency     | Per-action cost model (`experiments/cost_models.py`)                           | Kind          |
| ------------ | ------------------------------------------------------------------------------ | ------------- |
| `count`      | 1 (the original cardinality budget — reproduces the main table exactly)        | additive      |
| `input_tok`  | `set_size × prompt_len` (≈ model-independent; big slices cost more to feed)    | additive      |
| `output_tok` | `set_size × model_verbosity`                                                   | additive      |
| `latency`    | `set_size × (prompt+gen tokens) × model_params` (wall-clock to run the action) | additive      |
| `memory`     | `model_params` (peak resident GB; set-independent, paid per action)            | additive      |
| `storage`    | checkpoint GB counted **once per unique model** + small per-action cache       | **amortized** |


Costs are drawn from per-model (params, verbosity) and per-sample-set (examples,
prompt length) size factors with an rng decoupled from the label noise, so they are
reproducible; absolute scales are illustrative — only *ratios across actions* matter.

**Flow.** Selection is **cost-agnostic**: each method ranks the pool *once per seed*
(the ranking never sees the currency), and each currency only decides how far 15% of
its budget reaches into that fixed ranking:

```
  method.select(budget = 450)  ─────────►  order = [i₀, i₁, …, i₄₄₉]   (priority list of pair-indices)
                                                     │  same order reused for every currency
        ┌────────────────────────────────┬──────────┴──────────┬────────────────────────────────┐
        ▼ input_tok                       ▼ memory              ▼ storage (amortized)             │
 budget = 0.15·Σ tokens           budget = 0.15·Σ GB      budget = 0.15·(cache + checkpoints)     │
        │                                 │                     │  checkpoint paid once per model │
        ▼   greedy_fill: walk order front-to-back, KEEP i if its marginal cost fits the budget,   │
        │                    SKIP (don't stop) if not — so kept = a prefix-with-skips of order    │
     kept ≈ 252 actions               kept ≈ 264            kept ≈ 91 (checkpoints eat the budget) │
        └────────────────────────────────┴─────────────────────┴────────────────────────────────┘
                                          ▼
                       train ActiveEvaluator on kept → unseen MAE
```

`count` uses `budget = 0.15·P`; every other currency uses `0.15·pool_total`, the cost
of running the *whole* pool in that unit. Because the order is fixed, the table below
reflects only how far each budget stretches — not six different selection problems.
(Caveat: composite methods fix their internal split against the full 450-long order —
ActiveEval-Pair's 80/20 coverage/diversity, ActiveEval-S+M's keep-half-the-models — so
when a currency affords far fewer actions, e.g. storage's ~91, only the coverage phase
is ever reached; re-deriving the split per currency would be more faithful but 6× the
compute.)

**Measured results** (5 seeds, 15% of each currency's whole-pool cost; unseen-model
MAE in pp, lower is better; MetaEvaluator full = **3.60**). One table per currency,
each sorted by its own MAE — the top row is that currency's winner (**bold**).
`#actions` is the median number of labeled pairs the 15% budget actually bought.

> **On `count`.** `count` is *not* a computational cost — it is the plain
> **label-count** budget every other table in this README uses: each action costs
> exactly 1, so "15% of the whole-pool cost" is simply 15% of the ~1800 pairs = ~270
> labels. It is the degenerate currency where cost = cardinality, kept here as the
> reference; the other five re-price those same actions by a real resource (tokens /
> seconds / GB), and `#actions` then varies because a fixed budget of tokens/GB buys
> a different *number* of labels than a flat count does.

#### count — label-count budget (baseline = main table)

| Method | Unseen MAE (pp) | #actions |
| ------ | --------------- | -------- |
| **ActiveEval-S** | **4.08 ± 0.68** | 270 |
| ActiveEval-Pair | 4.11 ± 0.82 | 270 |
| Bayesian opt. design | 4.61 ± 0.67 | 270 |
| Submod. benchmark [^1] | 4.64 ± 0.22 | 270 |
| k-center | 4.96 ± 0.51 | 270 |
| Matrix completion | 5.14 ± 0.78 | 270 |
| Random | 5.47 ± 0.90 | 270 |
| Facility-location | 5.54 ± 0.57 | 270 |
| ActiveEval-S+M | 5.66 ± 1.11 | 270 |
| Greedy MI (Alg. 2) | 5.77 ± 0.52 | 270 |
| GRAD-MATCH | 5.96 ± 1.52 | 270 |
| Active testing | 6.10 ± 0.98 | 270 |

#### input_tok — input tokens (`set_size × prompt_len`)

| Method | Unseen MAE (pp) | #actions |
| ------ | --------------- | -------- |
| **ActiveEval-Pair** | **4.53 ± 0.46** | 252 |
| ActiveEval-S | 4.56 ± 0.38 | 252 |
| Submod. benchmark [^1] | 4.77 ± 0.70 | 241 |
| Bayesian opt. design | 4.77 ± 0.72 | 254 |
| k-center | 4.85 ± 0.21 | 266 |
| Matrix completion | 4.97 ± 0.76 | 268 |
| Facility-location | 5.06 ± 0.64 | 267 |
| Random | 5.20 ± 0.79 | 280 |
| ActiveEval-S+M | 5.34 ± 0.97 | 248 |
| Greedy MI (Alg. 2) | 5.67 ± 0.68 | 280 |
| Active testing | 6.11 ± 0.90 | 270 |
| GRAD-MATCH | 6.26 ± 1.35 | 287 |

#### output_tok — generated tokens (`set_size × verbosity`)

| Method | Unseen MAE (pp) | #actions |
| ------ | --------------- | -------- |
| **Submod. benchmark** [^1] | **4.44 ± 0.26** | 286 |
| ActiveEval-Pair | 4.70 ± 0.52 | 248 |
| Bayesian opt. design | 4.76 ± 0.95 | 293 |
| ActiveEval-S | 4.78 ± 0.58 | 248 |
| k-center | 4.81 ± 0.67 | 283 |
| Facility-location | 5.14 ± 0.54 | 252 |
| Matrix completion | 5.18 ± 0.87 | 268 |
| ActiveEval-S+M | 5.29 ± 0.99 | 259 |
| Random | 5.52 ± 0.78 | 277 |
| Greedy MI (Alg. 2) | 5.67 ± 0.62 | 272 |
| Active testing | 5.69 ± 0.86 | 267 |
| GRAD-MATCH | 5.89 ± 1.66 | 307 |

#### latency — wall-clock seconds (`tokens × params`)

| Method | Unseen MAE (pp) | #actions |
| ------ | --------------- | -------- |
| **k-center** | **4.38 ± 0.65** | 282 |
| Bayesian opt. design | 4.47 ± 0.93 | 271 |
| Submod. benchmark [^1] | 4.53 ± 0.35 | 277 |
| ActiveEval-Pair | 4.55 ± 0.65 | 255 |
| ActiveEval-S | 4.59 ± 0.54 | 255 |
| Facility-location | 4.81 ± 0.29 | 305 |
| Matrix completion | 5.21 ± 0.94 | 265 |
| Random | 5.28 ± 0.68 | 279 |
| Greedy MI (Alg. 2) | 5.45 ± 0.98 | 286 |
| ActiveEval-S+M | 5.47 ± 1.12 | 235 |
| Active testing | 5.79 ± 0.80 | 278 |
| GRAD-MATCH | 6.15 ± 1.22 | 314 |

#### memory — peak resident GB (`params`, per action)

| Method | Unseen MAE (pp) | #actions |
| ------ | --------------- | -------- |
| **ActiveEval-S** | **4.12 ± 0.58** | 264 |
| ActiveEval-Pair | 4.33 ± 0.49 | 264 |
| Submod. benchmark [^1] | 4.48 ± 0.42 | 312 |
| Bayesian opt. design | 4.53 ± 1.23 | 318 |
| k-center | 4.57 ± 0.66 | 295 |
| Random | 5.02 ± 0.50 | 259 |
| ActiveEval-S+M | 5.29 ± 1.09 | 264 |
| Facility-location | 5.43 ± 0.40 | 287 |
| Matrix completion | 5.46 ± 1.20 | 250 |
| Greedy MI (Alg. 2) | 5.55 ± 0.98 | 288 |
| Active testing | 5.79 ± 1.14 | 232 |
| GRAD-MATCH | 6.20 ± 1.55 | 275 |

#### storage — checkpoint + cache GB (**amortized per unique model**)

| Method | Unseen MAE (pp) | #actions |
| ------ | --------------- | -------- |
| **Submod. benchmark** [^1] | **4.99 ± 0.79** | 251 |
| Bayesian opt. design | 5.22 ± 0.68 | 232 |
| ActiveEval-S | 5.60 ± 0.84 | 98 |
| Random | 5.75 ± 1.43 | 109 |
| Greedy MI (Alg. 2) | 6.02 ± 0.97 | 157 |
| ActiveEval-S+M | 6.17 ± 1.28 | 122 |
| Matrix completion | 6.30 ± 1.74 | 106 |
| GRAD-MATCH | 6.34 ± 0.41 | 104 |
| Active testing | 6.34 ± 0.86 | 128 |
| k-center | 6.46 ± 2.22 | 177 |
| Facility-location | 6.72 ± 1.28 | 133 |
| ActiveEval-Pair | 6.81 ± 0.75 | 91 |

> `Greedy entropy (Alg. 1)` isn't in this table ([^1]), but it's worth naming what
> would happen if it were: 5.07 ± 0.99 pp on 249 actions — right next to `Submod.
> benchmark` (4.99, 251 actions), not a coincidence, since they're the same algorithm.
> That near-exact match is itself the confirmation: `select_logdet` has always used a
> ridge-regularized objective `log det(I + K/sigma^2)` with `sigma=1.0`
> ([baselines/_core.py:138](baselines/_core.py#L138)), while `greedy_entropy`/`greedy_mi`
> previously defaulted to a near-zero jitter (`+1e-6`) — a much weaker prior that left
> them more sensitive to noise in the RBF-bandwidth estimate — and, in this experiment
> specifically, `greedy_entropy` was also run through an unnecessary `cap=250`
> candidate-pool restriction that only the genuinely cubic-cost `greedy_mi` needed.
> `sigma` now defaults to `1.0` for both ([baselines/_core.py:364](baselines/_core.py#L364),
> [:394](baselines/_core.py#L394)), and `greedy_entropy` runs uncapped like
> `select_logdet` ([experiments/run_acquisition_benchmark.py:413](experiments/run_acquisition_benchmark.py#L413)).
> Full ablation isolating each fix: [docs/rq4-cost-currency-mechanism.md](docs/rq4-cost-currency-mechanism.md).


**What the currencies say.** Under the additive budgets (`input_tok`, `output_tok`,
`latency`, `memory`) the ordering barely moves from `count` — ActiveEval-S/-Pair stay
on top — because a cost-agnostic order draws a representative cost mix, so 15% of the
cost buys ≈15% of the actions (~270) whatever the unit. The `storage` **budget
inverts it**, and the reason is a checkpoint accounting detail, not a difference in
selection quality: `storage_cost = (#unique models touched) × checkpoint_gb + (#actions) ×
small_cache_gb`, so a fixed GB budget buys actions almost in proportion to how few
*new* models the selection order forces you to pay for
([docs/rq4-cost-currency-mechanism.md](docs/rq4-cost-currency-mechanism.md) for the
full derivation). `Submod. benchmark`'s pair-diversity objective happens to satisfy
itself by revisiting the *same* model across several sample-sets (16 models bought 250
actions, ≈15.6/model) — it was never trying to minimize model count, but pair
diversity doesn't require model diversity here, so it stays cheap. `ActiveEval-Pair`'s
coverage objective is explicitly indifferent to model identity (it optimizes purely
for target-similarity), but the few pairs closest to the target happen to be spread
across many different models, so chasing them forces a new checkpoint every ~7-8
actions (12 models bought only 90 actions) — collapsing it to near-worst (6.81 pp)
despite leading almost every other table.

This is an honest limitation of *count*-optimal target-aware selection under a storage
budget, and it points at the deferred next step: a **cost-benefit** (gain/cost
knapsack) variant of ActiveEval — the production pipeline already threads a `cost_fn`
through the greedy loops (`active_evaluator/active_selection.py`) — that trades
coverage off against the amortized checkpoint cost instead of ignoring it.
(`outputs/cost_budget_benchmark.json`; only `greedy_mi` is capped at 250 via
`_GREEDY_CAP` — its per-step complement-precision solve is genuinely cubic in the
candidate-pool size, unlike `greedy_entropy`'s `O(N)` pivoted-Cholesky step, which runs
uncapped like `select_logdet`; orders precomputed to `_N_ORDER = 450`, realized cost
fraction recorded per cell.)
For the code-level mechanism behind each table (why the additive currencies barely
reorder while `storage` inverts the ranking), see
[docs/rq4-cost-currency-mechanism.md](docs/rq4-cost-currency-mechanism.md).

### Fixed-model-pool fairness check (RQ4 companion)

```bash
python -m experiments.run_acquisition_benchmark --mode fixed_model_pool --seeds 5 --budget-frac 0.15
```

The `storage` currency above rewards methods that concentrate picks on few
reference models (checkpoint cost is paid once per unique model), which is why
`Submod. benchmark`/`Bayesian opt. design` win it and `ActiveEval-*` collapses to
near-worst. This leaves it unclear whether ActiveEval is *penalized for
spreading across more models*, or simply worse at picking pairs. This check
controls for that directly: freeze the reference-model set `Submod. benchmark`
buys under the storage budget, then let every other method reselect its own
pairs restricted to **only that model set**, at the same action count. Each
method is also measured **Free** (unrestricted pool, same action count, paired
RNG) as the fairness baseline. `Delta = Fixed − Free`; positive means being
confined to `Submod. benchmark`'s models hurts that method.

**Results** (5 seeds, 15% of storage pool cost). `Submod. benchmark` bought a
median of **251 actions across 16 models** (source MAE 5.08 ± 0.95 pp). Unseen
MAE (pp), lower is better; sorted by delta.

| Method | Free (unrestricted) | Fixed (Submod.'s models) | Delta |
| --- | ---: | ---: | ---: |
| **GRAD-MATCH** | 5.72 ± 1.63 | **5.02 ± 0.65** | **−0.70** |
| Random | 5.62 ± 0.66 | 5.33 ± 0.82 | −0.29 |
| Submod. benchmark (source) | 5.08 ± 0.95 | 5.08 ± 0.95 | 0 (defines the model set) |
| k-center | 5.04 ± 0.53 | 5.11 ± 0.55 | +0.06 |
| Bayesian opt. design | 4.82 ± 0.85 | 5.17 ± 0.88 | +0.35 |
| Facility-location | 5.00 ± 0.44 | 5.36 ± 0.79 | +0.36 |
| **ActiveEval-Pair** | **4.45 ± 0.36** | 5.02 ± 0.93 | +0.58 |
| ActiveEval-S+M | 5.03 ± 0.58 | 5.62 ± 1.20 | +0.59 |
| ActiveEval-S | 4.47 ± 0.47 | 5.06 ± 0.76 | +0.59 |
| Active testing | 4.62 ± 0.72 | 5.26 ± 1.06 | +0.64 |
| Greedy MI (Alg. 2) | 4.83 ± 0.46 | 5.54 ± 0.80 | +0.71 |
| Matrix completion | 5.20 ± 0.45 | 5.95 ± 1.13 | +0.75 |

**Best result:** lowest MAE overall is **ActiveEval-Pair, Free = 4.45 ± 0.36 pp**
(best of every cell in the table). All three ActiveEval variants get worse
(+0.58 to +0.59 pp) when confined to `Submod. benchmark`'s 16-model set —
consistent with the RQ4 mechanism: part of ActiveEval's advantage comes from
spreading picks across many models to stay close to the target region, which
the model-concentrated pool takes away. GRAD-MATCH and Random go the other way
(negative delta) — restricting the pool doesn't hurt methods that weren't
exploiting model diversity to begin with. CIs overlap across most rows at 5
seeds, so treat the ranking as directional. Full analysis:
[docs/fixed-model-pool-fairness-check.md](docs/fixed-model-pool-fairness-check.md)
(`outputs/fixed_model_pool_benchmark.json`).

### Adaptive (threshold-stopped) budget — is there a free lunch under a fixed cap?

```bash
python -m experiments.run_acquisition_benchmark --mode threshold_budget --seeds 5 --budget-frac 0.20
```

Every table above hands each method a fixed budget and forces it to spend all
of it (saturated greedy loops were padded with random picks up to the cap).
This experiment asks the reverse question: given a hard cap (20%), can a method
**stop itself early** — once its own marginal gain (coverage/log-det/mutual-info,
whichever it already computes) drops below a `gain_threshold` fraction of its
first pick's gain — and spend less while still matching full-cap accuracy? A
new `gain_threshold` kwarg was added to every greedy selector that tracks a
decreasing per-step gain (`select_facility`, `select_logdet`,
`select_greedy_entropy`/`select_greedy_mi`, and the `ActiveEval-S`/`-S+M`/`-Pair`
selectors built from them — [baselines/_core.py](baselines/_core.py)); when set,
the greedy loop breaks early and the pre-existing "pad back up to budget" step
is skipped, so the method can genuinely under-spend the cap. Because each of
these is a greedy *prefix* selector, stopping early at a threshold is
mathematically identical to fixing the budget at the realized spend from the
start — the sweep just locates that point automatically.

**Results** (5 seeds, 20% budget cap, log-spaced thresholds `1e-4`–`0.3`):

| Method | Baseline (full 20% cap) | Cheapest match found | Verdict |
| --- | --- | --- | --- |
| **ActiveEval-S** | 4.02 ± 0.49 pp @ 20% | **4.33 ± 0.61 pp @ 12.3%** (`gain_threshold=1e-4`) | **Genuine free lunch** — 39% less budget, MAE within the full-cap CI. |
| Greedy MI (Alg. 2) | 5.10 ± 0.37 pp @ 20% | 4.74 ± 0.47 pp @ 13.9% | Looks free, but is a `cap=250` search-universe artifact, not real gain decay — see the doc below. |
| Submod. benchmark | 4.67 ± 0.38 pp @ 20% | — never under-spends | Diversity gain never decays enough in this range; always spends the full cap. |
| ActiveEval-Pair | 4.79 ± 0.87 pp @ 20% | — never under-spends | Same — its diversity phase absorbs whatever budget the coverage phase leaves it. |
| Facility-location | 4.94 ± 0.42 pp @ 20% | — decays too fast | Collapses to ≤6.5% spend with much worse MAE at the smallest tested threshold; no usable middle ground. |
| ActiveEval-S+M | 4.61 ± 0.32 pp @ 20% | — decays too fast | Same pattern as Facility-location. |

**ActiveEval-S is the only genuine result** — its realized spend strictly
decreases as the threshold grows (20% → 12.3% → 8.5% → 3.7% → ...), a real
gain-decay curve, and its cheapest match uses ~39% less budget for
statistically indistinguishable accuracy. Greedy MI's apparent under-spend is a
measurement artifact worth flagging honestly: its realized spend is *identical*
(13.9%) across all 8 nonzero thresholds tested, because its search universe was
already capped at 250 candidates for tractability — the gain threshold never
actually triggered a break in this run, the pre-existing cap did. Full
per-threshold tables, the cap-artifact explanation, and reading notes:
[docs/adaptive-threshold-budget.md](docs/adaptive-threshold-budget.md)
(`outputs/threshold_budget_benchmark.json`).

### Representative-subset selection — which *sample-sets* should we pay to label at all?

```bash
python -m experiments.run_acquisition_benchmark --mode distance_formula_comparison --seeds 5
```

Every table above budgets **(model, sample-set) pairs** out of a fixed candidate
pool. This experiment asks a stage *earlier* question: getting a model's accuracy
on a sample-set is expensive (you must run every example in it), but a sample-set's
*shift descriptor* is cheap (no model run). So before spending any labeling budget,
can we cheaply pick the $K$ sample-sets that best **represent the whole
meta-dataset** — and only ever run models against those? This is deliberately
**target-blind**: the objective is "represent everything", not "resemble the
deployment target".

Each candidate sample-set becomes a point *cloud*; the selector needs a
**distance between two clouds**. Six selection rules are compared, all at
$K{=}60$ representatives (3600 of 6780 possible triples, a **47% cut** in
expensive evaluation actions):

| Method | Unseen MAE (pp) | Stage-1 select | Guarantee |
| --- | --- | ---: | --- |
| Full source pool (reference, 6780 triples) | **3.68 ± 0.63** | — | — |
| **Hausdorff** (worst-case point coverage) | **3.81 ± 0.65** | 0.88s | (1−1/e) |
| sum, Hausdorff-weighted (2:1:1) | 4.16 ± 0.78 | 2.86s | (1−1/e) |
| **Kernel herding** (global MMD matching) | **4.19 ± 0.90** | **0.09s** | none (O(1/T) rate) |
| Sliced Wasserstein | 4.20 ± 0.47 | 1.38s | (1−1/e) |
| *Random subset (control)* | *4.41 ± 0.76* | — | — |
| K-means (M=K, reps=1) | 4.61 ± 1.11 | 0.09s | none |
| Kernel mean (MMD) | 4.65 ± 0.89 | 0.59s | (1−1/e) |
| ProbCover (hard δ-ball coverage) | 5.09 ± 1.59 | 0.58s | none |

**Hausdorff wins** and nearly matches the full pool while labeling 47% fewer
triples. **Kernel herding is the standout efficiency result** — within 0.38 pp of
Hausdorff (overlapping CIs) at **~9× less selection compute**, and it needs no
metric-selection tuning. **ProbCover is the only method worse than random.**
K-means, the heuristic this whole line of work started from, is beaten by four of
the six principled rules.

**Where the time actually goes** (full end-to-end, 9 configs × 5 seeds = 36.3 min):
Stage-1 selection — the part these methods change — is a *rounding error*. The
`Facility-location` **baseline acquisition** call dominates at ≈28.8s per config
(15–19× `ActiveEval-Pair`'s ≈1.7s) because `cover_all=True` scores every greedy
round against the entire pool; on the full pool it balloons to **169.3s**, and that
single row is ~39% of the whole experiment. There are **no LLM calls anywhere** in
this benchmark, so *compute tokens are not applicable* — CPU wall-clock is the only
real cost unit.

⚠️ **The kernel-herding speed advantage is regime-specific and reverses on
realistic data.** It holds at this benchmark's 20 points/sample-set, but the kernel
Gram is $O(M^2)$ in points-per-sample-set while SciPy's Hausdorff is ~$O(M)$
average-case; measured crossover is $M\approx100$, and at $M{=}640$ Hausdorff is
**40× faster** than herding. Real workloads have hundreds of examples each, so
expect this ranking to flip.

**Why Hausdorff wins — and why it *lost* at $K{=}30$.** The two families optimise
**logically independent** guarantees (proved by counterexample in §29.5):
Hausdorff-induced facility location controls the **covering radius**
$\varepsilon(A)=\max_j\min_{i\in A}d_H(b_i,b_j)$, which bounds *worst-case*
label transfer uniformly over every unlabelled candidate
($\max_j|a^\star(b_j)-a^\star(\mathrm{rep}(j))|\le L\,\varepsilon(A)$, Theorem 2);
kernel herding controls $\lVert\mu_B-\mu_A\rVert$, which bounds only the
*pool-average* accuracy (Theorem 3). Measuring $\varepsilon(A)$ directly shows
Hausdorff has **no** coverage edge at $K{=}15$, a negligible 1.9% edge at
$K{=}30$, and a decisive 9–34% edge from $K{=}60$ — tracking its MAE record
(worst method at $K{=}30$, best at $K{=}60$) exactly. Caveat: covering radius
explains the extremes but **not** the middle of the ranking (§29.7).

Full derivations, proofs, profiling, and honest limits:
[docs/representative_meta_dataset_facility_location_full.md](docs/representative_meta_dataset_facility_location_full.md)
§27.9 (results), §28 (why herding is fast *and* accurate), §29 (what Hausdorff
provably guarantees), §30 (why herding's cost advantage reverses as $M$ grows —
$\Omega(M^2)$ vs $O(M)$, with the crossover *predicted* at $M^\star{=}82.4$ and
measured at 80) — `outputs/herding_probcover_comparison_K60_timed.json`.

## Quick start — reproduce the image-classification acquisition benchmark (CPU, no downloads)

```bash
python -m experiments.run_image_classification_benchmark --seeds 5 --budget-frac 0.15
```

Re-themes the same self-contained benchmark for the image-classification side of the
paper (CIFAR→TinyImageNet-style transfer over the README's 43-model pool: Classic CNN,
VGG/Residual/Wide/Dense, Efficient Mobile CNNs, Scaled/Lightweight CNNs, CIFAR-Standard
Robust Baselines). It reuses the exact selection math and meta-evaluator — `ActiveEvaluator`
(`active_evaluator/model.py`) and `baselines.ACQUISITION_REGISTRY` / `ESTIMATOR_REGISTRY`
are imported unchanged from the Text2SQL script — with `n_train_models=35` +
`n_unseen_models=8` seen/held-out classifiers (43 total) and sample-sets framed as
distribution-shift evaluation slices instead of Text2SQL workload slices. Same `--mode sweep|ablation|entropy_mi_sweep` options as the Text2SQL script (RQ3 budget curve, RQ5
ablation, greedy entropy vs. greedy MI). Regenerate with the command above
(`outputs/image_classification_acquisition_benchmark.json`); numbers are illustrative of
the design until replaced with a real torchvision/timm pipeline for camera-ready figures.

**Measured results** (5 seeds, 15% labeling budget, unseen-model MAE in percentage
points; lower is better). This variant uses fewer reference models (`n_train_models=35`),
so it is noisier and the budgeted methods are harder to separate (CIs ≳1 pp). Under the
held-out-target setup the full MetaEvaluator leads; among budgeted methods k-center is
narrowly best here and ActiveEval sits mid-pack — a reminder that the target-proximity
signal is weaker when the pool is small and noisy. Treat these as illustrative until
replaced by a real torchvision/timm pipeline.


| Method                  | Family      | Unseen MAE (pp) | Cost |
| ----------------------- | ----------- | --------------- | ---- |
| MetaEvaluator (full)    | reference   | **4.54 ± 1.10** | 100% |
| DoC                     | estimator   | 4.69 ± 0.50     | —    |
| k-center                | acquisition | 5.65 ± 1.27     | 15%  |
| **ActiveEval-S+M**      | ours        | 5.84 ± 1.72     | 15%  |
| **ActiveEval-S**        | ours        | 6.00 ± 1.84     | 15%  |
| Submod. benchmark [^1]  | acquisition | 6.20 ± 1.24     | 15%  |
| **ActiveEval-Pair**     | ours        | 6.20 ± 1.46     | 15%  |
| Random                  | acquisition | 6.39 ± 1.67     | 15%  |
| Greedy MI (Alg. 2)      | acquisition | 6.52 ± 1.18     | 15%  |
| Matrix completion       | acquisition | 6.58 ± 0.85     | 15%  |
| Bayesian opt. design    | acquisition | 6.66 ± 0.99     | 15%  |
| Active testing          | acquisition | 6.93 ± 1.34     | 15%  |
| Facility-location       | acquisition | 7.42 ± 2.03     | 15%  |
| GRAD-MATCH              | acquisition | 7.91 ± 0.64     | 15%  |
| ATC                     | estimator   | 8.28 ± 2.14     | —    |


Regenerate with the command above (`outputs/image_classification_acquisition_benchmark.json`).

## Quick start — reproduce the node-classification acquisition benchmark (CPU, no downloads)

```bash
python -m experiments.run_node_classification_benchmark --seeds 5 --budget-frac 0.15
```

Re-themes the same self-contained benchmark for node classification: reference models are
GNN architecture families (GCN, GraphSAGE, GAT, GIN, ChebNet, SGC, APPNP, JKNet, etc.) and
sample-sets are graph-shift evaluation slices (e.g. homophily/heterophily buckets,
feature-noise levels, or a citation-network-style transfer such as Cora → CiteSeer/PubMed)
rather than Text2SQL workload slices or image corruption slices. It reuses the exact
selection math and meta-evaluator — `ActiveEvaluator` (`active_evaluator/model.py`) and
`baselines.ACQUISITION_REGISTRY` / `ESTIMATOR_REGISTRY` are imported unchanged from the
Text2SQL script. No fixed model-pool size is documented for node classification elsewhere
in this repo, so it keeps the original script's `n_train_models=60` / `n_unseen_models=8`
defaults. Same `--mode sweep|ablation|entropy_mi_sweep` options as the other two scripts
(RQ3 budget curve, RQ5 ablation, greedy entropy vs. greedy MI). Regenerate with the command
above (`outputs/node_classification_acquisition_benchmark.json`); numbers are illustrative
of the design until replaced with a real PyG/DGL pipeline for camera-ready figures.

**Measured results** (5 seeds, 15% labeling budget, unseen-model MAE in percentage
points; lower is better). With the same `n_train_models=60` as the Text2SQL problem,
this variant tracks it closely: the full MetaEvaluator leads and ActiveEval-S/-Pair are
the strongest budgeted methods, ahead of whole-pool Random and every other baseline.


| Method                  | Family      | Unseen MAE (pp) | Cost |
| ----------------------- | ----------- | --------------- | ---- |
| MetaEvaluator (full)    | reference   | **3.60 ± 0.47** | 100% |
| **ActiveEval-S**        | ours        | 4.08 ± 0.68     | 15%  |
| **ActiveEval-Pair**     | ours        | 4.11 ± 0.39     | 15%  |
| DoC                     | estimator   | 4.39 ± 0.50     | —    |
| Random                  | acquisition | 4.47 ± 0.90     | 15%  |
| Bayesian opt. design    | acquisition | 4.61 ± 0.67     | 15%  |
| Submod. benchmark [^1]  | acquisition | 4.64 ± 0.22     | 15%  |
| k-center                | acquisition | 4.96 ± 0.51     | 15%  |
| Facility-location       | acquisition | 5.30 ± 0.48     | 15%  |
| Matrix completion       | acquisition | 5.38 ± 0.90     | 15%  |
| Greedy MI (Alg. 2)      | acquisition | 5.45 ± 0.79     | 15%  |
| **ActiveEval-S+M**      | ours        | 5.66 ± 1.11     | 15%  |
| GRAD-MATCH              | acquisition | 6.00 ± 1.48     | 15%  |
| Active testing          | acquisition | 6.10 ± 0.98     | 15%  |
| ATC                     | estimator   | 6.84 ± 3.37     | —    |


Regenerate with the command above (`outputs/node_classification_acquisition_benchmark.json`).

## Baseline method library

The paper compares against three families; each lives under [baselines/](baselines/)
with a `select`/`estimate` entry point. Implemented methods are in **bold**.


| Family                | Methods                                                                                                                                                                                                                          |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Label-free estimators | **ATC**, **DoC**, AutoEval, AETTA, SSME                                                                                                                                                                                          |
| Budgeted acquisition  | **Random**, **k-center**, **Facility-location**, **Matrix completion**, **Active testing**, **Bayesian optimal design**, **Submodular benchmark selection**, **GRAD-MATCH**, **Greedy entropy (Alg. 1)** [^1], **Greedy MI (Alg. 2)** |
| Reference / ours      | **MetaEvaluator (full budget)**, **ActiveEval-S**, **ActiveEval-S+M**, **ActiveEval-Pair**                                                                                                                                       |


The production Text2SQL pipeline (`active_evaluator/active_selection.py`) implements
the same facility-location / GRAD-MATCH / knapsack-budget selection on real
shift descriptors from cached LLM embeddings.

ActiveEvaluator Training Pipeline

## What's new in this version

The repository has been renamed from `meta_evaluator` to `active_evaluator` and a new active-selection module has been added. The base pipeline (shift descriptors → meta-trained MLP predictor) is unchanged; the new code is a strictly additive layer that runs *between* meta-training and held-out evaluation.


| Concern                               | Module                                                                       | Class / function                                                       |
| ------------------------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| Budget arithmetic and per-round split | [active_evaluator/budget.py](active_evaluator/budget.py)                     | `BudgetPlan`, `compute_budget`                                         |
| Selection algorithms (V1 + V2)        | [active_evaluator/active_selection.py](active_evaluator/active_selection.py) | `select_extension`, `lazy_greedy_facility`, `direct_greedy_validation` |
| Pipeline wiring + CLI flags           | [active_evaluator/pipeline.py](active_evaluator/pipeline.py)                 | `extend_test_tasks_with_active_selection`                              |
| Unit tests for selection              | [test/test_active_selection.py](test/test_active_selection.py)               | 13 tests                                                               |
| Unit tests for budget                 | [test/test_budget.py](test/test_budget.py)                                   | 8 tests                                                                |


The previously-existing modules ([active_evaluator/meta_learning.py](active_evaluator/meta_learning.py), [active_evaluator/model.py](active_evaluator/model.py), [active_evaluator/pipeline.py](active_evaluator/pipeline.py), [shift_descriptor/](shift_descriptor/)) were renamed but kept their behavior. When `--use-active-selection` is *not* passed, the pipeline runs identically to before.

## Repository Layout

- [active_evaluator/](active_evaluator/) — Meta-trained MLP predictor + active-selection module.
  - `active_selection.py` — V1 lazy facility location, V2 direct loss reduction, narrowing, influence weights.
  - `budget.py` — `BudgetPlan` dataclass and `compute_budget` (fraction / absolute / min-budget).
  - `pipeline.py` — End-to-end CLI; the test-time block calls `extend_test_tasks_with_active_selection` when the flag is set.
  - `meta_learning.py`, `model.py` — CAVIA-style meta-learner and 3-layer MLP predictor (unchanged behavior, renamed classes `ActiveEvaluator` / `ActiveEvaluatorLearner`).
- [shift_descriptor/](shift_descriptor/) — Frechet / Mahalanobis / Sliced Wasserstein descriptors from cached embeddings.
- [prompts/](prompts/) — Text2SQL prompt templates.
- [scripts/](scripts/) — Helpers (`inspect_active_evaluator.py`, `rename_model_outputs.py`).
- [test/](test/) — 24 unit tests (`test_active_selection.py`, `test_budget.py`, `test_active_learning.py`).

## Requirements

```bash
pip install torch transformers accelerate bitsandbytes scipy scikit-learn matplotlib tqdm numpy peft
```

`bitsandbytes` and `accelerate` are required for the 4-bit quantized SQL generator. Some checkpoints are gated on Hugging Face — pass `alias=model_id` to `--model-ids` to point at approved variants.

## Active Selection — the new contribution

### Problem statement

At test time, given a held-out *unseen* model with one labeled support pair `(descriptor, accuracy)`, we want the predictor's K-step inner adaptation to be as accurate as possible on a held-out target split. The base pipeline adapts on that single pair. ActiveEvaluator instead:

1. Builds a candidate pool **U** = labeled support pairs from training models, disjoint from the validation set **V**.
2. Narrows **V** to a target-aware subset **V_T** using a Gaussian similarity kernel against the unlabeled target features **T**.
3. Runs `n_rounds` of active selection, each round adding `B / n_rounds` examples to the extension set **S** ⊆ U.
4. Final adaptation runs on **S₀ ∪ S** (where **S₀** is the original support pair).

The invariant **S ∩ V = ∅** is enforced by an `assert` after every round.

### Two selection algorithms

**V1 — Influence-weighted facility location** (`--selection-method v1_facility`, default)

$$\tilde f_1(S) = \sum_{v \in V_T} I(v) \cdot \max_{s \in S \cup S_0} \exp\left(-\frac{\phi(v) - \phi(s)^2}{\tau}\right)$$

where `I(v) = ||∇φ L(h_φ; v)||₂` is the per-example gradient-norm influence weight and `τ` is the median-heuristic bandwidth. Monotone submodular by construction → lazy greedy is provably equivalent to plain greedy and gives a `(1 − 1/e)` approximation under cardinality (Nemhauser, Wolsey & Fisher 1978; Minoux 1978).

**V2 — Direct validation-loss reduction** (`--selection-method v2_direct`)

$$f_2(S) = \mathcal{L}(h_{\phi_K(S_0)}; V_T) - \mathcal{L}(h_{\phi_K(S_0 \cup S)}; V_T)$$

Non-submodular and non-monotone — uses plain greedy with a positive-gain abort rule. Each candidate evaluation requires a full K-step adaptation, so candidates are pre-filtered to the top `--selection-max-candidates-evaluated` by V1's surrogate (FASS-style filtering, Wei, Iyer & Bilmes 2015). This is an empirical upper-bound oracle.

### CLI flags


| Flag                                   | Default              | Meaning                                       |
| -------------------------------------- | -------------------- | --------------------------------------------- |
| `--use-active-selection`               | off                  | Enable the new test-time branch.              |
| `--selection-method`                   | `v1_facility`        | `v1_facility` or `v2_direct`.                 |
| `--selection-n-rounds`                 | `5`                  | Active-learning rounds.                       |
| `--selection-budget-fraction`          | `0.10`               | Total budget as a fraction of `               |
| `--selection-budget-absolute`          | `None`               | Absolute budget; overrides fraction when set. |
| `--selection-K-steps`                  | `--eval-inner-steps` | Inner adaptation steps used for V2 oracle.    |
| `--selection-narrowing-quantile`       | `0.7`                | Keep V examples with ρ ≥ this quantile.       |
| `--selection-max-candidates-evaluated` | `100`                | V2 FASS pre-filter cap.                       |
| `--selection-seed`                     | `42`                 | RNG seed (bandwidth pair sampling, ties).     |


### Output artifacts

When `--use-active-selection` is passed, per-test-model diagnostics are written to `outputs/<output-dir>/selection_<sanitized_model_id>/`:

- `budget_plan.json` — total budget, `n_rounds`, per-round split, fraction/absolute used, `cost_fn_name`.
- `narrowing_diagnostics.json` — `|V|`, `|V_T|`, `q`, ρ threshold/mean/min/max.
- `selection_trajectory.json` — per round, the full pick log:
  - `selected_keys` and `selected_source_models` (training-model alias parsed from the candidate key).
  - `picks[]` — one entry per selected candidate. For V1: `marginal_gain`, `cost`, `cumulative_cost_in_round`, `max_sim_to_VT`, `mean_sim_to_VT`, `most_influential_v_index`, `rank_in_round`, `source_model`. For V2: `loss_before`, `loss_after`, `loss_reduction`, `marginal_gain` (per-cost), `n_candidates_evaluated` (FASS pre-filter cap), `cost`, `rank_in_round`, `source_model`.
  - `round_cost_paid`, `round_gain_total`, `influence_stats` (mean/min/max/n of I(v) over V_T).
  - `val_loss_before`, `val_loss_after`, `elapsed_seconds`, `timings` (adapt/embed/influence/greedy seconds).
- `selected_examples.json` — final extension set: list of `{key, source_model, true_label, cost}` records plus the bare `selected_keys` list.
- `selection_summary.json` — top-level recap with: `method`, `tau`, `cost_fn_name`, `budget_total`, `cost_paid_total`, `cost_remaining`, `gain_total`, `n_selected`, `pool_size`, `val_size`, `val_VT_size`, `final_val_loss`, `total_selection_seconds`, `n_rounds_executed`, `source_models_chosen` (the chosen training-model aliases, in pick order), and a human-readable `rationale` paragraph explaining the V1/V2 selection criterion.

These logs answer "**what** got selected" (`source_models_chosen`, `selected[*].source_model`, `selected[*].true_label`), "**why** it got selected" (`picks[*].marginal_gain`, `picks[*].max_sim_to_VT`, `influence_stats`, `rationale`), and "**at what cost**" (`cost_paid_total`, `cost_remaining`, `picks[*].cost`, `picks[*].cumulative_cost_in_round`, `total_selection_seconds`).

### Run example

```bash
python -m active_evaluator.pipeline \
  --train-path data/sft_spider_train_text2sql.json \
  --dev-path data/sft_spider_dev_text2sql.json \
  --output-dir outputs/run_v1 \
  --max-train-samples 1500 --max-dev-samples 300 \
  --use-active-selection \
  --selection-method v1_facility \
  --selection-n-rounds 5 \
  --selection-budget-fraction 0.30 \
  --selection-narrowing-quantile 0.5 \
  --model-ids cycloneboy/SLM-SQL-0.5B Qwen/Qwen2-0.5B ... \
  --test-model-ids Qwen/Qwen2.5-0.5B-Instruct ...
```

### Theoretical guarantees


| Algorithm                   | Constraint             | Guarantee             | Reference                                    |
| --------------------------- | ---------------------- | --------------------- | -------------------------------------------- |
| V1 lazy greedy              | cardinality (`c(s)=1`) | `(1 − 1/e)` of OPT    | Nemhauser, Wolsey & Fisher (1978)            |
| V1 cost-benefit greedy      | knapsack               | `½(1 − 1/e)` of OPT   | Khuller, Moss & Naor (1999)                  |
| V1 with partial enumeration | knapsack               | `(1 − 1/e)` of OPT    | Sviridenko (2004)                            |
| V2 plain greedy             | cardinality            | none (non-submodular) | Wei, Iyer & Bilmes (2015) — empirical oracle |


## Tests

24 unit tests cover correctness invariants (no real-data MAE):

```bash
python -m pytest test/ -v
```


| Group             | Coverage                                                                                                                                              |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Budget arithmetic | fraction/absolute/min-floor/cap, JSON round-trip, consistency validation.                                                                             |
| V1 algorithmic    | lazy-greedy ≡ plain-greedy on synthetic 20×5 pool; submodularity (marginal gains non-increasing); disjointness `S ∩ V = ∅`; budget cap; log emission. |
| V2 algorithmic    | budget cap; FASS pre-filter cap (verified via `eval_trace` hook); positive-gain abort rule; pipeline-wrapper smoke.                                   |
| Shared machinery  | embedding shape; bandwidth scaling; narrowing produces `(1 − q) ·                                                                                     |


## Base Pipeline (unchanged behavior)

The base pipeline still:

1. Splits a labeled Text2SQL JSON into `meta_train` / `meta_val` / `meta_test`.
2. Runs each Text2SQL LLM in `--model-ids` to generate SQL on each split, computes execution accuracy.
3. Computes shift descriptors (Frechet, Mahalanobis, Sliced Wasserstein) from cached embeddings.
4. Meta-trains a 3-layer MLP predictor mapping `(model_descriptor, distribution_descriptor) → predicted accuracy`.
5. At test time, performs K context-adaptation steps on the held-out model's support descriptor and predicts on `meta_test` and the held-out target split.

```bash
python -m active_evaluator.pipeline \
  --train-path data/sft_spider_train_text2sql.json \
  --dev-path data/sft_spider_dev_text2sql.json \
  --output-dir outputs/run_baseline \
  --max-train-samples 1500 --max-dev-samples 300 \
  --model-ids ... \
  [--test-model-ids ...]
```

## Shift Descriptor Pipeline (auxiliary)

```bash
python -m shift_descriptor.pipeline \
  --train-path data/<train_filename>.json \
  --test-path data/<test_filename>.json \
  --output-dir outputs
```

Extracts pooled embeddings, computes descriptor metrics (Frechet, Mahalanobis, SWD), and reports similarity diagnostics. Outputs include embedding caches, descriptor matrices, PCA plots, and similarity diagnostics.

Key arguments: `--model-ids`, `--lora-r`, `--metric-max-points`, `--prompt-template`, `--context-fields`, `--use-plain-text`, `--text-field`, `--device`. Suffix `:remote` (e.g. `mamba=state-spaces/mamba-2.8b-slimpj:remote`) enables `trust_remote_code` for architectures like Mamba and RWKV.

## Loading Models With This Repo

The pipeline uses Hugging Face causal language models for SQL generation. For Text2SQL LLMs, pass the model IDs via `--model-ids`. Example lightweight models are already in [active_evaluator/pipeline.py](active_evaluator/pipeline.py).

Encoder-decoder Text2SQL models (T5/BART/PICARD/CodeT5p) and structured parsers (RAT-SQL, LGESQL, SmBoP) require custom inference wrappers. Run their generation and execution accuracy separately and plug the accuracy map into the pipeline by extending `run_inference_for_models` or loading a precomputed `model_accuracies.json`.

For image classification backbones, standard PyTorch loaders (e.g. `torchvision.models.resnet50(weights="DEFAULT")` or `timm.create_model("vit_tiny_patch16_224", pretrained=True)`) can be used to produce embeddings and descriptors in a vision-oriented pipeline.

## Mathematical References

- Nemhauser, Wolsey & Fisher (1978). "An analysis of approximations for maximizing submodular set functions." *Math. Programming* 14(1):265–294.
- Khuller, Moss & Naor (1999). "The budgeted maximum coverage problem." *Information Processing Letters* 70(1):39–45.
- Sviridenko (2004). "A note on maximizing a submodular set function subject to a knapsack constraint." *Operations Research Letters* 32(1):41–43.
- Minoux (1978). "Accelerated greedy algorithms for maximizing submodular set functions."
- Mirchandani & Francis (1990). *Discrete Location Theory*.
- Wei, Iyer & Bilmes (ICML 2015). "Submodularity in Data Subset Selection and Active Learning."
- Park, Park & Lee (ICLR 2025). "Active Learning for Continual Learning: Keeping the Past Alive in the Present."

## Legacy benchmarks (pre-active-selection)

The following tables report MAE numbers from the original MetaEvaluator paper (Text2SQL Spider→BIRD and image classification CIFAR→TinyImageNet transfer). They reflect the *base* predictor without active selection and are kept here for reference. New active-selection MAE numbers will be added once the scaled-up benchmark run completes.

### Text2SQL Model Pool (78 Total)


| Category                            | Count | Families and Models                                                                                                            |
| ----------------------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------ |
| Structured Text2SQL Parsers         | 8     | RAT-SQL; LGESQL; SmBoP; RESDSQL; Clause-SmBoP; IRNet; BRIDGE; ValueNet / RYANSQL                                               |
| Encoder-Decoder Models              | 10    | PICARD T5; FLAN-T5; BART NL2SQL; CodeT5p-770M; FronyAI/natural2sql-ko                                                          |
| SQLCoder / SLM-SQL / CscSQL / Hrida | 24    | SQLCoder family; SLM-SQL family; CscSQL-Merge / CscSQL-Grpo; Hrida-T2SQL                                                       |
| Other LLMs                          | 10    | DeepSeek-Coder; Snowflake Arctic-Text2SQL; DeepSeek-R1-Distill; WizardCoder; Mistral; Llama-3.1; Qwen2.5                       |
| Modern General LLM Backbones        | 26    | DeepSeek-V3; OLMo-2; gemma-3; Mistral-Small-3.1; Llama-4; Qwen3; SmolLM3; Kimi-K2; GPT-OSS; GLM-4.5/4.6; MiniMax; RWKV; Mamba2 |


### Image Classification Model Pool (43 Total)


| Category                        | Count | Families                                                                                       |
| ------------------------------- | ----- | ---------------------------------------------------------------------------------------------- |
| Classic CNN                     | 2     | LeNet-5; AlexNet                                                                               |
| VGG / Residual / Wide / Dense   | 17    | VGG-11..19; ResNet-18..152; WideResNet; DenseNet                                               |
| Efficient Mobile CNNs           | 10    | MobileNet-V1..V3; ShuffleNet-V2; EfficientNet-B0..B2                                           |
| Scaled / Lightweight CNNs       | 5     | SqueezeNet; RegNet                                                                             |
| CIFAR-Standard Robust Baselines | 9     | ResNet-20/56/110; PreAct-ResNet; PyramidNet; Shake-Shake; ResNeXt-29; DenseNet-BC; MobileNetV2 |


Full per-model MAE tables (DoC / ATC / AGD / PseudoAutoEval / AutoEval / NL2SQL-BUGS / SelfTrainEns vs. ActiveEvaluator base) are preserved in the git history of this README prior to commit `<active-selection-rewrite>`.
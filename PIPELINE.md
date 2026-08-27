# ActiveEvaluator — Full Pipeline Reference

This document describes, end to end and at implementation detail, how every
part of this repository fits together: what runs first, what data flows into
what, which files own which piece of math, and how the experimental artifacts
under `outputs/` and `logs/` were produced. It complements `README.md` (which
is user/results-facing) with an internals-facing map of the code.

---

## 1. The problem in one paragraph

We have a pool of **reference models** (e.g. small Text2SQL LLMs) and labeled
**support workloads** (prompt splits with known execution accuracy). We want
to predict the accuracy of **future, unseen models** on an **unlabeled
target** workload. Running a model against a workload to get a labeled
accuracy point is the expensive "evaluation action" we are budgeting. Given a
hard labeling budget, which (reference model, workload) pairs should we
actually run, so that a small predictor trained only on those acquired labels
generalizes best to new models? This is *supervision acquisition* for a
meta-evaluator — distinct from model selection (deployed model is fixed),
benchmark compression (train and test models are the same population), and
active testing (the target itself stays unlabeled, we never label it).

---

## 2. Two parallel tracks in the repo

The repository actually contains **two independent implementations** of the
same idea, at two different levels of realism:

1. **`active_evaluator/` + `shift_descriptor/`** — the *production* pipeline.
   Runs real Hugging Face LLMs on real Text2SQL (Spider) data, extracts real
   embeddings, computes real distribution-shift descriptors, meta-trains a
   real predictor, and (optionally) runs real active selection over real
   model/workload pairs. This is slow (needs GPUs, model downloads) and is
   what `logs/launch_v1.sh` / `run.sh` / `run_extended.sh` drive.

2. **`experiments/run_acquisition_benchmark.py` + `baselines/`** — a
   *synthetic, CPU-only, seconds-to-run* reproduction of the same selection
   math and the same `ActiveEvaluator` MLP class, but on a fabricated
   latent-factor problem instead of real LLMs. This is what produces the
   headline numbers in `README.md`'s results table and the RQ3/RQ5 tables
   (`outputs/acquisition_benchmark.json`, `outputs/acquisition_sweep.json`,
   `outputs/acquisition_ablation.json`). It exists so the paper's *ordering*
   of methods can be checked without a GPU cluster, and so every baseline
   (including ones never wired into the real pipeline, like Bayesian design
   or matrix completion) can be compared on equal footing.

Both tracks share one real module: `active_evaluator/model.py`'s
`ActiveEvaluator` MLP. Everything else in the synthetic track is a
self-contained NumPy re-implementation of the selection math that lives for
real in `active_evaluator/active_selection.py`.

---

## 3. Production pipeline — stage by stage

Entry point: `python -m active_evaluator.pipeline` (`active_evaluator/pipeline.py:main`).

```
train JSON, dev JSON
        │
        ▼
┌───────────────────┐
│ 1. Load + prompt   │  data_utils.py: load_raw_dataset, load_prompted_texts
│    + split         │  build_prompt_splits → meta_train / meta_val / meta_test (60/20/20 default)
└───────────────────┘  dev JSON is kept whole as the "dev"/real-test split
        │
        ▼
┌───────────────────┐
│ 2. SQL generation  │  generation.py: SQLGenerator (4-bit quantized HF causal LM, optional LoRA)
│    + execution acc │  evaluation.py: build_prediction_records (exact match + SQLite execution match)
└───────────────────┘  → model_accuracies.json  {model: {meta_val, meta_test, dev: accuracy}}
        │
        ▼
┌───────────────────┐
│ 3. Embeddings      │  embedding_cache.py: EmbeddingCache (on-disk .npz cache per model×split)
│                    │  shift_descriptor/embeddings.py: EmbeddingExtractor (mean-pooled last hidden state)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ 4. Shift           │  descriptors.py: compute_shift_descriptor(meta_train vs meta_val/meta_test/dev)
│    descriptors     │  shift_descriptor/metrics.py: Fréchet, Mahalanobis, Sliced-Wasserstein
└───────────────────┘  → one 6-dim feature vector per model per (support/query/transfer) pair
        │
        ▼
┌───────────────────┐
│ 5. Build tasks     │  pipeline.py: build_tasks → ShiftDescriptorTask per model
│                    │  (support_descriptor/label = meta_train↔meta_val; query = meta_train↔meta_test;
│                    │   transfer = meta_train↔dev). Descriptors z-normalized across all tasks.
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ 6. Meta-training   │  meta_learning.py: ActiveEvaluatorLearner.meta_train (CAVIA-style)
└───────────────────┘  → active_evaluator_model.pt (best val-MAE checkpoint)
        │
        ▼
┌───────────────────┐
│ 7. Eval on seen    │  meta_learner.evaluate (meta-test) + evaluate_transfer (dev/real-test)
│    models          │  → active_evaluator_metrics.json, *_predictions.json
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ 8. Held-out models │  build_tasks() again on --test-model-ids (never seen during meta-training)
│    (optional       │  IF --use-active-selection: extend_test_tasks_with_active_selection()
│    active          │    runs V1/V2/V3 selection per held-out model, concatenates picked
│    selection)       │    (descriptor, label) pairs onto that model's support tensors
└───────────────────┘  → active_evaluator_test_{meta,dev}_predictions.json
```

### 3.1 Data & prompting — `active_evaluator/data_utils.py`

- `load_raw_dataset`: reads a JSON list of Text2SQL examples (question,
  gold `sql`, `db_id`/`db_path`, optional `evidence`/`matched_contents`).
- `load_prompted_texts`: renders each example through the repo's prompt
  template (`prompts/`, resolved via `shift_descriptor.prompts.default_template_path`)
  using `context_fields` (defaults: `evidence`, `matched_contents`, `text`),
  or falls back to a raw `text_field` when `--use-plain-text` is set.
- `build_prompt_splits`: a **seeded permutation split** (`np.random.default_rng(seed)`)
  of the training file into `meta_train` / `meta_val` / `meta_test` at
  configurable ratios (default `0.6/0.2/0.2`). The dev JSON is *not* split —
  it's wrapped whole as a `PromptSplit` named `"dev"` (`create_dev_split`) and
  used as the "real transfer target" split.
- `maybe_cap_examples`: truncates to `--max-train-samples` / `--max-dev-samples`
  for fast iteration.

### 3.2 SQL generation & execution accuracy

- `generation.py::SQLGenerator` loads each HF causal LM in **4-bit
  (bitsandbytes NF4) quantization**, optionally wraps it in a **LoRA adapter**
  (`--lora-r`, targeting whichever of `q_proj/k_proj/v_proj/o_proj/gate_proj/
  up_proj/down_proj/...` modules exist on that architecture), and greedily (or
  nucleus-) decodes SQL for every prompt in a split.
- `evaluation.py::build_prediction_records` scores each prediction two ways:
  - **exact_match**: normalized string equality (`normalize_sql`: lowercase,
    strip trailing `;`, collapse whitespace).
  - **execution_correct**: runs both gold and predicted SQL against the
    example's SQLite DB (`SQLiteExecutor`, read-only `file:...?mode=ro`
    connections, cached per DB path) and compares **sorted row tuples** — this
    is the metric that matters (`execution_accuracy`, reported as the model's
    "accuracy" everywhere downstream).
- `pipeline.py::run_inference_for_models` runs this per model over
  `{meta_val, meta_test, dev}`, is **resumable** (`maybe_load_cached_metrics`
  skips a split if a cached prediction file with the right sample count
  already exists), and is **fault-isolated** (one model's OOM/download
  failure is caught, logged, and skipped — it just won't get a
  `ShiftDescriptorTask` later). Output: `outputs/<run>/predictions/<model>/
  {meta_val,meta_test,dev}.json` and the aggregated `model_accuracies.json`.

### 3.3 Embeddings — `embedding_cache.py` + `shift_descriptor/embeddings.py`

- `EmbeddingExtractor.encode`: batches texts through the same 4-bit HF model
  used for generation (optionally LoRA-wrapped), takes `output_hidden_states`,
  and **mean-pools the last hidden state over non-padding tokens**
  (`_select_hidden_tensor` degrades gracefully for architectures that don't
  expose `last_hidden_state`/`hidden_states`, e.g. falling back to logits or a
  flattened final-layer state tensor for exotic backbones like Mamba/RWKV).
  Non-finite values are zeroed (`nan_to_num`).
- `EmbeddingCache.load_or_compute`: **on-disk cache** keyed by
  `{alias}_{split}_embeddings.npz` under `--embedding-dir`; recomputation is
  skipped entirely if the file exists. Optionally subsamples to
  `--subsample-limit` points per split for speed on huge splits.

### 3.4 Shift descriptors — `active_evaluator/descriptors.py` + `shift_descriptor/metrics.py`

For a pair of embedding clouds (e.g. `meta_train` vs `meta_val`), six scalar
features are computed (`DEFAULT_FEATURE_ORDER`):

| Feature | Formula (from `shift_descriptor/metrics.py`) | Meaning |
| --- | --- | --- |
| `frechet_distance` | `‖μ_b - μ_a‖² + Σ(√(var_b/var_a) - 1)²` | Fréchet-style mean+variance-scale drift (diagonal approx, not full FID) |
| `frechet_mean_shift` | `‖μ_b - μ_a‖` | Pure mean shift magnitude |
| `mahalanobis_distance` | `√((μ_a-μ_b)ᵀ Σ_pooled⁺ (μ_a-μ_b))`, `Σ_pooled = 0.5(Σ_a+Σ_b)` | Covariance-normalized mean shift (pseudo-inverse for stability) |
| `swd_mean`, `swd_std`, `swd_max` | mean/std/max of 1-D Wasserstein distance over `num_projections` random unit directions | Sliced Wasserstein distance — a tractable proxy for full-distribution Wasserstein distance in high dimensions |

`compute_shift_descriptor` computes `DistributionStats` (mean, regularized
covariance `cov + εI`, clipped variance) once per embedding cloud and reuses
them across metrics (`_maybe_compute_stats` cache keyed by `id(emb)`).

Each model gets **three** such descriptors, all against `meta_train` as the
reference distribution:
- **support** = `meta_train` vs `meta_val` (label: `meta_val` accuracy) — this is `S_0`, the one labeled pair every task starts with.
- **query** = `meta_train` vs `meta_test` (label: `meta_test` accuracy) — meta-training's held-out target within the seen-model pool.
- **transfer** = `meta_train` vs `dev` (label: `dev` accuracy) — the "real" external test split.

`pipeline.py::_compute_descriptor_stats` / `_apply_descriptor_norm` then
z-normalize every descriptor (support/query/transfer, across *all* tasks)
using a single shared mean/std so the MLP sees a consistent input scale.

### 3.5 The predictor — `active_evaluator/model.py`

`ActiveEvaluator` is a plain **3-layer MLP**: `Linear(input_dim→128) → ReLU →
Dropout(0.1) → Linear(128→64) → ReLU → Dropout(0.1) → Linear(64→1) →
Sigmoid`. `input_dim = 6 (descriptor) + context_dim (default 32) = 38`.
Sigmoid output because execution accuracy ∈ [0,1] and unbounded regression
heads were found to drift outside that range and inflate MAE (see the code
comment at `model.py:35`). `functional_forward` re-implements the same
forward pass against an explicit list of 6 parameter tensors (rather than
`self.parameters()`) so it can be called with *manually adapted* weights —
this is what the meta-learning inner loop needs.

### 3.6 Meta-learning — `active_evaluator/meta_learning.py`

This is a **CAVIA-style** (Context Adaptation Via Meta-Learning, Zintgraf et
al. 2019) meta-learner, not vanilla MAML: instead of adapting *all* network
weights per task in the inner loop, it adapts a small **task-specific context
vector** `z ∈ ℝ^{32}` that gets concatenated onto the 6-dim descriptor before
the forward pass (`_augment_with_context`), while the MLP weights stay shared
(meta-)parameters updated only in the outer loop. This is cheaper and more
stable than full-weight MAML when there's only a handful of adaptation
examples per task (often just 1).

- **Inner loop** (`_adapt_context`): `context_dim`-many gradient steps of
  `context ← context - inner_lr · ∇_context MSE(f(support_x, context), support_y)`,
  computed via `torch.autograd.grad` so it stays differentiable
  (first-order MAML approximation by default — `cfg.first_order=True` detaches
  after each step, `create_graph=False`).
- **Outer loop** (`meta_train`): for each meta-batch (`tasks_per_batch=4`
  random tasks), adapt context on that task's **support** pair, predict on
  its **query** descriptor, accumulate `MSE(pred, query_label)` (+ optional
  `meta_reg_beta`-weighted KL-style regularizer `0.5·mean(pred²)` pulling
  predictions/activations toward 0, and `meta_reg_lambda`-weighted L2 on the
  MLP weights), average over the batch, and step `Adam(outer_lr)` on the
  **MLP weights only** (the context vector is task-local and discarded).
  Runs `num_epochs` (default 500) such steps; every `val_interval` epochs it
  evaluates on `val_tasks` and checkpoints the best MAE (`early_stopping_patience`
  epochs without improvement stops early), then reloads the best checkpoint
  at the end.
- **Context bank** (`build_context_bank`): after meta-training, adapt a
  context vector per *training* task once and cache it
  (`TaskContextEmbedding`) — used as a warm-start `context_init` when
  evaluating so cold-start adaptation isn't needed for every model at test
  time (`evaluate`/`evaluate_transfer` look up `context_by_model` by name,
  falling back to a fresh zero-context adapt for genuinely new models).
- **Evaluation** (`_evaluate_task`): clone base params (detached from the
  graph), optionally run `adapt_context_steps` context-adaptation steps and
  `adapt_weight_steps` **weight** adaptation steps (`_param_step` — plain SGD
  on the *cloned* weights, i.e. an outer-loop-style fine-tune, not part of
  meta-training) on the support pair, then predict on the target descriptor
  (`query_descriptor` for `evaluate`, `transfer_descriptor` for
  `evaluate_transfer`). Reports `|pred − true|` as MAE per task.

### 3.7 Active selection (the thesis's new contribution) — `active_evaluator/active_selection.py`

Only runs when `--use-active-selection` is passed, and only for the **held-out
test models** (`extend_test_tasks_with_active_selection` in `pipeline.py`).
For each held-out model:

- **S₀** = that model's own single labeled support pair (support_descriptor, support_label).
- **U** (pool) = every *training* task's support pair, wrapped as
  `SelectionExample`s (`_task_to_support_example`).
- **V** (validation) = every *validation-split* task's support pair (the
  `val_tasks` carved out of the meta-training pool in `pipeline.py:695-707`,
  roughly `max(3, 2n/5)` tasks). `U` and `V` are kept disjoint by key-filtering.
- **T** (target) = the held-out model's own unlabeled **query** descriptor
  (`_task_to_target_example`) — i.e. `meta_train` vs `meta_test` shift, with
  no label attached; this stands in for "the deployment workload we haven't
  labeled yet."

`select_extension(...)` (`active_selection.py:721`) then:

1. **Budgets** the run: `compute_budget(pool_size=|U|, n_rounds, budget_fraction
   or budget_absolute)` (see §4) and splits the total across `n_rounds`.
2. **Embeds** everything through the *predictor's penultimate layer*
   (`compute_embeddings` = `ReLU(fc2(ReLU(fc1(x))))`, run with descriptors
   zero-padded to the model's full context-augmented input width via
   `_zero_context` — selection never uses the task-adapted context, only the
   raw meta-trained weights) and sets the RBF kernel bandwidth `τ` via the
   **median heuristic** (`compute_bandwidth_median`: median of squared
   pairwise distances over up to 2000 random pairs).
3. **(Optional) narrows the pool U** to target-aligned candidates
   (`--selection-pool-narrow-quantile`, off by default): keep only pool
   examples whose mean RBF similarity to T is above the given quantile.
4. **Narrows V → V_T** (`narrow_validation`, "Strategy A"): keep validation
   examples whose mean similarity to T is above `--selection-narrowing-quantile`
   (default 0.7) — i.e. drop the ~70% of validation models that don't look
   like the deployment target, so the acquisition objective is scored against
   a *target-relevant* slice, not the whole validation pool. Falls back to the
   single best-matching example if the quantile threshold empties the set.
5. **Runs `n_rounds` of greedy selection**, each round:
   - Re-adapts the predictor for `K_steps` on `S₀ ∪ (already selected)` to get
     the "current" model state `φ_round` (mirrors what the final evaluation
     will actually do).
   - Re-embeds V_T and the remaining pool under `φ_round`.
   - Computes **influence weights** `I(v) = ‖∇_φ L(h_φ; v)‖₂` for each
     `v ∈ V_T` (`compute_influence_weights` — per-example gradient norm of
     the MSE loss w.r.t. all predictor params; used to up-weight validation
     examples the predictor is currently most sensitive to).
   - Dispatches to one of three selection algorithms (§3.8) for `round_budget`
     picks, extends `S`, removes picked items from the remaining pool, and
     logs a detailed per-pick trace (§3.9).
6. **Restores** the predictor's original (pre-selection) weights before
   returning (`_restore_state_dict`) — selection never permanently mutates
   the meta-trained model; only the caller's downstream evaluation adapts on
   the *extended* support set afterward, the normal way, through
   `meta_learner.evaluate(...)`.

Back in `pipeline.py::extend_test_tasks_with_active_selection`, the returned
`selected_S` examples are concatenated onto the held-out task's
`support_descriptor`/`support_label` tensors — so the ordinary CAVIA
adaptation loop (§3.6) picks up the enlarged support set with **zero special
casing**; it just sees more (descriptor, label) rows to adapt on.

### 3.8 The three selection algorithms

**V1 — `lazy_greedy_facility`** (default, `--selection-method v1_facility`):
monotone submodular, **influence-weighted facility location**:

```
f₁(S) = Σ_{v∈V_T} I(v) · max_{s∈S∪S₀} exp(-‖φ(v)-φ(s)‖² / τ)
```

Implemented as a **lazy greedy with a priority queue** (`heapq`): each
candidate's marginal gain `Δ(s|S) = Σ_v I(v)·max(0, sim(v,s) − coverage(v))`
is pushed with a stale "last computed at iteration" tag; when popped, if it
wasn't recomputed since the top of the heap last moved, it's re-scored lazily
before accepting (classic Minoux 1978 lazy-greedy trick — algebraically
identical to plain greedy under submodularity, just far fewer marginal-gain
evaluations). Score used for ranking is **gain-per-cost** (`gain / c(key)`)
to support non-uniform costs, though the default `cost_fn` is unit cost
(cardinality). Stops a round early if the best available true gain is ≤ 0.
Provable `(1 − 1/e)` approximation to the optimum under a cardinality budget
(Nemhauser–Wolsey–Fisher 1978); knapsack-budget variants have weaker but
still-provable guarantees (Khuller–Moss–Naor 1999; Sviridenko 2004).

**V2 — `direct_greedy_validation`** (`--selection-method v2_direct`):
**non-submodular, non-monotone** empirical oracle. Plain greedy over the
*actual* post-adaptation validation loss:

```
f₂(S) = L(h_{φ_K(S₀)}; V_T) − L(h_{φ_K(S₀∪S)}; V_T)
```

where `φ_K(X)` = predictor weights after K real SGD steps
(`_adapt_K_steps`) adapting on support set `X`, and `L` is MSE on `V_T`. For
each pick, only the top `--selection-max-candidates-evaluated` (default 100)
candidates **by V1's cheap surrogate ranking** are actually K-step-adapted
and scored (**FASS-style pre-filtering**, Wei/Iyer/Bilmes 2015) — evaluating
every remaining candidate this way is too expensive to do exhaustively.
Picks the candidate with the best per-cost loss reduction; a round aborts
early if no candidate yields a positive gain (`best_gain <= 0` → break),
because — unlike V1 — nothing here guarantees marginal gains stay positive
or diminishing (see the earlier conversation on why: SGD adaptation is a
nonlinear, path-dependent map, so it can be non-monotone/superadditive).
Optional `weight_decay` (regularizes the inner K-step SGD so it doesn't
overfit a tiny V_T) and `early_stop_patience` (stop the inner adaptation once
V_T loss stops improving) are exposed as extra safety valves.

**V3 — `gradient_match_omp`** (`--selection-method v3_gradmatch`):
**GRAD-MATCH** (Killamsetty et al. 2021) via **Orthogonal Matching Pursuit**.
Purely first-order — no nested K-step adaptation at all, so it's much cheaper
than V2 while still being data-dependent (unlike V1's kernel-only geometry).
Computes per-example loss gradients w.r.t. all predictor params for the pool
and for V_T (`compute_per_example_gradients`), sets the target as the mean
V_T gradient, then greedily picks the pool example whose gradient has the
largest `|inner product|` with the current residual
`r = g_target − Σᵢ wᵢ gᵢ`, refitting weights `w` after every pick via
regularized least squares (`(GGᵀ + λI)w = G·g_target`) — this is exactly
Algorithm 2 of the GRAD-MATCH paper. Minimizes
`‖Σᵢ wᵢ gᵢ − g_target‖² + λ‖w‖²` subject to `|S| ≤ budget`.

| Algorithm | Objective structure | Guarantee | Cost per pick |
| --- | --- | --- | --- |
| V1 facility | monotone submodular | `(1-1/e)`-approx (cardinality); weaker knapsack bounds | 1 kernel eval |
| V2 direct | non-submodular, non-monotone | none — empirical upper-bound oracle | up to 1 K-step retrain × 100 candidates |
| V3 GRAD-MATCH | convex OMP relaxation | OMP recovery guarantees under RIP-like conditions, not submodular | 1 gradient + 1 small linear solve |

### 3.9 Selection diagnostics / output artifacts

When `--use-active-selection` runs with a `log_dir`, per held-out model it
writes to `outputs/<run>/selection_<sanitized_model_id>/`:

- `budget_plan.json` — `BudgetPlan` (total, n_rounds, per-round split, fraction/absolute, cost fn).
- `narrowing_diagnostics.json` — `|V|`, `|V_T|`, quantile `q`, ρ threshold/mean/min/max for both V-narrowing and (if enabled) U-narrowing.
- `selection_trajectory.json` — per round: which keys/source models were picked, full per-pick trace (method-specific fields — `marginal_gain`/`max_sim_to_VT` for V1, `loss_before/after` for V2, `inner_product_score`/`residual_norm_after` for V3), influence-weight stats, `val_loss_before/after`, per-stage timings.
- `selected_examples.json` — final `S`: `{key, source_model, true_label, cost}` per pick.
- `selection_summary.json` — top-level recap (`method`, `τ`, budget totals, `n_selected`, `final_val_loss`, `source_models_chosen`, and a **human-readable `rationale` string** explaining the objective in plain language).

These answer *what* got picked, *why* (marginal gain / loss reduction /
inner-product score, plus influence stats), and *at what cost*.

---

## 4. Budget arithmetic — `active_evaluator/budget.py`

`compute_budget(pool_size, n_rounds, budget_fraction=0.10, budget_absolute=None,
min_budget=1)`:

```
total = budget_absolute if given else round(budget_fraction * pool_size)
total = clamp(total, min_budget, pool_size)
per_round[i] = total // n_rounds, with the remainder added to the last round
```

Returned as a `BudgetPlan` dataclass (JSON round-trippable via
`.save`/`.load`), which self-validates in `__post_init__` that
`len(per_round) == n_rounds` and `sum(per_round) == total`.

---

## 5. Baseline method library — `baselines/`

`baselines/_core.py` is a **compact, dependency-free NumPy reimplementation**
of every acquisition/estimator strategy compared against ActiveEval, sharing
one call signature:

```python
select(X, pair_model, pair_sample, target_mask, budget, *, rng, **kw) -> List[int]
```

where `X` is `(P, d)` shift descriptors for every candidate (reference-model,
sample-set) *pair*, `pair_model`/`pair_sample` are integer ids per pair, and
`target_mask` flags pairs aligned with the deployment workload.

| Registry key | Function | Idea |
| --- | --- | --- |
| `random` | `select_random` | Uniform random — naive floor. |
| `kcenter` | `select_kcenter` | Greedy k-center: maximize min distance to the chosen set (diversity only, no target awareness). |
| `facility_location` | `select_facility` | Target-aware (optionally influence-weighted) greedy facility location — the NumPy twin of `lazy_greedy_facility`. |
| `matrix_completion` | `select_matrix_completion` | Leverage-score sampling from a rank-r SVD of `X` (Candès & Recht 2009). |
| `active_testing` | `select_active_testing` | Kossen et al. 2021: bagged ridge fits on a small random seed, acquire highest predictive-variance pairs. |
| `bayesian_design` | `select_bayesian_design` | Greedy D-optimal design: maximize `log det(XₛᵀXₛ + λI)` via a rank-1 inverse update. |
| `submodular_benchmark` | `select_logdet` | Log-determinant / DPP-style diversity, greedy incremental Cholesky. |
| `gradmatch` | `select_gradmatch` | GRAD-MATCH OMP — NumPy twin of `gradient_match_omp`. |
| `activeeval_s` | `select_activeeval_sample` | **Ours (S)**: target-aware, influence-weighted facility location over sample-sets, model axis left uniform. |
| `activeeval_sm` | `select_activeeval_sm` | **Ours (S+M)**: k-center over per-model centroid descriptors to keep a behaviorally diverse half of the reference models, then target-aware facility location within them. |
| `activeeval_pair` | `select_activeeval_pair` | **Ours (Pair, strongest)**: `α·budget` via target-aware facility location, remaining `(1-α)·budget` via log-determinant diversity restricted to the target-aligned pairs only. |

Label-free estimators (no acquisition, just a point estimate from unlabeled
target confidences): `estimate_atc` (Average Thresholded Confidence, Garg et
al. 2022 — calibrate a confidence threshold on source so source accuracy
matches, then measure the fraction of target points above it) and
`estimate_doc` (Difference of Confidences, Guillory et al. 2021 — subtract
the source/target mean-confidence gap from the source accuracy). Both
registered in `ESTIMATOR_REGISTRY` (`baselines/__init__.py`).

`AutoEval`, `AETTA`, `SSME` are **not implemented** — their subpackages
(`baselines/autoeval/`, `baselines/aetta/`, `baselines/ssme/`) contain only a
`README.md` pointing at the official upstream repos; they appear in the
README's method table but not in `ACQUISITION_REGISTRY`/`ESTIMATOR_REGISTRY`.

The production pipeline (`active_evaluator/active_selection.py`) implements
the same facility-location / GRAD-MATCH / knapsack-budget selection math on
*real* shift descriptors from cached LLM embeddings; `baselines/_core.py`
exists so every method (including ones the production pipeline doesn't wire
up as a `--selection-method` choice, e.g. Bayesian design) can be compared on
one synthetic benchmark.

---

## 6. Synthetic acquisition benchmark — `experiments/run_acquisition_benchmark.py`

A **self-contained, CPU-only, seconds-long** reproduction used to produce
`README.md`'s main results table and the RQ3/RQ5 tables, without downloading
any model.

**`make_problem(seed, n_train_models=60, n_unseen_models=8, n_samplesets=40,
d_lat=6, base_noise=0.08)`** synthesizes a controlled meta-evaluation matrix:

1. Latent factors: `U` (train-model factors, `60×6`), `Uo` (unseen-model
   factors, `8×6`, **shifted by +0.15** to simulate a real train/unseen
   family gap), `V` (sample-set factors, `40×6`).
2. A random **target direction** in factor space picks the top quarter of
   sample-sets as `target_sets` — the simulated deployment workload cluster.
3. A fixed random 2-layer tanh/sigmoid network `true_acc(u, v)` is the
   ground-truth accuracy function — shared by every method, unknown to all of
   them, standing in for "the true relationship between a model, a workload,
   and how well that model does on it."
4. `descriptor(u, v) = [u, v, u⊙v] + noise` (18-dim) stands in for the shift
   descriptor a real pipeline would compute — factors plus their elementwise
   interaction plus small observation noise.
5. **Every** `(train model, sample-set)` pair is built (`60×40 = 2400`
   candidate "evaluation actions"), each with a noisy accuracy label
   (`a_noisy`) — critically, **off-target pairs get ~4.4× more label noise**
   (`base_noise * 2.0` vs `* 0.45`) and make up 3/4 of the pool, modeling the
   real-world fact that labeling everything dilutes signal with noisy,
   off-target evaluations, while a budget spent only on target-aligned pairs
   stays clean.
6. The **eval set** is `unseen models × target sample-sets only`, with clean
   (noise-free) labels — this is the "operational" MAE every method is
   scored on.
7. Per-unseen-model synthetic confidence distributions (`conf_src`,
   `conf_tgt`) are generated for ATC/DoC, and a cheap "influence" proxy
   (`|true - seed_linear_fit|`) stands in for the real gradient-norm
   influence weights used by `select_facility`/ActiveEval variants.

**`train_eval(prob, selected_indices, seed, epochs=250)`**: instantiates the
*real* `active_evaluator.model.ActiveEvaluator` MLP, trains it with plain
Adam (no meta-learning/CAVIA here — this benchmark tests the *selection*, not
the meta-learner) on only the acquired `(X[selected], a_noisy[selected])`
pairs, and reports `mean(|pred - a_eval|) * 100` (MAE in percentage points)
on the clean unseen-model eval set.

**Three CLI modes** (`--mode {main,sweep,ablation}`):
- `main` (default): every method in `METHODS` gets the same `--budget-frac`
  (default 0.15) and 5 seeds → `outputs/acquisition_benchmark.json`, the
  table in README §"Quick start".
- `sweep`: `run_sweep` — ActiveEval-Pair vs Facility-location vs Random across
  `[5%, 10%, 15%, 20%, 30%, 50%]` budgets, plus the flat full-budget
  MetaEvaluator reference → `outputs/acquisition_sweep.json` (README's RQ3
  table — "budget efficiency").
- `ablation`: `run_ablation` — ActiveEval-Pair (full) vs "− target-aware
  narrowing" (same method but `target_mask` forced to cover the whole pool,
  i.e. `cover_all=True`) vs "− all structure" (plain `random`) →
  `outputs/acquisition_ablation.json` (README's RQ5 table).

---

## 7. Auxiliary shift-descriptor pipeline — `shift_descriptor/`

A standalone CLI (`python -m shift_descriptor.pipeline`) that runs *just*
stages 3–4 of §3 (embeddings → descriptors) between two arbitrary datasets,
without any meta-learning or active selection — useful for exploring shift
diagnostics on their own. Key files:

- `config.py` — `ModelSpec` dataclass + `default_model_specs()` (3 lightweight
  default HF models: TinyLlama-1.1B-Chat, Zephyr-3B-beta, Phi-3-mini-4k).
- `datasets.py` — `load_prompt_texts` / `load_text_field` (shared with
  `active_evaluator/data_utils.py`).
- `prompts.py` — the default Text2SQL prompt template + `default_template_path()`.
- `embeddings.py` — `EmbeddingExtractor` (same class used by `embedding_cache.py`).
- `metrics.py` — Fréchet/Mahalanobis/SWD math (same module used by `descriptors.py`),
  plus extras not used in the main pipeline: `tail_drift`, `compute_pairwise_similarities`,
  `pca_project`, `linear_cka`.
- `diagnostics.py` — higher-level similarity/coverage diagnostics between splits.
- `visualize.py` — PCA scatter plots of embedding clouds.
- `pipeline.py` — the CLI entry point tying the above together; writes embedding
  caches, descriptor matrices, PCA plots, and similarity diagnostics to `--output-dir`.

CLI flags mirror the main pipeline: `--model-ids`, `--lora-r`,
`--metric-max-points`, `--prompt-template`, `--context-fields`,
`--use-plain-text`, `--text-field`, `--device`. A `alias=model_id:remote`
suffix enables `trust_remote_code=True` for architectures needing it (Mamba,
RWKV, etc.).

---

## 8. Repository layout

```
active_evaluator/         production pipeline
  accuracy.py               ModelAccuracySource — loads/validates model_accuracies.json
  active_selection.py       V1/V2/V3 selection algorithms, narrowing, influence weights, budget wiring
  budget.py                 BudgetPlan dataclass + compute_budget
  data_utils.py             dataset loading, prompting, meta_train/val/test split
  descriptors.py            6-feature shift descriptor (Fréchet/Mahalanobis/SWD)
  embedding_cache.py        on-disk embedding cache keyed by model×split
  evaluation.py             SQL exact-match + SQLite execution-accuracy scoring
  generation.py             4-bit quantized + optional LoRA SQL generation
  meta_learning.py          CAVIA-style meta-learner (ActiveEvaluatorLearner)
  model.py                  ActiveEvaluator MLP (3-layer, sigmoid head)
  pipeline.py                CLI + end-to-end orchestration (main())

shift_descriptor/         auxiliary, standalone shift-diagnostics pipeline
  config.py, datasets.py, prompts.py, embeddings.py, metrics.py,
  diagnostics.py, visualize.py, pipeline.py

baselines/                 NumPy reference implementations of every compared method
  _core.py                    all `select_*`/`estimate_*` functions + registries
  random/, kcenter/, facility_location/, matrix_completion/, active_testing/,
  bayesian_design/, submodular_benchmark/, gradmatch/, atc/, doc/, metaevaluator/,
  aetta/, autoeval/, ssme/     thin per-method subpackages (last three are stub READMEs only)

experiments/
  run_acquisition_benchmark.py   synthetic CPU benchmark reproducing the paper's main/RQ3/RQ5 tables

data/                      Spider Text2SQL train/dev JSON (sft_spider_{train,dev}_text2sql.json)
prompts/                   Text2SQL prompt templates
resources/                 figures (e.g. training_pipeline.png)
scripts/                   inspect_active_evaluator.py, rename_model_outputs.py, compare_runs.py
test/                      24 unit tests: test_budget.py, test_active_selection.py, test_active_learning.py
logs/                      run scripts (launch_v1/v2/v3.sh, run_all.sh, run_extended.sh) + captured logs/comparison notes
outputs/                   all run artifacts: predictions, embeddings, metrics, selection diagnostics, benchmark JSONs
```

---

## 9. Running it

**Production pipeline, no active selection (base behavior, unchanged from
the pre-active-selection version):**

```bash
python -m active_evaluator.pipeline \
  --train-path data/sft_spider_train_text2sql.json \
  --dev-path data/sft_spider_dev_text2sql.json \
  --output-dir outputs/run_baseline \
  --max-train-samples 1500 --max-dev-samples 300 \
  --model-ids ... --test-model-ids ...
```

**With active selection (the new contribution):**

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
  --model-ids ... --test-model-ids ...
```

Swap `--selection-method v2_direct` or `v3_gradmatch` to compare selection
strategies; see `active_evaluator/pipeline.py:parse_args` for every
`--selection-*` knob (K-steps, weight decay, early-stop patience, pool
narrowing, FASS candidate cap, seed).

**Synthetic benchmark (no GPU, no downloads, seconds):**

```bash
python -m experiments.run_acquisition_benchmark --seeds 5 --budget-frac 0.15
python -m experiments.run_acquisition_benchmark --mode sweep --seeds 5
python -m experiments.run_acquisition_benchmark --mode ablation --seeds 5
```

**Auxiliary shift-descriptor diagnostics only:**

```bash
python -m shift_descriptor.pipeline \
  --train-path data/<train>.json --test-path data/<test>.json --output-dir outputs
```

**Tests** (24 unit tests, no real-data MAE — correctness invariants only:
budget arithmetic, lazy-greedy ≡ plain-greedy equivalence, submodularity
[non-increasing marginal gains], `S ∩ V = ∅` disjointness, FASS pre-filter
cap, positive-gain abort rule, embedding shapes, bandwidth scaling, narrowing
set sizes, non-negative influence weights):

```bash
python -m pytest test/ -v
```

---

## 10. Key design decisions worth remembering

- **Why CAVIA instead of full MAML**: with often a *single* labeled support
  pair per model, adapting all MLP weights per task would badly overfit;
  adapting a small 32-dim context vector while keeping the MLP weights
  shared is far more sample-efficient (`meta_learning.py`).
- **Why active selection only touches held-out test models**: the seen
  training/validation models already have full labeled support in the
  meta-training pool; active selection exists specifically to answer "given
  a brand-new unseen model with only one labeled point, what *else* should we
  label to help the predictor adapt to it" — hence it only runs inside
  `extend_test_tasks_with_active_selection`, never during meta-training.
- **Why V_T narrowing exists at all**: without it, the facility-location
  objective would try to cover *all* validation models uniformly, including
  ones with nothing in common with the actual deployment target — narrowing
  restricts the scoring set to the ~30% of validation models that most
  resemble the target's descriptor, so picks are chosen for target-relevance,
  not global coverage.
- **Why V2/V3 exist alongside V1**: V1 has a provable approximation
  guarantee but only optimizes a *geometric proxy* (kernel coverage) for
  validation-loss reduction; V2 optimizes the *real* quantity directly (at
  large compute cost, no guarantee); V3 is a cheap first-order compromise
  (data-dependent like V2, but no nested retraining, no guarantee like V1).
  They're three different points on the cost/fidelity/guarantee trade-off
  triangle, all wired through the same `select_extension` entry point so
  they're trivially swappable via `--selection-method`.
- **Why the synthetic benchmark exists separately from the real pipeline**:
  the real pipeline needs GPU + model downloads + hours; the synthetic
  benchmark reuses the *exact* selection math (`baselines/_core.py` mirrors
  `active_selection.py` function-for-function) and the *exact* predictor
  class (`active_evaluator.model.ActiveEvaluator`), so its method ordering is
  a trustworthy stand-in for the full pipeline's ordering, fast enough to
  regenerate on every change.

# Submodularity audit — what's proven, what's implemented, what's missing

This note answers two questions precisely: (1) does this repo actually use
submodular maximization, and can we prove the objective is monotone and
submodular; (2) does the code follow the paper's three-term joint objective
`F = αF_sample + βF_model + γF_info` (`method.tex`, line 110), especially the
log-determinant (`F_info`) term. Short answer: **yes to the sample-coverage
term (proven below, with a working `(1-1/e)` guarantee), yes to the log-det
term in isolation (proven below), but the three-term joint objective as one
combined greedy does not exist anywhere in the code** — what's implemented
are three separate, only loosely related mechanisms.

---

## Part 1 — V1 facility location: proof of monotone submodularity

`V1` (`active_evaluator/active_selection.py::lazy_greedy_facility`, the
default `--selection-method v1_facility`, and the coverage stage inside
ActiveEval-S / S+M / Pair) is genuine submodular maximization, and it is
provably both monotone and submodular.

### Setup (matching the code)

Ground set of candidates: the pool `U` (`pool_features`). For each round,
define for a chosen set `A ⊆ U`:

```
f(A) = Σ_{v∈V_T} I(v) · max_{s∈A∪S₀} k(v,s),   k(v,s) = exp(-‖φ(v)-φ(s)‖² / τ)
```

- `S₀` is a **fixed baseline** set (already-selected items ∪ the model's own
  support pair) — it just shifts the starting "coverage" floor and doesn't
  affect the argument below.
- `I(v) ≥ 0` — asserted non-negative in code
  (`compute_influence_weights`, `active_selection.py:395`:
  `assert (out >= 0).all()`).
- `k(v,s) ∈ (0, 1]` — a bounded similarity kernel (`_gaussian_similarity`).

This is exactly what `_coverage_from` + `_marginal_gain` compute:
`coverage(v) = max_{s∈A} k(v,s)`, and the marginal gain of adding candidate
`c` is:

```python
delta = (sim_to_candidate - coverage).clamp_min(0.0)      # active_selection.py:417
gain  = (influence * delta).sum()
```

i.e. `Δ(c ∣ A) = f(A∪{c}) − f(A) = Σ_v I(v)·max(0, k(v,c) − coverage(v∣A))`.
That closed form is the crux of both proofs below.

### Claim 1 — `f` is monotone nondecreasing

For `A ⊆ B ⊆ U`: since `A∪S₀ ⊆ B∪S₀`, taking a max over a superset can only
increase (or keep) the max:

```
max_{s∈A∪S₀} k(v,s) ≤ max_{s∈B∪S₀} k(v,s)   for every v
```

Multiply by `I(v) ≥ 0` and sum over `v∈V_T` — nonnegative weights preserve
the inequality — giving `f(A) ≤ f(B)`. ∎

### Claim 2 — `f` is submodular

Need: for `A ⊆ B ⊆ U` and any `c ∉ B`,
`f(A∪{c}) − f(A) ≥ f(B∪{c}) − f(B)` (diminishing returns).

Fix one `v` and let `g(A) = max_{s∈A∪S₀} k(v,s)`. Adding a single element
`c` updates the max by:

```
g(A∪{c}) − g(A) = max(g(A), k(v,c)) − g(A) = [k(v,c) − g(A)]₊
```

(exactly `clamp_min(sim_to_candidate − coverage, 0)` in the code). By
Claim 1, `g(A) ≤ g(B)` since `A ⊆ B`. Since `x ↦ [x]₊` is nondecreasing in
`x`, and `k(v,c) − g(A) ≥ k(v,c) − g(B)`:

```
[k(v,c)−g(A)]₊ ≥ [k(v,c)−g(B)]₊   ⟹   g(A∪{c})−g(A) ≥ g(B∪{c})−g(B)
```

That's diminishing returns for a single `v`. Sum over `v∈V_T` with weights
`I(v) ≥ 0` — a nonnegative-weighted sum of functions that each individually
satisfy diminishing returns still satisfies diminishing returns (the
inequality is preserved termwise, then summed):

```
f(A∪{c})−f(A) = Σ_v I(v)[g_v(A∪{c})−g_v(A)] ≥ Σ_v I(v)[g_v(B∪{c})−g_v(B)] = f(B∪{c})−f(B)
```

∎ `f` is submodular.

(Equivalently: each `g_v` is a "coverage" function —
`g_v(A) = ∫₀¹ 𝟙[max_{s∈A} k(v,s) ≥ t] dt`, an integral of monotone
set-coverage indicators, the textbook way facility-location functions are
shown submodular — but the direct max-of-similarities argument above maps
1:1 onto the actual code, which is why it's used here.)

### Why this matters operationally

Because `f` is monotone submodular:

- **Lazy greedy ≡ plain greedy exactly** (Minoux 1978) — a stale marginal
  gain computed earlier is always an upper bound on the true current gain
  (submodularity ⇒ gains only shrink as the set grows), so the heap in
  `lazy_greedy_facility` can skip most recomputation and still return the
  identical selection plain greedy would. This is unit-tested
  (`test/test_active_selection.py`: "lazy-greedy ≡ plain-greedy on synthetic
  20×5 pool" and "submodularity (marginal gains non-increasing)").
- The `(1 − 1/e) ≈ 0.632` approximation ratio (Nemhauser, Wolsey & Fisher
  1978) applies **exactly because** both properties hold — invalid without
  both monotonicity and submodularity, which is why it's the only one of the
  three selection methods (`V1`/`V2`/`V3`) the README's theory table credits
  with a proof.

### What in this repo is *not* submodular maximization

- **V2 (`direct_greedy_validation`)** — optimizes real post-adaptation
  validation loss via K-step SGD. SGD's trajectory is nonlinear and
  path-dependent, so nothing forces diminishing returns; the code's own
  positive-gain abort rule (`best_gain <= 0: break`) exists precisely
  because monotonicity isn't guaranteed. Non-submodular, non-monotone,
  empirical upper-bound oracle only.
- **V3 (`gradient_match_omp`, GRAD-MATCH)** — orthogonal matching pursuit on
  gradients. Its own convergence theory (sparse-recovery / OMP guarantees),
  entirely separate from submodularity — never claimed to be submodular.
- **`select_kcenter`** (baselines) — greedy farthest-point traversal solves
  a different combinatorial objective (min the covering radius); its
  guarantee is Gonzalez (1985)'s 2-approximation for k-center, not a
  submodular-coverage bound.

---

## Part 2 — does the code implement the paper's joint `F = αF_sample + βF_model + γF_info`?

**No — not as one combined greedy.** What exists instead:

| Term in `method.tex` | Code that claims to implement it | Faithful to the formula? | Target-aware? | Jointly weighted with the others? |
| --- | --- | --- | --- | --- |
| `F_sample` (facility location) | `active_evaluator/active_selection.py::lazy_greedy_facility` (production) and `baselines/_core.py::select_facility` (synthetic) | **Yes** — literally `Σ_v I(v)·max_s k(v,s)`, proven in Part 1 | Yes | — (it's the whole objective on its own in V1 / ActiveEval-S) |
| `F_model` (model coverage `Σ_ℓ max_{(i,j)∈Ω} k_M(ψ(m_i),ψ(m_ℓ))`) | `select_activeeval_sm`'s stage 1 (`select_kcenter` over model centroids) | **No** — a different function entirely | No — target-blind by construction | No — hard sequential pre-filter, not a summand |
| `F_info` (log-det diversity) | `baselines/_core.py::select_logdet` | **Yes**, verified in Part 3 | Only when called from inside `select_activeeval_pair`; **no** when run standalone as `"Submod. benchmark"` | No — separate phase, separate budget slice |

Concretely: **nowhere in the repo is there a function that computes
`α·Δ_sample(e∣Ω) + β·Δ_model(e∣Ω) + γ·Δ_info(e∣Ω)` for a candidate `e` and
picks the argmax of that combined score at each greedy step.**

- `active_selection.py` (production V1): `F_sample` alone. No `β`, no `γ`.
- `select_activeeval_sample` (**S**): `F_sample` alone, over the whole pool.
- `select_activeeval_sm` (**S+M**): a **hard, target-blind k-center cutoff**
  on the model axis first (permanently discards half the models), *then*
  `F_sample` on the survivors. This is not `βF_model` added to the
  objective — it's a cardinality pre-filter using a *different* algorithm
  (Gonzalez farthest-first, 2-approx on covering radius, not facility
  location) that runs before `F_sample` ever sees the discarded models.
  That's exactly why it can discard target-relevant models before the
  target-aware term gets a vote, and is the mechanism behind its weaker MAE
  in the results tables (README: "ActiveEval-S+M's model-axis narrowing
  hurts in this regime").
- `select_activeeval_pair` (**Pair**): spends `α_budget = round(0.8·budget)`
  on `F_sample` first, *then* spends the leftover `0.2·budget` on `F_info`
  (`select_logdet`), restricted post hoc to whatever target-aligned
  candidates remain. Two sequential single-term optimizations, not one
  joint score. There's also a subtle mismatch worth flagging: phase 1
  estimates its RBF bandwidth `τ` from the whole pool (`select_facility`'s
  own `_median_bandwidth` call), phase 2 re-estimates a *different* `τ` from
  just the leftover target-aligned candidates (`select_logdet`'s own
  `_median_bandwidth` call) — so the two phases aren't even scoring in a
  shared metric, another sign this is two separate objectives glued
  end-to-end, not one.

---

## Part 3 — the log-det (`F_info`) term itself: correctness check

`select_logdet` (`baselines/_core.py`) claims to greedily maximize
`log det(I + σ⁻²K_S)`. Walking the update:

```python
K = _rbf(X, X, tau) / (sigma ** 2)
cur_diag = 1.0 + diag                     # cur_diag[e] = 1 + σ⁻²k(e,e), the prior "gain" before conditioning
...
gains = cur_diag.copy()                    # picks argmax of cur_diag, not log(cur_diag)
pick = argmax(gains)
...
u = np.linalg.solve(L, k)                  # L u = k  →  u = L⁻¹ k
d = sqrt(cur_diag[pick])
newcol = (K[:, pick] - cov @ u) / d
L = block([[L, 0], [u, d]])                # extends the Cholesky factor of (I + σ⁻²K_S)
cov = concat([cov, newcol])
cur_diag = clip(cur_diag - newcol**2, 1e-9, None)   # rank-1 downdate of remaining conditional variances
```

This is the standard **incremental Cholesky** greedy for log-det (the same
recursion used in Krause–Singh–Guestrin-style near-optimal sensor
placement). It's correct:

1. **`cur_diag[e]` after conditioning on `S` really is
   `1 + σ⁻²·Var(e ∣ S)`** — `u = L⁻¹k` and `cur_diag[e] − newcol[e]²` are
   exactly the Schur-complement update
   `M[e,e] − k_S(e)ᵀ M_S⁻¹ k_S(e)` for `M = I + σ⁻²K`, i.e. the
   conditional-variance recursion the paper's proof invokes.
2. **Ranking by `cur_diag` instead of `log(cur_diag)` doesn't change which
   element gets picked** — `log` is strictly increasing on `(0, ∞)` and
   `cur_diag > 0` is maintained throughout (clipped at `1e-9`), so
   `argmax(cur_diag) = argmax(log(cur_diag))`. Skipping the `log()` call is
   a valid micro-optimization, not a bug — the true marginal log-det gain is
   `Δ(e∣S) = log(cur_diag[pick])`, just never explicitly materialized since
   it isn't needed for selection.
3. **Monotonicity holds**: `Δ(e∣S) = log(1 + σ⁻²Var(e∣S)) ≥ 0` always, since
   `Var ≥ 0` ⇒ `1+σ⁻²Var ≥ 1` ⇒ `log(·) ≥ 0`. So `F_info` is genuinely
   monotone nondecreasing — matches the paper's claim.
4. **Submodularity holds**: for a PSD kernel `K` (RBF is PSD by Mercer's
   theorem, and `σ⁻²K` stays PSD), `Var(e ∣ S)` is nonincreasing as `S`
   grows — conditioning on more variables never increases posterior
   variance for a jointly-Gaussian-covariance-structured model. Hence
   `Δ(e∣S) = log(1+σ⁻²Var(e∣S))` is nonincreasing in `S`, which is exactly
   diminishing returns. This is the standard result behind D-optimal design
   / max-entropy sensor placement (Krause & Guestrin), and holds here
   because `K` is RBF-PSD.

**So in isolation, `select_logdet` is a mathematically correct
monotone-submodular greedy for `F_info`, exactly as the paper describes.**
The gap isn't in that term's math — it's that it's never wired into a
combined `α/β/γ` objective anywhere in the code; it only ever runs alone (as
the untargeted `"Submod. benchmark"` baseline, where `target_mask` is
accepted as a parameter but never referenced in the function body) or as a
disjoint second phase inside Pair.

---

## What this means for the thesis

The `(1-1/e)` guarantee for the combined `F = αF_sample + βF_model + γF_info`
is currently a **paper-level claim with no matching implementation**. You
have proofs and correct code for two of the three individual terms
(`F_sample`, `F_info`), but:

- `F_model` as literally specified (`Σ_ℓ max k_M(...)`) isn't implemented at
  all — what runs instead is a different, non-facility-location
  model-diversity heuristic (`select_kcenter` over model centroids).
- No function anywhere computes the actual weighted sum
  `αΔ_sample + βΔ_model + γΔ_info` per candidate and greedily maximizes
  *that*, so the "nonnegative weighted sum of monotone submodular functions
  is monotone submodular" closure argument, while true in the abstract, isn't
  something the repo currently demonstrates end-to-end.

**To get code-paper parity**, the missing piece is a single new selection
function that:

1. Implements `F_model` as the actual facility-location coverage-of-`M`
   term (`Σ_ℓ max_{(i,j)∈Ω} k_M(ψ(m_i),ψ(m_ℓ))`), not a k-center pre-filter.
2. Computes all three `Δ`'s for each candidate under a **shared**
   kernel/bandwidth (fixing the `τ` mismatch noted in Part 2).
3. Combines them with `α, β, γ ≥ 0` (optionally `/ c_e` for the
   cost-benefit knapsack variant from `method.tex`), and greedily picks the
   argmax of that combined score.

That's the one function code and proof would both point at for the joint
`(1-1/e)` guarantee.

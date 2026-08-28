# Submodularity audit — what's proven, what's implemented, what's missing

This note answers two questions precisely: (1) does this repo actually use
submodular maximization, and can we prove the objective is monotone and
submodular; (2) does the code follow the paper's three-term joint objective
$F = \alpha F_{\text{sample}} + \beta F_{\text{model}} + \gamma F_{\text{info}}$
(`method.tex`, line 110), especially the log-determinant ($F_{\text{info}}$)
term. Short answer: **yes to the sample-coverage term (proven below, with a
working $(1-1/e)$ guarantee), yes to the log-det term in isolation (proven
below), but the three-term joint objective as one combined greedy does not
exist anywhere in the code** — what's implemented are three separate, only
loosely related mechanisms.

See [`docs/kmeans-submodular-warmstart.md`](kmeans-submodular-warmstart.md)
for the extension of Part 1's facility-location proof to a k-means-quantized
candidate pool (when the sample-set/workload universe is too expensive to
label in full) — it proves a quantization-stability bound in terms of the
number of clusters $M$, and reports a real, honestly-mixed empirical result
from testing it.

---

## Part 1 — V1 facility location: proof of monotone submodularity

`V1` (`active_evaluator/active_selection.py::lazy_greedy_facility`, the
default `--selection-method v1_facility`, and the coverage stage inside
ActiveEval-S / S+M / Pair) is genuine submodular maximization, and it is
provably both monotone and submodular.

### Setup (matching the code)

Ground set of candidates: the pool $U$ (`pool_features`). For each round,
define for a chosen set $A \subseteq U$:

$$
f(A) \;=\; \sum_{v \in V_T} I(v) \cdot \max_{s \,\in\, A \,\cup\, S_0} k(v,s),
\qquad
k(v,s) = \exp\!\left(-\frac{\lVert \phi(v)-\phi(s) \rVert^2}{\tau}\right)
$$

- $S_0$ is a **fixed baseline** set (already-selected items ∪ the model's
  own support pair) — it just shifts the starting "coverage" floor and
  doesn't affect the argument below.
- $I(v) \ge 0$ — asserted non-negative in code
  (`compute_influence_weights`, `active_selection.py:395`:
  `assert (out >= 0).all()`).
- $k(v,s) \in (0, 1]$ — a bounded similarity kernel (`_gaussian_similarity`).

This is exactly what `_coverage_from` + `_marginal_gain` compute:
$\mathrm{coverage}(v \mid A) = \max_{s \in A} k(v,s)$, and the marginal gain
of adding candidate $c$ is:

```python
delta = (sim_to_candidate - coverage).clamp_min(0.0)      # active_selection.py:417
gain  = (influence * delta).sum()
```

i.e.

$$
\Delta(c \mid A) \;=\; f(A\cup\{c\}) - f(A)
\;=\; \sum_{v} I(v)\cdot \max\!\big(0,\; k(v,c) - \mathrm{coverage}(v\mid A)\big)
$$

That closed form is the crux of both proofs below.

### Claim 1 — $f$ is monotone nondecreasing

For $A \subseteq B \subseteq U$: since $A\cup S_0 \subseteq B\cup S_0$,
taking a max over a superset can only increase (or keep) the max:

$$
\max_{s\,\in\,A\cup S_0} k(v,s) \;\le\; \max_{s\,\in\,B\cup S_0} k(v,s)
\qquad \text{for every } v
$$

Multiply by $I(v) \ge 0$ and sum over $v \in V_T$ — nonnegative weights
preserve the inequality — giving

$$
f(A) \le f(B). \qquad \blacksquare
$$

### Claim 2 — $f$ is submodular

Need: for $A \subseteq B \subseteq U$ and any $c \notin B$,

$$
f(A\cup\{c\}) - f(A) \;\ge\; f(B\cup\{c\}) - f(B) \qquad \text{(diminishing returns)}
$$

Fix one $v$ and let $g(A) = \max_{s\in A\cup S_0} k(v,s)$. Adding a single
element $c$ updates the max by:

$$
g(A\cup\{c\}) - g(A) \;=\; \max\big(g(A),\, k(v,c)\big) - g(A) \;=\; \big[\,k(v,c) - g(A)\,\big]_+
$$

(exactly `clamp_min(sim_to_candidate − coverage, 0)` in the code, where
$[x]_+ := \max(0,x)$). By Claim 1, $g(A) \le g(B)$ since $A \subseteq B$.
Since $x \mapsto [x]_+$ is nondecreasing in $x$, and
$k(v,c) - g(A) \ge k(v,c) - g(B)$:

$$
\big[k(v,c)-g(A)\big]_+ \;\ge\; \big[k(v,c)-g(B)\big]_+
\quad\Longrightarrow\quad
g(A\cup\{c\})-g(A) \;\ge\; g(B\cup\{c\})-g(B)
$$

That's diminishing returns for a single $v$. Sum over $v \in V_T$ with
weights $I(v) \ge 0$ — a nonnegative-weighted sum of functions that each
individually satisfy diminishing returns still satisfies diminishing
returns (the inequality is preserved termwise, then summed):

$$
f(A\cup\{c\})-f(A) = \sum_v I(v)\big[g_v(A\cup\{c\})-g_v(A)\big]
\;\ge\;
\sum_v I(v)\big[g_v(B\cup\{c\})-g_v(B)\big] = f(B\cup\{c\})-f(B)
$$

$\blacksquare$ $f$ is submodular.

Equivalently: each $g_v$ is a "coverage" function,

$$
g_v(A) = \int_0^1 \mathbb{1}\!\Big[\max_{s\in A} k(v,s) \ge t\Big]\,dt,
$$

an integral of monotone set-coverage indicators — the textbook way
facility-location functions are shown submodular — but the direct
max-of-similarities argument above maps 1:1 onto the actual code, which is
why it's used here.

### Why this matters operationally

Because $f$ is monotone submodular:

- **Lazy greedy ≡ plain greedy exactly** (Minoux 1978) — a stale marginal
  gain computed earlier is always an upper bound on the true current gain
  (submodularity ⇒ gains only shrink as the set grows), so the heap in
  `lazy_greedy_facility` can skip most recomputation and still return the
  identical selection plain greedy would. This is unit-tested
  (`test/test_active_selection.py`: "lazy-greedy ≡ plain-greedy on synthetic
  20×5 pool" and "submodularity (marginal gains non-increasing)").
- The approximation ratio

$$
f(A_{\text{greedy}}) \;\ge\; \left(1-\frac{1}{e}\right) f(A^\star)
$$

  (Nemhauser, Wolsey & Fisher 1978) applies **exactly because** both
  properties hold — invalid without both monotonicity and submodularity,
  which is why it's the only one of the three selection methods
  (V1/V2/V3) the README's theory table credits with a proof.

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

## Part 2 — does the code implement the paper's joint objective?

$$
F(\Omega) \;=\; \alpha\, F_{\text{sample}}(\Omega) \;+\; \beta\, F_{\text{model}}(\Omega) \;+\; \gamma\, F_{\text{info}}(\Omega),
\qquad \Omega \subseteq \mathcal M \times \mathcal X
$$

**No — not as one combined greedy.** What exists instead:

| Term in `method.tex` | Formula | Code that claims to implement it | Faithful? | Target-aware? | Jointly weighted with the others? |
| --- | --- | --- | --- | --- | --- |
| $F_{\text{sample}}$ | $\displaystyle\sum_{q\in Q(\mathcal T)} \max_{(m_i,X_j)\in\Omega} k_X(q,\phi(X_j))$ | `lazy_greedy_facility` (production), `select_facility` (synthetic) | **Yes** — proven in Part 1 | Yes | — (whole objective on its own in V1 / ActiveEval-S) |
| $F_{\text{model}}$ | $\displaystyle\sum_{m_\ell\in\mathcal M} \max_{(m_i,X_j)\in\Omega} k_M(\psi(m_i),\psi(m_\ell))$ | `select_activeeval_sm`'s stage 1 (`select_kcenter` over model centroids) | **No** — a different function entirely | No — target-blind by construction | No — hard sequential pre-filter, not a summand |
| $F_{\text{info}}$ | $\log\det(I+\sigma^{-2}K_\Omega)$ | `select_logdet` | **Yes**, verified in Part 3 | Only inside `select_activeeval_pair`; **no** when run standalone as `"Submod. benchmark"` | No — separate phase, separate budget slice |

Concretely: **nowhere in the repo is there a function that computes**

$$
\alpha\cdot\Delta_{\text{sample}}(e\mid\Omega) \;+\; \beta\cdot\Delta_{\text{model}}(e\mid\Omega) \;+\; \gamma\cdot\Delta_{\text{info}}(e\mid\Omega)
$$

**for a candidate $e$ and picks the argmax of that combined score at each
greedy step.**

- `active_selection.py` (production V1): $F_{\text{sample}}$ alone. No $\beta$, no $\gamma$.
- `select_activeeval_sample` (**S**): $F_{\text{sample}}$ alone, over the whole pool.
- `select_activeeval_sm` (**S+M**): a **hard, target-blind k-center cutoff**
  on the model axis first (permanently discards half the models), *then*
  $F_{\text{sample}}$ on the survivors. This is not $\beta F_{\text{model}}$
  added to the objective — it's a cardinality pre-filter using a
  *different* algorithm (Gonzalez farthest-first, 2-approx on covering
  radius, not facility location) that runs before $F_{\text{sample}}$ ever
  sees the discarded models. That's exactly why it can discard
  target-relevant models before the target-aware term gets a vote, and is
  the mechanism behind its weaker MAE in the results tables (README:
  "ActiveEval-S+M's model-axis narrowing hurts in this regime").
- `select_activeeval_pair` (**Pair**): spends
  $\alpha_{\text{budget}} = \mathrm{round}(0.8\cdot\text{budget})$ on
  $F_{\text{sample}}$ first, *then* spends the leftover
  $0.2\cdot\text{budget}$ on $F_{\text{info}}$ (`select_logdet`), restricted
  post hoc to whatever target-aligned candidates remain. Two sequential
  single-term optimizations, not one joint score. There's also a subtle
  mismatch worth flagging: phase 1 estimates its RBF bandwidth $\tau$ from
  the whole pool (`select_facility`'s own `_median_bandwidth` call), phase 2
  re-estimates a *different* $\tau$ from just the leftover target-aligned
  candidates (`select_logdet`'s own `_median_bandwidth` call) — so the two
  phases aren't even scoring in a shared metric, another sign this is two
  separate objectives glued end-to-end, not one.

---

## Part 3 — the log-det ($F_{\text{info}}$) term itself: correctness check

`select_logdet` (`baselines/_core.py`) claims to greedily maximize
$\log\det(I + \sigma^{-2}K_S)$. Walking the update:

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

1. **`cur_diag[e]` after conditioning on $S$ really is**
   $1 + \sigma^{-2}\cdot\mathrm{Var}(e \mid S)$ — `u = L⁻¹k` and
   `cur_diag[e] − newcol[e]²` are exactly the Schur-complement update

   $$
   M[e,e] - k_S(e)^\top M_S^{-1} k_S(e), \qquad M = I + \sigma^{-2}K,
   $$

   i.e. the conditional-variance recursion the paper's proof invokes.
2. **Ranking by `cur_diag` instead of $\log(\text{cur\_diag})$ doesn't
   change which element gets picked** — $\log$ is strictly increasing on
   $(0, \infty)$ and `cur_diag` $> 0$ is maintained throughout (clipped at
   $10^{-9}$), so
   $\arg\max(\text{cur\_diag}) = \arg\max(\log(\text{cur\_diag}))$. Skipping
   the `log()` call is a valid micro-optimization, not a bug — the true
   marginal log-det gain is $\Delta(e\mid S) = \log(\text{cur\_diag}[\text{pick}])$,
   just never explicitly materialized since it isn't needed for selection.
3. **Monotonicity holds**:

   $$
   \Delta(e\mid S) = \log\big(1 + \sigma^{-2}\mathrm{Var}(e\mid S)\big) \;\ge\; 0 \quad \text{always,}
   $$

   since $\mathrm{Var}\ge 0 \Rightarrow 1+\sigma^{-2}\mathrm{Var}\ge 1 \Rightarrow \log(\cdot)\ge 0$.
   So $F_{\text{info}}$ is genuinely monotone nondecreasing — matches the
   paper's claim.
4. **Submodularity holds**: for a PSD kernel $K$ (RBF is PSD by Mercer's
   theorem, and $\sigma^{-2}K$ stays PSD), $\mathrm{Var}(e \mid S)$ is
   nonincreasing as $S$ grows — conditioning on more variables never
   increases posterior variance for a jointly-Gaussian-covariance-structured
   model. Hence $\Delta(e\mid S) = \log(1+\sigma^{-2}\mathrm{Var}(e\mid S))$
   is nonincreasing in $S$, which is exactly diminishing returns. This is
   the standard result behind D-optimal design / max-entropy sensor
   placement (Krause & Guestrin), and holds here because $K$ is RBF-PSD.

**So in isolation, `select_logdet` is a mathematically correct
monotone-submodular greedy for $F_{\text{info}}$, exactly as the paper
describes.** The gap isn't in that term's math — it's that it's never wired
into a combined $\alpha/\beta/\gamma$ objective anywhere in the code; it
only ever runs alone (as the untargeted `"Submod. benchmark"` baseline,
where `target_mask` is accepted as a parameter but never referenced in the
function body) or as a disjoint second phase inside Pair.

---

## What this means for the thesis

The $(1-1/e)$ guarantee for the combined
$F = \alpha F_{\text{sample}} + \beta F_{\text{model}} + \gamma F_{\text{info}}$
is currently a **paper-level claim with no matching implementation**. You
have proofs and correct code for two of the three individual terms
($F_{\text{sample}}$, $F_{\text{info}}$), but:

- $F_{\text{model}}$ as literally specified
  ($\sum_\ell \max k_M(\cdot,\cdot)$) isn't implemented at all — what runs
  instead is a different, non-facility-location model-diversity heuristic
  (`select_kcenter` over model centroids).
- No function anywhere computes the actual weighted sum
  $\alpha\Delta_{\text{sample}} + \beta\Delta_{\text{model}} + \gamma\Delta_{\text{info}}$
  per candidate and greedily maximizes *that*, so the "nonnegative weighted
  sum of monotone submodular functions is monotone submodular" closure
  argument, while true in the abstract, isn't something the repo currently
  demonstrates end-to-end.

**To get code-paper parity**, the missing piece is a single new selection
function that:

1. Implements $F_{\text{model}}$ as the actual facility-location
   coverage-of-$\mathcal M$ term
   ($\sum_{m_\ell\in\mathcal M} \max_{(m_i,X_j)\in\Omega} k_M(\psi(m_i),\psi(m_\ell))$),
   not a k-center pre-filter.
2. Computes all three $\Delta$'s for each candidate under a **shared**
   kernel/bandwidth (fixing the $\tau$ mismatch noted in Part 2).
3. Combines them with $\alpha, \beta, \gamma \ge 0$ (optionally divided by
   $c_e$ for the cost-benefit knapsack variant from `method.tex`:

   $$
   e^\star = \arg\max_{e\notin\Omega} \frac{F(\Omega\cup\{e\})-F(\Omega)}{c_e}
   $$

   ), and greedily picks the argmax of that combined score.

That's the one function code and proof would both point at for the joint
$(1-1/e)$ guarantee.

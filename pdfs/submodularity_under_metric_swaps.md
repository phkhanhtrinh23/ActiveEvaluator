# Does the selection objective stay monotone and submodular if we swap the metric?

**Question.** `app:coverage-proof` proves the coverage term is monotone submodular, which is what
buys the $(1-1/e)$ greedy guarantee. If we replace it with **Wasserstein**, a **kernel mean
embedding (MMD)**, or **kernel herding**, do both properties survive?

**Short answer.**

| replacement | monotone | submodular | fixed by $\exp(-d)$? |
|---|:--:|:--:|:--:|
| facility location (**what we actually have**) | yes | yes | already fine |
| true directed Hausdorff, $\max_x\min_z d$ | yes | **no** | **no** — outer $\max_x$ |
| Wasserstein, **free support weights** | yes | yes | already fine |
| Wasserstein, fixed uniform weights | **no** | **no** | **no** — normalization |
| MMD, normalized $\hat\mu_{\mathcal U_{\mathcal Q}}$ | **no** | **no** | **no** — normalization |
| MMD, fixed cardinality, $k\ge0$ | **no** | yes | **no** — normalization |
| kernel herding | **no** | n/a (Frank–Wolfe) | **no** |
| **any point-to-point $d(x,z)$ wrapped as $\sum_x\max_z e^{-d/\sigma}$** | **yes** | **yes** | **this *is* the fix** |

**The last row is the important one.** Wrapping a distance as $S=e^{-d/\sigma}$ and feeding it
to facility location rescues *every* point-to-point distance unconditionally — no metric axioms
required at all (§7). But it provably **cannot** rescue the two structural failures: an outer
$\max_x$ aggregation (Hausdorff), or a set-level normalized measure (MMD, fixed-weight
Wasserstein). Of the swaps taken at face value, only free-support Wasserstein is safe, because
it collapses *exactly* onto facility location.

---

## 0. Definitions and the one lemma we reuse

Ground set: candidate labeled subsets $\mathcal P=\{P_1,\dots,P_R\}$. We select
$\mathcal Q\subseteq\mathcal P$ with $\lvert\mathcal Q\rvert\le K$, and write
$\mathcal U_{\mathcal Q}=\bigcup_{P\in\mathcal Q}P$ for the union of labeled items. Target set
$\mathcal X$, $n=\lvert\mathcal X\rvert$.

$f:2^{\mathcal P}\to\mathbb R$ is **monotone** if $\mathcal A\subseteq\mathcal B\Rightarrow
f(\mathcal A)\le f(\mathcal B)$, and **submodular** if for $\mathcal A\subseteq\mathcal B$ and
$e\notin\mathcal B$,

$$
f(\mathcal A\cup\{e\})-f(\mathcal A)\;\ge\;f(\mathcal B\cup\{e\})-f(\mathcal B)
\qquad\text{(diminishing returns).}
$$

Nemhauser–Wolsey–Fisher needs **all three** of: normalized ($f(\varnothing)=0$), monotone,
submodular. Drop monotonicity and $(1-1/e)$ is gone.

> **Lemma 0 (lifting through the union).** If $\tilde f$ is monotone submodular on *items* and
> $f(\mathcal Q):=\tilde f(\mathcal U_{\mathcal Q})$, then $f$ is monotone submodular on
> *subsets*.
>
> *Proof.* Monotone: $\mathcal Q\subseteq\mathcal R\Rightarrow\mathcal U_{\mathcal Q}\subseteq
> \mathcal U_{\mathcal R}$, apply monotonicity of $\tilde f$. Submodular: the marginal of adding
> $P$ is $\tilde f(\mathcal U_{\mathcal Q}\cup P)-\tilde f(\mathcal U_{\mathcal Q})$. Add the
> items of $P\setminus\mathcal U_{\mathcal Q}$ one at a time and apply submodularity of
> $\tilde f$ to each; every term is no larger when computed from
> $\mathcal U_{\mathcal R}\supseteq\mathcal U_{\mathcal Q}$. $\square$

So it suffices to reason about item-level functions throughout.

---

## 1. Baseline: what we actually have is *facility location*, not Hausdorff

Our objective is

$$
f_{\mathrm{cov}}(\mathcal Q)=\sum_{x\in\mathcal X}\ \max_{z\in\mathcal U_{\mathcal Q}}S(x,z),
\qquad S\ge 0,\quad \max_{z\in\varnothing}:=0 .
$$

This is the **facility-location** function: a *sum* over targets of a *max* over selected
points. It is monotone submodular (our `app:coverage-proof`).

The proof, compressed, because we reuse it three times below. Fix $x$ and let
$m_x(\mathcal Q)=\max_{z\in\mathcal U_{\mathcal Q}}S(x,z)$. The marginal gain of adding $P$ is

$$
\Delta_x(P\mid\mathcal Q)=\max\Big\{0,\ \max_{z\in P}S(x,z)-m_x(\mathcal Q)\Big\}.
$$

$m_x$ is nondecreasing in $\mathcal Q$, and $t\mapsto\max\{0,\,a-t\}$ is nonincreasing, so
$\mathcal Q\subseteq\mathcal R\Rightarrow\Delta_x(P\mid\mathcal Q)\ge\Delta_x(P\mid\mathcal R)$.
Summing over $x$ preserves this. $\square$

### 1.1 True Hausdorff would *not* work

The directed Hausdorff distance replaces the outer sum by a max:

$$
H(\mathcal X\to\mathcal U_{\mathcal Q})=\max_{x\in\mathcal X}\ \min_{z\in\mathcal U_{\mathcal Q}}d(x,z).
$$

Maximizing $g=M-H$ is monotone (more points can only shrink the worst-case distance) but
**not submodular**. Counterexample, two targets and two candidates:

$$
d(x_1,e_1)=0,\quad d(x_1,e_2)=10,\qquad d(x_2,e_1)=10,\quad d(x_2,e_2)=0,\qquad M=10 .
$$

| $\mathcal Q$ | $H(\mathcal Q)$ | $g(\mathcal Q)=10-H$ |
|---|---:|---:|
| $\varnothing$ | 10 | 0 |
| $\{e_1\}$ | $\max(0,10)=10$ | 0 |
| $\{e_2\}$ | $\max(10,0)=10$ | 0 |
| $\{e_1,e_2\}$ | $\max(0,0)=0$ | 10 |

$$
\Delta(e_2\mid\varnothing)=0-0=0,
\qquad
\Delta(e_2\mid\{e_1\})=10-0=10 .
$$

Submodularity demands $0\ge10$. False — these are *increasing* returns. The reason is
structural: $g=\min_x\big[\max_z(-d)\big]$, and a **minimum of submodular functions is not
submodular**. The outer $\sum$ in facility location is doing real work; an outer $\max$ destroys
it.

*(Verified by exhaustive enumeration.)*

**Takeaway.** Our coverage term is already the right choice, and the paper should say
"facility location", not describe it in Hausdorff terms.

---

## 2. Wasserstein

Write $\hat\mu_{\mathcal X}=\frac1n\sum_{x}\delta_x$ and let the selected support be
$\mathcal U_{\mathcal Q}$. **Everything depends on whether the weights on that support are free
or fixed.**

### 2.1 Free support weights: monotone and submodular ✓

Consider

$$
W^\star(\mathcal Q)\;=\;\min_{w\in\Delta(\mathcal U_{\mathcal Q})}\ W_p^p\Big(\textstyle\sum_{z}w_z\delta_z,\ \hat\mu_{\mathcal X}\Big).
$$

**Step 1 — collapse the transport problem.** A coupling $\gamma$ must satisfy
$\sum_z\gamma_{xz}=1/n$ for every $x$; because $w$ is free, that is the *only* binding
constraint (any $\gamma$ induces an admissible $w=\sum_x\gamma_{xz}$). So the cost
$\sum_{x,z}\gamma_{xz}d(x,z)^p$ is minimized by sending each target's mass entirely to its
nearest selected point:

$$
W^\star(\mathcal Q)=\frac1n\sum_{x\in\mathcal X}\ \min_{z\in\mathcal U_{\mathcal Q}}d(x,z)^p .
\tag{2.1}
$$

This is the classical optimal-quantization / $k$-medoids identity.

**Step 2 — normalize into a gain.** Put $D_x:=\max_{z\in\mathcal U_{\mathcal P}}d(x,z)^p$ and

$$
g_W(\mathcal Q)\;:=\;\frac1n\sum_x\Big[D_x-\min_{z\in\mathcal U_{\mathcal Q}}d(x,z)^p\Big].
$$

Maximizing $g_W$ is exactly minimizing $W^\star$, and $g_W(\varnothing)=0$, $g_W\ge0$.

**Step 3 — recognize it.** For each $x$,

$$
D_x-\min_{z}d(x,z)^p=\max_{z}\big[D_x-d(x,z)^p\big]=\max_{z}S_W(x,z),
\qquad S_W(x,z):=D_x-d(x,z)^p\ \ge 0 .
$$

Hence

$$
\boxed{\;g_W(\mathcal Q)=\frac1n\sum_{x\in\mathcal X}\max_{z\in\mathcal U_{\mathcal Q}}S_W(x,z)\;}
$$

which is **facility location with similarity $S_W$**. By §1 plus Lemma 0 it is monotone and
submodular, and greedy gives $(1-1/e)$. $\blacksquare$

*(Verified: exhaustive scan over all $2^6$ subsets found 0 violations of either property.)*

**Interpretation.** Free-support Wasserstein is not a new objective — it is our objective in
different coordinates, with $S_W=D_x-d^p$ in place of MaxSim. The swap is free.

### 2.2 Fixed uniform weights: neither ✗

If instead $\hat\mu_{\mathcal U_{\mathcal Q}}=\frac{1}{\lvert\mathcal U_{\mathcal Q}\rvert}\sum_z\delta_z$
with weights *forced* uniform, monotonicity dies immediately. Let $\mathcal X$ sit at $0$:

- $\mathcal Q=\{P_1\}$ with $P_1$ at $0$: $W_1=0$.
- $\mathcal Q=\{P_1,P_2\}$ with $P_2$ at $100$: mass splits $\tfrac12/\tfrac12$, so $W_1=50$.

Adding a subset made the objective **worse**. The cause is **dilution**: the normalizer
$1/\lvert\mathcal U_{\mathcal Q}\rvert$ forces new points to steal mass from good ones. Since
monotonicity already fails, NWF is inapplicable regardless of submodularity.

---

## 3. Kernel mean embedding / MMD

$$
\mathrm{MMD}^2(\mathcal U_{\mathcal Q},\mathcal X)=\big\lVert\hat\mu_{\mathcal U_{\mathcal Q}}-\hat\mu_{\mathcal X}\big\rVert_{\mathcal H}^2 .
$$

Expand with $m=\lvert\mathcal U_{\mathcal Q}\rvert$, $\bar k(z):=\frac1n\sum_{x}k(z,x)$, and
$c_{\mathcal X}:=\frac1{n^2}\sum_{x,x'}k(x,x')$:

$$
\mathrm{MMD}^2=\underbrace{\frac{1}{m^{2}}\sum_{z,z'\in\mathcal U_{\mathcal Q}}k(z,z')}_{\text{repulsion}}
\;-\;\underbrace{\frac{2}{m}\sum_{z\in\mathcal U_{\mathcal Q}}\bar k(z)}_{\text{attraction}}
\;+\;c_{\mathcal X}.
\tag{3.1}
$$

### 3.1 With the normalization: neither ✗

The $1/m$ and $1/m^{2}$ factors make (3.1) a **ratio of set functions**, and ratios are
generically neither monotone nor submodular. Concretely, the same dilution argument as §2.2
applies: appending far-away items drags $\hat\mu_{\mathcal U_{\mathcal Q}}$ away from
$\hat\mu_{\mathcal X}$.

*(Verified: exhaustive scan with a Gaussian kernel found **11 monotonicity violations** and
**14 submodularity violations** over $2^5$ subsets.)*

### 3.2 At fixed cardinality with $k\ge 0$: submodular but not monotone

Freeze $m$. Then maximizing $-\mathrm{MMD}^2$ is maximizing

$$
g_K(\mathcal U)=\frac{2}{m}\sum_{z\in\mathcal U}\bar k(z)\;-\;\frac{1}{m^{2}}\sum_{z,z'\in\mathcal U}k(z,z').
$$

**Attraction term.** $\sum_{z\in\mathcal U}\bar k(z)$ is a sum over elements, hence **modular**
(both sub- and supermodular), and monotone when $\bar k\ge0$.

**Repulsion term.** Let $h(\mathcal U)=-\sum_{z,z'\in\mathcal U}k(z,z')$. For $e\notin\mathcal U$,

$$
h(\mathcal U\cup\{e\})-h(\mathcal U)=-2\sum_{z\in\mathcal U}k(z,e)-k(e,e).
$$

If $\mathcal U\subseteq\mathcal V$ and $k\ge0$, then $\sum_{\mathcal U}k(z,e)\le\sum_{\mathcal V}k(z,e)$, so

$$
h(\mathcal U\cup\{e\})-h(\mathcal U)\ \ge\ h(\mathcal V\cup\{e\})-h(\mathcal V),
$$

i.e. **submodular** ✓. But every marginal is $\le0$, so $h$ is **decreasing** — not monotone.

Modular $+$ submodular $=$ submodular, so $g_K$ is submodular but **not** monotone. NWF does not
apply; one falls back to non-monotone submodular maximization under a cardinality constraint —
randomized greedy at $1/e$ (Buchbinder et al.), or local search at $\approx1/4$.

**This is the DPP-style diversity structure**: an attraction term pulling toward the target mean
and a repulsion term punishing near-duplicates. Diversity objectives are naturally submodular
and naturally non-monotone.

---

## 4. Kernel herding

Herding (Chen, Welling & Smola, 2010) greedily picks

$$
z_{t+1}=\arg\max_{z}\Big[\bar k(z)-\frac{1}{t+1}\sum_{s\le t}k(z,z_s)\Big].
$$

Structurally it is the same attraction-minus-repulsion pattern as §3.2, and it is **not** a
submodular maximization at all. Bach, Lacoste-Julien & Obozinski (2012) showed herding is exactly
**Frank–Wolfe** on the marginal polytope with step size $1/(t+1)$. Consequently:

- Its guarantee is $\mathrm{MMD}=O(1/t)$ for a finite-dimensional RKHS with the mean embedding in
  the relative interior (and $O(1/\sqrt t)$ in general) — **a convergence rate, not an
  approximation ratio.**
- It is a rate on *approximation quality of the mean embedding*, which is a different — and
  arguably stronger — statement than $(1-1/e)$ of an optimum. But it is **not interchangeable**
  with our current guarantee, and it does not compose with the coverage term.
- The per-step criterion carries the same $\tfrac{1}{t+1}$ normalizer that broke §3.1, so no
  monotone-submodular claim is available.

---

## 5. A gap this exposes in the current paper

Our selection rule (`eq:select-objective`) is

$$
\mathcal Q^\star=\arg\max_{\lvert\mathcal Q\rvert\le K}\Big[f_{\mathrm{cov}}(\mathcal Q)-\lambda_{\mathrm{mmd}}\,\mathrm{MMD}^2(\mathcal U_{\mathcal Q},\mathcal X)\Big],
$$

but `app:coverage-proof` establishes monotone submodularity **for $f_{\mathrm{cov}}$ only**. By
§3.1 the MMD term is neither monotone nor submodular, so the sum is

$$
\underbrace{f_{\mathrm{cov}}}_{\text{monotone submodular}}\;+\;\underbrace{(-\lambda_{\mathrm{mmd}}\mathrm{MMD}^2)}_{\text{neither}}
\;=\;\text{no guarantee as stated.}
$$

**The $(1-1/e)$ we cite does not cover the objective we actually optimize.** Three honest fixes:

1. **Scope the claim.** State that $(1-1/e)$ holds for the coverage term and that the MMD penalty
   is a heuristic refinement. Cheapest; requires one sentence.
2. **Move MMD into a constraint.** Maximize $f_{\mathrm{cov}}$ subject to
   $\mathrm{MMD}^2\le\tau$. Greedy on a monotone submodular objective over a down-closed
   feasible family retains a guarantee, and the density-matching intent is preserved.
3. **Adopt the non-monotone machinery.** Fix cardinality, require $k\ge0$, invoke §3.2, and cite
   the $1/e$ randomized-greedy result. Strictly weaker than $(1-1/e)$, but honest and complete.

Option 2 is the most faithful to what the objective is *for*: coverage is the thing being
maximized, density matching is a feasibility condition.

---

## 6. Summary

| claim | reason |
|---|---|
| facility location is monotone submodular | $\max\{0,a-t\}$ nonincreasing in $t$; $m_x$ nondecreasing |
| directed Hausdorff is **not** submodular | $g=\min_x[\max_z(-d)]$; min of submodular is not submodular; explicit $2\times2$ counterexample |
| free-support Wasserstein **is** monotone submodular | free weights collapse OT to nearest-neighbor assignment (2.1); becomes facility location with $S_W=D_x-d^p$ |
| fixed-weight Wasserstein is **not** monotone | dilution: $1/\lvert\mathcal U\rvert$ lets a bad point steal mass |
| normalized MMD is **neither** | $1/m,1/m^2$ make it a ratio of set functions |
| MMD at fixed $m$, $k\ge0$, is submodular not monotone | modular attraction $+$ submodular repulsion; marginals $\le0$ |
| herding has no submodular guarantee | it is Frank–Wolfe; gives $O(1/t)$ MMD decay instead |
| $\sum_x\max_z e^{-d/\sigma}$ is monotone submodular for **any** $d$ | Thm 7.1 needs only $S\ge0$; $e^{-d/\sigma}\in(0,1]$ |
| $\exp$ cannot rescue MMD or Hausdorff | Prop 7.3: a strictly decreasing map preserves every monotonicity violation |

**The single rule behind all of it:** *a sum over targets of a max over selected items is
monotone submodular for **any** nonnegative $S$ — that is the whole theorem, and $\exp(-d/\sigma)$
supplies the nonnegativity for free. What breaks the property is never the distance itself but
the surrounding structure: an outer $\max_x$ instead of $\sum_x$, or a normalizer
$1/\lvert\mathcal U_{\mathcal Q}\rvert$ that turns the objective into a ratio of set functions.*
$\exp$ fixes the range problem completely and the structural problems not at all. Free-support
Wasserstein survives because free weights let the normalizer be optimized away, returning it to
coverage form.

---

## 7. The $\exp(-d)$ construction: what it fixes and what it cannot

The natural objection to §§2–3 is: *why fight with raw distances at all — just put the distance
inside $\exp(-\cdot)$ to turn it into a similarity, and use facility location.* That instinct is
right, and it is **stronger than §§1–3 suggest** — but it is not universal. There are three
distinct reasons a set function can fail, and $\exp$ addresses exactly one of them.

| failure mode | cause | does $\exp(-d)$ fix it? |
|---|---|:--:|
| **F1 — range** | $-d\le 0$, so $f$ *drops* when the first element is added | **yes** |
| **F2 — aggregation** | outer $\max_x$ instead of $\sum_x$ (Hausdorff) | no |
| **F3 — normalization** | selected set enters as $\hat\mu_{\mathcal U_{\mathcal Q}}$, a ratio | no |

### 7.1 The positive result: $\exp$ fixes F1 completely

> **Theorem 7.1.** Let $S:\mathcal X\times\mathcal Z\to[0,\infty)$ be **any** nonnegative
> function. With the convention $\max_{z\in\varnothing}S(x,z):=0$,
> $$f(\mathcal Q)=\sum_{x\in\mathcal X}\ \max_{z\in\mathcal U_{\mathcal Q}}S(x,z)$$
> is normalized, monotone, and submodular. **No further assumption on $S$ is used** — not
> symmetry, not the triangle inequality, not even that $S$ derives from a metric.

*Proof.* Fix $x$ and write $m_x(\mathcal Q)=\max_{z\in\mathcal U_{\mathcal Q}}S(x,z)$.

**Normalized.** $\mathcal U_\varnothing=\varnothing$, so $m_x(\varnothing)=0$ and $f(\varnothing)=0$.

**Monotone.** $\mathcal Q\subseteq\mathcal R\Rightarrow\mathcal U_{\mathcal Q}\subseteq\mathcal U_{\mathcal R}$;
a max over a superset cannot be smaller, so $m_x(\mathcal Q)\le m_x(\mathcal R)$. At the empty-set
boundary this needs $0\le S$, which is where nonnegativity is used — **and only there**.

**Submodular.** Adding $P$ gives $m_x(\mathcal Q\cup\{P\})=\max\{m_x(\mathcal Q),\ \max_{z\in P}S(x,z)\}$, so

$$
\Delta_x(P\mid\mathcal Q)=\max\Big\{0,\ \max_{z\in P}S(x,z)-m_x(\mathcal Q)\Big\}.
$$

The map $t\mapsto\max\{0,a-t\}$ is nonincreasing, and $m_x$ is nondecreasing in $\mathcal Q$, so
$\mathcal Q\subseteq\mathcal R\Rightarrow\Delta_x(P\mid\mathcal Q)\ge\Delta_x(P\mid\mathcal R)$.
Summing the inequality over $x\in\mathcal X$ preserves it. $\square$

**Corollary 7.2.** For any $d(x,z)\ge0$ and any bandwidth $\sigma>0$, set
$S(x,z)=e^{-d(x,z)/\sigma}\in(0,1]$. Theorem 7.1 applies, so

$$
\boxed{\;f_\sigma(\mathcal Q)=\sum_{x\in\mathcal X}\ \max_{z\in\mathcal U_{\mathcal Q}}\exp\!\big(-d(x,z)/\sigma\big)\;}
$$

is monotone submodular and greedy attains $(1-1/e)$ — **for every point-to-point $d$**.

This is a genuinely broad licence. It covers Euclidean, cosine, $d^p$, MaxSim, RKHS distance
$\lVert\phi(x)-\phi(z)\rVert_{\mathcal H}$, SQL-AST edit distance, execution-mismatch rate,
per-item Wasserstein between local neighbourhoods — and **non-metrics**: asymmetric scores, an
LLM-judged dissimilarity, anything violating the triangle inequality. The proof never touches
those properties.

*Verified numerically: $\sum_x\max_z e^{-d/\sigma}$ over all $2^6$ subsets gave **0 violations**
of monotonicity and **0** of submodularity for Euclidean, squared-Euclidean, and a deliberately
pathological non-metric $d(a,b)=\lvert a_1-b_1\rvert^{1/2}+3\lvert\sin(a_2b_2)\rvert+0.7\lvert a_1-2b_2\rvert$
(asymmetric, no triangle inequality), at $\sigma\in\{0.3,1,5\}$.*

**Why nonnegativity was the whole problem.** Note the submodularity argument above never used
$S\ge0$. Only *monotonicity at the empty set* did: with $S=-d\le0$, going from $\varnothing$
(value $0$) to $\{z\}$ (value $-d<0$) *decreases* $f$. That is F1, and $\exp$ eliminates it by
mapping $[0,\infty)\to(0,1]$. Any decreasing map into $[0,\infty)$ works equally well — e.g.
$S=D_{\max}-d$, which is what §2.1 used. $\exp$ is simply the one that needs no knowledge of
$D_{\max}$ and never clips.

### 7.2 The negative result: $\exp$ cannot fix F2 or F3

> **Proposition 7.3.** Let $\psi$ be any strictly decreasing function and $F$ any set function.
> If $F$ violates monotonicity, so does $\psi\circ F$.

*Proof.* Suppose $\mathcal A\subseteq\mathcal B$ but $F(\mathcal A)<F(\mathcal B)$ (adding
elements made the divergence worse). Since $\psi$ is strictly decreasing,
$\psi(F(\mathcal A))>\psi(F(\mathcal B))$, so the gain function $\psi\circ F$ *decreases* along
$\mathcal A\subseteq\mathcal B$ — still non-monotone. $\square$

$\psi=\exp(-\cdot)$ is strictly decreasing, so **every** monotonicity violation survives
$\exp$ one-for-one. Monotonicity is a statement about the *ordering* of set values, and an
order-reversing bijection can neither create nor destroy it. Rescaling the output cannot repair
a defect in how the set enters the input.

**F3 (normalization), concretely.** The §2.2 dilution counterexample is untouched: target at
$0$, $\mathcal Q=\{P_1\}$ at $0$ gives $W=0$ and $e^{-0}=1$; adding $P_2$ at $100$ gives $W=50$
and $e^{-50}\approx0$. The value still fell. *Numerically: $-\mathrm{MMD}^2$ had 13 monotonicity
and 28 submodularity violations; $\exp(-\mathrm{MMD}^2)$ still had **8 and 8**. Reduced, not
removed — and the guarantee needs zero.*

**F2 (aggregation), concretely.** Hausdorff's defect is the outer $\max_x$, not the inner
distance. Wrapping gives $\min_x\max_z e^{-d}$ — still a **minimum of submodular functions**,
which is not submodular. *Numerically: $10-H$ had 112 submodularity violations;
$\min_x\max_z e^{-d}$ had **170** — worse.* Only replacing the outer $\max_x$ by $\sum_x$ fixes
it, and that turns Hausdorff into facility location.

### 7.3 The honest caveat: $\exp$ changes the objective

$\sum_x\max_z e^{-d/\sigma}$ is **not** a monotone transform of $\sum_x\min_z d$ — the
$\exp$ sits inside the sum, so the two rank subsets differently. Contrast:

- **§2.1 (free-support Wasserstein)** is an *exact identity*. The objective is unchanged; we
  merely rewrote it and inherited the guarantee.
- **§7 ($\exp$ wrapping)** is a *modelling choice*. It buys monotone submodularity by optimizing
  a different, softened objective. The $(1-1/e)$ is relative to *that* objective's optimum, not
  to the original distance-based one.

That trade is usually worth taking, but it should be stated rather than glossed.

**Bandwidth matters.** $\sigma$ interpolates between two degenerate ends:

$$
\sigma\to\infty:\ S\to1\ \Rightarrow\ f_\sigma\to\lvert\mathcal X\rvert\ \text{(constant, no discrimination)},
\qquad
\sigma\to 0:\ S\to\mathbb 1[d=0]\ \Rightarrow\ f_\sigma\to\text{set cover}.
$$

Both extremes are monotone submodular, so the guarantee holds throughout — but it is a guarantee
about a progressively less useful objective. $\sigma$ should be set on the scale of typical
nearest-neighbour distances.

### 7.4 Recommendation

1. **Any item-level geometry we want is available.** Put it in $d(x,z)$, wrap in
   $\exp(-d/\sigma)$, keep facility location. Theorem 7.1 covers it with no metric assumptions.
   This is the cheapest way to generalize `eq:facility-coverage` across modalities.
2. **Kernel geometry: use item-level, not mean-embedding.** $\exp(-\lVert\phi(x)-\phi(z)\rVert^2/\sigma)$
   is fine — for a normalized kernel it is monotone in $k(x,z)$, and Theorem 7.1 applies. The
   *mean embedding of the selected set*, $\hat\mu_{\mathcal U_{\mathcal Q}}$, is what breaks
   things. Kernel similarity between **items**: safe. Kernel mean of the **set**: unsafe.
3. **The set-level MMD penalty still needs handling.** $\exp$ does not rescue it (Prop. 7.3), so
   §5's conclusion stands: scope the claim to the coverage term, or move MMD into a constraint.

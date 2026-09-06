# Notes on `Trinh_proof_facility_location.tex` — three questions

Answers keyed to the source at `../Trinh_proof_facility_location.tex`.
Math is written in `$…$` / `$$…$$`; render with a Markdown previewer that
supports KaTeX/MathJax (VS Code: *Markdown: Open Preview*).

---

## Q1 — Is the §24 greedy algorithm selecting items or subsets? What is $\mathcal{R}$? And why append when $\Delta \le 0$?

### 1.1 Items or subsets: both, at two different levels

This is the source of the confusion, and it resolves cleanly once the two
levels are separated (§2, lines 116–147):

$$B \;\longrightarrow\; b_i \;\longrightarrow\; x_{im}$$

- $B=\{b_1,\dots,b_N\}$ is the **ground set** — the meta-dataset, $N=113$ workloads.
- $b_i=\{x_{i1},\dots,x_{iM_i}\}$ is one **element** of that ground set, and is
  itself a set of examples (a point cloud in $\mathbb{R}^d$).
- $x_{im}\in\mathbb{R}^d$ is a single embedded example.

Therefore:

| Object | Level | What it is |
|---|---|---|
| $A \subseteq B$, $\lvert A\rvert \le K$ | the **answer** | a subset of the ground set |
| $b_{i_t}$ picked at round $t$ | the **move** | a single element of the ground set |
| $x_{im}$ | inside an element | never touched by the greedy loop |

So both readings are correct and there is no contradiction:

> The final answer $A$ **is** a subset of $B$. Greedy constructs that subset
> **one element at a time**, and each element happens to be a dataset.

`for i ∈ R` iterates over single candidate workloads because that is how a
subset is built greedily: $A_0 \subset A_1 \subset \cdots \subset A_K$. The set
function $F:2^B \to \mathbb{R}$ is defined on subsets; greedy simply never
evaluates more than $N$ of them per round.

### 1.2 What $\mathcal{R}$ is

$\mathcal{R}$ is the set of **remaining candidate indices** — those not yet
selected. It starts as $\{1,\dots,N\}$, and step (4) removes $i_t$:

$$\mathcal{R} \;=\; \{1,\dots,N\}\setminus\{i_1,\dots,i_{t-1}\}\quad\text{at round } t.$$

In code this is literally:

```python
remaining = set(range(n))       # R
remaining.discard(best_idx)     # R <- R \ {i_t}
```

Three observations worth recording:

1. **$\mathcal{R}$ holds indices, not workloads.** Hence step (4) mixes both
   notations in one line: $A_t \leftarrow A_{t-1}\cup\{b_{i_t}\}$ (workloads)
   but $\mathcal{R}\leftarrow\mathcal{R}\setminus\{i_t\}$ (indices). Consistent
   with the initialization $\mathcal{R}\leftarrow\{1,\dots,N\}$, but the box
   would read more cleanly if this were stated once.

2. **Removing $i_t$ is purely an efficiency measure, not a correctness one.**
   After the coverage update (5), $c(j)\ge S_{i_t j}$ for every $j$, so by
   Theorem `thm:gain` an already-selected element has marginal gain exactly $0$
   and could never win the $\arg\max$ again.

3. **The sum in step (1) runs over all of $B$, not over $\mathcal{R}$.**

$$\Delta(b_i \mid A_{t-1}) \;=\; \sum_{j=1}^{N}\max\bigl(0,\;S_{ij}-c(j)\bigr)$$

   This is the $i$/$j$ role distinction from §2.1 (line 166): $i$ indexes the
   **candidate being bought**, $j$ indexes the **workload being judged**. A
   workload already purchased must still be counted as covered, so $j$ ranges
   over all $N$.

### 1.3 Step (3): the control flow — we do **not** append

The premise of the question is a misreading of the pseudocode. Step (3) reads

> (3) **if** $\Delta(b_{i_t}\mid A_{t-1})\le 0$ **then stop**

and *stop* terminates the loop, so step (4) never executes in that round. The
code makes this unambiguous — `break` precedes `append`:

```python
best_gain = float(gains[best_local])
if best_gain <= 0:              # (3) stop
    break
selected.append(best_idx)       # (4) unreachable when the gain is 0
```

### 1.4 The gain can never be negative

Separately from the control flow: **$\Delta \ge 0$ always.** By Theorem
`thm:gain`,

$$\Delta(b_x \mid A) \;=\; \sum_{j=1}^{N}\max\bigl(0,\;S_{xj}-c_A(j)\bigr),$$

and every summand is clipped at zero, so the whole sum is non-negative. This is
Corollary `cor:gainpos` (lines 1112–1119). Consequently:

$$\Delta \le 0 \;\;\Longleftrightarrow\;\; \Delta = 0,$$

which means *no remaining candidate improves the objective at all*. Since
$S_{ij}=\exp(-D_{ij}^2/\tau) > 0$ strictly (the Gaussian never vanishes), this
happens only in degenerate floating-point situations. It is a safety valve, not
part of the normal path — exactly as the remark following the code says.

### 1.5 One genuine imprecision to fix

The pseudocode box ends with

> **return** $A_K$

but on an early stop at round $t$ the algorithm returns $A_{t-1}$, not $A_K$.
The code is correct (`return selected`); the pseudocode is not. Suggested fix:
`return A_{t-1}` or `return the current A`. (The output line already hedges with
$\lvert A\rvert \le K$ rather than $=K$, so the intent is there.)

---

## Q2 — With Sliced Wasserstein or kernel mean / MMD under $\exp(-\cdot/\tau)$, do monotonicity and submodularity still hold?

### 2.1 Yes — unconditionally

The document already contains the reason, in §15 (lines 788–795):

> Every proof in Part V uses only two facts about $S$: that its entries are
> **non-negative**, and that $\max$ is monotone.

Facility location

$$F(A)\;=\;\sum_{j=1}^{N}\max_{b_i \in A} S_{ij}$$

is monotone and submodular for **any** non-negative matrix $S$. Not required:
symmetry, the triangle inequality, positive semi-definiteness, or any metric
axiom whatsoever. This is precisely why the document can already drive facility
location with the non-Hilbertian Hausdorff metric.

So both of the following are monotone, submodular, normalized
($F(\emptyset)=0$) and non-negative:

$$S_{ij}=\exp\!\Bigl(-\frac{\mathrm{SW}_2(b_i,b_j)^2}{\tau}\Bigr),
\qquad
S_{ij}=\exp\!\Bigl(-\frac{\mathrm{MMD}_k(b_i,b_j)^2}{\tau}\Bigr).$$

Nemhauser–Wolsey–Fisher and the $1-1/e$ guarantee therefore transfer with **zero
change** to Part V or Part VI.

**One caution.** If kernel mean embeddings are used as a raw inner product
$S_{ij}=\langle \mu_i,\mu_j\rangle$ rather than through $\exp(-\mathrm{MMD}^2/\tau)$,
then $S_{ij}\ge 0$ must be checked. It holds for a non-negative base kernel (e.g.
Gaussian) but not in general — and non-negativity is the one hypothesis that
cannot be dropped.

### 2.2 What does **not** transfer for free: Part VIII

| Result | What it needs | Survives SW / MMD? |
|---|---|---|
| Lemma `lem:Fdist` | $\phi$ strictly decreasing | Yes, verbatim |
| Lemma `lem:cover`(b) | $\phi$ decreasing, $\phi(0)=1$ | Yes, verbatim |
| Assumption `ass:lip` ($L$-Lipschitz accuracy) | the **specific** metric | **The weak link** |
| Theorem `thm:cover` | `ass:lip` | Only as strong as the assumption |
| Prop `prop:cost` (early termination) | the $\max\min$ structure | **No** — Hausdorff-only |

The remark at lines 1970–1985 already states the issue precisely. The
plausibility argument for

$$\bigl\lvert a^\star(b)-a^\star(b')\bigr\rvert \;\le\; L\, d(b,b')$$

works for $d_H$ **because $d_H$ is worst-case**: small $d_H$ means every example
in $b$ has a nearby counterpart in $b'$ and vice versa. Under an average-case
distance such as SW or MMD, a small distance is compatible with a subpopulation
being wildly unmatched, and accuracy can move a great deal on that
subpopulation.

Both $\mathrm{SW}_p$ and $\mathrm{MMD}$ with a characteristic kernel are genuine
metrics, so $\varepsilon(A)$ and the Lipschitz statement remain **well-formed** —
the theorem still holds as stated. What degrades is the *defensibility* of the
assumption: the constant $L$ needed is much larger, possibly unusably so.

### 2.3 Two practical notes

- **Re-run the median heuristic per distance.** $\tau=\operatorname{median}\{D_{ij}^2 : i<j\}$
  has the units of the distance squared, and SW / MMD / Hausdorff units are not
  comparable. The code already does this from the data, so it is handled
  automatically — but the value of $\tau$ is not portable between distances.
- **The early-termination cost argument dies.** Proposition `prop:cost`
  exploits the $\max\min$ structure for an $O(M)$ expected inner loop. SW costs
  $O(LM\log M)$ per pair for $L$ projections; MMD costs $O(M^2 d)$ with no early
  exit.

### 2.4 Summary

> Submodularity is cheap and universal. The **transfer guarantee** is what
> actually earns Hausdorff its place in the pipeline.

---

## Q3 — Is this the true Hausdorff, and does $\exp(-\cdot/\tau)$ make it monotone and submodular?

### 3.1 Yes, it is the true Hausdorff — exactly, and computed exactly

§7–§8 define

$$h(b_i,b_j)\;=\;\max_{x\in b_i}\ \min_{y\in b_j}\ \lVert x-y\rVert_2
\qquad\text{(directed)}$$

$$D^{\mathrm H}_{ij}\;=\;d_H(b_i,b_j)\;=\;\max\bigl\{h(b_i,b_j),\;h(b_j,b_i)\bigr\}
\qquad\text{(symmetric)}$$

That is the textbook symmetric Hausdorff distance on non-empty finite subsets of
$\mathbb{R}^d$. §10 (Theorem `thm:metric`) proves all four metric axioms,
including the triangle inequality — which is what licenses the Lipschitz
condition in Theorem `thm:cover`.

The implementation uses SciPy's `directed_hausdorff`, whose early termination is
an **exact** optimization (Taha & Hanbury 2015), symmetrized by `max`. No
subsampling, no approximation, no relaxation.

### 3.2 No — the exponential is **not** what makes the objective submodular

This is the same misconception as Q2, and it is worth correcting explicitly.
Monotonicity and submodularity come from the **max-of-non-negatives structure of
facility location**, and would hold for any non-negative affinity whatsoever.

What the transform actually buys is the four requirements of §12 (lines
657–666), which are about making the *optimization problem well-posed*, not
about submodularity:

1. **Reverses the sense.** Distance is small-is-good; the objective is
   maximized. Without a decreasing $\phi$ you would be maximizing in the wrong
   direction.
2. **Non-negativity and boundedness**, $S\in(0,1]$ with $S_{ii}=1$. *This* is the
   load-bearing property — but what is needed is $S\ge 0$, not the exponential
   specifically.
3. **Saturation.** Distant workloads contribute $\approx 0$ rather than
   dominating, which is what turns the sum into a coverage count.

### 3.3 The sharpest way to see it

Suppose $S_{ij}=-D_{ij}$ were used directly. Then

$$F(A)\;=\;\sum_{j=1}^{N}\max_{b_i\in A}(-D_{ij})\;=\;-\sum_{j=1}^{N}\min_{b_i\in A} D_{ij},$$

the $k$-medians objective. This is **still monotone and submodular** — $\max$
over a growing set is monotone-submodular for arbitrary real entries, positive
or negative. What breaks is *normalization*: $F(\emptyset)=-\infty$ and $F<0$, so
the hypotheses of Theorem `thm:nwf` ($F:2^B\to\mathbb{R}_{\ge 0}$ with
$F(\emptyset)=0$) fail and the $1-1/e$ bound becomes unavailable.

> **That is the precise job the exponential does:** it supplies non-negativity
> and normalization, which NWF requires — not submodularity, which comes for
> free from the $\max$.

### 3.4 Where the functional form *does* matter: downstream

The specific choice $\phi(t)=\exp(-t^2/\tau)$ is genuinely load-bearing in
Part VIII, because it is **invertible**. Lemma `lem:Fdist` gives

$$F(A)\;=\;\sum_{j=1}^{N}\exp\!\Bigl(-\frac{d(j,A)^2}{\tau}\Bigr),$$

and inverting it yields the covering-radius bound of Lemma `lem:cover`(b): with
deficit $\Delta := N-F(A) < 1$,

$$\varepsilon(A)\;\le\;\sqrt{\ \tau\,\ln\frac{1}{1-\Delta}\ },$$

which chains into Corollary `cor:chain`:

$$\max_{j}\bigl\lvert a^\star(b_j)-a^\star(b_{i^\star(j)})\bigr\rvert
\;\le\; L\sqrt{\tau\,\ln\tfrac{1}{1-\Delta}}.$$

A different decreasing $\phi$ would give a different — still valid — inversion
and a different bound.

### 3.5 Summary

> The transform is **structurally irrelevant** in Part V (submodularity) and
> **quantitatively load-bearing** in Part VIII (the covering-radius inversion),
> with one exception in between: non-negativity, which Part VI's $1-1/e$
> guarantee genuinely requires.

---

## Suggested edits to the source

1. **§24 pseudocode, `return` line** — change `return $A_K$` to `return $A_{t-1}$`
   (or "the current $A$") so the early-stop path is stated correctly.
2. **After Corollary `cor:gainpos`** — add one sentence making explicit that
   since $\Delta \ge 0$ always, the step-(3) test $\Delta \le 0$ should be read as
   $\Delta = 0$, i.e. "nothing left to gain", not "the gain went negative".
3. **§24 step (4)/(5)** — optionally state once that $\mathcal{R}$ holds indices
   while $A$ holds workloads, since the two notations appear on the same line.

---
---

# Addendum — reconciling with `submodularity_under_metric_swaps.md`

**Question.** `../../active_model_eval/pdfs/submodularity_under_metric_swaps.md` states that
*"true directed Hausdorff, $\max_x\min_z d$ — monotone: yes, submodular: **no**, fixed by
$\exp(-d)$: **no**"*. The notes above state that Hausdorff-driven facility location **is**
monotone submodular. Is one of them wrong?

**Answer: no. Both are correct.** They are statements about two different functions, at two
different levels of the hierarchy. The word "Hausdorff" appears in each, but in a structurally
different position.

---

## A.1 The two objects

### Object A — Hausdorff as the *affinity* (this project, `active_eval`)

$$D_{ij}=d_H(b_i,b_j)=\max\Bigl\{\max_{x\in b_i}\min_{y\in b_j}\lVert x-y\rVert,\ \max_{y\in b_j}\min_{x\in b_i}\lVert x-y\rVert\Bigr\}$$

$$S_{ij}=\exp\!\bigl(-D_{ij}^2/\tau\bigr),
\qquad
F(A)=\sum_{j=1}^{N}\ \max_{b_i\in A} S_{ij}.$$

The $\max\min$ lives **inside the computation of one scalar** $D_{ij}$ — it ranges over
*examples* $x,y$ within a fixed pair of workloads. It happens entirely **below** the level of
the set function. By the time $F$ sees anything, $D_{ij}$ is just a number.

The set-function-level aggregation is $\sum_j \max_{b_i\in A}$ — a **sum of maxes**. That is
facility location. **Monotone and submodular.**

### Object B — Hausdorff-shaped *aggregation* (the `active_model_eval` doc, §1.1)

$$H(\mathcal X \to \mathcal U_{\mathcal Q})=\max_{x\in\mathcal X}\ \min_{z\in\mathcal U_{\mathcal Q}}d(x,z),
\qquad g(\mathcal Q)=M-H(\mathcal Q).$$

Here the outer $\max_x$ ranges over the **targets being covered**, i.e. it *is* the aggregation
the set function performs. That makes $g$ a **minimum of submodular functions**, which is not
submodular in general. Their $2\times2$ counterexample is correct:

| $\mathcal Q$ | $H$ | $g=10-H$ |
|---|---:|---:|
| $\varnothing$ | 10 | 0 |
| $\{e_1\}$ | 10 | 0 |
| $\{e_2\}$ | 10 | 0 |
| $\{e_1,e_2\}$ | 0 | 10 |

$\Delta(e_2\mid\varnothing)=0$ but $\Delta(e_2\mid\{e_1\})=10$ — *increasing* returns.
Submodularity would demand $0\ge 10$. This is the standard fact that **$k$-center is not
submodular**, and it is right.

### The distinction in one line

> **Hausdorff as the edge weight $S_{ij}$: submodular. Hausdorff as the outer aggregation:
> not submodular.** The failure is never in the distance; it is in what sits outside the sum.

The metric-swaps document says exactly this itself, in its own §7 taxonomy: F2 ("outer $\max_x$
instead of $\sum_x$") is a property of the **aggregation**, not of $d$. And its own final table
row — *"any point-to-point $d(x,z)$ wrapped as $\sum_x\max_z e^{-d/\sigma}$: monotone yes,
submodular yes"* — **is** Object A, because $d_H(b_i,b_j)$ is a point-to-point distance once the
"points" are workloads. Theorem 7.1 of that document therefore *confirms* the notes above rather
than contradicting them.

---

## A.2 Why the two projects land in different places

| | `active_eval` (this project) | `active_model_eval` |
|---|---|---|
| Ground set | $B=\{b_1,\dots,b_N\}$, workloads | $\mathcal P=\{P_1,\dots,P_R\}$, labeled subsets |
| Element = | one point cloud $b_i$ | one subset $P$, items pooled into $\mathcal U_{\mathcal Q}$ |
| "Points" the objective ranges over | **workloads** $b_j$ | **items** $x\in\mathcal X$ |
| Where Hausdorff sits | *below* the ground set (between examples inside two workloads) | *at* the objective level (over items) |
| Consequence | affinity → facility location ✓ | aggregation → $k$-center ✗ |

Same word, different level. Nothing to reconcile beyond noticing which level is meant.

---

## A.3 The Hausdorff-shaped object *does* exist in this project — and §34 handles it correctly

The `active_eval` proof does contain a genuine Object-B quantity. It is the **covering radius**
of §34:

$$\varepsilon(A)=\max_{1\le j\le N}\ \min_{b_i\in A} d_H(b_i,b_j).$$

This is $\max\min$ **at the set-function level**, over workloads — structurally identical to the
metric-swaps §1.1 counterexample. So $-\varepsilon(A)$ is **not submodular**, and greedy carries
no $(1-1/e)$ guarantee for it.

The document already flags the structure explicitly (§34 remark: *"$\varepsilon(A)$ is itself a
$\max\min$, exactly like the Hausdorff distance of Part II — but one level higher, over workloads
rather than over points"*), and — crucially — **it never optimizes $\varepsilon(A)$.**

The architecture is:

$$\underbrace{\max_{|A|\le K} F(A)}_{\text{submodular; greedy gets } 1-1/e}
\ \xrightarrow[\ \text{Lemma \texttt{lem:cover}(b)}\ ]{}\
\underbrace{\varepsilon(A)\le\sqrt{\tau\ln\tfrac{1}{1-\Delta}}}_{\text{derived bound, not optimized}}
\ \xrightarrow[\ \text{Thm \texttt{thm:cover}}\ ]{}\
L\,\varepsilon(A).$$

Optimize the submodular surrogate $F$; **bound** the non-submodular quantity $\varepsilon$ as a
corollary. That is precisely the move the metric-swaps document prescribes when it writes *"only
replacing the outer $\max_x$ by $\sum_x$ fixes it, and that turns Hausdorff into facility
location."* The two documents agree on both the diagnosis and the remedy.

**Naming check.** The metric-swaps takeaway warns: *"the paper should say 'facility location',
not describe it in Hausdorff terms."* This paper already complies — `abstract.tex`, `intro.tex`
and `method.tex` consistently say **"Hausdorff dataset distance"** and **"Hausdorff facility
location"**, which correctly names Hausdorff as the *distance* and facility location as the
*objective*. No edit needed.

---

## A.4 One convention difference, not a disagreement

The notes above say: if you used $S_{ij}=-D_{ij}$ directly, $F$ stays monotone submodular and
what breaks is **normalization** ($F(\varnothing)=-\infty$, $F<0$).

The metric-swaps §7.1 says: with $S=-d\le 0$, **monotonicity** breaks, because
$\varnothing\mapsto 0$ and $\{z\}\mapsto -d<0$.

Both are correct; they use different empty-set conventions:

| Convention | $F(\varnothing)$ | What fails |
|---|---|---|
| $\max_{\varnothing}:=-\infty$ (natural for $\max$) | $-\infty$ | normalization + non-negativity |
| $\max_{\varnothing}:=0$ (metric-swaps §0) | $0$ | monotonicity, **at the empty set only** |

On non-empty sets the function is monotone either way — a max over a larger set cannot shrink,
whatever the sign of the entries. The operational conclusion is identical and is what matters:

> **NWF needs $S\ge 0$; the exponential is what supplies it. Submodularity itself never uses the
> sign.**

---

## A.5 Two genuine defects in `submodularity_under_metric_swaps.md`

Both are minor and repairable; neither changes any conclusion.

### (i) Proposition 7.3 is false as literally stated

> *"Let $\psi$ be any strictly decreasing function and $F$ any set function. If $F$ violates
> monotonicity, so does $\psi\circ F$."*

Counterexample: take $F(\mathcal A)=-\lvert\mathcal A\rvert$, which violates monotonicity
(increasing). With $\psi=\exp(-\cdot)$ we get $\psi(F(\mathcal A))=e^{\lvert\mathcal A\rvert}$,
which **is** monotone. So the statement as written is wrong.

The intended — and correct — content is visible in the proof, which silently assumes $F$ is a
*divergence* to be minimized:

> **Proposition 7.3′.** For strictly decreasing $\psi$, the composite $\psi\circ F$ is monotone
> non-decreasing **if and only if** $F$ is monotone non-increasing. Hence if the divergence $F$
> (e.g. $\mathrm{MMD}^2$) *increases* along some chain $\mathcal A\subseteq\mathcal B$, then
> $\psi\circ F$ *decreases* along that chain and is non-monotone.

Restated this way the argument is airtight and the §7.2 conclusion ($\exp$ cannot repair F3)
stands unchanged.

### (ii) The numerical report in §7.2 contradicts the proposition it illustrates

> *"$-\mathrm{MMD}^2$ had 13 monotonicity and 28 submodularity violations;
> $\exp(-\mathrm{MMD}^2)$ still had **8 and 8**."*

Monotonicity violation counts **must be identical**, not merely similar. A violation of
$-\mathrm{MMD}^2$ is a pair $\mathcal A\subseteq\mathcal B$ with
$\mathrm{MMD}^2(\mathcal A)<\mathrm{MMD}^2(\mathcal B)$; a violation of $\exp(-\mathrm{MMD}^2)$
is $\exp(-\mathrm{MMD}^2(\mathcal A))>\exp(-\mathrm{MMD}^2(\mathcal B))$ — **the same condition**,
since $\exp(-\cdot)$ is a strictly decreasing bijection. So the count should read $13\to 13$.

Reporting $13\to 8$ contradicts the document's own claim that violations survive "one-for-one".
Almost certainly the checker used an absolute tolerance: $\exp$ compresses large
$\mathrm{MMD}^2$ gaps into differences near $0$, so 5 genuine violations fell below the epsilon.
**Fix:** use a relative tolerance, or compare orderings rather than value gaps, and report
$13\to 13$ — which strengthens the section rather than weakening it.

(The submodularity counts $28\to 8$ are fine and need no correction: submodularity is *not*
preserved by monotone transforms, so that number may legitimately move in either direction.)

---

## A.6 Summary

| claim | status |
|---|---|
| Hausdorff *as affinity* $S_{ij}=e^{-d_H^2/\tau}$ → facility location is monotone submodular | **correct** (these notes; and their own Thm 7.1) |
| Hausdorff *as aggregation* $\max_x\min_z d$ is not submodular | **correct** (their §1.1; standard $k$-center fact) |
| the two are in conflict | **no** — different functions, different levels |
| $\varepsilon(A)$ in `active_eval` §34 is Object B, hence not submodular | **true, and the document never optimizes it** |
| the $\exp$ supplies non-negativity, not submodularity | **both documents agree** |
| Prop 7.3 as literally stated | **false**; repair as 7.3′ |
| $13\to 8$ monotonicity counts | **inconsistent**; should be $13\to 13$ |

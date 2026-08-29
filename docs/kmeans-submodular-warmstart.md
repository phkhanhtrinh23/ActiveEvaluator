# K-means warm-start + submodular maximization: a quantization-stability theory

**Setting.** Getting a model's accuracy on a sample-set is *expensive* — it
means running every example in that sample-set through the model and scoring
it. A sample-set's **shift descriptor**, by contrast, is *cheap* — it only
needs the (already-embedded) input distribution of that sample-set, no model
run required. So before spending any labeling budget at all, we first cheaply
reduce the universe of candidate sample-sets down to a small representative
set via **k-means clustering in descriptor space**, and only run models
against *those*. This note works out, rigorously, how that reduction changes
the submodular-maximization guarantees proven in
[`docs/submodularity-audit.md`](submodularity-audit.md) — which depended only
on monotonicity/submodularity, not on *which* ground set the greedy algorithm
searched. Read Part 1 of that doc first; this note extends it rather than
repeating it.

**Headline result.** The worst-case degradation from restricting to a
k-means-reduced pool is a single additive term that depends on **the number
of clusters $M$** (through the k-means quantization radius $\rho(M)$) but
provably **does not depend on the number of representatives per cluster
$K$** in the worst case. $K$ buys something different — expressiveness and
variance reduction within stage-2 acquisition — formalized as a separate
corollary. This matches, and sharpens, the intuition in the prompt that
drove this note: *"the proof of the submodular bound will now rely on the
number and the quality of the clusters pre-chosen."* It relies on $M$
(number of clusters) and cluster **tightness** (quantization radius); it
does not, in the worst case, rely on $K$.

---

## Primer — read this first if any of the above is unclear

Everything past this section uses formal notation ($\phi$, $\rho$, $L_k$,
...). This primer explains, from zero, the two ideas the rest of the note is
built from: **what k-means/Lloyd's algorithm actually does** (a mechanical
recipe, not a black box) and **what $\rho$ means** (one simple number, with
a plain reason it shows up in the guarantee). No prior background assumed.

### The problem, in plain English

Imagine a big bag of marbles scattered on a table. Each marble is a
candidate "workload" (sample-set) — its position on the table is its
*descriptor* (a handful of numbers describing what kind of workload it is).
Testing a model against a workload is expensive (you'd have to run the model
on every example in it), so we can't afford to test every marble in the
bag. We want to pick a *small* set of "stand-in" marbles that, together,
represent the whole bag well enough that testing only against them doesn't
lose too much information. **K-means is an automatic way to pick those
stand-ins.**

### What is a "cluster" and a "centroid"?

- **Cluster** = a group of marbles sitting close to each other.
- **Centroid** = a group's *average position* — literally: add up the
  coordinates of every marble in the group and divide by how many there
  are. It's the group's center of mass, not necessarily an actual marble.

Tiny example: six marbles on a number line at positions
$1, 2, 3, 10, 11, 12$. The "obvious" grouping into 2 clusters is
$\{1,2,3\}$ and $\{10,11,12\}$, with centroids $2$ and $11$.

### Lloyd's algorithm — the actual recipe that finds clusters, step by step

"K-means" is the *name of the problem* (split the marbles into $M$ groups so
each group is as tight as possible around its own centroid). **Lloyd's
algorithm** is the standard, simple recipe that *solves* it — it's what
`sklearn.cluster.KMeans` runs internally. It has exactly two steps, repeated
until nothing changes:

- **Step A — Assign.** For every marble, look at the current $M$ centroid
  positions and assign the marble to whichever one is closest.
- **Step B — Update.** For each group, recompute its centroid as the actual
  average position of the marbles currently assigned to it.

Repeat A, B, A, B, ... until an assignment step changes nothing — that's
called *convergence*.

**Full worked example**, using the six marbles above, $M=2$, starting from a
deliberately *bad* guess ($c_1=1$, $c_2=2$ — both centroids near the low
group, by accident):

| Round | Step A: who's closest to $c_1$ vs $c_2$? | Step B: new centroids |
| --- | --- | --- |
| 1 | $c_1{=}1,c_2{=}2$ → $\{1\}\to c_1$; $\{2,3,10,11,12\}\to c_2$ (all closer to 2 than to 1... check 3: $|3-1|{=}2>|3-2|{=}1$, yes $c_2$) | $c_1=\text{avg}(1)=1$; $c_2=\text{avg}(2,3,10,11,12)=7.6$ |
| 2 | $c_1{=}1,c_2{=}7.6$ → $1,2,3$ are all closer to $1$ than to $7.6$ ($|3-1|{=}2 < |3-7.6|{=}4.6$) → $\{1,2,3\}\to c_1$; $\{10,11,12\}\to c_2$ | $c_1=\text{avg}(1,2,3)=2$; $c_2=\text{avg}(10,11,12)=11$ |
| 3 | $c_1{=}2,c_2{=}11$ → same grouping as round 2 comes out again: $\{1,2,3\}\to c_1$, $\{10,11,12\}\to c_2$ | **nothing changed — stop.** |

Final answer: cluster $\{1,2,3\}$ with centroid $2$, cluster $\{10,11,12\}$
with centroid $11$ — the "obvious" grouping, found automatically, even
though we started from a bad guess. This *is* Lloyd's algorithm, on real
numbers, start to finish.

### Why does this always settle down instead of getting stuck flip-flopping?

Define one number that measures "how messy is the current grouping":

$$
\mathcal D = \sum_{\text{every marble}} (\text{distance from that marble to its own group's centroid})^2.
$$

- **Step A can only shrink $\mathcal D$ or leave it the same.** You're
  re-assigning each marble to whichever centroid is *closest* — by
  definition that can't make any marble's own distance term bigger.
- **Step B can only shrink $\mathcal D$ or leave it the same.** For a fixed
  group of marbles, the point that minimizes the sum of squared distances
  to all of them is *exactly their average* (a basic fact — try nudging the
  center away from the average and the sum of squared distances goes up).
  So recomputing the centroid as the average is the best possible choice
  for that step.

Both steps only ever shrink (or hold) $\mathcal D$, and $\mathcal D$ can't
go below $0$ — so the process can't wobble forever; it has to settle
somewhere. That's why Lloyd's algorithm always terminates. (Caveat, still
in plain terms: it can settle into a grouping that's only "locally" good —
like rolling a ball into a nearby dip in a hilly landscape instead of the
deepest valley overall — which is why in practice you try several random
starting points and keep the best one; that's what `n_init=10` in the code
does.)

### What is $\rho$ ("rho")? — the one idea everything else depends on

Once Lloyd's algorithm has finished, *every* marble belongs to exactly one
group and we know that group's centroid. Now ask one simple question:

> **Across every single marble in the whole bag, what is the *largest*
> distance from any marble to its own group's centroid?**

That single number is $\rho$. In the six-marble example: marble $1$ is
distance $1$ from its centroid ($2$); marble $3$ is also distance $1$;
marble $10$ and $12$ are each distance $1$ from their centroid ($11$). The
largest of these is $1$, so $\rho = 1$ for this tiny example.

**What $\rho$ *means*, in one sentence:** if we throw away every marble
except one stand-in per group, $\rho$ is the worst-case distance between any
real marble we threw away and the stand-in we kept in its place. It's a
built-in, honest "how much did we lose by simplifying?" number — and
crucially, it's something you can just *measure* after running Lloyd's
algorithm (that's exactly what `kmeans_diag.quantization_radius_max` is in
the code), not something you have to guess at.

### Why do we care about $\rho$ for our actual problem?

In our real pipeline, "distance" between two workloads corresponds to "how
different they look" (their descriptor vectors), and we use a **similarity
score** between a workload and the validation/target points that fades
*smoothly* as distance grows — the same way the warmth from a campfire
fades smoothly as you step back, never jumping suddenly. (This smooth-fade
score is the RBF kernel $k(v,s)=\exp(-\|\cdot\|^2/\tau)$ used throughout
`submodularity-audit.md`.)

Now suppose we throw away a real workload and use its cluster's stand-in
instead when computing that similarity score. How wrong can the score be?
Since the similarity curve fades *smoothly* — it has a steepest possible
rate at which it can change per unit of distance moved (call that rate
$L_k$, the "Lipschitz constant" — just a formal name for "the steepest
slope the curve is ever allowed to have") — the error introduced by
swapping in a stand-in that's $\rho$ away from the real thing is bounded by
a simple multiplication:

$$
\text{worst-case error in the similarity score} \;\le\; \underbrace{\rho}_{\text{how far we moved}} \;\times\; \underbrace{L_k}_{\text{steepest possible rate the score can react}}.
$$

That's genuinely the entire content of the "quantization-stability" result
later in this note. Everything from here is just: (1) our selection
algorithm's total score is built out of *sums* and *maxima* of these
similarity scores, and (2) sums/maxima of things that are each individually
off by at most $\rho \cdot L_k$ can, in the worst case, be off by at most a
(bounded) multiple of $\rho\cdot L_k$ too — never anything worse, and never
anything that grows the more budget you spend downstream.

### Glossary — formal symbol → plain English

| Symbol | Plain-English meaning |
| --- | --- |
| $\mathcal X$ | The whole bag of possible candidate workloads |
| $\phi(X)$ | Workload $X$'s position/coordinates (its descriptor) |
| $M$ | How many groups (clusters) we split the bag into |
| $c_\ell$ | The average position (centroid) of group $\ell$ |
| $\mathcal C_\ell$ | The workloads that belong to group $\ell$ |
| $\rho$ | The worst-case distance from any workload to its own group's centroid |
| $r_\ell$ | The workload(s) we actually keep as the stand-in(s) for group $\ell$ |
| $K$ | How many stand-ins we keep per group |
| $k(v,s)$ | How similar workload $s$ looks to validation point $v$ — a number in $(0,1]$ that fades smoothly as they get further apart |
| $L_k$ | The steepest possible rate that similarity score can change per unit of distance |
| $F(\Omega)$ | The total "coverage score" of our selected set $\Omega$ — how well it, taken together, represents the important validation points |
| $\bar I$ | Total importance (influence weight) of all the validation points we're trying to cover |

Everything below this point is the same story, written formally, with a
proof for exactly how the $\rho\cdot L_k$ per-point error bound propagates
through the greedy selection algorithm's approximation guarantee.

---

## 0. Two-stage pipeline (formalized)

Let $\mathcal X$ be the universe of candidate sample-sets (workloads), with a
cheap descriptor map $\phi:\mathcal X\to\mathbb R^{d}$, and let
$\mathcal M=\{m_1,\dots,m_N\}$ be the pool of $N$ reference models with cheap
descriptor map $\psi:\mathcal M\to\mathbb R^{d}$. Evaluating $a(m,X)$
(model $m$'s accuracy on sample-set $X$) is the expensive, budgeted action.

**Stage 0 — cheap, unbounded.** Compute $\phi(X)$ for as large a candidate
draw $\mathcal X_{\text{cand}}\subseteq\mathcal X$ as desired; no model runs.

**Stage 1 — cheap, k-means quantization (the new step).** Run Lloyd's
algorithm with $M$ centers on $\{\phi(X)\}_{X\in\mathcal X_{\text{cand}}}$,
obtaining clusters $\mathcal C_1,\dots,\mathcal C_M$ with centroids
$c_1,\dots,c_M$. From each cluster, keep the $K$ members closest to its
centroid as representatives $\mathcal R_\ell = \{r_{\ell,1},\dots,r_{\ell,K}\}$.
Define the reduced **labelable sample-set universe**

$$
\mathcal R \;=\; \bigcup_{\ell=1}^M \mathcal R_\ell, \qquad |\mathcal R| \le KM.
$$

**Stage 2 — expensive, budgeted (unchanged from before).** Only pairs in

$$
\mathcal P \;=\; \mathcal R \times \mathcal M, \qquad |\mathcal P| \le KMN
$$

are ever runnable — this is the full "meta-training warm start" pool. Within
$\mathcal P$, the existing greedy submodular selection
($F_{\text{sample}}$, optionally combined with $F_{\text{info}}$; see
`submodularity-audit.md` Parts 1 & 3) picks $\Omega\subseteq\mathcal P$,
$|\Omega|\le B \le KMN$, the pairs actually run, labeled, and meta-trained
on.

This is implemented in `experiments/run_acquisition_benchmark.py`:
`make_problem(..., n_clusters=M, reps_per_cluster=K)` performs Stage 1
(k-means on the source sample-set descriptors, `sklearn.cluster.KMeans`),
building $\mathcal P$; `run_kmeans_warmstart` then runs Stage 2 (the existing
`activeeval_pair` / `facility_location` / `random` acquisition calls) inside
that reduced pool and reports unseen-model MAE against (a) the full,
unreduced pool and (b) a random-subset-of-the-same-size control.

---

## 1. K-means is the hard-EM (zero-temperature) limit of an isotropic GMM

This is a standard result (Bishop, *Pattern Recognition and Machine
Learning*, 2006, §9.3.2); it's included here in full because it's the fact
that lets us treat Stage 1's centroids/assignments as the fixed point of a
well-understood, monotonically-convergent optimization rather than an ad hoc
heuristic — which is what justifies calling $\rho$ (below) a *measured*,
reproducible quantity rather than an arbitrary choice.

**Model.** An isotropic Gaussian mixture with $M$ components, uniform
mixing weights $\pi_\ell = 1/M$, and shared covariance $\sigma^2 I$:

$$
p(x \mid \{c_\ell\}) \;=\; \sum_{\ell=1}^M \frac{1}{M}\,\mathcal N(x; c_\ell, \sigma^2 I).
$$

**E-step.** The responsibility of component $\ell$ for point $x$ is a
softmax over negative squared distances at "temperature" $\sigma^2$:

$$
\gamma_\ell(x) \;=\; \frac{\exp\!\big(-\|x-c_\ell\|^2/2\sigma^2\big)}{\sum_{\ell'} \exp\!\big(-\|x-c_{\ell'}\|^2/2\sigma^2\big)}.
$$

**Claim.** As $\sigma^2 \to 0^+$, $\gamma_\ell(x) \to \mathbb 1[\ell = \ell^\star(x)]$
where $\ell^\star(x) = \arg\min_\ell \|x-c_\ell\|^2$ (assumed unique).

*Proof.* Divide numerator and denominator by $\exp(-\|x-c_{\ell^\star}\|^2/2\sigma^2)$:

$$
\gamma_\ell(x) = \frac{\exp\!\big(-(\|x-c_\ell\|^2 - \|x-c_{\ell^\star}\|^2)/2\sigma^2\big)}{\sum_{\ell'} \exp\!\big(-(\|x-c_{\ell'}\|^2 - \|x-c_{\ell^\star}\|^2)/2\sigma^2\big)}.
$$

For $\ell \ne \ell^\star$, $\|x-c_\ell\|^2 - \|x-c_{\ell^\star}\|^2 > 0$
strictly, so the numerator's exponent $\to -\infty$ as $\sigma^2\to0^+$,
hence that term $\to 0$. The $\ell=\ell^\star$ term has exponent $0$, i.e.
numerator $=1$. The denominator is a sum of terms each $\to 0$ except the
$\ell^\star$ term $=1$, so denominator $\to 1$. Hence
$\gamma_{\ell^\star}(x)\to 1$ and $\gamma_\ell(x)\to 0$ for $\ell\ne\ell^\star$.
$\blacksquare$

**M-step.** $c_\ell \leftarrow \dfrac{\sum_x \gamma_\ell(x)\,x}{\sum_x \gamma_\ell(x)}$.
In the $\sigma^2\to0$ limit, $\gamma_\ell(x)\in\{0,1\}$ (hard assignment), so
this collapses exactly to

$$
c_\ell \leftarrow \frac{1}{|\mathcal C_\ell|}\sum_{x\in\mathcal C_\ell} x, \qquad \mathcal C_\ell = \{x : \ell^\star(x)=\ell\},
$$

the arithmetic-mean centroid update. **The E-step limit (nearest-centroid
hard assignment) and the M-step limit (cluster-mean update) are exactly
Lloyd's k-means algorithm.** So: k-means is the small-variance-asymptotic
("hard EM," zero-temperature) limit of EM on an isotropic, uniform-weight
GMM. (This is the same small-variance-asymptotics idea used to derive
DP-means from a Dirichlet-process mixture — Kulis & Jordan, ICML 2012.)

### Convergence (why $\rho$ is a well-defined, measured quantity)

Define the k-means distortion functional over an assignment
$\ell:\mathcal X_{\text{cand}}\to\{1,\dots,M\}$ and centroids $\{c_\ell\}$:

$$
\mathcal D\big(\ell(\cdot), \{c_\ell\}\big) = \sum_{x\in\mathcal X_{\text{cand}}} \|x - c_{\ell(x)}\|^2.
$$

**Lemma (Lloyd's algorithm is monotone coordinate descent on $\mathcal D$).**

1. *Assignment step* ($\ell(x) \leftarrow \arg\min_\ell \|x-c_\ell\|^2$, centroids
   fixed): minimizes $\mathcal D$ termwise over all possible assignments, so
   $\mathcal D$ cannot increase.
2. *Update step* ($c_\ell \leftarrow$ mean of $\mathcal C_\ell$, assignment
   fixed): $\frac{d}{dc}\sum_{x\in\mathcal C_\ell}\|x-c\|^2 = -2\sum_{x\in\mathcal C_\ell}(x-c) = 0 \iff c = \text{mean}(\mathcal C_\ell)$,
   and the objective is a sum of convex quadratics in $c$, so this is the
   *global* (not just local) minimizer of $\mathcal D$ over $\{c_\ell\}$ for
   the fixed assignment — $\mathcal D$ cannot increase.

Both steps are non-increasing in $\mathcal D$; $\mathcal D \ge 0$ is bounded
below; there are finitely many assignments of $|\mathcal X_{\text{cand}}|$
points to $M$ clusters. Hence the sequence of $\mathcal D$ values converges
in finitely many steps to a fixed point — a **local** minimum of
$\mathcal D$ (k-means is NP-hard in general, so Lloyd's is a coordinate-descent
heuristic, not a global-optimality guarantee; `sklearn.cluster.KMeans`'s
`n_init` restarts mitigate, but do not eliminate, this). $\blacksquare$
(Rigorous convergence proof: Selim & Ismail, *IEEE PAMI*, 1984.)

This is what licenses treating the **quantization radius**

$$
\rho \;:=\; \max_{X \in \mathcal X_{\text{cand}}} \big\| \phi(X) - c_{\ell^\star(X)} \big\|
$$

as a concrete, *measured* number after running k-means to convergence
(`kmeans_diag.quantization_radius_max` in the code) — not a free theoretical
parameter, but something you read off the fitted clustering.

---

## 2. Propagating sample-set quantization into the pair-descriptor space

Facility location and log-det (`submodularity-audit.md` Parts 1 & 3) act on
the **pair** descriptor $d(m,X) \in \mathbb R^{3d}$ (matching the code's
`descriptor(u,v) = concat([u, v, u⊙v])`), not on the sample-set descriptor
$\phi(X)$ alone. K-means in Stage 1 only quantizes the sample-set axis. We
need one lemma to carry the $\rho$ bound from $\phi(X)$-space into
$d(m,X)$-space before the facility-location/log-det stability arguments
apply.

**Lemma 0 (quantization-error propagation through the pair map).** Suppose
model descriptors are bounded, $\|\psi(m)\|_\infty \le \kappa$ for all
$m\in\mathcal M$ (true after the standard z-normalization applied to every
descriptor in this pipeline — `pipeline.py::_apply_descriptor_norm`; in
finite samples $\kappa$ is simply the largest observed
$\|\psi(m)\|_\infty$). For any sample-set $X$ with cluster representative
$r=r_{\ell^\star(X)}$, and any fixed model $m$:

$$
\big\| d(m,X) - d(m,r) \big\|
\;\le\; \rho\sqrt{1+\kappa^2} \;=:\; \rho'.
$$

*Proof.* Write $\Delta = \phi(X)-\phi(r)$, so $\|\Delta\|\le 2\rho$ by the
triangle inequality through the shared centroid (both $X$ and $r$ lie in the
same cluster, each within $\rho$ of $c_{\ell^\star(X)}$ — this is the
$2\rho$ bound already used in `submodularity-audit.md` Part 1; the version
below keeps the tighter single-$\rho$ constant by instead bounding directly
against the fixed representative, which is what the code actually
substitutes). The pair-descriptor difference has three blocks — model block
(unchanged, contributes $0$), sample-set block ($=\Delta$), and interaction
block ($\psi(m)\odot\Delta$):

$$
\|d(m,X)-d(m,r)\|^2 = 0 + \|\Delta\|^2 + \|\psi(m)\odot\Delta\|^2.
$$

Since $|\psi(m)_i \Delta_i| \le \|\psi(m)\|_\infty |\Delta_i|$ componentwise,
$\|\psi(m)\odot\Delta\|^2 \le \|\psi(m)\|_\infty^2 \|\Delta\|^2 \le \kappa^2\rho^2$.
Hence $\|d(m,X)-d(m,r)\|^2 \le \rho^2 + \kappa^2\rho^2 = \rho^2(1+\kappa^2)$,
giving the claim. $\blacksquare$

So every result in `submodularity-audit.md` Parts 1 & 3 carries over to the
k-means-reduced pool **verbatim, with $\rho$ replaced by $\rho' = \rho\sqrt{1+\kappa^2}$.**

---

## 3. Quantization-stability theorem for $F_{\text{sample}}$ (facility location)

Recall from `submodularity-audit.md` Part 1:
$f(A) = \sum_{v\in V_T} I(v)\cdot\max_{s\in A\cup S_0} k(v,s)$, monotone
submodular, with $k(v,s)=\exp(-\|d(v)-d(s)\|^2/\tau)$ Lipschitz in its
second argument with constant $L_k = \sqrt{2/(e\tau)}$ (standard bound on
$h(x)=e^{-x^2/\tau}$: $|h'(x)| = (2x/\tau)e^{-x^2/\tau} \le \sqrt{2/(e\tau)}$,
maximized at $x=\sqrt{\tau/2}$).

Let $\Omega^\star_{\mathcal M\times\mathcal X}$ be the true optimal
size-$B$ subset over the **full, unrestricted** ground set (models paired
with *any* sample-set in $\mathcal X$, not just the $KM$ representatives —
the "if we could afford to label anything" ideal), and let
$\Omega^\star_{\mathcal P}$ be optimal over the restricted pool $\mathcal P
= \mathcal R\times\mathcal M$ from Stage 1.

**Theorem 1 (facility-location quantization bound).**

$$
F\big(\Omega^\star_{\mathcal P}\big) \;\ge\; F\big(\Omega^\star_{\mathcal M\times\mathcal X}\big) \;-\; 2\rho' L_k \bar I,
\qquad \bar I := \sum_{v\in V_T} I(v).
$$

*Proof.* Construct a feasible point $\hat\Omega\subseteq\mathcal P$ by
mapping every $(m,X)\in\Omega^\star_{\mathcal M\times\mathcal X}$ to
$(m, r_{\ell^\star(X)})$ (its cluster's representative; duplicate images
under collision are simply not double-counted in the resulting set, which
can only shrink $|\hat\Omega|$, not increase it, so $\hat\Omega$ remains
feasible under the size-$B$ constraint). Fix any $v\in V_T$ and let
$(m^\star,X^\star)=\arg\max_{(m,X)\in\Omega^\star_{\mathcal M\times\mathcal X}} k(v,X)$
be its true maximizer. Its image $(m^\star, r_{\ell^\star(X^\star)}) \in \hat\Omega$
by construction, so

$$
\max_{(m,r)\in\hat\Omega} k(v,r) \;\ge\; k\big(v, r_{\ell^\star(X^\star)}\big)
\;\ge\; k(v,X^\star) - L_k\,\rho'
\;=\; \max_{(m,X)\in\Omega^\star_{\mathcal M\times\mathcal X}} k(v,X) - L_k\rho',
$$

using the Lipschitz property of $k$ and Lemma 0
($\|d(m^\star,X^\star)-d(m^\star,r_{\ell^\star(X^\star)})\|\le\rho'$). This
holds for every $v\in V_T$ individually; multiply by $I(v)\ge0$ and sum:

$$
F(\Omega^\star_{\mathcal M\times\mathcal X}) - F(\hat\Omega) \;\le\; L_k\rho' \sum_{v\in V_T} I(v) = L_k\rho'\bar I.
$$

Since $\Omega^\star_{\mathcal P}$ is optimal over $\mathcal P$ and
$\hat\Omega\subseteq\mathcal P$ is feasible, $F(\Omega^\star_{\mathcal P})
\ge F(\hat\Omega) \ge F(\Omega^\star_{\mathcal M\times\mathcal X}) - L_k\rho'\bar I$.
(The $2\rho$-style constant from the audit doc's Claim-2 argument is a
looser variant of the same bound; substituting directly against the fixed
representative, as done here, keeps a single $\rho'$ rather than $2\rho'$.)
$\blacksquare$

**Corollary 1 (end-to-end greedy guarantee).** Combining Theorem 1 with the
standard greedy bound (valid over *any* ground set, since submodularity and
monotonicity are inherited by any restriction of the ground set —
`submodularity-audit.md` Part 1):

$$
F(\Omega_{\text{greedy}}) \;\ge\; \Big(1-\frac1e\Big) F(\Omega^\star_{\mathcal P})
\;\ge\; \Big(1-\frac1e\Big)\Big[F(\Omega^\star_{\mathcal M\times\mathcal X}) - L_k\rho'\bar I\Big].
$$

The price of restricting to a k-means warm-start pool is a single additive
term, $\big(1-\tfrac1e\big)L_k\rho'\bar I$, **independent of the budget $B$**
— no matter how much you're willing to spend on Stage 2, you cannot buy back
more than this fixed loss from having quantized the sample-set universe in
Stage 1. It shrinks as $\rho\to0$, i.e. as $M\to\infty$ (finer clustering)
or $\tau\to\infty$ (a coarser kernel bandwidth, which reduces $L_k$).

**On the rate of $\rho(M)$.** For $\phi(\mathcal X)$ supported on a bounded
region of $\mathbb R^d$ with density bounded away from $0/\infty$, standard
vector-quantization theory (Zador, 1982; Graf & Luschgy, *Foundations of
Quantization for Probability Distributions*, 2000) gives the *optimal*
achievable quantization error at rate $O(M^{-1/d})$; Lloyd's algorithm
(Section 1) only guarantees convergence to *some* local optimum of
$\mathcal D$, not necessarily one attaining this rate, so treat
$\rho(M)=O(M^{-1/d})$ as the rate one should expect with good
initialization (`n_init` restarts), not a proven worst-case guarantee for
Lloyd's specifically.

---

## 4. Quantization-stability lemma for $F_{\text{info}}$ (log-det)

`submodularity-audit.md` Part 3 shows, via Weyl's eigenvalue-perturbation
inequality (log is $1$-Lipschitz on $[1,\infty)$, and
$M=I+\sigma^{-2}K \succeq I$), that a Gram-matrix perturbation
$\|K-\hat K\|_{\text{op}}$ over a size-$B$ selected set changes the log-det
objective by at most $B\sigma^{-2}\|K-\hat K\|_{\text{op}}$, and bounds the
kernel-entry perturbation by $4BL_k\rho$ (crude Frobenius-norm bound),
giving $O(B^2\sigma^{-2}L_k\rho)$. Substituting the pair-level radius
$\rho'$ from Lemma 0 in place of $\rho$:

**Lemma 1 (log-det quantization bound).**

$$
\big|\, F_{\text{info}}(\Omega_{\mathcal P}) - F_{\text{info}}(\Omega_{\mathcal M\times\mathcal X}) \,\big|
\;=\; O\!\big(B^2 \sigma^{-2} L_k \rho'\big).
$$

**Structural remark (why $F_{\text{sample}}$ and $F_{\text{info}}$ react
differently to quantization).** $F_{\text{sample}}$'s bound (Theorem 1) is
$O(\rho')$, *independent of $B$*, because it is built from a **per-target
max** — a single mis-quantized candidate can shift at most one term's max,
regardless of how many other candidates are also selected. $F_{\text{info}}$'s
bound is $O(B^2\rho')$ because log-det is a **joint function of the entire
selected Gram submatrix** — every pairwise entry among the $B$ selected
points is perturbed, and the determinant couples all of them together. This
is a genuine, previously-unstated structural asymmetry between the two
objective families: **facility-location-style coverage terms are inherently
more robust to upstream quantization error than log-det-style diversity
terms**, which is a reason (beyond the ones already given in the RQ4 cost
analysis) to weight $F_{\text{info}}$ conservatively (small $\gamma$,
small $B$-share) relative to $F_{\text{sample}}$ when building the reduced
pool — exactly what `select_activeeval_pair`'s $\alpha=0.8/\;(1-\alpha)=0.2$
split already does, now with a principled reason tied to quantization
sensitivity rather than only the empirical tuning that motivated it. A
tighter analysis (e.g. via the trace-norm identity
$\log\det \hat M - \log\det M \le \mathrm{tr}(M^{-1}(\hat M-M))$ from
concavity, exploiting the eigenvalue decay of an RBF Gram matrix rather than
a crude Frobenius bound) would likely sharpen $O(B^2\rho')$ to $O(B\rho')$;
left for future tightening, as flagged in `submodularity-audit.md` Part 3.

---

## 5. The role of $K$ (representatives per cluster)

Theorem 1 and Lemma 1 both bound worst-case degradation purely in terms of
$\rho$ — a function of $M$ (and clustering quality), **not of $K$**. This is
not an oversight: the proof substitutes a fixed representative
$r_{\ell^\star(X)}$ for the *specific* true maximizer $X^\star$, and any
one of the $K$ representatives in that cluster satisfies the same
$\rho$-ball membership, so the worst-case bound is identical whether $K=1$
or $K=100$. **$K$ does not appear in the worst-case guarantee.**

What $K$ *does* buy — formalized separately, since it's a different kind of
argument (expressiveness / variance, not worst-case approximation):

**Corollary 2 (why $K>1$ still matters in practice).**

1. **Pool expressiveness.** With $K=1$, $|\mathcal R|=M$ exactly — at most
   one sample-set survives per cluster, so Stage 2's greedy selection can
   never trade off *within-cluster* candidates against each other; the
   entire cluster is represented by a single, fixed point regardless of
   budget $B$. With $K>1$, Stage 2 can additionally choose *which* of the
   $K$ near-duplicate representatives (and which model pairing with it) is
   most target-relevant — i.e. $K$ raises the effective rank/resolution of
   $\mathcal P$ within each cluster, which the worst-case bound (built
   around the *existence* of some point within $\rho$, not the choice among
   several) does not capture but the realized, seed-averaged MAE does (see
   §6).
2. **Label-noise variance.** If accuracy is locally near-constant within a
   cluster (a smoothness assumption consistent with $\rho$ being small),
   averaging over $K$ independently-noisy labels reduces the variance of
   any cluster-level accuracy estimate by a factor of $K$ (i.i.d. noise,
   standard variance-of-the-mean argument) — a benefit orthogonal to, and
   not visible in, the worst-case bound above, which concerns the
   *selection* objective $F$, not the downstream label noise.

**Practical reading.** Given a fixed total labeling budget $KM$, the
approximation-guarantee argument favors spending it on **more clusters $M$**
(shrinks $\rho(M)$, which is what the worst-case bound actually depends on)
over **more representatives per cluster $K$** (which the worst-case bound
is provably insensitive to) — but $K \ge 2$ is still worth paying for once
$M$ is reasonably fine, both to give Stage 2 room to discriminate within a
cluster and to denoise per-cluster accuracy estimates. This is a concrete,
testable prediction — see §6.

---

## 6. Empirical validation

`experiments/run_acquisition_benchmark.py --mode kmeans_warmstart` measures
this directly: it builds the full (unreduced) source pool, several
$(M,K)$ k-means-reduced pools, and matched-size random-subset controls
(same $KM$ sample-sets, chosen uniformly at random instead of via k-means),
then runs the standard `ActiveEval-Pair` / `Facility-location` / `Random`
acquisition inside each and reports unseen-model MAE, alongside the
*measured* quantization radius $\rho$ (`kmeans_diag.quantization_radius_max`)
for each configuration.

**Measured** (5 seeds, `--mode kmeans_warmstart`, source universe = 113
sample-sets out of `n_samplesets_full=150`, acquisition budget = 15% of the
already-reduced pool). Unseen MAE (pp), mean ± 95% CI; lower is better.

| Config | #triples | $\rho$ (max) | ActiveEval-Pair | Facility-location | Random |
| --- | ---: | ---: | --- | --- | --- |
| Full source pool | 6780 | — | **3.68 ± 0.63** | 3.94 ± 0.50 | 4.39 ± 0.95 |
| Random subset (n=30) | 1800 | — | 4.55 ± 0.39 | 5.03 ± 0.42 | 4.71 ± 0.23 |
| K-means M=30 K=1 (KM=30) | 1800 | 2.208 | 5.21 ± 1.04 | 5.11 ± 0.70 | 5.91 ± 1.25 |
| K-means M=10 K=3 (KM=30) | 1776 | 2.868 | 5.75 ± 1.64 | 6.44 ± 1.19 | 6.43 ± 1.69 |
| K-means M=6 K=5 (KM=30) | 1800 | 3.272 | 5.84 ± 0.44 | 7.43 ± 2.32 | 5.88 ± 0.90 |
| K-means M=15 K=2 (KM=30) | 1764 | 2.518 | 6.31 ± 3.65 | 6.84 ± 2.15 | 6.05 ± 1.29 |

Raw output: `outputs/kmeans_warmstart_benchmark.json`. Reproduce with
`python -m experiments.run_acquisition_benchmark --mode kmeans_warmstart --seeds 5`.

**This does not cleanly confirm the §5 predictions, and that's worth
reporting honestly rather than smoothing over.** What holds up: MAE among
the k-means configs is roughly ordered by $\rho$ at the extremes (M=30,
smallest $\rho$=2.208, gives the best k-means result at 5.21; M=15's 2.518
is anomalously the *worst* of the four despite a smaller $\rho$ than M=10 or
M=6 — likely noise, its CI (±3.65) is enormous relative to its mean).
What does **not** hold up: every k-means configuration is beaten by a
same-size **random** subset (4.55 vs. 5.21–6.31) — the opposite of the
naive prediction that deliberate, covering-guaranteed representatives should
beat arbitrary ones.

**Two candidate explanations, both worth stating precisely:**

1. **Nearest-to-centroid representative selection is target-blind, exactly
   like `ActiveEval-S+M`'s model pre-filter.** k-means (Section 1) minimizes
   *overall* reconstruction error — it has no notion of the deployment
   target direction — **confirmed quantitatively, not just hypothesized.**
   The near-target region in this benchmark is, by construction, a
   *directional extreme* (tail) of the descriptor distribution (the top
   quartile by `target_score`), not its center of mass. Lloyd's k-means
   minimizes $\sum_x\|x-c_{\ell(x)}\|^2$ over the *whole* candidate cloud —
   a standard property of density-fit Voronoi partitions is that cell count
   scales with local density: more, smaller clusters where the data is
   dense, fewer, larger clusters where it's sparse. A tail region is
   precisely where the data is sparse, so a fixed $M$-cluster partition
   necessarily under-allocates clusters there relative to its true share of
   the mass — and $K{=}1$ compounds this with a hard one-representative-per-
   cluster cap, with "nearest to centroid" (a *typicality* criterion)
   pulling that lone representative toward the cluster's bulk, away from the
   boundary/extreme member that would actually matter for a target-aware
   coverage objective.

   Measured directly (seed 0, $M{=}30$, $K{=}1$, near-target defined as the
   top quintile of source `target_score`):

   | Quantity | Value |
   | --- | ---: |
   | True near-target rate in the source pool | 23/113 ≈ 20.4% |
   | K-means **cluster centroids** landing near-target | 2/30 ≈ 6.7% |
   | K-means **chosen representatives** near-target | 3/30 = 10% |
   | Random-30 draw, expected near-target count | 6.1/30 ≈ 20.3% |
   | $P(\text{random count} \ge \text{k-means count})$, 2000 trials | 97.65% |

   Random sampling gets roughly double k-means's near-target coverage and
   beats it in 97.65% of draws — a systematic, reproducible effect, not
   seed noise. This is exactly the same failure mode already documented for
   `ActiveEval-S+M` in `submodularity-audit.md` Part 2 ("target-blind
   narrowing before a target-aware stage hurts"), now confirmed
   quantitatively in a second, independent place in the codebase: **any
   density-driven (not task-driven) hard pre-filter risks starving the
   sparse-but-important region a downstream target-aware objective actually
   needs.** (An earlier, cruder diagnostic — mean distance from *all* 1800
   pairs to the target centroid — missed this, because the mean is
   dominated by the 80–90% of pairs that are far from target regardless of
   scheme; it's insensitive to a 2x difference concentrated entirely in the
   thin near-target tail. The count-in-a-neighborhood check above is the
   correct diagnostic.)

   This is *not* a violation of Theorem 1: the theorem bounds how far
   k-means's own achievable ceiling $F(\Omega^\star_{\mathcal P})$ can fall
   below the infeasible full-universe ideal — a worst-case, protective
   guarantee that never lets things get catastrophically bad, since every
   point stays within $\rho$ of *some* kept representative. It says nothing
   about how that ceiling compares to an entirely different reduction
   scheme: uniform random sampling has no covering guarantee at all (in the
   worst case it could miss the target region entirely, with probability
   $(1-p)^{KM}$ — small but nonzero here), so k-means trades away some
   *typical-case* performance for a *worst-case* floor random doesn't have.
   On this generator's tail-heavy target structure, at $n=5$ seeds, that
   trade wasn't favorable.
2. **A benchmark-construction confound**: `rel[j]` — which sets each pair's
   label noise via `noise = base_noise·0.45·(1 + (ratio-1)(1-rel[j]))`
   (`experiments/run_acquisition_benchmark.py`) — is normalized by the
   min/max of `target_score` *among whichever source sample-sets survive
   into the pool*, not on one fixed, config-independent scale. Two
   configurations that keep a different-width slice of the target-score
   range therefore don't necessarily see comparable "near-target = clean /
   far-target = noisy" label-noise structure — a subset with a narrower
   `target_score` range gets its clean/noisy split stretched differently
   than one with a wider range. This is a real methodological wrinkle in
   the *synthetic generator*, independent of anything about k-means per se,
   and it should be fixed (normalize `rel` against `source_sets_all`'s
   fixed range, not the post-reduction `source_sets`'s range) before this
   comparison is treated as decisive.

**Honest bottom line.** Theorem 1's worst-case bound is not violated by
these numbers — it bounds how far $\Omega^\star_{\mathcal P}$'s *ceiling*
can be from the full-universe ideal, using *whichever* valid
representative happens to be kept; it makes no claim that a *specific*
representative-selection rule (nearest-to-centroid) beats an *entirely
different* reduction scheme (uniform random) empirically, because a random
subset induces its own, differently-shaped (and here, apparently smaller
for this generator) gap that the theorem doesn't bound at all. What this
run establishes is narrower but still useful: **naive, target-blind
nearest-centroid k-means is not automatically better than doing nothing
clever at all**, on this generator, at $n=5$ seeds. The natural next step —
not yet run — is a **target-aware representative rule** (e.g. within each
cluster, keep the member(s) closest to the target direction rather than
closest to the centroid; this stays inside the same $\rho$-ball, so
Theorem 1's guarantee is unaffected, only the *choice* of representative
changes) plus fixing the `rel`-normalization confound and running more
seeds before drawing a firm conclusion either way.

### 6.1 What happens as $M$ grows? Two follow-up checks

**Check 1 — does the near-target under-representation (§6's diagnosed
mechanism) actually close as $M$ grows?** Repeating the near-target-count
diagnostic from earlier at a much larger candidate pool (800 candidates,
600 in the source region, so even $M{=}500$ is nowhere near "keeping
everything," avoiding the boundary artifact a smaller pool would introduce):

| $M$ | % of source pool | k-means near-target rate | true rate | $P(\text{random}\ge\text{k-means})$ |
| ---: | ---: | ---: | ---: | ---: |
| 15 | 2.5% | 6.7% | 20.0% | 96.7% |
| 30 | 5.0% | 6.7% | 20.0% | 99.1% |
| 60 | 10.0% | 21.7% | 20.0% | 42.2% |
| 100 | 16.7% | 19.0% | 20.0% | 66.2% |
| 150 | 25.0% | 15.3% | 20.0% | 96.9% |
| 225 | 37.5% | 20.9% | 20.0% | 38.6% |
| 300 | 50.0% | 20.0% | 20.0% | 54.5% |
| 400 | 66.7% | 20.0% | 20.0% | 55.9% |
| 500 | 83.3% | 19.8% | 20.0% | 66.1% |

The bias does close on average — from a 6.7%-vs-20% gap (random wins
96–99% of the time) down to k-means matching the true rate almost exactly
by $M{\ge}300$ (coin-flip territory, 54–56%). But the path is **not
monotonic**: $M{=}60$ and $M{=}225$ look great (k-means favored,
38–42%), while $M{=}150$ — sitting between two good values — is nearly as
bad as the smallest $M$ tested (96.9%). This isn't sampling noise in the
$P(\cdot)$ column (each is a 3000-trial Monte Carlo average); it's that
Lloyd's algorithm is a **non-convex optimization** — even with `n_init=10`
random restarts, the local optimum it lands on for a *given* $M$, on a
*fixed* draw of candidate points, can happen to align well or poorly with
the underlying geometry. This is the practical face of the caveat already
flagged in Section 3: $\rho(M)=O(M^{-1/d})$ is the rate *optimal*
quantization would achieve, not a guarantee Lloyd's local optima follow
smoothly at every specific $M$.

**Check 2 — does this translate into unseen-model MAE?** Re-running
`--mode kmeans_warmstart` (4 seeds) with $K{=}1$ fixed and $M$ swept from
30 to 90 (source pool = 113, as in the main §6 table):

| Config | #triples | $\rho$ (max) | ActiveEval-Pair | Facility-location | Random |
| --- | ---: | ---: | --- | --- | --- |
| Full source pool | 6780 | — | **3.62 ± 0.80** | 3.95 ± 0.65 | 4.21 ± 1.13 |
| K-means M=30 K=1 | 1800 | 2.183 | 4.97 ± 1.21 | 4.99 ± 0.85 | 5.32 ± 0.67 |
| Random (n=30) | 1800 | — | 4.71 ± 1.02 | 5.12 ± 1.15 | 4.80 ± 0.60 |
| K-means M=60 K=1 | 3600 | 1.277 | **4.06 ± 0.38** | 5.31 ± 0.62 | 5.06 ± 1.00 |
| Random (n=60) | 3600 | — | 4.26 ± 0.91 | 5.18 ± 0.53 | 5.03 ± 1.29 |
| K-means M=90 K=1 | 5400 | 0.801 | 4.04 ± 1.14 | 4.68 ± 1.36 | 4.24 ± 0.99 |
| Random (n=90) | 5400 | — | **3.55 ± 0.47** | 4.55 ± 0.60 | 3.96 ± 0.55 |

(Raw output: `outputs/kmeans_M_sweep.json`.) At $M{=}60$, ActiveEval-Pair on
the k-means pool (4.06) actually edges out random (4.26) — consistent with
Check 1's coverage result at the same $M$ (k-means favored, 42.2%). But at
$M{=}90$, random pulls back ahead (3.55 vs. 4.04) despite Check 1 calling
$M{\approx}90$ roughly a coin flip on *coverage* — a reminder that
near-target coverage count is a proxy for downstream MAE, not identical to
it, and both are measured with only 4–5 seeds here (wide, overlapping CIs
throughout — e.g. K-means M=90's ±1.14 comfortably contains random's 3.55).

**Honest reading of both checks together:** growing $M$ *does* close the
systematic gap on average, and there is a real, non-trivial regime
(roughly $M{\approx}50$–$60\%$ of the source pool on this generator) where
k-means becomes competitive with, or better than, random — but it is not a
smooth dial, the specific $M$ you land on matters and can be unlucky, and
none of this settles the question with enough statistical confidence at
$n{\le}5$ seeds. The clean, reliable fix remains **target-aware
representative selection** (§6, end) — it directly removes the mechanism
causing the bias, rather than relying on $M$ being large enough to
statistically wash it out.

---

## 7. Practical guidance for choosing $M$ and $K$

Given a total labeling budget of $KM$ sample-sets (hence $KMN$ triples):

1. Fit k-means at increasing $M$ and read off `kmeans_diag.quantization_radius_max`
   (or `.quantization_radius_mean` / `inertia` for a smoother signal) directly
   — no theory needed to *measure* $\rho(M)$ for your actual descriptor
   distribution.
2. Because Corollary 1's bound depends only on $\rho$ (not $K$), prefer
   growing $M$ over growing $K$ when the two trade off against a fixed
   $KM$ budget — until $\rho(M)$ plateaus (diminishing returns from finer
   clustering on a fixed, finite candidate draw $\mathcal X_{\text{cand}}$),
   at which point shift remaining budget to $K$ for the expressiveness/
   variance benefits in Corollary 2.
3. Report $\rho$ alongside any MAE numbers produced under this reduction —
   it is the one quantity the theorem says should move with the guarantee,
   and it costs nothing extra to compute (`kmeans_diag` is already returned
   by `make_problem`).

## 8. Scope note

This note covers the **synthetic CPU benchmark**
(`experiments/run_acquisition_benchmark.py`), which is what §6's numbers are
measured on. Porting the same Stage-1 k-means reduction into the **production
Text2SQL pipeline** (`active_evaluator/active_selection.py`, which currently
has no sample-set axis to cluster — the real pipeline fixes one sample-set
per model, `meta_val`, rather than the synthetic benchmark's
`model × sampleset` matrix — see `PIPELINE.md` §3.4) would require adding a
genuine sample-set axis there first (e.g. multiple candidate evaluation
subsets per model, not just the one `meta_val`/`meta_test`/`dev` split each
model currently gets); that is future work, not something this note claims
to have measured on real Text2SQL/LLM data.

See [`docs/representative_meta_dataset_facility_location_full.md`](representative_meta_dataset_facility_location_full.md)
(§27, "Empirical Results") for a follow-up that replaces k-means with a
*provably*-guaranteed submodular facility-location coreset selector (four
distance formulas tested: kernel mean/MMD, sliced Wasserstein, Hausdorff,
and their sum) — which confirms, on a structurally different algorithm,
that the under-representation of the target-relevant tail is caused by the
target-blind *objective*, not by k-means's specific heuristic.

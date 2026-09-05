# Do we have a theory for choosing the effective sample size?

**Analysis of:** Morita, Thall & Müller (2008), *Determining the Effective Sample Size of a
Parametric Prior*, **Biometrics** 64(2):595–602.
**Local copy:** [`MTM08_effective_sample_size_prior.pdf`](MTM08_effective_sample_size_prior.pdf)
· extracted text: [`MTM08.txt`](MTM08.txt)
**Bearing on:** `appendix_formulation.tex` §`app:beta-priors`, §`app:calibration-variance`;
`method.tex` §`sec:method-prior`

---

## 1. Short answer: no

We cite Morita–Thall–Müller (MTM08) **exactly once**, at `appendix_formulation.tex:178`, as a
passing appeal to authority:

> *"This is the usual effective-sample-size view of Bayesian priors: the prior contributes
> pseudo-observations to the posterior update~\citep{morita2008determining}."*

We never state a definition of ESS, never verify that our prior satisfies one, and never
justify the discount factor $\operatorname{clip}(\bar S_{\mathcal Q^\star},0,1)$. That is a
citation, not a theory. A reviewer who knows this literature will spot the gap.

The good news: MTM08 hands us the theory almost for free, because **our construction is
literally their Example 6.**

---

## 2. What MTM08 actually provides

### Definition 1 (their §2)

The ESS of a prior $p(\theta \mid \tilde\theta)$ **with respect to a likelihood**
$f_m(Y_m \mid \theta)$ is the value $m$ minimizing

$$
\delta(m, \bar\theta, p, q_0) \;=\; D_{p,+}(\bar\theta) \;-\; D_{q,+}(m, \bar\theta),
$$

where the curvature terms are

$$
D_{p,j}(\theta) = -\frac{\partial^2 \log p(\theta \mid \tilde\theta)}{\partial \theta_j^2},
\qquad
D_{q,j}(m,\theta,Y_m) = -\frac{\partial^2 \log q_m(\theta \mid \tilde\theta_0, Y_m)}{\partial \theta_j^2},
$$

summed over coordinates ($D_{p,+}=\sum_j D_{p,j}$, and $D_{q,+}$ additionally averaged over the
marginal $f_m$). Here $q_0$ is an **$\varepsilon$-information prior** — same mean and
correlations as $p$, but inflated variances — and $q_m$ is its posterior after $m$
observations.

In words: *find the sample size $m$ such that updating a deliberately vague prior with $m$
observations produces a posterior as sharply peaked as your actual prior.*

They validate that this reproduces the textbook answers. **Example 1:**
$\theta \sim \mathrm{Be}(3,7) \Rightarrow \mathrm{ESS} = 10 = a+b$.

---

## 3. Three results that bear directly on our paper

### 3.1 Our prior *is* their power-prior example — the main find

**MTM08 Example 6.** For a power prior
$p(\theta \mid D_0, a_0) \propto L(\theta \mid D_0)^{a_0}\, p_0(\theta \mid c_0)$
(Ibrahim & Chen, 2000), the effective sample size is

$$
\boxed{\;\mathrm{ESS}(p) \;=\; a_0 \cdot \mathrm{ESS}\{L(\theta \mid D_0)\} \;+\; \mathrm{ESS}\{p_0\}\;}
$$

Their worked case: $\mathrm{Be}(1,1)$ initial prior, $D_0$ = 3 successes in 10 trials, giving
$\mathrm{ESS} = 10\,a_0 + 2$.

**The mapping to PoolEvaluator is exact:**

| MTM08 | PoolEvaluator |
|---|---|
| historical data $D_0$ | calibration subset $\mathcal S^\star$: $M_j$ correct on $\pi_j\lvert\mathcal S^\star\rvert$ of $\lvert\mathcal S^\star\rvert$ items |
| $\mathrm{ESS}\{L(\theta \mid D_0)\}$ | $\lvert\mathcal S^\star\rvert$ |
| power exponent $a_0$ | $\operatorname{clip}(\bar S_{\mathcal Q^\star},0,1)$ — **our target-match discount** |
| initial prior $p_0 = \mathrm{Be}(1,1)$, ESS $=2$ | the "$1+$" in $\operatorname{Beta}(1+s_j\pi_j,\;1+s_j(1-\pi_j))$ |

Multiplying out with $a_0 = \operatorname{clip}(\bar S_{\mathcal Q^\star},0,1)$:

$$
L(\alpha_j \mid D_0)^{a_0}\, p_0(\alpha_j)
\;=\;
\alpha_j^{\,a_0 \pi_j \lvert\mathcal S^\star\rvert}\,(1-\alpha_j)^{\,a_0 (1-\pi_j) \lvert\mathcal S^\star\rvert}
\;=\;
\alpha_j^{\,s_j \pi_j}\,(1-\alpha_j)^{\,s_j(1-\pi_j)},
$$

which is $\operatorname{Beta}\!\big(1+s_j\pi_j,\; 1+s_j(1-\pi_j)\big)$ with

$$
s_j \;=\; a_0\,\lvert\mathcal S^\star\rvert
\;=\; \lvert\mathcal S^\star\rvert \operatorname{clip}(\bar S_{\mathcal Q^\star},0,1),
$$

**verbatim our `eq:appendix-prior-weight-discounted`.** Therefore

$$
\mathrm{ESS}\Big(\operatorname{Beta}\big(1+s_j\pi_j,\,1+s_j(1-\pi_j)\big)\Big) \;=\; s_j + 2 .
$$

**Two payoffs.**

1. The target-match discount stops being a heuristic and becomes the **Ibrahim–Chen
   power-prior exponent**, with a closed-form ESS. This is real theory, not an analogy.
2. Our stated ESS is off by exactly $2$ — the uniform baseline. Harmless, but far better
   stated by us than found by a reviewer.

---

### 3.2 ESS is a property of the (prior, likelihood) **pair** — this justifies the $J$ in $s_\beta$

Direct quote (their §2, after Definition 1):

> *"An essential point is that the ESS is defined as a property of a prior and likelihood
> pair, so that, for example, a given prior might have two different ESS values in the
> context of two different likelihoods."*

This settles the question I flagged when we set $s_\beta$:

| parameter | its likelihood counts | prior strength |
|---|---|---|
| $\alpha_j$ | $N$ items | $s_j = a_0\lvert\mathcal S^\star\rvert$ |
| $\beta$ | $NJ$ model–item pairs | $s_\beta = a_0 \, J \lvert\mathcal S^\star\rvert$ |

Same Beta family, different likelihood $\Rightarrow$ legitimately different ESS. The factor
$J$ is exactly the pair-dependence MTM08 predicts — **not** a fudge factor.

> **Caveat to state honestly in the paper.** MTM08 assumes an i.i.d. sample,
> $f_m(Y_m\mid\theta) = \prod_{i=1}^m f(Y_i \mid \theta)$. Our $NJ$ agreement observations are
> **not** independent: the $J$ observations on item $i$ share the single pseudo-label
> $\hat y_i$. So $J\lvert\mathcal S^\star\rvert$ matches the *nominal* pair count while the
> true information content is lower — we err toward **over**-weighting the prior. The
> rigorous fix is their Algorithm 1 (numerical, Monte Carlo) run against the actual
> agreement likelihood.

---

### 3.3 Their "Case 3" is our exact setting

MTM08 distinguish three cases. **Case 3** is: $d \ge 2$ with the prior factorizing as
$p(\theta\mid\tilde\theta) = \prod_{k=1}^{K} p_k(\theta_k \mid \tilde\theta_k, \theta_1,\dots,\theta_{k-1})$,
for which *a vector of $K$ ESS values, one per subvector, is the meaningful object.*

Ours is

$$
\theta = (\alpha_1,\dots,\alpha_J,\beta),
\qquad
p(\theta) = \Big[\textstyle\prod_{j=1}^{J} p(\alpha_j)\Big]\, p(\beta),
$$

so our $(s_1,\dots,s_J,s_\beta)$ **is** that vector. We simply never say so.

Their phase-I trial example is instructive: overall $m = 1.5$ alongside subvector values
$m_1 = 547.3$, $m_2 = 756.8$, $m_3 = 0.01$. Orders-of-magnitude spread across subvectors is
expected and meaningful, not a bug — the analogue for us is that a per-model $s_j$ can be
large while the pool-level prior remains weak.

---

## 4. A diagnostic their Guidelines expose that we are missing

MTM08 §4, Guidelines 5 and 6: calibrate the prior so that **the data, not the prior,
dominates**; and when reviewing a Bayesian design, report the ESS so the reader can judge how
much the prior drives the conclusions.

Our MAP update is a convex combination:

$$
\alpha_j^{(t+1)}
=
\underbrace{\frac{N}{N+s_j}}_{\text{target evidence}}\;\hat\mu_j
\;+\;
\underbrace{\frac{s_j}{N+s_j}}_{\text{calibration prior}}\;\pi_j .
$$

If $\lvert\mathcal S^\star\rvert \approx N$ and $a_0 \approx 1$, then $s_j \approx N$ and the
prior takes **~50 % of the weight** — PoolEvaluator would be half-reporting calibration
accuracy rather than evaluating on the target.

**We report this ratio nowhere.** It belongs in the experimental setup. Their Guideline 4
(ESS as the index in a sensitivity analysis) also makes an $a_0$-sweep a natural ablation.

---

## 5. Verdict and proposed additions

**Verdict.** We have the *right* construction and cite the *right* paper, but no theory —
only an assertion. Everything needed is in MTM08 Definition 1 + Example 6.

| # | Where | What to add | Touches results? |
|---|---|---|---|
| 1 | `app:beta-priors` | State Definition 1; note $a+b = 2+s$, hence $\mathrm{ESS} = s+2$ | no |
| 2 | `app:calibration-variance` | Derive $s_j$ as the power-prior ESS via Ex. 6; identify $\operatorname{clip}(\bar S_{\mathcal Q^\star},0,1)$ as $a_0$ | no |
| 3 | `app:calibration-variance` | Two sentences on pair-dependence justifying $J$ in $s_\beta$ + the non-i.i.d. caveat | no |
| 4 | `experiments.tex` | Report $s_j/(N+s_j)$; optional $a_0$ sweep as sensitivity ablation | yes (new number) |

Item 2 requires adding **`ibrahim2000power`** (Ibrahim & Chen, *Power Prior Distributions for
Regression Models*, Statistical Science 15(1):46–60, 2000) to `ref.bib`.

Items 1–3 are appendix-only and change no reported result.

---

## 6. Unrelated note

The last sentence of `exp_benchmark.tex` currently reads:

> *"…which is the failure mode the judge in `\autoref{sec:method-ambiguity}`."*

It lost its verb (previously "…is designed to absorb"). Looks like an accidental truncation
during editing rather than an intended trim.

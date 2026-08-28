# Ideas from the Handbook of Bayesian Deep Learning

Source: `docs/handbook_of_bdl.pdf` (818 pages, 56 chapters). Reading scope,
stated honestly so nothing below is overclaimed:

- **Full table of contents read** (all 56 chapter/section titles).
- **Deep-read in full**: Chapter 53 "Active Learning and Experimental
  Design" (pp. 603–625) — by far the most relevant chapter, essentially
  the entire thing. Chapter 39 "Last-layer inference" (pp. 427–432).
  Chapter 44 "Confidence-based selective classification" (§44.1–44.2.2,
  pp. 489–492).
- **TOC-level only, not verified in depth** — flagged explicitly wherever
  used below: Chapter 27 "Meta-learning deep kernel Gaussian processes,"
  Chapter 22 "PAC-Bayesian deep neural network ensembles," Chapter 55
  "Credal BDL and Imprecise Probability in BDL."

Every idea below cites the specific chapter/equation it comes from and ties
to a specific file/function in this repo — no generic advice.

---

## Group A — directly fixes/reframes the open gaps already identified in this repo

### A1. Formalize the selection framework via Theorem 53.1 (proper scoring rules ⟺ Bayes-optimal experimental design)

**Source:** §53.2.2, Theorem 53.1 (p. 610); Table 53.1 (p. 612).

Every Bayes-optimal experimental-design policy is the minimizer of an
expected proper scoring rule on the posterior, and vice versa — any
integrable loss induces a corresponding scoring rule. This is exactly the
theoretical anchor missing for `method.tex`'s proposed
$F = \alpha F_{\text{sample}} + \beta F_{\text{model}} + \gamma F_{\text{info}}$,
which `docs/submodularity-audit.md` documents as implemented only as three
separate mechanisms, never jointly. Table 53.1 gives named
expected-uncertainty-reduction (EUR) objectives per scoring rule:
$F_{\text{sample}}$/facility-location maps to an (unweighted)
Shannon-entropy-style expected-information-gain surrogate;
$F_{\text{info}}$/log-det maps to the squared-distance-to-mean →
trace-of-covariance → expected-variance-reduction row — the same identity
already half-derived in `docs/kmeans-submodular-warmstart.md` Part 4, now
with a name and citation. Writing this correspondence out explicitly turns
"three separate heuristics" into "three different, named EUR objectives
under one decision-theoretic umbrella."

**Effort:** moderate — framing/writing, no new code.

### A2. Replace V2's expensive K-step-SGD retraining oracle with a Fisher-information / EVR analytic surrogate

**Source:** §53.4.2 "Variance-based methods" (p. 621) — Cohn's EVR/EFV,
Eq. 53.40–53.41: $\mathrm{EFV}(\xi) = \mathbb E_{p(x)p(d;\xi)}\big[\mathbb V_{p(y;x,d)}[y]\big]$
— expected *future* predictive variance after hypothetically adding
candidate $\xi$. Also §53.3.2, Fisher-information/D-optimality
(Eq. 53.27–53.28).

`active_evaluator/active_selection.py::direct_greedy_validation` (V2)
currently *empirically* estimates something in this spirit by literally
running K SGD steps per candidate and measuring realized validation MSE —
expensive (`--selection-max-candidates-evaluated` exists purely to bound
this) and, as proven in `docs/submodularity-audit.md`, non-submodular with
no guarantee. With a Bayesian last-layer posterior (A3), $\mathbb
V_{p(y;x,d)}[y]$ has a **closed form** (Eq. 39.14–39.15) — no retraining
loop needed. V2 becomes cheap (one linear-algebra formula per candidate
instead of K-step retraining × FASS-shortlist) *and* properly Bayesian,
while V1 keeps its submodular guarantee as the fast default.

**Effort:** substantial (depends on A3 first), but high payoff — directly
fixes V2's most-criticized property (no guarantee, expensive).

### A3. Give the ActiveEvaluator MLP a last-layer Bayesian posterior

**Source:** Chapter 39 "Last-layer inference," read in full (pp. 427–432).

Last-layer inference = ordinary Bayesian linear regression on the *frozen
penultimate-layer features* $\phi_\theta(x)$ (Eq. 39.2, Fig. 39.1), with a
closed-form Gaussian posterior:

$$
\mu_N = \Sigma_N\big(\Sigma_0^{-1}\mu_0 + \sigma_\varepsilon^{-2}\Phi^\top Y\big), \qquad
\Sigma_N = \big(\Sigma_0^{-1} + \sigma_\varepsilon^{-2}\Phi^\top\Phi\big)^{-1}
$$

(Eq. 39.14–39.15; a Gibbs sampler is also given, Algorithm 31 p. 432, but
the conjugate case needs no sampling).

**This maps almost exactly onto code that already exists.**
`active_evaluator/active_selection.py::compute_embeddings`/`_penultimate`
already extracts $\phi_\theta(x) = \mathrm{ReLU}(fc2(\mathrm{ReLU}(fc1(x))))$
for every candidate — that's literally $\Phi$ in Eq. 39.5. Fitting a
Gaussian posterior over `fc3`'s weights on top of this (instead of only
using it as a kernel-embedding space) gives, essentially for free:

- closed-form predictive variance per candidate (feeds A2),
- a proper Bayesian information-gain quantity to replace/complement the
  current gradient-norm influence weight (feeds A4/B2),
- confidence intervals on `predicted_accuracy` — currently a bare point
  estimate everywhere (`meta_learning.py::evaluate`/`evaluate_transfer`
  return only `predicted_accuracy` + `mae`, no uncertainty).

**Effort:** moderate — one new module wrapping the existing `fc3` layer, no
architecture change.

### A4. Replace the gradient-norm influence weight with an EPIG-style predictive information-gain score

**Source:** §53.4.2 "Information-based methods" (pp. 620–621) — BALD
(parameter EIG, Eq. 53.32–53.33) vs. EPIG (predictive EIG,
Eq. 53.37–53.39). §53.4.3 "Targeting predictions" (p. 624) explicitly
warns: *"One notable case is BALD, which targets uncertainty in a set of
stochastic model parameters, not uncertainty in the model's predictions
directly, and can thus be of limited use in improving predictive
performance."*

`compute_influence_weights` (`active_selection.py:364`) computes
$I(v) = \|\nabla_\phi \mathcal L(h_\phi; v)\|_2$ — a parameter-space,
gradient-norm heuristic with exactly the structural character the Handbook
flags as potentially mismatched to the actual goal (accurate *predictions*
of unseen-model accuracy, not parameter recovery). EPIG targets the
*predictive* quantity directly and is transductive-friendly — it can be
defined w.r.t. known unlabeled target inputs (§53.4.2 "Transductive
predictive EIGs"), which matches our setting exactly: we already have
unlabeled target descriptors ($T$/`target_T` in `select_extension`). With
A3's posterior, an EPIG-style score is computable in closed form for a
Gaussian-linear model.

**Effort:** moderate, depends on A3.

### A5. Use gradient-embedding k-means (BADGE), not raw-descriptor k-means, for the warm-start clustering stage

**Source:** §53.4.2 "Gradient-based methods → BADGE" (pp. 623–624),
Eq. 53.54. BADGE clusters candidates via **k-means++ on loss-gradient
embeddings**
$\mathcal E = \{\mathrm{vec}(-\partial/\partial\psi'_L \log p(y^{(1)};x,\psi)) \mid x \in \mathcal X_{\text{pool}}\}$
(gradient w.r.t. the *last layer only*, using the model's own pseudo-label)
— not raw input/feature space.

`docs/kmeans-submodular-warmstart.md` §6 diagnosed, quantitatively, exactly
why naive raw-descriptor k-means underperforms random subsampling: Lloyd's
algorithm allocates cluster resolution proportional to **candidate
density**, not **task relevance**, systematically starving the
sparse-but-important target-adjacent tail (measured: k-means put only 10%
of its representatives near-target vs. the true 20.4% base rate at
$M{=}30$). Gradient embeddings are a natural, literature-established fix
precisely because task/target-relevance is baked into the embedding space
itself — "how much would this point currently change what the model
believes" — rather than being something a downstream target-aware stage has
to recover after a target-blind pre-filter already discarded the useful
points. Concretely: swap `make_problem`'s k-means input from raw `V[j]`
descriptors to per-candidate last-layer gradients.
`compute_per_example_gradients` in `active_selection.py` already computes
this exact quantity for GRAD-MATCH (V3) — it would need restricting to
last-layer-only gradients (BADGE's actual recipe) and applying to sample-set
descriptors instead of full pairs.

**Effort:** moderate — the gradient machinery already exists (V3's code
path), needs re-wiring into the clustering stage. **This is probably the
single highest-leverage, most concretely actionable idea in this list** —
it directly targets a bug we already measured and diagnosed.

### A6. Cite MaxHerding / TypiClust as the established names for what V1 and the k-means warm-start are already doing

**Source:** §53.4.2 "Coverage-based methods" (p. 623).

**MaxHerding** (Eq. 53.51):
$\mathrm{KernelCoverage}(x_b^+,\mathcal X_{\text{lab}},\mathcal X_{\text{pool}},k) = \frac{1}{|\mathcal X_{\text{pool}}|}\sum_{x\in\mathcal X_{\text{pool}}} \max_{x_{\text{lab}}} k(\phi(x),\phi(x_{\text{lab}}))$
— structurally identical to `active_selection.py`'s influence-weighted
facility location, $f(A) = \sum_v I(v)\max_s k(v,s)$, just with uniform
weight instead of $I(v)$ and averaged over the whole pool instead of a
target-narrowed $V_T$. Citing MaxHerding strengthens
`docs/submodularity-audit.md` Part 1's proof section — it isn't a novel
construction from scratch, it's a validated instance of a named, published
active-learning family, with influence-weighting and target-narrowing as
this project's specific contribution on top.

**TypiClust** (Eq. 53.52, p. 623) is the closest published precedent to the
exact k-means-warm-start problem in `docs/kmeans-submodular-warmstart.md`:
cluster the pool, then within each cluster select the point maximizing
local *typicality* (density of $k$ nearest neighbors) — a genuinely
different representative-selection rule than "nearest to centroid," worth
comparing as a third variant alongside A5's target-aware fix.

**Effort:** cheap (citation) for the MaxHerding framing; moderate (one more
baseline implementation) for the TypiClust comparison.

### A7. Reweight the meta-training/adaptation loss to correct for acquisition-induced sampling bias

**Source:** §53.4.3 "Constructing accurate models" (p. 624–625):
*"An important factor in designing predictive algorithms is that elements
of an actively acquired data sequence are not independent and identically
distributed, leading to statistical bias in models produced by standard
empirical-risk minimisation... Reweighted risk estimators can be used to
prevent this bias... but interestingly these do not always lead to better
predictions."*

Once V1/V2/V3 select a *non-i.i.d.* subset $\Omega$ (deliberately biased
toward target-relevant, high-influence points), `meta_learning.py`'s
adaptation loop (`_adapt_context`/`_param_step`) trains on it with plain
unweighted MSE — no correction for the fact the sample is no longer
representative of the underlying (model, workload) population. An
inverse-propensity-weighted loss (weight each selected example by
$1/$its acquisition-score-derived selection probability) is the standard
fix — though the Handbook is honest it doesn't reliably help. Worth trying
and reporting either way, as an honest ablation in the style already
established in `docs/kmeans-submodular-warmstart.md`.

**Effort:** cheap to try (a loss-reweighting change in `meta_learning.py`),
valuable either as a positive result or as a documented honest negative
result matching the Handbook's own caveat.

---

## Group B — new angles, not previously flagged as gaps

### B1. Add confidence intervals to `predicted_accuracy`

**Source:** §53.5 Conclusion (p. 625): *"One complex problem that warrants
special attention is active testing... there is more work to be done to
handle more complex notions of model behaviour than the basic measures of
predictive performance that have been the focus of most prior work"* —
citing work on providing confidence intervals on performance estimates.

Every prediction record in this codebase (`meta_learning.py::evaluate`,
`evaluate_transfer`, and every `outputs/*_predictions.json` artifact)
currently reports only a bare scalar `predicted_accuracy` + `mae` — never a
calibrated interval. With A3's posterior this is a direct, cheap add-on:
report $\big(\mu_N^\top\phi(x_*),\; \phi(x_*)^\top\Sigma_N\phi(x_*) + \sigma_\varepsilon^2\big)$
per prediction — exactly the kind of concrete, currently-missing
deliverable the Handbook flags as valuable and underdeveloped in the
active-testing literature specifically.

**Effort:** cheap once A3 exists.

### B2. A calibrated "trust this prediction or abstain" wrapper, selective-classification style, repurposed for regression

**Source:** Chapter 44 "Confidence-based selective classification," read
through §44.2.2 (pp. 489–492).

The chapter's core move — combine several independent confidence/uncertainty
signals into one calibrated rejection score, then threshold it to trade off
coverage vs. reliability — transfers directly to ActiveEvaluator's *output*
side, not just its acquisition side. Several signals already exist without
new machinery: the narrowing diagnostics' $\rho$/similarity threshold (how
well-covered is this unseen model by the narrowed validation set $V_T$),
the influence-weighted coverage achieved by the final selected support set,
and (with A3) the last-layer posterior variance. None of these are
currently combined into a single "should you trust this specific
unseen-model prediction" score — `selection_summary.json` reports them as
separate diagnostics. This is a genuinely new framing (the chapter is about
classification; transferring to regression/selective-prediction needs
adapting the risk-coverage machinery, §44.2.4–44.2.6), not a small patch.

**Effort:** substantial — a real, self-contained mini-contribution if
pursued (own experiment + doc).

### B3. Chapter 27 "Meta-learning deep kernel Gaussian processes" — likely formalizes the meta-learning + kernel combination this pipeline already does informally

**Not verified in depth — TOC-level only.** From the table of contents:
§27.3 "Adaptive Deep Kernel Fitting with Implicit Function Theorem"
(described as "a unification framework"), §27.4.2 cross-references
"Meta-learning," §27.4.3 "Multi-Task Gaussian Processes."

ActiveEvaluator's CAVIA-style inner-loop context adaptation
(`meta_learning.py`) *is* a meta-learning system, and its selection stage
(`active_selection.py`) builds an RBF kernel directly on the same network's
learned features (`compute_embeddings`) — informally, a deep-kernel-learning
setup bolted onto a meta-learner. This chapter appears to be the formal
literature this combination should be positioned against, but its actual
content hasn't been read.

**Recommend:** a follow-up deep-read of pp. 267–277 (printed) before citing
it substantively.

### B4. Chapter 22 "PAC-Bayesian deep neural network ensembles" — a possible route to a provable bound for influence weights or the ATC/DoC baselines

**Not verified in depth — TOC-level only.** From the table of contents:
covers the "PAC-Bayes-kl inequality," "Oracle bounds for the weighted
majority vote," "Empirical bounds for the weighted majority vote."

PAC-Bayes gives high-probability, finite-sample guarantees rather than only
asymptotic/worst-case-set arguments — potentially stronger footing than the
current empirical-only validation of the influence-weight heuristic, or a
way to put a real confidence bound on the ATC/DoC label-free estimator
baselines (`baselines/_core.py::estimate_atc/estimate_doc`), which
currently report bare point estimates with no theoretical guarantee at all.

**Recommend:** a real read before committing to this direction.

### B5. Chapter 55 "Credal BDL and Imprecise Probability in BDL" — a possible principled split of aleatoric vs. epistemic uncertainty

**Not verified in depth — TOC-level only.** From the table of contents:
§55.2.3 explicitly covers "Quantifying and Disentangling Aleatoric and
Epistemic Uncertainties."

This distinction is directly relevant but currently informal in this
codebase: the synthetic benchmark's heteroscedastic label noise
(`base_noise` scaled by `rel[j]`, an *aleatoric*, irreducible-per-pair
quantity) is conceptually different from "how much do we not know about a
region of descriptor space because we haven't sampled it" (*epistemic*,
what active selection is actually supposed to reduce) — but nothing in the
current selection objectives ($F_{\text{sample}}$, $F_{\text{info}}$,
influence weights) explicitly separates the two. A credal/imprecise-
probability framing might formalize this split properly rather than
leaving it implicit in the noise-generation code.

**Recommend:** a real read before committing to this direction.

---

## Priority read-out

**Quick wins, high confidence:** A6 (citations), A7 (cheap ablation to try
immediately, own honest-negative-result value either way).

**Moderate effort, high payoff, best-grounded in verified reading:** A5
(gradient-embedding clustering — directly fixes the diagnosed
density-vs-relevance bug in `docs/kmeans-submodular-warmstart.md`), A3
(last-layer Bayesian posterior — unlocks A2/A4/B1, reuses existing
`compute_embeddings` infrastructure with no architecture change).

**Substantial new contributions:** A2 (Bayesian EVR replacing V2's
retraining oracle), A4 (EPIG-style influence weight), B2 (selective-
prediction wrapper).

**Needs a real chapter read before acting on:** B3, B4, B5 — flagged
honestly as table-of-contents-level pointers, not verified content.

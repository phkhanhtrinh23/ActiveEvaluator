# Deriving the Beta ESS from scratch

Two things to prove, from MTM08's Definition 1, with nothing assumed:

- **(A)** $\mathrm{Beta}(3,7)$ has $\mathrm{ESS}=10$ — and more generally
  $\mathrm{Beta}(\tilde\alpha,\tilde\beta)$ has $\mathrm{ESS}=\tilde\alpha+\tilde\beta$.
- **(B)** The power prior built from $\mathrm{Beta}(1,1)$ and 3-successes-in-10 has
  $\mathrm{ESS}=10\,a_0+2$.

(B) turns out to be a one-line corollary of (A) once you see what the power prior actually
*is*. So (A) is where the work happens.

---

## 0. The definition we are working from

MTM08 Definition 1. The ESS of a prior $p$ relative to a likelihood $f_m$ is the $m$
minimizing the **distance**

$$
\delta(m,\bar\theta,p,q_0) \;=\; \Big|\, D_p(\bar\theta) \;-\; D_q(m,\bar\theta) \,\Big| ,
\qquad
D(\theta) \;=\; -\frac{d^2}{d\theta^2}\log(\text{density}) .
$$

Three moving parts:

| symbol | what it is |
|---|---|
| $p$ | the prior whose ESS we want |
| $q_0$ | an **$\varepsilon$-information prior**: same mean as $p$, variance blown up so it carries ~no information |
| $q_m$ | the posterior you get by updating $q_0$ with $m$ observations |

**The idea in one sentence.** Curvature of a log-density measures how sharply peaked — how
*informative* — it is. So ask: *how many observations $m$ must I feed a deliberately vague
prior before its posterior is as sharply peaked as my actual prior $p$?* That $m$ is the ESS.

We evaluate curvature at the prior mean $\bar\theta=E_p(\theta)$, and average the posterior
curvature over the marginal distribution of the data.

---

## Part A — $\mathrm{Beta}(\tilde\alpha,\tilde\beta)$ has ESS $=\tilde\alpha+\tilde\beta$

### Step 1. Curvature of the actual prior

$p(\theta)=\mathrm{Beta}(\tilde\alpha,\tilde\beta)$ has density
$p(\theta)\propto\theta^{\tilde\alpha-1}(1-\theta)^{\tilde\beta-1}$, so

$$
\log p(\theta)=(\tilde\alpha-1)\log\theta+(\tilde\beta-1)\log(1-\theta)+\text{const}.
$$

Differentiate once:

$$
\frac{d\log p}{d\theta}=\frac{\tilde\alpha-1}{\theta}-\frac{\tilde\beta-1}{1-\theta} .
$$

Differentiate again (note the second term's sign flips again by the chain rule):

$$
\frac{d^{2}\log p}{d\theta^{2}}=-\frac{\tilde\alpha-1}{\theta^{2}}-\frac{\tilde\beta-1}{(1-\theta)^{2}} .
$$

Hence

$$
\boxed{\;D_p(\theta)=\frac{\tilde\alpha-1}{\theta^{2}}+\frac{\tilde\beta-1}{(1-\theta)^{2}}\;}
\tag{A.1}
$$

The prior mean is $\bar\theta=\dfrac{\tilde\alpha}{\tilde\alpha+\tilde\beta}$.

### Step 2. Build the $\varepsilon$-information prior $q_0$

MTM08's Table 1 prescribes, for the beta case,

$$
q_0=\mathrm{Beta}\!\left(\frac{\tilde\alpha}{c},\;\frac{\tilde\beta}{c}\right),
\qquad c\ \text{a large constant.}
$$

Check it does what an $\varepsilon$-prior must:

- **Same mean.** $\dfrac{\tilde\alpha/c}{\tilde\alpha/c+\tilde\beta/c}=\dfrac{\tilde\alpha}{\tilde\alpha+\tilde\beta}=\bar\theta$. ✓
- **Inflated variance.** $\operatorname{Var}\mathrm{Beta}(a,b)=\dfrac{ab}{(a+b)^2(a+b+1)}$. Substituting $a\to\tilde\alpha/c,\;b\to\tilde\beta/c$, the $c^{-2}$ cancels top and bottom and leaves

$$
\operatorname{Var}(q_0)=\frac{\tilde\alpha\tilde\beta}{(\tilde\alpha+\tilde\beta)^{2}\big(\frac{\tilde\alpha+\tilde\beta}{c}+1\big)}
\;\xrightarrow[c\to\infty]{}\;
\frac{\tilde\alpha\tilde\beta}{(\tilde\alpha+\tilde\beta)^{2}},
$$

which is the **largest** variance any distribution on $[0,1]$ with mean $\bar\theta$ can have.
Maximum spread at fixed mean = minimum information. ✓

The total pseudo-count is $(\tilde\alpha+\tilde\beta)/c\to0$: this is the "$\varepsilon$".

### Step 3. Update $q_0$ with $m$ Bernoulli observations

The likelihood is $m$ i.i.d. Bernoulli draws, $Y=\sum_{i=1}^m Y_i$ successes:

$$
f_m(Y_m\mid\theta)=\theta^{Y}(1-\theta)^{m-Y}.
$$

By Beta–Bernoulli conjugacy (add successes to the first parameter, failures to the second):

$$
q_m(\theta\mid Y)=\mathrm{Beta}\!\left(\frac{\tilde\alpha}{c}+Y,\;\;\frac{\tilde\beta}{c}+m-Y\right).
$$

Its curvature comes from the *same* formula (A.1), just with these parameters:

$$
D_q(m,\theta,Y)=\frac{\frac{\tilde\alpha}{c}+Y-1}{\theta^{2}}+\frac{\frac{\tilde\beta}{c}+m-Y-1}{(1-\theta)^{2}} .
\tag{A.2}
$$

### Step 4. Average over the data

$D_q$ still depends on the random $Y$, so average over the marginal of $Y$. Marginalizing the
Bernoulli likelihood against the **actual** prior $p=\mathrm{Beta}(\tilde\alpha,\tilde\beta)$
gives a Beta-Binomial, whose mean is

$$
E[Y]=m\cdot\frac{\tilde\alpha}{\tilde\alpha+\tilde\beta}=m\bar\theta .
$$

$D_q$ is **linear in $Y$**, so averaging is just substituting $E[Y]$:

$$
D_q(m,\bar\theta)=\frac{\frac{\tilde\alpha}{c}+m\bar\theta-1}{\bar\theta^{2}}
+\frac{\frac{\tilde\beta}{c}+m(1-\bar\theta)-1}{(1-\bar\theta)^{2}} .
\tag{A.3}
$$

### Step 5. Subtract, and watch things cancel

$$
D_p(\bar\theta)-D_q(m,\bar\theta)
=\underbrace{\frac{(\tilde\alpha-1)-\big(\tfrac{\tilde\alpha}{c}+m\bar\theta-1\big)}{\bar\theta^{2}}}_{\text{the }-1\text{'s cancel}}
+\frac{(\tilde\beta-1)-\big(\tfrac{\tilde\beta}{c}+m(1-\bar\theta)-1\big)}{(1-\bar\theta)^{2}}
$$

$$
=\frac{\tilde\alpha-\tfrac{\tilde\alpha}{c}-m\bar\theta}{\bar\theta^{2}}
+\frac{\tilde\beta-\tfrac{\tilde\beta}{c}-m(1-\bar\theta)}{(1-\bar\theta)^{2}} .
$$

Now let $c\to\infty$, killing the $\varepsilon$ terms:

$$
D_p(\bar\theta)-D_q(m,\bar\theta)=\frac{\tilde\alpha-m\bar\theta}{\bar\theta^{2}}
+\frac{\tilde\beta-m(1-\bar\theta)}{(1-\bar\theta)^{2}} .
\tag{A.4}
$$

**This is already the whole answer, if you read it right.** The two numerators vanish
simultaneously exactly when

$$
m\bar\theta=\tilde\alpha
\quad\text{and}\quad
m(1-\bar\theta)=\tilde\beta ,
$$

i.e. when the $m$ observations deliver, *in expectation*, precisely $\tilde\alpha$ successes
and $\tilde\beta$ failures — the prior's own pseudo-counts. Adding the two equations gives
$m=\tilde\alpha+\tilde\beta$. And because $\bar\theta=\tilde\alpha/(\tilde\alpha+\tilde\beta)$
was *defined* as the prior mean, both equations are satisfied by that same $m$ — they don't
conflict. That consistency is why the answer is unambiguous.

### Step 6. Finish algebraically

Write $S=\tilde\alpha+\tilde\beta$, so $\bar\theta=\tilde\alpha/S$ and $1-\bar\theta=\tilde\beta/S$.
First term of (A.4):

$$
\frac{\tilde\alpha-m\frac{\tilde\alpha}{S}}{\big(\frac{\tilde\alpha}{S}\big)^{2}}
=\frac{\tilde\alpha\big(1-\frac{m}{S}\big)S^{2}}{\tilde\alpha^{2}}
=\frac{S^{2}-mS}{\tilde\alpha}
=\frac{S(S-m)}{\tilde\alpha}.
$$

Identically, the second term is $\dfrac{S(S-m)}{\tilde\beta}$. Therefore

$$
\boxed{\;\delta(m)=\Big|\,S(S-m)\Big(\frac{1}{\tilde\alpha}+\frac{1}{\tilde\beta}\Big)\Big|\;}
\tag{A.5}
$$

A V-shaped function of $m$, vanishing at exactly one point:

$$
\delta(m)=0 \iff m=S=\tilde\alpha+\tilde\beta .
$$

$$
\boxed{\;\mathrm{ESS}\big(\mathrm{Beta}(\tilde\alpha,\tilde\beta)\big)=\tilde\alpha+\tilde\beta\;}
$$

This V-shape is exactly MTM08's Figure 1.

### Step 7. Numerical check with $\mathrm{Beta}(3,7)$

$\tilde\alpha=3$, $\tilde\beta=7$, $S=10$, $\bar\theta=3/10=0.3$.

**Prior curvature**, from (A.1):

$$
D_p(0.3)=\frac{3-1}{0.3^{2}}+\frac{7-1}{0.7^{2}}=\frac{2}{0.09}+\frac{6}{0.49}=22.222+12.245=34.467 .
$$

**Posterior curvature** at $m=10$, from (A.3) with $c\to\infty$:

$$
D_q(10,0.3)=\frac{10(0.3)-1}{0.09}+\frac{10(0.7)-1}{0.49}=\frac{2}{0.09}+\frac{6}{0.49}=34.467 .
$$

Identical — $\delta(10)=0$. ✓

Sanity-check two other values against formula (A.5), which predicts
$\delta(m)=\big|10(10-m)(\tfrac13+\tfrac17)\big|=4.762\,|10-m|$:

| $m$ | $D_q(m,0.3)$ direct | $\delta=\lvert 34.467-D_q\rvert$ | formula (A.5) |
|---:|---:|---:|---:|
| 0 | $\frac{-1}{0.09}+\frac{-1}{0.49}=-13.152$ | 47.619 | 47.619 ✓ |
| 5 | $\frac{0.5}{0.09}+\frac{2.5}{0.49}=10.658$ | 23.810 | 23.810 ✓ |
| 10 | $34.467$ | **0** | **0** ✓ |
| 15 | $\frac{3.5}{0.09}+\frac{9.5}{0.49}=58.277$ | 23.810 | 23.810 ✓ |

Minimum at $m=10$. **$\mathrm{Beta}(3,7)\Rightarrow\mathrm{ESS}=10$.** $\blacksquare$

---

## Part B — the power prior, $\mathrm{ESS}=10\,a_0+2$

### Step 1. What a power prior is

Ibrahim & Chen (2000): you have historical data $D_0$ you only partly trust. Raise its
likelihood to a fractional power $a_0\in[0,1]$ and use that as a prior, on top of some initial
prior $p_0$:

$$
p(\theta\mid D_0,a_0)\;\propto\;L(\theta\mid D_0)^{a_0}\;p_0(\theta) .
$$

$a_0=1$ trusts the historical data fully; $a_0=0$ discards it and leaves $p_0$; fractional
$a_0$ downweights it continuously. **This is exactly a discount knob.**

### Step 2. Instantiate MTM08's Example 6

Their setup: $p_0=\mathrm{Beta}(1,1)$ (uniform, density $\propto\theta^{0}(1-\theta)^{0}=1$),
and $D_0$ = **3 successes in 10 trials**, so

$$
L(\theta\mid D_0)=\theta^{3}(1-\theta)^{7}.
$$

Multiply out — this is the entire proof:

$$
p(\theta\mid D_0,a_0)
\;\propto\;\big\{\theta^{3}(1-\theta)^{7}\big\}^{a_0}\cdot 1
\;=\;\theta^{3a_0}(1-\theta)^{7a_0}
\;=\;\theta^{(1+3a_0)-1}(1-\theta)^{(1+7a_0)-1}.
$$

That last form is a Beta density kernel. So the power prior is **just a Beta distribution**:

$$
p(\theta\mid D_0,a_0)=\mathrm{Beta}\big(1+3a_0,\;\,1+7a_0\big).
$$

### Step 3. Apply Part A

Part A says ESS $=$ sum of the two parameters:

$$
\mathrm{ESS}=(1+3a_0)+(1+7a_0)=2+10a_0 .
$$

$$
\boxed{\;\mathrm{ESS}=10\,a_0+2\;}\qquad\blacksquare
$$

So there is no separate machinery here at all. **The power prior is a Beta, and Part A already
told us every Beta's ESS.**

Sanity check the endpoints: $a_0=0$ gives ESS $=2$, the uniform $\mathrm{Beta}(1,1)$, which has
$1+1=2$. ✓ And $a_0=1$ gives ESS $=12$: the 10 historical observations *plus* the 2 already
carried by the uniform prior. ✓

### Step 4. The general rule, derived

Historical data: $y_0$ successes in $n_0$ trials. Initial prior $\mathrm{Beta}(a_{00},b_{00})$.

$$
p\;\propto\;\big\{\theta^{y_0}(1-\theta)^{n_0-y_0}\big\}^{a_0}\;\theta^{a_{00}-1}(1-\theta)^{b_{00}-1}
\;=\;\theta^{\,a_0y_0+a_{00}-1}(1-\theta)^{\,a_0(n_0-y_0)+b_{00}-1},
$$

that is $p=\mathrm{Beta}\big(a_0y_0+a_{00},\;a_0(n_0-y_0)+b_{00}\big)$. Sum the parameters —
the $y_0$ terms cancel:

$$
\mathrm{ESS}=a_0y_0+a_{00}+a_0(n_0-y_0)+b_{00}=a_0\,n_0+(a_{00}+b_{00}).
$$

Since $n_0=\mathrm{ESS}\{L(\theta\mid D_0)\}$ ($n_0$ real observations) and
$a_{00}+b_{00}=\mathrm{ESS}\{p_0\}$ (Part A):

$$
\boxed{\;\mathrm{ESS}(p)\;=\;a_0\cdot\mathrm{ESS}\{L(\theta\mid D_0)\}\;+\;\mathrm{ESS}\{p_0\}\;}
$$

which is MTM08's stated Example 6 identity, now proved rather than quoted. $\blacksquare$

> **Note on a typo.** The PDF renders this example as
> $\propto\{\theta^3(1-\theta^7)\}^{a_0}\theta(1-\theta)$. Two glitches: $(1-\theta^7)$ should
> be $(1-\theta)^7$; and the trailing $\theta(1-\theta)$ is the kernel of
> $\mathrm{Beta}(2,2)$, not the $\mathrm{Beta}(1,1)$ they name in the text. Taking
> $\mathrm{Beta}(2,2)$ literally would give $10a_0+4$, contradicting their own stated answer.
> Their $\mathrm{Beta}(1,1)$ is the correct reading, and it reproduces $10a_0+2$ exactly.

---

## Part C — our case, now fully grounded

Our prior (`appendix_formulation.tex`, §`app:beta-priors`) is

$$
\alpha_j\sim\mathrm{Beta}\big(1+s_j\pi_j,\;1+s_j(1-\pi_j)\big).
$$

**Route 1 — directly via Part A.** Sum the parameters:

$$
(1+s_j\pi_j)+(1+s_j(1-\pi_j))=2+s_j\quad\Longrightarrow\quad \mathrm{ESS}=s_j+2 .
$$

**Route 2 — via Part B**, which additionally explains *where $s_j$ comes from*:

| Example 6 | PoolEvaluator |
|---|---|
| historical data $D_0$ | calibration subset $\mathcal S^\star$ |
| $n_0$ trials | $\lvert\mathcal S^\star\rvert$ labeled items |
| $y_0$ successes | $\pi_j\lvert\mathcal S^\star\rvert$ items $M_j$ got right |
| power $a_0$ | $\operatorname{clip}(\bar S_{\mathcal Q^\star},0,1)$ — **our target-match discount** |
| initial prior $p_0$ | $\mathrm{Beta}(1,1)$ |

Substituting into the general rule of Step 4:

$$
\mathrm{ESS}=\underbrace{\operatorname{clip}(\bar S_{\mathcal Q^\star},0,1)\cdot\lvert\mathcal S^\star\rvert}_{=\;s_j\ \text{by our \texttt{eq:appendix-prior-weight-discounted}}}\;+\;2\;=\;s_j+2 .
$$

Both routes agree. Two consequences for the paper:

1. **The discount is not a heuristic.** $\operatorname{clip}(\bar S_{\mathcal Q^\star},0,1)$
   *is* the Ibrahim–Chen power-prior exponent $a_0$. Down-weighting a poorly-aligned
   calibration subset is formally identical to partially trusting historical data, and MTM08
   gives its ESS in closed form.
2. **Our stated ESS is off by exactly 2.** We call $s_j$ the effective sample size; the
   MTM08 ESS is $s_j+2$. The $+2$ is the uniform $\mathrm{Beta}(1,1)$ baseline. Harmless, but
   worth stating before a reviewer finds it.

---

## Summary

| claim | why |
|---|---|
| $\mathrm{Beta}(\tilde\alpha,\tilde\beta)\Rightarrow\mathrm{ESS}=\tilde\alpha+\tilde\beta$ | curvature difference (A.5) is $S(S-m)(\tfrac1{\tilde\alpha}+\tfrac1{\tilde\beta})$, zero iff $m=\tilde\alpha+\tilde\beta$ |
| $\mathrm{Beta}(3,7)\Rightarrow 10$ | $3+7$; verified numerically at $m=0,5,10,15$ |
| power prior $\Rightarrow 10a_0+2$ | it *is* $\mathrm{Beta}(1+3a_0,1+7a_0)$; apply the sum rule |
| general power prior | parameters sum to $a_0n_0+(a_{00}+b_{00})$; the $y_0$ terms cancel |
| ours $\Rightarrow s_j+2$ | $a_0=\operatorname{clip}(\bar S_{\mathcal Q^\star},0,1)$, $n_0=\lvert\mathcal S^\star\rvert$, $p_0=\mathrm{Beta}(1,1)$ |

The single load-bearing fact behind all of it: **for a Beta prior against a Bernoulli
likelihood, curvature is linear in the pseudo-counts, so "matching curvature" reduces to
"matching counts."**

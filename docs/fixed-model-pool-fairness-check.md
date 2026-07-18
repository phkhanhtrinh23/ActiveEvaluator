# Fixed-model-pool fairness check

Companion to the storage-currency finding in
[docs/rq4-cost-currency-mechanism.md](rq4-cost-currency-mechanism.md): under the
amortized `storage` budget, model-concentrating baselines (`Submod. benchmark`,
`Bayesian opt. design`) buy far more actions than target-aware methods
(`ActiveEval-*`), because checkpoint cost is paid once per unique reference model.
That leaves it unclear whether ActiveEval is *penalized for spreading across more
models*, or simply worse at picking pairs.

This experiment controls for the model set directly: freeze the reference-model
set `Submod. benchmark` (`select_logdet`) buys under the storage budget, then let
every other method reselect its own pairs restricted to *only that model set*, at
the same action count `Submod. benchmark` bought. Each method is also run at that
same action count on the full, unrestricted pool (paired RNG draw) as the fairness
baseline. `Delta = Fixed − Free`; positive means being confined to
`Submod. benchmark`'s models hurts that method.

Run: `python -m experiments.run_acquisition_benchmark --mode fixed_model_pool --seeds 5 --budget-frac 0.15`
Code: [experiments/run_acquisition_benchmark.py](../experiments/run_acquisition_benchmark.py) (`run_fixed_model_pool`)

## Results (5 seeds, 15% of storage pool cost)

`Submod. benchmark` bought a median of **251 actions across 16 models** (source MAE
5.08 ± 0.95 pp). Unseen MAE (pp), lower is better; sorted by delta.

| Method | Free (unrestricted) | Fixed (Submod.'s models) | Delta |
|---|---:|---:|---:|
| **GRAD-MATCH** | 5.72 ± 1.63 | **5.02 ± 0.65** | **−0.70** |
| Random | 5.62 ± 0.66 | 5.33 ± 0.82 | −0.29 |
| **Submod. benchmark (source)** | 5.08 ± 0.95 | 5.08 ± 0.95 | 0 (defines the model set) |
| k-center | 5.04 ± 0.53 | 5.11 ± 0.55 | +0.06 |
| Bayesian opt. design | 4.82 ± 0.85 | 5.17 ± 0.88 | +0.35 |
| Facility-location | 5.00 ± 0.44 | 5.36 ± 0.79 | +0.36 |
| **ActiveEval-Pair** | **4.45 ± 0.36** | **5.02 ± 0.93** | +0.58 |
| ActiveEval-S+M | 5.03 ± 0.58 | 5.62 ± 1.20 | +0.59 |
| ActiveEval-S | 4.47 ± 0.47 | 5.06 ± 0.76 | +0.59 |
| Active testing | 4.62 ± 0.72 | 5.26 ± 1.06 | +0.64 |
| Greedy MI (Alg. 2) | 4.83 ± 0.46 | 5.54 ± 0.80 | +0.71 |
| Matrix completion | 5.20 ± 0.45 | 5.95 ± 1.13 | +0.75 |

**Best results:** lowest MAE overall is **ActiveEval-Pair, Free = 4.45 ± 0.36 pp**
(best of every cell in the table). Lowest MAE under the *fixed* model-set constraint
is a tie between **GRAD-MATCH and ActiveEval-Pair at 5.02 pp**. Best (most negative)
delta — the method least hurt, in fact helped, by being confined to
`Submod. benchmark`'s models — is **GRAD-MATCH at −0.70 pp**.

## Reading

- **`Submod. benchmark (source)`** is the reference row: it's the method whose
  storage-budget picks *define* the frozen model set, so its Free and Fixed columns
  are identical by construction (5.08 ± 0.95 pp either way) — it's the anchor every
  other row's "Fixed" column is being compared against, and against which the other
  rows' "Free" MAE is what RQ4's original storage-budget table already reported.
- All three **ActiveEval variants** get worse when confined to `Submod. benchmark`'s
  16-model set (+0.58 to +0.59 pp) — consistent with the RQ4 mechanism: ActiveEval's
  advantage comes partly from spreading picks across many models to stay close to
  the target region, which the model-concentrated pool takes away.
- **GRAD-MATCH and Random go the other way** (negative delta) — restricting the pool
  does not hurt (even slightly helps) methods that were not exploiting model
  diversity in the first place.
- **k-center is flat** (+0.06), as expected for a pure diversity method indifferent
  to which specific models are available.
- CIs overlap across most rows at 5 seeds, so treat the ranking/sign as directional,
  not yet a paired-significant result. Re-run with more seeds (e.g. 15, matching the
  greedy-MI ablation in `rq4-cost-currency-mechanism.md`) before citing this in the
  paper.

Raw output: `outputs/fixed_model_pool_benchmark.json`.

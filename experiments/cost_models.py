"""Per-action cost models for *real-currency* labeling budgets.

The acquisition benchmark (``run_acquisition_benchmark.py``) treats the budget as
a **count** of evaluation actions -- every (reference-model, sample-set) pair
costs exactly 1. In practice, running a model on a workload slice costs different
*currencies*:

* ``input_tok``  -- input tokens fed to the model  (scales with the sample-set
  size and prompt length; roughly model-independent),
* ``output_tok`` -- generated tokens               (sample-set size x model
  verbosity),
* ``latency``    -- wall-clock seconds             (tokens x model compute),
* ``memory``     -- peak resident GB               (model parameter count; set-
  independent, paid per action),
* ``storage``    -- checkpoint + cache GB          (checkpoint counted **once per
  unique model**, plus a small per-action cache -- a non-additive *set* cost that
  rewards concentrating the budget on few models).

Each currency is a :class:`Currency` = an additive per-action ``pair_cost`` array
plus an optional per-model ``model_cost`` (non-zero only for amortized storage).
The total cost of a selected set ``S`` is::

    sum(pair_cost[S]) + sum(model_cost[m] for m in unique_models(S))

so ``count`` (``pair_cost = 1``, ``model_cost = 0``) recovers the original
cardinality budget exactly. Costs are derived from per-model "size" (parameter
count) and per-sample-set "size" (number of examples, prompt length) factors
drawn with an rng offset from the accuracy-noise rng, so they are reproducible
and independent of the label noise. Absolute scales are illustrative (billions of
params, fp16 GB, ~seconds); only the *ratios across actions* drive selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


# fp16 weights: 2 bytes/param; params expressed in billions -> GB = params_b * 2.
_BYTES_PER_PARAM_GB = 2.0
# latency constant: tokens * params_b * C_LAT ~ seconds (a 7B model on 1000
# examples x ~400 tok/example lands around a few hundred seconds).
_C_LAT = 1.4e-4
# cached-generation storage: 2 bytes/token -> GB.
_CACHE_GB_PER_TOK = 2e-9


@dataclass
class Currency:
    """One budget currency over the candidate pool.

    ``pair_cost[i]`` is the additive cost of action ``i``; ``model_cost[m]`` is a
    one-time cost paid the first time any action of model ``m`` is selected
    (zero for the additive currencies, non-zero only for amortized storage).
    """

    name: str
    unit: str
    pair_cost: np.ndarray            # (P,)
    model_cost: np.ndarray           # (n_models,) -- zeros if additive
    amortized: bool = False

    def total(self, idx, pair_model: np.ndarray) -> float:
        """Total cost of selecting the actions ``idx``."""
        idx = np.asarray(idx, int)
        c = float(self.pair_cost[idx].sum())
        if self.amortized and idx.size:
            c += float(self.model_cost[np.unique(pair_model[idx])].sum())
        return c

    def pool_total(self, pair_model: np.ndarray) -> float:
        """Cost of selecting the *entire* pool (denominator for a budget fraction)."""
        return self.total(np.arange(len(self.pair_cost)), pair_model)


def greedy_fill(order: List[int], currency: Currency, budget: float,
                pair_model: np.ndarray) -> tuple[List[int], float]:
    """Walk a method's (cost-agnostic) greedy ``order`` and take every action whose
    *marginal* cost fits the remaining budget (skip-and-continue).

    Selection order comes from the acquisition method; the currency only decides
    *how far the budget stretches*. For amortized currencies the marginal cost of
    an action is free of the checkpoint term once that model has already been
    selected, so concentrating on few models buys more actions.
    """
    kept: List[int] = []
    paid = 0.0
    used: set = set()
    pc = currency.pair_cost
    mc = currency.model_cost
    for idx in order:
        idx = int(idx)
        m = int(pair_model[idx])
        marg = float(pc[idx]) + (0.0 if (not currency.amortized or m in used) else float(mc[m]))
        if paid + marg <= budget + 1e-9:
            kept.append(idx)
            paid += marg
            used.add(m)
    if not kept and order:                     # guarantee at least one action
        kept = [int(order[0])]
        paid = currency.total(kept, pair_model)
    return kept, paid


def make_cost_model(prob: dict, seed: int) -> Dict[str, Currency]:
    """Build the per-currency cost model for a problem instance.

    Uses ``prob['pair_model']`` / ``prob['pair_sample']`` to know which model and
    sample-set each action touches, then draws log-uniform size factors (params,
    examples, prompt/output lengths) with an rng offset from the label-noise rng.
    """
    pair_model = np.asarray(prob["pair_model"])
    pair_sample = np.asarray(prob["pair_sample"])
    P = len(pair_model)
    n_models = int(pair_model.max()) + 1
    sample_ids = np.unique(pair_sample)
    s_index = {int(s): k for k, s in enumerate(sample_ids)}
    S = len(sample_ids)

    rng = np.random.default_rng(seed + 7919)   # decoupled from label noise

    def _logu(lo, hi, size):
        return np.exp(rng.uniform(np.log(lo), np.log(hi), size=size))

    params_b = _logu(0.5, 70.0, n_models)      # per-model parameter count (billions)
    g_len = _logu(32.0, 256.0, n_models)       # per-model generated tokens / example
    n_ex = _logu(200.0, 3000.0, S)             # per-set example count
    p_len = _logu(128.0, 1024.0, S)            # per-set input tokens / example
    # broadcast per-model / per-set factors to per-action vectors
    pm = pair_model
    ps = np.array([s_index[int(s)] for s in pair_sample])
    params_pair = params_b[pm]
    glen_pair = g_len[pm]
    nex_pair = n_ex[ps]
    plen_pair = p_len[ps]
    
    input_tok = nex_pair * plen_pair
    output_tok = nex_pair * glen_pair
    latency = nex_pair * (plen_pair + glen_pair) * params_pair * _C_LAT
    memory = params_pair * _BYTES_PER_PARAM_GB
    cache_gb = output_tok * _CACHE_GB_PER_TOK
    checkpoint_gb = params_b * _BYTES_PER_PARAM_GB

    zero_model = np.zeros(n_models)
    return {
        "count":      Currency("count", "actions", np.ones(P), zero_model),
        "input_tok":  Currency("input_tok", "tokens", input_tok.astype(float), zero_model),
        "output_tok": Currency("output_tok", "tokens", output_tok.astype(float), zero_model),
        "latency":    Currency("latency", "sec", latency.astype(float), zero_model),
        "memory":     Currency("memory", "GB", memory.astype(float), zero_model),
        "storage":    Currency("storage", "GB", cache_gb.astype(float),
                               checkpoint_gb.astype(float), amortized=True),
    }

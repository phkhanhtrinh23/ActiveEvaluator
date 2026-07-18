"""Unit tests for the real-currency cost models (experiments/cost_models.py)."""

import numpy as np
import pytest

from experiments.cost_models import Currency, greedy_fill, make_cost_model


def _toy_prob(seed=0):
    # 3 models x 2 sample-sets = 6 actions
    pair_model = np.array([0, 0, 1, 1, 2, 2])
    pair_sample = np.array([3, 5, 3, 5, 3, 5])
    return {"pair_model": pair_model, "pair_sample": pair_sample,
            "X": np.zeros((6, 4), np.float32)}


def test_count_currency_is_cardinality():
    prob = _toy_prob()
    cm = make_cost_model(prob, 0)
    count = cm["count"]
    assert count.pool_total(prob["pair_model"]) == 6
    # every action costs exactly 1, no model term
    assert np.allclose(count.pair_cost, 1.0)
    assert not count.amortized


def test_additive_total_is_sum_of_pair_costs():
    prob = _toy_prob()
    cm = make_cost_model(prob, 1)
    tok = cm["input_tok"]
    idx = [0, 2, 4]
    assert tok.total(idx, prob["pair_model"]) == pytest.approx(tok.pair_cost[idx].sum())


def test_storage_amortizes_checkpoint_per_unique_model():
    prob = _toy_prob()
    cm = make_cost_model(prob, 2)
    st = cm["storage"]
    assert st.amortized
    pm = prob["pair_model"]
    # two actions of the SAME model pay one checkpoint + two caches
    one_model = st.total([0, 1], pm)
    expected = st.model_cost[0] + st.pair_cost[[0, 1]].sum()
    assert one_model == pytest.approx(expected)
    # two actions of DIFFERENT models pay two checkpoints -> strictly more
    two_models = st.total([0, 2], pm)
    assert two_models > one_model


def test_greedy_fill_respects_additive_budget():
    prob = _toy_prob()
    cm = make_cost_model(prob, 3)
    mem = cm["memory"]
    pm = prob["pair_model"]
    budget = 0.5 * mem.pool_total(pm)
    kept, paid = greedy_fill(list(range(6)), mem, budget, pm)
    assert paid <= budget + 1e-9
    assert mem.total(kept, pm) == pytest.approx(paid)


def test_greedy_fill_amortization_buys_more_when_concentrated():
    # storage budget stretches further along an order that reuses one model
    prob = _toy_prob()
    cm = make_cost_model(prob, 4)
    st = cm["storage"]
    pm = prob["pair_model"]
    budget = st.model_cost.min() * 1.0 + st.pair_cost.sum()  # ~one cheapest checkpoint
    # order that stays on the single cheapest model first
    cheap_m = int(np.argmin(st.model_cost))
    concentrated = [i for i in range(6) if pm[i] == cheap_m] + \
                   [i for i in range(6) if pm[i] != cheap_m]
    spread = [i for i in range(6) if pm[i] != cheap_m] + \
             [i for i in range(6) if pm[i] == cheap_m]
    kept_c, _ = greedy_fill(concentrated, st, budget, pm)
    kept_s, _ = greedy_fill(spread, st, budget, pm)
    assert len(kept_c) >= len(kept_s)


def test_greedy_fill_guarantees_at_least_one_action():
    prob = _toy_prob()
    cm = make_cost_model(prob, 5)
    mem = cm["memory"]
    pm = prob["pair_model"]
    kept, paid = greedy_fill(list(range(6)), mem, 0.0, pm)
    assert len(kept) == 1


def test_costs_are_reproducible_for_a_seed():
    prob = _toy_prob()
    a = make_cost_model(prob, 7)["latency"].pair_cost
    b = make_cost_model(prob, 7)["latency"].pair_cost
    assert np.allclose(a, b)

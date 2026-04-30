import json
from pathlib import Path

import pytest

from active_evaluator.budget import BudgetPlan, compute_budget


def test_compute_budget_fraction_default():
    plan = compute_budget(pool_size=200, n_rounds=5, budget_fraction=0.10)
    assert plan.total == 20
    assert plan.n_rounds == 5
    assert sum(plan.per_round) == plan.total
    assert plan.per_round[:-1] == [4, 4, 4, 4]
    assert plan.per_round[-1] == 4  # 20 // 5 == 4 with no remainder


def test_compute_budget_remainder_folds_into_last_round():
    plan = compute_budget(pool_size=23, n_rounds=4, budget_fraction=1.0)
    assert plan.total == 23
    assert sum(plan.per_round) == 23
    assert plan.per_round[-1] >= plan.per_round[0]


def test_compute_budget_absolute_overrides_fraction():
    plan = compute_budget(pool_size=200, n_rounds=2, budget_fraction=0.5, budget_absolute=7)
    assert plan.total == 7
    assert sum(plan.per_round) == 7


def test_compute_budget_min_budget_floor():
    plan = compute_budget(pool_size=2, n_rounds=1, budget_fraction=0.001, min_budget=1)
    assert plan.total >= 1


def test_compute_budget_caps_at_pool_size():
    plan = compute_budget(pool_size=3, n_rounds=2, budget_absolute=500)
    assert plan.total <= 3


def test_compute_budget_invalid_rounds():
    with pytest.raises(ValueError):
        compute_budget(pool_size=10, n_rounds=0)


def test_budget_plan_round_trip(tmp_path: Path):
    plan = compute_budget(pool_size=100, n_rounds=3, budget_fraction=0.1)
    path = tmp_path / "plan.json"
    plan.save(path)
    loaded = BudgetPlan.load(path)
    assert loaded.total == plan.total
    assert loaded.per_round == plan.per_round
    assert loaded.n_rounds == plan.n_rounds


def test_budget_plan_validates_consistency():
    with pytest.raises(ValueError):
        BudgetPlan(total=5, n_rounds=3, per_round=[1, 1])
    with pytest.raises(ValueError):
        BudgetPlan(total=5, n_rounds=2, per_round=[1, 1])

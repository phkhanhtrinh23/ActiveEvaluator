"""Budget arithmetic for active selection.

Provides a small dataclass that records the total budget plus its
per-round split, with simple JSON serialization for reproducibility.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class BudgetPlan:
    """Records the budget used for an active-selection run."""

    total: int
    n_rounds: int
    per_round: List[int]
    cost_fn_name: str = "cardinality"
    pool_size: Optional[int] = None
    budget_fraction: Optional[float] = None
    budget_absolute: Optional[int] = None
    extras: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_rounds != len(self.per_round):
            raise ValueError(
                f"per_round length {len(self.per_round)} does not match n_rounds {self.n_rounds}."
            )
        if sum(self.per_round) != self.total:
            raise ValueError(
                f"per_round sum {sum(self.per_round)} does not match total {self.total}."
            )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "BudgetPlan":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(**payload)


def compute_budget(
    pool_size: int,
    n_rounds: int,
    *,
    budget_fraction: float = 0.10,
    budget_absolute: Optional[int] = None,
    min_budget: int = 1,
) -> BudgetPlan:
    """Compute a per-round budget plan for active selection.

    Args:
        pool_size: Size of the candidate pool U.
        n_rounds: Number of active-learning rounds (>= 1).
        budget_fraction: Fraction of pool_size to allow as the extension.
        budget_absolute: Absolute budget; overrides budget_fraction when provided.
        min_budget: Floor for the total budget so at least one round runs.

    Returns:
        A BudgetPlan with total, per-round split, and metadata.
    """
    if n_rounds < 1:
        raise ValueError(f"n_rounds must be >= 1, got {n_rounds}.")
    if pool_size < 0:
        raise ValueError(f"pool_size must be >= 0, got {pool_size}.")

    if budget_absolute is not None:
        total = max(min_budget, int(budget_absolute))
    else:
        total = max(min_budget, int(budget_fraction * pool_size))
    total = min(total, max(0, pool_size))

    base = total // n_rounds
    per_round = [base] * n_rounds
    remainder = total - base * n_rounds
    if n_rounds > 0:
        per_round[-1] += remainder

    return BudgetPlan(
        total=total,
        n_rounds=n_rounds,
        per_round=per_round,
        cost_fn_name="cardinality",
        pool_size=pool_size,
        budget_fraction=budget_fraction,
        budget_absolute=budget_absolute,
    )

"""Text2SQL examples, prompts and meta-dataset workloads.

Every dataset is a JSON list of examples with at least `db_id`, `question`,
`sql`, and a `schema` ({"schema_items": [...], "foreign_keys": [...]}).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

SYSTEM = (
    "You are a Text-to-SQL expert. Given a SQLite database schema and a question, "
    "return one valid SQLite query that answers the question, with no explanation."
)


def load(path: str | Path, limit: int | None = None, seed: int = 0) -> List[dict]:
    data = json.loads(Path(path).read_text())
    if limit and len(data) > limit:
        keep = np.sort(np.random.default_rng(seed).choice(len(data), limit, replace=False))
        data = [data[i] for i in keep]
    return data


def schema_text(sample: dict, with_types: bool = True) -> str:
    schema = sample.get("schema") or {}
    lines = []
    for table in schema.get("schema_items", []):
        cols = table.get("column_names", [])
        types = table.get("column_types", [""] * len(cols))
        cols = [f"{c} {t}" if with_types and t else c for c, t in zip(cols, types)]
        lines.append(f"{table.get('table_name')}({', '.join(cols)})")
    for fk in schema.get("foreign_keys", []) if with_types else []:
        if len(fk) == 4:
            lines.append(f"{fk[0]}.{fk[1]} = {fk[2]}.{fk[3]}")
    return "\n".join(lines)


def _question(sample: dict) -> str:
    evidence = sample.get("evidence")
    return sample["question"] + (f"\nHint: {evidence}" if evidence else "")


def prompt(sample: dict) -> List[Dict[str, str]]:
    user = f"Database schema:\n{schema_text(sample)}\n\nQuestion: {_question(sample)}\n\nSQL:"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def serialize(sample: dict) -> str:
    """Input content fed to the frozen workload encoder."""
    return f"{_question(sample)}\n{schema_text(sample, with_types=False)}"


def make_workloads(pool: Sequence[dict], n: int, size: Sequence[int], dbs: Sequence[int], seed: int = 0) -> List[np.ndarray]:
    """Workloads of `size` examples drawn from `dbs` databases of the labeled pool."""
    rng = np.random.default_rng(seed)
    by_db: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(pool):
        by_db[s["db_id"]].append(i)
    db_ids = sorted(by_db)
    freq = np.array([len(by_db[d]) for d in db_ids], dtype=float)
    workloads = []
    for _ in range(n):
        k = min(int(rng.integers(dbs[0], dbs[1] + 1)), len(db_ids))
        chosen = rng.choice(len(db_ids), size=k, replace=False, p=freq / freq.sum())
        candidates = np.concatenate([by_db[db_ids[c]] for c in chosen])
        m = int(rng.integers(size[0], size[1] + 1))
        workloads.append(np.sort(rng.choice(candidates, size=min(m, len(candidates)), replace=False)))
    return workloads

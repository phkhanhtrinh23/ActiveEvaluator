"""SQLite execution, EX and EX-S (agreement on the original database and its variants)."""

from __future__ import annotations

import hashlib
import re
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

Result = Optional[Tuple]


def _canonical(rows: List[tuple], ordered: bool) -> Tuple:
    rows = [tuple("NULL" if v is None else repr(v) for v in row) for row in rows]
    return tuple(rows) if ordered else tuple(sorted(rows))


class Executor:
    def __init__(self, db_root: str | Path | None = None, variants_root: str | Path | None = None,
                 timeout: float = 30.0):
        self.db_root = Path(db_root) if db_root else None
        self.variants_root = Path(variants_root) if variants_root else None
        self.timeout = timeout
        self._conns: Dict[Path, sqlite3.Connection] = {}

    def databases(self, sample: dict) -> List[Path]:
        """Original database followed by its controlled variants."""
        db_id = sample.get("db_id", "")
        candidates = [self.db_root / db_id / f"{db_id}.sqlite"] if self.db_root else []
        if sample.get("db_path"):
            candidates.append(Path(sample["db_path"]))
        original = next((p for p in candidates if p.exists()), None)
        if original is None:
            raise FileNotFoundError(f"database for {db_id} not found")
        variants = sorted((self.variants_root / db_id).glob("*.sqlite")) if self.variants_root else []
        return [original, *variants]

    def _conn(self, path: Path) -> sqlite3.Connection:
        if path not in self._conns:
            conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True, check_same_thread=False)
            conn.text_factory = lambda b: b.decode(errors="replace")
            self._conns[path] = conn
        return self._conns[path]

    def execute(self, sql: str, db: Path, ordered: bool = False) -> Result:
        if not sql.strip():
            return None
        conn = self._conn(db)
        deadline = time.monotonic() + self.timeout
        conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)
        try:
            return _canonical(conn.execute(sql).fetchall(), ordered)
        except Exception:
            return None
        finally:
            conn.set_progress_handler(None, 0)

    def score(self, pred: str, gold: str, sample: dict) -> Tuple[bool, bool, int]:
        """(EX, EX-S, hash of the predicted result on the original database)."""
        ordered = bool(re.search(r"\border\s+by\b", gold, re.I))
        dbs = self.databases(sample)
        matches = []
        pred_hash = None
        for i, db in enumerate(dbs):
            p = self.execute(pred, db, ordered)
            if i == 0:
                pred_hash = result_hash(p, fallback=pred)
            matches.append(p is not None and p == self.execute(gold, db, ordered))
            if not matches[-1]:
                break
        return matches[0], all(matches) and len(matches) == len(dbs), pred_hash

    def close(self) -> None:
        for conn in self._conns.values():
            conn.close()
        self._conns.clear()


def result_hash(result: Result, fallback: str) -> int:
    """Stable 63-bit hash of an execution result; failed queries agree only on identical SQL."""
    key = repr(result) if result is not None else f"error::{fallback}"
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big") >> 1


def extract_sql(text: str) -> str:
    """First SQL statement in a model completion."""
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    if fenced:
        text = fenced.group(1)
    match = re.search(r"\bwith\s+\w+\s+as\s*\(.*|\bselect\b.*", text, re.S | re.I)
    sql = match.group(0) if match else text
    return sql.split(";")[0].strip()

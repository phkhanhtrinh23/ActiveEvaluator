import sqlite3

import numpy as np

from text2sql import data
from text2sql.execution import Executor, extract_sql

ROWS = [("Mia", "Data Science", 60), ("Noah", "Data Science", 78), ("Olivia", "Design", 36), ("Parker", "Biology", 108)]


def _db(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE student (name TEXT, program TEXT, credits INT)")
    conn.executemany("INSERT INTO student VALUES (?, ?, ?)", rows)
    conn.commit()
    conn.close()


def test_ex_s_rejects_spurious_match(tmp_path):
    """The student example of App. C: EX accepts the prediction, EX-S rejects it."""
    _db(tmp_path / "db" / "school" / "school.sqlite", ROWS)
    _db(tmp_path / "variants" / "school" / "v1.sqlite", ROWS + [("Quinn", "Data Science", 24)])
    _db(tmp_path / "variants" / "school" / "v2.sqlite", ROWS + [("Riley", "Design", 54)])
    ex = Executor(tmp_path / "db", tmp_path / "variants")
    sample = {"db_id": "school"}
    gold = "SELECT name FROM student WHERE program = 'Data Science'"
    pred = "SELECT name FROM student WHERE credits BETWEEN 45 AND 90"
    ex_match, exs_match, h = ex.score(pred, gold, sample)
    assert ex_match and not exs_match
    assert ex.score(gold, gold, sample)[:2] == (True, True)
    assert ex.score("SELECT nope", gold, sample)[:2] == (False, False)
    assert h == ex.score(gold, gold, sample)[2]  # same result on the original database


def test_extract_sql():
    assert extract_sql("```sql\nSELECT a FROM t;\n```") == "SELECT a FROM t"
    assert extract_sql("Answer: SELECT count(*) FROM singer; -- done") == "SELECT count(*) FROM singer"
    assert extract_sql("WITH x AS (SELECT 1) SELECT * FROM x").startswith("WITH x AS")


def test_workloads_are_deterministic_subsets():
    pool = [{"db_id": f"db{i % 7}"} for i in range(300)]
    a = data.make_workloads(pool, 20, (10, 30), (1, 3), seed=0)
    b = data.make_workloads(pool, 20, (10, 30), (1, 3), seed=0)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    for w in a:
        assert 10 <= len(w) <= 30 and len(set(w)) == len(w)
        assert len({pool[i]["db_id"] for i in w}) <= 3


def test_prompt_and_serialization():
    sample = {"question": "How many singers?", "evidence": "", "db_id": "x",
              "schema": {"schema_items": [{"table_name": "singer", "column_names": ["id", "name"],
                                           "column_types": ["int", "text"]}], "foreign_keys": []}}
    assert "singer(id int, name text)" in data.prompt(sample)[1]["content"]
    assert data.serialize(sample) == "How many singers?\nsinger(id, name)"

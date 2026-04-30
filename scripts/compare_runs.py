"""Compare baseline / V1 / V2 runs side-by-side.

Reads the three output directories produced by the orchestrator and emits:
- A summary table of meta-test MAE and real-test MAE for each method.
- Per-test-model deltas (predicted vs true accuracy) for each method.
- Active-selection cost / gain / time / source-model summary for V1 and V2.
- A wins/losses tally between V1 and V2.

Writes the comparison to outputs/comparison.json and prints a Markdown table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path("outputs")
RUNS = {
    "baseline": ROOT / "run_baseline",
    "v1_facility": ROOT / "run_active",
    "v2_direct": ROOT / "run_active_v2",
}


def _safe_load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _abs(x: float) -> float:
    return abs(float(x))


def collect(run_dir: Path) -> Dict[str, Any]:
    metrics = _safe_load(run_dir / "active_evaluator_metrics.json") or {}
    test_meta = _safe_load(run_dir / "active_evaluator_test_meta_predictions.json") or {}
    test_dev = _safe_load(run_dir / "active_evaluator_test_dev_predictions.json") or {}

    selection_summaries: Dict[str, Dict[str, Any]] = {}
    for sub in run_dir.glob("selection_*"):
        s = _safe_load(sub / "selection_summary.json")
        if s:
            selection_summaries[sub.name.replace("selection_", "")] = s

    return {
        "meta_test_mae_seen": metrics.get("meta_test_mae"),
        "real_test_mae_seen": metrics.get("real_test_mae"),
        "test_meta_mae_unseen": test_meta.get("mae"),
        "test_dev_mae_unseen": test_dev.get("mae"),
        "test_meta_records": test_meta.get("results", []),
        "test_dev_records": test_dev.get("results", []),
        "selection_summaries": selection_summaries,
    }


def md_table(rows: List[List[str]], header: List[str]) -> str:
    align = ["---"] * len(header)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    data = {name: collect(path) for name, path in RUNS.items()}

    # --- Top-line MAE table ---
    rows = []
    for name, d in data.items():
        rows.append([
            name,
            f"{d['meta_test_mae_seen']:.4f}" if d['meta_test_mae_seen'] is not None else "n/a",
            f"{d['real_test_mae_seen']:.4f}" if d['real_test_mae_seen'] is not None else "n/a",
            f"{d['test_meta_mae_unseen']:.4f}" if d['test_meta_mae_unseen'] is not None else "n/a",
            f"{d['test_dev_mae_unseen']:.4f}" if d['test_dev_mae_unseen'] is not None else "n/a",
        ])
    print("## Top-line MAE comparison\n")
    print(md_table(rows, ["Run", "Seen meta-test MAE", "Seen real-test MAE", "Unseen meta-test MAE", "Unseen real-test MAE"]))
    print()

    # --- Per-test-model deltas for the unseen real-test split ---
    by_model: Dict[str, Dict[str, Any]] = {}
    for run_name, d in data.items():
        for r in d["test_dev_records"]:
            m = r["model"]
            by_model.setdefault(m, {"true": r["true_accuracy"]})[run_name] = r["predicted_accuracy"]
            by_model[m][f"{run_name}_mae"] = r["mae"]

    print("## Per-test-model real-test predictions (unseen)\n")
    rows = []
    for model, vals in by_model.items():
        rows.append([
            model,
            f"{vals.get('true', 0):.3f}",
            f"{vals.get('baseline', float('nan')):.3f}" if 'baseline' in vals else "n/a",
            f"{vals.get('baseline_mae', float('nan')):.3f}" if 'baseline_mae' in vals else "n/a",
            f"{vals.get('v1_facility', float('nan')):.3f}" if 'v1_facility' in vals else "n/a",
            f"{vals.get('v1_facility_mae', float('nan')):.3f}" if 'v1_facility_mae' in vals else "n/a",
            f"{vals.get('v2_direct', float('nan')):.3f}" if 'v2_direct' in vals else "n/a",
            f"{vals.get('v2_direct_mae', float('nan')):.3f}" if 'v2_direct_mae' in vals else "n/a",
        ])
    print(md_table(rows, ["Model", "True", "BL pred", "BL MAE", "V1 pred", "V1 MAE", "V2 pred", "V2 MAE"]))
    print()

    # --- V1 vs V2 win/loss/tie on per-model MAE ---
    v1_v2 = []
    v1_wins = v2_wins = ties = 0
    for model, vals in by_model.items():
        if "v1_facility_mae" in vals and "v2_direct_mae" in vals:
            d_v1 = vals["v1_facility_mae"]
            d_v2 = vals["v2_direct_mae"]
            if abs(d_v1 - d_v2) < 1e-9:
                ties += 1
                winner = "tie"
            elif d_v1 < d_v2:
                v1_wins += 1
                winner = "V1"
            else:
                v2_wins += 1
                winner = "V2"
            v1_v2.append((model, d_v1, d_v2, winner))

    print("## V1 vs V2 per-model winner (smaller real-test MAE wins)\n")
    rows = [[m, f"{a:.4f}", f"{b:.4f}", w] for m, a, b, w in v1_v2]
    print(md_table(rows, ["Model", "V1 MAE", "V2 MAE", "Winner"]))
    print(f"\n**V1 wins: {v1_wins} | V2 wins: {v2_wins} | Ties: {ties}**\n")

    # --- Selection cost / gain / time per run, averaged across test models ---
    print("## Selection economics (per test model, averaged)\n")
    rows = []
    for run_name in ("v1_facility", "v2_direct"):
        ss = data[run_name]["selection_summaries"]
        if not ss:
            rows.append([run_name, "n/a", "n/a", "n/a", "n/a", "n/a", "n/a"])
            continue
        n = len(ss)
        budget = sum(s.get("budget_total", 0) for s in ss.values()) / n
        cost = sum(s.get("cost_paid_total", 0) for s in ss.values()) / n
        gain = sum(s.get("gain_total", 0) for s in ss.values()) / n
        n_sel = sum(s.get("n_selected", 0) for s in ss.values()) / n
        time_s = sum(s.get("total_selection_seconds", 0) for s in ss.values()) / n
        final_loss = sum(s.get("final_val_loss") or 0 for s in ss.values()) / n
        rows.append([run_name, f"{budget:.1f}", f"{cost:.1f}", f"{n_sel:.1f}", f"{gain:.3f}", f"{final_loss:.4f}", f"{time_s:.2f}s"])
    print(md_table(rows, ["Run", "Budget", "Cost paid", "n picked", "Gain total", "Final val loss", "Sel. time"]))
    print()

    # --- Source-model frequency for V1 and V2 ---
    print("## Most-picked training models (across all test models)\n")
    for run_name in ("v1_facility", "v2_direct"):
        ss = data[run_name]["selection_summaries"]
        if not ss:
            print(f"### {run_name}: no selection logs"); continue
        freq: Dict[str, int] = {}
        for s in ss.values():
            for m in s.get("source_models_chosen", []):
                freq[m] = freq.get(m, 0) + 1
        ordered = sorted(freq.items(), key=lambda kv: -kv[1])
        print(f"### {run_name}\n")
        rows = [[m, str(c)] for m, c in ordered]
        print(md_table(rows, ["Source training model", "Times picked"]))
        print()

    # --- Persist the full comparison ---
    out = ROOT / "comparison.json"
    out.write_text(json.dumps({
        "summary": {
            name: {k: v for k, v in d.items() if k != "selection_summaries"}
            for name, d in data.items()
        },
        "v1_vs_v2": {"wins_v1": v1_wins, "wins_v2": v2_wins, "ties": ties},
        "selection_summaries": {
            name: data[name]["selection_summaries"] for name in RUNS
        },
    }, indent=2), encoding="utf-8")
    print(f"\nFull comparison written to {out}")


if __name__ == "__main__":
    main()

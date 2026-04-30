#!/bin/bash
set -e
echo "=== Waiting for baseline (PID 623624) to finish ==="
while kill -0 623624 2>/dev/null; do sleep 30; done

echo "=== Launching V1 ==="
bash logs/launch_active.sh > logs/v1.log 2>&1
echo "V1 done"

echo "=== Launching V2 ==="
bash logs/launch_v2.sh > logs/v2.log 2>&1
echo "V2 done"

echo "=== ALL DONE ==="
python scripts/compare_runs.py | tee logs/comparison.md
echo
echo "Comparison artifacts:"
echo "  - logs/comparison.md  (Markdown tables, copy/paste-able)"
echo "  - outputs/comparison.json  (full structured payload)"

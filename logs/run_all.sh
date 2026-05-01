#!/bin/bash
set -e
echo "=== [$(date +%H:%M:%S)] Launching baseline ==="
bash logs/launch_baseline.sh > logs/baseline_v2.log 2>&1
echo "=== [$(date +%H:%M:%S)] Baseline done ==="

echo "=== [$(date +%H:%M:%S)] Launching V1 ==="
bash logs/launch_v1.sh > logs/v1.log 2>&1
echo "=== [$(date +%H:%M:%S)] V1 done ==="

echo "=== [$(date +%H:%M:%S)] Launching V2 ==="
bash logs/launch_v2.sh > logs/v2.log 2>&1
echo "=== [$(date +%H:%M:%S)] V2 done ==="

echo "=== [$(date +%H:%M:%S)] ALL DONE ==="
python scripts/compare_runs.py | tee logs/comparison.md
echo
echo "Comparison artifacts:"
echo "  - logs/comparison.md  (Markdown tables)"
echo "  - outputs/comparison.json  (structured payload)"

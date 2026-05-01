#!/bin/bash
# Extended-scale benchmark: 12 train + 6 test models = 18 total.
#   Stage 1: baseline (handles new model generation; cached models skipped).
#   Stage 2: V1 facility location.
#   Stage 3: V2 direct loss (with weight-decay + early-stop improvements).
#   Stage 4: V3 GRAD-MATCH (Killamsetty et al. 2021).
#   Stage 5: write side-by-side comparison to logs/comparison.md.
#
# Detached: launch with `nohup bash logs/run_extended.sh > logs/extended.log 2>&1 &`.
# Survives terminal close because it's reparented to init.

set -u  # unset vars are errors; do NOT use -e because we want to keep going on per-stage failures

cd "$(dirname "$0")/.."  # repo root regardless of where we're invoked from

LOG=logs/extended.log
mark() { echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] $1 ===" | tee -a "$LOG"; }

mark "starting extended benchmark"
mark "GPU at start: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"
mark "disk at start: $(df -h / | tail -1 | awk '{print $4}') free on /"

# Stage 1: baseline (does new generation + meta-train + held-out eval)
mark "stage 1/5: baseline"
bash logs/launch_baseline.sh > logs/extended_baseline.log 2>&1
RC1=$?
mark "stage 1 exit code: $RC1"
if [ $RC1 -ne 0 ]; then
  mark "BASELINE FAILED — see logs/extended_baseline.log; abort"
  exit 1
fi

# Stage 2: V1 (reuses prediction cache from baseline)
mark "stage 2/5: V1 facility location"
bash logs/launch_v1.sh > logs/extended_v1.log 2>&1
RC2=$?
mark "stage 2 exit code: $RC2"

# Stage 3: V2
mark "stage 3/5: V2 direct loss"
bash logs/launch_v2.sh > logs/extended_v2.log 2>&1
RC3=$?
mark "stage 3 exit code: $RC3"

# Stage 4: V3
mark "stage 4/5: V3 GRAD-MATCH"
bash logs/launch_v3.sh > logs/extended_v3.log 2>&1
RC4=$?
mark "stage 4 exit code: $RC4"

# Stage 5: comparison (works even if some stages failed; missing rows show 'n/a')
mark "stage 5/5: comparison"
python scripts/compare_runs.py | tee logs/comparison.md
mark "comparison done"

mark "ALL STAGES COMPLETE   baseline=$RC1 v1=$RC2 v2=$RC3 v3=$RC4"
echo
echo "Read results with:   cat logs/comparison.md"
echo "Per-stage logs in:   logs/extended_{baseline,v1,v2,v3}.log"

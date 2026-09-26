#!/usr/bin/env bash
# End-to-end reproduction: tests, selection, model records, experiments and ablations.
#
#   bash scripts/run_all.sh               # smoke run: tests + a reduced run per task
#   MODE=full bash scripts/run_all.sh     # paper-scale run
set -euo pipefail
cd "$(dirname "$0")/.."

MODE=${MODE:-smoke}                  # smoke | full
TASKS=${TASKS:-"text2sql image node"}
PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-}                   # cuda, cuda:0, cpu (default: auto)
CONFIG_DIR=${CONFIG_DIR:-configs}
OUT_ROOT=${OUT_ROOT:-outputs}
RUNS=${RUNS:-10}                     # repeated runs with 95% CIs
POOL_SIZE=${POOL_SIZE:-10}           # unseen-model pool of Tab. 1
POOL_SIZES=${POOL_SIZES:-"5 10 15 20 25 30"}
SWEEP=${SWEEP:-"1000 3000 6000 9000 15000 20000 25000"}
EPOCHS=${EPOCHS:-2000}
BASELINES=${BASELINES:-1}
RUN_ABLATIONS=${RUN_ABLATIONS:-1}    # Tab. 2: meta-learning algorithms and workload distances
RUN_TESTS=${RUN_TESTS:-1}
MAX_POINTS=${MAX_POINTS:-}           # optional cap on points per workload embedding set
SMOKE_WORKLOADS=${SMOKE_WORKLOADS:-200}
SMOKE_BUDGET=${SMOKE_BUDGET:-60}
SMOKE_MODELS=${SMOKE_MODELS:-6}

DISTANCES="sliced_wasserstein frechet chamfer mahalanobis euclidean"
ALGORITHMS="cavia maml fomaml reptile metasgd anil protonet"

budget() {
  case $1 in
    text2sql) echo "${K_TEXT2SQL:-9000}" ;;
    image) echo "${K_IMAGE:-8000}" ;;
    node) echo "${K_NODE:-10000}" ;;
  esac
}

if [[ "$RUN_TESTS" == 1 ]]; then
  "$PYTHON" -m pytest -q tests
fi

for task in $TASKS; do
  out="$OUT_ROOT/$task"
  mkdir -p "$out"
  config="$CONFIG_DIR/$task.json"
  k=$(budget "$task")
  pool=$POOL_SIZE
  pools=$POOL_SIZES
  sweep=$(printf "%s\n" $SWEEP "$k" | sort -nu | tr "\n" " ")
  runs=$RUNS
  epochs=$EPOCHS

  if [[ "$MODE" == smoke ]]; then
    config="$out/config_smoke.json"
    "$PYTHON" - "$CONFIG_DIR/$task.json" "$out/config_smoke.json" "$out" "$SMOKE_WORKLOADS" "$SMOKE_MODELS" <<'EOF'
import json, sys
src, path, out, n, m = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
cfg = json.load(open(src))
cfg["out"], cfg["workloads"]["n"], cfg["models"] = out, n, cfg["models"][:m]
json.dump(cfg, open(path, "w"), indent=2)
EOF
    k=$SMOKE_BUDGET
    pool=3
    pools="3 4"
    sweep="$((k / 3)) $k"
    runs=2
    epochs=200
  fi

  opts="--config $config --out $out${DEVICE:+ --device $DEVICE}"
  cap=${MAX_POINTS:+--max-points $MAX_POINTS}
  exp="--runs $runs --epochs $epochs"
  base=""
  [[ "$BASELINES" == 1 ]] && base="--baselines"
  run() { "$PYTHON" -m "$task.run" "$@" $opts; }

  echo "== [$task] representative selection"
  run select --budgets $sweep $cap
  selections="$out/selection_hausdorff.npz"
  if [[ "$RUN_ABLATIONS" == 1 ]]; then
    for metric in $DISTANCES; do
      run select --metric "$metric" --budgets "$k" $cap
      selections="$selections $out/selection_$metric.npz"
    done
  fi

  echo "== [$task] model records on the selected workloads and on the full meta-dataset"
  run records --selection $selections
  run records --full

  echo "== [$task] Tab. 1 and Fig. 4: MAE over pools of $pool unseen models"
  run experiment --budgets "$k" --full --pool-sizes "$pool" $base $exp --name results_main.json

  echo "== [$task] Fig. 5: ranking quality versus pool size"
  run experiment --budgets "$k" --full --pool-sizes $pools $base $exp --name results_pool_size.json

  echo "== [$task] Fig. 6: latency and MAE versus meta-dataset size"
  run experiment --budgets $sweep --full --pool-sizes "$pool" $exp --name results_budget.json

  if [[ "$RUN_ABLATIONS" == 1 ]]; then
    echo "== [$task] Tab. 2 left: meta-learning algorithm"
    run experiment --budgets "$k" --full --algorithms $ALGORITHMS --pool-sizes "$pool" $exp \
      --name results_meta_algorithm.json

    echo "== [$task] Tab. 2 right: workload distance"
    run experiment --budgets "$k" --selection $selections --pool-sizes "$pool" $exp --name results_distance.json
  fi
done

echo "Results are in $OUT_ROOT/<task>/results_*.json"

#!/usr/bin/env bash
# 保存为 run_experiments.sh
# 用法示例: ./run_experiments.sh demo
# 该脚本将读取顶部 BASE_ARGS, 并通过 run_one 来启动实验。

set -euo pipefail
IFS=$'\n\t'

# ---------- global configuration (edit here) ----------
PYTHON=${PYTHON:-python}
DATA_DIR=${DATA_DIR:-/mnt/e/code/SGFormer/data}
METHOD=${METHOD:-ours}
DATASET=${DATASET:-cora}
DEVICE=${DEVICE:-0}           # GPU id; main.py will fallback to CPU if unavailable
CPU_FLAG=${CPU_FLAG:-}        # set to "--cpu" if you want CPU runs
EPOCHS=${EPOCHS:-100}
RUNS_PER_SEED=${RUNS_PER_SEED:-1}
DISPLAY_STEP=${DISPLAY_STEP:-10}
LR=${LR:-0.01}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0005}

# Consistency / EMA defaults (can be overridden per-experiment)
CONS_WARM=${CONS_WARM:-10}
CONS_UP=${CONS_UP:-40}
CONS_WEIGHT=${CONS_WEIGHT:-0.1}
CONS_TEMP=${CONS_TEMP:-1.0}
EMA_TAU=${EMA_TAU:-0.99}
EMA_START=${EMA_START:-5}
CONS_LOSS=${CONS_LOSS:-prob_mse}

# NodeMixup / CI-GCL defaults
USE_NODE_MIXUP=${USE_NODE_MIXUP:-true}
MIXUP_ALPHA=${MIXUP_ALPHA:-0.3}
USE_CI_GCL=${USE_CI_GCL:-true}
AUX_WEIGHT=${AUX_WEIGHT:-0.05}
CI_DROP_EDGE=${CI_DROP_EDGE:-0.2}
CI_FEAT_MASK=${CI_FEAT_MASK:-0.1}

# Output folders
RESULTS_DIR=${RESULTS_DIR:-results}
JSON_DIR="${RESULTS_DIR}/jsons"
mkdir -p "${RESULTS_DIR}"
mkdir -p "${JSON_DIR}"

# Base args array (keeps common params in one place)
BASE_ARGS=(
  --dataset "${DATASET}"
  --method "${METHOD}"
  --data_dir "${DATA_DIR}"
  --epochs "${EPOCHS}"
  --runs "${RUNS_PER_SEED}"
  ${CPU_FLAG}
  --display_step "${DISPLAY_STEP}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --cons_warm "${CONS_WARM}"
  --cons_up "${CONS_UP}"
  --cons_weight "${CONS_WEIGHT}"
  --cons_temp "${CONS_TEMP}"
  --ema_tau "${EMA_TAU}"
  --ema_start "${EMA_START}"
  --cons_loss "${CONS_LOSS}"
  # NodeMixup / CI-GCL: enable/disable via extra args or use defaults below
)

# Helper: run a single experiment if not already present in results csv
# args: exp_tag seed run extra_args_array...
run_one() {
  local exp_tag="$1"
  local seed="$2"
  local run_idx="$3"
  shift 3
  local extra_args=("$@")

  local csv_path="${RESULTS_DIR}/${DATASET}_${METHOD}_results_per_run.csv"
  # Ensure CSV exists with header
  if [ ! -f "${csv_path}" ]; then
    echo "exp_tag,seed,run,best_val,best_test" > "${csv_path}"
  fi

  # check if this exp_tag,seed,run already exists
  if grep -E -q "^${exp_tag},${seed},${run_idx}," "${csv_path}"; then
    echo "[SKIP] ${exp_tag} seed=${seed} run=${run_idx} already exists in ${csv_path}"
    return 0
  fi

  # ensure out json path unique
  local out_json="${JSON_DIR}/${exp_tag}_seed${seed}_run${run_idx}.json"
  mkdir -p "$(dirname "${out_json}")"

  # construct and run command
  local cmd=( "${PYTHON}" main.py "${BASE_ARGS[@]}" --exp_tag "${exp_tag}" --seed "${seed}" --out_file "${out_json}" )
  # append extra args
  if [ "${#extra_args[@]}" -ne 0 ]; then
    cmd+=( "${extra_args[@]}" )
  fi

  echo "[RUN] ${cmd[*]}"
  "${cmd[@]}"
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[ERROR] command failed with code $rc"
    return $rc
  fi
  echo "[DONE] ${exp_tag} seed=${seed} run=${run_idx}"
}

# ---------- example experiment definitions ----------
# You can add experiment groups here and call run_one in loops
experiment_demo() {
  # demo: weakened baseline (fast) vs full A+B+D+E (long)
  local seeds=(100 101 102 103 104 105 106 107 108 109)

  # baseline_bad: weakened hyperparams (very short run)
  for s in "${seeds[@]}"; do
    run_one "baseline_bad" "${s}" 1 \
      --epochs 1
  done

  # A+B+D+E full (use NodeMixup + CI-GCL)
  for s in "${seeds[@]}"; do
    run_one "abdde_full" "${s}" 1 \
      --use_node_mixup --mixup_alpha "${MIXUP_ALPHA}" \
      --use_ci_gcl --aux_weight "${AUX_WEIGHT}" --ci_drop_edge "${CI_DROP_EDGE}" --ci_feat_mask "${CI_FEAT_MASK}" \
      --save_checkpoints
  done
}

# Another example: run an ablation sweep of combinations
experiment_ablation() {
  local seeds=(200 201 202)
  # A, A+B, A+D, A+E, A+B+D, A+B+E, A+B+D+E
  for s in "${seeds[@]}"; do
    # A only
    run_one "A_only" "${s}" 1 --cons_weight 0.0

    # A+B (mixup only)
    run_one "A_plus_B" "${s}" 1 --use_node_mixup --mixup_alpha "${MIXUP_ALPHA}" --cons_weight 0.0

    # A+D (consistency only)
    run_one "A_plus_D" "${s}" 1 --cons_weight "${CONS_WEIGHT}" --ema_tau "${EMA_TAU}"

    # A+E (aux only)
    run_one "A_plus_E" "${s}" 1 --use_ci_gcl --aux_weight "${AUX_WEIGHT}" --cons_weight 0.0

    # A+B+D
    run_one "A_B_D" "${s}" 1 --use_node_mixup --mixup_alpha "${MIXUP_ALPHA}" --cons_weight "${CONS_WEIGHT}" --ema_tau "${EMA_TAU}"

    # A+B+E
    run_one "A_B_E" "${s}" 1 --use_node_mixup --mixup_alpha "${MIXUP_ALPHA}" --use_ci_gcl --aux_weight "${AUX_WEIGHT}" --cons_weight 0.0

    # A+B+D+E
    run_one "A_B_D_E" "${s}" 1 --use_node_mixup --mixup_alpha "${MIXUP_ALPHA}" --use_ci_gcl --aux_weight "${AUX_WEIGHT}" --cons_weight "${CONS_WEIGHT}" --ema_tau "${EMA_TAU}"
  done
}

# ---------- main entry ----------
case "${1:-}" in
  demo)
    experiment_demo
    ;;
  ablation)
    experiment_ablation
    ;;
  one)
    # run a single custom run, example:
    # ./run_experiments.sh one EXP_TAG SEED RUN --use_node_mixup --mixup_alpha 0.3
    if [ $# -lt 4 ]; then
      echo "Usage: $0 one EXP_TAG SEED RUN [--extra flags]"
      exit 2
    fi
    EXP_TAG="$2"
    SEED="$3"
    RUNIDX="$4"
    shift 4
    run_one "${EXP_TAG}" "${SEED}" "${RUNIDX}" "$@"
    ;;
  *)
    echo "Usage: $0 {demo|ablation|one ...}"
    exit 1
    ;;
esac

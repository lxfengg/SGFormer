#!/usr/bin/env bash
# 保存为 run_s_vs_no_s.sh
# 用法: ./run_s_vs_no_s.sh
set -euo pipefail
IFS=$'\n\t'

PYTHON=${PYTHON:-python}
DATASET=${DATASET:-cora}
METHOD=${METHOD:-ours}
DATA_DIR=${DATA_DIR:-/mnt/d/File/code/SGFormer/data}
EPOCHS=${EPOCHS:-200}
DISPLAY_STEP=${DISPLAY_STEP:-10}
RUNS_PER_SEED=${RUNS_PER_SEED:-1}
SEEDS=(100 101 102 103 104)   # adjust / extend as needed

# base args shared
BASE_ARGS=(
  --dataset "${DATASET}"
  --method "${METHOD}"
  --data_dir "${DATA_DIR}"
  --epochs "${EPOCHS}"
  --runs "${RUNS_PER_SEED}"
  --display_step "${DISPLAY_STEP}"
)

# function to run one cmd
run_once() {
  exp_tag=$1
  seed=$2
  extra_args=("${@:3}")
  csv="results/${DATASET}_${METHOD}_results_per_run.csv"
  mkdir -p results/checkpoints results/epoch_logs
  if grep -E -q "^${exp_tag},${seed}," "${csv}" 2>/dev/null; then
    echo "[SKIP] ${exp_tag}, seed ${seed} already present in ${csv}"
    return
  fi
  cmd=( "${PYTHON}" main.py "${BASE_ARGS[@]}" --exp_tag "${exp_tag}" --seed "${seed}" --save_checkpoints "${extra_args[@]}" )
  echo "[RUN] ${cmd[*]}"
  "${cmd[@]}"
}

for s in "${SEEDS[@]}"; do
  # A_fair: no consistency, no mixup, no ci_gcl
  run_once "A_fair" "${s}" --cons_weight 0.0

  # A_plus_S: only consistency (S-channel) enabled
  run_once "A_plus_S" "${s}" --cons_weight 0.1 --cons_warm 10 --cons_up 40 --cons_loss prob_mse --cons_confidence 0.0 --ema_tau 0.99
done
 
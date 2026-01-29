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
SEEDS=(100 101 102 103 104)

# CLEAN_RESULTS behavior: delete (default) | backup | none
CLEAN_RESULTS=${CLEAN_RESULTS:-delete}

BASE_ARGS=(
  --dataset "${DATASET}"
  --method "${METHOD}"
  --data_dir "${DATA_DIR}"
  --epochs "${EPOCHS}"
  --runs "${RUNS_PER_SEED}"
  --display_step "${DISPLAY_STEP}"
)

# perform cleaning according to CLEAN_RESULTS
if [ "${CLEAN_RESULTS}" = "delete" ]; then
  if [ -d "results" ]; then
    echo "[CLEAN] Deleting existing results/ (irreversible)"
    rm -rf results
  fi
elif [ "${CLEAN_RESULTS}" = "backup" ]; then
  if [ -d "results" ]; then
    BACK="results_backup_$(date +%Y%m%d-%H%M%S)"
    echo "[CLEAN] Backing up existing results -> ${BACK}"
    mv results "${BACK}"
  fi
else
  echo "[CLEAN] CLEAN_RESULTS=${CLEAN_RESULTS}: leaving results/ as-is"
fi

mkdir -p results/checkpoints results/epoch_logs results/logs
csv="results/${DATASET}_${METHOD}_results_per_run.csv"
if [ ! -f "${csv}" ]; then
  echo "exp_tag,seed,run,best_val,best_test" > "${csv}"
  echo "[INFO] Created CSV header at ${csv}"
fi

run_once() {
  exp_tag=$1
  seed=$2
  extra_args=("${@:3}")
  if grep -E -q "^${exp_tag},${seed}," "${csv}" 2>/dev/null; then
    echo "[SKIP] ${exp_tag}, seed ${seed} already present in ${csv}"
    return
  fi
  cmd=( "${PYTHON}" main.py "${BASE_ARGS[@]}" --exp_tag "${exp_tag}" --seed "${seed}" --save_checkpoints "${extra_args[@]}" )
  logfile="results/logs/${DATASET}_${METHOD}_${exp_tag}_seed${seed}.log"
  echo "[RUN] ${cmd[*]}"
  "${cmd[@]}" 2>&1 | tee "${logfile}"
}

for s in "${SEEDS[@]}"; do
  run_once "A_fair" "${s}" --cons_weight 0.0
  run_once "A_plus_S" "${s}" --cons_weight 0.1 --cons_warm 10 --cons_up 40 --cons_loss prob_mse --cons_confidence 0.0 --ema_tau 0.99
done

echo "=== Experiments complete. Starting automatic analysis ==="

# call the external analysis script
python scripts/analyze_pairwise_postrun.py --csv "${csv}" --tagA A_fair --tagB A_plus_S --metric best_test --out_dir results/figs_pair

echo "=== Analysis complete. Check results/figs_pair for figures and JSON summary. ==="

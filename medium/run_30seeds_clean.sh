#!/usr/bin/env bash
# 保存为 run_30seeds_clean.sh
# 一键跑 30 seeds 的 A_fair vs A_plus_S 对比，并保存 checkpoints
# 用法:
#   ./run_30seeds_clean.sh
# 可选环境变量:
#   CLEAN_RESULTS=delete|backup|skip   (default: delete)
#   PYTHON=python3
#   DATA_DIR=/path/to/data
#   EPOCHS=200
#   SEED_START / SEED_END (default 100..129)
set -euo pipefail
IFS=$'\n\t'

PYTHON=${PYTHON:-python}
DATASET=${DATASET:-cora}
METHOD=${METHOD:-ours}
DATA_DIR=${DATA_DIR:-/mnt/d/File/code/SGFormer/data}
EPOCHS=${EPOCHS:-200}
DISPLAY_STEP=${DISPLAY_STEP:-10}
RUNS_PER_SEED=${RUNS_PER_SEED:-1}
SEED_START=${SEED_START:-100}
SEED_END=${SEED_END:-129}
CLEAN=${CLEAN_RESULTS:-delete}   # delete | backup | skip

echo "[INFO] run_30seeds_clean.sh starting. CLEAN_RESULTS=${CLEAN}"
timestamp() { date +%Y%m%d-%H%M%S; }

if [ "${CLEAN}" = "backup" ]; then
  if [ -d results ]; then
    mv results results_backup_$(timestamp)
    echo "[INFO] moved existing results/ -> results_backup_$(timestamp)"
  fi
elif [ "${CLEAN}" = "delete" ]; then
  echo "[CLEAN] Deleting existing result artifacts (irreversible):"
  rm -rf results/figs_pair results/logs results/checkpoints results/epoch_logs results/diagnosis_per_ckpt results/diagnosis_all_checkpoints_python* results/*.csv || true
  echo "[DONE] deleted old results/"
else
  echo "[INFO] CLEAN_RESULTS=skip: keeping existing results/"
fi

mkdir -p results/logs results/checkpoints results/epoch_logs results/figs_pair results/diagnosis_per_ckpt

BASE_ARGS=(
  --dataset "${DATASET}"
  --method "${METHOD}"
  --data_dir "${DATA_DIR}"
  --epochs "${EPOCHS}"
  --runs "${RUNS_PER_SEED}"
  --display_step "${DISPLAY_STEP}"
  --save_checkpoints
)

echo "[INFO] Running seeds ${SEED_START}..${SEED_END} for dataset=${DATASET}, method=${METHOD}"
for s in $(seq ${SEED_START} ${SEED_END}); do
  echo "------------------------------------------------------------"
  echo "[RUN] Seed ${s} - A_fair (no S)"
  LOGFN="results/logs/${DATASET}_${METHOD}_A_fair_seed${s}.log"
  "${PYTHON}" main.py "${BASE_ARGS[@]}" --exp_tag A_fair --seed "${s}" --cons_weight 0.0 2>&1 | tee "${LOGFN}"

  echo "[RUN] Seed ${s} - A_plus_S (with S channel)"
  LOGFN="results/logs/${DATASET}_${METHOD}_A_plus_S_seed${s}.log"
  "${PYTHON}" main.py "${BASE_ARGS[@]}" --exp_tag A_plus_S --seed "${s}" \
    --cons_weight 0.1 --cons_warm 10 --cons_up 40 --cons_loss prob_mse --cons_confidence 0.0 --ema_tau 0.99 2>&1 | tee "${LOGFN}"
done

echo "[INFO] All experiments finished for seeds ${SEED_START}..${SEED_END}."

# Automatic analysis (if scripts available)
if [ -f scripts/analyze_pairwise_postrun.py ]; then
  echo "[ANALYSIS] Running pairwise analysis..."
  "${PYTHON}" scripts/analyze_pairwise_postrun.py --csv results/${DATASET}_${METHOD}_results_per_run.csv --tagA A_fair --tagB A_plus_S --metric best_test --out_dir results/figs_pair || echo "[WARN] pairwise analysis failed"
else
  echo "[ANALYSIS] scripts/analyze_pairwise_postrun.py not found; skipping pairwise analysis."
fi

if [ -f scripts/bootstrap_and_tests.py ]; then
  echo "[ANALYSIS] Running bootstrap robustness test..."
  "${PYTHON}" scripts/bootstrap_and_tests.py --rows results/figs_pair/pairwise_rows_A_fair_vs_A_plus_S_best_test.csv --n_boot 10000 || echo "[WARN] bootstrap failed"
else
  echo "[ANALYSIS] scripts/bootstrap_and_tests.py not found; skipping bootstrap."
fi

echo "[DONE] Full run completed. Results: results/, Figures: results/figs_pair/"

#!/usr/bin/env bash
# 保存为 run_clean_and_rerun.sh
# 用途：备份旧 results -> 清理 -> 依次跑 A_fair 与 A_plus_S (seeds list) -> 运行诊断脚本
# 用法: ./run_clean_and_rerun.sh [DATA_DIR] [DATASET]
# 例: ./run_clean_and_rerun.sh /mnt/d/File/code/SGFormer/data cora

set -euo pipefail
IFS=$'\n\t'

PYTHON=${PYTHON:-python}
DATA_DIR_ARG=${1:-""}
DATASET=${2:-cora}

# seeds & hyperparams（按需改）
SEEDS=(100 101 102 103 104)
EPOCHS=${EPOCHS:-200}
DISPLAY_STEP=${DISPLAY_STEP:-10}
METHOD=${METHOD:-ours}

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
ROOT="$(pwd)"

# Resolve data dir: prefer arg, else common candidates
if [ -n "${DATA_DIR_ARG}" ]; then
  DATA_DIR="${DATA_DIR_ARG}"
else
  CANDIDATES=(
    "${ROOT}/data"
    "/mnt/e/code/SGFormer/data"
    "/mnt/d/File/code/SGFormer/data"
    "/mnt/d/file/code/SGFormer/data"
  )
  DATA_DIR=""
  for d in "${CANDIDATES[@]}"; do
    if [ -d "${d}/Planetoid/raw" ]; then
      DATA_DIR="${d}"
      break
    fi
  done
  if [ -z "${DATA_DIR}" ]; then
    echo "[ERROR] Could not locate Planetoid/raw. Provide data_dir as first argument."
    exit 1
  fi
fi

echo "[INFO] Using data dir: ${DATA_DIR}"
echo "[INFO] Dataset: ${DATASET}"
echo "[INFO] Seeds: ${SEEDS[*]}, epochs=${EPOCHS}"

# backup & clear old results
if [ -d results ]; then
  BACKUP="results_backup_${TIMESTAMP}"
  echo "[INFO] Backing up existing results -> ${BACKUP}"
  mv results "${BACKUP}"
fi

# recreate directories
mkdir -p results/epoch_logs results/checkpoints results/logs

# helper to run a single experiment and log
run_exp() {
  local exp_tag=$1
  local seed=$2
  local extra_args=("${@:3}")
  local logf="results/logs/${DATASET}_${METHOD}_${exp_tag}_seed${seed}.log"
  echo "[RUN] exp=${exp_tag} seed=${seed} -> log=${logf}"
  # command
  cmd=( "${PYTHON}" main.py \
    --dataset "${DATASET}" \
    --method "${METHOD}" \
    --data_dir "${DATA_DIR}" \
    --seed "${seed}" \
    --exp_tag "${exp_tag}" \
    --epochs "${EPOCHS}" \
    --runs 1 \
    --display_step "${DISPLAY_STEP}" \
    --save_checkpoints "${extra_args[@]}" )
  # print cmd to log and run
  echo "CMD: ${cmd[*]}" > "${logf}"
  "${cmd[@]}" >> "${logf}" 2>&1 || { echo "[WARN] Command failed for ${exp_tag} seed ${seed} (see ${logf})"; }
  echo "[DONE] ${exp_tag} seed ${seed} (log: ${logf})"
}

# main loop: for each seed run A_fair then A_plus_S
for s in "${SEEDS[@]}"; do
  # A_fair: consistency off
  run_exp "A_fair" "${s}" --cons_weight 0.0

  # A_plus_S: enable S (consistency) only
  run_exp "A_plus_S" "${s}" --cons_weight 0.1 --cons_warm 10 --cons_up 40 --cons_loss prob_mse --cons_confidence 0.0 --ema_tau 0.99
done

# After runs: run diagnostics (append to a single diagnosis log)
DIAG_LOG="results/logs/diagnosis.txt"
echo "=== Diagnosis run at $(date) ===" > "${DIAG_LOG}"

if [ -f analyze_teacher_coverage.py ]; then
  echo "---- analyze_teacher_coverage.py output ----" >> "${DIAG_LOG}"
  python analyze_teacher_coverage.py >> "${DIAG_LOG}" 2>&1 || echo "[WARN] analyze script failed" >> "${DIAG_LOG}"
else
  echo "[WARN] analyze_teacher_coverage.py not found" >> "${DIAG_LOG}"
fi

if [ -f reachability.py ]; then
  echo "---- reachability.py output ----" >> "${DIAG_LOG}"
  python reachability.py >> "${DIAG_LOG}" 2>&1 || echo "[WARN] reachability failed" >> "${DIAG_LOG}"
else
  echo "[WARN] reachability.py not found" >> "${DIAG_LOG}"
fi

# try evaluate checkpoint for last A_plus_S seed
CKPT="results/checkpoints/${DATASET}_A_plus_S_seed${SEEDS[-1]}_run0_best.pth"
if [ -f "${CKPT}" ] && [ -f eval_checkpoint_predictions_fixed.py ]; then
  echo "---- eval_checkpoint_predictions_fixed.py output (ckpt: ${CKPT}) ----" >> "${DIAG_LOG}"
  python eval_checkpoint_predictions_fixed.py --ckpt "${CKPT}" --dataset "${DATASET}" >> "${DIAG_LOG}" 2>&1 || echo "[WARN] eval_checkpoint_predictions_fixed failed" >> "${DIAG_LOG}"
else
  echo "[WARN] checkpoint ${CKPT} or eval script not found; skipping checkpoint eval" >> "${DIAG_LOG}"
fi

echo "[INFO] All done. Logs: results/logs/, epoch logs: results/epoch_logs/, checkpoints: results/checkpoints/"
echo "[INFO] Diagnosis written to ${DIAG_LOG}"

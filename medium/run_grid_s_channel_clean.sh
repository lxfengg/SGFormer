#!/usr/bin/env bash
# 保存为 run_grid_s_channel_clean.sh
# 一键做 S-channel 超参网格搜索。默认会删除 results/ 下旧数据（以避免脏数据）。
# 如不想删除，请设置 CLEAN_RESULTS=keep
set -euo pipefail
IFS=$'\n\t'

# ------------ user-editable settings ------------
PYTHON=${PYTHON:-python}
DATASET=${DATASET:-cora}
METHOD=${METHOD:-ours}
DATA_DIR=${DATA_DIR:-/mnt/d/File/code/SGFormer/data}
EPOCHS=${EPOCHS:-200}
DISPLAY_STEP=${DISPLAY_STEP:-10}
RUNS_PER_SEED=${RUNS_PER_SEED:-1}   # usually 1
# seeds for quick grid (phase 1). Expand later to 30 for final.
SEEDS=(100 101 102)

# Grid: adjust values as you like. Keep these small for fast phase-1 runs.
CONS_WEIGHTS=(0.02 0.05 0.1)
CONS_CONFS=(0.0 0.5 0.7)
EMA_TAUS=(0.99 0.95)
CONS_WARM=${CONS_WARM:-5}   # supervised epochs before consistency
CONS_UP=${CONS_UP:-20}      # epochs to ramp to target weight

# Whether to remove old results by default. Default = delete (as you requested earlier)
CLEAN_RESULTS=${CLEAN_RESULTS:-delete}

# Other runtime args
SAVE_CHECKPOINTS_FLAG="--save_checkpoints"
LOG_DIR=results
CHECKPOINT_DIR=results/checkpoints
EPOCH_LOG_DIR=results/epoch_logs
FIG_DIR=results/figs_pair
# ------------------------------------------------

echo "[INFO] CLEAN_RESULTS=$CLEAN_RESULTS"
if [ "${CLEAN_RESULTS}" = "delete" ]; then
  echo "[CLEAN] Deleting existing results/ (irreversible)"
  rm -rf results || true
  mkdir -p "${CHECKPOINT_DIR}" "${EPOCH_LOG_DIR}" "${FIG_DIR}"
else
  mkdir -p "${CHECKPOINT_DIR}" "${EPOCH_LOG_DIR}" "${FIG_DIR}"
fi

mkdir -p results
CSV="results/${DATASET}_${METHOD}_results_per_run.csv"
# ensure header
if [ ! -f "${CSV}" ]; then
  echo "exp_tag,seed,run,best_val,best_test" > "${CSV}"
fi

run_once() {
  local exp_tag=$1
  local seed=$2
  shift 2
  local extra_args=("$@")
  # check if already in csv (avoid rerun)
  if grep -E -q "^${exp_tag},${seed}," "${CSV}" 2>/dev/null; then
    echo "[SKIP] ${exp_tag}, seed ${seed} already present in ${CSV}"
    return 0
  fi
  cmd=( "${PYTHON}" main.py \
    --dataset "${DATASET}" \
    --method "${METHOD}" \
    --data_dir "${DATA_DIR}" \
    --epochs "${EPOCHS}" \
    --runs "${RUNS_PER_SEED}" \
    --display_step "${DISPLAY_STEP}" \
    --exp_tag "${exp_tag}" \
    --seed "${seed}" \
    "${SAVE_CHECKPOINTS_FLAG}" \
    "${extra_args[@]}" )
  echo "[RUN] ${cmd[*]}"
  "${cmd[@]}"
}

# Phase 1: quick grid
EXP_PREFIX="grid_s"
for cw in "${CONS_WEIGHTS[@]}"; do
  for cc in "${CONS_CONFS[@]}"; do
    for et in "${EMA_TAUS[@]}"; do
      TAG="${EXP_PREFIX}_w${cw}_conf${cc}_tau${et}"
      for s in "${SEEDS[@]}"; do
        run_once "${TAG}" "${s}" \
          --cons_weight "${cw}" \
          --cons_confidence "${cc}" \
          --ema_tau "${et}" \
          --cons_warm "${CONS_WARM}" \
          --cons_up "${CONS_UP}" \
          --cons_loss prob_mse \
          --cons_temp 1.0
      done
    done
  done
done

echo "[DONE] Grid complete. You can now run analysis:"
echo "  python scripts/analyze_pairwise_postrun.py --csv results/${DATASET}_${METHOD}_results_per_run.csv --tagA A_fair --tagB <your_best_tag> --metric best_test --out_dir results/figs_pair"
echo "Or run the one-click analysis:"
echo "  python scripts/oneclick_delta_pseudo_analysis.py"

exit 0

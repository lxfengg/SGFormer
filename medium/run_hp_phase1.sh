#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_hp_phase1.sh <dataset>
DATASET=${1:-}
if [ -z "$DATASET" ]; then
  echo "Usage: $0 <dataset>"
  exit 1
fi

DATA_DIR=${DATA_DIR:-/mnt/e/code/SGFormer/data}
PYTHON=${PYTHON:-python}
OUTDIR=${OUTDIR:-results_hp_phase1}
RUNS_PER_CALL=1
USE_CPU_FLAG="--cpu"   # set to empty to use GPU

mkdir -p "${OUTDIR}"
RAW_OUT="${OUTDIR}/${DATASET}_phase1_raw.csv"
echo "config_tag,exp_group,seed,run,best_val,best_test" > "${RAW_OUT}"

SEEDS=(100 101 102 103 104)

# Baseline small set
BASELINE_CONFIGS=(
  "--cons_weight 0 --cons_warm 0 --cons_up 0"
  "--cons_weight 0 --cons_warm 0 --cons_up 0 --lr 0.005"
)

# Schannel wider grid (weights, tempos, ema params). also include use_graph variants.
SCH_CONFIGS=(
  "--cons_weight 0.02 --ema_tau 0.99 --ema_start 10 --cons_warm 5 --cons_up 40 --cons_temp 1.0"
  "--cons_weight 0.05 --ema_tau 0.99 --ema_start 10 --cons_warm 10 --cons_up 40 --cons_temp 1.0"
  "--cons_weight 0.10 --ema_tau 0.99 --ema_start 5  --cons_warm 5  --cons_up 20 --cons_temp 1.0"
  "--cons_weight 0.20 --ema_tau 0.99 --ema_start 0  --cons_warm 0  --cons_up 40 --cons_temp 1.0"
  "--cons_weight 0.40 --ema_tau 0.99 --ema_start 0  --cons_warm 0  --cons_up 20 --cons_temp 1.0"
  "--cons_weight 0.10 --ema_tau 0.95 --ema_start 5  --cons_warm 5  --cons_up 20 --cons_temp 0.5"
  "--cons_weight 0.10 --ema_tau 0.90 --ema_start 5  --cons_warm 5  --cons_up 20 --cons_temp 2.0"
  # with graph enabled
  "--cons_weight 0.10 --ema_tau 0.99 --ema_start 5 --cons_warm 5 --cons_up 20 --cons_temp 1.0 --use_graph"
  "--cons_weight 0.20 --ema_tau 0.99 --ema_start 0 --cons_warm 0 --cons_up 40 --cons_temp 1.0 --use_graph"
  # try different cons_loss (kl, logit_mse)
  "--cons_weight 0.10 --cons_loss kl  --ema_tau 0.99 --ema_start 5 --cons_warm 5 --cons_up 20 --cons_temp 1.0"
  "--cons_weight 0.10 --cons_loss logit_mse --ema_tau 0.99 --ema_start 5 --cons_warm 5 --cons_up 20 --cons_temp 1.0"
)

# helper to run and salvage results
run_config() {
  local group="$1"
  local tag="$2"
  local args="$3"
  local seed="$4"

  echo "---- run: group=${group} tag=${tag} seed=${seed}"
  cmd=( ${PYTHON} main.py --dataset "${DATASET}" --method ours --data_dir "${DATA_DIR}" --runs "${RUNS_PER_CALL}" --seed "${seed}" ${USE_CPU_FLAG} --exp_tag "${tag}" ${args} )
  echo "CMD: ${cmd[*]}"
  if "${cmd[@]}"; then
    echo "[INFO] finished normally"
  else
    echo "[WARN] main.py exited non-zero (continue)"
  fi

  # salvage last line from results csv
  RAW_CSV="results/${DATASET}_ours_results_per_run.csv"
  run_idx=0; best_val=0; best_test=0
  if [ -f "${RAW_CSV}" ]; then
    last_line=$(tail -n +2 "${RAW_CSV}" | sed '/^\s*$/d' | tail -n 1 || true)
    if [ -n "${last_line}" ]; then
      IFS=',' read -r f1 f2 f3 f4 f5 <<< "$(echo "${last_line}" | tr -d '\r\n')"
      if [ -n "${f5}" ]; then
        run_idx="${f3}"
        best_val="${f4}"
        best_test="${f5}"
      else
        run_idx="${f1}"
        best_val="${f2}"
        best_test="${f3}"
      fi
    fi
  fi

  echo "${tag},${group},${seed},${run_idx},${best_val},${best_test}" >> "${RAW_OUT}"
  echo "[APPENDED] ${tag},${group},${seed},${run_idx},${best_val},${best_test}"
}

# run loops
for seed in "${SEEDS[@]}"; do
  for i in "${!BASELINE_CONFIGS[@]}"; do
    cfg="${BASELINE_CONFIGS[$i]}"
    run_config "baseline" "baseline_q${i}" "${cfg}" "${seed}"
  done

  for i in "${!SCH_CONFIGS[@]}"; do
    cfg="${SCH_CONFIGS[$i]}"
    run_config "schannel" "sch_q${i}" "${cfg}" "${seed}"
  done
done

echo "Phase1 finished. Raw results: ${RAW_OUT}"
echo "Next: run 'python select_and_test.py ${RAW_OUT}' to pick per-seed best & do pairwise test."

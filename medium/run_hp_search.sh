#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_hp_search.sh <dataset>
# Example:
#   ./run_hp_search.sh cora

DATASET=${1:-}
if [ -z "${DATASET}" ]; then
  echo "Usage: $0 <dataset>";
  exit 1
fi

DATA_DIR=${DATA_DIR:-/mnt/e/code/SGFormer/data}
PYTHON=${PYTHON:-python}
OUTDIR=${OUTDIR:-results_hp_search}
RUNS_PER_CALL=1
USE_CPU_FLAG="--cpu"   # set empty to use GPU

# Seeds to run (30 seeds typical). Edit as needed.
SEEDS=(100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129)

mkdir -p "${OUTDIR}"
rm -f "${OUTDIR}/${DATASET}_hp_raw.csv"
echo "config_tag,exp_group,seed,run,best_val,best_test" > "${OUTDIR}/${DATASET}_hp_raw.csv"

# --------------------------
# Define hyperparam grids
# --------------------------
# Baseline group (fair: we include a tiny grid; expand as needed)
BASELINE_CONFIGS=(
  "--cons_weight 0 --cons_warm 0 --cons_up 0"                # strict baseline
  "--cons_weight 0 --cons_warm 0 --cons_up 0 --lr 0.005"     # try a different lr
)

# Schannel group (search space — expand/contract as you like)
SCH_CONFIGS=(
  "--cons_weight 0.05 --ema_tau 0.99 --ema_start 10 --cons_warm 10 --cons_up 40 --cons_temp 1.0"
  "--cons_weight 0.05 --ema_tau 0.9  --ema_start 5  --cons_warm 5  --cons_up 20 --cons_temp 0.5"
  "--cons_weight 0.10 --ema_tau 0.9  --ema_start 5  --cons_warm 5  --cons_up 20 --cons_temp 0.5"
  "--cons_weight 0.10 --ema_tau 0.99 --ema_start 10 --cons_warm 10 --cons_up 40 --cons_temp 1.0"
)

# helper to run one config
run_config() {
  local exp_group="$1"   # "baseline" or "schannel"
  local config_tag="$2"  # short tag
  local extra_args="$3"
  local seed="$4"

  echo "------------------------------------------------------------"
  echo "[INFO] Running ${exp_group} tag=${config_tag} dataset=${DATASET} seed=${seed}"
  cmd=( "${PYTHON}" main.py --dataset "${DATASET}" --method ours --data_dir "${DATA_DIR}" --runs "${RUNS_PER_CALL}" --seed "${seed}" ${USE_CPU_FLAG} --exp_tag "${config_tag}" ${extra_args} )
  echo "CMD: ${cmd[*]}"
  # run
  if "${cmd[@]}"; then
    echo "[INFO] main.py exited normally"
  else
    echo "[WARN] main.py exited with non-zero code; will still try to salvage epoch log."
  fi

  # Read the per-run results CSV (the script main.py writes results/<dataset>_ours_results_per_run.csv)
  RAW_CSV="results/${DATASET}_ours_results_per_run.csv"
  # robust: pick last non-empty non-header line
  if [ -f "${RAW_CSV}" ]; then
    last_line=$(tail -n +2 "${RAW_CSV}" | sed '/^\s*$/d' | tail -n 1 || true)
    if [ -n "${last_line}" ]; then
      # Normalize fields; support both 3-field and 5-field formats
      IFS=',' read -r f1 f2 f3 f4 f5 <<< "$(echo "${last_line}" | tr -d '\r\n')"
      if [ -n "${f5}" ]; then
        run_idx=$(echo "${f3}" | xargs)
        best_val=$(echo "${f4}" | xargs)
        best_test=$(echo "${f5}" | xargs)
      elif [ -n "${f3}" ]; then
        run_idx=$(echo "${f1}" | xargs)
        best_val=$(echo "${f2}" | xargs)
        best_test=$(echo "${f3}" | xargs)
      else
        run_idx=0
        best_val=0
        best_test=0
      fi
    else
      run_idx=0; best_val=0; best_test=0
    fi
  else
    run_idx=0; best_val=0; best_test=0
  fi

  # Append a clean line to our hp_raw CSV
  echo "${config_tag},${exp_group},${seed},${run_idx},${best_val},${best_test}" >> "${OUTDIR}/${DATASET}_hp_raw.csv"
  echo "[INFO] Appended to ${OUTDIR}/${DATASET}_hp_raw.csv: ${config_tag},${exp_group},${seed},${run_idx},${best_val},${best_test}"
}

# ---------------
# Run loops
# ---------------
for seed in "${SEEDS[@]}"; do
  # baseline configs
  for i in "${!BASELINE_CONFIGS[@]}"; do
    cfg="${BASELINE_CONFIGS[$i]}"
    tag="baseline_cfg${i}"
    run_config "baseline" "${tag}" "${cfg}" "${seed}"
  done

  # schannel configs
  for i in "${!SCH_CONFIGS[@]}"; do
    cfg="${SCH_CONFIGS[$i]}"
    tag="sch_cfg${i}"
    run_config "schannel" "${tag}" "${cfg}" "${seed}"
  done
done

echo "DONE. Raw HP results: ${OUTDIR}/${DATASET}_hp_raw.csv"
echo "Next: run select_and_test.py ${OUTDIR}/${DATASET}_hp_raw.csv to pick per-seed best configs and run paired test."

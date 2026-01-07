#!/usr/bin/env bash
set -euo pipefail

# Quick HP sweep: runs a small grid on 3 seeds for fast screening.
# Usage: ./run_quick_hp.sh <dataset>
# Example: ./run_quick_hp.sh cora

DATASET=${1:-}
if [ -z "${DATASET}" ]; then
  echo "Usage: $0 <dataset>"
  exit 1
fi

DATA_DIR=${DATA_DIR:-/mnt/e/code/SGFormer/data}
PYTHON=${PYTHON:-python}
OUTDIR=${OUTDIR:-results_hp_quick}
RUNS_PER_CALL=1
USE_CPU_FLAG="--cpu"   # set to empty string to use GPU

# seeds for quick sweep (small set)
SEEDS=(100 101 102)

mkdir -p "${OUTDIR}"
RAW_OUT="${OUTDIR}/${DATASET}_hp_raw_quick.csv"
# header
echo "config_tag,exp_group,seed,run,best_val,best_test" > "${RAW_OUT}"

# --------------------------
# Baseline configs (fair, small set)
# --------------------------
BASELINE_CONFIGS=(
  "--cons_weight 0 --cons_warm 0 --cons_up 0"          # strict baseline
  "--cons_weight 0 --cons_warm 0 --cons_up 0 --lr 0.005"
)

# --------------------------
# Schannel configs -- expanded to include stronger weights & temps
# --------------------------
SCH_CONFIGS=(
  # moderate weights, conservative EMA
  "--cons_weight 0.05 --ema_tau 0.99  --ema_start 10 --cons_warm 10 --cons_up 40 --cons_temp 1.0"
  "--cons_weight 0.05 --ema_tau 0.95  --ema_start 5  --cons_warm 5  --cons_up 20 --cons_temp 0.5"

  # stronger consistency
  "--cons_weight 0.10 --ema_tau 0.99 --ema_start 10 --cons_warm 10 --cons_up 40 --cons_temp 1.0"
  "--cons_weight 0.20 --ema_tau 0.99 --ema_start 5  --cons_warm 5  --cons_up 20 --cons_temp 1.0"

  # aggressive: very strong weight (check stability)
  "--cons_weight 0.40 --ema_tau 0.99 --ema_start 0  --cons_warm 0  --cons_up 20 --cons_temp 1.0"

  # different temperature (smoother teacher)
  "--cons_weight 0.10 --ema_tau 0.95 --ema_start 5  --cons_warm 5  --cons_up 40 --cons_temp 2.0"
)

# helper to run one config and append a clean record
run_config() {
  local exp_group="$1"   # baseline | schannel
  local cfg_tag="$2"     # tag string
  local extra_args="$3"
  local seed="$4"

  echo "------------------------------------------------------------"
  echo "[INFO] Running ${exp_group} tag=${cfg_tag} dataset=${DATASET} seed=${seed}"
  cmd=( "${PYTHON}" main.py --dataset "${DATASET}" --method ours --data_dir "${DATA_DIR}" --runs "${RUNS_PER_CALL}" --seed "${seed}" ${USE_CPU_FLAG} --exp_tag "${cfg_tag}" ${extra_args} )
  echo "CMD: ${cmd[*]}"
  # run (allow failure but continue)
  if "${cmd[@]}"; then
    echo "[INFO] main.py exited normally"
  else
    echo "[WARN] main.py exited with non-zero code; continue and attempt salvage"
  fi

  # try to salvage latest produced per-run CSV
  RAW_CSV="results/${DATASET}_ours_results_per_run.csv"
  run_idx=0; best_val=0; best_test=0
  if [ -f "${RAW_CSV}" ]; then
    last_line=$(tail -n +2 "${RAW_CSV}" | sed '/^\s*$/d' | tail -n 1 || true)
    if [ -n "${last_line}" ]; then
      # split to tokens (support both 3-field and 5-field)
      IFS=',' read -r a b c d e <<< "$(echo "${last_line}" | tr -d '\r\n')"
      if [ -n "${e}" ]; then
        run_idx=$(echo "${c}" | xargs)
        best_val=$(echo "${d}" | xargs)
        best_test=$(echo "${e}" | xargs)
      elif [ -n "${c}" ]; then
        run_idx=$(echo "${a}" | xargs)
        best_val=$(echo "${b}" | xargs)
        best_test=$(echo "${c}" | xargs)
      fi
    fi
  fi

  echo "${cfg_tag},${exp_group},${seed},${run_idx},${best_val},${best_test}" >> "${RAW_OUT}"
  echo "[INFO] Appended: ${cfg_tag},${exp_group},${seed},${run_idx},${best_val},${best_test}"
}

# ---------------
# Run loops
# ---------------
for seed in "${SEEDS[@]}"; do
  for i in "${!BASELINE_CONFIGS[@]}"; do
    cfg="${BASELINE_CONFIGS[$i]}"
    tag="baseline_q${i}"
    run_config "baseline" "${tag}" "${cfg}" "${seed}"
  done

  for i in "${!SCH_CONFIGS[@]}"; do
    cfg="${SCH_CONFIGS[$i]}"
    tag="sch_q${i}"
    run_config "schannel" "${tag}" "${cfg}" "${seed}"
  done
done

echo "DONE. Raw quick results: ${RAW_OUT}"
echo "Now run: python select_and_test.py ${RAW_OUT}  to pick per-seed best configs and do paired test."

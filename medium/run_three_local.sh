#!/usr/bin/env bash
set -euo pipefail

# run_three_local_fixed.sh
# Usage: ./run_three_local_fixed.sh <dataset>
# Defaults:
DATA_DIR="${DATA_DIR:-/mnt/e/code/SGFormer/data}"
PYTHON="${PYTHON:-python}"
USE_CPU="${USE_CPU:-1}"

if [ $# -lt 1 ]; then
  echo "Usage: $0 <dataset>"
  exit 1
fi
DATASET="$1"

OUTDIR="results"
RAW_CSV="${OUTDIR}/${DATASET}_ours_results_per_run.csv"
TAGGED_CSV="${OUTDIR}/${DATASET}_ours_results_per_run_tagged.csv"

mkdir -p "${OUTDIR}"
mkdir -p "${OUTDIR}/epoch_logs"

SEEDS=(100 101 102)   # 三个 seed（示例，可改）
SCH_CONS_WEIGHT=0.05
SCH_CONS_WARM=20
SCH_CONS_UP=60
SCH_EMA_START=10
DISPLAY_STEP=50

CPU_FLAG=""
if [ "${USE_CPU}" = "1" ]; then
  CPU_FLAG="--cpu"
fi

# ensure header
if [ ! -f "${TAGGED_CSV}" ]; then
  echo "exp_tag,seed,run,best_val,best_test" > "${TAGGED_CSV}"
fi

_trim() { echo "$1" | awk '{$1=$1;print}'; }

run_one() {
  local exp_tag="$1"
  local seed="$2"
  shift 2
  local extra_args=("$@")

  echo "------------------------------------------------------------"
  echo "[INFO] Running exp_tag=${exp_tag} dataset=${DATASET} seed=${seed}"
  echo "cmd: ${PYTHON} main.py --dataset ${DATASET} --method ours --data_dir ${DATA_DIR} --runs 1 --seed ${seed} ${CPU_FLAG} --exp_tag ${exp_tag} ${extra_args[*]}"

  # count lines before
  pre_lines=0
  if [ -f "${RAW_CSV}" ]; then
    pre_lines=$(wc -l < "${RAW_CSV}" || echo 0)
  fi

  # run main.py and capture exit code
  if ${PYTHON} main.py --dataset "${DATASET}" --method ours --data_dir "${DATA_DIR}" \
      --runs 1 --seed "${seed}" ${CPU_FLAG} --exp_tag "${exp_tag}" "${extra_args[@]}"; then
    rc=0
  else
    rc=$?
  fi

  if [ "${rc}" -ne 0 ]; then
    echo "[ERROR] main.py exited with code ${rc} for seed ${seed}. Will NOT append to tagged CSV."
    return 1
  fi

  # small pause to ensure file flush
  sleep 0.1

  if [ ! -f "${RAW_CSV}" ]; then
    echo "[ERROR] expected raw results file ${RAW_CSV} not found after successful run."
    return 1
  fi

  post_lines=$(wc -l < "${RAW_CSV}" || echo 0)
  new_lines=$(( post_lines - pre_lines ))
  if [ "${new_lines}" -le 0 ]; then
    last_line=$(tail -n +2 "${RAW_CSV}" | sed '/^\s*$/d' | tail -n 1 || true)
  else
    last_line=$(tail -n "${new_lines}" "${RAW_CSV}" | sed '/^\s*$/d' | tail -n 1 || true)
  fi

  last_line=$(echo "${last_line}" | tr -d '\r' || true)
  if [ -z "${last_line}" ]; then
    echo "[WARN] could not extract last result line from ${RAW_CSV}. Skipping append."
    return 1
  fi

  # parse last_line robustly
  num_fields=$(echo "${last_line}" | awk -F',' '{print NF}')
  if [ "${num_fields}" -ge 5 ]; then
    run_idx=$(echo "${last_line}" | awk -F',' '{print $3}' | xargs)
    best_val=$(echo "${last_line}" | awk -F',' '{print $4}' | xargs)
    best_test=$(echo "${last_line}" | awk -F',' '{print $5}' | xargs)
  elif [ "${num_fields}" -eq 3 ]; then
    run_idx=$(echo "${last_line}" | awk -F',' '{print $1}' | xargs)
    best_val=$(echo "${last_line}" | awk -F',' '{print $2}' | xargs)
    best_test=$(echo "${last_line}" | awk -F',' '{print $3}' | xargs)
  else
    # salvage numeric tokens: find last 3 numeric-looking tokens
    IFS=',' read -r -a toks <<< "${last_line}"
    n=${#toks[@]}
    out_list=()
    found=0
    for ((i=n-1;i>=0;i--)); do
      tok=$(echo "${toks[i]}" | tr -d ' \t\r\n')
      if [[ "${tok}" =~ ^-?[0-9]+(\.[0-9]+)?$ ]]; then
        out_list=("${tok}" "${out_list[@]}")
        ((found++))
        if [ "${found}" -ge 3 ]; then break; fi
      fi
    done
    if [ "${found}" -ge 3 ]; then
      run_idx="${out_list[0]}"
      best_val="${out_list[1]}"
      best_test="${out_list[2]}"
    else
      echo "[WARN] cannot parse last_line: ${last_line}"
      return 1
    fi
  fi

  # sanity checks: numeric and plausible
  if ! [[ "${best_test}" =~ ^-?[0-9]+(\.[0-9]+)?$ ]]; then
    echo "[WARN] best_test is not numeric: ${best_test}. Skipping append."
    return 1
  fi

  echo "${exp_tag},${seed},${run_idx},${best_val},${best_test}" >> "${TAGGED_CSV}"
  echo "[INFO] appended to ${TAGGED_CSV}: ${exp_tag},${seed},${run_idx},${best_val},${best_test}"
  return 0
}

# baseline runs
echo "================  RUN GROUP: baseline (cons_weight=0)  ================"
for s in "${SEEDS[@]}"; do
  run_one "baseline" "${s}" --cons_weight 0 --cons_warm 0 --cons_up 0 --display_step ${DISPLAY_STEP} || echo "[WARN] baseline run failed for seed ${s}"
done

# schannel runs
echo "================  RUN GROUP: schannel (S-channel)  ================"
for s in "${SEEDS[@]}"; do
  run_one "schannel" "${s}" \
    --cons_weight "${SCH_CONS_WEIGHT}" \
    --cons_warm "${SCH_CONS_WARM}" \
    --cons_up "${SCH_CONS_UP}" \
    --ema_start "${SCH_EMA_START}" \
    --display_step "${DISPLAY_STEP}" || echo "[WARN] schannel run failed for seed ${s}"
done

echo "DONE. Tagged CSV: ${TAGGED_CSV}"
echo "Run analyses: python paired_analysis.py ${TAGGED_CSV}  or python stats_postproc.py ${TAGGED_CSV}"

#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_exps.sh <dataset>   (e.g. ./run_exps.sh citeseer)
# Defaults (可通过环境变量覆盖)
DATA_DIR=${DATA_DIR:-/mnt/e/code/SGFormer/data}
PYTHON=${PYTHON:-python}
METHOD=${METHOD:-ours}
OUTDIR=${OUTDIR:-results}

if [ $# -lt 1 ]; then
  echo "Usage: $0 <dataset>  (example: $0 citeseer)"
  exit 1
fi

DATASET=$1

# Seeds to use (示例：30 个 seeds). 如需改为 15 或更少，请修改此行。
SEEDS=(100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129)

# Per-run settings
RUNS_PER_CALL=1
USE_CPU_FLAG="--cpu"   # 若想用 GPU，把此置空：USE_CPU_FLAG=""
DISPLAY_STEP=50

# S-channel hyperparams (与你之前实验一致)
SCH_CONS_WEIGHT=0.05
SCH_CONS_WARM=20
SCH_CONS_UP=60
SCH_EMA_START=10

mkdir -p "${OUTDIR}"
mkdir -p "${OUTDIR}/epoch_logs"

TAGGED_CSV="${OUTDIR}/${DATASET}_${METHOD}_results_per_run_tagged.csv"
# ---------- 重要：RAW_CSV 必须与 main.py 写入的文件名一致 ----------
RAW_CSV="${OUTDIR}/${DATASET}_${METHOD}_results_per_run.csv"
# -------------------------------------------------------------------

# Initialize tagged CSV if missing
if [ ! -f "${TAGGED_CSV}" ]; then
  echo "exp_tag,seed,run,best_val,best_test" > "${TAGGED_CSV}"
fi

# helper: trim whitespace
_trim() { echo "$1" | awk '{$1=$1;print}'; }

run_one() {
  local exp_tag="$1"
  shift
  local extra_args=("$@")
  local seed="$SEED"

  echo "------------------------------------------------------------"
  echo "[INFO] Running exp_tag=${exp_tag} dataset=${DATASET} seed=${seed}"

  out_file="${OUTDIR}/${DATASET}_${METHOD}_${exp_tag}_seed${seed}.json"

  echo "cmd: ${PYTHON} main.py --dataset \"${DATASET}\" --method \"${METHOD}\" --data_dir \"${DATA_DIR}\" --runs \"${RUNS_PER_CALL}\" --seed \"${seed}\" ${USE_CPU_FLAG} --exp_tag \"${exp_tag}\" --out_file \"${out_file}\" ${extra_args[*]}"

  # count pre-lines
  if [ -f "${RAW_CSV}" ]; then
    pre_lines=$(wc -l < "${RAW_CSV}" || echo 0)
  else
    pre_lines=0
  fi

  # Run main.py (don't exit whole script if it fails; we handle later)
  ${PYTHON} main.py --dataset "${DATASET}" --method "${METHOD}" --data_dir "${DATA_DIR}" \
      --runs "${RUNS_PER_CALL}" --seed "${seed}" ${USE_CPU_FLAG} --exp_tag "${exp_tag}" --out_file "${out_file}" "${extra_args[@]}" || true

  sleep 0.1

  if [ ! -f "${RAW_CSV}" ]; then
    echo "[ERROR] expected raw results file ${RAW_CSV} not found after run. Will write placeholder entry."
    echo "${exp_tag},${seed},unknown,0.0,0.0" >> "${TAGGED_CSV}"
    return 0
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
    echo "[WARN] could not determine last result line from ${RAW_CSV}. Writing placeholder."
    echo "${exp_tag},${seed},unknown,0.0,0.0" >> "${TAGGED_CSV}"
    return 0
  fi

  num_fields=$(echo "${last_line}" | awk -F',' '{print NF}')
  if [ "${num_fields}" -ge 5 ]; then
    exp_tag_raw=$(echo "${last_line}" | awk -F',' '{print $1}')
    seed_raw=$(echo "${last_line}" | awk -F',' '{print $2}')
    run_idx_raw=$(echo "${last_line}" | awk -F',' '{print $3}')
    best_val_raw=$(echo "${last_line}" | awk -F',' '{print $4}')
    best_test_raw=$(echo "${last_line}" | awk -F',' '{print $5}')
    exp_tag_raw=$(_trim "${exp_tag_raw}")
    seed_raw=$(_trim "${seed_raw}")
    run_idx_raw=$(_trim "${run_idx_raw}")
    best_val_raw=$(_trim "${best_val_raw}")
    best_test_raw=$(_trim "${best_test_raw}")

    if [ "${seed_raw}" != "${seed}" ]; then
      echo "[WARN] seed in raw CSV (${seed_raw}) != requested seed (${seed}). Using requested seed for tagged CSV."
    fi
    run_idx="${run_idx_raw}"
    best_val="${best_val_raw}"
    best_test="${best_test_raw}"
  elif [ "${num_fields}" -eq 3 ]; then
    run_idx=$(echo "${last_line}" | awk -F',' '{print $1}' | xargs)
    best_val=$(echo "${last_line}" | awk -F',' '{print $2}' | xargs)
    best_test=$(echo "${last_line}" | awk -F',' '{print $3}' | xargs)
  else
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
      echo "[WARN] cannot parse last_line reliably: ${last_line}. Writing placeholder."
      echo "${exp_tag},${seed},unknown,0.0,0.0" >> "${TAGGED_CSV}"
      return 0
    fi
  fi

  run_idx=$(_trim "${run_idx}")
  best_val=$(_trim "${best_val}")
  best_test=$(_trim "${best_test}")

  echo "${exp_tag},${seed},${run_idx},${best_val},${best_test}" >> "${TAGGED_CSV}"
  echo "[INFO] appended to ${TAGGED_CSV}: ${exp_tag},${seed},${run_idx},${best_val},${best_test}"
  return 0
}

# Baseline runs (disable consistency)
echo "================  RUN GROUP: baseline (cons_weight=0)  ================"
for SEED in "${SEEDS[@]}"; do
  run_one "baseline" --cons_weight 0 --cons_warm 0 --cons_up 0 --display_step "${DISPLAY_STEP}" || echo "[WARN] run_one failed for seed ${SEED}"
done

# S-channel runs (enable consistency)
echo "================  RUN GROUP: schannel (S-channel + EMA)  ================"
for SEED in "${SEEDS[@]}"; do
  run_one "schannel" \
    --cons_weight "${SCH_CONS_WEIGHT}" \
    --cons_warm "${SCH_CONS_WARM}" \
    --cons_up "${SCH_CONS_UP}" \
    --ema_start "${SCH_EMA_START}" \
    --display_step "${DISPLAY_STEP}" || echo "[WARN] run_one failed for seed ${SEED}"
done

echo "DONE. Tagged CSV: ${TAGGED_CSV}"
echo "Run analyses: python paired_analysis.py ${TAGGED_CSV}  or python stats_postproc.py ${TAGGED_CSV}"

#!/usr/bin/env bash
set -euo pipefail

# === Configuration (修改这些变量以适配你环境) ===
DATA_DIR=${DATA_DIR:-/mnt/e/code/SGFormer/data}   # <- 你的数据目录
PYTHON=${PYTHON:-python}                         # <- python 可执行文件（可写绝对路径）
METHOD=${METHOD:-ours}
OUTDIR=${OUTDIR:-results}
DISPLAY_STEP=${DISPLAY_STEP:-50}
USE_CPU_FLAG="--cpu"      # 如果想用 GPU，把这行改为 USE_CPU_FLAG="" 并在 main.py 调用时传 --device <id>

# === Dataset / seeds ===
if [ $# -lt 1 ]; then
  echo "Usage: $0 <dataset>  (example: $0 pubmed)"
  exit 1
fi
DATASET="$1"

# 推荐 30 个 seed（若算力不足可缩短）
SEEDS=(100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129)

# S-channel 超参（你之前实验中使用的值）
SCH_CONS_WEIGHT=${SCH_CONS_WEIGHT:-0.05}
SCH_CONS_WARM=${SCH_CONS_WARM:-20}
SCH_CONS_UP=${SCH_CONS_UP:-60}
SCH_EMA_START=${SCH_EMA_START:-10}

# 生成结果目录
mkdir -p "${OUTDIR}"
mkdir -p "${OUTDIR}/epoch_logs"

RAW_CSV="${OUTDIR}/${DATASET}_${METHOD}_results_per_run.csv"
TAGGED_CSV="${OUTDIR}/${DATASET}_${METHOD}_results_per_run_tagged.csv"

# 如果 tagged csv 不存在，写 header（确保 file 格式统一）
if [ ! -f "${TAGGED_CSV}" ]; then
  echo "exp_tag,seed,run,best_val,best_test" > "${TAGGED_CSV}"
fi

# helper: trim CR/LF and spaces
_trim() { echo "$1" | tr -d '\r' | awk '{$1=$1;print}'; }

# run single config and append a tagged row to TAGGED_CSV
run_one() {
  local exp_tag="$1"; shift
  local extra_args=("$@")
  local seed="$SEED"

  echo "------------------------------------------------------------"
  echo "[INFO] exp=${exp_tag} dataset=${DATASET} seed=${seed}"
  echo "cmd: ${PYTHON} main.py --dataset ${DATASET} --method ${METHOD} --data_dir ${DATA_DIR} --runs 1 --seed ${seed} ${USE_CPU_FLAG} --exp_tag ${exp_tag} ${extra_args[*]}"

  # run main.py (it appends to RAW_CSV)
  "${PYTHON}" main.py --dataset "${DATASET}" --method "${METHOD}" --data_dir "${DATA_DIR}" \
      --runs 1 --seed "${seed}" ${USE_CPU_FLAG} --exp_tag "${exp_tag}" "${extra_args[@]}"

  # ensure raw CSV exists and get last non-empty line
  if [ ! -f "${RAW_CSV}" ]; then
    echo "[ERROR] expected raw results file ${RAW_CSV} not found after run."
    exit 2
  fi
  last_line=$(tail -n +2 "${RAW_CSV}" | sed '/^\s*$/d' | tail -n 1 || true)
  last_line=$(_trim "${last_line}")
  if [ -z "${last_line}" ]; then
    echo "[WARN] no non-empty last line in ${RAW_CSV}; skipping append."
    return
  fi

  # parse last_line expecting: exp_tag,seed,run,best_val,best_test  OR run,best_val,best_test
  # normalize by removing CRs
  last_line=$(echo "${last_line}" | tr -d '\r')
  num_fields=$(echo "${last_line}" | awk -F',' '{print NF}')
  if [ "${num_fields}" -ge 5 ]; then
    # assume it's already tagged (exp_tag,seed,run,best_val,best_test)
    parsed_exp_tag=$(echo "${last_line}" | awk -F',' '{print $1}')
    parsed_seed=$(echo "${last_line}" | awk -F',' '{print $2}')
    run_idx=$(echo "${last_line}" | awk -F',' '{print $3}')
    best_val=$(echo "${last_line}" | awk -F',' '{print $4}')
    best_test=$(echo "${last_line}" | awk -F',' '{print $5}')
  elif [ "${num_fields}" -eq 3 ]; then
    # format run,best_val,best_test
    run_idx=$(echo "${last_line}" | awk -F',' '{print $1}')
    best_val=$(echo "${last_line}" | awk -F',' '{print $2}')
    best_test=$(echo "${last_line}" | awk -F',' '{print $3}')
    parsed_exp_tag="${exp_tag}"
    parsed_seed="${seed}"
  else
    # fallback: attempt to extract last 3 numeric tokens
    toks=$(echo "${last_line}" | tr ',' '\n' | sed 's/^[ \t]*//;s/[ \t]*$//' | tac)
    # pick first 3 numeric-ish tokens
    nums=()
    while IFS= read -r t; do
      if [[ "${t}" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        nums+=("${t}")
      fi
      if [ "${#nums[@]}" -ge 3 ]; then break; fi
    done <<< "${toks}"
    if [ "${#nums[@]}" -ge 3 ]; then
      run_idx="${nums[2]}"
      best_val="${nums[1]}"
      best_test="${nums[0]}"
    else
      run_idx="unknown"; best_val="0"; best_test="0"
    fi
    parsed_exp_tag="${exp_tag}"
    parsed_seed="${seed}"
  fi

  # final trim and append with our canonical exp_tag and seed (ensures pairing)
  run_idx=$(_trim "${run_idx}")
  best_val=$(_trim "${best_val}")
  best_test=$(_trim "${best_test}")

  echo "${exp_tag},${seed},${run_idx},${best_val},${best_test}" >> "${TAGGED_CSV}"
  echo "[INFO] appended to ${TAGGED_CSV}: ${exp_tag},${seed},${run_idx},${best_val},${best_test}"
}

# === 1) Baseline runs ===
echo "================  RUN GROUP: baseline (cons_weight=0)  ================"
for SEED in "${SEEDS[@]}"; do
  run_one "baseline" --cons_weight 0 --cons_warm 0 --cons_up 0 --display_step "${DISPLAY_STEP}" || echo "[WARN] run failed for seed ${SEED}"
done

# === 2) S-channel runs ===
echo "================  RUN GROUP: schannel (S-channel + EMA)  ================"
for SEED in "${SEEDS[@]}"; do
  run_one "schannel" \
    --cons_weight "${SCH_CONS_WEIGHT}" \
    --cons_warm "${SCH_CONS_WARM}" \
    --cons_up "${SCH_CONS_UP}" \
    --ema_start "${SCH_EMA_START}" \
    --display_step "${DISPLAY_STEP}" || echo "[WARN] run failed for seed ${SEED}"
done

echo "DONE. Tagged CSV: ${TAGGED_CSV}"
echo "Run analyses: python paired_analysis.py ${TAGGED_CSV}  OR python stats_postproc.py ${TAGGED_CSV}"

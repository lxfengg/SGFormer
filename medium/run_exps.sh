#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_replication.sh <dataset>   (e.g. ./run_replication.sh citeseer)
# Default data dir - 修改为你当前的数据路径（你告诉过我是 /mnt/e/code/SGFormer/data）
DATA_DIR=${DATA_DIR:-/mnt/e/code/SGFormer/data}
PYTHON=${PYTHON:-python}   # 或者指定 full path to python 环境
METHOD=${METHOD:-ours}
OUTDIR=${OUTDIR:-results}

if [ $# -lt 1 ]; then
  echo "Usage: $0 <dataset>  (example: $0 citeseer)"
  exit 1
fi

DATASET=$1

# Seeds to use (建议 30 个做最终统计；这里示例给 30 个 seed).
# 如需 fewer runs，把列表缩短；如需更多，扩展列表。
SEEDS=(100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129)

# Run-level settings:
RUNS_PER_CALL=1   # main.py 内仍为 --runs 1 每次运行以方便 pairing（我们用 seed 做 pairing）
USE_CPU_FLAG="--cpu"   # 如果希望使用 GPU，置空：USE_CPU_FLAG=""
DISPLAY_STEP=50

# S-channel hyperparams (你之前实验里用的)
SCH_CONS_WEIGHT=0.05
SCH_CONS_WARM=20
SCH_CONS_UP=60
SCH_EMA_START=10

# Make sure result folders exist
mkdir -p ${OUTDIR}
mkdir -p ${OUTDIR}/epoch_logs

# Master tagged CSV we'll produce (exp_tag,seed,run,best_val,best_test)
TAGGED_CSV="${OUTDIR}/${DATASET}_${METHOD}_results_per_run_tagged.csv"
RAW_CSV="${OUTDIR}/${DATASET}_${METHOD}_results_per_run.csv"

if [ ! -f "${TAGGED_CSV}" ]; then
  echo "exp_tag,seed,run,best_val,best_test" > "${TAGGED_CSV}"
fi

# helper: run a single config (arguments are $1 = exp_tag, $2.. = extra args for main.py)
run_one() {
  local exp_tag="$1"; shift
  local extra_args=("$@")
  local seed=$SEED

  echo "------------------------------------------------------------"
  echo "[INFO] exp=${exp_tag} dataset=${DATASET} seed=${seed}"
  echo "cmd: ${PYTHON} main.py --dataset ${DATASET} --method ${METHOD} --data_dir ${DATA_DIR} --runs ${RUNS_PER_CALL} --seed ${seed} ${USE_CPU_FLAG} ${extra_args[*]}"
  # run main.py (it will append to raw results CSV under results/)
  ${PYTHON} main.py --dataset ${DATASET} --method ${METHOD} --data_dir ${DATA_DIR} \
      --runs ${RUNS_PER_CALL} --seed ${seed} ${USE_CPU_FLAG} ${extra_args[*]}

  # Wait for file to be present
  if [ ! -f "${RAW_CSV}" ]; then
    echo "[ERROR] expected raw results file ${RAW_CSV} not found after run."
    exit 2
  fi

  # Extract last non-empty non-header line from raw csv (robust)
  last_line=$(tail -n +2 "${RAW_CSV}" | sed '/^\s*$/d' | tail -n 1)
  if [ -z "${last_line}" ]; then
    echo "[WARN] no data line found in ${RAW_CSV}"
    return
  fi

  # parse CSV line -> run,best_val,best_test
  IFS=',' read -r run_idx best_val best_test <<< "$(echo "${last_line}" | tr -d '\r\n')"

  # Append tagged row to master CSV
  echo "${exp_tag},${seed},${run_idx},${best_val},${best_test}" >> "${TAGGED_CSV}"
  echo "[INFO] appended to ${TAGGED_CSV}: ${exp_tag},${seed},${run_idx},${best_val},${best_test}"
}

# 1) Baseline runs (consistency disabled)
echo "================  RUN GROUP: baseline (cons_weight=0)  ================"
for SEED in "${SEEDS[@]}"; do
  # baseline args: cons_weight=0, cons_warm=0, cons_up=0
  run_one "baseline" --cons_weight 0 --cons_warm 0 --cons_up 0 --display_step ${DISPLAY_STEP}
done

# 2) S-channel runs (enable consistency with your chosen hyperparams)
echo "================  RUN GROUP: schannel (S-channel + EMA)  ================"
for SEED in "${SEEDS[@]}"; do
  run_one "schannel" \
    --cons_weight ${SCH_CONS_WEIGHT} \
    --cons_warm ${SCH_CONS_WARM} \
    --cons_up ${SCH_CONS_UP} \
    --ema_start ${SCH_EMA_START} \
    --display_step ${DISPLAY_STEP}
done

echo "DONE. Tagged CSV is: ${TAGGED_CSV}"
echo "You can now run your analysis scripts, e.g.:"
echo "  python paired_analysis.py ${TAGGED_CSV}"
echo "  python stats_postproc.py ${TAGGED_CSV}"

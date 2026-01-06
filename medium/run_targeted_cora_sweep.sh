#!/usr/bin/env bash
set -euo pipefail

# run_targeted_cora_sweep.sh
# 目的：在 Cora 上做小规模超参扫（快速判断能否放大 S-channel 效应）
# 使用说明：
#   修改 DATA_DIR / PYTHON / SEEDS / N_SEEDS_PER_CONFIG 根据你的机器调整
# Example:
#   ./run_targeted_cora_sweep.sh

DATA_DIR=${DATA_DIR:-/mnt/e/code/SGFormer/data}
PYTHON=${PYTHON:-python}
METHOD=${METHOD:-ours}
OUTDIR=${OUTDIR:-results}
DATASET=${1:-cora}   # 默认 cora，可改为 citeseer/pubmed 运行同样脚本
RUNS_PER_CALL=1

# Seeds to use for each config (示例 10 个)
SEEDS=(100 101 102 103 104 105 106 107 108 109)
N_SEEDS=${#SEEDS[@]}

# Output CSV (tagged)
TAGGED_CSV="${OUTDIR}/${DATASET}_${METHOD}_results_per_run_tagged.csv"
RAW_CSV="${OUTDIR}/${DATASET}_${METHOD}_results_per_run.csv"

mkdir -p "${OUTDIR}"
mkdir -p "${OUTDIR}/epoch_logs"

if [ ! -f "${TAGGED_CSV}" ]; then
  echo "exp_tag,seed,run,best_val,best_test" > "${TAGGED_CSV}"
fi

# helper trim
_trim() { echo "$1" | awk '{$1=$1;print}'; }

run_one() {
  local exp_tag="$1"; shift
  local seed="$1"; shift
  local extra_args=("$@")

  echo "------------------------------------------------------------"
  echo "[INFO] exp=${exp_tag} dataset=${DATASET} seed=${seed}"
  echo "cmd: ${PYTHON} main.py --dataset ${DATASET} --method ${METHOD} --data_dir ${DATA_DIR} --runs ${RUNS_PER_CALL} --seed ${seed} --cpu --exp_tag ${exp_tag} ${extra_args[*]}"

  # run
  ${PYTHON} main.py --dataset "${DATASET}" --method "${METHOD}" --data_dir "${DATA_DIR}" \
      --runs "${RUNS_PER_CALL}" --seed "${seed}" --cpu --exp_tag "${exp_tag}" "${extra_args[@]}"

  # robustly extract last data line from RAW_CSV
  if [ ! -f "${RAW_CSV}" ]; then
    echo "[WARN] raw csv ${RAW_CSV} not found after run (main.py may have written elsewhere). Skipping append."
    return
  fi

  last_line=$(tail -n +2 "${RAW_CSV}" | sed '/^\s*$/d' | tail -n 1 || true)
  last_line=$(echo "${last_line}" | tr -d '\r')
  if [ -z "${last_line}" ]; then
    echo "[WARN] no data line found in ${RAW_CSV} after run"
    return
  fi

  # parse last_line: support both "exp_tag,seed,run,best_val,best_test" and "run,best_val,best_test"
  num_fields=$(echo "${last_line}" | awk -F',' '{print NF}')
  if [ "${num_fields}" -ge 5 ]; then
    exp_tag_raw=$(echo "${last_line}" | awk -F',' '{print $1}')
    seed_raw=$(echo "${last_line}" | awk -F',' '{print $2}')
    run_idx_raw=$(echo "${last_line}" | awk -F',' '{print $3}')
    best_val_raw=$(echo "${last_line}" | awk -F',' '{print $4}')
    best_test_raw=$(echo "${last_line}" | awk -F',' '{print $5}')
    run_idx=$(_trim "${run_idx_raw}")
    best_val=$(_trim "${best_val_raw}")
    best_test=$(_trim "${best_test_raw}")
  elif [ "${num_fields}" -eq 3 ]; then
    run_idx=$(echo "${last_line}" | awk -F',' '{print $1}' | xargs)
    best_val=$(echo "${last_line}" | awk -F',' '{print $2}' | xargs)
    best_test=$(echo "${last_line}" | awk -F',' '{print $3}' | xargs)
  else
    echo "[WARN] cannot parse last_line: ${last_line}"
    run_idx="unknown"; best_val="0"; best_test="0"
  fi

  echo "${exp_tag},${seed},${run_idx},${best_val},${best_test}" >> "${TAGGED_CSV}"
  echo "[INFO] appended: ${exp_tag},${seed},${run_idx},${best_val},${best_test}"
}

# ---- configs to test ----
# baseline first
echo "==== RUN GROUP: baseline ===="
for s in "${SEEDS[@]}"; do
  run_one "baseline" "${s}" --cons_weight 0 --cons_warm 0 --cons_up 0 --display_step 50
done

# S-channel default (your earlier good config)
echo "==== RUN GROUP: schannel_default (cons_weight=0.05,warm=20,up=60) ===="
for s in "${SEEDS[@]}"; do
  run_one "schannel_default" "${s}" --cons_weight 0.05 --cons_warm 20 --cons_up 60 --ema_start 10 --display_step 50
done

# small hyperparam grid (only a few combos) -- add/remove combos as needed
CONS_WEIGHTS=(0.01 0.05 0.10)
EMA_STARTS=(10 50)
CONS_WARMS=(10 20)

for w in "${CONS_WEIGHTS[@]}"; do
  for es in "${EMA_STARTS[@]}"; do
    for cw in "${CONS_WARMS[@]}"; do
      tag="sch_w${w}_ema${es}_warm${cw}"
      echo "==== RUN GROUP: ${tag} ===="
      for s in "${SEEDS[@]}"; do
        run_one "${tag}" "${s}" --cons_weight "${w}" --ema_start "${es}" --cons_warm "${cw}" --cons_up 60 --display_step 50
      done
    done
  done
done

echo "DONE. Tagged CSV: ${TAGGED_CSV}"
echo "建议随后运行： python paired_analysis.py ${TAGGED_CSV}  或 stats_postproc.py ${TAGGED_CSV}"

#!/usr/bin/env bash
set -euo pipefail

# run_targeted_cora_sweep.sh
PYTHON=${PYTHON:-python}
DATA_DIR=${DATA_DIR:-/mnt/e/code/SGFormer/data}
OUTDIR=${OUTDIR:-results}
DATASET=${DATASET:-cora}
METHOD=${METHOD:-ours}
DISPLAY_STEP=50

# seeds for each config (10 seeds for quick but meaningful stats)
SEEDS=(100 101 102 103 104 105 106 107 108 109)

# grid to sweep (small grid; 可按需扩展)
CONS_WEIGHTS=(0.01 0.05 0.10)
EMA_STARTS=(5 10)
CONS_WARMS=(10 20)
CONS_UPS=(40 60)

mkdir -p "${OUTDIR}"
mkdir -p "${OUTDIR}/epoch_logs"

for w in "${CONS_WEIGHTS[@]}"; do
  for ema in "${EMA_STARTS[@]}"; do
    for warm in "${CONS_WARMS[@]}"; do
      for up in "${CONS_UPS[@]}"; do
        TAG="sch_w${w}_ema${ema}_warm${warm}_up${up}"
        echo "=== RUN CONFIG: ${TAG} ==="
        for seed in "${SEEDS[@]}"; do
          echo "[INFO] ${TAG} seed ${seed}"
          ${PYTHON} main.py \
            --dataset "${DATASET}" \
            --method "${METHOD}" \
            --data_dir "${DATA_DIR}" \
            --runs 1 \
            --seed "${seed}" \
            --cpu \
            --exp_tag "${TAG}" \
            --cons_weight "${w}" \
            --ema_start "${ema}" \
            --cons_warm "${warm}" \
            --cons_up "${up}" \
            --display_step "${DISPLAY_STEP}"
        done
      done
    done
  done
done

echo "DONE sweep. Raw per-run CSV is: results/${DATASET}_${METHOD}_results_per_run.csv"

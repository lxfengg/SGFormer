#!/usr/bin/env bash
ROOT=/mnt/d/File/code/SGFormer
MAIN=${ROOT}/medium/main.py
DATA=${ROOT}/data
LOGDIR=${ROOT}/medium/logs
mkdir -p ${LOGDIR}
MODE=${1:-debug}  # debug | full

if [ "${MODE}" = "debug" ]; then
  RUNS=3
else
  RUNS=10
fi

echo "======================================"
echo "[RUN A] Baseline (no consistency)"
echo "======================================"
python ${MAIN} \
  --dataset cora --method ours --epochs 1200 --runs ${RUNS} \
  --cpu \
  --data_dir ${DATA} \
  --cons_warm 0 --cons_up 0 --cons_weight 0.0 \
  --display_step 50 2>&1 | tee ${LOGDIR}/runA_baseline.log

echo "======================================"
echo "[RUN B] S-channel + EMA consistency"
echo "======================================"
python ${MAIN} \
  --dataset cora --method ours --epochs 1200 --runs ${RUNS} \
  --cpu \
  --data_dir ${DATA} \
  --cons_warm 20 --cons_up 60 --cons_weight 0.05 \
  --ema_start 10 --ema_tau 0.99 \
  --cons_loss prob_mse \
  --display_step 50 2>&1 | tee ${LOGDIR}/runB_schannel.log

echo "[INFO] Done. Logs at ${LOGDIR}, results at $(pwd)/results/"

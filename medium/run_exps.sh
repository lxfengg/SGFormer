#!/usr/bin/env bash
set -e

############################
# 基本路径配置（只做变量替换）
############################
ROOT_DIR=/mnt/d/File/code/SGFormer
DATA_DIR=${ROOT_DIR}/data
MEDIUM_DIR=${ROOT_DIR}/medium
LOG_DIR=${MEDIUM_DIR}/logs

PY=python
MAIN_PY=${MEDIUM_DIR}/main.py

mkdir -p ${LOG_DIR}

echo "[INFO] ROOT_DIR=${ROOT_DIR}"
echo "[INFO] DATA_DIR=${DATA_DIR}"
echo "[INFO] MAIN_PY=${MAIN_PY}"
echo "[INFO] LOG_DIR=${LOG_DIR}"

############################
# 通用参数（A / B 共用）
############################
COMMON_ARGS="
  --dataset cora
  --method ours
  --epochs 1200
  --runs 10
  --patience 400
  --cpu
  --data_dir ${DATA_DIR}
  --display_step 50
"

############################
# ========== A 方案 ==========
# Baseline（等价于你之前“直接 python main.py 能跑”的方式）
############################
echo "======================================"
echo "[RUN A] Baseline (no consistency)"
echo "======================================"

${PY} ${MAIN_PY} \
  ${COMMON_ARGS} \
  --cons_warm 0 \
  --cons_up 0 \
  --cons_weight 0 \
  2>&1 | tee ${LOG_DIR}/A_baseline.log

############################
# ========== B 方案 ==========
# S-channel + EMA teacher（一致性训练）
############################
echo "======================================"
echo "[RUN B] S-channel + EMA consistency"
echo "======================================"

${PY} ${MAIN_PY} \
  ${COMMON_ARGS} \
  --cons_warm 20 \
  --cons_up 60 \
  --cons_weight 0.05 \
  --cons_loss prob_mse \
  --cons_temp 1.0 \
  --ema_start 10 \
  --ema_tau 0.99 \
  --cons_confidence 0.0 \
  2>&1 | tee ${LOG_DIR}/B_s_channel_w0.05.log

echo "[INFO] All experiments finished."

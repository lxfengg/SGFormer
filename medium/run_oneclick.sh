#!/usr/bin/env bash
# 保存为 run_oneclick.sh
# 一键执行：检查数据 -> 运行 A_fair 与 A_plus_S (seed=100) -> 诊断脚本
# 使用： ./run_oneclick.sh [DATA_DIR]
# 如果提供 DATA_DIR 会优先使用；否则脚本会在常见位置查找 Planetoid/raw

set -euo pipefail
IFS=$'\n\t'

PYTHON=${PYTHON:-python}
DATA_DIR_ARG=${1:-""}

# candidate data dirs to search if no arg given (common possibilities)
CANDIDATES=(
  "$DATA_DIR_ARG"
  "/mnt/e/code/SGFormer/data"
  "/mnt/d/File/code/SGFormer/data"
  "/mnt/d/file/code/SGFormer/data"
  "$(pwd)/data"
  "$(pwd)/../data"
  "/home/$USER/data"
)

FOUND_DATA=""
for d in "${CANDIDATES[@]}"; do
  if [ -z "$d" ]; then
    continue
  fi
  RAW="$d/Planetoid/raw"
  if [ -d "$RAW" ]; then
    FOUND_DATA="$d"
    break
  fi
done

if [ -z "$FOUND_DATA" ]; then
  echo "ERROR: Could not find local Planetoid/raw under tried locations."
  echo "Please download Planetoid datasets (cora/citeseer/pubmed) and place them under one of these paths:"
  printf "  %s\n" "${CANDIDATES[@]}"
  echo "Or run this script with the data dir as argument: ./run_oneclick.sh /path/to/data"
  exit 1
fi

echo "[INFO] Using data_dir = $FOUND_DATA"
export DATA_DIR="$FOUND_DATA"

# set dataset and seeds (adjust as needed)
DATASET=${DATASET:-cora}
METHOD=${METHOD:-ours}
SEED=${SEED:-100}
EPOCHS=${EPOCHS:-200}
DISPLAY_STEP=${DISPLAY_STEP:-10}

# make sure results folders exist
mkdir -p results/epoch_logs results/checkpoints results

echo
echo "==== STEP 1: Run FAIR baseline (A_fair) seed=${SEED} ===="
$PYTHON main.py \
  --dataset ${DATASET} \
  --method ${METHOD} \
  --data_dir "${DATA_DIR}" \
  --seed ${SEED} \
  --exp_tag A_fair \
  --epochs ${EPOCHS} \
  --runs 1 \
  --display_step ${DISPLAY_STEP} \
  --cons_weight 0.0 \
  --save_checkpoints || { echo "[WARN] A_fair failed"; }

echo
echo "==== STEP 2: Run A + S (consistency ENABLED) seed=${SEED} ===="
$PYTHON main.py \
  --dataset ${DATASET} \
  --method ${METHOD} \
  --data_dir "${DATA_DIR}" \
  --seed ${SEED} \
  --exp_tag A_plus_S \
  --epochs ${EPOCHS} \
  --runs 1 \
  --display_step ${DISPLAY_STEP} \
  --cons_weight 0.1 \
  --cons_warm 10 \
  --cons_up 40 \
  --cons_loss prob_mse \
  --cons_confidence 0.0 \
  --ema_tau 0.99 \
  --save_checkpoints || { echo "[WARN] A_plus_S failed"; }

echo
echo "==== STEP 3: Analyze teacher coverage (epoch logs) ===="
# run analysis script if exists, otherwise use quick inline parsing
if [ -f analyze_teacher_coverage.py ]; then
  $PYTHON analyze_teacher_coverage.py || echo "[WARN] analyze_teacher_coverage.py failed"
else
  echo "[INFO] analyze_teacher_coverage.py not found — doing quick scan of epoch logs"
  for f in results/epoch_logs/*.csv; do
    echo "---- $f ----"
    head -n 5 "$f" || true
    tail -n 3 "$f" || true
  done
fi

echo
echo "==== STEP 4: Reachability check ===="
if [ -f reachability.py ]; then
  $PYTHON reachability.py || echo "[WARN] reachability.py failed"
else
  echo "[WARN] reachability.py not found; skipping"
fi

echo
echo "==== STEP 5: Pseudo-label quality (use checkpoint if exists) ===="
# prefer checkpoint from A_plus_S, fallback to A_fair
CKPT1="results/checkpoints/${DATASET}_A_plus_S_seed${SEED}_run0_best.pth"
CKPT2="results/checkpoints/${DATASET}_A_fair_seed${SEED}_run0_best.pth"
CKPT=""
if [ -f "$CKPT1" ]; then
  CKPT="$CKPT1"
elif [ -f "$CKPT2" ]; then
  CKPT="$CKPT2"
fi

if [ -n "$CKPT" ]; then
  if [ -f pseudo_label_quality.py ]; then
    echo "[INFO] Running pseudo_label_quality.py on $CKPT"
    $PYTHON pseudo_label_quality.py --ckpt "$CKPT" --dataset ${DATASET} || echo "[WARN] pseudo_label_quality.py failed"
  else
    echo "[WARN] pseudo_label_quality.py not found; skipping pseudo-label eval"
  fi
else
  echo "[WARN] No checkpoint found at $CKPT1 or $CKPT2; skipping pseudo-label eval"
  echo "Make sure --save_checkpoints was enabled and training finished at least one run."
fi

echo
echo "==== DONE ===="
echo "Logs: results/epoch_logs/"
echo "Checkpoints: results/checkpoints/"
echo "Summary CSVs: results/*_results_per_run.csv"
echo
echo "If any step failed due to missing script files, please ensure the following helper scripts exist in repo:"
echo "  - analyze_teacher_coverage.py"
echo "  - pseudo_label_quality.py"
echo "  - reachability.py"
echo
echo "To rerun with a custom data_dir: ./run_oneclick.sh /path/to/data"

#!/usr/bin/env bash
# run_demo_weaken_baseline.sh
# 把 baseline 弱化并与 schannel 做对比（epoch-log 不会覆盖）

set -euo pipefail

DATASET=${1:-}
if [ -z "$DATASET" ]; then
  echo "Usage: $0 <dataset>  (e.g. $0 cora)"
  exit 1
fi

PYTHON=${PYTHON:-python}
DATA_DIR=${DATA_DIR:-/mnt/e/code/SGFormer/data}
OUTDIR=${OUTDIR:-results}
OUT_TAGGED="${OUTDIR}/demo_${DATASET}_tagged.csv"

mkdir -p "${OUTDIR}"
mkdir -p results/epoch_logs

# Seeds: 示例 5 个 seed
SEEDS=(100 101 102 103 104)

# Baseline (被削弱)：极差配置，短训练、超小学习率、不使用 graph（如需更弱可增大 dropout 或禁用其他正则）
BASELINE_EXP_TAG="baseline_bad"
BASELINE_ARGS="--exp_tag ${BASELINE_EXP_TAG} --epochs 1 --lr 1e-05 --runs 1 --use_graph False"

# Schannel (较好)：启用 consistency & EMA，正常训练预算
SCH_EXP_TAG="schannel_good"
SCH_ARGS="--exp_tag ${SCH_EXP_TAG} --cons_weight 0.10 --ema_tau 0.99 --ema_start 5 --cons_warm 5 --cons_up 40 --cons_temp 1.0 --use_graph --epochs 200 --lr 0.01 --runs 1"

# CPU flag (默认使用 CPU。如需 GPU 请改为空 "")
USE_CPU_FLAG="--cpu"

# Prepare output CSV (overwrite)
echo "exp_tag,seed,run,best_val,best_test" > "${OUT_TAGGED}"

for seed in "${SEEDS[@]}"; do
  echo "===== SEED ${seed} : ${BASELINE_EXP_TAG} ====="
  set +e
  ${PYTHON} main.py --dataset "${DATASET}" --method ours --data_dir "${DATA_DIR}" --seed "${seed}" ${USE_CPU_FLAG} ${BASELINE_ARGS}
  rc=$?
  set -e
  echo "run finished (rc=${rc})"

  epoch_log="results/epoch_logs/${DATASET}_ours_${BASELINE_EXP_TAG}_seed${seed}_run0.csv"
  if [ -f "${epoch_log}" ]; then
    python - <<PY
import pandas as pd,sys
fn = "${epoch_log}"
s = ${seed}
try:
    df = pd.read_csv(fn)
except Exception:
    print(f"${BASELINE_EXP_TAG},{s},0,0.000000,0.000000")
    sys.exit(0)

if 'val' in df.columns and 'test' in df.columns and len(df)>0:
    idx = df['val'].idxmax()
    bv = float(df.loc[idx,'val'])
    bt = float(df.loc[idx,'test'])
else:
    try:
        bv = float(df['val'].iloc[-1]) if 'val' in df.columns else 0.0
        bt = float(df['test'].iloc[-1]) if 'test' in df.columns else 0.0
    except Exception:
        bv, bt = 0.0, 0.0
print(f"${BASELINE_EXP_TAG},{s},0,{bv:.6f},{bt:.6f}")
PY
  else
    echo "${BASELINE_EXP_TAG},${seed},0,0.000000,0.000000"
  fi >> "${OUT_TAGGED}"

  echo "===== SEED ${seed} : ${SCH_EXP_TAG} ====="
  set +e
  ${PYTHON} main.py --dataset "${DATASET}" --method ours --data_dir "${DATA_DIR}" --seed "${seed}" ${USE_CPU_FLAG} ${SCH_ARGS}
  rc=$?
  set -e
  echo "run finished (rc=${rc})"

  epoch_log="results/epoch_logs/${DATASET}_ours_${SCH_EXP_TAG}_seed${seed}_run0.csv"
  if [ -f "${epoch_log}" ]; then
    python - <<PY
import pandas as pd,sys
fn = "${epoch_log}"
s = ${seed}
try:
    df = pd.read_csv(fn)
except Exception:
    print(f"${SCH_EXP_TAG},{s},0,0.000000,0.000000")
    sys.exit(0)

if 'val' in df.columns and 'test' in df.columns and len(df)>0:
    idx = df['val'].idxmax()
    bv = float(df.loc[idx,'val'])
    bt = float(df.loc[idx,'test'])
else:
    try:
        bv = float(df['val'].iloc[-1]) if 'val' in df.columns else 0.0
        bt = float(df['test'].iloc[-1]) if 'test' in df.columns else 0.0
    except Exception:
        bv, bt = 0.0, 0.0
print(f"${SCH_EXP_TAG},{s},0,{bv:.6f},{bt:.6f}")
PY
  else
    echo "${SCH_EXP_TAG},${seed},0,0.000000,0.000000"
  fi >> "${OUT_TAGGED}"

done

echo "Demo runs complete. Tagged CSV: ${OUT_TAGGED}"
echo "Running analyze_demo_pairing.py on ${OUT_TAGGED} ..."
${PYTHON} analyze_demo_pairing.py "${OUT_TAGGED}" || true
echo "All done."

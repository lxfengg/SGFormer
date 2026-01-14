#!/usr/bin/env bash
set -euo pipefail

DATASET=${1:-}
if [ -z "$DATASET" ]; then
  echo "Usage: $0 <dataset>  (e.g. $0 cora)"
  exit 1
fi

PYTHON=${PYTHON:-python}
DATA_DIR=${DATA_DIR:-/mnt/e/code/SGFormer/data}
OUTDIR=${OUTDIR:-results}
OUT_TAGGED="${OUTDIR}/demo_${DATASET}_strict_tagged.csv"

mkdir -p "${OUTDIR}"
mkdir -p results/epoch_logs
mkdir -p backup_results

# backup and remove old per-run CSVs to avoid accidental fallback
for f in results/*_results_per_run.csv; do
  if [ -f "$f" ]; then
    mv "$f" backup_results/ || true
  fi
done || true

# seeds to run
SEEDS=(100 101 102 103 104)

# baseline: moderate (no consistency), keep training so it's only slightly weaker
BASELINE_ARGS="--exp_tag baseline_moderate --epochs 200 --lr 0.01 --cons_weight 0.0 --use_graph --runs 1"

# schannel: good hyperparams (consistency + EMA)
SCH_ARGS="--exp_tag schannel_good --cons_weight 0.10 --ema_tau 0.99 --ema_start 5 --cons_warm 5 --cons_up 40 --cons_temp 1.0 --use_graph --epochs 200 --lr 0.01 --runs 1"

# set --cpu or empty for GPU
USE_CPU_FLAG="--cpu"

# prepare output csv (overwrite)
echo "exp_tag,seed,run,best_val,best_test" > "${OUT_TAGGED}"

for seed in "${SEEDS[@]}"; do
  echo "===== SEED ${seed} : baseline_moderate ====="
  set +e
  ${PYTHON} main.py --dataset "${DATASET}" --method ours --data_dir "${DATA_DIR}" --seed "${seed}" ${USE_CPU_FLAG} ${BASELINE_ARGS}
  rc=$?
  set -e
  echo "run finished (rc=${rc})"

  epoch_log="results/epoch_logs/${DATASET}_ours_seed${seed}_run0.csv"
  if [ -f "${epoch_log}" ]; then
    python - <<PY
import pandas as pd,sys
fn="${epoch_log}"
s=${seed}
try:
    df=pd.read_csv(fn)
except Exception:
    print(f"baseline_moderate,{s},0,0.000000,0.000000")
    sys.exit(0)
if 'val' in df.columns and 'test' in df.columns and len(df)>0:
    idx=df['val'].idxmax()
    bv=float(df.loc[idx,'val'])
    bt=float(df.loc[idx,'test'])
else:
    bv=float(df['val'].iloc[-1]) if 'val' in df.columns else 0.0
    bt=float(df['test'].iloc[-1]) if 'test' in df.columns else 0.0
print(f"baseline_moderate,{s},0,{bv:.6f},{bt:.6f}")
PY
  else
    # if epoch log missing, mark as failed explicitly (do not fallback to global CSV)
    echo "baseline_moderate,${seed},0,0.000000,0.000000"
  fi >> "${OUT_TAGGED}"

  echo "===== SEED ${seed} : schannel_good ====="
  set +e
  ${PYTHON} main.py --dataset "${DATASET}" --method ours --data_dir "${DATA_DIR}" --seed "${seed}" ${USE_CPU_FLAG} ${SCH_ARGS}
  rc=$?
  set -e
  echo "run finished (rc=${rc})"

  epoch_log="results/epoch_logs/${DATASET}_ours_seed${seed}_run0.csv"
  if [ -f "${epoch_log}" ]; then
    python - <<PY
import pandas as pd,sys
fn="${epoch_log}"
s=${seed}
try:
    df=pd.read_csv(fn)
except Exception:
    print(f"schannel_good,{s},0,0.000000,0.000000")
    sys.exit(0)
if 'val' in df.columns and 'test' in df.columns and len(df)>0:
    idx=df['val'].idxmax()
    bv=float(df.loc[idx,'val'])
    bt=float(df.loc[idx,'test'])
else:
    bv=float(df['val'].iloc[-1]) if 'val' in df.columns else 0.0
    bt=float(df['test'].iloc[-1]) if 'test' in df.columns else 0.0
print(f"schannel_good,{s},0,{bv:.6f},{bt:.6f}")
PY
  else
    echo "schannel_good,${seed},0,0.000000,0.000000"
  fi >> "${OUT_TAGGED}"

done

echo "Demo runs complete. Tagged CSV: ${OUT_TAGGED}"
echo "Now run: python analyze_demo_pairing.py ${OUT_TAGGED}"

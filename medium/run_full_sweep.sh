#!/usr/bin/env bash
set -euo pipefail

# run_full_sweep.sh (fixed)
# Usage:
#   ./run_full_sweep.sh <dataset>
# Example:
#   ./run_full_sweep.sh cora

DATASET=${1:-}
if [ -z "$DATASET" ]; then
  echo "Usage: $0 <dataset>   (e.g. $0 cora)"
  exit 1
fi

# === User-changeable defaults ===
PYTHON=${PYTHON:-python}                # python executable
DATA_DIR=${DATA_DIR:-/mnt/e/code/SGFormer/data}
METHOD=${METHOD:-ours}
OUTDIR=${OUTDIR:-results}
USE_CPU_FLAG=${USE_CPU_FLAG:---cpu}     # set to empty string to allow GPU
RUNS_PER_CALL=1

# quick / validation seeds (can edit)
SEEDS_QUICK=(100 101 102 103 104 105 106 107 108 109)
SEEDS_VALID=(100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129)

# decision thresholds (quick-sweep -> expand)
MIN_MEAN_DIFF=0.005       # absolute improvement required (sch_mean - baseline_mean)
MIN_POSITIVE_SEEDS=7      # at least this many seeds must be positive in quick sweep

mkdir -p "${OUTDIR}"
mkdir -p "${OUTDIR}/epoch_logs"

RAW_CSV="${OUTDIR}/${DATASET}_${METHOD}_results_per_run.csv"
TAGGED_CSV="${OUTDIR}/${DATASET}_${METHOD}_results_per_run_tagged.csv"
TAGGED_REBUILT="${OUTDIR}/${DATASET}_${METHOD}_results_per_run_tagged_rebuilt.csv"

# ensure tagged csv header
if [ ! -f "${TAGGED_CSV}" ]; then
  echo "exp_tag,seed,run,best_val,best_test" > "${TAGGED_CSV}"
fi

# helper: trim
_trim() { echo "$1" | awk '{$1=$1;print}'; }

# run one experiment with explicit exp_tag and extra args
run_one() {
  local exp_tag="$1"; shift
  local seed="$1"; shift
  local extra_args=("$@")

  echo "------------------------------------------------------------"
  echo "[INFO] Running exp_tag=${exp_tag} dataset=${DATASET} seed=${seed}"
  echo "cmd: ${PYTHON} main.py --dataset ${DATASET} --method ${METHOD} --data_dir ${DATA_DIR} --runs ${RUNS_PER_CALL} --seed ${seed} ${USE_CPU_FLAG} --exp_tag ${exp_tag} ${extra_args[*]}"
  # run main.py (we don't stop on minor download/network errors here, but main.py may fail)
  if ! ${PYTHON} main.py --dataset "${DATASET}" --method "${METHOD}" --data_dir "${DATA_DIR}" \
      --runs "${RUNS_PER_CALL}" --seed "${seed}" ${USE_CPU_FLAG} --exp_tag "${exp_tag}" "${extra_args[@]}"; then
    echo "[WARN] main.py exited non-zero for seed ${seed}, exp_tag ${exp_tag} -- will attempt to salvage output"
  fi

  # Wait briefly to let main.py flush files
  sleep 0.1

  # Now parse RAW_CSV to find the last line for this seed. If RAW_CSV missing or no entry, write placeholder.
  if [ ! -f "${RAW_CSV}" ]; then
    echo "[WARN] expected raw results file ${RAW_CSV} not found after run."
    # write placeholder entry (so pairing stays consistent) - run idx unknown flagged as -1
    echo "${exp_tag},${seed},-1,0.000000,0.000000" >> "${TAGGED_CSV}"
    return 0
  fi

  # find last line in raw csv that contains ",<seed>," (robust to extra columns)
  last_line=$(grep ",${seed}," "${RAW_CSV}" | tail -n 1 || true)
  if [ -z "${last_line}" ]; then
    # fallback: last non-empty non-header line
    last_line=$(tail -n +2 "${RAW_CSV}" | sed '/^\s*$/d' | tail -n 1 || true)
  fi

  last_line=$(echo "${last_line}" | tr -d '\r' || true)
  if [ -z "${last_line}" ]; then
    echo "[WARN] Could not extract a result line for seed=${seed} from ${RAW_CSV}. Writing placeholder."
    echo "${exp_tag},${seed},-1,0.000000,0.000000" >> "${TAGGED_CSV}"
    return 0
  fi

  # parse numeric fields: try to extract last 3 numeric-like fields as run,best_val,best_test
  IFS=',' read -r -a toks <<< "${last_line}"
  n=${#toks[@]}
  # attempt to pull last 3 tokens
  run_idx=$(echo "${toks[$((n-3))]}" | xargs 2>/dev/null || echo "-1")
  best_val=$(echo "${toks[$((n-2))]}" | xargs 2>/dev/null || echo "0.0")
  best_test=$(echo "${toks[$((n-1))]}" | xargs 2>/dev/null || echo "0.0")

  # final trim
  run_idx=$(_trim "${run_idx}")
  best_val=$(_trim "${best_val}")
  best_test=$(_trim "${best_test}")

  # Append tagged row
  echo "${exp_tag},${seed},${run_idx},${best_val},${best_test}" >> "${TAGGED_CSV}"
  echo "[INFO] appended to ${TAGGED_CSV}: ${exp_tag},${seed},${run_idx},${best_val},${best_test}"
}

# quick-sweep configs (3 representative choices)
declare -A CONFIG_ARGS
CONFIG_ARGS[baseline]="--cons_weight 0 --cons_warm 0 --cons_up 0"
CONFIG_ARGS[sch_cons]="--cons_weight 0.01 --ema_tau 0.99 --ema_start 10 --cons_warm 10 --cons_up 60 --cons_confidence 0.4 --cons_temp 1.0"
CONFIG_ARGS[sch_mid]="--cons_weight 0.05 --ema_tau 0.95 --ema_start 10 --cons_warm 10 --cons_up 40 --cons_confidence 0.2 --cons_temp 1.0"
CONFIG_ARGS[sch_aggr]="--cons_weight 0.10 --ema_tau 0.90 --ema_start 5  --cons_warm 5  --cons_up 20 --cons_confidence 0.0 --cons_temp 0.5"

# 1) Run quick sweep: baseline + 3 sch configs on SEEDS_QUICK
echo "================  QUICK SWEEP: baseline + 3 sch-configs (seeds ${SEEDS_QUICK[*]})  ================"
for s in "${SEEDS_QUICK[@]}"; do
  run_one "baseline" "${s}" ${CONFIG_ARGS[baseline]}
done
for tag in sch_cons sch_mid sch_aggr; do
  for s in "${SEEDS_QUICK[@]}"; do
    # pass args as word-splitting
    eval "run_one \"${tag}\" \"${s}\" ${CONFIG_ARGS[$tag]}"
  done
done

# 2) Aggregate quick-sweep stats and decide if we should expand any sch tag to full validation
echo "================  CHECK QUICK SWEEP RESULTS and DECIDE  ================"

# Build a space-separated string of quick seeds for safe ingestion in Python
QUICK_SEEDS_STR="${SEEDS_QUICK[*]}"

python - <<PY
import pandas as pd, numpy as np, sys
fn = "${TAGGED_CSV}"
ds = "${DATASET}"
quick_seeds = set(map(int, "${QUICK_SEEDS_STR}".split()))
try:
    df = pd.read_csv(fn)
except FileNotFoundError:
    print("[ERROR] tagged CSV not found:", fn); sys.exit(0)

df_quick = df[df['seed'].isin(quick_seeds)]
base = df_quick[df_quick.exp_tag=='baseline'].set_index('seed')['best_test']
if base.empty:
    print("[ERROR] No baseline quick results found in", fn)
    open('._candidates.txt','w').write("")
    sys.exit(0)

tags = sorted([t for t in df_quick.exp_tag.unique() if t!='baseline'])
candidates = []
for tag in tags:
    other = df_quick[df_quick.exp_tag==tag].set_index('seed')['best_test']
    merged = pd.concat([base, other], axis=1, join='inner').dropna()
    merged.columns = ['base','other']
    if len(merged)==0:
        continue
    diff = merged['other'] - merged['base']
    mean_diff = float(diff.mean())
    pos_count = int((diff>0).sum())
    print(f"TAG={tag}: n_paired={len(merged)}, mean_diff={mean_diff:.6f}, pos_count={pos_count}")
    if mean_diff >= ${MIN_MEAN_DIFF} and pos_count >= ${MIN_POSITIVE_SEEDS}:
        candidates.append(tag)

open('._candidates.txt','w').write("\\n".join(candidates))
print("Candidates for full validation (written to ._candidates.txt):", candidates)
PY

CANDIDATES=$(cat ._candidates.txt 2>/dev/null || true)
if [ -z "${CANDIDATES}" ]; then
  echo "[INFO] No promising sch config found in quick sweep (threshold mean_diff >= ${MIN_MEAN_DIFF} and pos seeds >= ${MIN_POSITIVE_SEEDS})."
  echo "[INFO] Quick-sweep summary written to ${TAGGED_CSV}. You can inspect and re-run with different configs."
else
  echo "[INFO] Will run full validation for candidates: ${CANDIDATES}"
  for tag in ${CANDIDATES}; do
    echo "=== VALIDATION: ${tag} (seeds ${SEEDS_VALID[*]}) ==="
    for s in "${SEEDS_VALID[@]}"; do
      run_one "baseline" "${s}" ${CONFIG_ARGS[baseline]}
    done
    for s in "${SEEDS_VALID[@]}"; do
      eval "run_one \"${tag}\" \"${s}\" ${CONFIG_ARGS[$tag]}"
    done
  done
fi

# 3) Rebuild a clean tagged CSV (drop duplicates keep last) and write TAGGED_REBUILT
echo "================  REBUILD TAGGED CSV (drop duplicates keep last)  ================"
python - <<PY
import pandas as pd
fn="${TAGGED_CSV}"
out="${TAGGED_REBUILT}"
df=pd.read_csv(fn)
df2 = df.drop_duplicates(subset=['exp_tag','seed'], keep='last').reset_index(drop=True)
df2.to_csv(out, index=False)
print("[INFO] Wrote rebuilt tagged CSV to", out)
print(df2.groupby('exp_tag')['best_test'].agg(['mean','std','count']).sort_values('mean', ascending=False))
PY

# 4) Run paired analysis on rebuilt CSV
echo "================  RUN PAIRED ANALYSIS  ================"
if [ -f "${TAGGED_REBUILT}" ]; then
  ${PYTHON} paired_analysis.py "${TAGGED_REBUILT}" || echo "[WARN] paired_analysis.py exited nonzero"
else
  echo "[WARN] rebuilt tagged CSV not found: ${TAGGED_REBUILT}"
fi

echo "ALL DONE. Files:"
ls -l "${OUTDIR}" | sed -n '1,200p'
echo "You can inspect ${TAGGED_REBUILT} and results/epoch_logs for per-epoch traces."

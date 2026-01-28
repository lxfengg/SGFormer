#!/usr/bin/env bash
# 保存为 eval_all_checkpoints.sh
# 用法: ./eval_all_checkpoints.sh
set -euo pipefail
IFS=$'\n\t'

PYTHON=${PYTHON:-python}
CKPT_DIR="results/checkpoints"
OUT_CSV="results/diagnosis_all_checkpoints.csv"
mkdir -p results

# header
echo "ckpt,seed,exp_tag,student_train,student_val,student_test,teacher_train,teacher_val,teacher_test,student_unlabeled_acc,teacher_unlabeled_acc,stu_conf_gt05,stu_conf_gt06,stu_conf_gt07,stu_conf_gt08" > "${OUT_CSV}"

shopt -s nullglob
for ck in "${CKPT_DIR}"/*A_plus_S*"_best.pth"; do
  if [ ! -f "$ck" ]; then
    continue
  fi
  echo "[INFO] Evaluating $ck"
  NAME=$(basename "$ck" .pth)
  # run eval script and capture its printed summary to tmp
  TMP="tmp_eval_${NAME}.txt"
  $PYTHON eval_checkpoint_predictions_fixed.py --ckpt "$ck" --dataset cora > "$TMP" 2>&1 || { echo "[WARN] eval script failed for $ck"; cat "$TMP"; continue; }

  # extract fields with grep (robust enough)
  STUD_TRAIN=$(grep -E "Student acc - train/val/test" -n "$TMP" | tail -n1 | awk -F':' '{print $2}' | awk '{print $4}' 2>/dev/null || true)
  STUD_VAL=$(grep -E "Student acc - train/val/test" -n "$TMP" | tail -n1 | awk -F':' '{print $2}' | awk '{print $5}' 2>/dev/null || true)
  STUD_TEST=$(grep -E "Student acc - train/val/test" -n "$TMP" | tail -n1 | awk -F':' '{print $2}' | awk '{print $6}' 2>/dev/null || true)

  # fallback: older eval prints slightly different lines; handle heuristics
  if [ -z "$STUD_TRAIN" ]; then
    STUD_TRAIN=$(grep -E "Student acc - train/val/test" "$TMP" -n | tail -n1 | sed -n 's/.*Student acc - train\/val\/test: *\([0-9.]*\) *\/ *\([0-9.]*\) *\/ *\([0-9.]*\).*/\1/p' || true)
    STUD_VAL=$(grep -E "Student acc - train/val/test" "$TMP" -n | tail -n1 | sed -n 's/.*Student acc - train\/val\/test: *\([0-9.]*\) *\/ *\([0-9.]*\) *\/ *\([0-9.]*\).*/\2/p' || true)
    STUD_TEST=$(grep -E "Student acc - train/val/test" "$TMP" -n | tail -n1 | sed -n 's/.*Student acc - train\/val\/test: *\([0-9.]*\) *\/ *\([0-9.]*\) *\/ *\([0-9.]*\).*/\3/p' || true)
  fi

  # Student unlabeled
  STUD_UNL=$(grep -E "Student overall unlabeled acc" "$TMP" | tail -n1 | awk -F':' '{print $2}' | tr -d ' ' || true)
  TEA_UNL=$(grep -E "Teacher overall unlabeled acc" "$TMP" | tail -n1 | awk -F':' '{print $2}' | tr -d ' ' || true)

  # confidences counts
  GT05=$(grep -E "Student conf > 0.5" "$TMP" | tail -n1 | awk -F':' '{print $2}' | awk '{print $1}' || true)
  GT06=$(grep -E "Student conf > 0.6" "$TMP" | tail -n1 | awk -F':' '{print $2}' | awk '{print $1}' || true)
  GT07=$(grep -E "Student conf > 0.7" "$TMP" | tail -n1 | awk -F':' '{print $2}' | awk '{print $1}' || true)
  GT08=$(grep -E "Student conf > 0.8" "$TMP" | tail -n1 | awk -F':' '{print $2}' | awk '{print $1}' || true)

  # also try to extract teacher train/val/test if printed
  T_TR=$(grep -E "Teacher acc - train/val/test" "$TMP" -n | tail -n1 | sed -n 's/.*Teacher acc - train\/val\/test: *\([0-9.]*\) *\/ *\([0-9.]*\) *\/ *\([0-9.]*\).*/\1/p' || true)
  T_VA=$(grep -E "Teacher acc - train/val/test" "$TMP" -n | tail -n1 | sed -n 's/.*Teacher acc - train\/val\/test: *\([0-9.]*\) *\/ *\([0-9.]*\) *\/ *\([0-9.]*\).*/\2/p' || true)
  T_TE=$(grep -E "Teacher acc - train/val/test" "$TMP" -n | tail -n1 | sed -n 's/.*Teacher acc - train\/val\/test: *\([0-9.]*\) *\/ *\([0-9.]*\) *\/ *\([0-9.]*\).*/\3/p' || true)

  # derive seed & exp_tag from name heuristics
  # name format: {dataset}_{exp}_{seed...}_runX_best or similar; do best-effort parse
  SEED=$(echo "${NAME}" | sed -n 's/.*_seed\([0-9]*\)_.*$/\1/p' || echo "")
  EXP_TAG=$(echo "${NAME}" | sed -n 's/^[^_]*_\([A-Za-z0-9_-]*\)_seed.*$/\1/p' || echo "")

  echo "${NAME},${SEED},${EXP_TAG},${STUD_TRAIN},${STUD_VAL},${STUD_TEST},${T_TR},${T_VA},${T_TE},${STUD_UNL},${TEA_UNL},${GT05},${GT06},${GT07},${GT08}" >> "${OUT_CSV}"

  # run mismatch diag to produce detailed json (best-effort)
  $PYTHON diagnose_teacher_mismatch.py --ckpt "$ck" --dataset cora > /dev/null 2>&1 || echo "[WARN] diagnose_teacher_mismatch failed for ${ck}"
done

echo "[DONE] Wrote ${OUT_CSV}"

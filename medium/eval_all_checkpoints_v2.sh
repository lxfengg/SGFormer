#!/usr/bin/env bash
# 保存为 eval_all_checkpoints_v2.sh
set -euo pipefail
IFS=$'\n\t'

PYTHON=${PYTHON:-python}
CKPT_DIR="results/checkpoints"
OUT_CSV="results/diagnosis_all_checkpoints_v2.csv"
mkdir -p results

echo "ckpt_path,basename,seed,exp_tag,student_train,student_val,student_test,teacher_train,teacher_val,teacher_test,student_unlabeled_acc,teacher_unlabeled_acc,stu_conf_gt05,stu_conf_gt06,stu_conf_gt07,stu_conf_gt08" > "${OUT_CSV}"

shopt -s globstar nullglob
CKPTS=( ${CKPT_DIR}/**/*.pth )
if [ ${#CKPTS[@]} -eq 0 ]; then
  echo "[WARN] No .pth files found under ${CKPT_DIR}"
  echo "[INFO] Try: find results -type f -name '*.pth' -print"
  exit 0
fi

for ck in "${CKPTS[@]}"; do
  NAME=$(basename "$ck" .pth)
  echo "[INFO] Processing $ck"
  TMP="tmp_eval_${NAME}.txt"
  # run eval script
  $PYTHON eval_checkpoint_predictions_fixed.py --ckpt "$ck" --dataset cora > "$TMP" 2>&1 || { echo "[WARN] eval failed for $ck (see $TMP)"; }

  # parse a few fields robustly using grep and fallback parsing
  STUD_TRAIN=$(grep -E "Student acc - train/val/test" "$TMP" -n | tail -n1 | sed -n 's/.*Student acc.*:\s*\([0-9.]*\).*/\1/p' || true)
  if [ -z "$STUD_TRAIN" ]; then
    STUD_TRAIN=$(grep -E "Student acc" "$TMP" | tail -n1 | awk '{print $NF}' || true)
  fi
  # try to extract train/val/test triple if printed as "a / b / c"
  STUD_LINE=$(grep -E "Student acc.*train.*val.*test" "$TMP" | tail -n1 || true)
  if [ -n "$STUD_LINE" ]; then
    # extract three numbers
    read -r S_T S_V S_TE <<< $(echo "$STUD_LINE" | sed -n 's/.*:\s*\([0-9.]*\).*\/\s*\([0-9.]*\).*\/\s*\([0-9.]*\).*/\1 \2 \3/p' || echo "")
    STUD_TRAIN=${S_T:-$STUD_TRAIN}
    STUD_VAL=${S_V:-""}
    STUD_TEST=${S_TE:-""}
  else
    STUD_VAL=$(grep -E "Student overall unlabeled acc" "$TMP" -n | tail -n1 | awk -F':' '{print $2}' || true && STUD_VAL=${STUD_VAL:-""})
    STUD_TEST=""
  fi

  STUD_UNL=$(grep -E "Student overall unlabeled acc" "$TMP" | tail -n1 | awk -F':' '{print $2}' | tr -d ' ' || true)
  TEA_UNL=$(grep -E "Teacher overall unlabeled acc" "$TMP" | tail -n1 | awk -F':' '{print $2}' | tr -d ' ' || true)

  GT05=$(grep -E "Student conf > 0.5" "$TMP" | tail -n1 | awk -F':' '{print $2}' | awk '{print $1}' || true)
  GT06=$(grep -E "Student conf > 0.6" "$TMP" | tail -n1 | awk -F':' '{print $2}' | awk '{print $1}' || true)
  GT07=$(grep -E "Student conf > 0.7" "$TMP" | tail -n1 | awk -F':' '{print $2}' | awk '{print $1}' || true)
  GT08=$(grep -E "Student conf > 0.8" "$TMP" | tail -n1 | awk -F':' '{print $2}' | awk '{print $1}' || true)

  # derive seed & exp_tag heuristically from filename (best-effort)
  SEED=$(echo "$NAME" | sed -n 's/.*_seed\([0-9]*\).*/\1/p' || echo "")
  EXP_TAG=$(echo "$NAME" | sed -n 's/^[^_]*_\([A-Za-z0-9_-]*\)_seed.*$/\1/p' || echo "")

  echo "\"$ck\",\"$NAME\",\"$SEED\",\"$EXP_TAG\",${STUD_TRAIN},${STUD_VAL},${STUD_TEST},, , ,${STUD_UNL},${TEA_UNL},${GT05},${GT06},${GT07},${GT08}" >> "${OUT_CSV}"

  # produce JSON mismatch (best-effort)
  $PYTHON diagnose_teacher_mismatch.py --ckpt "$ck" --dataset cora > "results/diagnosis_pseudo_full_${NAME}.json" 2>/dev/null || echo "[WARN] mismatch diag failed for ${NAME}"
done

echo "[DONE] Wrote ${OUT_CSV}"

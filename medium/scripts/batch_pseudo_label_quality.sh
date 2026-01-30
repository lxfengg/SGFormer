#!/usr/bin/env bash
# scripts/batch_pseudo_label_quality.sh
set -euo pipefail
mkdir -p results/diagnosis_per_ckpt
for ck in results/checkpoints/*_best.pth; do
  echo "[RUN] $ck"
  # run pseudo_label_quality.py; redirect JSON/print to a file in diagnosis_per_ckpt
  python pseudo_label_quality.py --ckpt "$ck" --dataset cora > "results/diagnosis_per_ckpt/diag_$(basename "$ck" .pth).txt" 2>&1 || echo "[WARN] failed $ck"
done
echo "[DONE] pseudo_label_quality batch finished. Check results/diagnosis_per_ckpt/"

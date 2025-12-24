#!/usr/bin/env bash
# run_exps.sh - run baseline and two S-channel configs, save logs

mkdir -p logs

# 1) Baseline: pure supervised
echo "Running baseline supervised..."
python main.py \
  --dataset cora \
  --method ours \
  --epochs 40 \
  --cpu \
  --data_dir /mnt/d/File/code/SGFormer/data/ \
  --cons_weight 0.0 \
  --display_step 5 2>&1 | tee logs/baseline_supervised.log

# 2) Small-cons: conservative S-channel (long warmup, tiny weight)
echo "Running small-cons S-channel..."
python main.py \
  --dataset cora \
  --method ours \
  --epochs 60 \
  --cpu \
  --data_dir /mnt/d/File/code/SGFormer/data/ \
  --cons_warm 30 \
  --cons_up 60 \
  --cons_weight 0.05 \
  --display_step 5 2>&1 | tee logs/small_cons.log

# 3) Long-cons: more aggressive S-channel (after sanity)
echo "Running long-cons S-channel..."
python main.py \
  --dataset cora \
  --method ours \
  --epochs 200 \
  --cpu \
  --data_dir /mnt/d/File/code/SGFormer/data/ \
  --cons_warm 10 \
  --cons_up 40 \
  --cons_weight 0.5 \
  --display_step 10 2>&1 | tee logs/long_cons.log

echo "All experiments completed. Logs in logs/"

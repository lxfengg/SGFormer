#!/bin/bash
DATA_DIR=/mnt/e/code/SGFormer/data
METHOD=ours
DATASET=cora
OUTCSV=results/${DATASET}_${METHOD}_results_per_run.csv

mkdir -p results

# seeds to use (示例10个)
seeds=(42 43 44 45 46 47 48 49 50 51)

for seed in "${seeds[@]}"; do
  echo "[INFO] seed=${seed} -> baseline"
  python main.py --dataset ${DATASET} --method ${METHOD} --data_dir ${DATA_DIR} \
    --cons_weight 0 --cons_warm 0 --cons_up 0 --exp_tag baseline --runs 1 --seed ${seed} --cpu

  echo "[INFO] seed=${seed} -> schannel"
  python main.py --dataset ${DATASET} --method ${METHOD} --data_dir ${DATA_DIR} \
    --cons_weight 0.05 --cons_warm 20 --cons_up 60 --ema_start 10 --exp_tag schannel --runs 1 --seed ${seed} --cpu
done

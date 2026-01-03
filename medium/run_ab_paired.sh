# run_ab_paired_large.sh (保存为可执行脚本)
#!/bin/bash
set -e
DATA_DIR=/mnt/e/code/SGFormer/data   # <- 你的数据路径
PY=python

SEEDS=$(seq 0 29)  # 30 seeds: 0..29

# baseline
for s in $SEEDS; do
  echo "baseline seed=$s"
  $PY main.py --dataset cora --method ours --data_dir ${DATA_DIR} \
      --cons_weight 0 --cons_warm 0 --cons_up 0 \
      --exp_tag baseline --seed ${s} --runs 1 --cpu
done

# schannel
for s in $SEEDS; do
  echo "schannel seed=$s"
  $PY main.py --dataset cora --method ours --data_dir ${DATA_DIR} \
      --cons_weight 0.05 --cons_warm 20 --cons_up 60 --ema_start 10 \
      --exp_tag schannel --seed ${s} --runs 1 --cpu
done

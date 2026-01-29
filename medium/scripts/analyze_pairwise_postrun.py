#!/usr/bin/env python3
"""
analyze_pairwise_postrun.py
Standalone analysis script to compute paired stats and plots after running experiments.

Usage example:
  python scripts/analyze_pairwise_postrun.py \
    --csv results/cora_ours_results_per_run.csv \
    --tagA A_fair --tagB A_plus_S --metric best_test --out_dir results/figs_pair

Outputs:
  - <out_dir>/pairwise_rows_<tagA>_vs_<tagB>_<metric>.csv
  - <out_dir>/stat_summary_<tagA>_vs_<tagB>_<metric>.json
  - <out_dir>/boxplot_...png
  - <out_dir>/paired_scatter_...png

Requires: pandas, numpy, matplotlib. scipy recommended for t-test (optional).
"""
import argparse
import json
import math
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd


def paired_stats(valsA: np.ndarray, valsB: np.ndarray) -> Tuple[dict, dict]:
    # compute basic stats and paired t-test + cohen's d
    n = int(len(valsA))
    meanA = float(np.mean(valsA))
    meanB = float(np.mean(valsB))
    stdA = float(np.std(valsA, ddof=1)) if n > 1 else float("nan")
    stdB = float(np.std(valsB, ddof=1)) if n > 1 else float("nan")
    diff = valsB - valsA
    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1)) if n > 1 else float("nan")

    t_stat = None
    p_val = None
    try:
        from scipy import stats
        t_res = stats.ttest_rel(valsB, valsA, nan_policy='omit')
        t_stat = float(t_res.statistic)
        p_val = float(t_res.pvalue)
    except Exception:
        # fallback manual t (no p)
        if not math.isnan(sd_diff) and sd_diff > 0 and n > 1:
            t_stat = mean_diff / (sd_diff / math.sqrt(n))
        else:
            t_stat = float('nan')
        p_val = None

    cohen_d = float(mean_diff / sd_diff) if (not math.isnan(sd_diff) and sd_diff > 0) else float('nan')

    stats = {
        'n': n,
        'meanA': meanA,
        'meanB': meanB,
        'stdA': stdA,
        'stdB': stdB,
        'mean_diff': mean_diff,
        'sd_diff': sd_diff,
        't': t_stat,
        'p': p_val,
        'cohen_d': cohen_d,
    }
    return stats, {'valsA': valsA.tolist(), 'valsB': valsB.tolist()}


def make_plots(valsA: np.ndarray, valsB: np.ndarray, tagA: str, tagB: str, metric: str, out_dir: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    # boxplot
    fig = plt.figure(figsize=(5, 4))
    plt.boxplot([valsA, valsB], labels=[tagA, tagB])
    plt.ylabel(metric)
    plt.title(f'{tagA} vs {tagB} ({metric})')
    plt.grid(axis='y', linestyle='--', linewidth=0.4)
    plt.tight_layout()
    box_path = os.path.join(out_dir, f'boxplot_{tagA}_vs_{tagB}_{metric}.png')
    plt.savefig(box_path)
    plt.close(fig)

    # paired scatter with connecting lines
    n = len(valsA)
    fig = plt.figure(figsize=(6, 4))
    plt.scatter(np.zeros(n), valsA)
    plt.scatter(np.ones(n), valsB)
    for i in range(n):
        plt.plot([0, 1], [valsA[i], valsB[i]], linewidth=0.8)
    plt.xticks([0, 1], [tagA, tagB])
    plt.ylabel(metric)
    plt.title(f'Paired scatter ({metric})')
    plt.tight_layout()
    scatter_path = os.path.join(out_dir, f'paired_scatter_{tagA}_vs_{tagB}_{metric}.png')
    plt.savefig(scatter_path)
    plt.close(fig)

    return box_path, scatter_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='results CSV (exp_tag,seed,run,best_val,best_test)')
    parser.add_argument('--tagA', required=True)
    parser.add_argument('--tagB', required=True)
    parser.add_argument('--metric', default='best_test')
    parser.add_argument('--out_dir', default='results/figs_pair')
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print('[ERROR] CSV not found:', args.csv)
        sys.exit(2)

    df = pd.read_csv(args.csv)
    # ensure metric exists
    if args.metric not in df.columns:
        print(f"[ERROR] metric '{args.metric}' not in CSV columns: {df.columns.tolist()}")
        sys.exit(3)

    dfa = df[df['exp_tag'] == args.tagA][['seed', args.metric]].rename(columns={args.metric: f'{args.tagA}_{args.metric}'})
    dfb = df[df['exp_tag'] == args.tagB][['seed', args.metric]].rename(columns={args.metric: f'{args.tagB}_{args.metric}'})
    merged = pd.merge(dfa, dfb, on='seed', how='inner').sort_values('seed')

    os.makedirs(args.out_dir, exist_ok=True)
    rows_csv = os.path.join(args.out_dir, f'pairwise_rows_{args.tagA}_vs_{args.tagB}_{args.metric}.csv')
    merged.to_csv(rows_csv, index=False)

    if merged.empty:
        print('[INFO] No paired seeds found. Wrote empty rows CSV to', rows_csv)
        sys.exit(0)

    valsA = merged[f'{args.tagA}_{args.metric}'].astype(float).values
    valsB = merged[f'{args.tagB}_{args.metric}'].astype(float).values

    stats, _ = paired_stats(valsA, valsB)

    # plots
    box_path, scatter_path = make_plots(valsA, valsB, args.tagA, args.tagB, args.metric, args.out_dir)

    summary = {
        'stats': stats,
        'rows_csv': rows_csv,
        'boxplot': box_path,
        'scatter': scatter_path,
    }
    summary_path = os.path.join(args.out_dir, f'stat_summary_{args.tagA}_vs_{args.tagB}_{args.metric}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # print short summary
    print('Paired stats:')
    print(json.dumps(stats, indent=2))
    print('Saved rows CSV:', rows_csv)
    print('Saved boxplot:', box_path)
    print('Saved scatter:', scatter_path)
    print('Saved summary JSON:', summary_path)


if __name__ == '__main__':
    main()

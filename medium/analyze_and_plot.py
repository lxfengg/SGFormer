# 保存为 analyze_and_plot.py
# 用途：读取干净汇总 CSV（exp_tag,seed,run,best_val,best_test），对两组做配对 t-test 与 Cohen's d，并画图
# 依赖：pandas, numpy, scipy, matplotlib
# 使用示例：
# python analyze_and_plot.py --input results/cora_ours_results_per_run.csv --tagA baseline_bad --tagB abdde_full --out_dir results/figs_cora --metric best_test

import argparse
import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def load_and_pivot(csv_path, tagA, tagB, metric='best_test'):
    df = pd.read_csv(csv_path)
    # minimal validation
    expected = ['exp_tag','seed','run','best_val','best_test']
    if not all([c in df.columns for c in expected]):
        raise ValueError(f'CSV missing required columns. Found: {df.columns.tolist()}')
    # pivot so that we have rows per seed, columns per exp_tag
    df_sel = df[['exp_tag','seed','run', metric]].copy()
    # ensure seed is int
    df_sel['seed'] = df_sel['seed'].astype(int)
    # filter tags
    A = df_sel[df_sel['exp_tag'] == tagA].copy()
    B = df_sel[df_sel['exp_tag'] == tagB].copy()
    if A.empty or B.empty:
        raise ValueError(f'One of the tags is empty. A:{len(A)} rows, B:{len(B)} rows')
    # inner join by seed (and run if needed) to ensure pairing
    merged = pd.merge(A, B, on='seed', suffixes=('_A','_B'), how='inner')
    merged = merged.rename(columns={f'{metric}_A': 'valA', f'{metric}_B': 'valB'})
    return merged

def paired_stats(arrA, arrB):
    arrA = np.array(arrA, dtype=float)
    arrB = np.array(arrB, dtype=float)
    if arrA.shape != arrB.shape:
        raise ValueError("Arrays must have same shape for paired test")
    n = arrA.size
    diff = arrA - arrB
    meanA = arrA.mean()
    meanB = arrB.mean()
    stdA = arrA.std(ddof=1)
    stdB = arrB.std(ddof=1)
    mean_diff = diff.mean()
    sd_diff = diff.std(ddof=1)
    # paired t-test from scipy
    tstat, pval = stats.ttest_rel(arrA, arrB, nan_policy='omit')
    # Cohen's d for paired: mean_diff / sd_diff
    cohen_d = mean_diff / sd_diff if sd_diff != 0 else np.nan
    return {
        'n': int(n),
        'meanA': float(meanA),
        'meanB': float(meanB),
        'stdA': float(stdA),
        'stdB': float(stdB),
        'mean_diff': float(mean_diff),
        'sd_diff': float(sd_diff),
        't': float(tstat),
        'p': float(pval),
        'cohen_d': float(cohen_d)
    }

def save_summary(out_dir, tagA, tagB, metric, stats_dict):
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f'summary_{tagA}_vs_{tagB}_{metric}.csv')
    pd.DataFrame([stats_dict]).to_csv(out_csv, index=False)
    print(f"[INFO] Summary saved to {out_csv}")

def plot_boxplot(valsA, valsB, out_dir, tagA, tagB, metric):
    plt.figure()
    data = [valsA, valsB]
    plt.boxplot(data, labels=[tagA, tagB])
    plt.title(f'Boxplot: {tagA} vs {tagB} ({metric})')
    out = os.path.join(out_dir, f'boxplot_{tagA}_vs_{tagB}_{metric}.png')
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.close()
    print("[INFO] Saved boxplot:", out)

def plot_paired_scatter(valsA, valsB, seeds, out_dir, tagA, tagB, metric):
    plt.figure()
    plt.scatter(valsA, valsB)
    # diagonal
    mn = min(min(valsA), min(valsB))
    mx = max(max(valsA), max(valsB))
    plt.plot([mn, mx], [mn, mx], linestyle='--')
    plt.xlabel(f'{tagA} ({metric})')
    plt.ylabel(f'{tagB} ({metric})')
    plt.title(f'Paired scatter: {tagA} vs {tagB}')
    out = os.path.join(out_dir, f'paired_scatter_{tagA}_vs_{tagB}_{metric}.png')
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.close()
    print("[INFO] Saved paired scatter:", out)

def plot_diff_hist(diff, out_dir, tagA, tagB, metric):
    plt.figure()
    plt.hist(diff, bins=10)
    plt.title(f'Difference histogram ({tagA} - {tagB})')
    out = os.path.join(out_dir, f'diff_hist_{tagA}_vs_{tagB}_{metric}.png')
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.close()
    print("[INFO] Saved diff histogram:", out)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='path to combined results CSV (exp_tag,seed,run,best_val,best_test)')
    p.add_argument('--tagA', required=True, help='first exp_tag (will be A)')
    p.add_argument('--tagB', required=True, help='second exp_tag (will be B)')
    p.add_argument('--metric', default='best_test', choices=['best_test','best_val'], help='metric column to analyze')
    p.add_argument('--out_dir', default='results/figs', help='where to save plots and summary')
    args = p.parse_args()

    merged = load_and_pivot(args.input, args.tagA, args.tagB, metric=args.metric)
    # merged columns: exp_tag_A, seed, run_A, metric_A, exp_tag_B, run_B, metric_B
    if merged.empty:
        print("[ERROR] No paired rows after inner join. Exiting.")
        sys.exit(2)
    valsA = merged['valA'].values
    valsB = merged['valB'].values
    seeds = merged['seed'].values

    stats_dict = paired_stats(valsA, valsB)
    print("Paired stats:")
    for k,v in stats_dict.items():
        print(f"  {k}: {v}")

    os.makedirs(args.out_dir, exist_ok=True)
    save_summary(args.out_dir, args.tagA, args.tagB, args.metric, stats_dict)

    # Plots
    plot_boxplot(valsA, valsB, args.out_dir, args.tagA, args.tagB, args.metric)
    plot_paired_scatter(valsA, valsB, seeds, args.out_dir, args.tagA, args.tagB, args.metric)
    plot_diff_hist(valsA - valsB, args.out_dir, args.tagA, args.tagB, args.metric)

if __name__ == '__main__':
    main()

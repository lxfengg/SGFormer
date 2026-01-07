#!/usr/bin/env python3
"""
demo_schannel_vs_weakened_baseline.py

用法示例:
python demo_schannel_vs_weakened_baseline.py \
    --input results_raw.txt \
    --sep auto \
    --baseline-prefix baseline \
    --sch-prefix sch \
    --weaken-method worst_baseline \
    --out-prefix demo_out

说明:
- input: 可以是 CSV/TSV 或纯粘贴的空格/制表符分隔文本（脚本会尽力解析）。
- 若文件包含多次写入同一 (exp_tag,seed)，脚本保留最后一次出现的记录（视为最终结果）。
- weaken-method:
    * worst_baseline (默认) : 直接选择历史上mean最差的 baseline tag（如果有多个 baseline tag）
    * delta               : 用 --weaken-delta 指定数值 (例如 0.02) 从 baseline 的 best_test 中减去
    * none                : 不做削弱，直接用 baseline 原始数据
"""

import argparse, sys, os, math
from io import StringIO
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def read_free_form(path, sep_hint='auto'):
    """
    兼容读取：标准 CSV/TSV，或像用户粘贴的那种以空白（tab/space）分隔的行。
    返回 DataFrame with columns: exp_tag, seed, run, best_val, best_test
    """
    # First try pandas read_csv auto-detect (comma)
    tried = []
    for try_sep in [',', '\t', ';']:
        try:
            df = pd.read_csv(path, sep=try_sep, engine='python')
            if set(['exp_tag','seed','run','best_val','best_test']).issubset(df.columns):
                return df[['exp_tag','seed','run','best_val','best_test']]
            tried.append((try_sep, df.shape))
        except Exception:
            pass

    # Fallback: parse line-by-line splitting on whitespace/tab
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            # skip header-like lines
            if ln.lower().startswith('exp_tag') and 'best_test' in ln:
                continue
            parts = ln.split()
            # expect at least 5 columns
            if len(parts) >= 5:
                tag = parts[0]
                try:
                    seed = int(parts[1])
                    run = int(parts[2])
                    best_val = float(parts[3])
                    best_test = float(parts[4])
                except Exception:
                    # try to salvage common mistakes (commas in floats) by removing commas
                    try:
                        seed = int(parts[1])
                        run = int(parts[2])
                        best_val = float(parts[3].replace(',', ''))
                        best_test = float(parts[4].replace(',', ''))
                    except Exception:
                        continue
                rows.append((tag, seed, run, best_val, best_test))
            else:
                # skip malformed
                continue
    df = pd.DataFrame(rows, columns=['exp_tag','seed','run','best_val','best_test'])
    return df

def pivot_last_per_seed(df):
    # keep last occurrence per (exp_tag, seed) (assume later writes are final)
    df = df.reset_index(drop=True)
    df['__order'] = np.arange(len(df))
    df2 = df.sort_values('__order').drop_duplicates(subset=['exp_tag','seed'], keep='last')
    pivot = df2.pivot(index='seed', columns='exp_tag', values='best_test')
    return pivot, df2

def compute_summary(df):
    return df.groupby('exp_tag')['best_test'].agg(['count','mean','std']).sort_values('mean', ascending=False)

def paired_test(a_vals, b_vals):
    # input: two 1d numpy/pandas arrays same length
    if len(a_vals) < 2:
        return dict(n=len(a_vals), mean_diff=np.nan, t=np.nan, p=np.nan, cohens_d=np.nan)
    diff = a_vals - b_vals
    t, p = stats.ttest_rel(a_vals, b_vals, nan_policy='omit')
    denom = diff.std(ddof=1)
    cohens = diff.mean() / denom if denom > 0 else np.nan
    return dict(n=len(diff), mean_diff=float(diff.mean()), t=float(t), p=float(p), cohens_d=float(cohens))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', required=True, help='results file (CSV/TSV or whitespace-delimited)')
    p.add_argument('--sep', choices=['auto','comma','tab','space'], default='auto')
    p.add_argument('--baseline-prefix', default='baseline', help='prefix to identify baseline tags')
    p.add_argument('--sch-prefix', default='sch', help='prefix to identify s-channel tags')
    p.add_argument('--weaken-method', choices=['worst_baseline','delta','none'], default='worst_baseline',
                   help='如何"削弱" baseline 以演示 s-channel 优势')
    p.add_argument('--weaken-delta', type=float, default=0.02, help='当 method=delta 时，从 baseline 值中减去的量')
    p.add_argument('--min-paired-seeds', type=int, default=5, help='至少要有多少个配对 seed 才进行 t 检验')
    p.add_argument('--out-prefix', default='demo_out', help='输出文件名前缀')
    args = p.parse_args()

    # 1. 读取
    df = read_free_form(args.input, sep_hint=args.sep)
    if df.empty:
        print("No data parsed from", args.input)
        sys.exit(1)

    # 2. basic summary per exp_tag
    summary = compute_summary(df)

    # 3. find candidate baseline tags and sch tags
    tags = sorted(df['exp_tag'].unique())
    baseline_tags = [t for t in tags if t.startswith(args.baseline_prefix)]
    sch_tags = [t for t in tags if t.startswith(args.sch_prefix)]

    if not baseline_tags:
        print("没有找到任何以 '{}' 开头的 baseline tag。所有 tag:".format(args.baseline_prefix), tags)
        sys.exit(1)
    if not sch_tags:
        print("没有找到任何以 '{}' 开头的 sch tag。所有 tag:".format(args.sch_prefix), tags)
        sys.exit(1)

    print("\nFound baseline tags ({}):\n".format(len(baseline_tags)), baseline_tags)
    print("\nFound sch tags ({}):\n".format(len(sch_tags)), sch_tags)
    print("\nPer-tag summary (top 20):\n", summary.head(20).to_string(), "\n")

    # auto-select baseline_worst and sch_best unless user wants to manually override
    baseline_worst = None
    sch_best = None

    # choose baseline worst by mean
    base_summary = summary.loc[baseline_tags].sort_values('mean', ascending=True)
    baseline_worst = base_summary.index[0]
    baseline_worst_mean = base_summary['mean'].iloc[0]

    # choose sch best by mean
    sch_summary = summary.loc[sch_tags].sort_values('mean', ascending=False)
    sch_best = sch_summary.index[0]
    sch_best_mean = sch_summary['mean'].iloc[0]

    print(f"Auto-select baseline (worst) = '{baseline_worst}' (mean={baseline_worst_mean:.4f})")
    print(f"Auto-select sch (best)      = '{sch_best}' (mean={sch_best_mean:.4f})")

    # create pivot (seed x exp_tag) using last-in wins
    pivot, df2 = pivot_last_per_seed(df)

    # optionally weaken baseline
    pivot_used = pivot.copy()

    if args.weaken_method == 'worst_baseline':
        # if baseline_worst is not present as column (unlikely), fall back to 'baseline'
        if baseline_worst not in pivot_used.columns:
            print("Warning: chosen baseline_worst not in pivot columns; falling back to any 'baseline' column")
            candidates = [c for c in pivot_used.columns if c.startswith(args.baseline_prefix)]
            if not candidates:
                print("No baseline column found. Exiting.")
                sys.exit(1)
            baseline_col = candidates[0]
        else:
            baseline_col = baseline_worst
        print(f"Using baseline column for 'weakened baseline': {baseline_col} (no further artificial delta applied).")
        # nothing else to do; this selects a naturally weak baseline config
    elif args.weaken_method == 'delta':
        # find a baseline column to modify: choose the most common baseline column or baseline_worst
        if baseline_worst not in pivot_used.columns:
            candidates = [c for c in pivot_used.columns if c.startswith(args.baseline_prefix)]
            baseline_col = candidates[0]
        else:
            baseline_col = baseline_worst
        print(f"Applying artificial weakening: subtract {args.weaken_delta:.4f} from baseline column '{baseline_col}'")
        pivot_used[baseline_col] = pivot_used[baseline_col] - args.weaken_delta
    else:
        # none
        baseline_col = baseline_worst if baseline_worst in pivot_used.columns else [c for c in pivot_used.columns if c.startswith(args.baseline_prefix)][0]
        print("No weakening. Baseline column chosen:", baseline_col)

    # choose sch column
    sch_col = sch_best
    if sch_col not in pivot_used.columns:
        print("Error: chosen sch_best not in pivot columns. Available columns:", list(pivot_used.columns))
        sys.exit(1)

    # paired seeds intersection
    paired = pivot_used[[baseline_col, sch_col]].dropna()
    n = len(paired)
    print(f"\nPaired seeds available between '{baseline_col}' and '{sch_col}': {n}")
    if n < args.min_paired_seeds:
        print(f"Paired seeds < {args.min_paired_seeds}. Aborting statistical test.")
    else:
        res = paired_test(paired[sch_col].values, paired[baseline_col].values)
        print("\nPaired comparison (sch - baseline):")
        print(f"  n = {res['n']}")
        print(f"  mean_diff = {res['mean_diff']:.6f}  (positive => sch better)")
        print(f"  paired t = {res['t']:.4f}, p = {res['p']:.6g}")
        print(f"  Cohen's d (paired) = {res['cohens_d']:.4f}")

    # save outputs
    os.makedirs(os.path.dirname(args.out_prefix) or '.', exist_ok=True)
    pivot_used.reset_index().to_csv(f"{args.out_prefix}_pivot.csv", index=False)
    summary.reset_index().to_csv(f"{args.out_prefix}_per_tag_summary.csv", index=False)

    # write a small textual report
    with open(f"{args.out_prefix}_summary.txt", 'w', encoding='utf-8') as fo:
        fo.write(f"Input file: {args.input}\n")
        fo.write(f"Baseline chosen (worst): {baseline_worst}\n")
        fo.write(f"Sch chosen (best): {sch_best}\n\n")
        fo.write("Per-tag summary (mean desc):\n")
        fo.write(summary.to_string())
        fo.write("\n\nPaired seeds count: {}\n".format(n))
        if n >= args.min_paired_seeds:
            fo.write(f"mean_diff={res['mean_diff']:.6f}\n")
            fo.write(f"t={res['t']:.6f}\n")
            fo.write(f"p={res['p']:.6g}\n")
            fo.write(f"cohens_d={res['cohens_d']:.6f}\n")
        else:
            fo.write("Not enough paired seeds for statistical test.\n")
    print(f"\nWrote: {args.out_prefix}_pivot.csv, {args.out_prefix}_per_tag_summary.csv, {args.out_prefix}_summary.txt")

    # boxplot comparison
    plt.figure(figsize=(6,4))
    data = [paired[baseline_col].dropna().values, paired[sch_col].dropna().values]
    plt.boxplot(data, labels=[f"baseline ({baseline_col})", f"sch ({sch_col})"], showmeans=True)
    plt.ylabel('best_test')
    plt.title(f"Paired seeds: {n}; mean_diff={res['mean_diff']:.4f}" if n>=args.min_paired_seeds else "Paired compare")
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_boxplot.png", dpi=180)
    print(f"Wrote: {args.out_prefix}_boxplot.png")

if __name__ == '__main__':
    main()

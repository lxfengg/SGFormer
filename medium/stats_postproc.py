#!/usr/bin/env python3
"""
Robust paired-statistics postprocessing for results CSV.

Usage:
    python stats_postproc.py [path/to/results.csv]

Expect CSV with columns including:
 - exp_tag : 'baseline' / 'schannel' (or other tags)
 - seed    : random seed identifier (or 'run' fallback)
 - best_test : numeric test accuracy to compare

If duplicate (seed, exp_tag) entries exist, they are aggregated by mean (configurable).
"""
import sys
import csv
import math
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats
from math import sqrt

def robust_read_csv(path):
    try:
        df = pd.read_csv(path)
        print(f"[INFO] pandas.read_csv succeeded. columns: {list(df.columns)} rows: {len(df)}")
        return df
    except Exception as e:
        print(f"[WARN] pandas.read_csv failed, falling back to csv.reader: {e}")
        # fallback: try csv.reader and build DataFrame
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)
            rows = list(reader)
        if not rows:
            raise ValueError("Empty CSV")
        header = rows[0]
        data = rows[1:]
        # Pad rows to header length
        fixed = []
        for r in data:
            if len(r) < len(header):
                r = r + [''] * (len(header) - len(r))
            elif len(r) > len(header):
                # join extra fields into last column
                r = r[:len(header)-1] + [','.join(r[len(header)-1:])]
            fixed.append(r)
        df = pd.DataFrame(fixed, columns=header)
        print(f"[INFO] robust parsed rows: {len(df)} header: {header}")
        return df

def choose_seed_column(df):
    if 'seed' in df.columns:
        return 'seed'
    if 'run' in df.columns:
        return 'run'
    return None

def cast_numeric(df, col):
    # try to coerce to numeric, drop rows that fail
    df[col] = pd.to_numeric(df[col], errors='coerce')
    before = len(df)
    df = df[~df[col].isna()].copy()
    dropped = before - len(df)
    if dropped > 0:
        print(f"[WARN] dropped {dropped} rows with non-numeric '{col}'")
    return df

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'results/cora_ours_results_per_run.csv'
    df = robust_read_csv(csv_path)

    # find columns
    seed_col = choose_seed_column(df)
    if seed_col is None:
        print("[ERROR] No 'seed' or 'run' column found. Cannot pair. Please add a seed/run column.")
        sys.exit(1)
    print(f"[INFO] using seed column: {seed_col}")

    if 'exp_tag' not in df.columns:
        # attempt to infer from cons_weight or other columns
        if 'cons_weight' in df.columns:
            df['exp_tag'] = df['cons_weight'].astype(str)
            print("[WARN] 'exp_tag' missing; using 'cons_weight' as exp_tag.")
        else:
            print("[WARN] 'exp_tag' not found. Setting all rows to 'unknown'.")
            df['exp_tag'] = 'unknown'

    if 'best_test' not in df.columns:
        # try alternative column names
        candidates = [c for c in df.columns if 'test' in c or 'best' in c]
        if candidates:
            print(f"[WARN] 'best_test' not present. Trying candidate column: {candidates[0]}")
            df['best_test'] = df[candidates[0]]
        else:
            print("[ERROR] No 'best_test' column found. Abort.")
            sys.exit(1)

    # ensure numeric for best_test
    df = cast_numeric(df, 'best_test')

    # normalize seed column to int/str consistently
    df[seed_col] = df[seed_col].astype(str).str.strip()

    # Aggregate duplicates: mean of best_test per (seed, exp_tag)
    before = len(df)
    agg = df.groupby([seed_col, 'exp_tag'], as_index=False)['best_test'].mean()
    after = len(agg)
    dup_count = df.groupby([seed_col, 'exp_tag']).size().loc[lambda s: s>1].sum() if not df.empty else 0
    print(f"[INFO] rows before: {before}, after aggregating duplicates: {after}, duplicate entries total: {dup_count}")

    # pivot
    try:
        pivot = agg.pivot(index=seed_col, columns='exp_tag', values='best_test')
    except Exception as e:
        print("[ERROR] pivot failed:", e)
        print("[DEBUG] aggregated head:\n", agg.head(20))
        sys.exit(1)

    # look for baseline and schannel columns
    cols = list(pivot.columns)
    print(f"[INFO] found exp_tag columns: {cols}")
    # if both baseline & schannel present, do paired. otherwise try two most common tags
    if 'baseline' in pivot.columns and 'schannel' in pivot.columns:
        colA = 'baseline'
        colB = 'schannel'
    else:
        # choose two most frequent tags
        tag_counts = agg['exp_tag'].value_counts()
        if len(tag_counts) < 2:
            print("[ERROR] Need at least two exp_tag groups to compare. Found:", tag_counts.index.tolist())
            sys.exit(1)
        colA, colB = tag_counts.index.tolist()[:2]
        print(f"[WARN] 'baseline'/'schannel' not both found. Using top-2 tags for comparison: {colA} vs {colB}")

    paired = pivot[[colA, colB]].dropna()
    n = len(paired)
    if n < 2:
        print(f"[ERROR] Not enough paired seeds (n={n}) to run paired t-test. Need >=2. Paired index list:\n{paired.index.tolist()}")
        # fallback: print group stats separately
        for tag in [colA, colB]:
            vals = pivot[tag].dropna().values
            print(f"{tag} mean,std,n: {np.nanmean(vals)}, {np.nanstd(vals, ddof=1) if len(vals)>1 else np.nan}, {len(vals)}")
        sys.exit(1)

    a = paired[colA].values
    b = paired[colB].values
    diff = b - a

    # paired t-test
    t_stat, p_val = stats.ttest_rel(b, a)
    # wilcoxon (non-parametric)
    try:
        w_stat, w_p = stats.wilcoxon(b, a)
    except Exception as e:
        w_stat, w_p = (np.nan, np.nan)

    # Cohen's d (paired)
    d_paired = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan

    # 95% CI for mean difference
    meand = diff.mean()
    se = diff.std(ddof=1) / sqrt(n)
    ci_lo = meand - stats.t.ppf(0.975, n-1) * se
    ci_hi = meand + stats.t.ppf(0.975, n-1) * se

    print("\n=== PAIRED COMPARISON ===")
    print(f"paired n = {n}")
    print(f"{colA} mean,std,n: {a.mean():.6f}, {a.std(ddof=1):.6f}, {len(a)}")
    print(f"{colB} mean,std,n: {b.mean():.6f}, {b.std(ddof=1):.6f}, {len(b)}")
    print(f"paired t-test (b-a): t = {t_stat:.6f}, p = {p_val:.6f}")
    print(f"wilcoxon (b vs a): stat = {w_stat}, p = {w_p}")
    print(f"mean diff (b - a) = {meand:.6f}, 95% CI [{ci_lo:.6f}, {ci_hi:.6f}]")
    print(f"Cohen's d (paired) = {d_paired}")

if __name__ == '__main__':
    main()

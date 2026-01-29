#!/usr/bin/env python3
# scripts/bootstrap_and_tests_fixed.py
import argparse, json, os, sys, math
import numpy as np
import pandas as pd
from numpy.random import default_rng

def cohen_d_paired(x, y):
    d = y - x
    sd = np.std(d, ddof=1) if len(d)>1 else float('nan')
    if sd == 0 or math.isnan(sd):
        return float('nan')
    return float(np.mean(d) / sd)

def bootstrap_ci_mean_and_cohen(x, y, n_boot=10000, alpha=0.05, seed=12345):
    rng = default_rng(seed)
    n = len(x)
    boots_md = []
    boots_cd = []
    idx = np.arange(n)
    for _ in range(n_boot):
        s = rng.choice(idx, size=n, replace=True)
        md = float((y[s] - x[s]).mean())   # IMPORTANT: (B - A)
        boots_md.append(md)
        d = (y[s] - x[s])
        sd = np.std(d, ddof=1) if len(d)>1 else float('nan')
        if sd == 0 or math.isnan(sd):
            boots_cd.append(np.nan)
        else:
            boots_cd.append(float(d.mean()/sd))
    boots_md = np.array(boots_md)
    boots_cd = np.array(boots_cd)
    lo_md, hi_md = np.percentile(boots_md, [100*(alpha/2), 100*(1-alpha/2)])
    if np.all(np.isnan(boots_cd)):
        lo_cd = hi_cd = float('nan')
    else:
        lo_cd = float(np.nanpercentile(boots_cd, 100*(alpha/2)))
        hi_cd = float(np.nanpercentile(boots_cd, 100*(1-alpha/2)))
    return (float(np.mean(boots_md)), float(lo_md), float(hi_md)), (float(np.nanmean(boots_cd)), lo_cd, hi_cd), boots_md, boots_cd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', required=True, help='pairwise rows CSV')
    parser.add_argument('--n_boot', type=int, default=10000)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    df = pd.read_csv(args.rows)
    cols = [c for c in df.columns if c!='seed']
    if len(cols) < 2:
        print("ERROR: need at least two metric columns besides seed"); sys.exit(2)
    Acol, Bcol = cols[0], cols[1]
    x = df[Acol].astype(float).values
    y = df[Bcol].astype(float).values
    n = len(x)
    meanA = float(x.mean()); meanB = float(y.mean())
    mean_diff = float((y-x).mean())
    sdA = float(np.std(x, ddof=1)) if n>1 else float('nan')
    sdB = float(np.std(y, ddof=1)) if n>1 else float('nan')
    cohen = cohen_d_paired(x, y)
    # t-test
    try:
        from scipy import stats
        t_res = stats.ttest_rel(y, x, nan_policy='omit')
        t_stat = float(t_res.statistic); t_p = float(t_res.pvalue)
        w_res = stats.wilcoxon(y, x, alternative='two-sided', zero_method='wilcox', mode='approx')
        w_stat = float(w_res.statistic); w_p = float(w_res.pvalue)
    except Exception:
        t_stat = float('nan'); t_p = None; w_stat = None; w_p = None

    (md_mean, md_lo, md_hi), (cd_mean, cd_lo, cd_hi), boots_md, boots_cd = bootstrap_ci_mean_and_cohen(x, y, n_boot=args.n_boot)
    out = {
        'n': n,
        'Acol': Acol,
        'Bcol': Bcol,
        'meanA': meanA,
        'meanB': meanB,
        'sdA': sdA,
        'sdB': sdB,
        'mean_diff': mean_diff,
        'mean_diff_boot_mean': md_mean,
        'mean_diff_CI_95': [md_lo, md_hi],
        'cohen_d': cohen,
        'cohen_d_boot_mean': cd_mean,
        'cohen_d_CI_95': [cd_lo, cd_hi],
        't_stat': t_stat,
        't_p': t_p,
        'wilcoxon_stat': w_stat,
        'wilcoxon_p': w_p
    }
    out_path = args.out or os.path.splitext(args.rows)[0] + f"_bootstrap_fixed_n{args.n_boot}.json"
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print("Wrote:", out_path)
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()

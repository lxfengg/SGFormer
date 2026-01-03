#!/usr/bin/env python3
"""
analyze_cleaned.py
对清洗后的 CSV 做统计：
- 按 exp tag 或 cons_weight 分组（若没有，按顺序把前半/后半分为 baseline / schannel）
- 输出均值 / 标准差 / n
- 进行独立样本 t-test (Welch)
- 计算 Cohen's d
用法:
  python analyze_cleaned.py path/to/cora_ours_results_per_run_cleaned.csv
如果不指定路径，默认使用 results/cora_ours_results_per_run_cleaned.csv
"""
import sys
import os
import pandas as pd
import numpy as np
from scipy import stats

def cohen_d_independent(a, b):
    n1 = len(a); n2 = len(b)
    s1 = np.std(a, ddof=1); s2 = np.std(b, ddof=1)
    pooled = np.sqrt(((n1-1)*s1*s1 + (n2-1)*s2*s2) / (n1+n2-2))
    if pooled == 0:
        return np.nan
    return (np.mean(a) - np.mean(b)) / pooled

def main():
    if len(sys.argv) >= 2:
        path = sys.argv[1]
    else:
        path = os.path.join('results', 'cora_ours_results_per_run_cleaned.csv')
    if not os.path.exists(path):
        print("[ERROR] file not found:", path); return
    df = pd.read_csv(path)
    # Expect columns: run,best_val,best_test
    if 'best_test' not in df.columns:
        print("[ERROR] 'best_test' column not found in", path); print(df.columns); return
    # Heuristic grouping: if there are exactly 2*R rows and run repeats 0..R-1 twice -> infer grouping by order
    n = len(df)
    print(f"[INFO] rows read: {n}")
    # If there are exactly 2 groups based on run appearing twice, try to split by order: first half = A, second half = B
    if n % 2 == 0 and n >= 2:
        half = n // 2
        a = df['best_test'].values[:half]
        b = df['best_test'].values[half:half*2]
        print(f"[INFO] using first {half} rows as group A, next {half} rows as group B (heuristic split)")
    else:
        # fallback: assume all are independent; can't compare
        print("[WARN] cannot infer two equal groups by row order. Printing overall stats.")
        vals = df['best_test'].values
        print("Overall mean,std,n:", np.mean(vals), np.std(vals, ddof=1), len(vals))
        return
    # stats
    mean_a = np.mean(a); std_a = np.std(a, ddof=1); n_a = len(a)
    mean_b = np.mean(b); std_b = np.std(b, ddof=1); n_b = len(b)
    t_res = stats.ttest_ind(a, b, equal_var=False)
    d = cohen_d_independent(a, b)
    print("Group A (first): mean, std, n:", mean_a, std_a, n_a)
    print("Group B (second): mean, std, n:", mean_b, std_b, n_b)
    print("Welch t-test: t = {:.4f}, p = {:.6f}".format(t_res.statistic, t_res.pvalue))
    print("Cohen's d (pooled): {:.4f}".format(d))
    # short conclusion
    alpha = 0.05
    if not np.isnan(t_res.pvalue):
        if t_res.pvalue < alpha:
            print(f"Conclusion: difference is statistically significant at alpha={alpha}.")
        else:
            print(f"Conclusion: no statistically significant difference at alpha={alpha}.")
    else:
        print("Conclusion: p-value is NaN; insufficient data.")
    print("Done.")

if __name__ == '__main__':
    main()

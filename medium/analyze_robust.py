#!/usr/bin/env python3
"""
analyze_robust.py
更鲁棒地分析清洗后的 per-run CSV，自动尝试：
1) 如果有 'exp_tag' 或 'cons_weight' 列，用它们分组；
2) 如果每个 run id 恰好出现两次，使用 per-run 对比 (配对/非配对视情况)；
3) 如果行数为偶数，且看起来是两块连续写入，按前半/后半分组；
4) 否则尝试用 2-cluster 聚类（仅作为最后手段），并警告用户。
输出每组 mean/std/n、Welch t-test、Cohen's d，及所用的启发式说明。
"""
import sys, os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans

def cohen_d(a,b):
    n1=len(a); n2=len(b)
    s1=np.std(a,ddof=1); s2=np.std(b,ddof=1)
    pooled=np.sqrt(((n1-1)*s1*s1 + (n2-1)*s2*s2)/(n1+n2-2)) if n1+n2-2>0 else np.nan
    return (np.mean(a)-np.mean(b))/pooled if pooled and not np.isnan(pooled) else np.nan

def try_grouping(df):
    # 1) if exp_tag exists
    if 'exp_tag' in df.columns:
        groups = df.groupby('exp_tag')
        if len(groups) >= 2:
            return {k: g['best_test'].values for k,g in groups}
    # 2) if cons_weight exists -> group by rounded cons_weight
    if 'cons_weight' in df.columns:
        df['cw_round'] = df['cons_weight'].round(6)
        groups = df.groupby('cw_round')
        if len(groups) >= 2:
            return {str(k): g['best_test'].values for k,g in groups}
    # 3) if run ids repeat exactly twice each
    if 'run' in df.columns:
        counts = df['run'].value_counts()
        if counts.max() == 2 and counts.min() == 2:
            # gather pairs by run order
            grouped = {}
            a=[]; b=[]
            for run_id in sorted(counts.index):
                sub = df[df['run']==run_id]
                vals = sub['best_test'].values
                if len(vals)==2:
                    a.append(vals[0]); b.append(vals[1])
                else:
                    return None
            return {'A': np.array(a), 'B': np.array(b)}
    # 4) if even number rows, split half/half
    n=len(df)
    if n%2==0 and n>=2:
        half=n//2
        return {'A': df['best_test'].values[:half], 'B': df['best_test'].values[half:]}
    # 5) fallback: kmeans cluster into 2 (last resort)
    vals = df['best_test'].values.reshape(-1,1)
    if len(vals)>=4:
        kmeans = KMeans(n_clusters=2, random_state=0).fit(vals)
        labels = kmeans.labels_
        return {'cluster0': vals[labels==0].flatten(), 'cluster1': vals[labels==1].flatten()}
    return None

def main():
    path = sys.argv[1] if len(sys.argv)>1 else os.path.join('results','cora_ours_results_per_run_cleaned.csv')
    if not os.path.exists(path):
        print("file not found:", path); return
    df = pd.read_csv(path)
    print("[INFO] columns:", df.columns.tolist(), "rows:", len(df))
    groups = try_grouping(df)
    if groups is None:
        print("[ERROR] 无法推断分组，请手动检查 CSV 或在 main.py 中加入 exp_tag 字段。")
        print(df.head(50))
        return
    print("[INFO] inferred groups:", list(groups.keys()))
    for k,v in groups.items():
        print(f"Group {k}: n={len(v)}, mean={np.mean(v):.6f}, std={np.std(v,ddof=1):.6f}")
    # select two largest groups for t-test
    sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
    if len(sorted_groups) < 2:
        print("[ERROR] 少于两个组，无法比较。")
        return
    g1_name,g1 = sorted_groups[0]
    g2_name,g2 = sorted_groups[1]
    t = stats.ttest_ind(g1, g2, equal_var=False)
    d = cohen_d(g1,g2)
    print("=== Comparison ===")
    print(f"{g1_name} mean={np.mean(g1):.6f} std={np.std(g1,ddof=1):.6f} n={len(g1)}")
    print(f"{g2_name} mean={np.mean(g2):.6f} std={np.std(g2,ddof=1):.6f} n={len(g2)}")
    print(f"Welch t-test: t={t.statistic:.4f}, p={t.pvalue:.6f}")
    print(f"Cohen's d (pooled)={d:.4f}")
    if t.pvalue < 0.05:
        print("Conclusion: difference is statistically significant at alpha=0.05")
    else:
        print("Conclusion: not significant at alpha=0.05")
    print("Note: 请审阅上面的分组方法与输出，若分组不合理请使用方案1 在 main.py 写入 exp_tag 后重跑实验。")

if __name__ == '__main__':
    main()

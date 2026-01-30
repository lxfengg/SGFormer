#!/usr/bin/env python3
# scripts/corr_delta_pseudo.py
import pandas as pd, os
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
os.makedirs('results/figs_pair', exist_ok=True)
pair = pd.read_csv('results/figs_pair/pairwise_rows_A_fair_vs_A_plus_S_best_test.csv')
# ensure seed column int
if 'seed' in pair.columns:
    pair['seed'] = pair['seed'].astype(int)
else:
    raise SystemExit("pairwise rows csv missing 'seed' column")
pseudo = pd.read_csv('results/diagnosis_pseudo_summary.csv')
# extract seed from ckpt name like cora_A_plus_S_seed100_run0_best
pseudo['seed'] = pseudo['ckpt'].str.extract(r'(\d{3}|\d+)').astype(int)
merged = pd.merge(pair, pseudo, on='seed', how='left')
print("Merged rows:", len(merged))
for col in ['teacher_unlabeled_acc','coverage_50_frac','coverage_60_frac','coverage_70_frac','coverage_80_frac']:
    if col in merged.columns:
        valid = merged.dropna(subset=['delta',col])
        if len(valid)>=3:
            pr = pearsonr(valid['delta'], valid[col])
            sr = spearmanr(valid['delta'], valid[col])
            print(f"{col}: Pearson r={pr[0]:.3f}, p={pr[1]:.3f}; Spearman rho={sr.correlation:.3f}, p={sr.pvalue:.3f}")
            # scatter plot
            plt.figure(figsize=(5,4))
            plt.scatter(valid[col], valid['delta'])
            plt.xlabel(col); plt.ylabel('delta (A_plus_S - A_fair)')
            plt.axhline(0, color='k', linestyle='--', linewidth=0.6)
            plt.tight_layout()
            plt.savefig(f"results/figs_pair/delta_vs_{col}.png", dpi=150)
            print("Wrote plot for",col)
        else:
            print(col, "not enough data (n<3)")
    else:
        print(col, "not in pseudo summary")
merged.to_csv('results/figs_pair/merged_delta_pseudo.csv',index=False)
print("Wrote results/figs_pair/merged_delta_pseudo.csv")

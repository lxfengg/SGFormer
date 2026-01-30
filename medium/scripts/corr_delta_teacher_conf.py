#!/usr/bin/env python3
# 保存为 scripts/corr_delta_teacher_conf.py
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# 输入文件（你生成的）
PAIR_CSV = 'results/figs_pair/pairwise_rows_A_fair_vs_A_plus_S_best_test.csv'
DIAG_JSON = 'results/diagnosis_teacher_coverage.json'  # from analyze_teacher_coverage.py
OUT_DIR = 'results/figs_pair'
os.makedirs(OUT_DIR, exist_ok=True)

# load pairwise
pair = pd.read_csv(PAIR_CSV)
# expected columns: seed, A_fair_best_test, A_plus_S_best_test (or similar)
# normalize column names:
cols = pair.columns.tolist()
# locate seed and two metric cols:
if 'seed' not in cols:
    # try first col named something else
    pair = pair.rename(columns={cols[0]:'seed'})
# create delta
pair['delta'] = pair.iloc[:,2] - pair.iloc[:,1]

# load diag json (it contains per-seed lines; we try to read as jsonlines or plain json)
import json
with open(DIAG_JSON, 'r') as f:
    diag = json.load(f)

# diag may be a dict with per-seed lines or list; try flexible parsing:
# We expect diag to contain entries per CSV line printed by analyze_teacher_coverage: we'll parse file instead.
# As fallback, try to parse results/diagnosis_per_ckpt/*.json
if isinstance(diag, dict) and 'rows' in diag and isinstance(diag['rows'], list):
    rows = diag['rows']
else:
    # try to read results/diagnosis_per_ckpt/*.json summary if present
    import glob
    rows = []
    for p in glob.glob('results/diagnosis_per_ckpt/*.json'):
        try:
            j = json.load(open(p,'r'))
            s = j.get('summary', j)
            # try extract seed from filename
            import re
            m = re.search(r'seed(\d+)', p)
            if m:
                s['seed'] = int(m.group(1))
            rows.append(s)
        except Exception:
            pass

if len(rows) == 0:
    print("No per-checkpoint JSON summary found. Exiting.")
    raise SystemExit(1)

diag_df = pd.DataFrame(rows)
# try common keys
# choose teacher mean conf column name heuristically
possible_conf_cols = [c for c in diag_df.columns if 'teacher' in c.lower() and 'conf' in c.lower() or 'mtc' in c.lower() or 'mean_teacher' in c.lower()]
if not possible_conf_cols:
    # try 'teacher_unlabeled_acc' or 'mean_teacher_conf'
    for tryc in ['mean_teacher_conf','mtc_mean','mtc_last','teacher_unlabeled_acc']:
        if tryc in diag_df.columns:
            possible_conf_cols = [tryc]
            break
if not possible_conf_cols:
    print("Could not find teacher confidence column in diag json. Columns:", diag_df.columns.tolist())
    raise SystemExit(1)

conf_col = possible_conf_cols[0]
print("Using teacher conf column:", conf_col)

# ensure seed present
if 'seed' not in diag_df.columns:
    # try extract from filename column or json name
    if 'json' in diag_df.columns:
        diag_df['seed'] = diag_df['json'].str.extract(r'(\d{3}|\d+)').astype(float)
    else:
        print("No seed column found in diag. Columns:", diag_df.columns.tolist())
        raise SystemExit(1)

diag_df['seed'] = diag_df['seed'].astype(int)
merged = pd.merge(pair, diag_df[['seed', conf_col]], on='seed', how='left')

# drop NA
merged2 = merged.dropna(subset=['delta', conf_col])
print("Merged rows:", len(merged2))

# correlation
pear = pearsonr(merged2['delta'], merged2[conf_col])
spear = spearmanr(merged2['delta'], merged2[conf_col])
print("Pearson r, p:", pear)
print("Spearman rho, p:", spear)

# scatter plot
plt.figure(figsize=(6,5))
plt.scatter(merged2[conf_col], merged2['delta'])
for _, r in merged2.iterrows():
    plt.text(r[conf_col], r['delta'], str(int(r['seed'])), fontsize=7)
plt.axhline(0, color='k', linestyle='--', linewidth=0.6)
plt.xlabel('teacher_conf ('+conf_col+')')
plt.ylabel('delta (A_plus_S - A_fair)')
plt.title(f'Δ vs teacher_conf (n={len(merged2)})\\nPearson r={pear[0]:.3f}, p={pear[1]:.3f}')
out_png = os.path.join(OUT_DIR, 'delta_vs_teacher_conf.png')
plt.tight_layout()
plt.savefig(out_png, dpi=150)
print("Wrote", out_png)
merged2.to_csv(os.path.join(OUT_DIR,'merged_pair_diag.csv'),index=False)
print("Wrote merged csv to results/figs_pair/merged_pair_diag.csv")

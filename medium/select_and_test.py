#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
import scipy.stats as stats

if len(sys.argv) < 2:
    print("Usage: python select_and_test.py <hp_raw_csv>")
    sys.exit(1)

fn = sys.argv[1]
df = pd.read_csv(fn)

# Expect columns: config_tag,exp_group,seed,run,best_val,best_test
required = {'config_tag','exp_group','seed','run','best_val','best_test'}
if not required.issubset(set(df.columns)):
    print("Missing required columns in", fn)
    print("Found:", df.columns.tolist())
    sys.exit(1)

# convert numeric
df['seed'] = df['seed'].astype(int)
df['best_val'] = pd.to_numeric(df['best_val'], errors='coerce')
df['best_test'] = pd.to_numeric(df['best_test'], errors='coerce')

# For each exp_group and seed, choose the config with highest best_val (validation)
best_rows = []
for (exp_group, seed), g in df.groupby(['exp_group','seed']):
    # drop nan val
    g2 = g.dropna(subset=['best_val'])
    if len(g2) == 0:
        continue
    # pick row with max best_val; if tie, take max best_test
    idx = g2['best_val'].idxmax()
    cand = g2.loc[[idx]]
    best_rows.append(cand)

best_df = pd.concat(best_rows, ignore_index=True)
# pivot to compare baseline vs schannel (paired by seed)
pivot = best_df.pivot(index='seed', columns='exp_group', values='best_test')
pivot = pivot.dropna()  # only seeds present in both groups
print("Paired seeds:", len(pivot))
print(pivot.head())

# compute stats
groups = pivot.columns.tolist()
if 'baseline' not in groups or 'schannel' not in groups:
    print("Need baseline and schannel in exp_group. Found:", groups)
    sys.exit(1)

base = pivot['baseline'].values
sch = pivot['schannel'].values
diff = sch - base
mean_base = np.mean(base)
mean_sch = np.mean(sch)
print(f"baseline mean={mean_base:.6f}, schannel mean={mean_sch:.6f}, mean diff={np.mean(diff):.6f}")
# paired t-test
t, p = stats.ttest_rel(sch, base)
# Cohen's d (paired)
d = np.mean(diff) / (np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else np.nan)
print(f"paired t: t={t:.4f}, p={p:.6e}, cohen_d={d:.4f}")

# Save pivot and summary
out_pref = fn.replace('.csv','')
pivot.to_csv(out_pref + '_best_pivot.csv')
with open(out_pref + '_summary.txt','w') as f:
    f.write(f"paired seeds: {len(pivot)}\n")
    f.write(f"baseline mean: {mean_base:.6f}\n")
    f.write(f"schannel mean: {mean_sch:.6f}\n")
    f.write(f"mean diff (sch - base): {np.mean(diff):.6f}\n")
    f.write(f"paired t: t={t:.4f}, p={p:.6e}\n")
    f.write(f"cohen d: {d:.4f}\n")

print("Wrote:", out_pref + "_best_pivot.csv", out_pref + "_summary.txt")

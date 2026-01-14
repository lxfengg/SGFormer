#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
from scipy import stats

if len(sys.argv) < 2:
    print("Usage: analyze_demo_pairing.py <tagged_csv>")
    sys.exit(1)

fn = sys.argv[1]
df = pd.read_csv(fn)

if 'exp_tag' not in df.columns or 'seed' not in df.columns or 'best_test' not in df.columns:
    print("ERROR: CSV missing required columns.")
    sys.exit(1)

pivot = df.pivot_table(index='seed', columns='exp_tag', values='best_test', aggfunc='last')
print("Pivot (seed x exp_tag):")
print(pivot)
print()

cols = pivot.columns.tolist()
baseline_cols = [c for c in cols if str(c).lower().startswith('baseline')]
schannel_cols = [c for c in cols if str(c).lower().startswith('schannel')]

if not baseline_cols:
    print("ERROR: no baseline-like columns found. Available columns:", cols)
    sys.exit(1)
if not schannel_cols:
    print("ERROR: no schannel-like columns found. Available columns:", cols)
    sys.exit(1)

baseline_col = baseline_cols[0]
schannel_col = schannel_cols[0]
print(f"Using baseline column: {baseline_col}, schannel column: {schannel_col}")
print()

paired = pivot[[baseline_col, schannel_col]].dropna()
print("Paired seeds:", len(paired))
if len(paired) == 0:
    print("No paired seeds found after dropping NA.")
    sys.exit(0)

b = paired[baseline_col].astype(float)
a = paired[schannel_col].astype(float)

print(f"baseline mean={b.mean():.6f}, schannel mean={a.mean():.6f}, mean diff={(a-b).mean():.6f}")

try:
    t, p = stats.ttest_rel(a, b)
except Exception:
    t, p = np.nan, np.nan

diff = a - b
try:
    d = diff.mean() / diff.std(ddof=1)
except Exception:
    d = np.nan

print(f"paired t: t={t:.4f}, p={p:.6e}, cohen_d={d:.4f}")

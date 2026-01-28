#!/usr/bin/env python3
# 保存为 analyze_pairwise.py
"""
Pairwise analysis script.
Reads a CSV like results/diagnosis_all_checkpoints_python.csv (one row per checkpoint).
Pairs rows by seed for tagA and tagB, computes paired t-test, Cohen's d, and plots.
"""
import argparse, os, csv, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--csv', required=True, help='Summary CSV (e.g. results/diagnosis_all_checkpoints_python.csv)')
parser.add_argument('--tagA', required=True, help='Label for control (e.g. A_fair)')
parser.add_argument('--tagB', required=True, help='Label for treatment (e.g. A_plus_S)')
parser.add_argument('--metric', default='student_test', help='Column name to use (student_test / student_val / student_unlabeled_acc etc.)')
parser.add_argument('--out_dir', default='results/figs_pair', help='Output dir for CSV + figures')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# load CSV robustly
df = pd.read_csv(args.csv)
# unify column names
df.columns = [c.strip() for c in df.columns]

# determine tag column or derive from basename
def infer_tag(row):
    if 'exp_tag' in row and pd.notna(row['exp_tag']) and str(row['exp_tag']).strip() != '':
        return str(row['exp_tag']).strip()
    # try to parse from basename
    name = ''
    if 'basename' in row and pd.notna(row['basename']):
        name = str(row['basename'])
    elif 'ckpt_path' in row and pd.notna(row['ckpt_path']):
        name = os.path.basename(str(row['ckpt_path']))
    # heuristics
    if 'A_plus_S' in name:
        return 'A_plus_S'
    if 'A_fair' in name:
        return 'A_fair'
    # fallback: try tokens separated by underscore
    toks = name.split('_')
    if len(toks) >= 2:
        return toks[1]
    return 'unknown'

df['inferred_tag'] = df.apply(infer_tag, axis=1)

# get seed column or extract
def infer_seed(row):
    for col in ['seed','Seed','SEED']:
        if col in row.index and pd.notna(row[col]):
            try:
                return int(row[col])
            except:
                pass
    # try from basename or ckpt_path
    s = ''
    if 'basename' in row and pd.notna(row['basename']):
        s = str(row['basename'])
    elif 'ckpt_path' in row and pd.notna(row['ckpt_path']):
        s = os.path.basename(str(row['ckpt_path']))
    import re
    m = re.search(r"_seed(\d+)", s)
    if m:
        return int(m.group(1))
    # last resort: try digits in string
    m2 = re.search(r"(\d{2,4})", s)
    if m2:
        return int(m2.group(1))
    return None

df['seed_infer'] = df.apply(infer_seed, axis=1)

# filter rows that have the metric
if args.metric not in df.columns:
    print(f"[WARN] Metric '{args.metric}' not found in CSV columns. Available columns: {list(df.columns)}")
    # try lowercase variants
    lowcols = {c.lower():c for c in df.columns}
    if args.metric.lower() in lowcols:
        args.metric = lowcols[args.metric.lower()]
        print(f"[INFO] Using metric column '{args.metric}' instead.")
    else:
        raise SystemExit(1)

# split by tags
A_rows = df[df['inferred_tag'] == args.tagA].copy()
B_rows = df[df['inferred_tag'] == args.tagB].copy()
if len(A_rows)==0 or len(B_rows)==0:
    print(f"[ERROR] No rows found for tagA={args.tagA} (n={len(A_rows)}) or tagB={args.tagB} (n={len(B_rows)}).")
    print("Available tags (inferred):", df['inferred_tag'].unique().tolist())
    raise SystemExit(1)

# index by seed
A_map = {int(r['seed_infer']): r for _, r in A_rows.iterrows() if not pd.isna(r['seed_infer'])}
B_map = {int(r['seed_infer']): r for _, r in B_rows.iterrows() if not pd.isna(r['seed_infer'])}
common_seeds = sorted(set(A_map.keys()).intersection(set(B_map.keys())))
if len(common_seeds)==0:
    print("[ERROR] No overlapping seeds found between groups. Seeds in A:", sorted(A_map.keys()), " Seeds in B:", sorted(B_map.keys()))
    raise SystemExit(1)

# collect metric values (convert to float or nan)
valsA = []
valsB = []
rows = []
for s in common_seeds:
    arow = A_map[s]; brow = B_map[s]
    try:
        a_val = float(arow.get(args.metric, float('nan')))
    except:
        a_val = float('nan')
    try:
        b_val = float(brow.get(args.metric, float('nan')))
    except:
        b_val = float('nan')
    # skip nan pairs
    if math.isnan(a_val) or math.isnan(b_val):
        continue
    valsA.append(a_val); valsB.append(b_val)
    rows.append({'seed':s, f'{args.tagA}_{args.metric}':a_val, f'{args.tagB}_{args.metric}':b_val, 'diff': (b_val - a_val)})

n = len(valsA)
if n==0:
    print("[ERROR] After filtering NaNs no paired data left.")
    raise SystemExit(1)

arrA = np.array(valsA)
arrB = np.array(valsB)
diff = arrB - arrA

meanA = arrA.mean(); meanB = arrB.mean()
stdA = arrA.std(ddof=1); stdB = arrB.std(ddof=1)
mean_diff = diff.mean(); sd_diff = diff.std(ddof=1)
# paired t-test: try scipy, else compute t
p_val = None
t_stat = None
dfree = n-1
try:
    from scipy import stats
    t_res = stats.ttest_rel(arrB, arrA, nan_policy='omit')
    t_stat = float(t_res.statistic)
    p_val = float(t_res.pvalue)
except Exception:
    # compute t manually
    t_stat = mean_diff / (sd_diff / math.sqrt(n)) if sd_diff>0 else float('inf')
    p_val = None

# Cohen's d (paired)
cohen_d = mean_diff / sd_diff if sd_diff > 0 else float('inf')

# write paired rows CSV
out_rows_df = pd.DataFrame(rows)
out_csv = os.path.join(args.out_dir, f'pairwise_rows_{args.tagA}_vs_{args.tagB}_{args.metric}.csv')
out_stats = {
    'n': n,
    'meanA': meanA,
    'meanB': meanB,
    'stdA': stdA,
    'stdB': stdB,
    'mean_diff': mean_diff,
    'sd_diff': sd_diff,
    't': t_stat,
    'p': p_val,
    'cohen_d': cohen_d
}
os.makedirs(args.out_dir, exist_ok=True)
out_rows_df.to_csv(out_csv, index=False)

# print concise summary (suitable to paste in paper)
print("Paired stats:")
print(f"  n: {n}")
print(f"  mean{args.tagA}: {meanA:.4f}")
print(f"  mean{args.tagB}: {meanB:.4f}")
print(f"  std{args.tagA}: {stdA:.6f}")
print(f"  std{args.tagB}: {stdB:.6f}")
print(f"  mean_diff: {mean_diff:.4f}")
print(f"  sd_diff: {sd_diff:.6f}")
print(f"  t: {t_stat}")
print(f"  p: {p_val}")
print(f"  cohen_d: {cohen_d}")

# plotting: boxplot and paired scatter
fig1 = plt.figure(figsize=(5,4))
plt.boxplot([arrA, arrB], labels=[args.tagA, args.tagB])
plt.ylabel(args.metric)
plt.title(f'{args.tagA} vs {args.tagB} ({args.metric})')
plt.grid(axis='y', linestyle='--', linewidth=0.4)
plt.tight_layout()
box_path = os.path.join(args.out_dir, f'boxplot_{args.tagA}_vs_{args.tagB}_{args.metric}.png')
plt.savefig(box_path)
plt.close(fig1)

# paired scatter (lines connecting pairs)
fig2 = plt.figure(figsize=(6,4))
x = np.arange(n)
plt.scatter(np.zeros(n), arrA, label=args.tagA)
plt.scatter(np.ones(n), arrB, label=args.tagB)
for i in range(n):
    plt.plot([0,1], [arrA[i], arrB[i]], linewidth=0.7)
plt.xticks([0,1], [args.tagA, args.tagB])
plt.ylabel(args.metric)
plt.title(f'Paired scatter {args.metric}')
plt.tight_layout()
scatter_path = os.path.join(args.out_dir, f'paired_scatter_{args.tagA}_vs_{args.tagB}_{args.metric}.png')
plt.savefig(scatter_path)
plt.close(fig2)

# write stats JSON
import json
with open(os.path.join(args.out_dir, f'stat_summary_{args.tagA}_vs_{args.tagB}_{args.metric}.json'), 'w') as jf:
    json.dump({'stats': out_stats, 'rows_csv': out_csv, 'boxplot': box_path, 'scatter': scatter_path}, jf, indent=2)

print(f"Saved rows CSV -> {out_csv}")
print(f"Saved boxplot -> {box_path}")
print(f"Saved paired scatter -> {scatter_path}")
print(f"Saved stats JSON -> {os.path.join(args.out_dir, f'stat_summary_{args.tagA}_vs_{args.tagB}_{args.metric}.json')}")

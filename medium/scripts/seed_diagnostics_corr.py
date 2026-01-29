#!/usr/bin/env python3
# scripts/seed_diagnostics_corr.py
"""
Merge pairwise results with per-checkpoint diagnostics and compute correlations.
- If results/diagnosis_all_checkpoints_python.csv exists, use it.
- Else, scan results/diagnosis_per_ckpt/*.json to build diagnostics table.
Outputs:
  - results/figs_pair/seed_diagnostics_with_delta.csv
  - results/figs_pair/delta_vs_<diag>.png (one per diag column)
  - results/figs_pair/seed_diagnostics_corr_summary.json
"""
import os, sys, json, glob, re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

PAIR_CSV = "results/figs_pair/pairwise_rows_A_fair_vs_A_plus_S_best_test.csv"
DIAG_CSV = "results/diagnosis_all_checkpoints_python.csv"
OUT_DIR = "results/figs_pair"
os.makedirs(OUT_DIR, exist_ok=True)

# load pair CSV
if not os.path.exists(PAIR_CSV):
    print(f"[ERROR] pair CSV not found: {PAIR_CSV}")
    sys.exit(2)
dfp = pd.read_csv(PAIR_CSV)

# load diagnostics: prefer aggregated CSV, else construct from json files
if os.path.exists(DIAG_CSV):
    print(f"[INFO] Loading diagnostics from {DIAG_CSV}")
    dfd = pd.read_csv(DIAG_CSV)
else:
    print(f"[WARN] {DIAG_CSV} not found; scanning results/diagnosis_per_ckpt/*.json")
    rows = []
    jfiles = sorted(glob.glob("results/diagnosis_per_ckpt/*.json"))
    for jf in jfiles:
        try:
            with open(jf, 'r') as f:
                data = json.load(f)
        except Exception:
            continue
        # standard fields we expect; fallback to keys if present
        row = {}
        row['json'] = jf
        # extract basename or infer from filename
        base = data.get('basename') or os.path.splitext(os.path.basename(jf))[0].replace('diagnosis_', '')
        row['basename'] = base
        # seed: try to parse "seedXXX" from basename
        m = re.search(r"seed(\d+)", base)
        row['seed'] = int(m.group(1)) if m else (data.get('seed') or np.nan)
        # copy some common metrics if present
        for k in ['student_train','student_val','student_test','teacher_train','teacher_val','teacher_test','student_unlabeled_acc','teacher_unlabeled_acc','stu_conf_gt_50','stu_conf_gt_60','stu_conf_gt_70','stu_conf_gt_80','mean_teacher_conf']:
            if k in data:
                row[k] = data[k]
        rows.append(row)
    if len(rows) == 0:
        print("[ERROR] No diagnosis json files found under results/diagnosis_per_ckpt/. Cannot proceed.")
        sys.exit(3)
    dfd = pd.DataFrame(rows)
    # try cast numeric where possible
    for c in dfd.columns:
        try:
            dfd[c] = pd.to_numeric(dfd[c], errors='ignore')
        except Exception:
            pass
    # save constructed diag csv for debugging
    try:
        dfd.to_csv("results/diagnosis_all_checkpoints_python_auto.csv", index=False)
        print("[INFO] Wrote reconstructed diagnostics to results/diagnosis_all_checkpoints_python_auto.csv")
    except Exception:
        pass

# ensure seed present in both
if 'seed' not in dfd.columns:
    print("[ERROR] diagnostics table has no 'seed' column. Here are columns:", dfd.columns.tolist())
    sys.exit(4)

# identify pair columns (the pair CSV likely has form: seed, A_fair_best_test, A_plus_S_best_test)
pair_cols = [c for c in dfp.columns if c != 'seed']
if len(pair_cols) < 2:
    print("[ERROR] pair CSV has unexpected columns:", dfp.columns.tolist())
    sys.exit(5)
Acol, Bcol = pair_cols[0], pair_cols[1]
dfp['delta'] = dfp[Bcol].astype(float) - dfp[Acol].astype(float)

# merge on seed
merged = pd.merge(dfp, dfd, on='seed', how='inner')
if merged.empty:
    print("[ERROR] No matching seeds between pair CSV and diagnostics. pair seeds:", sorted(dfp['seed'].unique()), "diag seeds:", sorted(dfd['seed'].unique()))
    sys.exit(6)

# choose diagnostic columns to analyze
candidates = []
for col in ['teacher_unlabeled_acc','stu_conf_gt_50','stu_conf_gt_60','stu_conf_gt_70','stu_conf_gt_80','mean_teacher_conf','teacher_val','teacher_test','student_unlabeled_acc']:
    if col in merged.columns:
        candidates.append(col)

summary = {}
for col in candidates:
    try:
        x = merged[col].astype(float).values
        y = merged['delta'].astype(float).values
    except Exception:
        continue
    # compute Pearson & Spearman
    try:
        pear_r, pear_p = stats.pearsonr(x, y)
    except Exception:
        pear_r, pear_p = (float('nan'), None)
    try:
        spear_r, spear_p = stats.spearmanr(x, y)
    except Exception:
        spear_r, spear_p = (float('nan'), None)
    summary[col] = {'pearson_r': pear_r, 'pearson_p': pear_p, 'spearman_r': spear_r, 'spearman_p': spear_p}
    # make scatter plot with seed labels
    plt.figure(figsize=(5,4))
    plt.scatter(x, y)
    for i, s in enumerate(merged['seed'].astype(int).tolist()):
        plt.text(x[i], y[i], str(s), fontsize=7, alpha=0.8)
    plt.xlabel(col); plt.ylabel('delta (A_plus_S - A_fair)')
    plt.title(f'delta vs {col}\\npearson r={pear_r:.3f}, p={pear_p}')
    plt.grid(ls='--', lw=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'delta_vs_{col}.png'))
    plt.close()

# save merged table and summary
merged.to_csv(os.path.join(OUT_DIR, 'seed_diagnostics_with_delta.csv'), index=False)
with open(os.path.join(OUT_DIR, 'seed_diagnostics_corr_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print("[DONE] Wrote seed_diagnostics_with_delta.csv, plots, and seed_diagnostics_corr_summary.json to", OUT_DIR)

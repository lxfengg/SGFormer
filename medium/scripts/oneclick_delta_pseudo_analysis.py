#!/usr/bin/env python3
# scripts/oneclick_delta_pseudo_analysis.py
"""
One-click analysis:
- Parse files under results/diagnosis_per_ckpt/ (txt or json) to extract pseudo-label metrics.
- Read pairwise rows CSV (results/figs_pair/pairwise_rows_A_fair_vs_A_plus_S_best_test.csv).
- Merge by seed.
- Compute Pearson & Spearman correlations between 'delta' and each numeric metric.
- Save merged CSV, correlation summary, and top scatter plots.
- Print a short human-readable summary and recommended next step.
"""
import os, glob, re, csv, json, traceback
from collections import defaultdict
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

# ------------ config ------------
PAIR_CSV = 'results/figs_pair/pairwise_rows_A_fair_vs_A_plus_S_best_test.csv'
DIAG_DIR = 'results/diagnosis_per_ckpt'
OUT_DIR = 'results/figs_pair'
MERGED_CSV = os.path.join(OUT_DIR, 'merged_delta_pseudo_oneclick.csv')
SUMMARY_CSV = os.path.join(OUT_DIR, 'delta_pseudo_corr_summary_oneclick.csv')
os.makedirs(OUT_DIR, exist_ok=True)
# ---------------------------------

def parse_diag_file(path):
    """Parse a single diagnosis file (txt or json) into a dict of candidate metrics."""
    d = defaultdict(lambda: None)
    name = os.path.basename(path)
    d['file'] = name
    # attempt to find ckpt name
    m = re.match(r'diagnosis_?(.+?)(?:\.json|\.txt)?$', name)
    if m:
        d['ckpt'] = m.group(1)
    else:
        d['ckpt'] = name.rsplit('.',1)[0]
    # attempt to extract seed
    m2 = re.search(r'seed[_-]?0*([0-9]+)', d['ckpt'])
    if not m2:
        m2 = re.search(r'(\d{3}|\d{2}|\d{1})', d['ckpt'])
    if m2:
        try:
            d['seed'] = int(m2.group(1))
        except:
            d['seed'] = None
    else:
        d['seed'] = None

    # read content
    txt = ''
    try:
        if path.lower().endswith('.json'):
            with open(path,'r',encoding='utf-8') as f:
                j = json.load(f)
            txt = json.dumps(j)
        else:
            with open(path,'r',encoding='utf-8',errors='ignore') as f:
                txt = f.read()
    except Exception as e:
        d['parse_error'] = str(e)
        return d

    # common extractions (robust)
    # Overall unlabeled acc
    m = re.search(r'Overall\s+unlabeled\s+acc[^\d\-]*([0-9]*\.[0-9]+|[0-9]+)', txt, re.IGNORECASE)
    if m:
        d['teacher_unlabeled_acc'] = float(m.group(1))
    m = re.search(r'teacher_unlabeled_acc\s*[:=]\s*([0-9]*\.[0-9]+|[0-9]+)', txt, re.IGNORECASE)
    if m:
        d['teacher_unlabeled_acc'] = float(m.group(1))

    # student unlabeled acc
    m = re.search(r'student_unlabeled_acc\s*[:=]\s*([0-9]*\.[0-9]+|[0-9]+)', txt, re.IGNORECASE)
    if m:
        d['student_unlabeled_acc'] = float(m.group(1))

    # mean teacher conf, mtc_mean
    for key in ['mean_teacher_conf','mtc_mean','mtc_mean']:
        m = re.search(rf'{key}\s*[:=]?\s*([0-9]*\.[0-9]+|[0-9]+)', txt, re.IGNORECASE)
        if m:
            d['mean_teacher_conf'] = float(m.group(1))
            break

    # thr lines like "thr 0.5: coverage 2466/2568 acc_on_conf=None"
    for thr in [0.5,0.6,0.7,0.8]:
        pat = rf"thr\s*{str(thr)}\s*[:\-\)]?.*?coverage\s*[:=]?\s*([0-9]+)\s*/\s*([0-9]+)(?:.*?acc_on_conf\s*[=:]?\s*([0-9.]+|None))?"
        m = re.search(pat, txt, re.IGNORECASE|re.S)
        if m:
            cov = int(m.group(1)); total = int(m.group(2))
            acc_conf = m.group(3)
            acc_val = None if acc_conf is None or str(acc_conf).strip().lower()=='none' else float(acc_conf)
            d[f'coverage_{int(thr*100)}'] = cov
            d[f'coverage_{int(thr*100)}_frac'] = cov/total if total>0 else None
            d[f'acc_on_conf_{int(thr*100)}'] = acc_val

    # stu_conf_gtXX counts
    for lab in ['50','60','70','80']:
        m = re.search(rf'stu_conf_gt[_ ]?{lab}\s*[:=]?\s*([0-9]+)', txt, re.IGNORECASE)
        if m:
            d[f'stu_conf_gt_{lab}'] = int(m.group(1))

    # fallback: direct numeric tokens "coverage 2466" etc
    m = re.search(r'coverage\s*[:=]?\s*([0-9]+)', txt, re.IGNORECASE)
    if m and not d.get('coverage_50'):
        d['coverage_any'] = int(m.group(1))

    return d

def main():
    try:
        # check inputs
        if not os.path.exists(PAIR_CSV):
            print(f"[ERR] pair csv not found: {PAIR_CSV}")
            return
        diag_files = sorted(glob.glob(os.path.join(DIAG_DIR, '*')))
        if not diag_files:
            print(f"[WARN] no per-ckpt diagnostics under {DIAG_DIR}. Run pseudo_label_quality batch first.")
        rows=[]
        for p in diag_files:
            rows.append(parse_diag_file(p))
        if not rows:
            print("[ERR] no parsed diagnostics.")
        pseudo_df = pd.DataFrame(rows)
        # write pseudo summary
        pseudo_out = os.path.join('results','diagnosis_pseudo_summary_oneclick.csv')
        pseudo_df.to_csv(pseudo_out, index=False)
        print("[INFO] Wrote parsed pseudo summary:", pseudo_out)
        # load pair csv
        pair = pd.read_csv(PAIR_CSV)
        if 'seed' not in pair.columns:
            print("[ERR] pair csv missing seed column")
            return
        pair['seed'] = pair['seed'].astype(int)
        # ensure pseudo seed exists
        if 'seed' not in pseudo_df.columns or pseudo_df['seed'].isnull().all():
            if 'ckpt' in pseudo_df.columns:
                pseudo_df['seed'] = pseudo_df['ckpt'].astype(str).str.extract(r'(\d{3}|\d{2}|\d{1})')[0]
                pseudo_df['seed'] = pd.to_numeric(pseudo_df['seed'], errors='coerce').astype('Int64')
        # merge
        merged = pd.merge(pair, pseudo_df, on='seed', how='left')
        merged.to_csv(MERGED_CSV, index=False)
        print("[INFO] Wrote merged CSV:", MERGED_CSV)

        # ensure delta exists
        if 'delta' not in merged.columns:
            # try compute if possible from metric columns
            metric_candidates = [c for c in merged.columns if re.search(r'(_best_test|best_test|best_val)', c)]
            if len(metric_candidates) >= 2:
                merged['delta'] = merged[metric_candidates[1]] - merged[metric_candidates[0]]
                print("[INFO] computed delta from metrics:", metric_candidates[:2])
            else:
                print("[ERR] no delta and cannot infer. Aborting correlation.")
                return

        # find numeric columns to test vs delta
        numeric_cols = []
        for c in merged.columns:
            if c in ('seed','delta','ckpt','file'): continue
            # test numeric
            try:
                vals = pd.to_numeric(merged[c], errors='coerce').dropna()
                if len(vals) >= 4:
                    numeric_cols.append(c)
            except:
                continue
        print("[INFO] numeric candidate columns:", numeric_cols)

        rows_summary=[]
        # compute correlations
        for c in numeric_cols:
            df = merged[['delta', c]].dropna()
            if len(df) < 4:
                continue
            try:
                pr = pearsonr(df['delta'], df[c])
                sr = spearmanr(df['delta'], df[c])
            except Exception as e:
                print("[WARN] stat error for", c, e); continue
            rows_summary.append({'metric': c, 'n': len(df), 'pearson_r': pr[0], 'pearson_p': pr[1], 'spearman_rho': sr.correlation, 'spearman_p': sr.pvalue})
            # safe filename creation (avoid f-string in-expression backslashes)
            safe = re.sub(r'[^0-9a-zA-Z_]+', '_', c)[:80]
            fn = os.path.join(OUT_DIR, "delta_vs_{}.png".format(safe))
            plt.figure(figsize=(5,4))
            plt.scatter(df[c], df['delta'])
            plt.axhline(0, color='k', linestyle='--', linewidth=0.6)
            plt.xlabel(c); plt.ylabel('delta (A_plus_S - A_fair)')
            plt.title(f"{c} n={len(df)} pearson r={pr[0]:.3f} p={pr[1]:.3f}")
            plt.tight_layout()
            plt.savefig(fn, dpi=150)
            plt.close()
            print("[INFO] saved plot", fn)

        if rows_summary:
            pd.DataFrame(rows_summary).sort_values(by='pearson_p').to_csv(SUMMARY_CSV, index=False)
            print("[INFO] Wrote correlation summary:", SUMMARY_CSV)
        else:
            print("[INFO] No numeric metrics with enough data to correlate.")

        # print short human summary (top 3)
        if rows_summary:
            dfsum = pd.DataFrame(rows_summary).sort_values(by='pearson_p')
            top = dfsum.head(3)
            print("\n=== Short summary (top metrics by Pearson p-value) ===")
            for _, r in top.iterrows():
                print(f"{r['metric']}  n={int(r['n'])}  pearson r={r['pearson_r']:.3f} (p={r['pearson_p']:.3e})  spearman rho={r['spearman_rho']:.3f} (p={r['spearman_p']:.3e})")
            best = dfsum.iloc[0]
            if best['pearson_p'] < 0.05:
                print("\nInterpretation: primary metric likely related to S-channel gains -> consider improving pseudo-label quality (e.g., increase cons_confidence, or tune cons_weight/ema_tau).")
            else:
                print("\nInterpretation: no strong correlation found -> S-channel gains not explained by parsed pseudo-quality metrics; consider prioritizing NodeMixup (B) or T-channel (CI-GCL).")
        else:
            print("\nNo metrics to summarize. Check diagnostics in results/diagnosis_per_ckpt/")

    except Exception:
        print("[FATAL] unexpected error:")
        traceback.print_exc()

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# 保存为 scripts/generate_diag_from_epoch_logs.py
"""
Scan results/epoch_logs/*.csv and produce per-run diagnosis json files
into results/diagnosis_per_ckpt/diagnosis_<basename>.json
Fields produced: basename, seed, exp_tag, mtc_mean, mtc_last, cons_last, cons_mean, epochs
"""
import os, glob, json, re
import pandas as pd
OUT_DIR = 'results/diagnosis_per_ckpt'
os.makedirs(OUT_DIR, exist_ok=True)

files = sorted(glob.glob('results/epoch_logs/*.csv'))
if not files:
    print("[WARN] no epoch log CSVs found in results/epoch_logs/. Aborting.")
    raise SystemExit(1)

n=0
for p in files:
    basename = os.path.basename(p)
    # try to parse dataset_method_tag_seed_run pattern
    # example: cora_ours_A_plus_S_seed100_run0.csv
    m = re.match(r'(?P<dataset>[^_]+)_(?P<method>[^_]+)_(?P<tag>.+)_seed(?P<seed>\d+)_run(?P<run>\d+)\.csv', basename)
    info = {
        'basename': basename,
        'path': p,
        'seed': None,
        'exp_tag': None,
        'epochs': None,
        'mtc_mean': None,
        'mtc_last': None,
        'cons_last': None,
        'cons_mean': None,
    }
    if m:
        info['seed'] = int(m.group('seed'))
        info['exp_tag'] = m.group('tag')
    try:
        df = pd.read_csv(p)
    except Exception as e:
        print(f"[WARN] failed to read {p}: {e}")
        continue
    # columns expected to include 'mean_teacher_conf' and 'cons_nodes' (or similar)
    # normalize column names
    cols = [c.strip() for c in df.columns.tolist()]
    df.columns = cols
    # prefer mean_teacher_conf, fallback to mtc_mean or 'mean_teacher_conf'
    mtc_col = None
    for cand in ['mean_teacher_conf', 'mtc', 'mtc_mean', 'mean_teacher', 'mean_teacher_confidence']:
        if cand in df.columns:
            mtc_col = cand
            break
    # cons nodes column
    cons_col = None
    for cand in ['cons_nodes', 'cons_node', 'cons_count', 'cons_last']:
        if cand in df.columns:
            cons_col = cand
            break
    # if mtc_col found compute stats
    try:
        if mtc_col and len(df[mtc_col].dropna())>0:
            info['mtc_mean'] = float(df[mtc_col].dropna().mean())
            info['mtc_last'] = float(df[mtc_col].dropna().iloc[-1])
        else:
            # try to infer from columns with 'mtc' substring
            possible = [c for c in df.columns if 'mtc' in c or 'teacher' in c and 'conf' in c]
            if possible:
                col = possible[0]
                info['mtc_mean'] = float(df[col].dropna().mean())
                info['mtc_last'] = float(df[col].dropna().iloc[-1])
    except Exception:
        pass
    try:
        if cons_col and len(df[cons_col].dropna())>0:
            info['cons_mean'] = float(df[cons_col].dropna().mean())
            info['cons_last'] = int(df[cons_col].dropna().iloc[-1])
    except Exception:
        pass
    try:
        info['epochs'] = int(df['epoch'].max()) if 'epoch' in df.columns else len(df)
    except Exception:
        info['epochs'] = len(df)
    # write json
    outf = os.path.join(OUT_DIR, f'diagnosis_{basename}.json')
    with open(outf, 'w') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    n+=1

print(f"[DONE] Wrote {n} per-run diagnosis json(s) to {OUT_DIR}/")

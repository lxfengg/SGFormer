#!/usr/bin/env python3
# verify_results_integrity.py
import pandas as pd
import glob
import os
import numpy as np

def inspect_tagged(path):
    print("----", path, "----")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # ensure numeric
    df['best_test'] = pd.to_numeric(df['best_test'], errors='coerce')
    df['seed'] = df['seed'].astype(str)
    # duplicates per (seed,exp_tag)
    dup = df.duplicated(subset=['seed','exp_tag'], keep=False)
    ndup = dup.sum()
    if ndup:
        print(f"[WARN] {ndup} duplicated lines for same (seed,exp_tag).")
        print(df[dup].sort_values(['seed','exp_tag']).head(20))
    # group pivot
    pivot = df.pivot_table(index='seed', columns='exp_tag', values='best_test', aggfunc='last')
    print("Seeds in file:", pivot.shape[0])
    # per-seed stats
    equal_count = 0
    zero_count = 0
    for seed, row in pivot.iterrows():
        vals = row.dropna().values
        if len(vals) >= 2:
            if np.allclose(vals, vals[0], atol=1e-9):
                equal_count += 1
        # placeholder
        for v in vals:
            try:
                if float(v) == 0.0:
                    zero_count += 1
            except:
                pass
    print("Identical-across-exp_tag seeds (exact):", equal_count)
    print("Placeholder (best_test==0) count:", zero_count)
    print("Sample pivot head:\n", pivot.head(15))
    return df, pivot

def recompute_from_epoch_logs(dataset):
    # looks for results/epoch_logs/<dataset>_*_seed{seed}_run{run}.csv
    path = os.path.join('results','epoch_logs')
    if not os.path.isdir(path):
        print("[INFO] no epoch_logs directory found:", path)
        return {}
    files = glob.glob(os.path.join(path,f'{dataset}_*_*seed*_*run*.csv'))
    # More permissive pattern:
    files = glob.glob(os.path.join(path, f'{dataset}_*seed*.csv')) + files
    by_seed = {}
    for f in files:
        base = os.path.basename(f)
        # try to parse seed/run
        # patterns like: {dataset}_{method}_seed{seed}_run{run}.csv
        try:
            parts = base.split('_')
            seed_part = next((p for p in parts if p.startswith('seed')), None)
            run_part = next((p for p in parts if p.startswith('run')), None)
            if seed_part:
                seed = seed_part.replace('seed','').replace('.csv','')
            else:
                seed = 'unknown'
            df = pd.read_csv(f)
            # expect columns: epoch,train,val,test,...
            if 'val' in df.columns and 'test' in df.columns:
                # choose row with max val
                idx = df['val'].idxmax()
                best_val = float(df.loc[idx,'val'])
                best_test = float(df.loc[idx,'test'])
            else:
                # fallback take last row
                best_val = float(df.iloc[-1]['val']) if 'val' in df.columns else np.nan
                best_test = float(df.iloc[-1]['test']) if 'test' in df.columns else np.nan
            by_seed.setdefault(seed, []).append({'file':f, 'best_val':best_val, 'best_test':best_test})
        except Exception as e:
            # ignore parse errors
            continue
    return by_seed

def main():
    tagged_files = glob.glob('results/*_results_per_run_tagged*.csv') + glob.glob('results/*_results_per_run.csv')
    if not tagged_files:
        print("No result CSVs found in results/.")
        return
    for f in tagged_files:
        df, pivot = inspect_tagged(f)
        dataset = os.path.basename(f).split('_')[0]
        by_seed = recompute_from_epoch_logs(dataset)
        if by_seed:
            print(f"[INFO] Found epoch logs for dataset {dataset}, checking consistency with tagged CSV:")
            # compare for seeds in pivot
            for seed in list(pivot.index)[:50]:
                if str(seed) in by_seed:
                    recs = by_seed[str(seed)]
                    # show if multiple runs exist
                    if len(recs) > 1:
                        print(f" seed {seed}: {len(recs)} epoch-logs found (showing last):", recs[-1])
                    else:
                        print(f" seed {seed}: epoch-log best_val/test:", recs[0])
            print("Note: if epoch-log best_test differs from tagged CSV, there was a write/merge issue.")
        else:
            print(f"[INFO] No epoch logs found for dataset {dataset} to cross-check.")

if __name__ == '__main__':
    main()

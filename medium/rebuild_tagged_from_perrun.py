#!/usr/bin/env python3
# rebuild_tagged_from_perrun.py
"""
Rebuilds results/*_results_per_run_tagged.csv from either:
  1) results/<dataset>_ours_results_per_run.csv  (preferred, has exp_tag column)
  2) results/epoch_logs/<dataset>_...csv (fallback)

Output path:
  results/<dataset>_ours_results_per_run_tagged_rebuilt.csv

It will backup existing tagged files (if any) to *.bak before writing.
"""
import os, glob, shutil, csv, math
import pandas as pd

RESULTS_DIR = "results"

def rebuild_from_perrun(perrun_path, out_path):
    df = pd.read_csv(perrun_path)
    # normalize columns
    df.columns = [c.strip() for c in df.columns]
    if 'exp_tag' not in df.columns:
        raise RuntimeError(f"{perrun_path} has no exp_tag column")
    # ensure numeric
    df['best_val'] = pd.to_numeric(df['best_val'], errors='coerce')
    df['best_test'] = pd.to_numeric(df['best_test'], errors='coerce')
    # pick best row per (exp_tag, seed) by highest best_val (fall back to last)
    rows = []
    grouped = df.sort_values(['exp_tag','seed','best_val','run']).groupby(['exp_tag','seed'], sort=False)
    for (exp_tag, seed), g in grouped:
        # choose row of max best_val, if tie pick last occurrence
        idx = g['best_val'].idxmax()
        chosen = g.loc[idx]
        run = chosen.get('run', 0)
        best_val = float(chosen.get('best_val', float('nan')))
        best_test = float(chosen.get('best_test', float('nan')))
        rows.append((exp_tag, str(seed), int(run), best_val, best_test))
    # write to out_path
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['exp_tag','seed','run','best_val','best_test'])
        for r in rows:
            writer.writerow(r)
    return len(rows)

def rebuild_from_epochlogs(dataset, out_path):
    # fallback: parse files like results/epoch_logs/{dataset}_*_seed{seed}_run{run}.csv
    pattern = os.path.join(RESULTS_DIR,'epoch_logs', f'{dataset}_*_seed*_run*.csv')
    files = glob.glob(pattern)
    if not files:
        # try permissive pattern
        files = glob.glob(os.path.join(RESULTS_DIR,'epoch_logs', f'{dataset}_*seed*.csv'))
    rows = []
    # We'll not be able to recover exp_tag reliably from epoch logs (no exp_tag in filename)
    # so we will write rows with exp_tag='unknown' unless multiple methods can be inferred.
    for f in sorted(files):
        base = os.path.basename(f)
        # try parse seed and run
        seed = 'unknown'
        run = 0
        try:
            if 'seed' in base:
                # find 'seed' token
                tokens = base.replace('.csv','').split('_')
                for t in tokens:
                    if t.startswith('seed'):
                        seed = t.replace('seed','')
                    if t.startswith('run'):
                        run = int(t.replace('run',''))
        except:
            pass
        try:
            df = pd.read_csv(f)
            if 'val' in df.columns and 'test' in df.columns:
                idx = df['val'].idxmax()
                best_val = float(df.loc[idx,'val'])
                best_test = float(df.loc[idx,'test'])
            else:
                # fallback last row
                best_val = float(df.iloc[-1]['val']) if 'val' in df.columns else float('nan')
                best_test = float(df.iloc[-1]['test']) if 'test' in df.columns else float('nan')
        except Exception as e:
            continue
        rows.append(('unknown', str(seed), run, best_val, best_test))
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['exp_tag','seed','run','best_val','best_test'])
        for r in rows:
            writer.writerow(r)
    return len(rows)

def main():
    datasets = []
    # find per-run CSVs available
    perrun_files = glob.glob(os.path.join(RESULTS_DIR, '*_ours_results_per_run.csv'))
    for p in perrun_files:
        datasets.append(os.path.basename(p).split('_')[0])
    # also check epoch_logs dataset names if perrun missing
    epoch_files = glob.glob(os.path.join(RESULTS_DIR,'epoch_logs','*.csv'))
    for f in epoch_files:
        name = os.path.basename(f).split('_')[0]
        if name not in datasets:
            datasets.append(name)
    datasets = sorted(set(datasets))
    if not datasets:
        print("No datasets found in results/. Nothing to rebuild.")
        return
    print("Datasets to process:", datasets)
    for ds in datasets:
        perrun = os.path.join(RESULTS_DIR, f'{ds}_ours_results_per_run.csv')
        out_rebuilt = os.path.join(RESULTS_DIR, f'{ds}_ours_results_per_run_tagged_rebuilt.csv')
        out_backup = os.path.join(RESULTS_DIR, f'{ds}_ours_results_per_run_tagged.csv.bak')
        orig_tagged = os.path.join(RESULTS_DIR, f'{ds}_ours_results_per_run_tagged.csv')
        if os.path.exists(orig_tagged):
            print(f"Backing up existing tagged file: {orig_tagged} -> {out_backup}")
            shutil.copyfile(orig_tagged, out_backup)
        try:
            if os.path.exists(perrun):
                n = rebuild_from_perrun(perrun, out_rebuilt)
                print(f"[OK] rebuilt {out_rebuilt} from {perrun} ({n} rows).")
            else:
                n = rebuild_from_epochlogs(ds, out_rebuilt)
                print(f"[OK] rebuilt {out_rebuilt} from epoch_logs ({n} rows).")
        except Exception as e:
            print(f"[ERROR] failed to rebuild for {ds}: {e}")
    print("Done. Review rebuilt CSVs in results/*. Re-run your analysis on *_rebuilt.csv")

if __name__ == '__main__':
    main()

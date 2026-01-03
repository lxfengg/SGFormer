#!/usr/bin/env python3
import csv
import os
import sys
import math
import traceback
import pandas as pd
import numpy as np
from scipy import stats

csv_path = 'results/cora_ours_results_per_run.csv'  # 修改为你的路径（如果不同）

def robust_read_csv(path):
    """
    Robustly read a CSV that may contain malformed lines.
    Strategy:
      1. Try pd.read_csv normally.
      2. If it fails, parse with csv.reader to locate a header row that contains
         expected column names and then parse subsequent rows, skipping lines
         that do not match the header column count.
    Returns: pandas.DataFrame
    """
    # 1) quick try pandas normal read first
    try:
        df = pd.read_csv(path)
        print(f"[INFO] pandas read_csv succeeded with shape {df.shape}")
        return df
    except Exception as e:
        print("[WARN] pandas.read_csv failed, trying robust parsing (csv.reader).")
        # continue to robust parse

    # 2) robust parsing
    rows = []
    header = None
    header_idx = None
    skipped = 0
    parsed = 0

    with open(path, 'r', newline='') as f:
        # read all lines via csv.reader so quotes are handled
        rdr = csv.reader(f)
        all_rows = list(rdr)

    # Try to find a header row that contains likely header tokens
    # Common header tokens in our produced file: run, seed, exp_tag, cons_weight, best_val, best_test
    expected_tokens = {'run', 'seed', 'best_test', 'best_val', 'exp_tag', 'cons_weight'}
    for idx, r in enumerate(all_rows):
        low = [c.strip().lower() for c in r]
        if any(tok in low for tok in ('best_test', 'best_val')) and 'run' in low:
            header = [c.strip() for c in r]
            header_idx = idx
            break

    if header is None:
        # fallback: assume first non-empty row is header
        for idx, r in enumerate(all_rows):
            if len(r) >= 2 and any(len(cell.strip())>0 for cell in r):
                header = [c.strip() for c in r]
                header_idx = idx
                print("[WARN] No clear header found; using first non-empty line as header.")
                break

    if header is None:
        raise ValueError("Could not find a header row in the CSV file. Please inspect the file manually.")

    ncols = len(header)
    # collect rows after header_idx
    for r in all_rows[header_idx+1:]:
        # skip entirely empty rows
        if len(r) == 0 or all(len(cell.strip()) == 0 for cell in r):
            continue
        if len(r) != ncols:
            skipped += 1
            # try a best-effort merge: if longer, join extras into last col
            if len(r) > ncols:
                merged = r[:ncols-1] + [",".join(r[ncols-1:])]
                if len(merged) == ncols:
                    rows.append(merged)
                    parsed += 1
                    continue
            # else skip
            continue
        rows.append(r)
        parsed += 1

    print(f"[INFO] header found at line {header_idx+1} with {ncols} columns")
    print(f"[INFO] parsed rows: {parsed}, skipped malformed rows: {skipped}")

    # Build DataFrame
    df = pd.DataFrame(rows, columns=header)
    return df

def ensure_columns(df):
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    cols = set(df.columns)
    # If 'exp_tag' missing, try to infer from cons_weight or create default
    if 'exp_tag' not in cols:
        if 'cons_weight' in cols:
            try:
                df['cons_weight'] = df['cons_weight'].astype(float)
                df['exp_tag'] = df['cons_weight'].apply(lambda x: 'baseline' if x == 0 else 'schannel')
                print("[INFO] Inferred exp_tag from cons_weight.")
            except Exception:
                df['exp_tag'] = 'unknown'
                print("[WARN] Could not cast cons_weight; exp_tag set to 'unknown'.")
        else:
            df['exp_tag'] = 'unknown'
            print("[WARN] No exp_tag or cons_weight column; exp_tag set to 'unknown' for all rows.")
    # If seed missing, try to use 'run' as seed fallback
    if 'seed' not in cols:
        if 'run' in cols:
            try:
                df['seed'] = df['run'].astype(int)
                print("[INFO] 'seed' column missing; used 'run' as fallback seed.")
            except Exception:
                df['seed'] = np.arange(len(df))
                print("[WARN] 'seed' & 'run' cannot be cast; generating synthetic seeds.")
        else:
            df['seed'] = np.arange(len(df))
            print("[WARN] 'seed' & 'run' missing; generating synthetic seeds.")

    # ensure best_test exists
    if 'best_test' not in cols and 'best_val_test' in cols:
        df['best_test'] = df['best_val_test']
        print("[INFO] Renamed best_val_test -> best_test")
    if 'best_test' not in df.columns:
        # try to find a column that likely contains test accuracy
        candidates = [c for c in df.columns if 'test' in c.lower()]
        if candidates:
            df['best_test'] = df[candidates[0]]
            print(f"[INFO] Mapped column {candidates[0]} -> best_test")
        else:
            raise ValueError("No 'best_test' column found and no candidate columns detected.")

    # convert best_test to numeric
    df['best_test'] = pd.to_numeric(df['best_test'], errors='coerce')
    # drop rows with NaN best_test
    before = len(df)
    df = df.dropna(subset=['best_test'])
    after = len(df)
    if after < before:
        print(f"[INFO] dropped {before-after} rows with non-numeric best_test")

    return df

def main():
    if not os.path.exists(csv_path):
        print(f"[ERROR] file not found: {csv_path}")
        sys.exit(1)

    try:
        df = robust_read_csv(csv_path)
    except Exception as e:
        print("[ERROR] failed to robustly parse CSV:", e)
        traceback.print_exc()
        sys.exit(1)

    try:
        df = ensure_columns(df)
    except Exception as e:
        print("[ERROR] column normalization failed:", e)
        traceback.print_exc()
        print("Top 30 lines of the file for inspection:")
        os.system(f"sed -n '1,30p' {csv_path}")
        sys.exit(1)

    # now we have a df with columns including: exp_tag, seed, best_test
    # Normalize best_test to percentage if it's in 0..1 range
    if df['best_test'].max() <= 1.0 + 1e-6:
        df['best_test_pct'] = df['best_test'] * 100.0
    else:
        df['best_test_pct'] = df['best_test'] * 1.0  # assume already percent

    # split groups
    baseline = df[df['exp_tag'].astype(str).str.lower() == 'baseline'].sort_values('seed')
    sch = df[df['exp_tag'].astype(str).str.lower() == 'schannel'].sort_values('seed')

    print(f"[INFO] baseline rows: {len(baseline)}, schannel rows: {len(sch)}")

    # if both present and seeds match -> paired
    paired = False
    if len(baseline) > 0 and len(baseline) == len(sch):
        if np.array_equal(baseline['seed'].values.astype(int), sch['seed'].values.astype(int)):
            paired = True

    if paired:
        base_vals = baseline['best_test_pct'].values
        sch_vals = sch['best_test_pct'].values
        diffs = sch_vals - base_vals
        tstat, pval = stats.ttest_rel(sch_vals, base_vals)
        mean_diff = np.mean(diffs)
        sd_diff = np.std(diffs, ddof=1)
        cohen_d = mean_diff / sd_diff if sd_diff != 0 else float('nan')
        ci_low, ci_high = stats.t.interval(0.95, len(diffs)-1, loc=mean_diff, scale=stats.sem(diffs))
        print("=== PAIRED TEST ===")
        print(f"Baseline mean, sd: {np.mean(base_vals):.4f}, {np.std(base_vals, ddof=1):.4f}")
        print(f"S-channel mean, sd: {np.mean(sch_vals):.4f}, {np.std(sch_vals, ddof=1):.4f}")
        print(f"Mean difference (sch - base): {mean_diff:.4f}")
        print(f"Paired t-test: t = {tstat:.4f}, p = {pval:.6f}")
        print(f"95% CI for mean diff: [{ci_low:.4f}, {ci_high:.4f}]")
        print(f"Cohen's d (paired): {cohen_d:.4f}")
    else:
        # independent test
        base_vals = baseline['best_test_pct'].values
        sch_vals = sch['best_test_pct'].values
        tstat, pval = stats.ttest_ind(sch_vals, base_vals, equal_var=False)
        print("=== INDEPENDENT TEST (fallback) ===")
        print(f"Baseline mean, sd, n: {np.mean(base_vals):.4f}, {np.std(base_vals, ddof=1):.4f}, {len(base_vals)}")
        print(f"S-channel mean, sd, n: {np.mean(sch_vals):.4f}, {np.std(sch_vals, ddof=1):.4f}, {len(sch_vals)}")
        print(f"Independent t-test (Welch): t = {tstat:.4f}, p = {pval:.6f}")

    # Print full per-seed table for record
    print("\n=== Per-seed summary (first 50 rows) ===")
    pd.set_option('display.max_rows', 200)
    merged = pd.merge(baseline[['seed','best_test_pct']].rename(columns={'best_test_pct':'base_test'}),
                      sch[['seed','best_test_pct']].rename(columns={'best_test_pct':'sch_test'}),
                      on='seed', how='outer')
    print(merged.head(50).to_string(index=False))

if __name__ == '__main__':
    main()

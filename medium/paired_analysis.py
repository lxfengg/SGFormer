#!/usr/bin/env python3
"""
Robust paired/independent analysis for results CSV.

Usage:
    python paired_analysis.py [path/to/results.csv]

This script tries pandas.read_csv first; on failure it falls back to csv.reader with
robust column detection and malformed-row skipping.
"""
import sys
import csv
import re
from collections import Counter, defaultdict
import math

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_PATH = 'results/cora_ours_results_per_run.csv'

def is_float(x):
    try:
        if x is None:
            return False
        s = str(x).strip()
        if s == '':
            return False
        float(s)
        return True
    except Exception:
        return False

def try_pandas_read(path):
    try:
        df = pd.read_csv(path)
        print(f"[INFO] pandas.read_csv succeeded. columns: {list(df.columns)} rows: {len(df)}")
        return df
    except pd.errors.ParserError as e:
        print("[WARN] pandas.read_csv failed with ParserError:", e)
        return None
    except Exception as e:
        print("[WARN] pandas.read_csv failed with Exception:", e)
        return None

def robust_csv_parse(path):
    print("[INFO] Trying robust csv.reader parsing...")
    rows = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        for r in reader:
            # strip BOM from first cell if present
            if len(r) > 0 and isinstance(r[0], str):
                r[0] = r[0].lstrip('\ufeff')
            rows.append([c.strip() for c in r])
    print(f"[INFO] total raw rows read: {len(rows)}")
    # find header
    header_idx = None
    header = None
    for i, r in enumerate(rows[:5]):  # check first 5 rows for header-like
        low = [c.lower() for c in r]
        if any('best' in c for c in low) or any('exp' in c for c in low):
            header_idx = i
            header = r
            break
    if header is None:
        # fallback: first row as header if it contains any alpha strings
        cand = rows[0]
        if any(re.search('[A-Za-z_]', c) for c in cand):
            header_idx = 0
            header = cand
        else:
            # no header found; create integer-column header
            max_cols = max(len(r) for r in rows)
            header_idx = 0
            header = [f'col{i}' for i in range(max_cols)]
            print("[WARN] No header detected; using artificial header:", header)
    print(f"[INFO] chosen header (row {header_idx}): {header}")

    data_rows = rows[header_idx+1:]

    # normalize rows to at least len(header) by padding with ''
    maxlen = max(len(header), max((len(r) for r in data_rows), default=0))
    norm_header = header + [f'col{i}' for i in range(len(header), maxlen)]
    norm_rows = []
    for r in data_rows:
        if len(r) < maxlen:
            r = r + [''] * (maxlen - len(r))
        elif len(r) > maxlen:
            # keep extra columns as-is (we'll try to interpret)
            pass
        norm_rows.append(r)
    # Build candidate dataframe and then try to identify cols
    df0 = pd.DataFrame(norm_rows, columns=norm_header[:len(norm_header)])
    # Try to find best_test / best_val columns by numeric detection
    col_scores = {}
    for col in df0.columns:
        vals = df0[col].astype(str).str.strip()
        numeric_count = vals.map(is_float).sum()
        col_scores[col] = numeric_count
    # choose best_test as column with highest numeric_count but also with float values present
    sorted_cols = sorted(col_scores.items(), key=lambda x: x[1], reverse=True)
    if not sorted_cols:
        raise RuntimeError("No columns parsed from CSV.")
    # pick top two numeric columns as best_val and best_test (if more than one)
    top_numeric_cols = [c for c, cnt in sorted_cols if cnt > 0]
    best_test_col = None
    best_val_col = None
    if len(top_numeric_cols) >= 2:
        best_val_col = top_numeric_cols[0]
        best_test_col = top_numeric_cols[1]
    elif len(top_numeric_cols) == 1:
        best_test_col = top_numeric_cols[0]
    else:
        # try to find any token-like numeric in whole row (rare)
        raise RuntimeError("Could not detect numeric columns for best_val/best_test.")
    # Try to detect exp_tag column (non-numeric, frequent text)
    text_scores = {}
    for col in df0.columns:
        vals = df0[col].astype(str).str.strip()
        nonnum = (~vals.map(is_float)) & (vals != '')
        text_scores[col] = nonnum.sum()
    exp_tag_col = None
    # pick the column with highest non-numeric count (but not zero) and not the numeric columns
    cand_text_cols = sorted(text_scores.items(), key=lambda x: x[1], reverse=True)
    for c, cnt in cand_text_cols:
        if cnt > 0 and c not in (best_test_col, best_val_col):
            exp_tag_col = c
            break
    # detect seed/run column: integer column (many ints)
    seed_col = None
    for col in df0.columns:
        vals = df0[col].astype(str).str.strip()
        intcount = 0
        total = 0
        for v in vals:
            if v == '':
                continue
            total += 1
            try:
                if float(v).is_integer():
                    intcount += 1
            except:
                pass
        if total > 0 and intcount / total > 0.6 and col not in (best_test_col, best_val_col, exp_tag_col):
            seed_col = col
            break

    print(f"[INFO] inferred columns -> best_val: {best_val_col}, best_test: {best_test_col}, exp_tag: {exp_tag_col}, seed/run: {seed_col}")

    # build cleaned records
    cleaned = []
    skipped = 0
    for idx, row in df0.iterrows():
        try:
            best_test_val = None
            best_val_val = None
            # try direct parse from columns
            if best_test_col in df0.columns:
                v = row.get(best_test_col, '')
                if is_float(v):
                    best_test_val = float(v)
            if best_val_col in df0.columns:
                v2 = row.get(best_val_col, '')
                if is_float(v2):
                    best_val_val = float(v2)
            # if fail, try to extract numeric tokens from whole row
            if best_test_val is None:
                tokens = []
                for c in row:
                    # find numeric inside string
                    found = re.findall(r'[-+]?\d*\.\d+|\d+', str(c))
                    for f in found:
                        try:
                            tokens.append(float(f))
                        except:
                            pass
                if len(tokens) >= 1:
                    # assume last numeric is best_test
                    best_test_val = float(tokens[-1])
                if len(tokens) >= 2:
                    best_val_val = float(tokens[-2])
            if best_test_val is None or (isinstance(best_test_val, float) and (math.isnan(best_test_val))):
                skipped += 1
                continue
            rec = {
                'best_test': float(best_test_val),
                'best_val': float(best_val_val) if best_val_val is not None else float('nan'),
                'exp_tag': str(row.get(exp_tag_col, '')).strip() if exp_tag_col is not None else '',
                'seed': None,
                'run': None,
                'raw_idx': idx
            }
            if seed_col is not None:
                sc = row.get(seed_col, '')
                if is_float(sc):
                    rec['seed'] = int(float(sc))
                else:
                    rec['run'] = sc
            cleaned.append(rec)
        except Exception as e:
            skipped += 1

    print(f"[INFO] parsed rows: {len(cleaned)}, skipped malformed rows: {skipped}")
    return pd.DataFrame(cleaned)

def prepare_dataframe(path):
    df = try_pandas_read(path)
    if df is not None:
        # normalize column names
        df_cols = [c.lower() for c in df.columns]
        colmap = {c: c for c in df.columns}
        lower_to_orig = dict(zip(df_cols, df.columns))
        # try to ensure best_test present
        if 'best_test' not in df_cols:
            # attempt to find numeric columns and pick last numeric as best_test
            numeric_cols = []
            for col in df.columns:
                series = df[col].astype(str).str.strip()
                numeric_ratio = series.map(is_float).mean()
                if numeric_ratio > 0.5:
                    numeric_cols.append(col)
            if len(numeric_cols) >= 1:
                chosen = numeric_cols[-1]
                df = df.rename(columns={chosen: 'best_test'})
                print(f"[WARN] Renamed detected numeric column {chosen} -> best_test")
            else:
                raise RuntimeError("Could not detect best_test column in CSV")
        # lowercase column names for convenience
        df.columns = [c.lower() for c in df.columns]
        return df
    else:
        # robust parse
        df = robust_csv_parse(path)
        # ensure columns lower-case
        df.columns = [c.lower() for c in df.columns]
        return df

def analyze(df):
    # standardize column names
    # seek exp_tag, seed, run, best_test, best_val
    cols = set(df.columns)
    print("[INFO] final columns in cleaned DF:", df.columns.tolist())
    if 'best_test' not in df.columns:
        print("[ERROR] cleaned dataframe has no 'best_test' column. abort.")
        return
    # fill exp_tag if missing
    if 'exp_tag' not in df.columns or df['exp_tag'].isnull().all() or (df['exp_tag'].astype(str).str.strip()=='').all():
        print("[WARN] 'exp_tag' missing or empty for all rows.")
        # attempt to create exp_tag from cons_weight if present
        if 'cons_weight' in df.columns:
            df['exp_tag'] = df['cons_weight'].astype(str)
            print("[INFO] created exp_tag from cons_weight column.")
        else:
            df['exp_tag'] = 'unknown'
    # identify baseline & schannel labels if present
    tags = df['exp_tag'].astype(str).unique().tolist()
    print(f"[INFO] found exp_tag values: {tags}")

    # coerce best_test to numeric
    df['best_test'] = pd.to_numeric(df['best_test'], errors='coerce')
    before = len(df)
    df = df.dropna(subset=['best_test'])
    dropped = before - len(df)
    if dropped > 0:
        print(f"[INFO] dropped {dropped} rows with non-numeric best_test")
    # try to use seed if present
    if 'seed' in df.columns and df['seed'].notna().any():
        key = 'seed'
    elif 'run' in df.columns and df['run'].notna().any():
        key = 'run'
    else:
        # make artificial run index
        df = df.reset_index().rename(columns={'index': 'run'})
        key = 'run'
    print(f"[INFO] using pairing key: {key}")

    # check groups
    group_vals = df['exp_tag'].astype(str).unique().tolist()
    if 'baseline' in group_vals and 'schannel' in group_vals:
        # preferred grouped analysis
        base_df = df[df['exp_tag'].astype(str)=='baseline'].set_index(key)
        sch_df = df[df['exp_tag'].astype(str)=='schannel'].set_index(key)
        # find intersection keys for pairing
        common = base_df.index.intersection(sch_df.index)
        if len(common) >= 2:
            base_vals = base_df.loc[common]['best_test'].astype(float).values
            sch_vals = sch_df.loc[common]['best_test'].astype(float).values
            print(f"[INFO] Paired analysis on {len(common)} matching {key} entries.")
            # stats
            print("baseline mean,std,n:", np.mean(base_vals), np.std(base_vals, ddof=1), len(base_vals))
            print("schannel mean,std,n:", np.mean(sch_vals), np.std(sch_vals, ddof=1), len(sch_vals))
            t, p = stats.ttest_rel(sch_vals, base_vals)
            print("paired t-test (sch - baseline): t = %.6f, p = %.6e" % (t, p))
            # effect size (paired Cohen's d)
            diff = sch_vals - base_vals
            d = np.mean(diff) / np.std(diff, ddof=1)
            print("Cohen's d (paired): %.6f" % d)
            return
        else:
            print("[WARN] Could not find >=2 matching keys between baseline and schannel for pairing.")
            # fallthrough to independent analysis
    # if no baseline/schannel tags, try to split by tag values
    # if more than 1 tag, pick two largest groups
    groups = df.groupby('exp_tag')['best_test'].agg(['count','mean','std'])
    print("[INFO] groups summary:\n", groups)
    # attempt to pick two groups: baseline & schannel if present, else top two by count
    if len(groups) >= 2:
        if 'baseline' in groups.index and 'schannel' in groups.index:
            a_tag, b_tag = 'baseline', 'schannel'
        else:
            sorted_by_n = groups.sort_values('count', ascending=False)
            a_tag, b_tag = sorted_by_n.index[0], sorted_by_n.index[1]
            print(f"[INFO] comparing groups '{a_tag}' vs '{b_tag}' (top two by count).")
        a = df[df['exp_tag']==a_tag]['best_test'].astype(float).values
        b = df[df['exp_tag']==b_tag]['best_test'].astype(float).values
        print(f"{a_tag} mean,std,n: {np.mean(a):.6f} {np.std(a, ddof=1):.6f} {len(a)}")
        print(f"{b_tag} mean,std,n: {np.mean(b):.6f} {np.std(b, ddof=1):.6f} {len(b)}")
        # independent t-test (Welch)
        t, p = stats.ttest_ind(a, b, equal_var=False)
        print("Welch t-test: t = %.6f, p = %.6e" % (t, p))
        # Cohen's d (pooled)
        na, nb = len(a), len(b)
        sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
        # pooled sd
        pooled = math.sqrt(((na-1)*sa*sa + (nb-1)*sb*sb) / (na+nb-2)) if (na+nb-2)>0 else float('nan')
        if not math.isnan(pooled) and pooled > 0:
            d = (np.mean(a) - np.mean(b)) / pooled
            print("Cohen's d (pooled): %.6f" % d)
        return
    else:
        # no grouping or single group: print overall stats
        arr = df['best_test'].astype(float).values
        print("[INFO] Only one group/few rows available. Overall mean,std,n:", np.mean(arr), np.std(arr, ddof=1) if len(arr)>1 else float('nan'), len(arr))
        return

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    print("[INFO] reading:", path)
    df = prepare_dataframe(path)
    analyze(df)

if __name__ == '__main__':
    main()

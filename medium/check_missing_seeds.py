#!/usr/bin/env python3
# check_missing_seeds.py
import pandas as pd
import glob
import sys
from pathlib import Path

def check_file(p):
    print("----", p, "----")
    df = pd.read_csv(p)
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    # ensure expected cols
    expected = {'exp_tag','seed','run','best_val','best_test'}
    if not expected.issubset(set(df.columns)):
        print("[WARN] unexpected columns:", df.columns.tolist())
    # cast seed to int if possible
    try:
        df['seed'] = df['seed'].astype(int)
    except:
        pass
    # group
    summary = df.groupby(['seed','exp_tag'])['best_test'].agg(['count','last']).reset_index()
    # pivot to see presence
    pivot = df.pivot_table(index='seed', columns='exp_tag', values='best_test', aggfunc='last')
    print("Seeds in file:", len(pivot))
    print(pivot.head(10))
    # find seeds missing baseline or schannel
    needs = []
    for seed, row in pivot.iterrows():
        base = row.get('baseline') if 'baseline' in row.index else None
        sch = row.get('schannel') if 'schannel' in row.index else None
        # treat NaN as missing
        if pd.isna(base) or pd.isna(sch):
            needs.append((seed, base, sch))
        else:
            # treat placeholder cases (best_test == 0 or extremely small) as missing
            try:
                if float(base) == 0.0 or float(sch) == 0.0:
                    needs.append((seed, base, sch))
            except:
                pass
    if needs:
        print("[NEEDS RERUN COUNT]", len(needs))
        for s,b,sc in needs:
            print("seed:", s, "baseline:", b, "schannel:", sc)
    else:
        print("All seeds paired and non-zero!")
    return pivot, needs

def main():
    files = glob.glob("results/*_results_per_run_tagged*.csv") + glob.glob("results/*_results_per_run.csv")
    if not files:
        print("No result tagged CSVs found in results/.")
        return
    for f in files:
        try:
            check_file(f)
        except Exception as e:
            print("Failed to check", f, e)

if __name__ == '__main__':
    main()

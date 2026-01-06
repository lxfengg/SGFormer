#!/usr/bin/env python3
"""
parse_run_txts.py

Usage:
  python parse_run_txts.py [--dir RESULTS_DIR] [--out OUT_CSV] [--keep_raw]

Example:
  python parse_run_txts.py --dir results --out results/run_summaries_parsed.csv

Notes:
 - The script tries to be robust: it extracts common labeled fields (method, hidden, lr, Highest Test, Final Test, run_time...)
 - If it cannot parse some files, it reports them to results/run_summaries_badfiles.txt
"""
import re
import csv
import argparse
from pathlib import Path
from collections import OrderedDict
import pandas as pd

# --- patterns to extract ---
PATTERNS = {
    # configs (key: regex, group to capture)
    "method": re.compile(r'\bmethod[:=]\s*([A-Za-z0-9_\-]+)', re.IGNORECASE),
    "exp_tag": re.compile(r'\bexp_tag[:=]\s*([A-Za-z0-9_\-]+)', re.IGNORECASE),
    "dataset": re.compile(r'\bdataset[:=]\s*([A-Za-z0-9_\-]+)', re.IGNORECASE),
    "hidden": re.compile(r'\bhidden[:=]\s*([0-9]+)', re.IGNORECASE),
    "ours_layers": re.compile(r'\bours_layers[:=]\s*([0-9]+)', re.IGNORECASE),
    "lr": re.compile(r'\blr[:=]\s*([0-9]*\.?[0-9]+(?:e[-+]?\d+)?)', re.IGNORECASE),
    "use_graph": re.compile(r'\buse_graph[:=]\s*(True|False|true|false)', re.IGNORECASE),
    "train_prop": re.compile(r'\btrain_prop[:=]\s*([0-9]*\.?[0-9]+)', re.IGNORECASE),
    "valid_prop": re.compile(r'\bvalid_prop[:=]\s*([0-9]*\.?[0-9]+)', re.IGNORECASE),
    # runs summary
    "highest_train": re.compile(r'Highest\s*Train[:\s]*([0-9]*\.?[0-9]+)', re.IGNORECASE),
    "highest_val_epoch": re.compile(r'Highest\s*val\s*epoch[:\s]*([0-9]+)', re.IGNORECASE),
    "highest_test": re.compile(r'Highest\s*Test[:\s]*([0-9]*\.?[0-9]+)', re.IGNORECASE),
    "final_test": re.compile(r'Final\s*Test[:\s]*([0-9]*\.?[0-9]+)', re.IGNORECASE),
    "run_time": re.compile(r'run_time[:\s]*([0-9]*\.?[0-9]+)', re.IGNORECASE),
    # alternative variants
    "final_test_alt": re.compile(r'Final\s*Test[:\s]*([0-9]*\.?[0-9]+)\s*±', re.IGNORECASE),
    "highest_test_alt": re.compile(r'Highest\s*Test[:\s]*([0-9]*\.?[0-9]+)\s*±', re.IGNORECASE),
    # also sometimes single-line: "1 runs: Highest Train: 22.86 ± nan Highest val epoch:0 Highest Test: 14.80 ± nan Final Test: 14.80 ± nan run_time: 75.70"
    "inline_summary": re.compile(r'Highest\s*Train[:\s]*([0-9]*\.?[0-9]+).*?Highest\s*val\s*epoch[:\s]*([0-9]+).*?Highest\s*Test[:\s]*([0-9]*\.?[0-9]+).*?Final\s*Test[:\s]*([0-9]*\.?[0-9]+).*?run_time[:\s]*([0-9]*\.?[0-9]+)', re.IGNORECASE|re.DOTALL),
}

# filename heuristics for dataset, seed, exp_tag
FILENAME_DATASET = re.compile(r'([^/_\\]+)_(?:ours|ours|method)')  # e.g., cora_ours_...
FILENAME_SEED = re.compile(r'seed(\d+)|_(\d{3})')  # try to match seedNNN


def extract_from_text(text):
    res = {}
    # inline summary full match first (most compact)
    m = PATTERNS['inline_summary'].search(text)
    if m:
        res['highest_train'] = float(m.group(1))
        res['highest_val_epoch'] = int(m.group(2))
        res['highest_test'] = float(m.group(3))
        res['final_test'] = float(m.group(4))
        res['run_time'] = float(m.group(5))
    # other keys
    for k,pat in PATTERNS.items():
        if k == 'inline_summary':
            continue
        if k in res:
            continue
        m = pat.search(text)
        if not m:
            continue
        try:
            if k in ('hidden','ours_layers','highest_val_epoch'):
                res[k] = int(m.group(1))
            elif k in ('use_graph','exp_tag','method','dataset'):
                val = m.group(1)
                if k == 'use_graph':
                    res[k] = val.lower() in ('true','1','yes')
                else:
                    res[k] = val
            elif k in ('lr','highest_train','highest_test','final_test','run_time','train_prop','valid_prop'):
                res[k] = float(m.group(1))
            else:
                res[k] = m.group(1)
        except Exception:
            res[k] = m.group(1)
    return res


def parse_file(path: Path):
    txt = path.read_text(errors='replace')
    parsed = extract_from_text(txt)
    # heuristics from filename
    name = path.name
    if 'dataset' not in parsed:
        # try to get leading token before first underscore
        m = re.match(r'([A-Za-z0-9\-]+)_', name)
        if m:
            parsed['dataset'] = m.group(1)
    if 'exp_tag' not in parsed:
        m = re.search(r'(baseline|schannel|quick_check|default|sch_w[0-9._-]+)', name, re.IGNORECASE)
        if m:
            parsed['exp_tag'] = m.group(1)
    # seed guess
    m = re.search(r'seed[_\-]?(\d+)', name, re.IGNORECASE)
    if m:
        parsed['seed'] = m.group(1)
    else:
        m = re.search(r'_(\d{3})', name)
        if m:
            parsed['seed'] = m.group(1)
    # include file name and raw text length
    parsed['file'] = str(path)
    parsed['raw_len'] = len(txt)
    parsed['raw_head'] = txt[:200].replace('\n',' ')
    return parsed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', '-d', default='results', help='directory to search (recursive) for txt/log files')
    ap.add_argument('--out', '-o', default='results/run_summaries_parsed.csv', help='output CSV path')
    ap.add_argument('--keep_raw', action='store_true', help='keep raw text in CSV (may be large)')
    args = ap.parse_args()

    root = Path(args.dir)
    if not root.exists():
        print("Directory not found:", root)
        return

    files = list(root.rglob('*.txt')) + list(root.rglob('*.log')) + list(root.rglob('*.out'))
    files = sorted(set(files), key=lambda p: p.name)
    print(f"Found {len(files)} candidate text files under {root}")

    rows = []
    bad = []
    for f in files:
        try:
            p = parse_file(f)
            # require at least one of numeric results present
            if ('final_test' not in p) and ('highest_test' not in p) and ('raw_len' in p):
                bad.append(f)
            rows.append(p)
        except Exception as e:
            bad.append(f)

    # normalize keys and create DataFrame
    keys = set()
    for r in rows:
        keys.update(r.keys())
    # prefer an ordered set of common columns
    cols = ['file','dataset','exp_tag','seed','method','hidden','ours_layers','lr','use_graph',
            'train_prop','valid_prop',
            'highest_train','highest_val_epoch','highest_test','final_test','run_time',
            'raw_len','raw_head']
    # add any other keys found
    for k in sorted(keys):
        if k not in cols:
            cols.append(k)

    records = []
    for r in rows:
        rec = OrderedDict()
        for c in cols:
            if c in r:
                rec[c] = r[c]
            else:
                rec[c] = None
        records.append(rec)

    df = pd.DataFrame.from_records(records)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if not args.keep_raw:
        # drop raw_head if huge
        if 'raw_head' in df.columns:
            df = df.drop(columns=['raw_head'])
    df.to_csv(outp, index=False)
    print("Wrote parsed CSV:", outp)
    # write badfiles
    badp = outp.parent / (outp.stem + '.badfiles.txt')
    with open(badp, 'w') as f:
        for b in bad:
            f.write(str(b) + '\n')
    print("Wrote badfiles list:", badp)
    # quick summary
    try:
        print("\nQuick group summary (final_test mean by dataset,exp_tag):")
        print(df.groupby(['dataset','exp_tag'])['final_test'].agg(['count','mean','std']).sort_values(['dataset','mean'],ascending=[True,False]).head(50))
    except Exception as e:
        print("Summary error:", e)

if __name__ == '__main__':
    main()

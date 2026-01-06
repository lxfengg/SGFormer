#!/usr/bin/env python3
"""
rebuild_tagged.py

Robustly parse a possibly-malformed per-run CSV and produce a cleaned
tagged CSV with columns: exp_tag,seed,run,best_val,best_test

Usage:
    python rebuild_tagged.py <input_raw_csv> <output_tagged_csv> [--debug]

If parsing fails for some lines they will be saved to <output>.badlines.txt
"""
import sys
import csv
import io
import re
from pathlib import Path
import argparse
import math
import pandas as pd

NUM_RE = re.compile(r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$')

def is_number(s):
    try:
        float(s)
        return True
    except Exception:
        return False

def parse_line_tokens(tokens):
    """
    Heuristic to map a list of comma-separated tokens to
    exp_tag, seed, run, best_val, best_test.
    Returns dict with keys; missing values set to 'unknown' or 0.
    """
    # Trim tokens
    toks = [t.strip().strip('"').strip("'").replace('\r','') for t in tokens if t is not None]
    n = len(toks)

    out = {"exp_tag":"unknown", "seed":"unknown", "run":"0", "best_val":"0", "best_test":"0", "raw":",".join(toks)}

    # direct mapping when >=5 fields
    if n >= 5:
        out["exp_tag"] = toks[0] if toks[0] != "" else "unknown"
        out["seed"] = toks[1] if toks[1] != "" else "unknown"
        out["run"] = toks[2] if toks[2] != "" else "0"
        out["best_val"] = toks[3] if toks[3] != "" else "0"
        out["best_test"] = toks[4] if toks[4] != "" else "0"
        return out

    # if exactly 4 fields: try to detect numeric fields
    if n == 4:
        # find numeric indices
        num_idx = [i for i,t in enumerate(toks) if is_number(t)]
        if len(num_idx) >= 3:
            # assume last two numeric are best_val and best_test
            bv = toks[num_idx[-2]]
            bt = toks[num_idx[-1]]
            # seed is numeric before them if present
            seed = toks[num_idx[-3]] if len(num_idx) >= 3 else "unknown"
            # exp_tag is the non-numeric token (first token that's not the numeric ones)
            exp = None
            for i,t in enumerate(toks):
                if i not in num_idx:
                    exp = t
                    break
            out.update({"exp_tag": exp or "unknown", "seed": seed, "run":"0", "best_val": bv, "best_test": bt})
            return out
        else:
            # fallback: treat fields as exp_tag,seed,best_val,best_test
            out.update({"exp_tag": toks[0] or "unknown", "seed": toks[1] or "unknown", "run":"0", "best_val": toks[2] or "0", "best_test": toks[3] or "0"})
            return out

    # if exactly 3 fields: common format run,best_val,best_test or run,best_val,best_test
    if n == 3:
        # detect numeric pattern: if first numeric > 1000 maybe it's run or seed - keep as run
        if is_number(toks[0]) and is_number(toks[1]) and is_number(toks[2]):
            out.update({"exp_tag":"unknown", "seed":"unknown", "run": toks[0], "best_val": toks[1], "best_test": toks[2]})
            return out
        else:
            # fallback: last two numeric tokens -> best_val,best_test
            nums = [t for t in toks if is_number(t)]
            if len(nums) >= 2:
                out.update({"exp_tag": toks[0] if not is_number(toks[0]) else "unknown", "seed":"unknown", "run":"0", "best_val": nums[-2], "best_test": nums[-1]})
                return out
            else:
                # give up, put everything in raw
                out.update({"exp_tag": toks[0] if toks else "unknown", "seed":"unknown", "run":"0", "best_val":"0", "best_test":"0"})
                return out

    # n == 1 or other weird: try to extract last 3 numeric substrings anywhere
    flat = " ".join(toks)
    nums = re.findall(r'-?\d+\.\d+|-?\d+|\d+\.\d+[eE][-+]?\d+', flat)
    if len(nums) >= 3:
        # choose last three numbers as run,best_val,best_test
        run, bv, bt = nums[-3], nums[-2], nums[-1]
        out.update({"exp_tag":"unknown", "seed":"unknown", "run": run, "best_val":bv, "best_test":bt})
        return out

    # fallback: cannot parse numbers
    out["exp_tag"] = toks[0] if toks else "unknown"
    return out

def robust_read_lines(path, debug=False):
    raw = Path(path).read_text(errors='replace').splitlines()
    # remove BOM & empty head lines
    lines = [l for l in raw]
    # find header: prefer a line containing 'exp_tag' or 'best_test'
    header_idx = None
    for i,l in enumerate(lines[:50]):
        if 'exp_tag' in l and 'best_test' in l:
            header_idx = i
            break
    if header_idx is None:
        # fallback: first non-empty line
        for i,l in enumerate(lines):
            if l.strip():
                header_idx = i
                break
    if header_idx is None:
        raise RuntimeError("Could not find header or any non-empty line in file")
    header = lines[header_idx].strip().replace('\r','')
    data_lines = lines[header_idx+1:]
    if debug:
        print(f"[debug] header discovered (line {header_idx}): {header}")
        print(f"[debug] {len(data_lines)} data lines")
    return header, data_lines

def parse_file(inpath, outpath, debug=False):
    header, data_lines = robust_read_lines(inpath, debug=debug)
    parsed = []
    bad_lines = []
    reader = csv.reader
    for i,line in enumerate(data_lines, start=1):
        if not line.strip():
            continue
        # try CSV parse first
        try:
            toks = next(reader([line]))
        except Exception:
            toks = [line]
        # if line contains stray spaces & no commas, try splitting by whitespace
        if len(toks) == 1 and ',' not in toks[0]:
            # maybe space-separated
            toks = toks[0].split()
        rec = parse_line_tokens(toks)
        # validate numeric fields
        try:
            # ensure floats convertible
            bv = float(rec["best_val"])
            bt = float(rec["best_test"])
            # seed maybe int
            seed_raw = rec["seed"]
            # normalize seed: if numeric -> int string, else keep as-is
            if is_number(seed_raw):
                rec["seed"] = str(int(float(seed_raw)))
            # run to int
            if is_number(rec["run"]):
                rec["run"] = str(int(float(rec["run"])))
            parsed.append(rec)
        except Exception as e:
            bad_lines.append((i, line, str(e)))
            if debug:
                print(f"[debug] bad line {i}: {line} -> {e}")
            # still append a best-effort rec
            parsed.append(rec)

    # build DataFrame
    df = pd.DataFrame(parsed, columns=["exp_tag","seed","run","best_val","best_test","raw"])
    # convert best_val/best_test to numeric safely
    df['best_val'] = pd.to_numeric(df['best_val'], errors='coerce').fillna(0.0)
    df['best_test'] = pd.to_numeric(df['best_test'], errors='coerce').fillna(0.0)
    # ensure seed is string (for grouping)
    df['seed'] = df['seed'].astype(str)

    # Save bad lines for inspection
    badpath = Path(outpath).with_suffix('.badlines.txt')
    if bad_lines:
        with open(badpath, 'w', encoding='utf-8') as f:
            for idx, line, err in bad_lines:
                f.write(f"{idx}\t{err}\t{line}\n")
        if debug:
            print(f"[debug] wrote {len(bad_lines)} bad lines to {badpath}")
    else:
        # remove old badlines if any
        if badpath.exists():
            try:
                badpath.unlink()
            except Exception:
                pass

    # drop duplicates keep last occurrence for each (exp_tag,seed)
    df2 = df.drop_duplicates(subset=['exp_tag','seed'], keep='last').copy()
    # write out only desired columns
    outcols = ['exp_tag','seed','run','best_val','best_test']
    df2.to_csv(outpath, index=False, columns=outcols)
    # print summary
    print(f"Wrote cleaned tagged CSV: {outpath}  (rows before={len(df)}, rows after dedup={len(df2)})")
    print("Per-exp_tag stats (mean,std,count) on best_test:")
    try:
        stats = df2.groupby('exp_tag')['best_test'].agg(['mean','std','count']).sort_values('mean', ascending=False)
        print(stats.to_string())
    except Exception as e:
        print("Failed to compute stats:", e)
    return df, df2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', help='raw CSV path')
    parser.add_argument('outfile', help='cleaned tagged CSV out path')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    infile = args.infile
    outfile = args.outfile
    df_all, df_clean = parse_file(infile, outfile, debug=args.debug)
    if args.debug:
        print("[debug] Done. Sample of cleaned rows:")
        print(df_clean.head(20).to_string(index=False))

if __name__ == '__main__':
    main()

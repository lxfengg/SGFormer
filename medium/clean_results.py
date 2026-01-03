#!/usr/bin/env python3
"""
clean_results.py
从可能被日志污染的 results CSV 中提取合法的三列数值行：
  run,best_val,best_test
并写入 <origname>_cleaned.csv

用法:
  python clean_results.py path/to/your_results_per_run.csv

如果不指定路径，会默认使用 "results/cora_ours_results_per_run.csv"
"""
import sys
import os
import re
import csv

# 正则：匹配 run(int), float, float（支持科学计数法）
LINE_RE = re.compile(r'\s*(\d+)\s*,\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)')

def clean_file(inpath, outpath):
    kept = []
    total = 0
    with open(inpath, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f, start=1):
            total += 1
            s = line.strip()
            if not s:
                continue
            # Try direct regex match anywhere in the line
            m = LINE_RE.search(s)
            if m:
                run = int(m.group(1))
                try:
                    best_val = float(m.group(2))
                    best_test = float(m.group(3))
                    kept.append((run, best_val, best_test, i, s))
                except Exception:
                    # parse fail -> skip
                    continue
            else:
                # try CSV-splitting fallback: split by comma and try first 3 tokens
                parts = [p.strip() for p in s.split(',')]
                if len(parts) >= 3:
                    try:
                        run = int(parts[0])
                        best_val = float(parts[1])
                        best_test = float(parts[2])
                        kept.append((run, best_val, best_test, i, s))
                    except Exception:
                        pass
    # write cleaned CSV
    os.makedirs(os.path.dirname(outpath) or '.', exist_ok=True)
    with open(outpath, 'w', newline='') as fo:
        writer = csv.writer(fo)
        writer.writerow(['run','best_val','best_test'])
        for (run, best_val, best_test, _, _) in kept:
            writer.writerow([run, f'{best_val:.6f}', f'{best_test:.6f}'])
    return total, len(kept), kept

def main():
    if len(sys.argv) >= 2:
        inpath = sys.argv[1]
    else:
        inpath = os.path.join('results', 'cora_ours_results_per_run.csv')
    if not os.path.exists(inpath):
        print("[ERROR] input file not found:", inpath)
        sys.exit(2)
    base, ext = os.path.splitext(inpath)
    outpath = base + '_cleaned' + ext
    total, kept_count, kept = clean_file(inpath, outpath)
    print(f"[INFO] parsed {total} lines; kept {kept_count} numeric rows -> wrote {outpath}")
    if kept_count == 0:
        print("[WARN] no numeric rows extracted. Inspect original file manually (head/tail).")
    else:
        print("[INFO] sample of extracted rows (run,best_val,best_test,line_no,raw_line):")
        for t in kept[:10]:
            print(" ", t)
    print("Done.")

if __name__ == '__main__':
    main()

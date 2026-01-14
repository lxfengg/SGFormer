# 保存为 verify_results.py
# 用途：校验 results demo CSV/汇总 CSV 是否满足格式（exp_tag,seed,run,best_val,best_test）
# 使用：python verify_results.py --csv results/cora_ours_results_per_run.csv
# 输出：stdout 报告并返回非0退出码当有严重问题

import argparse
import csv
import sys
from collections import defaultdict

def load_csv(path):
    rows = []
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for i, r in enumerate(reader, start=2):
            rows.append((i, r))
    return header, rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True, help='path to results CSV (expected header: exp_tag,seed,run,best_val,best_test)')
    p.add_argument('--require-seeds', nargs='*', type=int, default=None,
                   help='optional list of seeds to require; will report missing ones')
    args = p.parse_args()

    header, rows = load_csv(args.csv)
    if header is None:
        print("[ERROR] CSV is empty or missing header.")
        sys.exit(2)
    header = [h.strip() for h in header]
    expected = ['exp_tag', 'seed', 'run', 'best_val', 'best_test']
    if header[:5] != expected:
        print(f"[ERROR] Header mismatch. Expected first five columns: {expected}. Found: {header[:5]}")
        sys.exit(2)
    print(f"[OK] Header looks good: {header[:5]}")

    seen = set()
    duplicates = []
    malformed = []
    by_tag = defaultdict(list)

    for lineno, cols in rows:
        # strip and ignore empty trailing whitespace columns
        cols = [c.strip() for c in cols]
        if len(cols) < 5:
            malformed.append((lineno, cols))
            continue
        exp_tag, seed_str, run_str, best_val_str, best_test_str = cols[:5]
        key = (exp_tag, seed_str, run_str)
        if key in seen:
            duplicates.append((lineno, key))
        seen.add(key)
        try:
            seed = int(seed_str)
            run_idx = int(run_str)
            bv = float(best_val_str)
            bt = float(best_test_str)
        except Exception as e:
            malformed.append((lineno, cols))
            continue
        by_tag[exp_tag].append(seed)

    if duplicates:
        print("[ERROR] Found duplicate lines for (exp_tag,seed,run):")
        for ln, key in duplicates:
            print(f"  line {ln}: {key}")
    else:
        print("[OK] No duplicate (exp_tag,seed,run) found.")

    if malformed:
        print("[ERROR] Found malformed lines (wrong column count or non-numeric fields):")
        for ln, cols in malformed[:10]:
            print(f"  line {ln}: {cols}")
        if len(malformed) > 10:
            print(f"  ... total malformed lines: {len(malformed)}")
    else:
        print("[OK] No malformed lines detected.")

    # report seed coverage if requested
    if args.require_seeds:
        required = set(args.require_seeds)
        for tag, seeds in by_tag.items():
            sset = set(seeds)
            missing = sorted(list(required - sset))
            if missing:
                print(f"[WARN] For tag '{tag}', missing seeds: {missing}")
            else:
                print(f"[OK] For tag '{tag}', all required seeds present.")

    # summary
    print("Summary of tags and seed counts:")
    for tag, seeds in by_tag.items():
        uniq = sorted(set(seeds))
        print(f"  {tag}: unique seeds={len(uniq)}, seeds sample={uniq[:10]}")

    # final status
    if duplicates or malformed:
        print("[FAIL] CSV verification failed.")
        sys.exit(3)
    else:
        print("[PASS] CSV verification passed.")
        sys.exit(0)

if __name__ == '__main__':
    main()

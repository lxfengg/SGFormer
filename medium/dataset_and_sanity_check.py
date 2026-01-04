#!/usr/bin/env python3
# dataset_and_sanity_check.py
# Quick script to load a dataset via dataset.load_nc_dataset(args)
# and print diagnostics. Optional quick-run of main.py for end-to-end smoke test.

import os
import sys
import argparse
import subprocess
import traceback
import time

# Add project medium directory to PYTHONPATH if needed
# (so `import dataset` works when running from project/medium)
PROJ_MEDIUM_DIR = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
if PROJ_MEDIUM_DIR not in sys.path:
    sys.path.insert(0, PROJ_MEDIUM_DIR)

# import load_nc_dataset from your dataset loader
try:
    from dataset import load_nc_dataset
except Exception as e:
    print("[ERROR] failed to import load_nc_dataset from dataset.py:", e)
    print(traceback.format_exc())
    # allow the script to continue to show cleaner error
    load_nc_dataset = None

def parse_args():
    p = argparse.ArgumentParser(description="Dataset sanity check and optional quick main.py run")
    p.add_argument('--dataset', '-d', required=True, help='dataset name (e.g. cora, citeseer, pubmed)')
    p.add_argument('--data_dir', '-D', default=os.environ.get('DATA_DIR', '/mnt/e/code/SGFormer/data'),
                   help='root data dir (overrides DATAPATH). Default uses DATA_DIR env or /mnt/e/code/SGFormer/data')
    p.add_argument('--no_feat_norm', action='store_true', help='pass no_feat_norm to loader')
    p.add_argument('--run_main_quick', action='store_true',
                   help='after checks, run a short quick main.py run (3 seeds) to verify training pipeline')
    p.add_argument('--python', default='python', help='python executable for quick-run')
    p.add_argument('--use_cpu', action='store_true', help='pass --cpu to main.py when quick-running')
    p.add_argument('--num_quick_runs', type=int, default=3, help='how many quick runs/seeds for quick-run')
    return p.parse_args()

def make_args_namespace(dataset, data_dir, no_feat_norm=False):
    # create a minimal args namespace that dataset.load_nc_dataset expects
    ns = argparse.Namespace()
    ns.dataset = dataset
    ns.data_dir = data_dir
    ns.no_feat_norm = no_feat_norm
    return ns

def print_dataset_info(ds):
    try:
        g = ds.graph
        lab = ds.label
        print("=== Dataset diagnostics ===")
        print("num_nodes (graph['num_nodes']):", g.get('num_nodes', getattr(g.get('node_feat', None), 'shape', None)))
        print("edge_index shape/type:", type(g.get('edge_index')), getattr(g.get('edge_index'), 'shape', None))
        print("node_feat shape/type:", type(g.get('node_feat')), getattr(g.get('node_feat'), 'shape', None))
        if lab is not None:
            print("label dtype/shape:", lab.dtype if hasattr(lab, 'dtype') else type(lab), lab.shape if hasattr(lab, 'shape') else None)
            try:
                import torch
                if isinstance(lab, torch.Tensor):
                    if lab.dim() > 1:
                        lab_printable = lab.squeeze(1)
                    else:
                        lab_printable = lab
                    unique, counts = torch.unique(lab_printable, return_counts=True)
                    print("DEBUG: label unique:", unique.tolist())
                    print("DEBUG: label counts:", counts.tolist())
                else:
                    import numpy as np
                    arr = np.asarray(lab)
                    vals, cnts = np.unique(arr, return_counts=True)
                    print("DEBUG: label unique:", vals.tolist())
                    print("DEBUG: label counts:", cnts.tolist())
            except Exception:
                pass
    except Exception as e:
        print("[WARN] failed to print some dataset info:", e)
        print(traceback.format_exc())

def quick_run_main(python_exec, dataset, data_dir, use_cpu, seeds=3):
    print("=== Quick run: launching short main.py runs (smoke test) ===")
    base_cmd = [python_exec, 'main.py', '--dataset', dataset, '--method', 'ours', '--data_dir', data_dir, '--runs', '1', '--display_step', '50']
    if use_cpu:
        base_cmd += ['--cpu']
    # use small seeds to run quick checks
    seed_list = [100+i for i in range(seeds)]
    for s in seed_list:
        cmd = base_cmd + ['--seed', str(s), '--exp_tag', 'quick_check']
        print("[INFO] Running:", " ".join(cmd))
        try:
            # run and stream output
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            start = time.time()
            # stream for a short while (limit ~ 30s) to detect immediate failures
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                print(line, end='')
                # stop early if run finishes or we saw some epochs
                if time.time() - start > 30:  # let it run 30s max per quick run
                    proc.terminate()
                    print("[INFO] terminated quick-run (time limit reached).")
                    break
            proc.wait(timeout=5)
        except Exception as e:
            print("[ERROR] quick-run failed for seed", s, ":", e)
            print(traceback.format_exc())

def main():
    args = parse_args()
    print(f"[INFO] dataset={args.dataset} data_dir={args.data_dir} no_feat_norm={args.no_feat_norm}")
    if load_nc_dataset is None:
        print("[ERROR] dataset loader not available; cannot proceed.")
        sys.exit(1)

    # build minimal args namespace for loader
    loader_args = make_args_namespace(args.dataset, args.data_dir, args.no_feat_norm)
    try:
        ds = load_nc_dataset(loader_args)
    except Exception as e:
        print("[ERROR] load_nc_dataset raised exception:")
        print(traceback.format_exc())
        sys.exit(2)

    # print info
    print_dataset_info(ds)

    # print splits if available
    try:
        if hasattr(ds, 'train_idx'):
            print("train_idx length:", getattr(ds, 'train_idx').shape if hasattr(ds.train_idx, 'shape') else len(ds.train_idx))
        if hasattr(ds, 'valid_idx'):
            print("valid_idx length:", getattr(ds, 'valid_idx').shape if hasattr(ds.valid_idx, 'shape') else len(ds.valid_idx))
        if hasattr(ds, 'test_idx'):
            print("test_idx length:", getattr(ds, 'test_idx').shape if hasattr(ds.test_idx, 'shape') else len(ds.test_idx))
    except Exception:
        print("[WARN] could not print split idxs:", traceback.format_exc())

    print("=== Done dataset checks ===")

    if args.run_main_quick:
        quick_run_main(args.python, args.dataset, args.data_dir, args.use_cpu, seeds=args.num_quick_runs)

if __name__ == '__main__':
    main()

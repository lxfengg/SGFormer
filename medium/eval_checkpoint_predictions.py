#!/usr/bin/env python3
# Robust eval_checkpoint_predictions.py
# Replace the existing script with this file. It attempts to build a fallback args Namespace
# from checkpoint['args'] and fill missing attributes with safe defaults so parse_method won't fail.

import os
import sys
import json
import argparse
import torch
import numpy as np

from dataset import load_nc_dataset
from parse import parse_method  # your repo's parse_method
from data_utils import load_fixed_splits, evaluate, eval_acc
from types import SimpleNamespace

def dict_to_ns(d):
    """Convert dict to namespace, keep nested dicts as-is."""
    if isinstance(d, SimpleNamespace):
        return d
    if isinstance(d, dict):
        return SimpleNamespace(**d)
    return d

def ensure_defaults(ns):
    """Ensure a set of commonly-used args exist on the namespace with safe defaults."""
    defaults = {
        'method': getattr(ns, 'method', 'ours'),
        'alpha': getattr(ns, 'alpha', 0.5),
        'num_layers': getattr(ns, 'num_layers', 2),
        'hidden_channels': getattr(ns, 'hidden_channels', 64),
        'num_heads': getattr(ns, 'num_heads', 4),
        'backbone': getattr(ns, 'backbone', 'gcn'),
        'use_graph': getattr(ns, 'use_graph', True),
        'ours_layers': getattr(ns, 'ours_layers', 2),
        'ours_dropout': getattr(ns, 'ours_dropout', 0.5),
        'dropout': getattr(ns, 'dropout', 0.5),
        'lr': getattr(ns, 'lr', 0.01),
        'weight_decay': getattr(ns, 'weight_decay', 5e-4),
        'ours_weight_decay': getattr(ns, 'ours_weight_decay', 5e-4),
        'ours_use_weight': getattr(ns, 'ours_use_weight', False),
        'use_bn': getattr(ns, 'use_bn', False),
        'use_residual': getattr(ns, 'use_residual', False),
        'use_act': getattr(ns, 'use_act', True),
        'aggregate': getattr(ns, 'aggregate', 'sum'),
        'hops': getattr(ns, 'hops', 2),
        'gat_heads': getattr(ns, 'gat_heads', 1),
        'out_heads': getattr(ns, 'out_heads', 1),
        'lamda': getattr(ns, 'lamda', 0.0),
        'num_elayers': getattr(ns, 'num_elayers', 1),
        'encoder_emdim': getattr(ns, 'encoder_emdim', 64),
        'attention': getattr(ns, 'attention', 'dot'),
        'ours_dropout': getattr(ns, 'ours_dropout', 0.5),
        'cons_weight': getattr(ns, 'cons_weight', 0.0),
        'cons_loss': getattr(ns, 'cons_loss', 'prob_mse'),
        'cons_confidence': getattr(ns, 'cons_confidence', 0.0),
        'ema_tau': getattr(ns, 'ema_tau', 0.99),
        'ema_start': getattr(ns, 'ema_start', 10),
        'seed': getattr(ns, 'seed', 0),
    }
    for k, v in defaults.items():
        if not hasattr(ns, k):
            setattr(ns, k, v)
    return ns

def safe_load_ckpt(ckpt_path, device):
    try:
        ck = torch.load(ckpt_path, map_location=device)
        return ck
    except Exception as e:
        print(f"[ERROR] loading checkpoint {ckpt_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--data_dir', default=None)
    args = parser.parse_args()

    device = torch.device("cpu")
    ckpt = safe_load_ckpt(args.ckpt, device)
    if ckpt is None:
        print("[ERROR] cannot load checkpoint, exiting")
        return

    # If checkpoint contains args as dict, namespace-ify them; otherwise use as-is.
    raw_args = ckpt.get('args', {})
    if isinstance(raw_args, dict):
        dargs = dict_to_ns(raw_args)
    else:
        dargs = raw_args if raw_args is not None else SimpleNamespace()
    # Ensure defaults to avoid AttributeError in parse_method
    dargs = ensure_defaults(dargs)

    # Now try to create dataset and model
    try:
        # load dataset with provided options if possible
        if args.data_dir:
            setattr(dargs, 'data_dir', args.data_dir)
        else:
            # if ckpt args provide data_dir, keep it; else set reasonable default
            if not hasattr(dargs, 'data_dir') or dargs.data_dir is None:
                setattr(dargs, 'data_dir', os.path.join(os.getcwd(), 'data'))

        # load dataset using your repo loader
        dataset = load_nc_dataset(dargs)
        if len(dataset.label.shape) == 1:
            dataset.label = dataset.label.unsqueeze(1)
        c = int(max(dataset.label.max().item() + 1, dataset.label.shape[1]))
        d = int(dataset.graph['node_feat'].shape[1])
    except Exception as e:
        print(f"[WARN] failed to load dataset or infer c/d from data: {e}")
        c = None; d = None

    model = None
    try:
        # parse_method expects (method, args, c, d, device)
        model = parse_method(dargs.method, dargs, c, d, device)
    except Exception as e:
        print(f"[WARN] parse_method failed with provided dargs: {e}")
        # attempt a second fallback by setting minimal args
        fb = SimpleNamespace(method=getattr(dargs,'method','ours'),
                             alpha=getattr(dargs,'alpha',0.5),
                             num_layers=getattr(dargs,'num_layers',2),
                             hidden_channels=getattr(dargs,'hidden_channels',64),
                             use_graph=getattr(dargs,'use_graph',True))
        try:
            model = parse_method(fb.method, fb, c, d, device)
            print("[INFO] Fallback parse_method creation succeeded with minimal args.")
        except Exception as e2:
            print(f"[ERROR] Fallback parse_method also failed: {e2}")
            model = None

    # If model available, attempt to load model_state
    if model is not None:
        try:
            model_state = ckpt.get('model_state', None)
            if model_state is not None:
                model.load_state_dict(model_state)
            else:
                print("[WARN] checkpoint missing 'model_state' key.")
        except Exception as e:
            print(f"[WARN] failed loading model_state: {e}")

        # Optionally evaluate predictions and write JSON summary
        try:
            # reconstruct an eval split if possible (best-effort)
            # If dataset available, run evaluation
            if 'dataset' in locals() and dataset is not None:
                eval_func = eval_acc
                split_idx = dataset.get_idx_split() if hasattr(dataset, 'get_idx_split') else None
                res = evaluate(model, dataset, split_idx, eval_func, None, dargs)
                # write results summary
                out = {
                    'ckpt_path': args.ckpt,
                    'basename': os.path.basename(args.ckpt),
                    'seed': getattr(dargs, 'seed', None),
                    'exp_tag': getattr(dargs, 'exp_tag', None),
                    'eval_result': res
                }
            else:
                out = {'ckpt_path': args.ckpt, 'note': 'model created but dataset unavailable for eval'}
        except Exception as e:
            out = {'ckpt_path': args.ckpt, 'note': f'eval failed: {e}'}

    else:
        # No model; still save minimal info from checkpoint + pseudo_label_quality if any
        out = {
            'ckpt_path': args.ckpt,
            'basename': os.path.basename(args.ckpt),
            'seed': getattr(dargs, 'seed', None),
            'exp_tag': getattr(dargs, 'exp_tag', None),
            'note': 'model instantiation failed; see logs'
        }

    # save JSON under results/diagnosis_per_ckpt/
    os.makedirs('results/diagnosis_per_ckpt', exist_ok=True)
    base = os.path.basename(args.ckpt)
    out_path = os.path.join('results/diagnosis_per_ckpt', f'diagnosis_{base}.json')
    try:
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"[INFO] Wrote diagnosis JSON to {out_path}")
    except Exception as e:
        print(f"[ERROR] failed to write diagnosis JSON: {e}")

if __name__ == '__main__':
    main()

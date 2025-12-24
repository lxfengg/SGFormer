import argparse
import copy
import os
import random
import sys
import warnings
import time, subprocess

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils import class_rand_splits, eval_acc, eval_rocauc, evaluate, load_fixed_splits, class_rand_splits, to_sparse_tensor
from dataset import load_nc_dataset
from logger import Logger
from parse import parse_method, parser_add_default_args, parser_add_main_args
from torch_geometric.utils import (add_self_loops, remove_self_loops,
                                   to_undirected)

warnings.filterwarnings('ignore')

# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def get_gpu_memory_map():
    """Get the current gpu usage.
    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = np.array([int(x) for x in result.strip().split('\n')])
    # gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser.add_argument('--cons_warm', type=int, default=10, help='supervised pre-warm epochs before consistency')
parser.add_argument('--cons_up', type=int, default=40, help='epochs to linearly warm up consistency weight')
parser.add_argument('--cons_weight', type=float, default=1.0, help='final consistency weight (alpha)')
parser.add_argument('--cons_temp', type=float, default=1.0,
                    help='temperature for probability-based consistency loss')
parser.add_argument(
    '--ema_tau',
    type=float,
    default=0.99,
    help='EMA momentum for teacher model in step2'
)


parser_add_main_args(parser)
args = parser.parse_args()
parser_add_default_args(args)
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)
                          ) if torch.cuda.is_available() else torch.device("cpu")

### Load and preprocess data ###
dataset = load_nc_dataset(args)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)

dataset_name = args.dataset

if args.rand_split:
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                     for _ in range(args.runs)]
elif args.rand_split_class:
    split_idx_lst = [class_rand_splits(
        dataset.label, args.label_num_per_class, args.valid_num, args.test_num)]
else:
    split_idx_lst = load_fixed_splits(
        dataset, name=args.dataset, protocol=args.protocol)

dataset.label = dataset.label.to(device)

# ----------------- DEBUG: label & mask check -----------------
# place immediately after: dataset.label = dataset.label.to(device)
try:
    labels = dataset.label
    if labels.dim() > 1:
        labels_squeezed = labels.squeeze(1)
    else:
        labels_squeezed = labels
    unique, counts = torch.unique(labels_squeezed, return_counts=True)
    print("DEBUG: labels shape:", tuple(labels_squeezed.shape))
    print("DEBUG: label unique:", unique.tolist())
    print("DEBUG: label counts:", counts.tolist())
except Exception as e:
    print("DEBUG: failed to print labels info:", e)

# If you want to inspect the first split's train_idx counts (after split_idx_lst is created)
try:
    if 'split_idx_lst' in locals() and len(split_idx_lst) > 0:
        sample_split = split_idx_lst[0]
        if 'train' in sample_split:
            tr = sample_split['train']
            # tr may be a tensor of indices or bool mask
            if isinstance(tr, torch.Tensor):
                if tr.dtype == torch.bool:
                    print("DEBUG: sample_split train mask sum:", int(tr.sum().item()))
                else:
                    print("DEBUG: sample_split train idx length:", len(tr))
            else:
                print("DEBUG: sample_split train type:", type(tr))
except Exception as e:
    print("DEBUG: failed to inspect split_idx_lst:", e)
# -------------------------------------------------------------

n = dataset.graph['num_nodes']
# infer the number of classes for non one-hot and one-hot labels
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

_shape = dataset.graph['node_feat'].shape
print(f'features shape={_shape}')

# whether or not to symmetrize
if args.dataset not in {'deezer-europe'}:
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])

dataset.graph['edge_index'], dataset.graph['node_feat'] = \
    dataset.graph['edge_index'].to(
        device), dataset.graph['node_feat'].to(device)

if args.method == 'graphormer':
    dataset.graph['x'] = dataset.graph['x'].to(device)
    dataset.graph['in_degree'] = dataset.graph['in_degree'].to(device)
    dataset.graph['out_degree'] = dataset.graph['out_degree'].to(device)
    dataset.graph['spatial_pos'] = dataset.graph['spatial_pos'].to(device)
    dataset.graph['attn_bias'] = dataset.graph['attn_bias'].to(device)

print(f"num nodes {n} | num classes {c} | num node feats {d}")

### Load method ###
model = parse_method(args.method, args, c, d, device)

import copy
teacher = copy.deepcopy(model)
for p in teacher.parameters():
    p.requires_grad = False
teacher.eval()

# using rocauc as the eval function
if args.dataset in ('deezer-europe'):
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.NLLLoss()

eval_func = eval_acc

logger = Logger(args.runs, args)

model.train()

### Training loop ###
patience = 0
if args.method == 'ours' and args.use_graph:
    optimizer = torch.optim.Adam([
        {'params': model.params1, 'weight_decay': args.ours_weight_decay},
        {'params': model.params2, 'weight_decay': args.weight_decay}
    ],
        lr=args.lr)
else:
    optimizer = torch.optim.Adam(
        model.parameters(), weight_decay=args.weight_decay, lr=args.lr)

run_time_list = []

for run in range(args.runs):
    if args.dataset in ['cora', 'citeseer', 'pubmed'] and args.protocol == 'semi':
        split_idx = split_idx_lst[0]
    else:
        split_idx = split_idx_lst[run]
    train_idx = split_idx['train'].to(device)
    model.reset_parameters()

    best_val = float('-inf')
    patience = 0
    for epoch in range(args.epochs):
        start_time = time.perf_counter()
                # ====== START: S-channel (SimGRACE-style) integration ======
        # This block replaces the original single-forward -> loss -> backward part.
        # It performs two stochastic forwards and adds a logits-space MSE consistency loss.
        model.train()
        optimizer.zero_grad()
        emb = None

        # --- hyperparameters for S-channel (use getattr to keep compatibility with args) ---
        E_warm = getattr(args, "cons_warm", 10)      # pre-warm epochs (only supervised)
        E_alpha_up = getattr(args, "cons_up", 40)    # epochs to warm up alpha
        alpha_target = getattr(args, "cons_weight", 1.0)  # final consistency weight (alpha)

        # --- Two stochastic forwards for SimGRACE-style no-augmentation consistency ---
        # Note: model(dataset) may return different shapes for nodeformer; handle both cases.
        if args.method == 'nodeformer':
            out1, link_loss1 = model(dataset)
            out2, link_loss2 = model(dataset)
        else:
            out1 = model(dataset)
            out2 = model(dataset)

        # For graphormer dataset-specific squeeze handling (mirror original logic)
        if args.method == 'graphormer':
            # original code did: out = out.squeeze(0)
            out1 = out1.squeeze(0)
            out2 = out2.squeeze(0)

        # --- supervised loss computation (use out1 as supervised prediction to keep original behavior) ---
        # handle special 'deezer-europe' label formatting as in original code
        if args.dataset in ('deezer-europe'):
            if dataset.label.shape[1] == 1:
                true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
            else:
                true_label = dataset.label
            # supervised loss expects floats for this dataset (mirror original line)
            sup_loss = criterion(out1[train_idx], true_label.squeeze(1)[train_idx].to(torch.float))
        else:
            out1_logp = F.log_softmax(out1, dim=1)
            sup_loss = criterion(out1_logp[train_idx], dataset.label.squeeze(1)[train_idx])

        # optional nodeformer special link loss handling: average link losses from two forwards
        link_loss_avg = None
        if args.method == 'nodeformer':
            # ensure shape/format consistent with original expectation (list of values)
            # combine lists element-wise by averaging if both are lists
            try:
                # if link_loss1, link_loss2 are lists
                link_loss_avg = [(a + b) / 2.0 for a, b in zip(link_loss1, link_loss2)]
            except Exception:
                # fallback: if scalar
                link_loss_avg = [(link_loss1 + link_loss2) / 2.0]

        # --- consistency loss in logits space (more stable than prob space) ---
        # Use raw logits MSE between the two forward passes (SimGRACE style)
        # cons_loss = F.mse_loss(out1, out2)
        # Option A: normalize logits per-node then MSE
        # out1_n = F.normalize(out1, p=2, dim=1)   # each row normalized
        # out2_n = F.normalize(out2, p=2, dim=1)
        # cons_loss = F.mse_loss(out1_n, out2_n)
        # Option B: probability-space MSE with temperature
        # T = getattr(args, "cons_temp", 0.5)   # 新增 argparse 参数 --cons_temp
        # p1 = F.softmax(out1 / T, dim=1)
        # p2 = F.softmax(out2 / T, dim=1)
        # cons_loss = F.mse_loss(p1, p2)
        with torch.no_grad():
            if args.method == 'nodeformer':
                t_out, _ = teacher(dataset)
            else:
                t_out = teacher(dataset)
            if args.method == 'graphormer':
                t_out = t_out.squeeze(0)

        # normalize (或用 softmax as preferred)
        out1_n = F.normalize(out1, p=2, dim=1)
        t_out_n = F.normalize(t_out, p=2, dim=1)
        cons_loss = F.mse_loss(out1_n, t_out_n)


        # --- alpha warmup schedule (linear) ---
        if epoch <= E_warm:
            alpha = 0.0
        else:
            t = epoch - E_warm
            alpha = alpha_target * min(1.0, float(t) / float(E_alpha_up))

        # --- total loss: supervised + alpha * consistency (and nodeformer link loss subtraction if applicable) ---
        loss = sup_loss + alpha * cons_loss

        # incorporate nodeformer link loss same way original code did (loss -= args.lamda * mean(link_loss))
        if args.method == 'nodeformer' and link_loss_avg is not None:
            # preserve original form: loss -= args.lamda * sum(link_loss_) / len(link_loss_)
            loss -= args.lamda * sum(link_loss_avg) / max(1, len(link_loss_avg))

        try:
            sup_val = float(sup_loss.item()) if 'sup_loss' in locals() else float(loss.item())
        except:
            sup_val = None
        try:
            cons_val = float(cons_loss.item()) if 'cons_loss' in locals() else 0.0
        except:
            cons_val = None
        print(f"[DEBUG] Epoch {epoch:03d} | sup_loss={sup_val:.4f} | cons_loss={cons_val:.4f} | alpha={alpha:.4f} | total_loss={loss.item():.4f}")
        # Backprop and update
        loss.backward()

        total_grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                try:
                    total_grad_norm += float(p.grad.data.norm(2).item())
                except:
                    pass
        print(f"[DEBUG] Epoch {epoch:03d} | grad_norm={total_grad_norm:.4f}")
        # gradient clipping to avoid potential explosion when adding extra losses
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        tau = getattr(args, 'ema_tau', 0.99)
        for s_param, t_param in zip(model.parameters(), teacher.parameters()):
            t_param.data.mul_(tau).add_(s_param.data * (1.0 - tau))

        # ====== END: S-channel integration ======

        end_time = time.perf_counter()
        run_time = 1000 * (end_time - start_time)
        run_time_list.append(run_time)

        result = evaluate(model, dataset, split_idx,
                          eval_func, criterion, args)
        logger.add_result(run, result[:-1])

        if result[1] > best_val:
            best_val = result[1]
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break

        if epoch % args.display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * result[0]:.2f}%, '
                  f'Valid: {100 * result[1]:.2f}%, '
                  f'Test: {100 * result[2]:.2f}%')
    logger.print_statistics(run)

run_time = sum(run_time_list) / len(run_time_list)
results = logger.print_statistics()
print(results)
out_folder = 'results'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

def make_print(method):
    print_str = ''
    if args.rand_split_class:
        print_str += f'label per class:{args.label_num_per_class}, valid:{args.valid_num},test:{args.test_num}\n'
    else:
        print_str += f'train_prop:{args.train_prop}, valid_prop:{args.valid_prop}'
    if method == 'ours':
        use_weight=' ours_use_weight' if args.ours_use_weight else ''
        print_str += f'method: {args.method} hidden: {args.hidden_channels} ours_layers:{args.ours_layers} lr:{args.lr} use_graph:{args.use_graph} aggregate:{args.aggregate} graph_weight:{args.graph_weight} alpha:{args.alpha} ours_decay:{args.ours_weight_decay} ours_dropout:{args.ours_dropout} epochs:{args.epochs} use_feat_norm:{not args.no_feat_norm} use_bn:{args.use_bn} use_residual:{args.ours_use_residual} use_act:{args.ours_use_act}{use_weight}\n'
        if not args.use_graph:
            return print_str
        if args.backbone == 'gcn':
            print_str += f'backbone:{args.backbone}, layers:{args.num_layers} hidden: {args.hidden_channels} lr:{args.lr} decay:{args.weight_decay} dropout:{args.dropout}\n'
    else:
        print_str += f'method: {args.method} hidden: {args.hidden_channels} lr:{args.lr}\n'
    return print_str


file_name = f'{args.dataset}_{args.method}'
if args.method == 'ours' and args.use_graph:
    file_name += '_' + args.backbone
file_name += '.txt'
out_path = os.path.join(out_folder, file_name)
with open(out_path, 'a+') as f:
    print_str = make_print(args.method)
    f.write(print_str)
    f.write(results)
    f.write(f' run_time: { run_time }')
    f.write('\n\n')

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
from data_utils import class_rand_splits, eval_acc, eval_rocauc, evaluate, load_fixed_splits, to_sparse_tensor
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
# Note: we use normalized logits MSE; temperature is not used in step2
parser.add_argument('--cons_temp', type=float, default=1.0,
                    help='temperature for probability-based consistency loss')
parser.add_argument(
    '--ema_tau',
    type=float,
    default=0.99,
    help='EMA momentum for teacher model in step2'
)
parser.add_argument('--ema_start', type=int, default=10,
                    help='number of epochs before EMA teacher starts updating')
# 新增一致性相关开关/选项
parser.add_argument('--cons_confidence', type=float, default=0.0,
                    help='min prob threshold for using teacher predictions in consistency (0.0 = no confidence filtering)')
parser.add_argument('--cons_loss', type=str, default='prob_mse',
                    choices=['logit_mse', 'prob_mse', 'kl', 'norm_mse'],
                    help='type of consistency loss to use: logits-MSE, probability-MSE (with temp), KL, or normalized-logits-MSE')



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
# --- ensure teacher on same device and frozen ---
teacher = copy.deepcopy(model)
teacher.to(device)
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
    teacher.load_state_dict(model.state_dict())
    teacher.eval()

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

        # hyperparams
        E_warm = getattr(args, "cons_warm", 10)
        E_alpha_up = getattr(args, "cons_up", 40)
        alpha_target = getattr(args, "cons_weight", 1.0)
        T = getattr(args, "cons_temp", 1.0)
        cons_type = getattr(args, "cons_loss", "prob_mse")
        conf_thresh = getattr(args, "cons_confidence", 0.0)

        # Student forward
        if args.method == 'nodeformer':
            out1, link_loss1 = model(dataset)
        else:
            out1 = model(dataset)

        if args.method == 'graphormer':
            out1 = out1.squeeze(0)

        # supervised loss (unchanged)
        if args.dataset in ('deezer-europe'):
            if dataset.label.shape[1] == 1:
                true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
            else:
                true_label = dataset.label
            sup_loss = criterion(out1[train_idx], true_label.squeeze(1)[train_idx].to(torch.float))
        else:
            out1_logp = F.log_softmax(out1, dim=1)
            sup_loss = criterion(out1_logp[train_idx], dataset.label.squeeze(1)[train_idx])

        # Teacher forward (no grad)
        teacher.eval()
        with torch.no_grad():
            if args.method == 'nodeformer':
                t_out, _ = teacher(dataset)
            else:
                t_out = teacher(dataset)
            if args.method == 'graphormer':
                t_out = t_out.squeeze(0)

        # Prepare mask: unlabeled nodes only (avoid disturbing labeled samples)
        n_nodes = n
        unlabeled_mask = torch.ones(n_nodes, dtype=torch.bool, device=device)
        # train_idx 已经 .to(device) 了，直接用索引屏蔽
        unlabeled_mask[train_idx] = False


        # optionally filter by teacher confidence
        if conf_thresh > 0.0:
            # use softmax probabilities of teacher
            p_t = F.softmax(t_out / T, dim=1)
            conf_mask = (p_t.max(dim=1).values > conf_thresh)
            use_mask = unlabeled_mask & conf_mask
        else:
            use_mask = unlabeled_mask

        # compute consistency loss according to selected type
        if use_mask.sum() == 0:
            cons_loss = torch.tensor(0.0, device=device)
        else:
            if cons_type == 'logit_mse':
                # direct mse on logits (can be large-scale sensitive)
                cons_loss = F.mse_loss(out1[use_mask], t_out[use_mask])
            elif cons_type == 'prob_mse':
                p1 = F.softmax(out1 / T, dim=1)
                p2 = F.softmax(t_out / T, dim=1)
                cons_loss = F.mse_loss(p1[use_mask], p2[use_mask])
            elif cons_type == 'kl':
                # KL(p_teacher || p_student) with temperature correction
                logp1 = F.log_softmax(out1 / T, dim=1)
                p2 = F.softmax(t_out / T, dim=1)
                cons_loss = F.kl_div(logp1[use_mask], p2[use_mask], reduction='batchmean') * (T * T)
            else:  # 'norm_mse' normalized logits mse (default fallback)
                out1_n = F.normalize(out1, p=2, dim=1)
                t_out_n = F.normalize(t_out, p=2, dim=1)
                cons_loss = F.mse_loss(out1_n[use_mask], t_out_n[use_mask])

        # alpha warmup (linear)
        if epoch <= E_warm:
            alpha = 0.0
        else:
            t = epoch - E_warm
            alpha = alpha_target * min(1.0, float(t) / float(E_alpha_up))

        # total loss
        loss = sup_loss + alpha * cons_loss

        # optional nodeformer link loss handling as before
        if args.method == 'nodeformer':
            # if two values, average them; original code subtracts lamda * mean(link_loss)
            try:
                link_loss_avg = [(link_loss1_i) for link_loss1_i in link_loss1]
                loss -= args.lamda * sum(link_loss_avg) / max(1, len(link_loss_avg))
            except Exception:
                pass

        # Backprop
        loss.backward()

        # compute grad norm and clip (clip_grad_norm_ returns total_norm)
        max_norm = 5.0
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        # EMA teacher update (delayed by ema_start)
        if epoch >= getattr(args, 'ema_start', 0):
            tau = getattr(args, 'ema_tau', 0.99)
            for s_param, t_param in zip(model.parameters(), teacher.parameters()):
                t_param.data.mul_(tau).add_(s_param.data * (1.0 - tau))
        # ====== END: S-channel (teacher-student consistency) ======

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
            mask_count = int(use_mask.sum().item()) if 'use_mask' in locals() else -1
            grad_norm_val = float(grad_norm) if 'grad_norm' in locals() else -1.0
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {100 * result[0]:.2f}%, Valid: {100 * result[1]:.2f}%, Test: {100 * result[2]:.2f}%, cons_nodes:{mask_count}, grad_norm:{grad_norm_val:.4f}')

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

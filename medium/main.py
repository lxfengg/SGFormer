#!/usr/bin/env python3
import argparse
import copy
import os
import random
import sys
import warnings
import time
import subprocess
import json

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


def get_gpu_memory_map():
    """Get the current gpu usage (MB) as numpy array."""
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            encoding='utf-8')
        gpu_memory = np.array([int(x) for x in result.strip().split('\n')])
        return gpu_memory
    except Exception:
        return np.array([])


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed(seed)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True


def resolve_data_dir(data_dir):
    """Try to robustly find the Planetoid/raw directory given a user-provided data_dir."""
    candidates = [data_dir,
                  os.path.join(os.getcwd(), data_dir) if data_dir else None,
                  os.path.abspath(data_dir) if data_dir else None,
                  os.path.join(os.getcwd(), '..', 'data'),
                  os.path.join(os.getcwd(), '..', '..', 'data'),
                  os.path.join(os.getcwd(), 'data'),
                  os.path.join('/mnt', 'd', 'File', 'code', 'SGFormer', 'data'),
                  os.path.join(os.path.dirname(__file__), '..', 'data') if '__file__' in globals() else None]
    checked = []
    for c in candidates:
        if not c:
            continue
        if os.path.exists(c):
            raw = os.path.join(c, 'Planetoid', 'raw')
            if os.path.exists(raw):
                print(f"[INFO] Using data_dir = {c} (found Planetoid/raw).")
                return c
        checked.append(c)
    print("[WARN] Could not find 'Planetoid/raw' under tried locations. Tried:", checked)
    print("[WARN] Will proceed with provided data_dir; if dataset loader fails, please set correct --data_dir")
    return data_dir


def random_edge_dropout(edge_index, drop_prob, device):
    """Simple random edge dropout on edge_index tensor [2, E]. Returns new edge_index on same device."""
    if edge_index is None:
        return edge_index
    E = edge_index.shape[1]
    if drop_prob <= 0.0 or E == 0:
        return edge_index
    mask = (torch.rand(E, device=device) > drop_prob)
    if mask.sum() == 0:
        # keep at least one edge
        mask[torch.randint(0, E, (1,), device=device)] = True
    return edge_index[:, mask]


### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser.add_argument('--cons_warm', type=int, default=10, help='supervised pre-warm epochs before consistency')
parser.add_argument('--cons_up', type=int, default=40, help='epochs to linearly warm up consistency weight')
parser.add_argument('--cons_weight', type=float, default=1.0, help='final consistency weight (alpha)')
parser.add_argument('--cons_temp', type=float, default=1.0, help='temperature for probability-based consistency loss')
parser.add_argument('--ema_tau', type=float, default=0.99, help='EMA momentum for teacher model in step2')
parser.add_argument('--ema_start', type=int, default=10, help='number of epochs before EMA teacher starts updating')
parser.add_argument('--cons_confidence', type=float, default=0.0, help='min prob threshold for using teacher predictions in consistency (0.0 = no confidence filtering)')
parser.add_argument('--cons_loss', type=str, default='prob_mse', choices=['logit_mse', 'prob_mse', 'kl', 'norm_mse'], help='type of consistency loss to use')

# NodeMixup (B)
parser.add_argument('--use_node_mixup', action='store_true', help='enable NodeMixup on training nodes (feature-level mixup)')
parser.add_argument('--mixup_alpha', type=float, default=0.3, help='beta distribution alpha for mixup')
parser.add_argument('--mixup_on', type=str, default='feat', choices=['feat'], help='where to apply mixup; default: feat (input features)')

# CI-GCL / auxiliary loss (E) (simple placeholder implementation)
parser.add_argument('--use_ci_gcl', action='store_true', help='enable CI-GCL style auxiliary loss (simple implementation)')
parser.add_argument('--aux_weight', type=float, default=0.05, help='weight for auxiliary loss (w_aux)')
parser.add_argument('--ci_drop_edge', type=float, default=0.2, help='edge drop probability for CI-GCL augmentation')
parser.add_argument('--ci_feat_mask', type=float, default=0.1, help='feature mask probability for CI-GCL augmentation')

# Stage control & checkpoint
parser.add_argument('--stage', type=str, default='both', choices=['pretrain', 'finetune', 'both'], help='training stage: pretrain (D only), finetune (B+D+E), or both')
parser.add_argument('--save_checkpoints', action='store_true', help='save best model checkpoints per run')
parser.add_argument('--load_pretrain', type=str, default=None, help='path to pretrained checkpoint to load when finetune')

# main args from parse.py helper
parser_add_main_args(parser)

# Add exp_tag and out_file for robust downstream processing
parser.add_argument('--exp_tag', type=str, default='unknown', help="experiment tag (e.g. baseline, schannel) for postprocessing")
parser.add_argument('--out_file', type=str, default=None, help='optional per-run JSON output file path')

args = parser.parse_args()
parser_add_default_args(args)
print(args)

# Fix seed and device (initial)
fix_seed(args.seed)
if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
print("[INFO] device:", device)

# Try to resolve data dir robustly
args.data_dir = resolve_data_dir(args.data_dir)

### Load and preprocess data ###
dataset = load_nc_dataset(args)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)

dataset_name = args.dataset

if args.rand_split:
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop) for _ in range(args.runs)]
elif args.rand_split_class:
    split_idx_lst = [class_rand_splits(dataset.label, args.label_num_per_class, args.valid_num, args.test_num)]
else:
    split_idx_lst = load_fixed_splits(dataset, name=args.dataset, protocol=args.protocol)

dataset.label = dataset.label.to(device)

# ----------------- DEBUG: label & mask check -----------------
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

try:
    if 'split_idx_lst' in locals() and len(split_idx_lst) > 0:
        sample_split = split_idx_lst[0]
        if 'train' in sample_split:
            tr = sample_split['train']
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
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

_shape = dataset.graph['node_feat'].shape
print(f'features shape={_shape}')

if args.dataset not in {'deezer-europe'}:
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])

dataset.graph['edge_index'], dataset.graph['node_feat'] = dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)

if args.method == 'graphormer':
    for k in ('x', 'in_degree', 'out_degree', 'spatial_pos', 'attn_bias'):
        if k in dataset.graph:
            dataset.graph[k] = dataset.graph[k].to(device)

print(f"num nodes {n} | num classes {c} | num node feats {d}")

### Load method ###
model = parse_method(args.method, args, c, d, device)

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
# safer optimizer creation: use params1/params2 if model provides them, otherwise fallback
if args.method == 'ours' and args.use_graph and hasattr(model, 'params1') and hasattr(model, 'params2'):
    print("[INFO] Using grouped params optimizer (params1 + params2).")
    optimizer = torch.optim.Adam([
        {'params': model.params1, 'weight_decay': args.ours_weight_decay},
        {'params': model.params2, 'weight_decay': args.weight_decay}
    ], lr=args.lr)
else:
    print("[INFO] Using fallback optimizer over model.parameters().")
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)

run_time_list = []
best_val = float('-inf')
best_val_test = 0.0

# ensure result folders exist
os.makedirs('results/epoch_logs', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('results/checkpoints', exist_ok=True)

# utility: robust append/update to CSV with explicit format
def write_result_csv(csv_path, exp_tag, seed, run_idx, best_val, best_test):
    header = 'exp_tag,seed,run,best_val,best_test\n'
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write(header)
    # read existing lines reliably
    try:
        with open(csv_path, 'r') as f:
            lines = f.read().splitlines()
    except Exception:
        lines = []
    new_line = f'{exp_tag},{seed},{run_idx},{best_val:.6f},{best_test:.6f}'
    key_prefix = f'{exp_tag},{seed},{run_idx},'
    found = False
    for i, ln in enumerate(lines):
        if ln.startswith(key_prefix):
            lines[i] = new_line
            found = True
            break
    if not found:
        lines.append(new_line)
    # Ensure header present
    if len(lines) == 0 or not lines[0].startswith('exp_tag,seed,run'):
        lines.insert(0, header.strip())
    with open(csv_path, 'w') as f:
        f.write('\n'.join(lines).strip() + '\n')


for run in range(args.runs):
    # choose split
    if args.dataset in ['cora', 'citeseer', 'pubmed'] and args.protocol == 'semi':
        split_idx = split_idx_lst[0]
    else:
        split_idx = split_idx_lst[run]

    # adjust seed per run to avoid repeated identical seeds when using runs>1
    current_seed = int(args.seed) + int(run)
    fix_seed(current_seed)

    # set train_idx and other split tensors to device
    train_idx = split_idx['train'].to(device)

    model.reset_parameters()
    teacher.load_state_dict(model.state_dict())
    teacher.eval()

    best_val = float('-inf')
    patience = 0

    # prepare per-run epoch log path
    safe_tag = str(getattr(args, 'exp_tag', 'unknown')).replace('/', '_').replace(' ', '_')
    epoch_csv = os.path.join('results/epoch_logs', f'{args.dataset}_{args.method}_{safe_tag}_seed{current_seed}_run{run}.csv')
    if not os.path.exists(epoch_csv):
        with open(epoch_csv, 'w') as f:
            f.write('epoch,train,val,test,alpha,cons_loss,aux_loss,mixup_lambda,mean_teacher_conf,cons_nodes,grad_norm\n')


    for epoch in range(args.epochs):
        start_time = time.perf_counter()
        # ====== START: S-channel (SimGRACE-style) + B + E integration ======
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

        # Student forward (may be replaced if mixup applied)
        # We will potentially temporary replace node_feat in dataset for mixup augmentation.
        orig_node_feat = dataset.graph['node_feat']
        orig_edge_index = dataset.graph['edge_index']

        mixup_lambda = 0.0
        if args.use_node_mixup and args.stage in ('both', 'finetune'):
            try:
                # apply mixup on training node features
                alpha = float(args.mixup_alpha)
                if alpha > 0.0 and train_idx.numel() > 1:
                    lam = np.random.beta(alpha, alpha)
                    perm = train_idx[torch.randperm(train_idx.numel(), device=device)]
                    node_feat_mixed = orig_node_feat.clone()
                    # lam * x_i + (1-lam) * x_j  on train_idx
                    node_feat_mixed[train_idx] = lam * orig_node_feat[train_idx] + (1.0 - lam) * orig_node_feat[perm]
                    dataset.graph['node_feat'] = node_feat_mixed
                    mixup_lambda = float(lam)
                else:
                    mixup_lambda = 1.0
            except Exception as e:
                print("[WARN] mixup failed, proceeding without mixup:", e)
                dataset.graph['node_feat'] = orig_node_feat
                mixup_lambda = 0.0

        # Optionally apply CI-GCL augmentation for aux computation (but keep orig for teacher)
        aux_loss = torch.tensor(0.0, device=device)
        if args.use_ci_gcl and args.stage in ('both', 'finetune'):
            try:
                # create augmented view by dropping edges and masking features
                aug_edge_index = random_edge_dropout(orig_edge_index, args.ci_drop_edge, device)
                aug_node_feat = orig_node_feat.clone()
                if args.ci_feat_mask > 0.0:
                    mask = (torch.rand_like(aug_node_feat) > args.ci_feat_mask).float()
                    aug_node_feat = aug_node_feat * mask
                # prepare temporary augmented dataset object
                dataset_aug = copy.deepcopy(dataset)
                dataset_aug.graph['edge_index'] = aug_edge_index
                dataset_aug.graph['node_feat'] = aug_node_feat
            except Exception as e:
                print("[WARN] ci_gcl augmentation failed:", e)
                dataset_aug = None
        else:
            dataset_aug = None

        # Student forward
        if args.method == 'nodeformer':
            out1, link_loss1 = model(dataset)
        else:
            out1 = model(dataset)

        if args.method == 'graphormer':
            out1 = out1.squeeze(0)

        # supervised loss (NodeMixup handled because dataset.graph['node_feat'] may have been mixed)
        if args.dataset in ('deezer-europe'):
            if dataset.label.shape[1] == 1:
                true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
            else:
                true_label = dataset.label
            sup_loss = criterion(out1[train_idx], true_label.squeeze(1)[train_idx].to(torch.float))
        else:
            out1_logp = F.log_softmax(out1, dim=1)
            sup_loss = criterion(out1_logp[train_idx], dataset.label.squeeze(1)[train_idx])

        # Teacher forward (no grad) - teacher uses ORIGINAL features (so ensure orig restored)
        teacher.eval()
        with torch.no_grad():
            # ensure teacher uses orig features (not mixed)
            dataset.graph['node_feat'] = orig_node_feat
            if args.method == 'nodeformer':
                t_out, _ = teacher(dataset)
            else:
                t_out = teacher(dataset)
            if args.method == 'graphormer':
                t_out = t_out.squeeze(0)

        # compute teacher probabilities for confidence diagnostics (always compute for logging)
        mean_teacher_conf = 0.0
        p_t = None
        try:
            p_t = F.softmax(t_out / T, dim=1)
            mean_teacher_conf = float(p_t.max(dim=1).values.mean().item())
        except Exception:
            mean_teacher_conf = 0.0

        # Prepare mask: unlabeled nodes only
        n_nodes = n
        unlabeled_mask = torch.ones(n_nodes, dtype=torch.bool, device=device)
        unlabeled_mask[train_idx] = False

        # optionally filter by teacher confidence
        if conf_thresh > 0.0 and p_t is not None:
            conf_mask = (p_t.max(dim=1).values > conf_thresh)
            use_mask = unlabeled_mask & conf_mask
        else:
            use_mask = unlabeled_mask

        # compute consistency loss according to selected type (on teacher t_out and student out1)
        if use_mask.sum() == 0:
            cons_loss = torch.tensor(0.0, device=device)
        else:
            if cons_type == 'logit_mse':
                cons_loss = F.mse_loss(out1[use_mask], t_out[use_mask])
            elif cons_type == 'prob_mse':
                p1 = F.softmax(out1 / T, dim=1)
                p2 = F.softmax(t_out / T, dim=1)
                cons_loss = F.mse_loss(p1[use_mask], p2[use_mask])
            elif cons_type == 'kl':
                logp1 = F.log_softmax(out1 / T, dim=1)
                p2 = F.softmax(t_out / T, dim=1)
                cons_loss = F.kl_div(logp1[use_mask], p2[use_mask], reduction='batchmean') * (T * T)
            else:  # 'norm_mse'
                out1_n = F.normalize(out1, p=2, dim=1)
                t_out_n = F.normalize(t_out, p=2, dim=1)
                cons_loss = F.mse_loss(out1_n[use_mask], t_out_n[use_mask])

        # compute aux_loss (simple CI-GCL placeholder): L2 between normalized student outputs on original and augmented view
        if dataset_aug is not None:
            try:
                # dataset_aug uses augmented features/edges
                if args.method == 'nodeformer':
                    out_aug, _ = model(dataset_aug)
                else:
                    out_aug = model(dataset_aug)
                if args.method == 'graphormer':
                    out_aug = out_aug.squeeze(0)
                out1_n = F.normalize(out1, p=2, dim=1)
                out_aug_n = F.normalize(out_aug, p=2, dim=1)
                aux_loss = F.mse_loss(out1_n, out_aug_n)
            except Exception as e:
                print("[WARN] aux_loss computation failed:", e)
                aux_loss = torch.tensor(0.0, device=device)
        else:
            aux_loss = torch.tensor(0.0, device=device)

        # alpha warmup (linear). robust to E_alpha_up == 0
        # NOTE: use epoch < E_warm so that warm-up begins at epoch == E_warm
        if epoch < E_warm:
            alpha = 0.0
        else:
            t = epoch - E_warm
            if E_alpha_up <= 0:
                alpha = alpha_target
            else:
                alpha = alpha_target * min(1.0, float(t) / float(E_alpha_up))

        # total loss with aux
        loss = sup_loss + alpha * cons_loss + args.aux_weight * aux_loss

        # optional nodeformer link loss handling
        if args.method == 'nodeformer':
            try:
                link_loss_avg = [(link_loss1_i) for link_loss1_i in link_loss1]
                loss -= args.lamda * sum(link_loss_avg) / max(1, len(link_loss_avg))
            except Exception:
                pass

        # Backprop
        loss.backward()

        # compute grad norm and clip
        max_norm = 5.0
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm))

        optimizer.step()

        # EMA teacher update (delayed by ema_start)
        if epoch >= getattr(args, 'ema_start', 0):
            tau = getattr(args, 'ema_tau', 0.99)
            for s_param, t_param in zip(model.parameters(), teacher.parameters()):
                t_param.data.mul_(tau).add_(s_param.data * (1.0 - tau))

        # restore original dataset features/edges if we mutated them
        dataset.graph['node_feat'] = orig_node_feat
        dataset.graph['edge_index'] = orig_edge_index

        # ====== END S-channel + B + E block ======
        end_time = time.perf_counter()
        run_time = 1000 * (end_time - start_time)
        run_time_list.append(run_time)

        result = evaluate(model, dataset, split_idx, eval_func, criterion, args)
        logger.add_result(run, result[:-1])

        # update best/early stopping
        if result[1] > best_val:
            best_val = result[1]
            best_val_test = result[2]
            patience = 0
            # save checkpoint if desired
            if args.save_checkpoints:
                try:
                    ckpt_path = os.path.join('results', 'checkpoints', f'{safe_tag}_seed{current_seed}_run{run}_best.pth')
                    torch.save({'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'epoch': epoch,
                                'best_val': best_val,
                                'best_test': best_val_test}, ckpt_path)
                except Exception as e:
                    print("[WARN] failed to save checkpoint:", e)
        else:
            patience += 1
            if patience >= args.patience:
                break

        # write epoch CSV log
        try:
            curr_cons_nodes = int(use_mask.sum().item()) if 'use_mask' in locals() else -1
        except Exception:
            curr_cons_nodes = -1
        try:
            with open(epoch_csv, 'a') as f:
                f.write(f"{epoch},{result[0]:.6f},{result[1]:.6f},{result[2]:.6f},{alpha:.6f},{float(cons_loss):.6f},{float(aux_loss):.6f},{mixup_lambda:.6f},{mean_teacher_conf:.6f},{curr_cons_nodes},{grad_norm:.6f}\n")
        except Exception as e:
            print("[WARN] failed to write epoch log:", e)

        if epoch % args.display_step == 0:
            mask_count = curr_cons_nodes
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {100 * result[0]:.2f}%, Valid: {100 * result[1]:.2f}%, Test: {100 * result[2]:.2f}%, cons_nodes:{mask_count}, curr_cons_weight:{alpha:.4f}, cons_loss:{float(cons_loss):.6f}, aux_loss:{float(aux_loss):.6f}, mean_teacher_conf:{mean_teacher_conf:.4f}, grad_norm:{grad_norm:.4f}')

    logger.print_statistics(run)
    # append per-run summary to csv
    csv_path = os.path.join('results', f'{args.dataset}_{args.method}_results_per_run.csv')
    write_result_csv(csv_path, getattr(args, 'exp_tag', 'unknown'), current_seed, run, best_val, best_val_test)

    # optional per-run JSON file (useful for robust run collection)
    if args.out_file:
        try:
            os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
        except Exception:
            pass
        try:
            info = {
                'exp_tag': getattr(args, 'exp_tag', 'unknown'),
                'seed': current_seed,
                'run': run,
                'best_val': float(best_val),
                'best_test': float(best_val_test)
            }
            with open(args.out_file, 'w') as jf:
                json.dump(info, jf)
        except Exception as e:
            print('[WARN] failed to write out_file:', e)

# final stats
run_time = sum(run_time_list) / len(run_time_list) if len(run_time_list) > 0 else 0.0
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
        use_weight = ' ours_use_weight' if args.ours_use_weight else ''
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

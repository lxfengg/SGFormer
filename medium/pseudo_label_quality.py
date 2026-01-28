# 保存为 pseudo_label_quality.py
import torch, os, argparse
from dataset import load_nc_dataset
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', required=True)
parser.add_argument('--dataset', default='cora')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class A: pass
a = A(); a.dataset = args.dataset; a.data_dir = None
dataset = load_nc_dataset(a)
# try get splits
try:
    split_idx = dataset.get_idx_split(train_prop=0.05, valid_prop=0.185)
except:
    from data_utils import load_fixed_splits
    split_idx = load_fixed_splits(dataset, name=args.dataset, protocol='semi')[0]

# try to load ckpt
ck = torch.load(args.ckpt, map_location=device)
print("LOADED CKPT KEYS:", list(ck.keys()))
# We expect 'teacher_state' or 'model_state' in ck
state = None
if 'teacher_state' in ck:
    state = ck['teacher_state']
    which = 'teacher'
elif 'model_state' in ck:
    state = ck['model_state']
    which = 'student'
else:
    # if ck looks like a raw state_dict
    if isinstance(ck, dict) and any(k.startswith('module.') or k in ck for k in ck.keys()):
        # heuristic
        state = ck
        which = 'state_dict'
    else:
        raise RuntimeError("Unknown checkpoint format. Please inspect the ckpt content.")

# instantiate model via parse_method (best effort)
from parse import parse_method
from parse import parser_add_default_args, parser_add_main_args
import argparse as _arg
p = _arg.ArgumentParser()
parser_add_main_args(p)
_dummy_args = p.parse_known_args([])[0]
parser_add_default_args(_dummy_args)

# create model - use defaults
c = int(max(dataset.label.max().item() + 1, dataset.label.shape[1]))
d = int(dataset.graph['node_feat'].shape[1])
try:
    model = parse_method(_dummy_args.method if hasattr(_dummy_args,'method') else 'ours', _dummy_args, c, d, device)
except Exception as e:
    print("Failed to instantiate model via parse_method, please create model manually:", e)
    raise

# load state
try:
    model.load_state_dict(state, strict=False)
except Exception as e:
    print("state load error (non-strict):", e)
    try:
        # try nested dict keys
        if 'model' in ck:
            model.load_state_dict(ck['model'])
            which = 'model'
        else:
            raise
    except Exception as e2:
        print("Failed to load model state automatically. Please adapt script.")
        raise

model.to(device)
model.eval()
with torch.no_grad():
    out = model(dataset)
    if isinstance(out, tuple):
        out = out[0]
    probs = torch.softmax(out, dim=1).cpu().numpy()
    preds = probs.argmax(axis=1)

labels = dataset.label.squeeze(1).cpu().numpy()
train_idx = split_idx['train'].cpu().numpy().tolist()
n = dataset.graph['num_nodes']
all_idx = set(range(n))
unlabeled_idx = sorted(list(all_idx - set(train_idx)))

# overall unlabeled acc
acc_un = (preds[unlabeled_idx] == labels[unlabeled_idx]).mean()
print(f"Overall unlabeled acc (proxy): {acc_un:.4f}")
maxp = probs.max(axis=1)
for thr in [0.5,0.6,0.7,0.8]:
    idx_conf = [i for i in unlabeled_idx if maxp[i] > thr]
    if len(idx_conf)>0:
        acc_conf = (preds[idx_conf] == labels[idx_conf]).mean()
    else:
        acc_conf = None
    print(f"thr {thr}: coverage {len(idx_conf)}/{len(unlabeled_idx)} acc_on_conf={acc_conf}")

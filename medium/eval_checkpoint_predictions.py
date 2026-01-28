# 保存为 eval_checkpoint_predictions.py
import torch, os, argparse, numpy as np
from dataset import load_nc_dataset
from parse import parse_method
from data_utils import load_fixed_splits

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', required=True)
parser.add_argument('--dataset', default='cora')
parser.add_argument('--device', type=str, default=None, help='cuda:0 or cpu')
args = parser.parse_args()

# device
if args.device:
    device = torch.device(args.device)
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("dataset:", args.dataset)
# load dataset
class A: pass
a = A(); a.dataset = args.dataset; a.data_dir = None
dataset = load_nc_dataset(a)
# get splits
try:
    split_idx = dataset.get_idx_split(train_prop=0.05, valid_prop=0.185)
except Exception:
    split_idx = load_fixed_splits(dataset, name=args.dataset, protocol='semi')[0]

n = dataset.graph['num_nodes']
labels = dataset.label.squeeze(1).cpu().numpy()

# load ckpt
ck = torch.load(args.ckpt, map_location=device)
print("LOADED CKPT KEYS:", list(ck.keys()))

# build args object for parse_method from ckpt['args'] if present
ck_args = ck.get('args', {}) if isinstance(ck.get('args', {}), dict) else {}
class DArgs: pass
dargs = DArgs()

# Helper to set attribute with fallback
def set_attr(name, default=None):
    if name in ck_args:
        val = ck_args[name]
    else:
        val = default
    setattr(dargs, name, val)

# Common fields used by parse_method in many repos - set sensible defaults
set_attr('method', ck_args.get('method', 'ours'))
set_attr('backbone', ck_args.get('backbone', 'gcn'))
set_attr('num_layers', ck_args.get('num_layers', 2))
set_attr('hidden_channels', ck_args.get('hidden_channels', ck_args.get('hidden', 64)))
set_attr('use_graph', ck_args.get('use_graph', True))
set_attr('ours_layers', ck_args.get('ours_layers', ck_args.get('layers', 2)))
set_attr('ours_weight_decay', ck_args.get('ours_weight_decay', ck_args.get('weight_decay', 5e-4)))
set_attr('weight_decay', ck_args.get('weight_decay', 5e-4))
set_attr('dropout', ck_args.get('dropout', 0.5))
# Add any other commonly referenced flags with defaults
set_attr('hidden', getattr(dargs, 'hidden_channels', 64))
set_attr('device', 0)

# Now create model
c = int(max(dataset.label.max().item() + 1, dataset.label.shape[1]))
d = int(dataset.graph['node_feat'].shape[1])

try:
    model = parse_method(dargs.method, dargs, c, d, device)
except Exception as e:
    print("Failed to create model via parse_method with dargs; error:", e)
    print("Attempting to create model with minimal defaults...")
    # minimal fallback args object
    class FallbackArgs: pass
    fargs = FallbackArgs()
    fargs.method = getattr(dargs, 'method', 'ours')
    fargs.backbone = getattr(dargs, 'backbone', 'gcn')
    fargs.num_layers = getattr(dargs, 'num_layers', 2)
    fargs.hidden_channels = getattr(dargs, 'hidden_channels', 64)
    fargs.use_graph = getattr(dargs, 'use_graph', True)
    try:
        model = parse_method(fargs.method, fargs, c, d, device)
    except Exception as e2:
        print("Fallback model creation failed:", e2)
        raise RuntimeError("Could not instantiate model. Please inspect parse_method signature or provide a custom model loader.")

# load model state
state_loaded = False
if 'model_state' in ck:
    try:
        model.load_state_dict(ck['model_state'], strict=False)
        state_loaded = True
    except Exception as e:
        print("Warning: model_state load failed (non-strict) ->", e)
elif 'state_dict' in ck:
    try:
        model.load_state_dict(ck['state_dict'], strict=False)
        state_loaded = True
    except Exception as e:
        print("Warning: state_dict load failed ->", e)
else:
    # try raw dict
    if isinstance(ck, dict):
        try:
            model.load_state_dict(ck, strict=False)
            state_loaded = True
        except Exception as e:
            print("Warning: direct ck load failed ->", e)

if not state_loaded:
    print("Warning: model state not loaded cleanly. inspect checkpoint keys and shapes.")

# teacher if present
teacher_present = False
teacher = None
if 'teacher_state' in ck:
    teacher_present = True
    try:
        teacher = parse_method(dargs.method, dargs, c, d, device)
        teacher.load_state_dict(ck['teacher_state'], strict=False)
        teacher.to(device)
        teacher.eval()
    except Exception as e:
        print("Warning: teacher instantiation/load failed ->", e)
        teacher_present = False
        teacher = None

model.to(device)
model.eval()

# forward
with torch.no_grad():
    out = model(dataset)
    if isinstance(out, tuple):
        out = out[0]
    probs = torch.softmax(out, dim=1).cpu().numpy()
    preds = probs.argmax(axis=1)
    confs = probs.max(axis=1)

if teacher_present:
    with torch.no_grad():
        tout = teacher(dataset)
        if isinstance(tout, tuple):
            tout = tout[0]
        tprobs = torch.softmax(tout, dim=1).cpu().numpy()
        tpreds = tprobs.argmax(axis=1)
        tconfs = tprobs.max(axis=1)
else:
    tpreds = None; tconfs = None

# splits
train_idx = split_idx['train'].cpu().numpy().tolist()
val_idx = split_idx['valid'].cpu().numpy().tolist()
test_idx = split_idx['test'].cpu().numpy().tolist()
all_idx = set(range(n))
unlabeled_idx = sorted(list(all_idx - set(train_idx)))

# helper
def acc_on(idx_list, arr):
    if len(idx_list) == 0:
        return None
    return float((arr[idx_list] == labels[idx_list]).mean())

# print summary
print("=== Summary ===")
print("Num nodes:", n)
print("Train/Val/Test sizes:", len(train_idx), len(val_idx), len(test_idx))
print("Student acc - train/val/test: {:.4f} / {:.4f} / {:.4f}".format(
    acc_on(train_idx, preds) if acc_on(train_idx, preds) is not None else float('nan'),
    acc_on(val_idx, preds) if acc_on(val_idx, preds) is not None else float('nan'),
    acc_on(test_idx, preds) if acc_on(test_idx, preds) is not None else float('nan')
))
if teacher_present:
    print("Teacher acc - train/val/test: {:.4f} / {:.4f} / {:.4f}".format(
        acc_on(train_idx, tpreds) if acc_on(train_idx, tpreds) is not None else float('nan'),
        acc_on(val_idx, tpreds) if acc_on(val_idx, tpreds) is not None else float('nan'),
        acc_on(test_idx, tpreds) if acc_on(test_idx, tpreds) is not None else float('nan')
    ))

print("Student overall unlabeled acc: {:.4f}".format(acc_on(unlabeled_idx, preds)))
if teacher_present:
    print("Teacher overall unlabeled acc: {:.4f}".format(acc_on(unlabeled_idx, tpreds)))

# confidence summary (student)
unl_conf = confs[unlabeled_idx] if len(unlabeled_idx)>0 else np.array([])
if unl_conf.size>0:
    print("Student confidences on unlabeled: mean {:.4f} median {:.4f} 90p {:.4f}".format(
        float(unl_conf.mean()), float(np.median(unl_conf)), float(np.percentile(unl_conf,90))
    ))
    for thr in [0.5,0.6,0.7,0.8]:
        cnt = int((unl_conf > thr).sum())
        print(f"Student conf > {thr}: {cnt}/{len(unlabeled_idx)}")
else:
    print("No unlabeled nodes to evaluate/confidence stats.")

if teacher_present:
    t_unl_conf = tconfs[unlabeled_idx] if len(unlabeled_idx)>0 else np.array([])
    if t_unl_conf.size>0:
        print("Teacher confidences on unlabeled: mean {:.4f} median {:.4f} 90p {:.4f}".format(
            float(t_unl_conf.mean()), float(np.median(t_unl_conf)), float(np.percentile(t_unl_conf,90))
        ))
        for thr in [0.5,0.6,0.7,0.8]:
            cnt = int((t_unl_conf > thr).sum())
            print(f"Teacher conf > {thr}: {cnt}/{len(unlabeled_idx)}")
    else:
        print("Teacher present but no unlabeled confs computed.")

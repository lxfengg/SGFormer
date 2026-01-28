# 保存为 evaluate_checkpoints.py
"""
Scan results/checkpoints/**/*.pth, load each checkpoint, build model from ckpt['args'] (best-effort),
compute student & teacher predictions and metrics (train/val/test accuracy, unlabeled acc, conf counts),
write summary CSV and per-checkpoint JSON files.
"""
import os, glob, json, argparse, traceback
import numpy as np
import torch
from dataset import load_nc_dataset
from data_utils import load_fixed_splits
from parse import parse_method
from collections import defaultdict

OUT_CSV = "results/diagnosis_all_checkpoints_python.csv"
OUT_DIR_JSON = "results/diagnosis_per_ckpt"

def safe_set_attrs_from_dict(obj, d):
    for k, v in d.items():
        try:
            setattr(obj, k, v)
        except Exception:
            pass

def build_model_from_ckargs(ck_args, dataset, device):
    # create args-like object and set many reasonable defaults
    class DArgs: pass
    dargs = DArgs()
    safe_set_attrs_from_dict(dargs, ck_args if isinstance(ck_args, dict) else {})
    # sensible defaults for commonly used params in parse_method
    defaults = {
        'method': getattr(dargs, 'method', 'ours'),
        'backbone': getattr(dargs, 'backbone', 'gcn'),
        'num_layers': getattr(dargs, 'num_layers', 2),
        'hidden_channels': getattr(dargs, 'hidden_channels', getattr(dargs, 'hidden', 64)),
        'use_graph': getattr(dargs, 'use_graph', True),
        'ours_layers': getattr(dargs, 'ours_layers', getattr(dargs, 'layers', 2)),
    }
    for k,v in defaults.items():
        if not hasattr(dargs, k):
            setattr(dargs, k, v)
    # call parse_method
    c = int(max(dataset.label.max().item() + 1, dataset.label.shape[1]))
    d = int(dataset.graph['node_feat'].shape[1])
    model = parse_method(dargs.method, dargs, c, d, device)
    return model, dargs

def metrics_from_preds(preds, probs, labels, idx_list):
    if len(idx_list)==0:
        return None, {}
    arr = np.array(idx_list, dtype=int)
    acc = float((preds[arr] == labels[arr]).mean())
    return acc, {}

def per_conf_counts(probs, idxs, thresholds=(0.5,0.6,0.7,0.8)):
    maxp = probs.max(axis=1)
    res = {}
    for t in thresholds:
        res[f"conf_gt_{int(t*100)}"] = int((maxp[idxs] > t).sum())
    return res

def evaluate_ckpt(ckpt_path, dataset_name='cora', device=None):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f"[INFO] Loading dataset {dataset_name}")
    class A: pass
    a = A(); a.dataset = dataset_name; a.data_dir = None
    dataset = load_nc_dataset(a)
    # splits
    try:
        split_idx = dataset.get_idx_split(train_prop=0.05, valid_prop=0.185)
    except Exception:
        split_idx = load_fixed_splits(dataset, name=dataset_name, protocol='semi')[0]
    n = dataset.graph['num_nodes']
    labels = dataset.label.squeeze(1).cpu().numpy()

    # load ckpt
    ck = torch.load(ckpt_path, map_location=device)
    ck_args = ck.get('args', {}) if isinstance(ck.get('args', {}), dict) else {}
    basename = os.path.basename(ckpt_path).replace('.pth','')
    out = {'ckpt': ckpt_path, 'basename': basename, 'ck_args_present': isinstance(ck.get('args', {}), dict)}
    out.update({'student_train': None, 'student_val': None, 'student_test': None,
                'teacher_train': None, 'teacher_val': None, 'teacher_test': None,
                'student_unlabeled_acc': None, 'teacher_unlabeled_acc': None})
    try:
        model, dargs = build_model_from_ckargs(ck_args, dataset, device)
    except Exception as e:
        out['error'] = f"model_build_failed: {e}\n{traceback.format_exc()}"
        return out
    # load state
    try:
        if 'model_state' in ck:
            model.load_state_dict(ck['model_state'], strict=False)
        elif 'state_dict' in ck:
            model.load_state_dict(ck['state_dict'], strict=False)
        else:
            # try direct
            try:
                model.load_state_dict(ck, strict=False)
            except Exception:
                pass
    except Exception as e:
        out['error'] = f"model_load_failed: {e}\n{traceback.format_exc()}"
        return out

    model.to(device); model.eval()
    # teacher
    teacher = None
    if 'teacher_state' in ck:
        try:
            teacher, _ = build_model_from_ckargs(ck_args, dataset, device)
            teacher.load_state_dict(ck['teacher_state'], strict=False)
            teacher.to(device); teacher.eval()
            out['teacher_present'] = True
        except Exception as e:
            out['teacher_present'] = False
            out['teacher_error'] = f"{e}"
    else:
        out['teacher_present'] = False

    with torch.no_grad():
        stud_out = model(dataset)
        if isinstance(stud_out, tuple):
            stud_out = stud_out[0]
        stud_probs = torch.softmax(stud_out, dim=1).cpu().numpy()
        stud_preds = stud_probs.argmax(axis=1)

        if teacher is not None:
            t_out = teacher(dataset)
            if isinstance(t_out, tuple):
                t_out = t_out[0]
            t_probs = torch.softmax(t_out, dim=1).cpu().numpy()
            t_preds = t_probs.argmax(axis=1)
        else:
            t_probs = None; t_preds = None

    # indices
    train_idx = split_idx['train'].cpu().numpy().tolist()
    val_idx = split_idx['valid'].cpu().numpy().tolist()
    test_idx = split_idx['test'].cpu().numpy().tolist()
    all_idx = set(range(n))
    unlabeled_idx = sorted(list(all_idx - set(train_idx)))

    # compute accuracies
    def acc(idx_list, arr):
        if len(idx_list)==0: return None
        return float((arr[idx_list] == labels[idx_list]).mean())

    out['student_train'] = acc(train_idx, stud_preds)
    out['student_val'] = acc(val_idx, stud_preds)
    out['student_test'] = acc(test_idx, stud_preds)
    out['student_unlabeled_acc'] = acc(unlabeled_idx, stud_preds)
    if t_preds is not None:
        out['teacher_train'] = acc(train_idx, t_preds)
        out['teacher_val'] = acc(val_idx, t_preds)
        out['teacher_test'] = acc(test_idx, t_preds)
        out['teacher_unlabeled_acc'] = acc(unlabeled_idx, t_preds)
    # conf counts
    if stud_probs is not None:
        conf_counts = per_conf_counts(stud_probs, unlabeled_idx)
        out.update(conf_counts)
    # per-class unlabeled accuracy and confusion (small)
    num_classes = int(max(labels)+1)
    per_class = {}
    conf_mat = [[0]*num_classes for _ in range(num_classes)]
    for i in unlabeled_idx:
        p = int(stud_preds[i])
        t = int(labels[i])
        conf_mat[p][t] += 1
    per_class['student_unlabeled_per_class_acc'] = {}
    for cls in range(num_classes):
        ids = [i for i in unlabeled_idx if labels[i]==cls]
        per_class['student_unlabeled_per_class_acc'][str(cls)] = (float((stud_preds[ids]==labels[ids]).mean()) if len(ids)>0 else None)
    out['per_class'] = per_class
    out['confusion_matrix_unlabeled'] = conf_mat

    # save detailed json per ckpt
    os.makedirs(OUT_DIR_JSON, exist_ok=True)
    json_path = os.path.join(OUT_DIR_JSON, f"diagnosis_{basename}.json")
    with open(json_path, 'w') as jf:
        json.dump(out, jf, indent=2, default=lambda x: None)
    out['json'] = json_path
    return out

def main(dataset_name='cora'):
    ckpts = sorted(glob.glob("results/checkpoints/**/*.pth", recursive=True))
    if not ckpts:
        print("[WARN] No .pth found under results/checkpoints")
        return
    rows = []
    for ck in ckpts:
        print("[INFO] Evaluating", ck)
        try:
            r = evaluate_ckpt(ck, dataset_name)
        except Exception as e:
            r = {'ckpt': ck, 'error': f"uncaught:{e}\n{traceback.format_exc()}"}
        rows.append(r)
    # write CSV header + rows
    header = ["ckpt_path","basename","seed","exp_tag","student_train","student_val","student_test",
              "teacher_train","teacher_val","teacher_test","student_unlabeled_acc","teacher_unlabeled_acc",
              "stu_conf_gt_50","stu_conf_gt_60","stu_conf_gt_70","stu_conf_gt_80","json"]
    os.makedirs("results", exist_ok=True)
    with open(OUT_CSV, 'w') as fo:
        fo.write(','.join(header) + '\n')
        for r in rows:
            basename = r.get('basename','')
            seed = ''
            # try extract seed from basename
            import re
            m = re.search(r"_seed([0-9]+)", basename)
            if m:
                seed = m.group(1)
            exp_tag = ''
            m2 = re.search(r"^[^_]*_([^_]*)_seed", basename)
            if m2:
                exp_tag = m2.group(1)
            line = [
                r.get('ckpt',''),
                basename,
                seed,
                exp_tag,
                str(r.get('student_train','')),
                str(r.get('student_val','')),
                str(r.get('student_test','')),
                str(r.get('teacher_train','')),
                str(r.get('teacher_val','')),
                str(r.get('teacher_test','')),
                str(r.get('student_unlabeled_acc','')),
                str(r.get('teacher_unlabeled_acc','')),
                str(r.get('conf_gt_50', r.get('conf_gt_50', r.get('conf_gt_50','')) if False else r.get('conf_gt_50',''))),
                str(r.get('conf_gt_60','')),
                str(r.get('conf_gt_70','')),
                str(r.get('conf_gt_80','')),
                r.get('json','')
            ]
            fo.write(','.join(line) + '\n')
    print("[DONE] Wrote", OUT_CSV)

if __name__ == "__main__":
    import sys
    ds = sys.argv[1] if len(sys.argv)>1 else 'cora'
    main(ds)

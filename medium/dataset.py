# dataset.py
# Robust Planetoid loader compatible with cora/citeseer/pubmed
# Provides dataset.graph (edge_index, node_feat, num_nodes), dataset.label
# and also dataset.train_idx / dataset.valid_idx / dataset.test_idx for compatibility.

import os
import pickle
import numpy as np
import scipy.sparse as sp
import torch
from types import SimpleNamespace

def _read_pickle(path):
    with open(path, 'rb') as f:
        try:
            obj = pickle.load(f, encoding='latin1')
        except TypeError:
            obj = pickle.load(f)
    return obj

def _load_index_file(path):
    idx = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            idx.append(int(line))
    return np.array(idx, dtype=np.int64)

def sparse_to_torch_matrix(sparse_mx):
    # Convert scipy sparse matrix to dense torch.FloatTensor
    if sp.isspmatrix(sparse_mx):
        arr = sparse_mx.tocoo()
        dense = np.zeros(arr.shape, dtype=np.float32)
        if arr.data.size > 0:
            dense[arr.row, arr.col] = arr.data.astype(np.float32)
        return torch.from_numpy(dense)
    else:
        return torch.from_numpy(np.array(sparse_mx, dtype=np.float32))

def _graph_to_edge_index(graph_dict):
    rows = []
    cols = []
    for src, nbrs in graph_dict.items():
        for dst in nbrs:
            rows.append(src)
            cols.append(dst)
    if len(rows) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([rows, cols], dtype=torch.long)

def load_planetoid_dataset(name, no_feat_norm=False, data_dir=None):
    """
    Robust loader for ind.<name>.* Planetoid raw files.
    Returns a SimpleNamespace with attributes:
      - graph: {'edge_index': LongTensor(2,E), 'node_feat': FloatTensor(N,D), 'num_nodes': int}
      - label: LongTensor(N,1)  (category index per node) OR (N, C) for true multilabel datasets
      - train_idx, valid_idx, test_idx: LongTensor indices (when inferrable)
    """
    if data_dir is None:
        base = os.path.join('..', 'data')
    else:
        base = data_dir

    raw_dir = os.path.join(base, 'Planetoid', 'raw')
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Planetoid raw directory not found: {raw_dir}")

    pref = f"ind.{name}"
    def _p(s): return os.path.join(raw_dir, f"{pref}.{s}")

    required = ['x', 'tx', 'allx', 'y', 'ty', 'graph', 'test.index']
    for r in required:
        path = _p(r if r != 'test.index' else 'test.index')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file missing: {path}")

    x = _read_pickle(_p('x'))
    tx = _read_pickle(_p('tx'))
    allx = _read_pickle(_p('allx'))
    y = _read_pickle(_p('y'))
    ty = _read_pickle(_p('ty'))
    graph = _read_pickle(_p('graph'))
    test_idx = _load_index_file(_p('test.index'))  # keep original order

    # ensure lil matrices for assignment convenience
    try:
        allx_lil = allx.tolil()
    except Exception:
        allx_lil = sp.lil_matrix(allx)
    try:
        tx_lil = tx.tolil()
    except Exception:
        tx_lil = sp.lil_matrix(tx)

    # determine final number of nodes
    max_test_idx = int(test_idx.max()) if test_idx.size > 0 else -1
    n_rows_needed = max(max_test_idx + 1, allx_lil.shape[0] + tx_lil.shape[0])

    feat_dim = allx_lil.shape[1]
    features = sp.lil_matrix((n_rows_needed, feat_dim), dtype=np.float32)

    # put allx at front
    features[0:allx_lil.shape[0], :] = allx_lil

    # map each row of tx to the corresponding test_idx entry (preserve original test.index order)
    if tx_lil.shape[0] != test_idx.shape[0]:
        print("[WARN] tx rows and test.index length mismatch: tx_rows=", tx_lil.shape[0], "test_idx_len=", test_idx.shape[0])
    n_assign = min(tx_lil.shape[0], test_idx.shape[0])
    for i in range(n_assign):
        tidx = int(test_idx[i])
        if tidx >= features.shape[0]:
            # extend if needed
            extra_rows = tidx + 1 - features.shape[0]
            features = sp.vstack([features, sp.lil_matrix((extra_rows, feat_dim), dtype=np.float32)])
        features[tidx, :] = tx_lil[i, :]

    # build labels
    y_arr = np.array(y)
    ty_arr = np.array(ty)
    n_classes = y_arr.shape[1] if y_arr.ndim > 1 else (ty_arr.shape[1] if ty_arr.ndim > 1 else 1)
    labels = np.zeros((features.shape[0], n_classes), dtype=np.int64)
    labels[0:y_arr.shape[0], :] = y_arr
    n_assign_lbl = min(ty_arr.shape[0], test_idx.shape[0])
    for i in range(n_assign_lbl):
        tidx = int(test_idx[i])
        if tidx >= labels.shape[0]:
            # should not happen often, but guard
            extra_rows = tidx + 1 - labels.shape[0]
            labels = np.vstack([labels, np.zeros((extra_rows, n_classes), dtype=np.int64)])
        labels[tidx, :] = ty_arr[i, :]

    node_feat = sparse_to_torch_matrix(features)  # FloatTensor [N, D]
    labels_t = torch.from_numpy(labels).long()    # LongTensor [N, C] (currently possibly one-hot)

    # --- Convert one-hot labels to class index (Planetoid standard) ---
    # Many Planetoid files store labels as one-hot vectors (N, C). Training code expects
    # a per-node integer class index (N,1) that can be squeezed to 1D for NLLLoss.
    # If dataset truly is multi-label (rare for Planetoid), user may want to keep the (N,C) form.
    if labels_t.dim() == 2 and labels_t.shape[1] > 1:
        # check if rows look like one-hot (each row sum ==1 or 0)
        row_sums = labels_t.sum(dim=1)
        if torch.all((row_sums == 1) | (row_sums == 0)):
            # treat as one-hot -> convert via argmax
            cls = torch.argmax(labels_t, dim=1, keepdim=True)
            labels_t = cls  # shape (N,1)
            print(f"[INFO] Converted one-hot labels -> integer class indices, new label shape: {labels_t.shape}")
        else:
            # ambiguous case (not strictly one-hot). We'll still argmax but warn.
            cls = torch.argmax(labels_t, dim=1, keepdim=True)
            labels_t = cls
            print("[WARN] Labels appear multi-column and not strictly one-hot. Argmax used to produce single class per node.")

    elif labels_t.dim() == 1:
        labels_t = labels_t.unsqueeze(1)

    num_nodes = node_feat.shape[0]
    edge_index = _graph_to_edge_index(graph)

    ds = SimpleNamespace()
    ds.graph = {'edge_index': edge_index, 'node_feat': node_feat, 'num_nodes': num_nodes}
    ds.label = labels_t

    # Provide commonly used fixed Planetoid splits so downstream code (load_fixed_splits) works.
    # Standard Planetoid split: train: 0..139 (140), val: 140..639 (500), test: test.index
    # Clip ranges if dataset smaller.
    try:
        train_end = min(140, num_nodes)
        valid_end = min(640, num_nodes)
        ds.train_idx = torch.arange(0, train_end, dtype=torch.long)
        ds.valid_idx = torch.arange(train_end, valid_end, dtype=torch.long)
        # test_idx may be larger than num_nodes in malformed files; clip
        test_idx_clipped = test_idx[test_idx < num_nodes]
        ds.test_idx = torch.from_numpy(test_idx_clipped).long()
    except Exception:
        # fallback: provide simple split by proportions
        n = num_nodes
        t = int(0.1 * n) if n >= 10 else max(1, n // 10)
        v = int(0.2 * n)
        ds.train_idx = torch.arange(0, t, dtype=torch.long)
        ds.valid_idx = torch.arange(t, t + v, dtype=torch.long)
        ds.test_idx = torch.arange(t + v, n, dtype=torch.long)

    # Also give convenience function for random splits
    def get_idx_split(train_prop=0.5, valid_prop=0.25):
        n = num_nodes
        all_idx = np.arange(n)
        np.random.shuffle(all_idx)
        n_train = int(train_prop * n)
        n_valid = int(valid_prop * n)
        train = torch.tensor(all_idx[:n_train], dtype=torch.long)
        valid = torch.tensor(all_idx[n_train:n_train + n_valid], dtype=torch.long)
        test = torch.tensor(all_idx[n_train + n_valid:], dtype=torch.long)
        return {'train': train, 'valid': valid, 'test': test}

    ds.get_idx_split = get_idx_split
    return ds

def load_nc_dataset(args):
    dataname = args.dataset if hasattr(args, 'dataset') else args
    data_dir = args.data_dir if hasattr(args, 'data_dir') else None
    return load_planetoid_dataset(dataname, data_dir=data_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cora')
    parser.add_argument('--data_dir', default=os.path.join('..', 'data'))
    parsed = parser.parse_args()
    ds = load_planetoid_dataset(parsed.dataset, data_dir=parsed.data_dir)
    print("Loaded:", parsed.dataset, "num_nodes:", ds.graph['num_nodes'], "feat shape:", ds.graph['node_feat'].shape, "label shape:", ds.label.shape)

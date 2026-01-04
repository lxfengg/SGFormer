# Replacement dataset.py + sanity_check.py
# File: dataset.py (first part)
# -----------------------------
import os
import pickle as pkl
from os import path

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected


# Default DATAPATH will be overwritten at runtime by load_nc_dataset using args.data_dir
DATAPATH = '../../data/'


class NCDataset(object):
    def __init__(self, name, root=f'{DATAPATH}'):
        self.name = name
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25):
        # Not used for Planetoid fixed splits, but kept for compatibility
        raise NotImplementedError('Use dataset.train_idx / valid_idx / test_idx for fixed splits')

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


def load_nc_dataset(args):
    """ Public loader used by main.py. Accepts the argparse Namespace `args` and returns NCDataset.
    This implementation uses torch_geometric.datasets.Planetoid for cora/citeseer/pubmed and
    performs robust postprocessing (label shape normalization, mask extraction, checks).
    """
    global DATAPATH
    if hasattr(args, 'data_dir') and args.data_dir:
        DATAPATH = args.data_dir
    dataname = args.dataset
    print(dataname)
    if dataname in ('cora', 'citeseer', 'pubmed'):
        dataset = load_planetoid_dataset(dataname, no_feat_norm=getattr(args, 'no_feat_norm', False))
    else:
        raise ValueError(f'Unsupported dataname in this loader: {dataname}')
    return dataset


def _ensure_label_vector(y):
    """Ensure labels are a LongTensor of shape (N,1) with class indices, not one-hot."""
    if y is None:
        raise ValueError('Labels are None')
    if isinstance(y, torch.Tensor):
        if y.dim() == 1:
            y = y.long().unsqueeze(1)
        elif y.dim() == 2:
            # If already (N,1)
            if y.shape[1] == 1:
                y = y.long()
            else:
                # if floats or one-hot, convert to index
                if y.dtype.is_floating_point or y.max() > 1:
                    y = y.argmax(dim=1).long().unsqueeze(1)
                else:
                    # integer multi-label? still take argmax
                    y = y.argmax(dim=1).long().unsqueeze(1)
        else:
            raise ValueError('Unsupported label tensor shape: ' + str(y.shape))
    else:
        y = torch.tensor(y)
        return _ensure_label_vector(y)
    return y


def load_planetoid_dataset(name, no_feat_norm=False):
    """Load Planetoid dataset robustly and return NCDataset with attributes:
    dataset.graph: {'edge_index', 'node_feat', 'edge_feat', 'num_nodes'}
    dataset.label: LongTensor (N,1)
    dataset.train_idx / valid_idx / test_idx: 1D LongTensor indexes
    """
    global DATAPATH
    root = os.path.join(DATAPATH, 'Planetoid')
    print(f"[INFO] Planetoid root: {root}")

    if not no_feat_norm:
        transform = T.NormalizeFeatures()
        torch_dataset = Planetoid(root=root, name=name, transform=transform)
    else:
        torch_dataset = Planetoid(root=root, name=name)

    data = torch_dataset[0]

    # node features
    node_feat = data.x
    if node_feat is None:
        raise ValueError('Planetoid returned None for x')
    node_feat = node_feat.to(torch.float)

    # edge_index
    edge_index = data.edge_index
    if edge_index is None:
        raise ValueError('Planetoid returned None for edge_index')
    # Make undirected for downstream code if needed
    try:
        edge_index = to_undirected(edge_index)
    except Exception:
        pass

    # labels: robust conversion
    label = data.y
    label = _ensure_label_vector(label)

    num_nodes = data.num_nodes
    print(f"Num nodes: {num_nodes}")

    dataset = NCDataset(name)

    # masks -> indices
    if hasattr(data, 'train_mask') and data.train_mask is not None:
        dataset.train_idx = torch.where(data.train_mask)[0]
    else:
        dataset.train_idx = None
    if hasattr(data, 'val_mask') and data.val_mask is not None:
        dataset.valid_idx = torch.where(data.val_mask)[0]
    else:
        dataset.valid_idx = None
    if hasattr(data, 'test_mask') and data.test_mask is not None:
        dataset.test_idx = torch.where(data.test_mask)[0]
    else:
        dataset.test_idx = None

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label

    # Sanity checks and helpful debug prints
    try:
        n = num_nodes
        lab = label.squeeze(1)
        print('DEBUG: labels shape:', tuple(lab.shape))
        unique, counts = torch.unique(lab, return_counts=True)
        print('DEBUG: label unique:', unique.tolist())
        print('DEBUG: label counts:', counts.tolist())
    except Exception as e:
        print('[WARN] failed to print label debug info:', e)

    if dataset.train_idx is not None:
        print('DEBUG: train idx length:', len(dataset.train_idx))
    if dataset.valid_idx is not None:
        print('DEBUG: valid idx length:', len(dataset.valid_idx))
    if dataset.test_idx is not None:
        print('DEBUG: test idx length:', len(dataset.test_idx))

    # Final consistency checks
    if dataset.test_idx is None:
        print('[WARN] test_idx is None (unexpected)')
    else:
        if dataset.test_idx.max().item() >= num_nodes:
            raise IndexError('test_idx contains index >= num_nodes')

    return dataset

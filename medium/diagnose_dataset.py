# 保存为 diagnose_dataset.py
# 用法: python diagnose_dataset.py --csv_or_dataset_path ... (if using your loader, adapt)
import argparse
import torch
import numpy as np
import pandas as pd

# 修改以下两行以适配你的 dataset loader，如果你已有 load_nc_dataset(args) 可直接 import 使用
from dataset import load_nc_dataset

def homophily(edge_index, labels, num_nodes):
    # simple node-level homophily: fraction of edges connecting same-label nodes
    if edge_index is None:
        return None
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    same = 0
    total = len(src)
    labs = labels.squeeze(1).cpu().numpy()
    for s,d in zip(src,dst):
        if labs[s] == labs[d]:
            same += 1
    return float(same) / max(1, total)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_name', type=str, default='cora')
    p.add_argument('--data_dir', type=str, default=None)
    p.add_argument('--use_loader', action='store_true', help='use existing load_nc_dataset loader')
    args = p.parse_args()

    if args.use_loader:
        # assumes loader returns dataset like in your main.py
        class Args: pass
        a = Args()
        a.dataset = args.dataset_name
        a.data_dir = args.data_dir
        dataset = load_nc_dataset(a)
        graph = dataset.graph
        edge_index = graph['edge_index']
        x = graph['node_feat']
        labels = dataset.label
        n = graph['num_nodes']
    else:
        raise RuntimeError("Please set --use_loader and ensure load_nc_dataset is importable.")

    print("=== Dataset basic info ===")
    print("num_nodes:", n)
    print("num_edges:", int(edge_index.shape[1]))
    print("avg_degree:", float(edge_index.shape[1]) * 1.0 / n)
    isolated = ((torch.bincount(edge_index.view(-1)).cpu().numpy()==0).sum()) if False else None
    print("node_feat shape:", x.shape)
    labs = labels.squeeze(1).cpu().numpy()
    unique, counts = np.unique(labs, return_counts=True)
    print("num_classes:", len(unique))
    print("label counts (per class):", dict(zip(unique.tolist(), counts.tolist())))
    print("label fraction (labeled if mask exists): check split in your loader)")

    print("=== Feature stats ===")
    print("feature mean per-dim (first 5):", np.mean(x.cpu().numpy(), axis=0)[:5])
    print("feature std per-dim (first 5):", np.std(x.cpu().numpy(), axis=0)[:5])
    # simple feature-only baseline: kNN on features
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    X = x.cpu().numpy()
    Y = labs
    # If dataset has explicit splits, you'd use them. Here approximate random eval:
    trX, teX, trY, teY = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)
    knn = KNeighborsClassifier(n_neighbors=5).fit(trX, trY)
    acc = knn.score(teX, teY)
    print("kNN (5) accuracy on random split (feature-only baseline):", acc)

    print("=== Homophily ===")
    h = homophily(edge_index, labels, n)
    print("edge homophily fraction:", h)

    # degree distribution summary
    src = edge_index[0].cpu().numpy()
    deg = np.bincount(src, minlength=n)
    print("degree: min, median, mean, max:", deg.min(), np.median(deg), deg.mean(), deg.max())
    print("percent isolated (degree==0):", float((deg==0).sum())/n)

if __name__ == '__main__':
    main()

# 保存为 reachability.py
import numpy as np
from collections import deque
import torch
from dataset import load_nc_dataset

class Args: pass
a = Args()
a.dataset = 'cora'
a.data_dir = None
dataset = load_nc_dataset(a)

edge_index = dataset.graph['edge_index'].cpu().numpy()
n = dataset.graph['num_nodes']
# get train split robustly
try:
    split_idx = dataset.get_idx_split(train_prop=0.05, valid_prop=0.185)
except Exception:
    from data_utils import load_fixed_splits
    split_idx = load_fixed_splits(dataset, name=a.dataset, protocol='semi')[0]

train_idx = split_idx['train'].cpu().numpy().tolist()
adj = [[] for _ in range(n)]
for u,v in edge_index.T:
    adj[u].append(v); adj[v].append(u)

dist = [-1]*n
q = deque()
for t in train_idx:
    dist[t]=0; q.append(t)
while q:
    u = q.popleft()
    for v in adj[u]:
        if dist[v]==-1:
            dist[v]=dist[u]+1
            q.append(v)
d = np.array(dist)
print("num nodes:", n)
print("num training nodes:", len(train_idx))
print("fraction unreachable:", float((d==-1).sum())/n)
for k in [0,1,2,3,4,5]:
    print(f"frac within <={k} hops:", float((d<=k).sum())/n)
# show distribution stats for reachable distances
reachable = d[d>=0]
if reachable.size>0:
    print("reachable dist: min, median, mean, max:", reachable.min(), np.median(reachable), reachable.mean(), reachable.max())
else:
    print("No nodes reachable (unexpected).")

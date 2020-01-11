import dgl
from dgl.data.ppi import LegacyPPIDataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import networkx as nx
import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import os
import argparse
from transfers.utils import gen_graph

parser = argparse.ArgumentParser()
parser.add_argument("--lam", type=float, default=1.0)
parser.add_argument("--mu", type=float, default=0.1)
parser.add_argument("--p", type=int, default=128)
parser.add_argument("--n1", type=int, default=256)
parser.add_argument("--n2", type=int, default=256)
parser.add_argument('--seed', type=int, default=100)
args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

A1, F1, L1 = gen_graph(n=args.n1, p=args.p, lam=args.lam, mu=args.mu)
A2, F2, L2 = gen_graph(n=args.n2, p=args.p, lam=args.lam, mu=args.mu)
print(F1.shape)
A1 = torch.FloatTensor(A1).to(device)
A2 = torch.FloatTensor(A2).to(device)

# gen edgelist, labels, featuresh
# kmeans
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(F1)
kmeans_labels = kmeans.labels_
print("Cluster 0")
ids = np.argwhere(kmeans_labels == 0).flatten()
print(np.unique(L1[ids], return_counts=True))
print("Cluster 1")
ids = np.argwhere(kmeans_labels == 1).flatten()
print(np.unique(L1[ids], return_counts=True))

print("Save graphs")
X1 = F1
X2 = F2
features = X1
edgelist = np.argwhere(A1.detach().cpu().numpy() > 0)
labels = L1

outdir = f"data-transfers/syn-seed{args.seed}/0"
if not os.path.isdir(outdir):
    os.makedirs(outdir)

with open(outdir + f"/syn0.txt", "w+") as fp:
    for src, trg in edgelist:
        fp.write(f"{src} {trg}\n")
with open(outdir + "/labels.txt", "w+") as fp:
    for i, label in enumerate(labels):
        fp.write(f"{i} {label}\n")
np.savez_compressed(outdir + "/features.npz", features=features)
# np.savetxt(outdir+"/features.tsv", X1, fmt="%.4f", delimiter="\t")

features = X2
edgelist = np.argwhere(A2.detach().cpu().numpy() > 0)
labels = L2

outdir = f"data-transfers/syn-seed{args.seed}/1"
if not os.path.isdir(outdir):
    os.makedirs(outdir)

with open(outdir + f"/syn1.txt", "w+") as fp:
    for src, trg in edgelist:
        fp.write(f"{src} {trg}\n")
with open(outdir + "/labels.txt", "w+") as fp:
    for i, label in enumerate(labels):
        fp.write(f"{i} {label}\n")
np.savez_compressed(outdir + "/features.npz", features=features)


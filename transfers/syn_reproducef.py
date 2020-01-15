# reproduce f from knn or sigmoid

# -*- coding: utf-8 -*-
import os
from sklearn.metrics import f1_score
"""ppi.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1i8xRqB1bvfLoTvICrMtHFuvKqdvU5x6D
"""

# !pip install dgl-cu101

import dgl
from dgl.data.ppi import PPIDataset

from dgl.data.ppi import LegacyPPIDataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
import argparse 
from transfers.utils import gen_graph, generate_graph

parser = argparse.ArgumentParser()
parser.add_argument("--lam", type=float, default=1.0)
parser.add_argument("--mu", type=float, default=0)
parser.add_argument("--p", type=int, default=128)
parser.add_argument("--n", type=int, default=256)
parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--kind', default='knn', help="choose from knn sigmoid")
parser.add_argument("--threshold", type=float, default=0.6)
parser.add_argument("--n_graphs", type=int, default=256)
parser.add_argument("--epochs", type=int, default=2000)
args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

graphs = []
train_size = args.n_graphs - 2
u = np.random.multivariate_normal(np.zeros((args.p)), np.eye(args.p)/args.p, 1)
for i in range(args.n_graphs):
    _, X, L = gen_graph(n=args.n, p=args.p, lam=args.lam, mu=args.mu, u=u)
    edge_index = generate_graph(torch.FloatTensor(X), kind=args.kind, k=5, threshold=args.threshold)
    A = np.zeros((len(X), len(X)))
    A[edge_index[:,0], edge_index[:,1]] = 1
    graphs.append((A, X, L))
train_graphs = graphs[:train_size]
test_graphs = graphs[train_size:]

def compute_f1(pA, A):
    pA = pA.detach().cpu().numpy()
    pA[pA >= 0.5] = 1
    pA[pA < 0.5] = 0
    A = A.cpu().numpy()
    f1 = f1_score(A, pA, average="macro")
    return f1

class ModelSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Sequential(
            nn.Linear(args.n*(args.n-1)//2, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, args.n*(args.n-1)//2, bias=True)
        )

    def forward(self, graphs):
        Xs = [torch.FloatTensor(x).to(device) for _,x,_ in graphs]
        # xs = [F.normalize(x, dim=1) for x in Xs]
        xs = [self.W(torch.pdist(d)) for d in Xs]
        # xs = [(x-x.mean(dim=0))/x.std(dim=0) for x in xs]
        xs = [F.sigmoid(x) for x in xs]
        return xs

class ModelKNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.Xs = [
            torch.FloatTensor(x).to(device) for _,x,_ in train_graphs
        ]
        n_nodes, fdim = self.Xs[0].shape
        self.W0 = nn.Sequential(
            nn.Linear(fdim, fdim*2),
            nn.ReLU(),
            nn.Linear(fdim*2, fdim)
        )
        self.W = nn.Sequential(
            nn.Linear(n_nodes, n_nodes*2, bias=True),
            # nn.BatchNorm1d(n_nodes*2),
            nn.ReLU(),
            nn.Linear(n_nodes*2, n_nodes, bias=True),
            # nn.Softmax()
        )
        self.W2 = nn.Linear(n_nodes, n_nodes)

    def forward(self):
        xs = [F.normalize(x, dim=1) for x in self.Xs]
        # xs = [self.W0(x) for x in self.Xs]
        xs = [self.W(x.mm(x.t())) for x in xs]
        xs = [self.W2(F.log_softmax(x, dim=1)) for x in xs]
        xs = [F.sigmoid(x) for x in xs]
        # for x in xs:
        #     x[np.arange(len(x)), np.arange(len(x))] = 0
        # halfxs = []
        # for x in xs:
        #     inds = torch.triu(torch.ones(len(x),len(x))) 
        #     inds[np.arange(len(x)), np.arange(len(x))] = 0
        #     halfxs.append(x[inds == 1])
        # xs = [x for x in self.Xs]
        # xs = [x.mm(x.t()) for x in xs]
        # N = xs[0].shape[0]
        # xs = [torch.bmm(x.view(N, N, 1), x.view(N, 1, N)) for x in xs]


        return xs

if args.kind == "knn":
    model = ModelKNN().to(device)
else:
    model = ModelSigmoid().to(device)
# loss_fn = nn.MSELoss()
loss_fn = nn.BCELoss()
optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

halfAs = []
if args.kind == "sigmoid":
    for A,_,_ in graphs:
        inds = torch.triu(torch.ones(len(A),len(A))) 
        inds[np.arange(len(A)), np.arange(len(A))] = 0
        halfA = torch.FloatTensor(A[inds == 1]).to(device)
        halfAs.append(halfA)
elif args.kind == "knn":
    halfAs = []
    for A,_,_ in graphs:
        halfA = A.copy()
        # halfA[np.arange(len(A)), np.arange(len(A))] = 0
        halfAs.append(torch.FloatTensor(halfA).to(device))
train_halfAs = halfAs[:train_size]
test_halfAs = halfAs[train_size:]

for iter in range(args.epochs):
    model.train()
    optim.zero_grad()
    inds = np.random.permutation(len(train_graphs))[:16]
    batch_graphs = [train_graphs[x] for x in inds]
    batch_halfAs = [halfAs[x] for x in inds]

    pred_As = model(batch_graphs)
    loss = 0
    for pred_A, halfA in zip(pred_As, batch_halfAs):
        loss += loss_fn(pred_A, halfA)
    loss = loss / len(pred_As)
    loss.backward()
    optim.step()
    if iter % 50 == 0:
        # microf11 = compute_f1(pred_As[0], halfAs[0])
        # microf12 = compute_f1(pred_As[1], halfAs[1])
        microfs = [compute_f1(pred_A, halfA) for predA, halfA in zip(pred_As, batch_halfAs)]
        pred_As = model(test_graphs)
        microfs += [compute_f1(pred_A, halfA) for predA, halfA in zip(pred_As, test_halfAs)]
        microstr = " ".join([f"{f1:.2f}" for f1 in microfs])
        print(f"Iter {iter} - loss {loss:.4f} - f1 {microstr}")

def save_graphs(A, X, L, outdir):
    edgelist = np.argwhere(A > 0)
    dataname = outdir.split("/")[-1]
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    with open(outdir + f"/{dataname}.txt", "w+") as fp:
        for src, trg in edgelist:
            fp.write(f"{src} {trg}\n")
    with open(outdir + "/labels.txt", "w+") as fp:
        for i, label in enumerate(L):
            fp.write(f"{i} {label}\n")
    np.savez_compressed(outdir + "/features.npz", features=X)

print("save graphs")
# G(Atrain, Xtrain)
A, X, L = train_graphs[0]
save_graphs(A, X, L, f"data-transfers/synf-seed{args.seed}/Atrain-Xtrain")
# G(Xtest, f(Xtest))

_, X, L = test_graphs[0]
pdistA = model([test_graphs[0]])[0].detach().cpu().numpy()
A = np.zeros((len(X), len(X)))
inds = torch.triu(torch.ones(len(A),len(A))) 
inds[np.arange(len(A)), np.arange(len(A))] = 0
A[inds == 1] = pdistA
A[np.arange(len(A)), np.arange(len(A))] = 1
A[A>=0.5] = 1
A[A<0.5] = 0
save_graphs(A, X, L, f"data-transfers/synf-seed{args.seed}/A2-Xtest")

# G(Xtest, KNN(Xtest))
_, X, L = test_graphs[0]
edge_index = generate_graph(torch.FloatTensor(X), kind="knn", k=5)
A = np.zeros((len(X), len(X)))
A[edge_index[:,0], edge_index[:,1]] = 1
save_graphs(A, X, L, f"data-transfers/synf-seed{args.seed}/A3-Xtest")

# G(Xtest, sigmoid(Xtest))
_, X, L = test_graphs[0]
edge_index = generate_graph(torch.FloatTensor(X), kind="sigmoid", threshold=args.threshold)
A = np.zeros((len(X), len(X)))
A[edge_index[:,0], edge_index[:,1]] = 1
save_graphs(A, X, L, f"data-transfers/synf-seed{args.seed}/A4-Xtest")

# G(Xtest, Atest)
A, X, L = test_graphs[0]
save_graphs(A, X, L, f"data-transfers/synf-seed{args.seed}/Atest-Xtest")

# gen edgelist, labels, featuresh
# print("Save graphs")
# X1 = F1
# X2 = F2
# features = X1
# edgelist = np.argwhere(A1.detach().cpu().numpy() > 0)
# labels = L1

# outdir = f"data-transfers/synD-seed{args.seed}/0"
# if not os.path.isdir(outdir):
#     os.makedirs(outdir)

# with open(outdir + f"/0.txt", "w+") as fp:
#     for src, trg in edgelist:
#         fp.write(f"{src} {trg}\n")
# with open(outdir + "/labels.txt", "w+") as fp:
#     for i, label in enumerate(labels):
#         fp.write(f"{i} {label}\n")
# np.savez_compressed(outdir + "/features.npz", features=features)

# features = X2
# edgelist = np.argwhere(A2.detach().cpu().numpy() > 0)
# labels = L2

# outdir = f"data-transfers/synD-seed{args.seed}/1"
# if not os.path.isdir(outdir):
#     os.makedirs(outdir)

# with open(outdir + f"/1.txt", "w+") as fp:
#     for src, trg in edgelist:
#         fp.write(f"{src} {trg}\n")
# with open(outdir + "/labels.txt", "w+") as fp:
#     for i, label in enumerate(labels):
#         fp.write(f"{i} {label}\n")
# np.savez_compressed(outdir + "/features.npz", features=features)


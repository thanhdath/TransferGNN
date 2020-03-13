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

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=100)
args = parser.parse_args()


def collate(sample):
    graphs, feats, labels = map(list, zip(*sample))
    graph = dgl.batch(graphs)
    feats = torch.from_numpy(np.concatenate(feats))
    labels = torch.from_numpy(np.concatenate(labels))
    return graph, feats, labels


train_dataset = LegacyPPIDataset(mode="test")
import pdb; pdb.set_trace()


ids = np.random.permutation(len(train_dataset))
nnodes = [x[0].number_of_nodes() for x in train_dataset]
ids = [i for i in ids if train_dataset[i][0].number_of_nodes() < 15000000]
G1, F1, L1_ori = train_dataset[ids[0]]
G2, F2, L2_ori = train_dataset[ids[1]]
A1 = np.asarray(G1.adjacency_matrix_scipy().todense())
A2 = np.asarray(G2.adjacency_matrix_scipy().todense())
A1[A1 > 0] = 1
A2[A2 > 0] = 1
print(A1.shape, A2.shape)
A1 = torch.FloatTensor(A1).cuda()
A2 = torch.FloatTensor(A2).cuda()


def compute_f1(pA, A):
    pA = pA.detach().cpu().numpy()
    pA[pA >= 0.5] = 1
    pA[pA < 0.5] = 0
    A = A.cpu().numpy()
    f1 = f1_score(A, pA, average="micro")
    return f1


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.D1 = torch.FloatTensor(F1).cuda()
        # self.D2 = torch.FloatTensor(F2).cuda()
        self.D1 = nn.Parameter(torch.empty(F1.shape).normal_(mean=0., std=1.)).cuda()
        self.D2 = nn.Parameter(torch.empty(F2.shape).normal_(mean=0., std=1.)).cuda()
        fdim = self.D1.shape[1]
        self.W = nn.Sequential(
            nn.Linear(fdim, fdim*2, bias=True),
            nn.ReLU(),
            nn.Linear(fdim*2, fdim*2, bias=True),
            nn.ReLU(),
            nn.Linear(fdim*2, fdim, bias=True),
        )

    def forward(self):
        D1 = self.W(self.D1)
        x1 = D1.mm(D1.t())
        D2 = self.W(self.D2)
        x2 = D2.mm(D2.t())
        return x1, x2


for selected_label in range(121):
    print("Selected label: ", selected_label)
    L1 = L1_ori[:, selected_label]
    L2 = L2_ori[:, selected_label]

    model = Model().cuda()
    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for iter in range(400):
        model.train()
        optim.zero_grad()
        pred_A1, pred_A2 = model()
        loss = loss_fn(pred_A1, A1) + loss_fn(pred_A2, A2)
        loss.backward()
        optim.step()
        if iter % 50 == 0:
            microf11 = compute_f1(pred_A1, A1)
            microf12 = compute_f1(pred_A2, A2)
            print(f"Iter {iter} - loss {loss:.4f} - f1 {microf11:.3f}  {microf12:.3f}")

    # gen edgelist, labels, featuresh
    print("Save graphs")
    X1 = F1
    X2 = F2
    features = X1
    edgelist = np.argwhere(A1.detach().cpu().numpy() > 0)
    labels = L1

    outdir = f"data-transfers/ppi-seed{args.seed}/labels{selected_label}-0"
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    with open(outdir + f"/labels{selected_label}-0.txt", "w+") as fp:
        for src, trg in edgelist:
            fp.write(f"{src} {trg}\n")
    with open(outdir + "/labels.txt", "w+") as fp:
        for i, label in enumerate(labels):
            fp.write(f"{i} {label}\n")
    np.savez_compressed(outdir + "/features.npz", features=features)

    features = X2
    edgelist = np.argwhere(A2.detach().cpu().numpy() > 0)
    labels = L2

    outdir = f"data-transfers/ppi-seed{args.seed}/labels{selected_label}-1"
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    with open(outdir + f"/labels{selected_label}-1.txt", "w+") as fp:
        for src, trg in edgelist:
            fp.write(f"{src} {trg}\n")
    with open(outdir + "/labels.txt", "w+") as fp:
        for i, label in enumerate(labels):
            fp.write(f"{i} {label}\n")
    np.savez_compressed(outdir + "/features.npz", features=features)

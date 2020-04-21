# ppi, learn and transfer from multi graphs

import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import ChebConv, GATConv, GCNConv, SAGEConv
from data import load_graph
import argparse
import os
from sklearn.metrics import f1_score
import numpy as np
#import matplotlib.pyplot as plt
import tensorboardX
from sklearn.metrics import classification_report
# from sage import SAGEConv

import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer
from torch_geometric.data import NeighborSampler, Data
import re
from scipy.sparse import csr_matrix, vstack, hstack

import json
import networkx as nx
from networkx.readwrite import json_graph
from networkx.generators.random_graphs import connected_watts_strogatz_graph
from transfers.utils import gen_graph, generate_graph

def gen_sbm():
    graphs = []
    u = np.random.multivariate_normal(np.zeros((args.p)), np.eye(args.p)/args.p, 1)
    for i in range(args.n_graphs):
        if i % 100 == 0:
            print(f"{i+1}/{args.n_graphs}")
        Asbm, X, L = gen_graph(n=args.n, p=args.p, lam=args.lam, mu=args.mu, u=u)
        Asbm[np.arange(len(Asbm)), np.arange(len(Asbm))] = 0
        src, trg = Asbm.nonzero()
        edge_index = torch.LongTensor(np.concatenate([src.reshape(1, -1), trg.reshape(1,-1)], axis=0))
        features = torch.FloatTensor(X)
        labels = torch.LongTensor(L)
        graphs.append((edge_index, features, labels, None))
    n_train = int(args.n_graphs*0.8)
    train_graphs = graphs[:n_train]
    test_graphs = graphs[n_train:]

    if args.f != "ori":
        converted_test_dataset = []
        for edge_index, x, y, _ in test_graphs:
            k = edge_index.shape[1] // x.shape[0]
            if args.f == "knn":
                adj = generate_graph(x, kind="knn", k=k)
            if args.f == "sigmoid":
                adj = generate_graph(x, kind="sigmoid", k=k)
            if args.f == "random":
                G = connected_watts_strogatz_graph(x.shape[0], k+1, p=0.1)
                adj = nx.to_numpy_matrix(G)
            
            src, trg = adj.nonzero()
            new_edge_index = torch.LongTensor(np.concatenate([src.reshape(1, -1), trg.reshape(1,-1)], axis=0))
            print(f"N edges before {edge_index.shape[1]} - N edges then {new_edge_index.shape[1]} ")
            converted_test_dataset.append((new_edge_index, x, y, None))
        test_graphs = converted_test_dataset
    return train_graphs, test_graphs

parser = argparse.ArgumentParser()
parser.add_argument('--lam', type=float, default=1.1)
parser.add_argument('--mu', type=float, default=100)
parser.add_argument('--p', type=int, default=8)
parser.add_argument('--n', type=int, default=32)
parser.add_argument('--n-graphs', type=int, default=300)
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--hidden", default=64, type=int)
parser.add_argument("--seed", default=100, type=int)
parser.add_argument("--f", choices=["ori", "knn", "sigmoid", "random"])
args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)

args.multiclass = False
num_features = args.p 
num_classes = 2
print("Num features:", num_features)
print("Num classes:", num_classes)

train_graphs, test_graphs = gen_sbm()
# GCN

# class Net(torch.nn.Module):

#     def __init__(self):
#         super(Net, self).__init__()
#         id_features_dim = 128
#         self.mapping = nn.Linear(num_features, id_features_dim)
#         self.conv1 = GATConv(id_features_dim, 256, heads=4)
#         self.lin1 = torch.nn.Linear(id_features_dim, 4 * 256)
#         self.conv2 = GATConv(4 * 256, 256, heads=4)
#         self.lin2 = torch.nn.Linear(4 * 256, 4 * 256)
#         self.conv3 = GATConv(4 * 256, num_classes, heads=6, concat=False)
#         self.lin3 = torch.nn.Linear(4 * 256, num_classes)

#     def forward(self, x, edge_index, edge_attr=None):
#         x = self.mapping(x)
#         x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
#         x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
#         x = self.conv3(x, edge_index) + self.lin3(x)
#         return x

# class Net(torch.nn.Module):

#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = GATConv(num_features, 256, heads=4)
#         self.lin1 = torch.nn.Linear(num_features, 4 * 256)
#         self.conv2 = GATConv(4 * 256, num_classes, heads=4, concat=False)
#         self.lin2 = torch.nn.Linear(4 * 256, num_classes)
#         # self.conv3 = GATConv(4 * 256, num_classes, heads=6, concat=False)
#         # self.lin3 = torch.nn.Linear(4 * 256, num_classes)

#     def forward(self, x, edge_index):
#         x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
#         x = self.conv2(x, edge_index) + self.lin2(x)
#         # x = self.conv3(x, edge_index) + self.lin3(x)
#         return x


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # id_features_dim = 128
        # self.mapping = nn.Linear(num_features, id_features_dim)
        self.conv1 = SAGEConv(num_features, args.hidden, normalize=False)
        self.conv2 = SAGEConv(args.hidden, args.hidden * 2, normalize=False)
        self.conv3 = SAGEConv(args.hidden * 2, num_classes, normalize=False)

    def forward(self, x, edge_index, edge_attr=None):
        # x = self.mapping(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

def train(train_graphs):
    model.train()
    inds = np.random.permutation(len(train_graphs))
    train_graphs = [train_graphs[x] for x in inds]
    for edge_index, x, y, edge_attr in train_graphs:
        edge_index = edge_index.to(device)
        if edge_attr is not None:
            edge_attr = edge_attr.to(device)
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        outputs = model(x, edge_index, edge_attr=edge_attr)
        F.nll_loss(outputs, y).backward()
        optimizer.step()


def f1(output, labels, multiclass=False):
    if len(output) == 0:
        return 0, 0
    if not multiclass:
        preds = output.max(1)[1]
        preds = preds.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        micro = f1_score(labels, preds, average='micro')
        macro = f1_score(labels, preds, average='macro')
        return micro, macro
    else:
        # probs = torch.sigmoid(output)
        # probs[probs > 0.5] = 1
        # probs[probs <= 0.5] = 0
        probs = (output > 0).float()
        probs = probs.cpu().detach().numpy().astype(np.int32)
        labels = labels.cpu().detach().numpy().astype(np.int32)
        micro = f1_score(labels, probs, average='micro')
        macro = f1_score(labels, probs, average='macro')
        return micro, macro


def test(graphs, n_randoms=None):
    if len(graphs) == 0:
        return 0, 0
    model.eval()
    micros = []
    macros = []
    logitss = []
    ys = []
    if n_randoms is not None:
        inds = np.random.permutation(len(graphs))[:n_randoms]
        graphs = [graphs[x] for x in inds]

    for edge_index, x, y, edge_attr in graphs:
        edge_index = edge_index.to(device)
        if edge_attr is not None:
            edge_attr = edge_attr.to(device)
        x = x.to(device)
        y = y.to(device)
        logits = model(x, edge_index, edge_attr)
        logitss.append(logits)
        ys.append(y)
    logitss = torch.cat(logitss, dim=0)
    ys = torch.cat(ys, dim=0)
    micro, macro = f1(logitss, ys, multiclass=args.multiclass)
    return micro, macro


def getdirname(path):
    if path is None:
        return None
    return [x for x in path.split("/") if len(x) > 0][-1]


def getfilename(path):
    if path is None:
        return None
    return path.split("/")[-1].split(".")[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
lr = 0.005
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

epochs = args.epochs
train_acc, train_macro = test(train_graphs, n_randoms=2)
log = 'Epoch: 0, micro-macro Train: {:.4f}-{:.4f}'
print(log.format(train_acc, train_macro))

# best_val_acc = val_acc
for epoch in range(1, epochs):
    train(train_graphs)
    train_acc, train_macro = test(train_graphs, n_randoms=2)
    if epoch % 20 == 0 or epoch == epochs - 1:
        log = 'Epoch: {:03d}, micro-macro Train: {:.4f}-{:.4f}'
        print(log.format(epoch, train_acc, train_macro))

val_acc, val_macro = test(test_graphs)
log = 'Val: {:.4f}-{:.4f}'
print(log.format(val_acc, val_macro))

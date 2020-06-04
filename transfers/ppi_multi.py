import torch
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, GCNConv, SAGEConv, SGConv, GATConv
import argparse
import os
from sklearn.metrics import f1_score
import numpy as np
import torch.nn as nn
import networkx as nx
import re
import pickle
import json
from networkx.readwrite import json_graph
from transfers.utils import generate_graph
from networkx.generators.random_graphs import connected_watts_strogatz_graph

def get_graph(g):
    features = np.array([g.nodes()[x]['features'] for x in g.nodes()])
    node_labels = np.array([g.nodes()[x]['label'] for x in g.nodes()], dtype=np.float32)
    features = torch.FloatTensor(features)
    labels = torch.FloatTensor(node_labels)
    adj = nx.to_numpy_matrix(g)
    src, trg = adj.nonzero()
    edge_index = torch.LongTensor(np.concatenate([src.reshape(1, -1), trg.reshape(1,-1)], axis=0))
    return edge_index, features, labels


parser = argparse.ArgumentParser()
parser.add_argument("--data-path", default="data/ppi.pkl")
parser.add_argument("--epochs", default=300, type=int)
parser.add_argument("--batch-size", default=4, type=int)
parser.add_argument("--hidden", default=64, type=int)
parser.add_argument("--model", default='mean',
                    choices=['gat', 'mean', 'sum', 'sgc', 'gcn', 'mlp'])
parser.add_argument("--seed", default=100, type=int)
parser.add_argument("--f", default='ori', choices=['knn', 'sigmoid', 'ori', 'random'])
args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

obj = pickle.load(open(args.data_path, "rb"))
train_graphs = obj['train_graphs']
val_graphs = train_graphs[-2:]
train_graphs = train_graphs[:-2]
test_graphs = obj['test_graphs']
num_features = len(train_graphs[0].nodes()[0]['features'])
num_classes = obj["n_classes"]
multilabel = obj["multilabel"]

print("Num features:", num_features)
print("Num classes:", num_classes)

train_graphs = [get_graph(g) for g in train_graphs]
val_graphs = [get_graph(g) for g in val_graphs]
test_graphs = [get_graph(g) for g in test_graphs]

# select best graph for training and testing based on number of labels each classes
if args.f != "ori":
    converted_test_graphs = []
    for edge_index, features, labels in test_graphs:
        k = edge_index.shape[1] // features.shape[0]
        if args.f == "knn":
            adj = generate_graph(features, kind="knn", k=k)
        elif args.f == "sigmoid":
            adj = generate_graph(features, kind="sigmoid", k=k)
        elif args.f == 'random':
            G = connected_watts_strogatz_graph(features.shape[0], k+1, p=0.1)
            adj = nx.to_numpy_matrix(G)
        src, trg = adj.nonzero()
        new_edge_index = torch.LongTensor(np.concatenate([src.reshape(1, -1), trg.reshape(1,-1)], axis=0))
        converted_test_graphs.append((new_edge_index, features, labels))
    test_graphs = converted_test_graphs

class GraphsageMEAN(torch.nn.Module):

    def __init__(self):
        super(GraphsageMEAN, self).__init__()
        self.conv1 = SAGEConv(num_features, args.hidden, normalize=False)
        self.conv2 = SAGEConv(args.hidden, args.hidden * 2, normalize=False)
        self.conv3 = SAGEConv(args.hidden * 2, num_classes, normalize=False)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        if not multilabel:
            return F.log_softmax(x, dim=1)
        return x

def merge_graphs(graphs):
    edges = []
    xs = []
    ys = []
    offset = 0
    for edge_index, x, y in graphs:
        edges.append(edge_index + offset)
        xs.append(x)
        ys.append(y)
        offset += x.shape[0]
    edges = torch.cat(edges, dim=1)
    xs = torch.cat(xs, dim=0)
    ys = torch.cat(ys, dim=0)
    if not multilabel and len(ys.shape) == 2:
        ys = ys.argmax(dim=1)
    assert edges.shape[0] == 2
    return edges, xs, ys


def train(train_graphs):
    model.train()
    inds = np.random.permutation(len(train_graphs))
    train_graphs = [train_graphs[x] for x in inds]
    n_iter = int(np.ceil(len(train_graphs) / args.batch_size))
    for iter in range(n_iter):
        batch_graphs = train_graphs[iter * args.batch_size:(iter + 1) * args.batch_size]
        edge_index, x, y = merge_graphs(batch_graphs)
        edge_index = edge_index.to(device)
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        outputs = model(x, edge_index).view(len(x), -1)
        if not multilabel:
            F.nll_loss(outputs, y).backward()
        else:
            criterion(outputs, y).backward()
        
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

    n_iter = int(np.ceil(len(graphs) / args.batch_size))
    for iter in range(n_iter):
        batch_graphs = graphs[iter * args.batch_size:(iter + 1) * args.batch_size]
        edge_index, x, y = merge_graphs(batch_graphs)
        edge_index = edge_index.to(device)
        x = x.to(device)
        y = y.to(device)
        logits = model(x, edge_index).view(len(x), -1)
        logitss.append(logits)
        ys.append(y)
    logitss = torch.cat(logitss, dim=0)
    ys = torch.cat(ys, dim=0)
    micro, macro = f1(logitss, ys, multiclass=multilabel)
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
if args.model == "gat":
    model = GAT().to(device)
elif args.model == "sum":
    model = GraphsageSUM().to(device)
elif args.model == "sgc":
    model = SGC().to(device)
elif args.model == "gcn":
    model = GCN().to(device)
elif args.model == "mlp":
    model = MLP().to(device)
elif args.model == "mean":
    model = GraphsageMEAN().to(device)
print(model)

import os
import time
if not os.path.isdir("model"):
    os.makedirs("model")
model_path = f"model/{args.model}-{time.time()}.pkl"

lr = 0.005

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
epochs = args.epochs

train_acc, _ = test(train_graphs, n_randoms=4)
val_acc, _ = test(val_graphs)
log = 'Epoch: 0, Acc Train: {:.4f}, Val: {:.4f}'
print(log.format(train_acc, val_acc))
best_val_acc = 0

for epoch in range(1, epochs):
    train(train_graphs)

    if epoch % 10 == 0 or epoch == epochs - 1:
        train_acc, _ = test(train_graphs, n_randoms=4)
        val_acc, _ = test(val_graphs)
        log = 'Epoch: {:03d}, Acc Train: {:.4f}, Val: {:.4f}'
        print(log.format(epoch, train_acc, val_acc))

        if val_acc > best_val_acc:
            torch.save(model.state_dict(), model_path)
            best_val_acc = val_acc
model.load_state_dict(torch.load(model_path))
test_acc, _ = test(test_graphs)
print(f'Test: {test_acc:.3f}')

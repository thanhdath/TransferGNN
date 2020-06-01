"""
logdir=logs/transfer-ppi/
mkdir $logdir
for seed in 100 101 102 103 104
do
    for f in ori knn sigmoid
    do
        python -u transfer_ppi.py --seed $seed --f $f > $logdir/$f-seed$seed.log
    done
done
"""

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
parser.add_argument("--epochs", default=100, type=int)
# specify num features, due to previously padded with zeros
parser.add_argument("--hidden", default=64, type=int)
parser.add_argument("--model", default='mean',
                    choices=['gat', 'mean', 'sum', 'sgc', 'gcn', 'mlp'])
parser.add_argument("--seed", default=100, type=int)
parser.add_argument("--f", default='ori', choices=['knn', 'sigmoid', 'ori'])
parser.add_argument("--th", default=0.5, type=float)
# parser.add_argument("--setting", default='A', choices=['A', 'B', 'C', 'D', 'E'])
args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

obj = pickle.load(open(args.data_path, "rb"))
train_graphs = obj['train_graphs']
test_graphs = obj['test_graphs']
num_features = len(train_graphs[0].nodes()[0]['features'])
num_classes = obj["n_classes"]
multilabel = obj["multilabel"]

print("Num features:", num_features)
print("Num classes:", num_classes)


# select best graph for training and testing based on number of labels each classes
train_graph = max(train_graphs, key=lambda x: x.number_of_nodes())
test_graph = max(test_graphs, key=lambda x: x.number_of_nodes())
train_graph = get_graph(train_graph)
test_graph = get_graph(test_graph)

if args.f == "knn":
    edge_index, features, labels = test_graph
    k = edge_index.shape[1] // features.shape[0]
    adj = generate_graph(features, kind="knn", k=k)
    src, trg = adj.nonzero()
    new_edge_index = torch.LongTensor(np.concatenate([src.reshape(1, -1), trg.reshape(1,-1)], axis=0))
    test_graph = (new_edge_index, features, labels)
if args.f == "sigmoid":
    edge_index, features, labels = test_graph
    k = edge_index.shape[1] // features.shape[0]
    adj = generate_graph(features, kind="sigmoid", k=k)
    src, trg = adj.nonzero()
    new_edge_index = torch.LongTensor(np.concatenate([src.reshape(1, -1), trg.reshape(1,-1)], axis=0))
    test_graph = (new_edge_index, features, labels)

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

def train():
    model.train()
    edge_index, x, y = train_graph
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


def test(graph):
    model.eval()
    micros = []
    macros = []
    logitss = []
    ys = []
    edge_index, x, y = graph
    edge_index = edge_index.to(device)
    x = x.to(device)
    y = y.to(device)
    logits = model(x, edge_index).view(len(x), -1)
    micro, macro = f1(logits, y, multiclass=multilabel)
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

train_acc, _ = test(train_graph)
test_acc, _ = test(test_graph)
log = 'Epoch: 0, Acc Train: {:.4f}, Test: {:.4f}'
print(log.format(train_acc, test_acc))

for epoch in range(1, epochs):
    train()

    if epoch % 10 == 0 or epoch == epochs - 1:
        train_acc, _ = test(train_graph)
        test_acc, _ = test(test_graph)
        log = 'Epoch: {:03d}, Acc Train: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, test_acc))

test_acc, _ = test(test_graph)
print(f'Test: {test_acc:.3f}')

import pdb
import torch
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, GATConv, DenseGCNConv, SAGEConv, SGConv, GCNConv
import argparse
import os
from sklearn.metrics import f1_score
import numpy as np
import torch.nn as nn
import re
import pickle
import json
import networkx as nx
from networkx.readwrite import json_graph
import random
from tqdm import tqdm
from itertools import chain
import tensorboardX
import time

class Dataset():
    def __init__(self):
        obj = pickle.load(open(args.data_path, "rb"))
        train_graphs = obj['train_graphs']
        test_graphs = obj['test_graphs']
        n_val = int(len(train_graphs) * 0.1)
        random.shuffle(train_graphs)
        self.val_graphs = train_graphs[:n_val]
        self.train_graphs = train_graphs[n_val:]
        self.test_graphs = test_graphs
        self.num_features = len(self.train_graphs[0].nodes()[0]['features'])
        self.num_classes = obj["n_classes"]
        self.multilabel = obj["multilabel"]
        random.shuffle(self.train_graphs)
        self.cur_train_idx = 0
        self.cur_val_idx = 0
        self.cur_test_idx = 0

    def get_next_val(self):
        g = self.val_graphs[self.cur_val_idx]
        self.cur_val_idx = (self.cur_val_idx + 1) % len(self.val_graphs)
        features = np.array([g.nodes()[x]['features'] for x in g.nodes()])
        node_labels = np.array([g.nodes()[x]['label'] for x in g.nodes()], dtype=np.float32)
        edge_index = torch.tensor(list(g.edges())).t().contiguous() 
        features = torch.FloatTensor(features)
        labels = torch.FloatTensor(node_labels)
        if not self.multilabel and len(labels.shape) > 1:
            labels = labels.argmax(dim=1)
        adj = nx.to_numpy_matrix(g)
        src, trg = adj.nonzero()
        edge_index = torch.LongTensor(np.concatenate([src.reshape(1, -1), trg.reshape(1,-1)], axis=0))
        return edge_index, features, labels, adj, g

    def get_next_test(self):
        g = self.test_graphs[self.cur_test_idx]
        self.cur_test_idx = (self.cur_test_idx + 1) % len(self.val_graphs)
        features = np.array([g.nodes()[x]['features'] for x in g.nodes()])
        node_labels = np.array([g.nodes()[x]['label'] for x in g.nodes()], dtype=np.float32)
        edge_index = torch.tensor(list(g.edges())).t().contiguous() 
        features = torch.FloatTensor(features)
        labels = torch.FloatTensor(node_labels)
        if not self.multilabel and len(labels.shape) > 1:
            labels = labels.argmax(dim=1)
        adj = nx.to_numpy_matrix(g)
        src, trg = adj.nonzero()
        edge_index = torch.LongTensor(np.concatenate([src.reshape(1, -1), trg.reshape(1,-1)], axis=0))
        return edge_index, features, labels, adj, g

    def get_next_train(self):
        if self.cur_train_idx == 0:
            random.shuffle(self.train_graphs)
        g = self.train_graphs[self.cur_train_idx]
        self.cur_train_idx = (self.cur_train_idx + 1) % len(self.train_graphs)
        features = np.array([g.nodes()[x]['features'] for x in g.nodes()])
        node_labels = np.array([g.nodes()[x]['label'] for x in g.nodes()], dtype=np.float32)
        # edge_index = torch.tensor(list(g.edges())).t().contiguous() 
        features = torch.FloatTensor(features)
        labels = torch.FloatTensor(node_labels)
        if not self.multilabel and len(labels.shape) > 1:
            labels = labels.argmax(dim=1)
        adj = nx.to_numpy_matrix(g)
        src, trg = adj.nonzero()
        edge_index = torch.LongTensor(np.concatenate([src.reshape(1, -1), trg.reshape(1,-1)], axis=0))
        # adj = torch.FloatTensor(np.zeros((len(features), len(features))))
        # adj[edge_index[0], edge_index[1]] = 1
        return edge_index, features, labels, None, g

    def get_graph(self, g):
        features = np.array([g.nodes()[x]['features'] for x in g.nodes()])
        node_labels = np.array([g.nodes()[x]['label'] for x in g.nodes()], dtype=np.float32)
        edge_index = torch.tensor(list(g.edges())).t().contiguous() 
        features = torch.FloatTensor(features)
        labels = torch.FloatTensor(node_labels)
        if not self.multilabel and len(labels.shape) > 1:
            labels = labels.argmax(dim=1)
        adj = torch.FloatTensor(np.zeros((len(features), len(features))))
        adj[edge_index[0], edge_index[1]] = 1
        return edge_index, features, labels, adj

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", default='data/ppi.pkl', type=str)
parser.add_argument("--epochs", default=300, type=int)
parser.add_argument("--hidden", default=32, type=int)
parser.add_argument("--gnn", default="mean", choices=["mean", "gcn", "gat", "sgc"])
parser.add_argument("--batch-size", default=64, type=int)
parser.add_argument("--seed", default=100, type=int)
args = parser.parse_args()
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

print(args)

dataset = Dataset()
num_features = dataset.num_features
num_classes = dataset.num_classes
print("Num features: ", dataset.num_features)
print("Num classes:", dataset.num_classes)


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
        return x

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, args.hidden, cached=False, normalize=True)
        self.conv2 = GCNConv(args.hidden, args.hidden*2, cached=False, normalize=True)
        self.conv3 = GCNConv(args.hidden*2, num_classes, cached=False, normalize=True)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        return x

class SGC(torch.nn.Module):
    def __init__(self):
        super(SGC, self).__init__()
        self.conv1 = SGConv(num_features, num_classes, K=2, cached=False)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index)
        return x

def train():
    model.train()
    n_iter = int(np.ceil(len(dataset.train_graphs) / args.batch_size))
    total_loss = 0
    for iter in range(n_iter):
        optimizer.zero_grad()
        loss = 0
        for i in range(args.batch_size):
            edge_index, x, y, adj, _ = dataset.get_next_train()
            edge_index = edge_index.to(device)
            x = x.to(device)
            y = y.to(device)
            logits = model(x, edge_index)
            loss += criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss
    return total_loss/n_iter

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
        probs = (output > 0).float()
        probs = probs.cpu().detach().numpy().astype(np.int32)
        labels = labels.cpu().detach().numpy().astype(np.int32)
        micro = f1_score(labels, probs, average='micro')
        macro = f1_score(labels, probs, average='macro')
        return micro, macro

def test(generator_fn, n_test=2):
    model.eval()
    with torch.no_grad():
        logitss = []
        ys = []

        for iter in range(n_test):
            edge_index, x, y, adj, _ = generator_fn()
            edge_index = edge_index.to(device)
            x = x.to(device)
            y = y.to(device)
            logits = model(x, edge_index)
            logitss.append(logits)
            ys.append(y)
        logitss = torch.cat(logitss, dim=0)
        ys = torch.cat(ys, dim=0)
        micro, macro = f1(logitss, ys, multiclass=dataset.multilabel)
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
if args.gnn == "mean":
    model = GraphsageMEAN().to(device)
elif args.gnn == "gcn":
    model = GCN().to(device)
elif args.gnn == "gat":
    model = GATDense().to(device)
elif args.gnn == "sgc":
    model = SGC().to(device)

print(model)
lr = 0.005

if dataset.multilabel:
    print("Train multilabel")
    criterion = torch.nn.BCEWithLogitsLoss()
else:
    print("Train one-label")
    criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

best_val_acc = 0
epochs = args.epochs
if not os.path.isdir("model"):
    os.makedirs("model")
model_name = f"{getfilename(args.data_path)}-normal-{args.gnn}-{time.time()}"
model_path = f"model/{model_name}-1.pkl"
writer = tensorboardX.SummaryWriter(logdir=f"runs/{model_name}")

train_acc, train_macro = test(dataset.get_next_train, n_test=64)
val_acc, val_macro = test(dataset.get_next_val, n_test=64)
log = 'Epoch: 0, micro-macro Train: {:.4f}-{:.4f}, Val: {:.4f}-{:.4f}'
print(log.format(train_acc, train_macro, val_acc, val_macro))

best_val_acc = val_acc
for epoch in range(1, epochs + 1):
    stime = time.time()
    loss = train()
    etime = time.time() - stime

    if epoch % 20 == 0 or epoch == epochs:
        train_acc, train_macro = test(dataset.get_next_train, n_test=64)
        val_acc, val_macro = test(dataset.get_next_val, n_test=len(dataset.val_graphs))
        log = 'Epoch: {:03d}, loss: {:4f}, micro-macro Train: {:.4f}-{:.4f}, Val: {:.4f}-{:.4f} - time {:.3f}'
        print(log.format(epoch, loss, train_acc, train_macro, val_acc, val_macro, etime))
        writer.add_scalar("loss", loss, epoch)
        writer.add_scalar("node_train_f1_micro", train_acc, epoch)
        writer.add_scalar("node_train_f1_macro", train_macro, epoch)
        writer.add_scalar("node_val_f1_micro", val_acc, epoch)
        writer.add_scalar("node_val_f1_macro", val_macro, epoch)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)

model.load_state_dict(torch.load(model_path))
test_acc, test_macro = test(dataset.get_next_test, n_test=len(dataset.test_graphs))
log = 'Test: {:.4f}-{:.4f}'
print(log.format(test_acc, test_macro))

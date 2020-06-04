import pdb
import torch
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, GATConv, DenseGCNConv, SAGEConv
import argparse
import os
from sklearn.metrics import f1_score
import numpy as np
from sage import SAGECompletedGraph
import torch.nn as nn
import re
import pickle
import json
import networkx as nx
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
        self.val_graphs = train_graphs[:n_val]
        self.train_graphs = train_graphs[n_val:]
        self.test_graphs = test_graphs
        self.num_features = len(self.train_graphs[0].nodes()[0]['features'])
        self.num_classes = obj["n_classes"]
        self.multilabel = obj["multilabel"]
        self.inds_train = np.random.permutation(len(self.train_graphs)) # keep original order
        self.cur_train_idx = 0
        self.cur_val_idx = 0
        self.cur_test_idx = 0

    def get_next_val(self):
        g = self.val_graphs[self.cur_val_idx]
        self.cur_val_idx = (self.cur_val_idx + 1) % len(self.val_graphs)
        features = np.array([g.nodes()[x]['features'] for x in g.nodes()])
        node_labels = np.array([g.nodes()[x]['label'] for x in g.nodes()], dtype=np.float32)
        features = torch.FloatTensor(features)
        labels = torch.FloatTensor(node_labels)
        if not self.multilabel and len(labels.shape) > 1:
            labels = labels.argmax(dim=1)
        adj = nx.to_numpy_matrix(g)
        src, trg = adj.nonzero()
        edge_index = torch.LongTensor(np.concatenate([src.reshape(1, -1), trg.reshape(1,-1)], axis=0))
        adj = torch.FloatTensor(adj)
        return edge_index, features, labels, adj, g

    def get_next_test(self):
        g = self.test_graphs[self.cur_test_idx]
        self.cur_test_idx = (self.cur_test_idx + 1) % len(self.val_graphs)
        features = np.array([g.nodes()[x]['features'] for x in g.nodes()])
        node_labels = np.array([g.nodes()[x]['label'] for x in g.nodes()], dtype=np.float32)
        features = torch.FloatTensor(features)
        labels = torch.FloatTensor(node_labels)
        if not self.multilabel and len(labels.shape) > 1:
            labels = labels.argmax(dim=1)
        adj = nx.to_numpy_matrix(g)
        src, trg = adj.nonzero()
        edge_index = torch.LongTensor(np.concatenate([src.reshape(1, -1), trg.reshape(1,-1)], axis=0))
        adj = torch.FloatTensor(adj)
        return edge_index, features, labels, adj, g

    def get_next_train(self):
        if self.cur_train_idx == 0:
            np.random.shuffle(self.inds_train)
        g = self.train_graphs[self.inds_train[self.cur_train_idx]]
        self.cur_train_idx = (self.cur_train_idx + 1) % len(self.train_graphs)
        features = np.array([g.nodes()[x]['features'] for x in g.nodes()])
        node_labels = np.array([g.nodes()[x]['label'] for x in g.nodes()], dtype=np.float32)
        features = torch.FloatTensor(features)
        labels = torch.FloatTensor(node_labels)
        if not self.multilabel and len(labels.shape) > 1:
            labels = labels.argmax(dim=1)
        adj = nx.to_numpy_matrix(g)
        src, trg = adj.nonzero()
        edge_index = torch.LongTensor(np.concatenate([src.reshape(1, -1), trg.reshape(1,-1)], axis=0))
        adj = torch.FloatTensor(adj)
        return edge_index, features, labels, adj, g

    def get_graph(self, g):
        features = np.array([g.nodes()[x]['features'] for x in g.nodes()])
        node_labels = np.array([g.nodes()[x]['label'] for x in g.nodes()], dtype=np.float32)
        features = torch.FloatTensor(features)
        labels = torch.FloatTensor(node_labels)
        if not self.multilabel and len(labels.shape) > 1:
            labels = labels.argmax(dim=1)
        adj = nx.to_numpy_matrix(g)
        assert (adj>1).sum() == 0 and (adj<0).sum() == 0, "adj error"
        src, trg = adj.nonzero()
        edge_index = torch.LongTensor(np.concatenate([src.reshape(1, -1), trg.reshape(1,-1)], axis=0))
        adj = torch.FloatTensor(adj)
        return edge_index, features, labels, adj

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", default='data/ppi.pkl', type=str)
parser.add_argument("--ckpt-dir", default='ckpt-ppi', type=str)
parser.add_argument("--epochs", default=300, type=int)
parser.add_argument("--hidden", default=256, type=int)
parser.add_argument("--embedding_dim", default=256, type=int)
parser.add_argument("--gnn1", default="gat", choices=["gat", "mean"])
parser.add_argument("--gnn2", default="mean", choices=["mean", "gcn", "gat", "sgc"])
parser.add_argument("--n-layers", default=3, type=int)
parser.add_argument("--batch-size", default=64, type=int)
parser.add_argument("--seed", default=100, type=int)
parser.add_argument("--n-heads", default=4, type=int)
parser.add_argument("--adj-weight", default=1, type=float)
parser.add_argument("--classify-weight", default=1, type=float)
parser.add_argument("--adj-func", default='scaled_hacky_sigmoid_l2',
    choices=['scaled_hacky_sigmoid_l2', 'sigmoid_dot', 'scaled_sigmoid_absolute'])
parser.add_argument("--C", default=10, type=float)

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


class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.num_layers = args.n_layers
        self.conv1 = GATConv(dataset.num_features, args.hidden, heads=args.n_heads)
        self.lin1 = torch.nn.Linear(num_features, args.n_heads * args.hidden)
        self.middle_convs = nn.Sequential(*[
            GATConv(args.n_heads * args.hidden, args.hidden, heads=args.n_heads)
            for _ in range(self.num_layers - 2)])
        self.middle_lins = nn.Sequential(*[
            torch.nn.Linear(args.n_heads * args.hidden, args.n_heads * args.hidden)
            for _ in range(self.num_layers - 2)])
        self.conv3 = GATConv(args.n_heads * args.hidden, args.embedding_dim, heads=args.n_heads+2, concat=False)
        self.lin3 = torch.nn.Linear(args.n_heads * args.hidden, args.embedding_dim)
        
    def forward(self, x, edge_index, edge_attr=None):
        x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
        for conv, lin in zip(self.middle_convs, self.middle_lins):
            x = F.elu(conv(x, edge_index) + lin(x))
        x = self.conv3(x, edge_index) + self.lin3(x)
        return x

class GraphsageMEAN(torch.nn.Module):
    def __init__(self):
        super(GraphsageMEAN, self).__init__()
        self.conv1 = SAGEConv(num_features, args.hidden, normalize=False)
        self.conv2 = SAGEConv(args.hidden, args.hidden * 2, normalize=False)
        self.conv3 = SAGEConv(args.hidden * 2, args.embedding_dim, normalize=False)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        return x

class GraphsageMEANCompleted(torch.nn.Module):
    def __init__(self):
        super(GraphsageMEANCompleted, self).__init__()
        self.conv1 = SAGECompletedGraph (num_features, args.hidden, normalize=False)
        self.conv2 = SAGECompletedGraph(args.hidden, args.hidden * 2, normalize=False)
        self.conv3 = SAGECompletedGraph(args.hidden * 2, num_classes, normalize=False)

    def forward(self, x, adj):
        x = self.conv1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, adj)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, adj)
        return x

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = DenseGCNConv(num_features, args.hidden)
        self.conv2 = DenseGCNConv(args.hidden, args.hidden*2)
        self.conv3 = DenseGCNConv(args.hidden*2, num_classes)

    def forward(self, x, adj):
        x = self.conv1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, adj)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, adj)
        return x

class SGC(nn.Module):
    def __init__(self, degree=2):
        super(SGC, self).__init__()
        self.W = nn.Linear(num_features, num_classes)
        self.degree = degree

    def forward(self, x, adj):
        adj = self.aug_normalized_adjacency(adj)
        x = self.sgc_precompute(x, adj, self.degree)
        x = self.W(x)
        return x.unsqueeze(0)

    def row_normalize(self, x):
        rowsum = x.sum(dim=1)
        r_inv = 1 / rowsum
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        x = r_mat_inv.mm(x)
        return x

    def sgc_precompute(self, features, adj, degree):
        for i in range(degree):
            features = torch.mm(adj, features)
        return features

    def aug_normalized_adjacency(self, adj):
        adj = adj + torch.eye(adj.shape[0]).to(device)
        rowsum = adj.sum(dim=1)
        d_inv_sqrt = rowsum.pow(-0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        return d_mat_inv_sqrt.mm(adj).mm(d_mat_inv_sqrt)

def get_upper_triangle_matrix_except_diagonal(matrix):
    inds = torch.LongTensor(np.triu_indices(len(matrix), 1)).to(device)
    return matrix[inds[0], inds[1]]

def scaled_hacky_sigmoid_l2(embeddings):
    r = torch.sum(embeddings**2, dim=1).view(-1, 1)
    D = r - 2*embeddings.mm(embeddings.t()) + r.t()
    D /= np.sqrt(embeddings.shape[1])
    C = args.C
    adj = torch.sigmoid(C * (1-D))
    return adj

def sigmoid_dot(embeddings):
    dot = embeddings.mm(embeddings.t())
    # dot = dot * (1 - torch.eye(embeddings.shape[0]))
    return torch.sigmoid(dot - 0.5)

def scaled_sigmoid_absolute(embeddings):
    D = torch.norm(embeddings[:,None] - embeddings, dim=2, p=2)
    c = 10
    adj = torch.sigmoid(c * (1-D))
    return adj

def pred_adj(embeddings, upper_triangle=False):
    if args.adj_func == 'scaled_hacky_sigmoid_l2':
        adj = scaled_hacky_sigmoid_l2(embeddings)
    elif args.adj_func == 'sigmoid_dot':
        adj = sigmoid_dot(embeddings)
    elif args.adj_func == 'scaled_sigmoid_absolute':
        adj = scaled_sigmoid_absolute(embeddings)
    if upper_triangle:
        adj = get_upper_triangle_matrix_except_diagonal(adj)
    return adj

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
    assert edges.shape[0] == 2
    return edges, xs, ys

def train():
    model.train()
    model2.train()
    n_iter = int(np.ceil(len(dataset.train_graphs) / args.batch_size))
    total_loss = 0
    total_loss_adj = 0
    total_loss_classify = 0
    for iter in range(n_iter):
        optimizer.zero_grad()
        loss = 0
        for i in range(args.batch_size):
            edge_index, x, y, adj, _ = dataset.get_next_train()
            edge_index = edge_index.to(device)
            adj = adj.to(device)
            x = x.to(device)
            y = y.to(device)
            embeddings = model(x, edge_index)
            p_adj = pred_adj(embeddings, upper_triangle=False)
            loss_adj = criterion_adj(get_upper_triangle_matrix_except_diagonal(p_adj), 
                get_upper_triangle_matrix_except_diagonal(adj))
            logits = model2(x, p_adj)[0]
            loss_classify = criterion(logits, y)
            loss += args.adj_weight * loss_adj + args.classify_weight * loss_classify
            total_loss_adj += loss_adj*args.adj_weight
            total_loss_classify += loss_classify*args.classify_weight
        loss.backward()
        optimizer.step()
        total_loss += loss
    return total_loss/n_iter, total_loss_adj/n_iter, total_loss_classify/n_iter

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

def f1_adj(padj, adj):
    padj[padj >= 0.5] = 1
    padj[padj < 0.5] = 0
    padj = padj.cpu().detach().numpy().astype(np.int32)
    adj = adj.cpu().detach().numpy().astype(np.int32)
    micro = f1_score(adj, padj, average='micro')
    macro = f1_score(adj, padj, average='macro')
    return micro, macro


def gen_embeddings(suffix=""):
    model.eval()
    model2.eval()
    with torch.no_grad():
        train_graphs_networkx = []
        train_graphs_gen = []
        for g in (dataset.train_graphs + dataset.val_graphs):
            edge_index, x, y, adj = dataset.get_graph(g)
            edge_index = edge_index.to(device)
            adj = adj.to(device)
            x = x.to(device)
            y = y.to(device)
            embeddings = model(x, edge_index)
            p_adj = pred_adj(embeddings)
            p_labels = model2(x, p_adj)[0].argmax(dim=1).cpu().numpy()
            p_adj = p_adj.cpu().numpy()
            g_pred = nx.from_numpy_matrix(p_adj)
            embeddings = embeddings.detach().cpu().numpy()
            for node in g.nodes():
                g.nodes()[node]['embedding'] = embeddings[node]
                g_pred.nodes()[node]['p_label'] = p_labels[node]
            train_graphs_networkx.append(g)
            train_graphs_gen.append(g_pred)
        test_graphs_networkx = []
        test_graphs_gen = []
        for g in dataset.test_graphs:
            edge_index, x, y, adj = dataset.get_graph(g)
            edge_index = edge_index.to(device)
            adj = adj.to(device)
            x = x.to(device)
            y = y.to(device)
            embeddings = model(x, edge_index)
            p_adj = pred_adj(embeddings)
            p_labels = model2(x, p_adj)[0].argmax(dim=1).cpu().numpy()
            p_adj = p_adj.cpu().numpy()
            g_pred = nx.from_numpy_matrix(p_adj)
            embeddings = embeddings.detach().cpu().numpy()
            for node in g.nodes():
                g.nodes()[node]['embedding'] = embeddings[node]
                g_pred.nodes()[node]['p_label'] = p_labels[node]
            test_graphs_networkx.append(g)
            test_graphs_gen.append(g_pred)
        pickle.dump({
            "train_graphs": train_graphs_networkx, 
            "test_graphs": test_graphs_networkx,
            "train_graphs_gen": train_graphs_gen,
            "test_graphs_gen": test_graphs_gen,
            "gnn1": model.state_dict(),
            "gnn2": model2.state_dict()
            }, open(f"{args.ckpt_dir}/embeddings-{suffix}.pkl", "wb")) 

def test(generator_fn, n_test=2):
    model.eval()
    model2.eval()
    with torch.no_grad():
        logitss = []
        ys = []
        f1_adjs = []

        for iter in range(n_test):
            edge_index, x, y, adj, _ = generator_fn()
            edge_index = edge_index.to(device)
            adj = adj.to(device)
            x = x.to(device)
            y = y.to(device)
            embeddings = model(x, edge_index)
            p_adj = pred_adj(embeddings)
            logits = model2(x, p_adj)[0]
            logitss.append(logits)
            ys.append(y)
            micro_adj, macro_adj = f1_adj(p_adj, adj)
            f1_adjs.append((micro_adj, macro_adj))
        logitss = torch.cat(logitss, dim=0)
        ys = torch.cat(ys, dim=0)
        micro, macro = f1(logitss, ys, multiclass=dataset.multilabel)

        f1_adj_micros, f1_adj_macros = zip(*f1_adjs)
        f1_adj_micro, f1_adj_macro = np.mean(f1_adj_micros), np.mean(f1_adj_macros)

        return micro, macro, f1_adj_micro, f1_adj_macro


def getdirname(path):
    if path is None:
        return None
    return [x for x in path.split("/") if len(x) > 0][-1]


def getfilename(path):
    if path is None:
        return None
    return path.split("/")[-1].split(".")[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if args.gnn1 == "gat":
    model = GAT().to(device)
elif args.gnn1 == "mean":
    model = GraphsageMEAN().to(device)
if args.gnn2 == "mean":
    model2 = GraphsageMEANCompleted().to(device)
elif args.gnn2 == "gcn":
    model2 = GCN().to(device)
elif args.gnn2 == "gat":
    model2 = GATDense().to(device)
elif args.gnn2 == "sgc":
    model2 = SGC().to(device)

print(model)
print(model2)
lr = 0.001
# lr = 0.005

criterion_adj = torch.nn.BCELoss()
if dataset.multilabel:
    print("Train multilabel")
    criterion = torch.nn.BCEWithLogitsLoss()
else:
    print("Train one-label")
    criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(chain(model.parameters(), model2.parameters()), lr=lr, weight_decay=5e-4)

best_val_acc = 0
best_model = None
epochs = args.epochs
if not os.path.isdir("model"):
    os.makedirs("model")
if not os.path.isdir(args.ckpt_dir):
    os.makedirs(args.ckpt_dir)
model_name = f"{getfilename(args.data_path)}-{args.gnn1}-{args.gnn2}-{time.time()}"
model1_path = f"model/{model_name}-1.pkl"
model2_path = f"model/{model_name}-2.pkl"
writer = tensorboardX.SummaryWriter(logdir=f"runs/{model_name}")

train_acc, train_macro, train_f1_adj_micro, train_f1_adj_macro = test(dataset.get_next_train, n_test=64)
val_acc, val_macro, val_f1_adj_micro, val_f1_adj_macro = test(dataset.get_next_val, n_test=64)
log = 'Epoch: 0, micro-macro Train: {:.4f}-{:.4f}, Val: {:.4f}-{:.4f} - Adj Train: {:.4f}-{:.4f}, Val: {:.4f}-{:.4f}'
print(log.format(train_acc, train_macro, val_acc, val_macro, 
    train_f1_adj_micro, train_f1_adj_macro, val_f1_adj_micro, val_f1_adj_macro))

best_val_acc = val_acc
for epoch in range(1, epochs + 1):
    stime = time.time()
    loss, loss_adj, loss_classify = train()
    etime = time.time() - stime

    if epoch % 20 == 0 or epoch == epochs:
        train_acc, train_macro, train_f1_adj_micro, train_f1_adj_macro = test(dataset.get_next_train, n_test=64)
        val_acc, val_macro, val_f1_adj_micro, val_f1_adj_macro = test(dataset.get_next_val, n_test=len(dataset.val_graphs))
        log = 'Epoch: {:03d}, loss: {:4f}, micro-macro Train: {:.4f}-{:.4f}, Val: {:.4f}-{:.4f} - Adj Train: {:.4f}-{:.4f}, Val: {:.4f}-{:.4f} - time {:.3f}'
        print(log.format(epoch, loss, train_acc, train_macro, val_acc, val_macro, 
            train_f1_adj_micro, train_f1_adj_macro, val_f1_adj_micro, val_f1_adj_macro, etime))
        writer.add_scalar("loss", loss, epoch)
        writer.add_scalar("loss_adj", loss_adj, epoch)
        writer.add_scalar("loss_classify", loss_classify, epoch)
        writer.add_scalar("node_train_f1_micro", train_acc, epoch)
        writer.add_scalar("node_train_f1_macro", train_macro, epoch)
        writer.add_scalar("node_val_f1_micro", val_acc, epoch)
        writer.add_scalar("node_val_f1_macro", val_macro, epoch)
        writer.add_scalar("adj_train_f1_micro", train_f1_adj_micro, epoch)
        writer.add_scalar("adj_train_f1_macro", train_f1_adj_macro, epoch)
        writer.add_scalar("adj_val_f1_micro", val_f1_adj_micro, epoch)
        writer.add_scalar("adj_val_f1_macro", val_f1_adj_macro, epoch)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model1_path)
            torch.save(model2.state_dict(), model2_path)
    if epoch % 100 == 0:
        gen_embeddings(epoch)

model.load_state_dict(torch.load(model1_path))
model2.load_state_dict(torch.load(model2_path))
test_acc, test_macro, test_f1_adj_micro, test_f1_adj_macro = test(dataset.get_next_test, n_test=len(dataset.test_graphs))
log = 'Test micro-macro Node: {:.4f}-{:.4f} Adj: {:.4f}-{:.4f}'
print(log.format(test_acc, test_macro, test_f1_adj_micro, test_f1_adj_macro))
gen_embeddings("best")

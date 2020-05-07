import os.path as osp
from math import ceil
import random
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader, DataListLoader, Data
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
from torch_geometric.nn import GINConv, global_mean_pool, JumpingKnowledge
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
import torch.nn as nn
import numpy as np
from types import SimpleNamespace
import argparse

class CustomDataLoader():
    def __init__(self, datalist, batch_size=20, shuffle=False):
        self.datalist = datalist
        self.cur_idx = 0
        self.batch_size = batch_size
        self.shuffle = shuffle
        if self.shuffle:
            random.shuffle(self.datalist)
    def __len__(self):
        return len(self.datalist)
    def reset_idx(self):
        self.cur_idx = 0
    def get_graphs(self, datalist):
        edges = []
        adjs = []
        xs = []
        ys = []
        batch = []
        inc = 0
        for i, data in enumerate(datalist):
            edge_index = data.edge_index + inc
            x = data.x
            y = data.y
            inc += data.num_nodes
            adj = torch.zeros((data.num_nodes, data.num_nodes))
            adj[data.edge_index[0], data.edge_index[1]] = 1
            adjs.append(adj)
            edges.append(edge_index)
            xs.append(x)
            ys.append(y)
            batch += [i]*data.num_nodes
        edges = torch.cat(edges, dim=1)
        xs = torch.cat(xs, dim=0)
        ys = torch.cat(ys, dim=0)
        batch = torch.LongTensor(batch)
        adjs = torch.stack(adjs)
        return xs, ys, edges, adjs, batch

    def get_next_batch(self):
        if self.cur_idx + self.batch_size + 1 > len(self.datalist):
            if self.shuffle:
                random.shuffle(self.datalist)
            self.cur_idx = 0
        datalist = self.datalist[self.cur_idx:self.cur_idx+self.batch_size]
        self.cur_idx = (self.cur_idx+self.batch_size) % len(self.datalist)
        edges = []
        adjs = []
        xs = []
        ys = []
        batch = []
        inc = 0
        for i, data in enumerate(datalist):
            edge_index = data.edge_index + inc
            x = data.x
            y = data.y
            inc += data.num_nodes
            adj = torch.zeros((data.num_nodes, data.num_nodes))
            adj[data.edge_index[0], data.edge_index[1]] = 1
            adjs.append(adj)
            edges.append(edge_index)
            xs.append(x)
            ys.append(y)
            batch += [i]*data.num_nodes
        edges = torch.cat(edges, dim=1)
        xs = torch.cat(xs, dim=0)
        ys = torch.cat(ys, dim=0)
        batch = torch.LongTensor(batch)
        adjs = torch.stack(adjs)
        return xs, ys, edges, adjs, batch

class GIN(torch.nn.Module):
    def __init__(self, num_layers, hidden):
        super(GIN, self).__init__()
        features_dim = args.features_dim
        self.features_net = nn.Sequential(
            nn.Linear(max_nodes, features_dim*2),
            nn.ReLU(),
            nn.Linear(features_dim*2, features_dim)
        )
        self.conv1 = GINConv(Sequential(
            Linear(features_dim, hidden),
            ReLU(),
            Linear(hidden, hidden),
            ReLU(),
            BN(hidden),
        ), train_eps=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(Sequential(
                    Linear(hidden, hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                    ReLU(),
                    BN(hidden),
                ),
                        train_eps=True))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, batch):
        x = self.features_net(x)
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        node_embs = x
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return node_embs, F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

def get_upper_triangle_matrix_except_diagonal(matrix):
    inds = np.triu_indices(len(matrix), 1)
    return matrix[inds[0], inds[1]]

def pred_adj(all_embeddings, batch, upper_triangle=False): # upper triangle except diagonal
    graph_ids = np.unique(batch.cpu().numpy())
    pred_adjs = []
    for id in graph_ids:
        embeddings = all_embeddings[batch == id]
        r = torch.sum(embeddings**2, dim=1).view(-1, 1)
        D = r - 2*embeddings.mm(embeddings.t()) + r.t()
        D /= np.sqrt(embeddings.shape[1])
        r = 10
        adj = torch.sigmoid(r * (1-D))
        if upper_diagonal:
            adj = get_upper_triangle_matrix_except_diagonal(adj)
        pred_adjs.append(adj)
    return torch.stack(pred_adjs)

def train(epoch):
    model.train()
    loss_all = 0

    n_iters = int(np.ceil(len(train_loader) / args.batch_size))
    for iter in range(n_iters):
        x, y, edge_index, adj, batch = train_loader.get_next_batch()
        x = x.to(device)
        y = y.to(device)
        adj = adj.to(device)
        edge_index = edge_index.to(device)
        batch = batch.to(device)
        optimizer.zero_grad()
        embeddings, x = model(x, edge_index, batch)
        p_adj = pred_adj(embeddings, batch, upper_triangle=True)
        loss_adj = criterion_adj(p_adj, get_upper_triangle_matrix_except_diagonal(adj))
        loss_classify = F.nll_loss(x, y.view(-1))
        loss = loss_adj * args.adj_weight + loss_classify*args.classify_weight 
        loss.backward()
        loss_all += y.size(0) * loss.item()
        optimizer.step()
    return loss_all / len(train_loader)

@torch.no_grad()
def test(loader):
    model.eval()
    correct = 0

    n_iters = int(np.ceil(len(loader) / args.batch_size))
    for iter in range(n_iters):
        batch_datalist = loader.datalist[iter*args.batch_size:(iter+1)*args.batch_size]
        x, y, edge_index, adj, batch = loader.get_graphs(batch_datalist)
        x = x.to(device)
        y = y.to(device)
        adj = adj.to(device)
        edge_index = edge_index.to(device)
        batch = batch.to(device)
        embeddings, pred = model(x, edge_index, batch)
        pred = pred.max(dim=1)[1]
        assert pred.shape[0] == y.shape[0], "wrong shape"
        correct += (pred == y).sum().item()
    return correct / len(loader)

parser = argparse.ArgumentParser()
parser.add_argument('--features-dim', default=128, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--name', default='IMDB-BINARY')
parser.add_argument('--adj-weight', default=1, type=float)
parser.add_argument('--classify-weight', default=1, type=float)
parser.add_argument('--hidden', default=64, type=int)
parser.add_argument('--seed', default=100, type=int)
args = parser.parse_args()
print(args)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)

model_name = f"model/{args.name}.pkl"
dataset = TUDataset(f"data/{args.name}", name=args.name)
max_nodes = 0
for data in dataset:
    max_nodes = max(max_nodes, data.num_nodes)
print("Max num nodes: ", max_nodes)
dataset.transform = T.ToDense(max_nodes)
dataset = dataset.shuffle()
num_classes = dataset.num_classes

prep_dataset = []
for data in dataset:
    src, trg = torch.where(data.adj)
    edge_index = torch.cat([src.view(1,-1), trg.view(1,-1)], dim=0)
    new_data = Data(x=data.adj, y=data.y, edge_index=edge_index)
    data.x = data.adj
    prep_dataset.append(new_data)
dataset = prep_dataset

test_dataset = dataset[:len(dataset) // 10]
train_dataset = dataset[len(dataset) // 10:]
train_loader = CustomDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = CustomDataLoader(test_dataset, batch_size=args.batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GIN(5, args.hidden).to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion_adj = torch.nn.BCELoss()


for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)
    if epoch% 20 == 0:
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        print('Epoch: {:03d}, Train Loss: {:.3f}, Train Acc: {:.3f}, Test Acc: {:.3f}'.format(
            epoch, train_loss, train_acc, test_acc))

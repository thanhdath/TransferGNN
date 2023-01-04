import os.path as osp
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader, Data, DataListLoader
from torch_geometric.nn import GINConv, global_add_pool, GCNConv, DenseGraphConv, dense_mincut_pool, dense_diff_pool
import networkx as nx
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix, vstack, hstack
import numpy as np
import torch.nn as nn
import argparse
from math import ceil
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from transfers.utils import gen_graph, generate_graph

from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_tu_data
from networkx.generators.random_graphs import connected_watts_strogatz_graph

def halfA_to_A(halfA):
    A = np.zeros((len(X), len(X)))
    inds = torch.triu(torch.ones(len(A), len(A)))
    inds[np.arange(len(A)), np.arange(len(A))] = 0
    A[inds == 1] = halfA
    A = A + A.T
    # A[np.arange(len(A)), np.arange(len(A))] = 1
    return A

def remove_self_loops(edge_index):
    row, col = edge_index
    mask = row != col
    edge_index = edge_index[:, mask]
    return edge_index

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='ENZYMES')
parser.add_argument('--f', default="ori", choices=["ori", "random", "knn", "sigmoid"])
parser.add_argument('--hidden', default=32, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--seed', default=100, type=int)
args = parser.parse_args()
print(args)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

name = args.data

path = f"data/{name}"
dataset = TUDataset(path, name=name, use_node_attr=True).shuffle()
num_classes = dataset.num_classes
num_features = dataset.num_features

test_dataset = dataset[:len(dataset) // 10]
train_dataset = dataset[len(dataset) // 10:]

# gen adj on test dataset
if args.f != "ori":
    converted_test_dataset = []
    for data in test_dataset:
        if data.num_nodes == 1:
            converted_test_dataset.append(data)
            continue
        X = data.x
        k = data.edge_index.shape[1] // data.num_nodes
        if args.f == "knn":
            adj = generate_graph(data.x, kind="knn", k=k)
        if args.f == "sigmoid":
            adj = generate_graph(torch.FloatTensor(X), kind="sigmoid", k=k)
        if args.f == "random":
            G = connected_watts_strogatz_graph(data.num_nodes, k+1, p=0.1)
            adj = nx.to_numpy_matrix(G)
        
        src, trg = adj.nonzero()
        edge_index = torch.LongTensor(np.concatenate([src.reshape(1, -1), trg.reshape(1,-1)], axis=0))
        print(f"N edges before {data.edge_index.shape[1]} - N edges then {edge_index.shape[1]} ")
        new_data = Data(x=data.x, y=data.y, edge_index=edge_index)
        converted_test_dataset.append(new_data)
else:
    converted_test_dataset = test_dataset

test_loader = DataListLoader(converted_test_dataset, batch_size=1)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        dim = args.hidden
        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()

    if epoch == 51:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_dataset)


def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        if isinstance(data, list):
            data = data[0]
            batch = torch.zeros((data.num_nodes), dtype=torch.long).to(device)
        else:
            batch = data.batch.to(device)
        # print(edge_index)
        data = data.to(device)
        output = model(data.x, data.edge_index, batch)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)
    if epoch % 20 == 0:
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        print('Epoch: {:03d}, Train Loss: {:.7f}, '
            'Train Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss,
                                                        train_acc, test_acc))

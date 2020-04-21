import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader, Data
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
from torch_geometric.datasets.mnist_superpixels import MNISTSuperpixels

parser = argparse.ArgumentParser()
parser.add_argument('--init', default='real')
parser.add_argument('--feature_dim', default=128, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--seed', default=100, type=int)
args = parser.parse_args()
print(args)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

init = args.init  # degree, kcore
train_dataset = MNISTSuperpixels("data/MNISTSuper", train=True)
test_dataset = MNISTSuperpixels("data/MNISTSuper", train=False)
num_classes = train_dataset.num_classes
num_features = train_dataset.num_features
test_loader = DataLoader(test_dataset, batch_size=128)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)


def svd_features(graph, dim_size=128, alpha=0.5):
    adj = nx.to_scipy_sparse_matrix(graph, dtype=np.float32)
    if adj.shape[0] <= dim_size:
        padding_row = csr_matrix(
            (dim_size - adj.shape[0] + 1, adj.shape[1]), dtype=adj.dtype)
        adj = vstack((adj, padding_row))
        padding_col = csr_matrix(
            (adj.shape[0], dim_size - adj.shape[1] + 1), dtype=adj.dtype)
        adj = hstack((adj, padding_col))

    U, X, _ = svds(adj, k=dim_size)
    embedding = U * (X**(alpha))
    features = {node: embedding[i] for i, node in enumerate(graph.nodes())}
    return features


def degree_features(graph, dim_size=128):
    prep_dict = {}
    for idx, node in enumerate(graph.nodes()):
        prep_dict[node] = np.array([graph.degree(node)] + [1.] * (dim_size - 1))
    return prep_dict


def kcore_features(graph, dim_size=128):
    graph.remove_edges_from(nx.selfloop_edges(graph))
    kcore = nx.core_number(graph)
    prep_dict = {}
    for idx, node in enumerate(graph.nodes()):
        feature = np.ones((dim_size))
        feature[0] = kcore[node]
        prep_dict[node] = feature
    return prep_dict

prep_dataset = []

if init != "real":
    for data in dataset:
        edge_index = data.edge_index.t().numpy()
        adj = np.zeros((data.num_nodes, data.num_nodes))
        adj[edge_index[:, 0], edge_index[:, 1]] = 1
        G = nx.from_numpy_matrix(adj)
        G.remove_edges_from(nx.selfloop_edges(G))
        if init == "svd":
            features = svd_features(G, dim_size=args.feature_dim)
        elif init == "degree":
            features = degree_features(G, dim_size=args.feature_dim)
        elif init == "kcore":
            features = kcore_features(G, dim_size=args.feature_dim)
        x = np.array([features[node] for node in range(data.num_nodes)])
        x = torch.FloatTensor(x)
        prep_dataset.append(
            Data(x=x, y=data.y, edge_index=data.edge_index, edge_attr=data.edge_attr))
    # norm
    # xs = torch.cat([x.x for x in prep_dataset], dim=0)
    # mean = xs.mean(dim=0)
    # std = xs.std(dim=0)
    # for data in prep_dataset:
    #     temp_value = torch.FloatTensor((data.x-mean)/std)
    #     nan_indices = torch.isnan(temp_value)
    #     temp_value[nan_indices] = 0
    #     data.x = temp_value
    dataset = prep_dataset


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        dim = 32

        id_features_dim = 128
        self.mapping = nn.Linear(args.feature_dim, id_features_dim)
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
        if args.init != "real":
            x = self.mapping(x)
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


# class Net(torch.nn.Module):
#     def __init__(self, hidden_channels=32):
#         super(Net, self).__init__()
#         in_channels = num_features
#         out_channels = num_classes

#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         num_nodes = ceil(0.5 * average_nodes)
#         self.pool1 = Linear(hidden_channels, num_nodes)

#         self.conv2 = DenseGraphConv(hidden_channels, hidden_channels)
#         num_nodes = ceil(0.5 * num_nodes)
#         self.pool2 = Linear(hidden_channels, num_nodes)

#         self.conv3 = DenseGraphConv(hidden_channels, hidden_channels)

#         self.lin1 = Linear(hidden_channels, hidden_channels)
#         self.lin2 = Linear(hidden_channels, out_channels)

#     def forward(self, x, edge_index, batch):
#         x = F.relu(self.conv1(x, edge_index))

#         x, mask = to_dense_batch(x, batch)
#         adj = to_dense_adj(edge_index, batch)

#         s = self.pool1(x)
#         x, adj, mc1, o1 = dense_mincut_pool(x, adj, s, mask)

#         x = F.relu(self.conv2(x, adj))
#         s = self.pool2(x)

#         x, adj, mc2, o2 = dense_mincut_pool(x, adj, s)

#         x = self.conv3(x, adj)

#         x = x.mean(dim=1)
#         x = F.relu(self.lin1(x))
#         x = self.lin2(x)
#         return F.log_softmax(x, dim=-1), mc1 + mc2, o1 + o2

# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
        
#         if args.init == "real":
#             id_features_dim = num_features
#         else:
#             id_features_dim = 128
#         self.mapping = nn.Linear(args.feature_dim, id_features_dim)
#         self.conv1 = GraphConv(id_features_dim, 128)
#         self.pool1 = TopKPooling(128, ratio=0.8)
#         self.conv2 = GraphConv(128, 128)
#         self.pool2 = TopKPooling(128, ratio=0.8)
#         self.conv3 = GraphConv(128, 128)
#         self.pool3 = TopKPooling(128, ratio=0.8)

#         self.lin1 = torch.nn.Linear(256, 128)
#         self.lin2 = torch.nn.Linear(128, 64)
#         self.lin3 = torch.nn.Linear(64, num_classes)

#     def forward(self, x, edge_index, batch):
#         if args.init != "real":
#             x = self.mapping(x)
#         x = F.relu(self.conv1(x, edge_index))
#         x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
#         x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

#         x = F.relu(self.conv2(x, edge_index))
#         x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
#         x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

#         x = F.relu(self.conv3(x, edge_index))
#         x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
#         x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

#         x = x1 + x2 + x3

#         x = F.relu(self.lin1(x))
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = F.relu(self.lin2(x))
#         x = F.log_softmax(self.lin3(x), dim=-1)

#         return x

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
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print('Epoch: {:03d}, Train Loss: {:.7f}, '
          'Train Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss,
                                                       train_acc, test_acc))

torch.save(model, "model/mnist_supergrid.pth")

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
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from torch_geometric.nn import GINConv, global_mean_pool, JumpingKnowledge
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='ENZYMES')
parser.add_argument('--init', default='svd')
parser.add_argument('--features_dim', default=128, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--seed', default=100, type=int)
args = parser.parse_args()
print(args)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

name = args.data
init = args.init  # degree, kcore

path = f"data/{name}"
dataset = TUDataset(path, name=name).shuffle()
average_nodes = np.mean([data.num_nodes for data in dataset])
num_classes = dataset.num_classes
num_features = dataset.num_features


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
 
def random_uniform_features(graph, dim_size):
    prep_dict = {}
    for idx, node in enumerate(graph.nodes()):
        prep_dict[node] = np.random.uniform(-1.,1.,size=(dim_size))
    return prep_dict

def triangles_features(graph, dim_size):
    if nx.is_directed(graph):
        graph = nx.to_undirected(graph)
    triangles = nx.triangles(graph)
    
    prep_dict = {}
    for idx, node in enumerate(graph.nodes()):
        prep_dict[node] = np.array([triangles[node]]+[1.]*(dim_size-1))
    return prep_dict

import multiprocessing
import numpy as np
import multiprocessing
import networkx as nx
from gensim.models import Word2Vec


def deepwalk_walk_wrapper(class_instance, walk_length, start_node):
    class_instance.deepwalk_walk(walk_length, start_node)


def deepwalk_walk(args):
    '''
    Simulate a random walk starting from start node.
    '''
    walk_length = args["walk_length"]
    neibs = args["neibs"]
    nodes = args["nodes"]
    # if args["iter"] % 5 == 0:
    print("Iter:", args["iter"])  # keep printing, avoid moving process to swap

    walks = []
    for node in nodes:
        walk = [str(node)]
        if len(neibs[node]) == 0:
            walks.append(walk)
            continue
        while len(walk) < walk_length:
            cur = int(walk[-1])
            cur_nbrs = neibs[cur]
            if len(cur_nbrs) == 0:
                break
            walk.append(str(np.random.choice(cur_nbrs)))
        walks.append(walk)
    return walks


class BasicWalker:

    def __init__(self, G, workers):
        self.G = G
        if hasattr(G, 'neibs'):
            self.neibs = G.neibs
        else:
            self.build_neibs_dict()

    def build_neibs_dict(self):
        self.neibs = {}
        for node in self.G.nodes():
            self.neibs[node] = list(self.G.neighbors(node))

    def simulate_walks(self, num_walks, walk_length, num_workers):
        pool = multiprocessing.Pool(processes=num_workers)
        walks = []
        print('Walk iteration:')
        nodes = list(self.G.nodes())
        nodess = [np.random.shuffle(nodes)]
        for i in range(num_walks):
            _ns = nodes.copy()
            np.random.shuffle(_ns)
            nodess.append(_ns)
        params = list(map(lambda x: {'walk_length': walk_length, 'neibs': self.neibs, 'iter': x, 'nodes': nodess[x]},
                          list(range(1, num_walks + 1))))
        walks = pool.map(deepwalk_walk, params)
        pool.close()
        pool.join()
        # walks = np.vstack(walks)
        while len(walks) > 1:
            walks[-2] = walks[-2] + walks[-1]
            walks = walks[:-1]
        walks = walks[0]

        return walks
import networkx as nx


def deepwalk(G, dim_size, number_walks=20, walk_length=10,
             workers=multiprocessing.cpu_count() // 3):
    walk = BasicWalker(G, workers=workers)
    sentences = walk.simulate_walks(
        num_walks=number_walks, walk_length=walk_length, num_workers=workers)
    # for idx in range(len(sentences)):
    #     sentences[idx] = [str(x) for x in sentences[idx]]

    print("Learning representation...")
    word2vec = Word2Vec(sentences=sentences, min_count=0, workers=workers,
                        size=dim_size, sg=1)
    vectors = {}
    for word in G.nodes():
        vectors[word] = word2vec.wv[str(word)]
    return vectors

def init_features(data, features_dim):
    edge_index = data.edge_index.t().numpy()
    adj = np.zeros((data.num_nodes, data.num_nodes))
    adj[edge_index[:, 0], edge_index[:, 1]] = 1
    G = nx.from_numpy_matrix(adj)
    G.remove_edges_from(nx.selfloop_edges(G))
    if init == "svd":
        features = svd_features(G, dim_size=features_dim)
    elif init == "degree":
        features = degree_features(G, dim_size=features_dim)
    elif init == "kcore":
        features = kcore_features(G, dim_size=features_dim)
    elif init == "random":
        features = random_uniform_features(G, dim_size=features_dim)
    elif init == "triangle":
        features = triangles_features(G, dim_size=features_dim)
    elif init == "deepwalk":
        features = deepwalk(G, dim_size=features_dim, number_walks=40, walk_length=10, workers=4)
    elif init == "one":
        features = {node: np.ones((features_dim)) for node in G.nodes()}
    elif init == "degree-standard":
        features = degree_features(G, dim_size=features_dim)
        temp = np.array([features[node] for node in range(data.num_nodes)])
        mean = temp.mean(axis=0)
        std = temp.std(axis=0)
        features = {x: (y-mean)/std for x, y in features.items()}
    elif init == "degree-onehot":
        features = {node: np.zeros((max_deg)) for node in G.nodes()}
        for k, v in features.items():
            v[G.degree(k)-1] = 1

    x = np.array([features[node] for node in range(data.num_nodes)])
    x = torch.FloatTensor(x)
    new_data = Data(x=x, y=data.y, edge_index=data.edge_index, edge_attr=data.edge_attr)
    return new_data


prep_dataset = []

if init == "degree-onehot":
    max_deg = -1
    for data in dataset:
        edge_index = data.edge_index.t().numpy()
        adj = np.zeros((data.num_nodes, data.num_nodes))
        adj[edge_index[:, 0], edge_index[:, 1]] = 1
        G = nx.from_numpy_matrix(adj)
        G.remove_edges_from(nx.selfloop_edges(G))
        max_deg = max(max_deg, max([G.degree(x) for x in G.nodes()]))

if init != "real":
    num_features = args.features_dim
    dataset = [init_features(data, args.features_dim) for data in dataset]

import pickle
if not os.path.isdir("features_init"):
    os.makedirs("features_init")
pickle.dump(dataset, open(f"features_init/{name}-{init}-seed{args.seed}.pkl", "wb"))

inds = np.random.permutation(len(dataset))
dataset = [dataset[x] for x in inds]
test_dataset = dataset[:len(dataset) // 10]
train_dataset = dataset[len(dataset) // 10:]
test_loader = DataLoader(test_dataset, batch_size=64)
train_loader = DataLoader(train_dataset, batch_size=64)


class GIN(torch.nn.Module):
    def __init__(self, num_layers, hidden):
        super(GIN, self).__init__()
        self.conv1 = GINConv(Sequential(
            Linear(num_features, hidden),
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
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GIN(5, 64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train(epoch):
    model.train()
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
    if epoch%20 == 0:
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        print('Epoch: {:03d}, Train Loss: {:.7f}, '
            'Train Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss,
                                                        train_acc, test_acc))

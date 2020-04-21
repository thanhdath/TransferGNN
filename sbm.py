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

def remove_self_loops(edge_index, edge_attr=None):
    r"""Removes every self-loop in the graph given by :attr:`edge_index`, so
    that :math:`(i,i) \not\in \mathcal{E}` for every :math:`i \in \mathcal{V}`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    row, col = edge_index
    mask = row != col
    edge_attr = edge_attr if edge_attr is None else edge_attr[mask]
    edge_index = edge_index[:, mask]

    return edge_index, edge_attr


# def load_graph(input_dir, multiclass=None):
#     dataname = [x for x in input_dir.split("/") if len(x) > 0][-1]
#     graph_file = f"{input_dir}/{dataname}_graph.json"
#     graph_data = json.load(open(graph_file))
#     G = json_graph.node_link_graph(graph_data)

#     graph_id_file = f"{input_dir}/graph_id.npy"
#     graph_ids = np.load(graph_id_file)
#     nodes = np.array(sorted(G.nodes()))

#     feature_file = f"{input_dir}/feats.npy"
#     features = np.load(feature_file)

#     label_file = f"{input_dir}/labels.npy"
#     labels = np.load(label_file)

#     graphs_data = []
#     for id in set(graph_ids):
#         subgraph_nodes = nodes[graph_ids == id]
#         subgraph = G.subgraph(subgraph_nodes)
#         x = torch.FloatTensor(features[subgraph_nodes])
#         y = torch.FloatTensor(labels[subgraph_nodes])
#         mapping = {id: i for i, id in enumerate(subgraph_nodes)}
#         subgraph = nx.relabel_nodes(subgraph, mapping)
#         edge_index = torch.LongTensor(np.array(subgraph.edges())).t()
#         edge_index, _ = remove_self_loops(edge_index)
#         graphs_data.append((edge_index, x, y))
#     return graphs_data, x.shape[1], y.shape[1]

from scipy.sparse.linalg import svds
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
    print("Iter:", args["iter"]) # keep printing, avoid moving process to swap

    walks = []
    for node in nodes:
        walk = [str(node)]
        if len(neibs[node]) == 0:
            walks.append(walk)
            continue
        while len(walk) < walk_length:
            cur = int(walk[-1])
            cur_nbrs = neibs[cur]
            if len(cur_nbrs) == 0: break
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
            list(range(1, num_walks+1))))
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
    workers=multiprocessing.cpu_count()//3):
    walk = BasicWalker(G, workers=workers)
    sentences = walk.simulate_walks(num_walks=number_walks, walk_length=walk_length, num_workers=workers)
    # for idx in range(len(sentences)):
    #     sentences[idx] = [str(x) for x in sentences[idx]]

    print("Learning representation...")
    word2vec = Word2Vec(sentences=sentences, min_count=0, workers=workers,
                            size=dim_size, sg=1)
    vectors = {}
    for word in G.nodes():
        vectors[word] = word2vec.wv[str(word)]
    return vectors

def load_graph(input_dir):
    dataname = [x for x in input_dir.split("/") if len(x) > 0][-1]
    path = f"{input_dir}/{dataname}_graph.json"
    with open(path, 'r') as f:
        G = nx.DiGraph(json_graph.node_link_graph(json.load(f)))

    x = np.load(f"{input_dir}/feats.npy")
    x = torch.from_numpy(x).to(torch.float)

    y = np.load(f"{input_dir}/labels.npy")
    y = torch.from_numpy(y).to(torch.long).flatten()

    data_list = []
    path = f"{input_dir}/graph_id.npy"
    idx = torch.from_numpy(np.load(path)).to(torch.long)
    idx = idx - idx.min()

    for i in range(idx.max().item() + 1):
        mask = idx == i

        G_s = G.subgraph(mask.nonzero().view(-1).tolist())

        # extra_features = svd_features(G_s)
        # extra_features = deepwalk(G_s, 128, number_walks=40, walk_length=80)
        # extra_features = np.array([extra_features[x] for x in G_s.nodes()])
        # extra_features = torch.FloatTensor(extra_features)
        # features = torch.cat([x[mask], extra_features], dim=1)
        features = x[mask]

        edge_index = torch.tensor(list(G_s.edges)).t().contiguous()
        edge_index = edge_index - edge_index.min()
        edge_index, _ = remove_self_loops(edge_index)
        try:
            edge_attr = torch.FloatTensor([G_s.edges()[x]['weight'] for x in G_s.edges()])
        except:
            edge_attr = None

        data = (edge_index, features, y[mask], edge_attr)
        data_list.append(data)
    return data_list, features.shape[1], int(max(y) + 1)

parser = argparse.ArgumentParser()
parser.add_argument("--transfer", default=None)
parser.add_argument("--input-dir", default="../graphrnn/data-ppi/synf-seed104/Atrain-Xtrain/")
parser.add_argument("--epochs", default=300, type=int)
parser.add_argument("--hidden", default=64, type=int)
parser.add_argument("--feature-only", action='store_true')
parser.add_argument("--seed", default=100, type=int)
parser.add_argument("--f", default="ori")
parser.add_argument("--is-test-graphs", action="store_true")
args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
args.multiclass = False
graphs, num_features, num_classes = load_graph(args.input_dir)
print("Num features:", num_features)
print("Num classes:", num_classes)
if graphs[0][-1] is not None:
    print("Graph has edge attributes")


if args.is_test_graphs:
    train_graphs = []
    val_graphs = graphs
    test_graphs = []

    if args.f != "ori":
        converted_val_graphs = []
        for edge_index, X, y, edge_attr in val_graphs:
            k = edge_index.shape[1] // X.shape[0]
            if args.f == "knn":
                adj = generate_graph(X, kind="knn", k=k)
            if args.f == "sigmoid":
                adj = generate_graph(X, kind="sigmoid", k=k)
            if args.f == "random":
                G = connected_watts_strogatz_graph(X.shape[0], k+1, p=0.1)
                adj = nx.to_numpy_matrix(G)
            
            src, trg = adj.nonzero()
            new_edge_index = torch.LongTensor(np.concatenate([src.reshape(1, -1), trg.reshape(1,-1)], axis=0))
            print(f"N edges before {edge_index.shape[1]} - N edges then {new_edge_index.shape[1]} ")
            converted_val_graphs.append((new_edge_index, X, y, None))
        val_graphs = converted_val_graphs
else:
    train_graphs = graphs[:-10]
    val_graphs = graphs[-10:]
    test_graphs = []

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
        id_features_dim = 128
        self.mapping = nn.Linear(num_features, id_features_dim)
        self.conv1 = SAGEConv(id_features_dim, args.hidden, normalize=False)
        self.conv2 = SAGEConv(args.hidden, args.hidden * 2, normalize=False)
        self.conv3 = SAGEConv(args.hidden * 2, num_classes, normalize=False)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.mapping(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)


class SoftmaxRegression(torch.nn.Module):

    def __init__(self):
        super(SoftmaxRegression, self).__init__()
        self.model = torch.nn.Linear(num_features, num_classes)

    def forward(self):
        x = data.x
        x = self.model(x)
        if multiclass:
            return x
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

        edge_index = edge_index.cpu()
        x = x.cpu()
        y = y.cpu()
        torch.cuda.empty_cache()
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
if args.feature_only:
    print("Using softmax regression")
    model = SoftmaxRegression().to(device)
else:
    model = Net().to(device)
lr = 0.005
if args.transfer is not None:
    print("Load pretrained model", args.transfer)
    pretrained_state_dict = torch.load(args.transfer)
    differ_shape_params = []
    model_state_dict = model.state_dict()
    for k in pretrained_state_dict.keys():
        if pretrained_state_dict[k].shape != model_state_dict[k].shape:
            differ_shape_params.append(k)
    pretrained_state_dict.update(
        {k: v for k, v in model.state_dict().items() if k in differ_shape_params})
    model.load_state_dict(pretrained_state_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

best_val_acc = 0
best_model = None
epochs = args.epochs
if not os.path.isdir("model"):
    os.makedirs("model")
if args.transfer is not None:
    model_name = f"model/{getdirname(args.input_dir)}-transfer-from-{getfilename(args.transfer)}.pkl"
else:
    model_name = f"model/{getdirname(args.input_dir)}.pkl"
writer = tensorboardX.SummaryWriter(
    logdir="runs/" + model_name.split("/")[-1].split(".")[0])

train_acc, train_macro = test(train_graphs, n_randoms=2)
val_acc, val_macro = test(val_graphs)
log = 'Epoch: 0, micro-macro Train: {:.4f}-{:.4f}, Val: {:.4f}-{:.4f}'
torch.save(model.state_dict(), model_name)
print(log.format(train_acc, train_macro, val_acc, val_macro))

best_val_acc = val_acc
for epoch in range(1, epochs):
    # if args.transfer is not None and epoch < epochs//3:
    #     model.conv1.requires_grad = False
    # else:
    #     model.conv1.requires_grad = True
    train(train_graphs)
    if epoch == epochs - 1:
        model.load_state_dict(torch.load(model_name))
        print("Load best model")

    train_acc, train_macro = test(train_graphs, n_randoms=2)
    val_acc, val_macro = test(val_graphs)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), model_name)
        # test_acc = tmp_test_acc
    if epoch % 20 == 0 or epoch == epochs - 1:
        log = 'Epoch: {:03d}, micro-macro Train: {:.4f}-{:.4f}, Val: {:.4f}-{:.4f}'
        print(log.format(epoch, train_acc, train_macro, val_acc, val_macro))
        writer.add_scalar("train_acc", train_acc, epoch)
        writer.add_scalar("val_acc", val_acc, epoch)
# torch.save(model.state_dict(), model_name)
print("Best val acc: {:.3f}".format(best_val_acc))
print("Model has been saved to", model_name)

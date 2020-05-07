# ppi, learn and transfer from multi graphs

import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import ChebConv, GCNConv, SAGEConv
# from gat_custom import GATConv
from data import load_graph
import argparse
import os
from sklearn.metrics import f1_score
import numpy as np
#import matplotlib.pyplot as plt
import tensorboardX
from sklearn.metrics import classification_report
from sage import SAGECompletedGraph

import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer
from torch_geometric.data import NeighborSampler, Data
import re
from scipy.sparse import csr_matrix, vstack, hstack
import pickle
import json
import networkx as nx
from networkx.readwrite import json_graph
torch.backends.cudnn.deterministic = True

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
    path = f"{input_dir}/{dataname}_graph.pkl"
    with open(path, 'rb') as f:
        # G = nx.DiGraph(json_graph.node_link_graph(json.load(f)))
        G = pickle.load(f)

    x = np.load(f"{input_dir}/feats.npy")
    x = torch.from_numpy(x).to(torch.float)

    y = np.load(f"{input_dir}/labels.npy")
    y = torch.from_numpy(y).to(torch.float)

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

        # edge_index = torch.tensor(list(G_s.edges)).t().contiguous()
        # edge_index = edge_index - edge_index.min()
        # edge_index, _ = remove_self_loops(edge_index)
        # try:
        #     edge_attr = torch.FloatTensor([G_s.edges()[x]['weight'] for x in G_s.edges()])
        # except:
        #     edge_attr = None
        adj = torch.FloatTensor(nx.to_numpy_matrix(G_s))

        data = (adj, features, y[mask])
        data_list.append(data)
    return data_list, features.shape[1], y.shape[1]

parser = argparse.ArgumentParser()
parser.add_argument("--transfer", default=None)
parser.add_argument(
    "--input-dir", default="../graphrnn/data-ppi/synf-seed104/Atrain-Xtrain/")
parser.add_argument("--epochs", default=300, type=int)
parser.add_argument("--hidden", default=64, type=int)
parser.add_argument("--gat", action='store_true')
parser.add_argument("--seed", default=100, type=int)
parser.add_argument("--is-test-graphs", action="store_true")
parser.add_argument("--sbm", action="store_true")
args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
args.multiclass = True
graphs, num_features, num_classes = load_graph(args.input_dir)
print("Num features:", num_features)
print("Num classes:", num_classes)
if graphs[0][-1] is not None:
    print("Graph has edge attributes")


if args.is_test_graphs:
    train_graphs = []
    val_graphs = graphs
    test_graphs = []
else:
    if args.sbm:
        train_graphs = graphs[:-32]
        val_graphs = graphs[-32:]
        test_graphs = []
    else:
        train_graphs = graphs[:-2]
        val_graphs = graphs[-2:]
        test_graphs = []

# GCN

# class GAT(torch.nn.Module):
#     def __init__(self):
#         super(GAT, self).__init__()
#         id_features_dim = 128
#         self.mapping = nn.Linear(num_features, id_features_dim)
#         self.conv1 = GATConv(id_features_dim, 256, heads=4)
#         self.lin1 = torch.nn.Linear(id_features_dim, 4 * 256)
#         self.conv2 = GATConv(4 * 256, 256, heads=4)
#         self.lin2 = torch.nn.Linear(4 * 256, 4 * 256)
#         self.conv3 = GATConv(4 * 256, num_classes, heads=6, concat=False)
#         self.lin3 = torch.nn.Linear(4 * 256, num_classes)

#     def forward(self, x, adj):
#         x = self.mapping(x)
#         x = F.elu(self.conv1(x, adj) + self.lin1(x))
#         x = F.elu(self.conv2(x, adj) + self.lin2(x))
#         x = self.conv3(x, adj) + self.lin3(x)
#         return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GAT(nn.Module):
    def __init__(self):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        id_features_dim = 128
        nhid = 64
        dropout = 0.6
        alpha = 0.2
        nheads = 4

        self.dropout = dropout
        self.mapping = nn.Linear(num_features, id_features_dim)
        self.attentions = [GraphAttentionLayer(id_features_dim, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, num_classes, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = self.mapping(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x

# class GAT(torch.nn.Module):

#     def __init__(self):
#         super(GAT, self).__init__()
#         id_features_dim = 128
#         self.mapping = nn.Linear(num_features, id_features_dim)
#         self.conv1 = GATConv(id_features_dim, 256, heads=4)
#         self.lin1 = torch.nn.Linear(id_features_dim, 4 * 256)
#         self.conv2 = GATConv(4 * 256, num_classes, heads=4, concat=False)
#         self.lin2 = torch.nn.Linear(4 * 256, num_classes)

#     def forward(self, x, adj):
#         x = self.mapping(x)
#         x = F.elu(self.conv1(x, adj) + self.lin1(x))
#         x = self.conv2(x, adj) + self.lin2(x)
#         return x

class GraphsageMEAN(torch.nn.Module):

    def __init__(self):
        super(GraphsageMEAN, self).__init__()
        id_features_dim = 128
        self.mapping = nn.Linear(num_features, id_features_dim)
        self.conv1 = SAGECompletedGraph (id_features_dim, args.hidden, normalize=False)
        self.conv2 = SAGECompletedGraph(args.hidden, args.hidden * 2, normalize=False)
        self.conv3 = SAGECompletedGraph(args.hidden * 2, num_classes, normalize=False)

    def forward(self, x, adj):
        x = self.mapping(x)
        x = self.conv1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, adj)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, adj)
        return x


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
    for adj, x, y in train_graphs:
        adj = adj.to(device)
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        outputs = model(x, adj).view(len(x), -1)
        criterion(outputs, y).backward()
        optimizer.step()

        adj = adj.cpu()
        x = x.cpu()
        y = y.cpu()
        torch.cuda.empty_cache()


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

    for adj, x, y in graphs:
        adj = adj.to(device)
        x = x.to(device)
        y = y.to(device)
        logits = model(x, adj).view(len(x), -1)
        logitss.append(logits)
        ys.append(y)

        adj = adj.cpu()
        x = x.cpu()
        y = y.cpu()
        torch.cuda.empty_cache()
    logitss = torch.cat(logitss, dim=0)
    ys = torch.cat(ys, dim=0)
    micro, macro = f1(logitss, ys, multiclass=True)
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
# if args.feature_only:
#     print("Using softmax regression")
#     model = SoftmaxRegression().to(device)
# else:
if args.gat:
    model = GAT().to(device)
else:
    model = GraphsageMEAN().to(device)
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

criterion = torch.nn.BCEWithLogitsLoss()
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

# model.eval()
# with torch.no_grad():
#     logitss = []
#     ys = []
#     for edge_index, x, y in graphs:
#         edge_index = edge_index.to(device)
#         x = x.to(device)
#         y = y.to(device)
#         logits = model(x, edge_index)
#         logitss.append(logits)
#         ys.append(y)
#     preds = torch.cat(logitss, dim=0)
#     y = torch.cat(ys, dim=0)
#     # preds = torch.sigmoid(preds).cpu().numpy()
#     # preds[preds >= 0.5] = 1
#     # preds[preds < 0.5] = 0
#     preds = (preds > 0).float().cpu().numpy()
#     print(classification_report(y.cpu().numpy(), preds))

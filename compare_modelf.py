# ppi, learn and transfer from multi graphs

import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import ChebConv, GCNConv, SAGEConv, SGConv, GATConv
from sage import SAGESumConv
from data import load_graph
import argparse
import os
from sklearn.metrics import f1_score
import numpy as np
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
import torch.utils.checkpoint as checkpoint
# torch.backends.cudnn.deterministic = True

import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros


# class GATConv(MessagePassing):
#     r"""The graph attentional operator from the `"Graph Attention Networks"
#     <https://arxiv.org/abs/1710.10903>`_ paper
#     .. math::
#         \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
#         \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},
#     where the attention coefficients :math:`\alpha_{i,j}` are computed as
#     .. math::
#         \alpha_{i,j} =
#         \frac{
#         \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
#         [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
#         \right)\right)}
#         {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
#         \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
#         [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
#         \right)\right)}.
#     Args:
#         in_channels (int): Size of each input sample.
#         out_channels (int): Size of each output sample.
#         heads (int, optional): Number of multi-head-attentions.
#             (default: :obj:`1`)
#         concat (bool, optional): If set to :obj:`False`, the multi-head
#             attentions are averaged instead of concatenated.
#             (default: :obj:`True`)
#         negative_slope (float, optional): LeakyReLU angle of the negative
#             slope. (default: :obj:`0.2`)
#         dropout (float, optional): Dropout probability of the normalized
#             attention coefficients which exposes each node to a stochastically
#             sampled neighborhood during training. (default: :obj:`0`)
#         bias (bool, optional): If set to :obj:`False`, the layer will not learn
#             an additive bias. (default: :obj:`True`)
#         **kwargs (optional): Additional arguments of
#             :class:`torch_geometric.nn.conv.MessagePassing`.
#     """
#     def __init__(self, in_channels, out_channels, heads=1, concat=True,
#                  negative_slope=0.2, dropout=0, bias=True, **kwargs):
#         super(GATConv, self).__init__(aggr='add', **kwargs)

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.heads = heads
#         self.concat = concat
#         self.negative_slope = negative_slope
#         self.dropout = dropout

#         self.weight = Parameter(torch.Tensor(in_channels,
#                                              heads * out_channels))
#         self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

#         if bias and concat:
#             self.bias = Parameter(torch.Tensor(heads * out_channels))
#         elif bias and not concat:
#             self.bias = Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)

#         self.reset_parameters()

#     def reset_parameters(self):
#         glorot(self.weight)
#         glorot(self.att)
#         zeros(self.bias)

#     def forward(self, x, edge_index, size=None,
#                 return_attention_weights=False):
#         """"""
#         if size is None and torch.is_tensor(x):
#             edge_index, _ = remove_self_loops(edge_index)
#             edge_index, _ = add_self_loops(edge_index,
#                                            num_nodes=x.size(self.node_dim))

#         if torch.is_tensor(x):
#             x = torch.matmul(x, self.weight)
#         else:
#             x = (None if x[0] is None else torch.matmul(x[0], self.weight),
#                  None if x[1] is None else torch.matmul(x[1], self.weight))

#         out = self.propagate(edge_index, size=size, x=x,
#                              return_attention_weights=return_attention_weights)

#         if return_attention_weights:
#             alpha, self.alpha = self.alpha, None
#             return out, alpha
#         else:
#             return out

#     def message(self, edge_index_i, x_i, x_j, size_i,
#                 return_attention_weights):
#         bs = 128
#         # Compute attention coefficients.
#         x_j = x_j.view(-1, self.heads, self.out_channels)
#         if x_i is None:
#             alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
#         else:
#             x_i = x_i.view(-1, self.heads, self.out_channels)
#             # alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
#             alpha = torch.zeros((x_i.shape[0], self.heads)).to(device)
#             n_iters = int(np.ceil(x_i.shape[0]/bs))
#             for iter in range(n_iters):
#                 batch_x_i = x_i[iter*bs:(iter+1)*bs]
#                 batch_x_j = x_j[iter*bs:(iter+1)*bs]
#                 alpha[iter*bs:(iter+1)*bs] = (torch.cat([batch_x_i, batch_x_j], dim=-1) * self.att).sum(dim=-1)

#         alpha = F.leaky_relu(alpha, self.negative_slope)
#         alpha = softmax(alpha, edge_index_i, size_i)

#         if return_attention_weights:
#             self.alpha = alpha

#         # Sample attention coefficients stochastically.
#         alpha = F.dropout(alpha, p=self.dropout, training=self.training)
#         res = torch.zeros_like(x_j).to(device)
#         n_iters = int(np.ceil(x_j.shape[0]/bs))
#         for iter in range(n_iters):
#             batch_x_j = x_j[iter*bs:(iter+1)*bs]
#             batch_alpha = alpha[iter*bs:(iter+1)*bs].view(-1, self.heads, 1)
#             res[iter*bs:(iter+1)*bs] = batch_x_j *  batch_alpha
#         return res

#     def update(self, aggr_out):
#         if self.concat is True:
#             aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
#         else:
#             aggr_out = aggr_out.mean(dim=1)

#         if self.bias is not None:
#             aggr_out = aggr_out + self.bias
#         return aggr_out

#     def __repr__(self):
#         return '{}({}, {}, heads={})'.format(self.__class__.__name__,
#                                              self.in_channels,
#                                              self.out_channels, self.heads)

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


def load_graph(data_path):
    def graphs2data(graphs):
        data = []
        for g in graphs:
            adj = nx.to_numpy_matrix(g)
            edge_index = np.argwhere(adj >= args.th).T
            print(f"Number of edges: {edge_index.shape[1]}")
            features = np.array([g.nodes()[x]['features']
                                 for x in g.nodes()])[:, :args.num_features]
            labels = np.array([g.nodes()[x]['label'] for x in g.nodes()])
            if len(labels.shape) == 2:
                labels = torch.FloatTensor(labels)
            #     # labels = labels.reshape(-1)
            #     labels = labels.argmax(axis=1)
                # assert labels.shape[0] == features.shape[0], "Error! Multilabel"
            else:
                labels = torch.LongTensor(labels)
            data.append((torch.LongTensor(edge_index), torch.FloatTensor(features), labels))
        return data

    obj = pickle.load(open(data_path, "rb"))
    train_graphs_gen = obj["train_graphs_gen"]
    test_graphs_gen = obj["test_graphs_gen"]
    train_graphs_original = obj["train_graphs_original"]
    test_graphs_original = obj["test_graphs_original"]

    train_graphs_gen = graphs2data(train_graphs_gen)
    test_graphs_gen = graphs2data(test_graphs_gen)
    train_graphs_original = graphs2data(train_graphs_original)
    test_graphs_original = graphs2data(test_graphs_original)

    return train_graphs_original, test_graphs_original, train_graphs_gen, test_graphs_gen

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", default="/home/datht/gnf/ckpt-sbm/n128-p8-lam1.5-mu32.0/gnf/sbm_gnf.pkl")
parser.add_argument("--epochs", default=300, type=int)
parser.add_argument("--batch_size", default=64, type=int)
# specify num features, due to previously padded with zeros
parser.add_argument("--num_features", default=8, type=int)
parser.add_argument("--num_labels", default=2, type=int)
parser.add_argument("--hidden", default=64, type=int)
parser.add_argument("--multiclass", action='store_true')
parser.add_argument("--model", default='mean',
                    choices=['gat', 'mean', 'sum', 'sgc', 'gcn'])
parser.add_argument("--seed", default=100, type=int)
parser.add_argument("--th", default=0.5, type=float)
parser.add_argument("--setting", default='A', choices=['A', 'B', 'C', 'D'])
args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

train_graphs_original, test_graphs_original, train_graphs_gen, test_graphs_gen = load_graph(
    args.data_path)
num_features = args.num_features
num_classes = args.num_labels
print("Num features:", num_features)
print("Num classes:", num_classes)

if args.setting == 'A':
    train_graphs = train_graphs_original
    test_graphs = test_graphs_original
elif args.setting == 'B':
    train_graphs = train_graphs_gen
    test_graphs = test_graphs_original
elif args.setting == 'C':
    train_graphs = train_graphs_gen
    test_graphs = test_graphs_gen
elif args.setting == 'D':
    train_graphs = train_graphs_original
    test_graphs = test_graphs_gen


class GAT(torch.nn.Module):

    def __init__(self):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, args.hidden, heads=4)
        self.lin1 = torch.nn.Linear(num_features, 4 * args.hidden)
        # self.conv2 = GATConv(4 * args.hidden, args.hidden, heads=4)
        # self.lin2 = torch.nn.Linear(4 * args.hidden, 4 * args.hidden)
        self.conv3 = GATConv(4 * args.hidden, num_classes, heads=6, concat=False)
        self.lin3 = torch.nn.Linear(4 * args.hidden, num_classes)

    def forward(self, x, edge_index, edge_attr=None):
        x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
        # x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
        x = self.conv3(x, edge_index) + self.lin3(x)
        if not args.multiclass:
            return F.log_softmax(x, dim=1)
        return x

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
        if not args.multiclass:
            return F.log_softmax(x, dim=1)
        return x


class GraphsageSUM(torch.nn.Module):

    def __init__(self):
        super(GraphsageSUM, self).__init__()
        self.conv1 = SAGESumConv(num_features, args.hidden, normalize=False)
        self.conv2 = SAGESumConv(args.hidden, args.hidden * 2, normalize=False)
        self.conv3 = SAGESumConv(args.hidden * 2, num_classes, normalize=False)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        if not args.multiclass:
            return F.log_softmax(x, dim=1)
        return x


class SGC(torch.nn.Module):

    def __init__(self):
        super(SGC, self).__init__()
        self.conv1 = SGConv(num_features, num_classes, K=2, cached=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        if not args.multiclass:
            return F.log_softmax(x, dim=1)
        return x


class GCN(torch.nn.Module):

    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, args.hidden, cached=False,
                             normalize=True)
        self.conv2 = GCNConv(args.hidden, num_classes, cached=False,
                             normalize=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index, None))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, None)
        if not args.multiclass:
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
        if not args.multiclass:
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
if args.model == "gat":
    model = GAT().to(device)
elif args.model == "sum":
    model = GraphsageSUM().to(device)
elif args.model == "sgc":
    model = SGC().to(device)
elif args.model == "gcn":
    model = GCN().to(device)
else:
    model = GraphsageMEAN().to(device)
print(model)
lr = 0.005

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

best_val_acc = 0
best_model = None
epochs = args.epochs
train_acc, _ = test(train_graphs, n_randoms=4)
test_acc, _ = test(test_graphs)
log = 'Epoch: 0, Acc Train: {:.4f}, Test: {:.4f}'
print(log.format(train_acc, test_acc))

for epoch in range(1, epochs):
    train(train_graphs)
    train_acc, _ = test(train_graphs, n_randoms=2)

    if epoch % 1 == 0 or epoch == epochs - 1:
        test_acc, _ = test(test_graphs)
        log = 'Epoch: {:03d}, Acc Train: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, test_acc))


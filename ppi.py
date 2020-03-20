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
import numpy as np
import networkx as nx
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer
from torch_geometric.data import NeighborSampler, Data
import re

import json
import networkx as nx
from networkx.readwrite import json_graph


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

def load_graph(input_dir):
    dataname = [x for x in input_dir.split("/") if len(x) > 0][-1]
    path = f"{input_dir}/{dataname}_graph.json"
    with open(path, 'r') as f:
        G = nx.DiGraph(json_graph.node_link_graph(json.load(f)))

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
        edge_index = torch.tensor(list(G_s.edges)).t().contiguous()
        edge_index = edge_index - edge_index.min()
        edge_index, _ = remove_self_loops(edge_index)

        data = (edge_index, x[mask], y[mask])
        data_list.append(data)
    return data_list, x.shape[1], y.shape[1]

parser = argparse.ArgumentParser()
parser.add_argument("--transfer", default=None)
parser.add_argument(
    "--input-dir", default="../graphrnn/data-ppi/synf-seed104/Atrain-Xtrain/")
parser.add_argument("--epochs", default=300, type=int)
parser.add_argument("--hidden", default=64, type=int)
parser.add_argument("--feature-only", action='store_true')
parser.add_argument("--seed", default=100, type=int)
parser.add_argument("--is-test-graphs", action="store_true")
args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
args.multiclass = True
graphs, num_features, num_classes = load_graph(args.input_dir)
print("Num features:", num_features)
print("Num classes:", num_classes)


if args.is_test_graphs:
    train_graphs = []
    val_graphs = graphs
    test_graphs = []
else:
    train_graphs = graphs[:-2]
    val_graphs = graphs[-2:]
    test_graphs = []

# GCN


# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = GATConv(num_features, 256, heads=4)
#         self.lin1 = torch.nn.Linear(num_features, 4 * 256)
#         self.conv2 = GATConv(4 * 256, 256, heads=4)
#         self.lin2 = torch.nn.Linear(4 * 256, 4 * 256)
#         self.conv3 = GATConv(
#             4 * 256, num_classes, heads=6, concat=False)
#         self.lin3 = torch.nn.Linear(4 * 256, num_classes)

#     def forward(self):
#         x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
#         x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
#         x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
#         x = self.conv3(x, edge_index) + self.lin3(x)
#         if multiclass:
#             return x
#         return F.log_softmax(x, dim=1)

# class Net(torch.nn.Module):

#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = GATConv(num_features, 256, heads=4)
#         self.lin1 = torch.nn.Linear(num_features, 4 * 256)
#         self.conv2 = GATConv(4 * 256, 256, heads=4)
#         self.lin2 = torch.nn.Linear(4 * 256, 4 * 256)
#         self.conv3 = GATConv(
#             4 * 256, num_classes, heads=6, concat=False)
#         self.lin3 = torch.nn.Linear(4 * 256, num_classes)

#     def forward(self, x, edge_index):
#         x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
#         x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
#         x = self.conv3(x, edge_index) + self.lin3(x)
#         return x

class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SAGEConv(num_features, args.hidden, normalize=False)
        self.conv2 = SAGEConv(args.hidden, args.hidden * 2, normalize=False)
        self.conv3 = SAGEConv(args.hidden * 2, num_classes, normalize=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
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
    for edge_index, x, y in train_graphs:
        edge_index = edge_index.to(device)
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        outputs = model(x, edge_index)
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

    for edge_index, x, y in graphs:
        edge_index = edge_index.to(device)
        x = x.to(device)
        y = y.to(device)
        logits = model(x, edge_index)
        logitss.append(logits)
        ys.append(y)
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

model.eval()
with torch.no_grad():
    logitss = []
    ys = []
    for edge_index, x, y in graphs:
        edge_index = edge_index.to(device)
        x = x.to(device)
        y = y.to(device)
        logits = model(x, edge_index)
        logitss.append(logits)
        ys.append(y)
    preds = torch.cat(logitss, dim=0)
    y = torch.cat(ys, dim=0)
    # preds = torch.sigmoid(preds).cpu().numpy()
    # preds[preds >= 0.5] = 1
    # preds[preds < 0.5] = 0
    preds = (preds > 0).float()
    print(classification_report(y.cpu().numpy(), preds))

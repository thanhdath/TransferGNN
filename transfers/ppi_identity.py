import networkx as nx
from networkx.readwrite import json_graph
import json
import numpy as np
from torch_geometric.datasets import PPI
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import scipy.stats
import os
import matplotlib.pyplot as plt
from torch_geometric.nn import ChebConv, GATConv, GCNConv, SAGEConv
from scipy.sparse.linalg import svds
import torch.nn.functional as F
import torch_geometric.transforms as T
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score

def svd_features(graph, dim_size=128, alpha=0.5):
    adj = nx.to_scipy_sparse_matrix(graph, dtype=np.float32)
    if adj.shape[0] <= dim_size:
        padding_row = csr_matrix((dim_size-adj.shape[0]+1, adj.shape[1]), dtype=adj.dtype)
        adj = vstack((adj, padding_row))
        padding_col = csr_matrix((adj.shape[0], dim_size-adj.shape[1]+1), dtype=adj.dtype)
        adj = hstack((adj, padding_col))

    U, X,_ = svds(adj, k = dim_size)
    embedding = U*(X**(alpha))
    features = {node: embedding[i] for i, node in enumerate(graph.nodes())}
    return features

def load_ppi():
    def dataset2graphs(dataset):
        graphs = []
        for data in dataset:
            edge_index = data.edge_index.t().numpy()
            x = data.x.numpy()
            y = data.y.numpy()
            adj = np.zeros((len(x), len(x)))
            adj[edge_index[:, 0], edge_index[:, 1]] = 1
            G = nx.from_numpy_matrix(adj)
            for i, node in enumerate(G.nodes()):
                G.nodes()[node]['features'] = x[i].tolist()
                G.nodes()[node]['label'] = y[i].tolist()
            graphs.append(G)
        # graphs = [convert_nx_repr(g) for g in graphs]
        data = []
        for graph in graphs:
            features = svd_features(graph, dim_size=args.svd_features_dim, alpha=0.5)
            for node in graph.nodes():
                graph.nodes()[node]['features'] = features[node]
            graph.remove_edges_from(nx.selfloop_edges(graph))
            edge_index = np.array(graph.edges())
            edge_index = torch.LongTensor(edge_index).t().contiguous()
            x = torch.FloatTensor(np.array([features[node] for node in graph.nodes()]))
            y = torch.FloatTensor(np.array([graph.nodes()[node]['label'] for node in graph.nodes()]))
            data.append((edge_index, x, y))
        return data
    path = "dataset/"
    train_dataset = PPI(path, split='train')
    val_dataset = PPI(path, split='val')
    test_dataset = PPI(path, split='test')
    train_graphs = dataset2graphs(train_dataset)
    val_graphs = dataset2graphs(val_dataset)
    test_graphs = dataset2graphs(test_dataset)

    return train_graphs, val_graphs, test_graphs




# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         hidden = 128
#         num_classes = 121
#         svd_features_dim = args.svd_features_dim
#         id_features_dim = 128
#         self.mapping = nn.Linear(svd_features_dim, id_features_dim)
#         self.conv1 = SAGEConv(id_features_dim, hidden, normalize=False)
#         self.conv2 = SAGEConv(hidden, hidden * 2, normalize=False)
#         self.conv3 = SAGEConv(hidden * 2, num_classes, normalize=False)

#     def forward(self, x, edge_index):
#         x = self.mapping(x)
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv3(x, edge_index)
#         return x
    
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        hidden = 128
        num_classes = 121
        svd_features_dim = args.svd_features_dim
        id_features_dim = 128
        self.mapping = nn.Linear(svd_features_dim, id_features_dim)
        self.conv1 = GATConv(id_features_dim, 256, heads=4)
        self.lin1 = torch.nn.Linear(id_features_dim, 4 * 256)
        self.conv2 = GATConv(4 * 256, 256, heads=4)
        self.lin2 = torch.nn.Linear(4 * 256, 4 * 256)
        self.conv3 = GATConv(4 * 256, num_classes, heads=6, concat=False)
        self.lin3 = torch.nn.Linear(4 * 256, num_classes)

    def forward(self, x, edge_index):
        x = self.mapping(x)
        x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
        x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
        x = self.conv3(x, edge_index) + self.lin3(x)
        return x

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
        edge_index = edge_index.cpu()
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


"""
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=300, type=int)
# parser.add_argument("--hidden", default=64, type=int)
parser.add_argument("--seed", default=100, type=int)
args = parser.parse_args()
"""
from types import SimpleNamespace
args = {
    "epochs": 300,
    "seed": 100,
    "svd_features_dim": 128
}
args = SimpleNamespace(**args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
args.multiclass = True


train_graphs, val_graphs, test_graphs = load_ppi()

train_graphs = train_graphs + val_graphs
val_graphs = test_graphs


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
lr = 0.005
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

best_val_acc = 0
best_model = None
epochs = args.epochs
if not os.path.isdir("model"):
    os.makedirs("model")
model_name = f"model/model.pkl"
# writer = tensorboardX.SummaryWriter(logdir="runs/" + model_name.split("/")[-1].split(".")[0])

train_acc, train_macro = test(train_graphs, n_randoms=2)
val_acc, val_macro = test(val_graphs)
log = 'Epoch: 0, micro-macro Train: {:.4f}-{:.4f}, Val: {:.4f}-{:.4f}'
torch.save(model.state_dict(), model_name)
print(log.format(train_acc, train_macro, val_acc, val_macro))

best_val_acc = val_acc
for epoch in range(1, epochs + 1):
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
# torch.save(model.state_dict(), model_name)
print("Best val acc: {:.3f}".format(best_val_acc))
print("Model has been saved to", model_name)

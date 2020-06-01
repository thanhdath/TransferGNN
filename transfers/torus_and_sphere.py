import networkx as nx
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import math
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import torch.nn.functional as F
import os 
from transfers.utils import generate_graph
import argparse
from torch_geometric.data import DataLoader, Data, DataListLoader
from torch_geometric.nn import GINConv, global_mean_pool, JumpingKnowledge
from torch.nn import Sequential, Linear, ReLU
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN

def sample_sphere(num_nodes):
    N = num_nodes

    phi = np.random.uniform(low=0,high=2*np.pi, size=N)
    costheta = np.random.uniform(low=-1,high=1,size=N)
    u = np.random.uniform(low=0,high=1,size=N)

    theta = np.arccos( costheta )
    r = 1.0

    x = r * np.sin( theta) * np.cos( phi )
    y = r * np.sin( theta) * np.sin( phi )
    z = r * np.cos( theta )

    return torch.tensor(list(zip(x, y, z)))

def sample_torus(R, r, n_nodes):
    angle = np.linspace(0, 2*np.pi, 32)
    theta, phi = np.meshgrid(angle, angle)
    X = (R + r * np.cos(phi)) * np.cos(theta)
    Y = (R + r * np.cos(phi)) * np.sin(theta)  
    Z = r * np.sin(phi)

    ps_x = []
    ps_y = []
    ps_z = []
    for _ in range(n_nodes):
        u = random.random()
        v = random.random()
        w = random.random()
        omega = 2*np.pi*u
        theta = 2*np.pi*v 
        threshold = (R + r*math.cos(omega))/(R+r)
        if w <= threshold:
            x = (R+r*math.cos(omega))*math.cos(theta)
            y = (R+r*math.cos(omega))*math.sin(theta)
            z = r*math.sin(omega)
            ps_x.append(x)
            ps_y.append(y)
            ps_z.append(z)
    return torch.tensor(list(zip(ps_x, ps_y, ps_z)))

def generate_graph_with_noise(features, kind="sigmoid", k=5, noise=0.0):
    adj = generate_graph(features, kind, k, log=False)
    if noise > 0:
        n_added_edges = int(len(adj)**2 * noise)
        no_edge_index = np.argwhere(adj == 0)
        add_edge_index = np.random.permutation(no_edge_index)[:n_added_edges]
        adj[add_edge_index[:,0], add_edge_index[:,1]] = 1
        src, trg = adj.nonzero()
        edge_index = np.concatenate([src.reshape(-1,1), trg.reshape(-1,1)], axis=1)
        print(f"Random add {n_added_edges} edges")
    return adj

def generate_torus_and_sphere():
    print("Generate torus & sphere")
    pre_dataset = []
    dataset = []
    # Generate sphere graphs
    for _ in range(args.num_graphs//2):
        n_nodes = np.random.randint(100, 200)
        features = sample_sphere(n_nodes)
        pre_dataset.append((features, 0))
    for _ in range(args.num_graphs//2):
        n_nodes = np.random.randint(100, 200)
        features = sample_torus(80, 40, n_nodes) / 120
        pre_dataset.append((features, 1))
    # Split train-val-test before build f function on features
    inds = np.random.permutation(len(pre_dataset)).tolist()
    pre_dataset = [pre_dataset[x] for x in inds]
    n_train = int(len(pre_dataset)*0.8)
    n_val = int(len(pre_dataset)*0.1)
    train_predataset = pre_dataset[:n_train]
    val_predataset = pre_dataset[n_train:n_train+n_val]
    test_predataset = pre_dataset[n_train+n_val:]

    # Build f function
    train_dataset = []
    val_dataset = []
    test_dataset = []
    for features, label in train_predataset:
        adj = generate_graph_with_noise(features, kind=args.from_data, k=args.from_k, noise=args.from_noise)
        features = torch.FloatTensor(features)
        labels = torch.LongTensor([label])
        src, trg = adj.nonzero()
        edge_index = torch.LongTensor(np.concatenate([src.reshape(1, -1), trg.reshape(1,-1)], axis=0))
        new_data = Data(x=features, y=labels, edge_index=edge_index)
        train_dataset.append(new_data)
    for features, label in val_predataset:
        adj = generate_graph_with_noise(features, kind=args.from_data, k=args.from_k, noise=args.from_noise)
        features = torch.FloatTensor(features)
        labels = torch.LongTensor([label])
        src, trg = adj.nonzero()
        edge_index = torch.LongTensor(np.concatenate([src.reshape(1, -1), trg.reshape(1,-1)], axis=0))
        new_data = Data(x=features, y=labels, edge_index=edge_index)
        val_dataset.append(new_data)
    for features, label in test_predataset:
        adj = generate_graph_with_noise(features, kind=args.to_data, k=args.to_k, noise=args.to_noise)
        features = torch.FloatTensor(features)
        labels = torch.LongTensor([label])
        src, trg = adj.nonzero()
        edge_index = torch.LongTensor(np.concatenate([src.reshape(1, -1), trg.reshape(1,-1)], axis=0))
        new_data = Data(x=features, y=labels, edge_index=edge_index)
        test_dataset.append(new_data)
    return train_dataset, val_dataset, test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--num-graphs', type=int, default=1000)
parser.add_argument('--from-data', default='knn')
parser.add_argument('--to-data', default='sigmoid')
parser.add_argument('--from-noise', type=float, default=0.0)
parser.add_argument('--to-noise', type=float, default=0.0)
parser.add_argument('--from-k', type=int, default=5)
parser.add_argument('--to-k', type=int, default=5)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--seed', type=int, default=100)
args = parser.parse_args()

print(args)
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

if args.from_data == "knn":
    args.from_k = 5
elif args.from_data == "sigmoid":
    args.from_k = 10

if args.to_data == "knn":
    args.to_k = 5
elif args.to_data == "sigmoid":
    args.to_k = 10

# num_graphs = 1000
# # num_nodes_per_graph = 500
# graph_method = 'knn'
# k = 5
# noises = [0.0, 0.0001, 0.001, 0.01, 0.1]

train_dataset, val_dataset, test_dataset = generate_torus_and_sphere()
num_features = train_dataset[0].x.shape[1]
num_classes = 2
print(f"Num features: {num_features}")
print(f"Num classes: {num_classes}")
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

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

model_path = f"model/torus-sphere-from-{args.from_data}_{args.from_k}_{args.from_noise}-to-{args.to_data}_{args.to_k}_{args.to_noise}-seed{args.seed}.pkl"
if not os.path.isdir('model'):
    os.makedirs('model')

train_acc = test(train_loader)
val_acc = test(val_loader)
test_acc = test(test_loader)
print('Epoch: 0, Train Acc: {:.3f}, Val Acc: {:.3f} Test Acc: {:.3f}'.format(train_acc, val_acc, test_acc))

best_val_acc = 0
for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)
    if epoch%20 == 0:
        val_acc = test(val_loader)
        if val_acc > best_val_acc:
            torch.save(model.state_dict(), model_path)
            best_val_acc = val_acc
        print('Epoch: {:03d}, Train Loss: {:.7f}, Val Acc: {:.7f}'.format(epoch, train_loss, val_acc))
model.load_state_dict(torch.load(model_path))
test_acc = test(test_loader)
print(f'Test Acc: {test_acc:.3f}')

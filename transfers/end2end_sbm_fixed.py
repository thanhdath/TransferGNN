# -*- coding: utf-8 -*-
from itertools import chain
import os
from sklearn.metrics import f1_score

from torch.utils.data import DataLoader
import torch
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
import argparse
from transfers.utils import gen_graph, generate_graph
from types import SimpleNamespace
from sage import SAGECompletedGraph
import time
import random
import string
import eval.stats
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--lam', type=float, default=1.1)
parser.add_argument('--mu', type=float, default=100)
parser.add_argument('--p', type=int, default=8)
parser.add_argument('--n', type=int, default=32)
parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--kind', type=str, default='sigmoid')
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--n-graphs', type=int, default=128)
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--iter-gnn', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--beta', type=float, default=5)
parser.add_argument('--alpha', type=float, default=1.0)
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class GNN(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_classes,
                 normalize=False,
                 add_loop=True):
        super(GNN, self).__init__()
        self.add_loop = add_loop
        self.conv1 = SAGECompletedGraph(in_channels, hidden_channels, normalize)
        self.conv2 = SAGECompletedGraph(hidden_channels, num_classes, normalize)

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()
        x = F.relu(self.conv1(x, adj, mask, self.add_loop))
        x = F.relu(self.conv2(x, adj, mask, self.add_loop))
        return F.log_softmax(x, dim=1)


class ModelSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        d = args.n*(args.n-1)//2
        self.d = d
        name = ''.join([random.choice(string.ascii_letters + string.digits)
                        for n in range(4)])
        layers = [
            (f'{name}-bn1', nn.BatchNorm1d(d)),
            (f'{name}-linear1', nn.Linear(d, d*2)),
            (f'{name}-relu1', nn.LeakyReLU(0.2)),
            (f'{name}-linear4', nn.Linear(d*2, d)),
        ]
        self.W = nn.Sequential()
        [self.W.add_module(n, l) for n, l in layers]
        self.W.apply(init_weights)

    def forward(self, Xs):
        xs = [torch.FloatTensor(x).to(device) for x in Xs]
        xs = [torch.pdist(x) for x in xs]
        xs = torch.stack(xs, dim=0)
        xs = self.W(xs)
        xs = 1 - torch.sigmoid(xs)  # lower the distance, higher probabilitity of edge
        return xs

    def predict_adj(self, Xs, absolute=True):
        xs = self.forward(Xs)
        if absolute:
            xs[xs < 0.5] = 0
            xs[xs >= 0.5] = 1
        return xs


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        pdist_size = args.n*(args.n-1)//2
        layers = [
            nn.Dropout(0.1),
            nn.Linear(pdist_size, pdist_size*2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(pdist_size*2, pdist_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(pdist_size, 1),
            nn.Sigmoid()
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x).view(-1)


def weighted_cross_entropy_with_logits(logits, targets, beta):
    l = logits
    t = targets
    loss = -(beta*t*torch.log(torch.sigmoid(l)+1e-10) + (1-beta)
             * (1-t)*torch.log(torch.sigmoid(1-l)+1e-10))
    return loss


def f1_adj(pA, A):
    pA[pA >= 0.5] = 1
    pA[pA < 0.5] = 0
    f1 = f1_score(A, pA, average="macro")
    return f1


def f1_gnn(ypred, ytrue):
    ypred = ypred.detach().cpu().numpy().astype(np.int)
    ypred = np.argmax(ypred, axis=1)
    ytrue = ytrue.cpu().numpy().astype(np.int)
    f1 = f1_score(ytrue, ypred, average="micro")
    return f1


def batch_pdist_to_adjs(batch_halfAs):
    batch_adjs = torch.zeros((len(batch_halfAs), args.n, args.n)).to(device)
    for i, halfA in enumerate(batch_halfAs):
        inds = torch.triu(torch.ones(args.n, args.n))
        inds[torch.arange(args.n), torch.arange(args.n)] = 0
        batch_adjs[i][inds == 1] = halfA
        batch_adjs[i] = batch_adjs[i].T + batch_adjs[i]
    return batch_adjs


def halfA_to_A(halfA):
    A = np.zeros((len(X), len(X)))
    inds = torch.triu(torch.ones(len(A), len(A)))
    inds[np.arange(len(A)), np.arange(len(A))] = 0
    A[inds == 1] = halfA
    A = A + A.T
    A[np.arange(len(A)), np.arange(len(A))] = 1
    return A


def A_to_halfA(A):
    inds = torch.triu(torch.ones(len(A), len(A)))
    inds[np.arange(len(A)), np.arange(len(A))] = 0
    halfA = A[inds == 1]
    return halfA


def count_number_of_parameters(model):
    return sum(p.numel() for p in model.parameters())


graphs = []
train_size = args.n_graphs - 20
u = np.random.multivariate_normal(np.zeros((args.p)), np.eye(args.p)/args.p, 1)
print("u", u)
n_edges = []
for i in range(args.n_graphs):
    if i % 100 == 0:
        print(f"{i+1}/{args.n_graphs}")
    Asbm, X, L = gen_graph(n=args.n, p=args.p, lam=args.lam, mu=args.mu, u=u)
    n_edges.append(Asbm.sum())
    graphs.append((None, X, L, Asbm))
train_graphs = graphs[:train_size]
test_graphs = graphs[train_size:]
print(
    f"Number of edges: min {np.min(n_edges)} - max {np.max(n_edges)} - ave {np.mean(n_edges):.2f}")
print("Auto adjust k")
ave_degree = np.mean(n_edges) / args.n
args.k = int(ave_degree)
print(f"Select k = {args.k}")

model1 = ModelSigmoid().to(device)  # function F
print(model1)
print('Number of parameters: ', count_number_of_parameters(model1))
# discriminator = Discriminator().to(device)
# print(discriminator)


def loss_adj_fn(predA, A, dis_smooth=0.2):
    bs = len(predA)
    x = torch.cat([predA, A], dim=0)
    y = torch.FloatTensor(2 * bs).zero_().to(device)
    y[bs:] = dis_smooth  # must classify as fake
    y[:bs] = 1 - dis_smooth  # real
    preds = discriminator(x)
    loss = F.binary_cross_entropy(preds, y)
    return loss


def loss_reverse_distance(predAs):
    #     predAs BxP
    return -torch.pdist(predAs).mean()


def loss_shuffle_features(xs):
    xs = [x.detach().cpu().numpy() for x in xs]
    inds = [np.random.permutation(args.n) for _ in range(len(xs))]
    xs_shuffle = [x[ind] for x, ind in zip(xs, inds)]
    pred_As = model1.predict_adj(xs, absolute=False)
    pred_As_shuffle = model1.predict_adj(xs_shuffle, absolute=False)
    pred_adjs = batch_pdist_to_adjs(pred_As)
    pred_adjs_shuffle = batch_pdist_to_adjs(pred_As_shuffle)

    pred_As_ = torch.stack([pred_adj[ind][:, ind]
                            for pred_adj, ind in zip(pred_adjs, inds)])
    dists = torch.sqrt(torch.sum((pred_As_ - pred_adjs_shuffle)**2)).mean()
    return dists


def get_pred_As(batch_xs, batch_ys, batch_as, training=True):
    if training:  # shuffle
        for i in range(len(batch_xs)):
            shuffle_inds = np.random.permutation(args.n)
            batch_xs[i] = batch_xs[i][shuffle_inds]
            batch_ys[i] = batch_ys[i][shuffle_inds]
            batch_as[i] = batch_as[i][shuffle_inds][:, shuffle_inds]
    pred_As = model1.predict_adj(batch_xs, absolute=False)
    if training:
        batch_halfas = torch.stack(
            [torch.FloatTensor(A_to_halfA(x)).to(device) for x in batch_as])
        loss_adj = loss_adj_fn(pred_As, batch_halfas)
        return pred_As, loss_adj, batch_halfas
    return pred_As, None, None


# learn model1 | end2end model
loss_gnns = []
loss_distances = []
loss_shuffles = []
loss_adjs = []


optim = torch.optim.Adam(model1.parameters(), lr=0.01)  # weight_decay=5e-4
# loss_adj_fn = nn.MSELoss()
loss_adj_fn = nn.BCELoss()
for iter in range(args.epochs):
    model1.eval()
    inds = np.random.permutation(len(train_graphs))[:args.batch_size]
    batch_xs = [train_graphs[x][1] for x in inds]
    batch_ys = [train_graphs[x][2] for x in inds]
    batch_as = [train_graphs[x][3] for x in inds]
    pred_As, _, _ = get_pred_As(batch_xs, batch_ys, batch_as, training=False)

    print("=== Start learn GNN")
    model_gnn = GNN(args.p, args.p*2, 2).to(device)  # use to learn model1
    optim_gnn = torch.optim.Adam(model_gnn.parameters(), lr=0.01)  # weight_decay=5e-4
    batch_xs_torch = torch.FloatTensor(batch_xs).to(device)
    batch_ys_torch = torch.LongTensor(batch_ys).to(device)
    pred_adjs = batch_pdist_to_adjs(pred_As).detach()
    for iter_gnn in range(args.iter_gnn):
        model_gnn.train()
        optim_gnn.zero_grad()
        # A to graphsage
        outputs = model_gnn(batch_xs_torch, pred_adjs)
        loss_gnn = F.nll_loss(outputs.view(-1, 2), batch_ys_torch.view(-1))
        if iter_gnn < args.iter_gnn - 1:  # not backward at the last iteration
            loss_gnn.backward()
            optim_gnn.step()
    print("=== End learn GNN")

    # find loss adj
    model1.train()
    optim.zero_grad()
    pred_As, loss_adj, batch_halfas = get_pred_As(
        batch_xs, batch_ys, batch_as, training=True)
    # loss_distance = loss_reverse_distance(pred_As)  # prevent Adj -> I
    # loss_shuffle = loss_shuffle_features(batch_xs[:4])
    loss = loss_adj + loss_gnn  # + loss_shuffle*0.1
#     loss = loss_gnn + loss_adj

    if iter % 100 == 0 and iter_gnn == args.iter_gnn - 1:
        print(pred_As[0, :10])

    loss.backward()
    optim.step()

    loss_adjs.append(loss_adj.item())
    loss_gnns.append(loss_gnn.item())
    # loss_distances.append(loss_distance.item())
    # loss_shuffles.append(loss_shuffle.item())

#     evaluate
    if iter % 100 == 0:
        model1.eval()
        model_gnn = GNN(args.p, args.p*2, 2).to(device)
        with torch.no_grad():
            pred_As, _, _ = get_pred_As(
                batch_xs[:5] + [x[1] for x in test_graphs],
                batch_ys[:5] + [x[2] for x in test_graphs],
                [x.cpu().numpy() for x in batch_halfas[:5]] +
                [A_to_halfA(x[3]) for x in test_graphs],
                training=False
            )
            # acc gnn
            batch_xs = torch.FloatTensor(batch_xs).to(device)
            batch_ys = torch.LongTensor(batch_ys).to(device)
            pred_adjs = batch_pdist_to_adjs(pred_As)
            optim_gnn = torch.optim.Adam(
                model_gnn.parameters(), lr=0.005)  # weight_decay=5e-4
            for iter_gnn in range(args.iter_gnn):
                model_gnn.train()
                optim_gnn.zero_grad()
            #     pred_As to

                if iter % 100 == 0 and iter_gnn == args.iter_gnn - 1:
                    print(pred_As[0, :10])
                # A to graphsage
                outputs = model_gnn(batch_xs, pred_adjs)
                loss_gnn = F.nll_loss(outputs.view(-1, 2), batch_ys.view(-1))
                if iter_gnn < args.iter_gnn - 1:  # not backward at the last iteration
                    loss_gnn.backward()
                    optim_gnn.step()

            print(pred_adjs.sum() / len(pred_adjs))
            fs_gnn = [f1_gnn(output, y) for output, y in zip(outputs, batch_ys)]
            fs_gnn_str = " ".join([f"{f1:.2f}" for f1 in fs_gnn])

            print(f"Iter {iter} - loss_adj {loss_adj:.3f} loss_gnn {loss_gnn:.3f} - loss_dist {loss_distance:.3f} - loss_shuffle {loss_shuffle:.4f} - gnn {fs_gnn_str}")


# learn model1 done, frozen model1
for i in model1.parameters():
    model1.requires_grad = False
model1.eval()
# generate adj for all graphs
with torch.no_grad():
    print("u", u)
    for i in range(args.n_graphs):
        _, X, L, Asbm = graphs[i]
        halfA = model1.predict_adj([X], absolute=True)[0].cpu().numpy()
        A = halfA_to_A(halfA)
        graphs[i] = (A, X, L, Asbm)

train_graphs = graphs[:train_size]
test_graphs = graphs[train_size:]

# evaluate MMD


def evaluate_mmd(graphs_real, graphsF):
    # convert graph to networkx
    import networkx as nx  # networkx==1.11
    graphs_real = [nx.from_numpy_matrix(x) for x in graphs_real]
    graphsF = [nx.from_numpy_matrix(x) for x in graphsF]
    mmd_degree = eval.stats.degree_stats(graphs_real, graphsF)
    mmd_clustering = eval.stats.clustering_stats(graphs_real, graphsF)
    try:
        mmd_4orbits = eval.stats.orbit_stats_all(graphs_real, graphsF)
    except:
        mmd_4orbits = -1
    print(f'MMD degree: {mmd_degree:.3f}')
    print(f'MMD clustering: {mmd_clustering:.3f}')
    print(f'MMD 4orbits: {mmd_4orbits:.3f}')


evaluate_mmd(
    [x[-1] for x in test_graphs],
    [x[0] for x in test_graphs]
)


def save_graphs(A, X, L, outdir):
    print(f"\n==== Save graphs to {outdir}")
    dataname = outdir.split("/")[-1]
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    np.savetxt(outdir + f"/{dataname}.txt", A, fmt="%.4f", delimiter=" ")
    with open(outdir + "/labels.txt", "w+") as fp:
        for i, label in enumerate(L):
            fp.write(f"{i} {label}\n")
    np.savez_compressed(outdir + "/features.npz", features=X)
    print("=== Done ===\n")


# G(Atrain, Xtrain), Atrain = Asbm, Xtrain = Xsbm
Af, X, L, Asbm = train_graphs[0]
print('Atrain-Xtrain')
print(f"Number of edges: {Asbm.sum()}")
save_graphs(Asbm, X, L, f"data-transfers/synf-sbm-seed{args.seed}/Atrain-Xtrain")

print('AtrainF-Xtrain')
print(f"Number of edges: {Af.sum()}")
save_graphs(Af, X, L, f"data-transfers/synf-sbm-seed{args.seed}/AtrainF-Xtrain")

print('A2-Xtest')  # A2 = Af
Af, X, L, _ = test_graphs[0]
save_graphs(Af, X, L, f"data-transfers/synf-sbm-seed{args.seed}/A2-Xtest")

print('A3-Xtest')  # A3 = KNN(X)
_, X, L, _ = test_graphs[0]
A3 = generate_graph(torch.FloatTensor(X), kind="knn", k=args.k)
save_graphs(A3, X, L, f"data-transfers/synf-sbm-seed{args.seed}/A3-Xtest")

print('A4-Xtest')  # A4 = Sigmoid(X)
_, X, L, _ = test_graphs[0]
A4 = generate_graph(torch.FloatTensor(X), kind="sigmoid", k=args.k)
save_graphs(A4, X, L, f"data-transfers/synf-sbm-seed{args.seed}/A4-Xtest")

print('Atest-Xtest')  # A4 = Sigmoid(X)
_, X, L, Asbm = test_graphs[0]
save_graphs(Asbm, X, L, f"data-transfers/synf-sbm-seed{args.seed}/Atest-Xtest")

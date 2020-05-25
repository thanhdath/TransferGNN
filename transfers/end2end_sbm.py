# -*- coding: utf-8 -*-
from itertools import chain
import os
from sklearn.metrics import f1_score
import networkx as nx
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
parser.add_argument('--n-graphs', type=int, default=300)
parser.add_argument('--epochs', type=int, default=5000)
parser.add_argument('--batch-size', type=int, default=16)
args = parser.parse_args()

# from types import SimpleNamespace
# args = {
#     "lam": 0,
#     "mu": 0,
#     "p": 8,
#     "n": 128,
#     "seed": 100,
#     "n_graphs": 300,
#     "epochs": 10000,
#     "batch_size": 16
# }
# args= SimpleNamespace(**args)

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
        self.conv2 = SAGECompletedGraph(hidden_channels, hidden_channels, normalize)
        self.conv3 = SAGECompletedGraph(hidden_channels, num_classes, normalize)

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()
        x = F.relu(self.conv1(x, adj, mask, self.add_loop))
        x = F.relu(self.conv2(x, adj, mask, self.add_loop))
        x = F.relu(self.conv3(x, adj, mask, self.add_loop))
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

from torch.autograd import Variable
class FocalLoss(nn.modules.loss._WeightedLoss):

    def __init__(self, gamma=2, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean', balance_param=0.25):
        super(FocalLoss, self).__init__(weight, size_average, reduce, reduction)
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.balance_param = balance_param

    def forward(self, input, target):
        
        # inputs and targets are assumed to be BatchxClasses
        assert len(input.shape) == len(target.shape)
        assert input.size(0) == target.size(0)
        assert input.size(1) == target.size(1)
        
        weight = Variable(self.weight)
           
        # compute the negative likelyhood
        logpt = - F.binary_cross_entropy_with_logits(input, target, pos_weight=weight, reduction=self.reduction)
        pt = torch.exp(logpt)

        # compute the loss
        focal_loss = -( (1-pt)**self.gamma ) * logpt
        balanced_focal_loss = self.balance_param * focal_loss
        return balanced_focal_loss
    
graphs = []
train_size = args.n_graphs - 32
u = np.random.multivariate_normal(np.zeros((args.p)), np.eye(args.p)/args.p, 1)
print("u", u)
n_edges = []
n_positive_pdists = []
for i in range(args.n_graphs):
    if i % 100 == 0:
        print(f"{i+1}/{args.n_graphs}")
    Asbm, X, L = gen_graph(n=args.n, p=args.p, lam=args.lam, mu=args.mu, u=u)
    Asbm[np.arange(len(Asbm)), np.arange(len(Asbm))] = 0
    n_edges.append(Asbm.sum())
    halfAsbm = A_to_halfA(Asbm)
    n_positive_pdists.append(halfAsbm.sum())
    graphs.append((None, X, L, Asbm))
train_graphs = graphs[:train_size]
test_graphs = graphs[train_size:]
print(f"Number of edges: min {np.min(n_edges)} - max {np.max(n_edges)} - ave {np.mean(n_edges):.2f}")
print("Auto adjust k")
ave_degree = np.mean(n_edges) / args.n
args.k = int(ave_degree)
print(f"Select k = {args.k}")

model1 = ModelSigmoid().to(device)  # function F
print(model1)
print('Number of parameters: ', count_number_of_parameters(model1))
model_gnn = GNN(args.p, args.p*2, 2).to(device)  # use to learn model1
print(model_gnn)
# discriminator = Discriminator().to(device)
# print(discriminator)

optim = torch.optim.Adam(
    [
        {'params': model1.parameters(), 'lr': 0.01},
        {'params': model_gnn.parameters()},
        # {'params': discriminator.parameters(), 'lr': 5e-4}
    ],
    lr=0.01
)


weight_zeros= np.mean(n_positive_pdists) / (args.n * (args.n-1)//2) 
print("Weight zeros: ", weight_zeros)
def loss_adj_fn(predA, A):
    weight = torch.ones(A.shape)
    weight[A == 0] = weight_zeros
    return F.binary_cross_entropy(predA, A, 
#                                   weight=weight.to(device)
                                 )

# loss_adj_fn = FocalLoss(weight=torch.ones((args.batch_size, args.n*(args.n-1)//2))).cuda()

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


# learn model1 | end2end model
loss_gnns = []
loss_distances = []
loss_shuffles = []
loss_adjs = []

# loss_adj_fn = nn.MSELoss()
# loss_adj_fn = nn.BCELoss()

for iter in range(args.epochs):
    model1.train()
    model_gnn.train()
    optim.zero_grad()
    inds = np.random.permutation(len(train_graphs))[:args.batch_size]
    batch_xs = [train_graphs[x][1] for x in inds]
    batch_ys = [train_graphs[x][2] for x in inds]
    batch_as = [train_graphs[x][3] for x in inds]
    for i in range(len(batch_xs)):
        shuffle_inds = np.random.permutation(args.n)
        batch_xs[i] = batch_xs[i][shuffle_inds]
        batch_ys[i] = batch_ys[i][shuffle_inds]
        batch_as[i] = batch_as[i][shuffle_inds][:, shuffle_inds]
    batch_ys = torch.LongTensor(batch_ys).to(device)
    pred_As = model1.predict_adj(batch_xs, absolute=False)
    batch_halfas = torch.stack(
        [torch.FloatTensor(A_to_halfA(x)).to(device) for x in batch_as])
#     batch_as = torch.stack([torch.FloatTensor(x) for x in batch_as]).to(device)
    loss_adj = loss_adj_fn(pred_As, batch_halfas)
#     loss_adj = weighted_cross_entropy_with_logits(pred_As, batch_halfas, beta=0.8)

    batch_xs = torch.FloatTensor(batch_xs).to(device)
#     pred_As to
    pred_adjs = batch_pdist_to_adjs(pred_As)
    if iter % 100 == 0:
        print(pred_As[0, :10])
    # A to graphsage
    outputs = model_gnn(batch_xs, pred_adjs)
    loss_gnn = F.nll_loss(outputs.view(-1, 2), batch_ys.view(-1))
    loss_distance = loss_reverse_distance(pred_As)  # prevent Adj -> I
    loss_shuffle = loss_shuffle_features(batch_xs[:4])

    assert not torch.isnan(pred_adjs).any(), "Pred adjs contains nan"
    assert not torch.isinf(pred_adjs).any(), "Pred adjs inf"
    if torch.isnan(pred_adjs.sum()):
        print(pred_adjs)

#     reg_loss = torch.sqrt((pred_adjs.sum() / args.batch_size) - np.mean(n_edges))
    reg_loss = torch.sqrt(
        torch.sum((pred_As.sum(dim=[1]) - batch_halfas.sum(dim=[1]))**2)
    )
    loss = loss_adj + loss_distance + loss_gnn  # + loss_shuffle*0.1
#     loss += reg_loss*0.001
#     loss = loss_gnn + loss_adj

    loss.backward()
    optim.step()

    loss_adjs.append(loss_adj.item())
    loss_gnns.append(loss_gnn.item())
    loss_distances.append(loss_distance.item())
    loss_shuffles.append(loss_shuffle.item())

#     evaluate
    if iter % 100 == 0:
        model1.eval()
        model_gnn.eval()
        with torch.no_grad():
            batch_xs = [x.cpu().numpy() for x in batch_xs[:5]] + [x[1]
                                                                  for x in test_graphs]
            batch_ys = [x.cpu().numpy() for x in batch_ys[:5]] + [x[2]
                                                                  for x in test_graphs]
            batch_halfas = [x.cpu().numpy() for x in batch_halfas[:5]] + \
                [A_to_halfA(x[3]) for x in test_graphs]
            pred_As = model1.predict_adj(batch_xs, absolute=False)
            # acc gnn
            batch_xs = torch.FloatTensor(batch_xs).to(device)
            batch_ys = torch.LongTensor(batch_ys).to(device)
            pred_adjs = batch_pdist_to_adjs(pred_As)

#             pred adjs to nx
            graphs_nx = [nx.from_numpy_matrix(x.cpu().numpy()) for x in pred_adjs[:]]
            print("Components: ", [len(list(nx.connected_components(x))) for x in graphs_nx])

            print(f"Edges: {pred_adjs.sum() / len(pred_adjs):.2f}", )
#             print(pred_adjs[0])
            outputs = model_gnn(batch_xs, pred_adjs)
            fs_gnn = [f1_gnn(output, y) for output, y in zip(outputs[:5], batch_ys[:5])]
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
        halfA = model1.predict_adj([X], absolute=False)[0].cpu().numpy()
        A = halfA_to_A(halfA)
        graphs[i] = (A, X, L, Asbm)

train_graphs = graphs[:train_size]
test_graphs = graphs[train_size:]


import networkx as nx
from networkx.readwrite import json_graph
import json

def save_graphs_list(As, Xs, Ls, savedir):
    idx = 0
    graph_ids = []
    graph_id = 0
    edges_all = []
    for A, X, L in zip(As, Xs, Ls):
        edge_index = np.argwhere(A > 0)
        edges_weight = A[edge_index[:,0], edge_index[:,1]]
        edge_index += idx
        edges = np.concatenate([edge_index, edges_weight.reshape(-1, 1)], axis=1)
        edges_all.append(edges)
        graph_ids += [graph_id] * len(X)
        idx += len(X)
        graph_id += 1
    edges_all = np.concatenate(edges_all, axis=0)
    graph_ids = np.array(graph_ids, dtype=np.int)
    Xs = np.concatenate(Xs, axis=0)
    Ls = np.concatenate(Ls, axis=0).astype(np.int)
    if len(Ls.shape) == 1:
        Ls = Ls.reshape(-1, 1)
    adj = np.zeros((len(Xs), len(Xs)))
    adj[edges_all[:, 0].astype(np.int), edges_all[:, 1].astype(np.int)] = edges_all[:,2]
    G = nx.from_numpy_matrix(adj)
    print(f"""
        Info:
            - Number of nodes: {G.number_of_nodes()}
            - Number of edges: {G.number_of_edges()}
            - Xs shape: {Xs.shape}
            - Ls shape: {Ls.shape}
            - graph_ids shape: {graph_ids.shape}
    """)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    G_data = json_graph.node_link_data(G)
    dataname = savedir.split("/")[-1]
    with open(f"{savedir}/{dataname}_graph.json", "w+") as fp:
        fp.write(json.dumps(G_data))
    np.save(savedir + "/feats.npy", Xs)
    np.save(savedir + "/labels.npy", Ls)
    np.save(savedir + "/graph_id.npy", graph_ids)

def save_multiple_validation_graphs(savepath, seed, train_graphs, test_graphs):
  n_edges = [x[0].sum() for x in train_graphs]
  k = int(np.mean(n_edges)) / train_graphs[0][1].shape[0]
  print('Atrain-Xtrain')
  save_graphs_list(
      [x[3] for x in train_graphs],
      [x[1] for x in train_graphs],
      [x[2] for x in train_graphs],
      f"{savepath}/synf-seed{seed}-multigraphs/Atrain-Xtrain"
  )

  print('AtrainF-Xtrain')
  save_graphs_list(
      [x[0] for x in train_graphs],
      [x[1] for x in train_graphs],
      [x[2] for x in train_graphs],
      f"{savepath}/synf-seed{seed}-multigraphs/AtrainF-Xtrain"
  )

  print('Atest-Xtest')
  save_graphs_list(
      [x[3] for x in test_graphs],
      [x[1] for x in test_graphs],
      [x[2] for x in test_graphs],
      f"{savepath}/synf-seed{seed}-multigraphs/Atest-Xtest"
  )

  print('A2-Xtest')  # A2 = Af
  save_graphs_list(
      [x[0] for x in test_graphs],
      [x[1] for x in test_graphs],
      [x[2] for x in test_graphs],
      f"{savepath}/synf-seed{seed}-multigraphs/A2-Xtest"
  )

  print('A3-Xtest')  # A3 = KNN(X)
  A3_graphs = []
  for _, X, L, A in test_graphs:
      A3 = generate_graph(torch.FloatTensor(X), kind="knn", k=k)
      A3_graphs.append((A3, X, L, A))
  save_graphs_list(
      [x[0] for x in A3_graphs],
      [x[1] for x in A3_graphs],
      [x[2] for x in A3_graphs],
      f"{savepath}/synf-seed{seed}-multigraphs/A3-Xtest"
  )

  print('A4-Xtest')  # A4 = Sigmoid(X)
  A4_graphs = []
  for _, X, L, A in test_graphs:
      A4 = generate_graph(torch.FloatTensor(X), kind="sigmoid", k=k)
      A4_graphs.append((A4, X, L, A))
  save_graphs_list(
      [x[0] for x in A4_graphs],
      [x[1] for x in A4_graphs],
      [x[2] for x in A4_graphs],
      f"{savepath}/synf-seed{seed}-multigraphs/A4-Xtest"
  )


save_multiple_validation_graphs(f"data/gen-sbm-n{args.n}-p{args.p}-lam{args.lam}-mu{args.mu}", args.seed, train_graphs, test_graphs)

import torch
import numpy as np
import networkx as nx
import torch.nn.functional as F
import random
from networkx.readwrite import json_graph
import json
import argparse
import torch.nn as nn
from tqdm import tqdm
import scipy.stats
import os
import matplotlib.pyplot as plt
import random
import pickle

import multiprocessing
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
    workers=multiprocessing.cpu_count()//2):
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

def convert_nx_repr(graph):
    new_graph = nx.Graph(features=0)
    index_map = {}
    new_ind = 0
    for node in graph.nodes(data=True):
        if graph.degree(node[0]) == 0:
            continue
        index_map[node[0]] = new_ind
        new_graph.add_node(new_ind, features=node[1]['features'], label=node[1]['label'])
        # new_graph.add_edge(new_ind, new_ind, features=1) # features = 1 for true edge
        new_ind += 1
    for edge in graph.edges(data=True):
        new_graph.add_edge(index_map[edge[0]], index_map[edge[1]], features=1)
    # new_graph = nx.to_undirected(new_graph)
    return new_graph

def create_graphs(config):
    graphs = []
    p = config.p
    u = np.random.multivariate_normal(np.zeros((p)), np.eye(p) / p, 1)
    n = config.n
    for i in range(config.n_graphs):
        Asbm, X, L = gen_graph(n=n, p=p, lam=config.lam, mu=config.mu, u=u)
        L_onehot = np.zeros((len(L), 2), dtype=np.float32)
        L_onehot[:, 0][L==0] = 1
        L_onehot[:, 1][L==1] = 1
        X = X.astype(np.float32)
        G = nx.from_numpy_matrix(Asbm)
        if config.cat_deepwalk:
            extra_features = deepwalk(G, 32, number_walks=40, walk_length=80)
        for id, node in enumerate(G.nodes()):
            if config.cat_deepwalk:
                G.nodes()[node]['features'] = np.array(X[id].tolist() + extra_features[node].tolist(), dtype=np.float32)
            else:
                G.nodes()[node]['features'] = X[id]
            G.nodes()[node]['label'] = L_onehot[id]
        G.remove_edges_from(nx.selfloop_edges(G))
        G = convert_nx_repr(G)
        graphs.append(G)
    return graphs


def gen_graph(n=200, p=128, lam=1.0, mu=0.3, u=None):
    v = [1] * (n // 2) + [-1] * (n // 2)
    random.shuffle(v)
    d = 5
    """# Generate B (i.e. X)"""
    if u is None:
        u = np.random.multivariate_normal(np.zeros((p)), np.eye(p) / p, 1)
    Z = np.random.randn(n, p)
    B = np.zeros((n, p))

    for i in range(n):
        a = np.sqrt(mu / n) * v[i] * u
        b = Z[i] / np.sqrt(p)
        B[i, :] = a + b

    """# Generate A"""
    c_in = d + lam * np.sqrt(d)
    c_out = d - lam * np.sqrt(d)

    p_A = np.zeros((n, n))
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if v[i] == v[j]:
                p_A[i, j] = c_in / n
            else:
                p_A[i, j] = c_out / n
            if np.random.rand() <= p_A[i, j]:
                A[i, j] = 1
                A[j, i] = 1
    labels = np.array(v)
    labels[labels == -1] = 0
    return A, B, labels

parser = argparse.ArgumentParser()
parser.add_argument('--lam', type=float, default=1.1)
parser.add_argument('--mu', type=float, default=100)
parser.add_argument('--p', type=int, default=8)
parser.add_argument('--n', type=int, default=32)
parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--cat-deepwalk', action='store_true')
parser.add_argument('--n-graphs', type=int, default=300)
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

graphs = create_graphs(args)
n_train = int(len(graphs)*0.8)
train_graphs = graphs[:n_train]
test_graphs = graphs[n_train:]

save_data = {
    "train_graphs": train_graphs,
    "test_graphs": test_graphs,
    "multilabel": False,
    "n_classes": 2
}

if args.cat_deepwalk:
    filepath = f"data-sbm/n{args.n}-p{args.p}-lam{args.lam}-mu{args.mu}-deepwalk-seed{args.seed}.pkl"
else:
    filepath = f"data-sbm/n{args.n}-p{args.p}-lam{args.lam}-mu{args.mu}-seed{args.seed}.pkl"

pickle.dump(save_data, open(filepath, "wb"))

import numpy as np
import random
import torch
import torch.nn.functional as F

def gen_graph(n=200, p=128, lam=1.0, mu=0.3, u=None):
    v = [1]*(n//2) + [-1]*(n//2)
    random.shuffle(v)
    d = 5
    """# Generate B (i.e. X)"""
    if u is None:
        u = np.random.multivariate_normal(np.zeros((p)), np.eye(p)/p, 1)
    Z = np.random.randn(n, p)
    B = np.zeros((n, p))

    for i in range(n):
        a = np.sqrt(mu/ n)*v[i]*u
        b = Z[i]/np.sqrt(p)
        B[i,:] =  a + b
    
    """# Generate A"""
    c_in = d + lam*np.sqrt(d)
    c_out = d - lam*np.sqrt(d)

    p_A = np.zeros((n, n))
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if v[i] == v[j]:
                p_A[i,j] = c_in / n
            else:
                p_A[i,j] = c_out / n
            if np.random.rand() <= p_A[i, j]:
                A[i,j] = 1
                A[j,i] = 1
    labels = np.array(v)
    labels[labels == -1] = 0
    return A, B, labels

def generate_graph(features, kind="sigmoid", k=5, log=True):
    features_norm = F.normalize(features, dim=1))
    N = len(features_norm)
    scores = torch.pdist(features_norm)
    if log:
        print(f"Generate graph using {kind}")
    if kind == "sigmoid":
        scores = 1 - torch.sigmoid(scores)
        # find index to cut 
        n_edges = int((k*N - N)/2)
        threshold = scores[torch.argsort(-scores)[n_edges]]
        if log:
            print(f"Scores range: {scores.min():.3f}-{scores.max():.3f}")
            print(f"Expected average degree: {k} => Threshold: {threshold:.3f}")
        edges = scores >= threshold
        adj = np.zeros((len(features), len(features)), dtype=np.int)
        inds = torch.triu(torch.ones(len(adj),len(adj))) 
        inds[np.arange(len(adj)), np.arange(len(adj))] = 0
        adj[inds == 1] = edges.cpu().numpy().astype(np.int)
        adj = adj + adj.T
        adj[adj > 0] = 1
    elif kind == "knn":
        k = int(k)
        if log:
            print(f"Knn k = {k}")
        scores_matrix = np.zeros((len(features), len(features)))
        inds = torch.triu(torch.ones(len(features),len(features))) 
        inds[np.arange(len(features)), np.arange(len(features))] = 0
        scores_matrix[inds == 1] = scores
        scores_matrix = scores_matrix + scores_matrix.T
        if len(scores_matrix) > 60000: # avoid memory error
            edge_index = []
            for i, node_scores in enumerate(scores_matrix):
                candidate_nodes = np.argsort(node_scores)[:k]
                edge_index += [[i, node] for node in candidate_nodes]
            edge_index = np.array(edge_index, dtype=np.int32)
        else:
            sorted_scores = np.argsort(scores_matrix, axis=1)[:, :k]
            edge_index = np.zeros((len(scores_matrix)*k, 2), dtype=np.int32)
            N = len(scores_matrix)
            for i in range(k):
                edge_index[i*N:(i+1)*N, 0] = np.arange(N)
                edge_index[i*N:(i+1)*N, 1] = sorted_scores[:, i]
        adj = np.zeros((len(features), len(features)), dtype=np.int)
        adj[edge_index[:,0], edge_index[:,1]] = 1
    else:
        raise NotImplementedError
    if log:
        print("Number of edges: ", adj.sum())
    return adj

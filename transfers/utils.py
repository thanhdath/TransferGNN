import numpy as numpy
import random

def gen_graph(n=200, p=128, lam=1.0, mu=0.3):
    v = [1]*(n//2) + [-1]*(n//2)
    random.shuffle(v)
    d = 5
    """# Generate B (i.e. X)"""
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

    for i in range(n):
        for j in range(n):
            if v[i] == v[j]:
                p_A[i,j] = c_in / n
            else:
                p_A[i,j] = c_out / n

    p_samples = np.random.sample((n,n))
    A = np.zeros((n,n))
    A[p_A > p_samples] = 1
    labels = np.array(v)
    labels[labels == -1] = 0
    return A, B, labels

def generate_graph(features, kind="sigmoid", threshold=None, k=5):
    features_norm = F.normalize(features, dim=1)
    scores = features_norm.mm(features_norm.t())
    print(f"Generate graph using {kind}")
    if kind == "sigmoid":
        scores = torch.sigmoid(scores)
        if threshold is None:
            threshold = scores.mean()
        print(f"Scores range: {scores.min()}-{scores.max()}")
        print("Threshold: ", threshold)
        adj = scores > threshold
        adj = adj.int()
        edge_index = adj.nonzero().cpu().numpy()
    elif kind == "knn":
        print(f"Knn k = {k}")
        if len(scores) > 60000: # avoid memory error
            edge_index = []
            for i, node_scores in enumerate(scores):
                candidate_nodes = torch.argsort(-node_scores)[:k]
                edge_index += [[i, node] for node in candidate_nodes]
            edge_index = np.array(edge_index, dtype=np.int32)
        else:
            sorted_scores = torch.argsort(-scores, dim=1)[:, :k]
            edge_index = np.zeros((len(scores)*k, 2))
            N = len(scores)
            for i in range(k):
                edge_index[i*N:(i+1)*N, 0] = np.arange(N)
                edge_index[i*N:(i+1)*N, 1] = sorted_scores[:, i]
    else:
        raise NotImplementedError
    
    print("Number of edges: ", edge_index.shape[0])
    return edge_index

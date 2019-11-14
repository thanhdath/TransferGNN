import torch 
import numpy as np
import networkx as nx 
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer
from torch_geometric.data import NeighborSampler, Data
import re

labels2int = {
    "Determiner": 0,
    "Quantifier": 1,
    "Prepositions": 2,
    "Conjunction": 3,
    "Modal": 4,
    "Pronoun": 5
}

def load_adj(adj_file):
    adj = []
    with open(adj_file) as fp:
        for line in fp:
            elms = line.split()
            adj.append(elms)
    adj = np.array(adj, dtype=int)
    return adj

def load_graph(adj_file, feature_file, label_file):
    adj = load_adj(adj_file)
    nodes = np.arange(len(adj))
    
    node2label = {}
    multiclass = False
    for i, line in enumerate(open(label_file)):
        line = re.split("\s+", line.strip())
        node = i
        label = [labels2int[i] for i in line[1:]]
        if len(label)>1: multiclass = True
        node2label[node] = label
    print("Multiclass: ", multiclass)

    labels =[node2label[node] for node in nodes]
    if not multiclass:
        labels = [i[0] for i in labels]
        y = torch.LongTensor(labels)
    else:
        mlb = MultiLabelBinarizer()
        labels = mlb.fit_transform(labels)
        y = torch.FloatTensor(labels)
    
    x = np.load(feature_file, allow_pickle=True)["features"][()]
    x = torch.FloatTensor(x)
    edge_index = np.argwhere(adj > 0)
    edge_weight = torch.FloatTensor(adj[edge_index[:,0], edge_index[:,1]])
    edge_index = torch.LongTensor(edge_index.T)

    n_nodes = len(nodes)
    n_train = int(0.7*n_nodes)
    n_val = int(0.1*n_nodes)

    train_mask = np.zeros((n_nodes,))
    val_mask = np.zeros((n_nodes,))
    test_mask = np.zeros((n_nodes,))
    train_mask[:n_train] = 1
    val_mask[n_train:n_train+n_val] = 1
    test_mask[n_train+n_val:] = 1

    train_mask = torch.tensor(train_mask, dtype=torch.uint8)
    val_mask = torch.tensor(val_mask, dtype=torch.uint8)
    test_mask = torch.tensor(test_mask, dtype=torch.uint8)

    data = Data(x=x, y=y, edge_index=edge_index, 
        train_mask=train_mask, test_mask=test_mask, val_mask=val_mask,
        edge_weight=edge_weight)
    return data

if __name__ == '__main__':
    data = load_graph("twain_tramp/wan_twain_tramp_1.txt", "features.npz", "labels.txt")

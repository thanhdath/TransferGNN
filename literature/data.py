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
    "Pronoun": 5,
    "null": 6
}

def load_edgelist(adj_file):
    edgelist = []
    with open(adj_file) as fp:
        for line in fp:
            elms = re.split("[,\s]+", line)
            elms = [x for x in elms if len(x) > 0]
            edgelist.append(elms)
    edgelist = np.array(edgelist, dtype=float)
    if edgelist.shape[0] == edgelist.shape[1]:
        # adj
        nodes = np.arange(len(edgelist))
        edge_index = np.argwhere(edgelist > 0)
        edge_weight = edgelist[edge_index[:, 0], edge_index[:, 1]]
        edgelist = np.zeros((len(edge_index), 3))
        edgelist[:, :2] = edge_index
        edgelist[:, 2] = edge_weight
    else:
        edgelist = edgelist.astype(np.int)
        nodes = np.arange(edgelist[:, :2].max() + 1)
    if edgelist.shape[1] == 2:
        edges = np.zeros((len(edgelist), 3))
        edges[:, :2] = edgelist
        edges[:, 2] = 1.
        edgelist = edges
    return edgelist, nodes


def load_label(label_file):
    lines = open(label_file).readlines()
    start_line = 0
    end_line = None
    temp_line = ""
    node2label = {}
    for i in range(len(lines)):
        if "[" in lines[i]:
            start_line = i
        if "]" in lines[i]:
            end_line = i
        temp_line += lines[i]
        if start_line is not None and end_line is not None:
            temp_line = temp_line.replace("[", "").replace("]", "")
            elms = re.split("\s+", temp_line.strip())
            node = elms[0]
            label = [int(x) for x in elms[1:]]
            node2label[node] = label
            temp_line = ""
            start_line = None
            end_line = None
    return node2label, True


def load_graph(adj_file, feature_file, label_file, multiclass=None):
    edges, nodes = load_edgelist(adj_file)
    node2label = {}
    multiclass_found = False
    conversion = None

    for i, line in enumerate(open(label_file)):
        line = line.replace("[", "").replace("]", "")
        line = re.split("\s+", line.strip())
        node = i
        # if len(line) == 1:
        #     label = [labels2int["null"]]
        # else:
        #     label = [labels2int[i] for i in line[1:]]
        if conversion is None:
            try:
                [int(x) for x in line[1:]]
                conversion = int
            except:
                conversion = str
        label = [conversion(x) for x in line[1:]]
        if len(label) > 1:
            multiclass_found = True
        node2label[node] = label
    # node2label, multiclass = load_label(label_file)

    if multiclass is None:
        multiclass = multiclass_found
    print("Multiclass: ", multiclass)

    nodes = np.array(list(sorted(node2label.keys())))
    labels = [node2label[node] for node in nodes]
    if not multiclass:
        labels = [i[0] for i in labels]
        y = torch.LongTensor(labels)
    else:
        # mlb = MultiLabelBinarizer()
        # labels = mlb.fit_transform(labels)
        y = torch.FloatTensor(labels)

    x = np.load(feature_file, allow_pickle=True)["features"][()]
    x = torch.FloatTensor(x)
    edge_index = edges[:, :2]
    edge_weight = torch.FloatTensor(edges[:, 2])
    edge_index = torch.LongTensor(edge_index.T)

    n_nodes = len(nodes)
    n_train = int(0.5 * n_nodes)
    # n_val = int(0.2*n_nodes)
    n_val = n_nodes - n_train

    train_mask = np.zeros((n_nodes,))
    val_mask = np.zeros((n_nodes,))
    test_mask = np.zeros((n_nodes,))
    inds = np.random.permutation(n_nodes)
    train_inds = inds[:n_train]
    val_inds = inds[n_train:n_train + n_val]
    test_inds = inds[n_train + n_val:]
    train_mask[train_inds] = 1
    val_mask[val_inds] = 1
    test_mask[test_inds] = 1

    train_mask = torch.tensor(train_mask, dtype=torch.uint8)
    val_mask = torch.tensor(val_mask, dtype=torch.uint8)
    test_mask = torch.tensor(test_mask, dtype=torch.uint8)

    data = Data(x=x, y=y, edge_index=edge_index,
                train_mask=train_mask, test_mask=test_mask, val_mask=val_mask,
                edge_attr=edge_weight)
    return data, multiclass


if __name__ == '__main__':
    data = load_graph("twain_tramp/wan_twain_tramp_1.txt", "features.npz", "labels.txt")

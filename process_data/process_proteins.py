from torch_geometric.datasets import TUDataset
import pickle
import torch
import random
import networkx as nx
import numpy as np
import argparse

def convert_nx_repr(graph):
    new_graph = nx.Graph(features=0)
    index_map = {}
    new_ind = 0
    for node in graph.nodes(data=True):
        if graph.degree(node[0]) == 0:
            continue
        index_map[node[0]] = new_ind
        new_graph.add_node(new_ind, features=node[1]['features'], label=node[1]['label'])
        new_graph.add_edge(new_ind, new_ind, features=1) # features = 1 for true edge
        new_ind += 1
    for edge in graph.edges(data=True):
        new_graph.add_edge(index_map[edge[0]], index_map[edge[1]], features=1)
    # new_graph = nx.to_undirected(new_graph)
    return new_graph

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=100)
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)

name = "PROTEINS_full"
dataset = TUDataset(f'data/{name}', name=name, use_node_attr=True)

n_node_labels = dataset.num_node_labels
# check one-class or multi-class
node_labels = []
for data in dataset:
    node_labels.append(data.x[:, -n_node_labels:])
node_labels = torch.cat(node_labels, dim=0)

if any(node_labels.sum(dim=1) > 1) or any(node_labels.sum(dim=1) == 0):
    multiclass = True
else:
    multiclass = False
n_labels = node_labels.shape[1]

graphs = []
for data in dataset:
    adj = np.zeros((data.num_nodes, data.num_nodes))
    adj[data.edge_index[0], data.edge_index[1]] = 1
    G = nx.from_numpy_matrix(adj)
    features = data.x[:, :-n_node_labels].numpy().astype(np.float32)
    labels = data.x[:, -n_node_labels:].numpy().astype(np.float32)
    # if not multiclass:
    #     labels = np.argmax(labels, axis=1)
    for node in G.nodes():
        G.nodes()[node]['features'] = features[node]
        G.nodes()[node]['label'] = labels[node]
    G.remove_edges_from(nx.selfloop_edges(G))
    G = convert_nx_repr(G)
    graphs.append(G)

print(f"""Info:
    Max nodes: {max([x.number_of_nodes() for x in graphs])}
    Multilabel: {multiclass}
    Num labels: {n_labels}
    Num node attributes: {dataset[0].x.shape[1]-n_labels}
""")

n_train = int(len(graphs) * 0.8)
random.shuffle(graphs)
train_graphs = graphs[:n_train]
test_graphs = graphs[n_train:]

save_data = {
    "train_graphs": train_graphs,
    "test_graphs": test_graphs,
    "multilabel": multiclass,
    "n_classes": n_labels
}

pickle.dump(save_data,
    open(f"data/proteins_full-seed{args.seed}.pkl", "wb"))

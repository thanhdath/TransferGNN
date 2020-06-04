from networkx.readwrite import json_graph
import json
from torch_geometric.datasets import PPI
import pickle
import numpy as np
import networkx as nx

def load_ppi():
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
        return new_graph

    def dataset2graphs(dataset):
        graphs = []
        for data in dataset:
            edge_index = data.edge_index.t().numpy()
            x = data.x.numpy()
            y = data.y.numpy() 
            adj = np.zeros((len(x), len(x)))
            adj[edge_index[:,0], edge_index[:,1]] = 1
            G = nx.from_numpy_matrix(adj)
            for i, node in enumerate(G.nodes()):
                G.nodes()[node]['features'] = x[i].astype(np.float32)
                G.nodes()[node]['label'] = y[i].tolist()
            graphs.append(G)
        graphs = [convert_nx_repr(g) for g in graphs]
        return graphs
    path = "data/PPI"
    train_dataset = PPI(path, split='train')
    val_dataset = PPI(path, split='val')
    test_dataset = PPI(path, split='test')
    train_graphs = dataset2graphs(train_dataset)
    val_graphs = dataset2graphs(val_dataset)
    test_graphs = dataset2graphs(test_dataset)
    return train_graphs, val_graphs, test_graphs

train_graphs, val_graphs, test_graphs = load_ppi()
save_data = {
    "train_graphs": train_graphs+val_graphs,
    "test_graphs": test_graphs,
    "multilabel": True,
    "n_classes": 121
}

pickle.dump(save_data,
    open(f"data/ppi.pkl", "wb"))

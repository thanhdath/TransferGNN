import networkx as nx
from networkx.readwrite import json_graph
import json
import numpy as np
from torch_geometric.datasets import PPI
import argparse
from transfers.utils import gen_graph, generate_graph
import torch
import torch.nn as nn
from tqdm import tqdm
import scipy.stats
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--lam', type=float, default=1.1)
parser.add_argument('--mu', type=float, default=100)
parser.add_argument('--p', type=int, default=8)
parser.add_argument('--n', type=int, default=32)
parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--n-graphs', type=int, default=128)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--nested-batch-size', type=int, default=1024)
parser.add_argument('--evaluation-epochs', type=int, default=10)
parser.add_argument('--num-mix-component', type=int, default=16)
parser.add_argument('--edge-weight', type=float, default=1.0)
parser.add_argument('--weight-decay', type=float, default=5e-6)  # need tuning
parser.add_argument('--gen-multigraph', action='store_true')
parser.add_argument('--data-savepath', default="data-sbm/")
parser.add_argument('--ppi', action='store_true')
parser.add_argument('--export-imgs', action='store_true')
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)


def is_increasing_list(array):
    array = np.array(array)
    suffix = array[1:]
    prefix = array[:-1]
    return ((suffix - prefix) > 0).sum() == suffix.shape[0]


def load_ppi():
    def convert_nx_repr(graph):
        new_graph = nx.DiGraph(features=0)
        index_map = {}
        new_ind = 0
        for node in graph.nodes(data=True):
            index_map[node[0]] = new_ind
            new_graph.add_node(new_ind, features=node[1][
                               'features'], label=node[1]['label'])
            new_graph.add_edge(new_ind, new_ind, features=0)
            new_ind += 1
        for edge in graph.edges(data=True):
            new_graph.add_edge(index_map[edge[0]], index_map[edge[1]], features=0)
        return new_graph

    def dataset2graphs(dataset):
        graphs = []
        for data in dataset:
            edge_index = data.edge_index.t().numpy()
            x = data.x.numpy()
            y = data.y.numpy()
            adj = np.zeros((len(x), len(x)))
            adj[edge_index[:, 0], edge_index[:, 1]] = 1
            G = nx.from_numpy_matrix(adj)
            assert is_increasing_list(G.nodes()), "graph nodes is not increasing list!"
            for i, node in enumerate(G.nodes()):
                G.nodes()[node]['features'] = x[i].tolist()
                G.nodes()[node]['label'] = y[i].tolist()
            graphs.append(G)
        graphs = [convert_nx_repr(g) for g in graphs]
        return graphs
    path = "dataset/"
    train_dataset = PPI(path, split='train')
    val_dataset = PPI(path, split='val')
    test_dataset = PPI(path, split='test')
    train_graphs = dataset2graphs(train_dataset)
    val_graphs = dataset2graphs(val_dataset)
    test_graphs = dataset2graphs(test_dataset)
    return train_graphs, val_graphs, test_graphs


def graphs2data(graphs):
    data = []
    for graph in graphs:
        features = np.array([graph.nodes()[x]['features'] for x in graph.nodes()])
        adj = nx.to_numpy_array(graph)
        data.append((graph, features, adj))
    return data


def save_graphs(A, X, L, outdir, is_adj=True):
    print(f"\n==== Save graphs to {outdir}")
    dataname = outdir.split("/")[-1]
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    if is_adj:
        np.savetxt(outdir + f"/{dataname}.txt", A, fmt="%.4f", delimiter=" ")
    else:
        np.savetxt(outdir + f"/{dataname}.txt", A, fmt="%d", delimiter=" ")
    if len(L.shape) == 1:  # one class
        with open(outdir + "/labels.txt", "w+") as fp:
            for i, label in enumerate(L):
                fp.write(f"{i} {label}\n")
    elif len(L.shape) == 2:
        with open(outdir + "/labels.txt", "w+") as fp:
            for i, label in enumerate(L):
                fp.write("{} {}\n".format(i, " ".join(map(str, label))))
    np.savez_compressed(outdir + "/features.npz", features=X)
    print("=== Done ===\n")


def save_graphs_list(As, Xs, Ls, savedir):
    idx = 0
    graph_ids = []
    graph_id = 0
    edges_all = []
    for A, X, L in zip(As, Xs, Ls):
        edges = np.argwhere(A > 0) + idx
        edges_all.append(edges)
        graph_ids += [graph_id] * len(X)
        idx += len(X)
        graph_id += 1
    edges_all = np.concatenate(edges_all, axis=0)
    graph_ids = np.array(graph_ids, dtype=np.int)
    Xs = np.concatenate(Xs, axis=0)
    Ls = np.concatenate(Ls, axis=0).astype(np.int)
    adj = np.zeros((len(Xs), len(Xs)))
    adj[edges_all[:, 0], edges_all[:, 1]] = 1
    G = nx.from_numpy_matrix(adj)
    # assert create_graphs.is_increasing_list(
    #     G.nodes()), "graph nodes are not increasing list!"
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


def select_most_balanced_graph(graphs):
    entropys = []
    for _, _, L, _ in graphs:
        _, counts = np.unique(L, return_counts=True)
        entropy = scipy.stats.entropy(counts)
        entropys.append(entropy)
    select_ind = np.argmax(entropys)
    print("Labels distribution: ",
          np.unique(graphs[select_ind][2], return_counts=True)[1] / graphs[select_ind][2].shape[0])
    return select_ind


def merge_all_graphs(graphs):
    edgesf_all, Xall, Lall, edges_all = [], [], [], []
    idx = 0
    for Af, X, L, Asbm in graphs:
        Xall.append(X)
        Lall.append(L)
        edges_f = np.argwhere(Af > 0) + idx
        edges = np.argwhere(Asbm > 0) + idx
        edgesf_all.append(edges_f)
        edges_all.append(edges)
        idx += len(X)
    edgesf_all = np.concatenate(edgesf_all, axis=0)
    edges_all = np.concatenate(edges_all, axis=0)
    Xall = np.concatenate(Xall, axis=0)
    Lall = np.concatenate(Lall, axis=0)
    return edgesf_all, Xall, Lall, edges_all

# def save_all_validation_graphs(args, train_graphs, test_graphs, epoch):
#     # graph completion
#     # train_graph_completion(args,dataset_loader,rnn,output)
#     # import sys; sys.exit()
#     n_edges = [x[-1].sum() for x in train_graphs]
#     k = int(np.mean(n_edges)) / train_graphs[0][1].shape[0]
#     # nll evaluation
#     # train_nll(args, dataset_loader, dataset_loader, rnn, output, max_iter = 200, graph_validate_len=graph_validate_len,graph_test_len=graph_test_len)
#     if args.gen_multigraph:
#         print('Atrain-Xtrain')
#         save_graphs_list(
#             [x[3] for x in train_graphs],
#             [x[1] for x in train_graphs],
#             [x[2] for x in train_graphs],
#             f"{args.data_savepath}/synf-seed{args.seed}-epoch{epoch}/Atrain-Xtrain"
#         )

#         print('AtrainF-Xtrain')
#         save_graphs_list(
#             [x[0] for x in train_graphs],
#             [x[1] for x in train_graphs],
#             [x[2] for x in train_graphs],
#             f"{args.data_savepath}/synf-seed{args.seed}-epoch{epoch}/AtrainF-Xtrain"
#         )

#         print('Atest-Xtest')
#         save_graphs_list(
#             [x[3] for x in test_graphs],
#             [x[1] for x in test_graphs],
#             [x[2] for x in test_graphs],
#             f"{args.data_savepath}/synf-seed{args.seed}-epoch{epoch}/Atest-Xtest"
#         )

#         print('A2-Xtest')  # A2 = Af
#         save_graphs_list(
#             [x[0] for x in test_graphs],
#             [x[1] for x in test_graphs],
#             [x[2] for x in test_graphs],
#             f"{args.data_savepath}/synf-seed{args.seed}-epoch{epoch}/A2-Xtest"
#         )

#         print('A3-Xtest')  # A3 = KNN(X)
#         A3_graphs = []
#         for _, X, L, A in test_graphs:
#             A3 = generate_graph(torch.FloatTensor(X), kind="knn", k=k)
#             A3_graphs.append((A3, X, L, A))
#         save_graphs_list(
#             [x[0] for x in A3_graphs],
#             [x[1] for x in A3_graphs],
#             [x[2] for x in A3_graphs],
#             f"{args.data_savepath}/synf-seed{args.seed}-epoch{epoch}/A3-Xtest"
#         )

#         print('A4-Xtest')  # A4 = Sigmoid(X)
#         A4_graphs = []
#         for _, X, L, A in test_graphs:
#             A4 = generate_graph(torch.FloatTensor(X), kind="sigmoid", k=k)
#             A4_graphs.append((A4, X, L, A))
#         save_graphs_list(
#             [x[0] for x in A4_graphs],
#             [x[1] for x in A4_graphs],
#             [x[2] for x in A4_graphs],
#             f"{args.data_savepath}/synf-seed{args.seed}-epoch{epoch}/A4-Xtest"
#         )
#     else:
#         # G(Atrain, Xtrain), Atrain = Asbm, Xtrain = Xsbm
#         # ind_train = select_most_balanced_graph(train_graphs)
#         # ind_test = select_most_balanced_graph(test_graphs)

#         Af, X, L, Asbm = train_graphs[ind_train]
#         print('Atrain-Xtrain')
#         print(f"Number of edges: {Asbm.sum()}")
# save_graphs(Asbm, X, L,
# f"{args.data_savepath}/synf-seed{args.seed}-epoch{epoch}/Atrain-Xtrain")

#         print('AtrainF-Xtrain')
#         print(f"Number of edges: {Af.sum()}")
# save_graphs(Af, X, L,
# f"{args.data_savepath}/synf-seed{args.seed}-epoch{epoch}/AtrainF-Xtrain")

#         print('A2-Xtest')  # A2 = Af
#         Af, X, L, _ = test_graphs[ind_test]
# save_graphs(Af, X, L,
# f"{args.data_savepath}/synf-seed{args.seed}-epoch{epoch}/A2-Xtest")

#         print('A3-Xtest')  # A3 = KNN(X)
#         _, X, L, _ = test_graphs[ind_test]
#         A3 = generate_graph(torch.FloatTensor(X), kind="knn", k=k)
# save_graphs(A3, X, L,
# f"{args.data_savepath}/synf-seed{args.seed}-epoch{epoch}/A3-Xtest")

#         print('A4-Xtest')  # A4 = Sigmoid(X)
#         _, X, L, _ = test_graphs[ind_test]
#         A4 = generate_graph(torch.FloatTensor(X), kind="sigmoid", k=k)
# save_graphs(A4, X, L,
# f"{args.data_savepath}/synf-seed{args.seed}-epoch{epoch}/A4-Xtest")

#         print('Atest-Xtest')  # A4 = Sigmoid(X)
#         _, X, L, Asbm = test_graphs[ind_test]
# save_graphs(Asbm, X, L,
# f"{args.data_savepath}/synf-seed{args.seed}-epoch{epoch}/Atest-Xtest")


def save_single_validation_graphs(args, train_graph, test_graph, suffix, k):
    Af, X, L, Asbm = train_graph
    print('Atrain-Xtrain')
    print(f"Number of edges: {Asbm.sum()}")
    save_graphs(Asbm, X, L, f"{args.data_savepath}/synf-seed{args.seed}-{suffix}/Atrain-Xtrain")

    print('AtrainF-Xtrain')
    print(f"Number of edges: {Af.sum()}")
    save_graphs(Af, X, L, f"{args.data_savepath}/synf-seed{args.seed}-{suffix}/AtrainF-Xtrain")

    print('A2-Xtest')  # A2 = Af
    Af, X, L, _ = test_graph
    save_graphs(Af, X, L, f"{args.data_savepath}/synf-seed{args.seed}-{suffix}/A2-Xtest")

    print('A3-Xtest')  # A3 = KNN(X)
    _, X, L, _ = test_graph
    A3 = generate_graph(torch.FloatTensor(X), kind="knn", k=k)
    save_graphs(A3, X, L, f"{args.data_savepath}/synf-seed{args.seed}-{suffix}/A3-Xtest")

    print('A4-Xtest')  # A4 = Sigmoid(X)
    _, X, L, _ = test_graph
    A4 = generate_graph(torch.FloatTensor(X), kind="sigmoid", k=k)
    save_graphs(A4, X, L, f"{args.data_savepath}/synf-seed{args.seed}-{suffix}/A4-Xtest")

    print('Atest-Xtest')  # A4 = Sigmoid(X)
    _, X, L, Asbm = test_graph
    save_graphs(Asbm, X, L, f"{args.data_savepath}/synf-seed{args.seed}-{suffix}/Atest-Xtest")

# ppi
if args.ppi:
    train_graphs, val_graphs, test_graphs = load_ppi()
    train_data = graphs2data(train_graphs + val_graphs)
    test_data = graphs2data(test_graphs)
    args.p = 50
    n_edges = [x[-1].sum() for x in train_data]
    print(f"""
        Train ppi:
            - Number of edges (min-max-ave): {min(n_edges)} - {max(n_edges)} - {sum(n_edges)/len(n_edges)}
    """)
else:
    graphs = []
    n_train = int(args.n_graphs * 0.8)
    u = np.random.multivariate_normal(np.zeros((args.p)), np.eye(args.p) / args.p, 1)
    print("u", u)
    n_edges = []
    for i in range(args.n_graphs):
        if i % 100 == 0:
            print(f"{i+1}/{args.n_graphs}")
        Asbm, X, L = gen_graph(n=args.n, p=args.p, lam=args.lam, mu=args.mu, u=u)
        G = nx.from_numpy_matrix(Asbm)
        for i, node in enumerate(G.nodes()):
            G.nodes()[node]['features'] = X[i]
            G.nodes()[node]['label'] = L[i]
        n_edges.append(Asbm.sum())
        graphs.append(G)
    train_graphs = graphs[:n_train]
    test_graphs = graphs[n_train:]
    print(f"Number of edges: min {np.min(n_edges)} - max {np.max(n_edges)} - ave {np.mean(n_edges):.2f}")
    print("Auto adjust k")
    ave_degree = np.mean(n_edges) / args.n
    args.k = int(ave_degree)
    print(f"Select k = {args.k}")

    train_data = graphs2data(train_graphs)
    test_data = graphs2data(test_graphs)


class GRANMixtureBernoulli(nn.Module):

    def __init__(self):
        super(GRANMixtureBernoulli, self).__init__()
        self.hidden_dim = args.p  # ppi features size
        self.num_mix_component = args.num_mix_component
        self.edge_weight = args.edge_weight

        self.output_alpha = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.num_mix_component),
            nn.LogSoftmax(dim=1)
            # nn.Softmax()
        )

        self.output_theta = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.num_mix_component))
        pos_weight = torch.ones([1]) * self.edge_weight
        self.adj_loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

#     def forward(self, graphs):
#         adjs = [x[-1] for x in graphs]
#         features = [x[1] for x in graphs]
#         true_edges = [np.random.permutation(np.argwhere(adj == 1)) for adj in adjs]
#         false_edges = [np.random.permutation(np.argwhere(adj == 0)) for adj in adjs]
#         all_edges = [np.concatenate([x, y], axis=0) for x, y in zip(true_edges, false_edges)]
#         label = [np.array([1]*len(x)+[0]*len(y)) for x, y in zip(true_edges, false_edges)]

#         bs = 10000
#         sample_inds = [np.random.permutation(len(edges))[:bs] for edges in all_edges]
#         all_edges = [edges[ind] for edges, ind in zip(all_edges, sample_inds)]
#         label = torch.cat([torch.FloatTensor(x[ind]).cuda() for x, ind in zip(label, sample_inds)], dim=0)
# #
#         features = [torch.FloatTensor(x).cuda() for x in features]
#         diff = torch.cat([x[edges[:,0]] - x[edges[:,1]] for x, edges in zip(features, all_edges)], dim=0)

#         log_theta = self.output_theta(diff)
#         log_alpha = self.output_alpha(diff)  # B X (tt+K)K
#         log_theta = log_theta.view(-1, self.num_mix_component)  # B X CN(N-1)/2 X K
#         log_alpha = log_alpha.view(-1, self.num_mix_component)  # B X CN(N-1)/2 X K
#         adj_loss = mixture_bernoulli_loss(label, log_theta, log_alpha, self.adj_loss_func)
#         return adj_loss
    def forward(self, graphs):
        bs = args.nested_batch_size
        adjs = [x[-1] for x in graphs]
        features = [torch.FloatTensor(x[1]).cuda() for x in graphs]
        if bs > 0:
            all_lhs = [np.random.choice(len(x), size=bs, replace=True) for x in adjs]
            all_rhs = [np.random.choice(len(x), size=bs, replace=True) for x in adjs]
        else:
            all_lhs = [np.random.permutation(
                np.array(list(range(len(x))) * len(x))) for x in adjs]
            all_rhs = [np.random.permutation(
                np.array(list(range(len(x))) * len(x))) for x in adjs]
        label = torch.cat([torch.FloatTensor(adj[lhs,  rhs]).cuda()
                           for adj, lhs, rhs in zip(adjs, all_lhs, all_rhs)], dim=0)
        diff = torch.cat([x[lhs] - x[rhs]
                          for x, lhs, rhs in zip(features, all_lhs, all_rhs)], dim=0)
        log_theta = self.output_theta(diff)
        log_alpha = self.output_alpha(diff)  # B X (tt+K)K
        log_theta = log_theta.view(-1, self.num_mix_component)  # B X CN(N-1)/2 X K
        log_alpha = log_alpha.view(-1, self.num_mix_component)  # B X CN(N-1)/2 X K
        adj_loss = mixture_bernoulli_loss(label, log_theta, log_alpha, self.adj_loss_func)
        return adj_loss

    def predict_adj(self, features):
        temp = np.ones((len(features), len(features)))
        inds = np.triu_indices_from(temp, k=1)
        diff = torch.FloatTensor(
            features[inds[0]] - features[inds[1]]).unsqueeze(0).cuda()
        log_theta = self.output_theta(diff)
        log_alpha = self.output_alpha(diff)  # B X (tt+K)K

        B = 1
        log_theta = log_theta.view(B, -1, self.num_mix_component)  # B X K X (ii+K) X L
        log_alpha = log_alpha.view(B, -1, self.num_mix_component)  # B X K X (ii+K)
        prob_alpha = log_alpha.mean(dim=1).exp()
        alpha = torch.multinomial(prob_alpha, 1).squeeze(dim=1).long()

        A = torch.zeros((len(features), len(features)))
        prob = torch.sigmoid(log_theta[0, :, alpha[0]])
        A[inds[0], inds[1]] = torch.bernoulli(prob).cpu()

        # make it symmetric
        A = A + A.transpose(0, 1)
        return A


def mixture_bernoulli_loss(label, log_theta, log_alpha, adj_loss_func):
    """
    Compute likelihood for mixture of Bernoulli model

    Args:
      label: E X 1, see comments above
      log_theta: E X D, see comments above
      log_alpha: E X D, see comments above
      adj_loss_func: BCE loss
      subgraph_idx: E X 1, see comments above

    Returns:
      loss: negative log likelihood
    """
    K = log_theta.shape[1]
    reduce_adj_loss = torch.stack(
        [adj_loss_func(log_theta[:, kk], label) for kk in range(K)], dim=1)
    reduce_log_alpha = log_alpha
    log_prob = -reduce_adj_loss + reduce_log_alpha
    log_prob = torch.logsumexp(log_prob, dim=1)
    loss = -log_prob.sum() / float(log_theta.shape[0])
    return loss

model = GRANMixtureBernoulli().cuda()
optim = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=args.weight_decay)


def train_a_epoch():
    inds = np.random.permutation(len(train_data))
    data = [train_data[x] for x in inds]
    model.train()

    n_iter = len(data) // args.batch_size + 1
    loss_all = 0
    for iter in range(n_iter):
        optim.zero_grad()
        batch_data = data[iter * args.batch_size:(iter + 1) * args.batch_size]
        if len(batch_data) == 0:
            break
        loss = model(batch_data)
        loss.backward()
        optim.step()
        loss_all += loss
    return loss_all / n_iter


def evaluate(data):
    model.eval()

    n_iter = len(data) // args.batch_size + 1
    loss_all = 0
    for iter in range(n_iter):
        batch_data = data[iter * args.batch_size:(iter + 1) * args.batch_size]
        if len(batch_data) == 0:
            break
        loss = model(batch_data)
        loss_all += loss
    return loss_all / n_iter

if not os.path.isdir(f"{args.data_savepath}/imgs/"):
    os.makedirs(f"{args.data_savepath}/imgs/")

print("Start training")
for epoch in tqdm(range(1, args.epochs + 1)):
    loss = train_a_epoch()
    if epoch % args.evaluation_epochs == 0 or epoch == args.epochs:
        val_loss = evaluate(test_data)
        log_str = f"Epoch {epoch} - loss {loss:.3f} - val loss {val_loss:.3f}"
        print(log_str)
        # gen 1 train graph, 1 test graph, compare with 1 original train, 1 original test
        model.eval()
        with torch.no_grad():
            ind_train = np.random.randint(len(train_data))
            ind_test = np.random.randint(len(test_data))
            original_graphs = [train_data[ind_train][0], test_data[ind_test][0]]
            gen_adjs = [
                model.predict_adj(train_data[ind_train][1]).detach().cpu().numpy(),
                model.predict_adj(test_data[ind_test][1]).detach().cpu().numpy()
            ]
            gen_graphs = [nx.from_numpy_matrix(x) for x in gen_adjs]
            print("Gen graphs - number of edges: ", [x.sum() for x in gen_adjs])
            if args.export_imgs:
                plt.figure(dpi=150)
                plt.subplot(221)
                nx.draw(original_graphs[0], node_size=20)
                plt.subplot(222)
                nx.draw(original_graphs[1], node_size=20)
                plt.subplot(223)
                nx.draw(gen_graphs[0], node_color='g', node_size=20)
                plt.subplot(224)
                nx.draw(gen_graphs[1], node_color='g', node_size=20)
                plt.savefig(f"{args.data_savepath}/imgs/epoch-{epoch}.png")


print("Train done. Gen graphs from features")
gen_train_graphs = []
gen_test_graphs = []
model.eval()
with torch.no_grad():
    for graph, features, adj in train_data:
        gen_adj = model.predict_adj(features).detach().cpu().numpy()
        labels = np.array([graph.nodes()[node]['label'] for node in graph.nodes()])
        gen_train_graphs.append((gen_adj, features, labels, adj))
    for graph, features, adj in test_data:
        gen_adj = model.predict_adj(features).detach().cpu().numpy()
        labels = np.array([graph.nodes()[node]['label'] for node in graph.nodes()])
        gen_test_graphs.append((gen_adj, features, labels, adj))

# save_all_validation_graphs(args, gen_train_graphs, gen_test_graphs, epoch=args.epochs)

inds_train = np.random.permutation(len(gen_train_graphs))[:20]
inds_test = np.random.permutation(len(gen_test_graphs))[:20]
n_edges = [x[-1].sum() for x in gen_train_graphs]
k = int(np.mean(n_edges)) / gen_train_graphs[0][1].shape[0]
for i, (ind_train, ind_test) in enumerate(zip(inds_train, inds_test)):
    save_single_validation_graphs(args, gen_train_graphs[ind_train],
                                  gen_test_graphs[ind_test], suffix=i, k=k)

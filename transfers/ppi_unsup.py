import torch
import torch.nn.functional as F
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader
from sklearn.metrics import f1_score
from torch_geometric.nn import GCNConv, ChebConv
import os.path as osp
import numpy as np
from embed_algs import deepwalk

import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops

from torch_geometric.nn.inits import uniform


class SAGEConv(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{\hat{x}}_i &= \mathbf{\Theta} \cdot
        \mathrm{mean}_{j \in \mathcal{N(i) \cup \{ i \}}}(\mathbf{x}_j)

        \mathbf{x}^{\prime}_i &= \frac{\mathbf{\hat{x}}_i}
        {\| \mathbf{\hat{x}}_i \|_2}.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`False`, output features
            will not be :math:`\ell_2`-normalized. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 normalize=True,
                 bias=True,
                 **kwargs):
        super(SAGEConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)
        uniform(self.in_channels, self.bias)

    def forward(self, x, edge_index, size=None):
        """"""
        if size is None:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        x = x.unsqueeze(-1) if x.dim() == 1 else x
        x = torch.matmul(x, self.weight)

        return self.propagate(edge_index, size=size, x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        if self.normalize:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'PPI')
train_dataset = PPI(path, split='train')
# val_dataset = PPI(path, split='val')
# test_dataset = PPI(path, split='test')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

class Net(torch.nn.Module):
    def __init__(self, feature_dim):
        super(Net, self).__init__()
        self.conv1 = SAGEConv(feature_dim, 128, normalize=True)
        self.conv2 = SAGEConv(128, 128, normalize=True)

    def forward(self, X, A):
        edge_index = A.nonzero().t()
        x = X
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.normalize(x, p=2, dim=1)
        return x, None, None, None

        # neg = fixed_unigram_candidate_sampler(
        #     num_sampled=5,
        #     unique=False,
        #     range_max=len(A),
        #     distortion=0.75,
        #     unigrams=A.sum(axis=0).cpu().numpy()
        # )
        # neg_output = x[neg]
        # indices = np.random.permutation(edge_index.shape[1])
        # src, trg = edge_index[:, indices[:256]]
        # src_output = x[src]
        # trg_output = x[trg]
        # return x, src_output, trg_output, neg_output

def fixed_unigram_candidate_sampler(num_sampled, unique, range_max, distortion, unigrams):
    weights = unigrams**distortion
    prob = weights/weights.sum()
    sampled = np.random.choice(range_max, num_sampled, p=prob, replace=~unique)
    return sampled

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(train_dataset.num_features).to(device)
# loss_op = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
from prediction import BipartiteEdgePredLayer
loss_fn = BipartiteEdgePredLayer(is_normalized_input=False, device=device)
def loss_op(U, A):
    scores = torch.sigmoid(U.mm(U.t()))
    linkpred_loss = torch.norm(scores - A, p=2) / len(U)
    return linkpred_loss

def train(data):
    """
    data: list of (A,F)
    """
    model.train()

    total_loss = 0
    for A, X, _ in data:
        A = A.to(device)
        X = X.to(device)
        optimizer.zero_grad()
        U, srcE, trgE, negE = model(X, A)
        # loss = loss_fn.loss(srcE, trgE, negE)
        loss = loss_op(U, A)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len(data)


def test(data):
    model.eval()
    f1s = []
    with torch.no_grad():
        for A, X, _ in data:
            A = A.to(device)
            X = X.to(device)
            U, srcE, trgE, negE = model(X, A)
            pA = torch.sigmoid(U.mm(U.t()))
            pA[pA >= 0.5] = 1
            pA[pA < 0.5] = 0
            f1micro = f1_score(A.cpu().numpy(), pA.cpu().numpy(), average='micro')
            f1s.append(f1micro)
    return f1s

data = [x for x in train_loader][:2]
data = [[x.edge_index, x.x, x.y] for x in data]
for i in range(len(data)):
    edge_index, X, _ = data[i]
    A = torch.zeros((len(X), len(X)))
    src, trg = edge_index
    A[src, trg] = 1
    data[i][0] = A


for epoch in range(1, 101):
    loss = train(data)
    val_f1 = test(data)
    print('Epoch: {:02d}, Loss: {:.4f}, Train: {}'.format(epoch, 
    loss, " ".join(["{:.3f}".format(x) for x in val_f1])))


# networkx 
# import networkx as nx 
# Gs = [nx.from_numpy_array(adj.numpy()) for adj, x in data]

# for G in Gs:
#     vectors = deepwalk(G, 128)
#     U = np.array([vectors[x] for x in G.nodes()])
#     U = np.linalg.norm(U)
#     pA = U.dot(U.T)
#     pA[pA >= 0.5] = 1
#     pA[pA < 0.5] = 0
#     f1micro = f1_score(nx.to_numpy_array(G), pA, average='micro')
#     print("Graph x: f1 ", f1micro)

scores = 1
for _, _, L in data:
    pL = np.abs(L.sum(axis=0)/len(L) - 0.5)
    scores *= pL
selected_label = np.argmin(scores)
print("Selected label: ", selected_label)

print("Save graphs")
from tqdm import tqdm
for i, (A, X, L) in tqdm(enumerate(data)):
    U = model(X.to(device), A.to(device))[0].detach().cpu().numpy()
    features = X.cpu().numpy()
    edgelist = np.argwhere(A.detach().cpu().numpy() > 0)
    labels = L.cpu().numpy()
    labels = labels[:, selected_label]

    import os
    outdir = f"data-learnf/ppi/{i}"
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    with open(outdir + "/edgelist.txt", "w+") as fp:
        for src, trg in edgelist:
            fp.write(f"{src} {trg} 1\n")
    with open(outdir + "/labels.txt", "w+") as fp:
        for i, label in enumerate(labels):
            fp.write(f"{i} {int(label)}\n")
    np.savez_compressed(outdir + "/features.npz", features=features)
    np.savez_compressed(outdir + "/u.npz", features=U)


"""
1. PPI onelabel - transfer 1 1 - original features -> transferable
    Train acc ~ 0.71
    Without transfer: val acc at epoch 0 ~ 0.53 
    With transfer: val acc at epoch 0 : 0.69-0.61, higher best val acc 
2. PPI onelabel - transfer 1 1 - unsupervised features -> transferable
    Train acc ~ 0.64
    Without transfer: val acc at epoch 0 ~ 0.52
    With transfer: val acc at epoch 0 ~ 0.61-0.62, higher best val acc
"""

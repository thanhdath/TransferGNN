import os.path as osp
from math import ceil

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
from torch_geometric.io import read_tu_data
import tensorboardX

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from torch_geometric.nn.inits import uniform
import time
import torch
import sys


EPS = 1e-15
def dense_diff_pool(x, adj, s, mask=None):
    r"""Differentiable pooling operator from the `"Hierarchical Graph
    Representation Learning with Differentiable Pooling"
    <https://arxiv.org/abs/1806.08804>`_ paper

    .. math::
        \mathbf{X}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{X}

        \mathbf{A}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})

    based on dense learned assignments :math:`\mathbf{S} \in \mathbb{R}^{B
    \times N \times C}`.
    Returns pooled node feature matrix, coarsened adjacency matrix and the
    auxiliary link prediction objective :math:`\| \mathbf{A} -
    \mathrm{softmax}(\mathbf{S}) \cdot {\mathrm{softmax}(\mathbf{S})}^{\top}
    \|_F`.

    Args:
        x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
            \times N \times F}` with batch-size :math:`B`, (maximum)
            number of nodes :math:`N` for each graph, and feature dimension
            :math:`F`.
        adj (Tensor): Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B
            \times N \times N}`.
        s (Tensor): Assignment tensor :math:`\mathbf{S} \in \mathbb{R}^{B
            \times N \times C}` with number of clusters :math:`C`. The softmax
            does not have to be applied beforehand, since it is executed
            within this method.
        mask (ByteTensor, optional): Mask matrix
            :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
            the valid nodes for each graph. (default: :obj:`None`)

    :rtype: (:class:`Tensor`, :class:`Tensor`, :class:`Tensor`,
        :class:`Tensor`)
    """

    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    batch_size, num_nodes, _ = x.size()

    s = torch.softmax(s, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    link_loss = adj - torch.matmul(s, s.transpose(1, 2))
    link_loss = torch.norm(link_loss, p=2)
    link_loss = link_loss / adj.numel()

    ent_loss = (-s * torch.log(s + EPS)).sum(dim=-1).mean()

    return out, out_adj, link_loss, ent_loss, s


class SAGESum(torch.nn.Module):
    r"""See :class:`torch_geometric.nn.conv.SAGEConv`.

    :rtype: :class:`Tensor`
    """

    def __init__(self, in_channels, out_channels, normalize=True, bias=True):
        super(SAGESum, self).__init__()

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

    def forward(self, x, adj, mask=None, add_loop=True):
        r"""
        Args:
            x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
                \times N \times F}`, with batch-size :math:`B`, (maximum)
                number of nodes :math:`N` for each graph, and feature
                dimension :math:`F`.
            adj (Tensor): Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B
                \times N \times N}`. The adjacency tensor is broadcastable in
                the batch dimension, resulting in a shared adjacency matrix for
                the complete batch.
            mask (ByteTensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
            add_loop (bool, optional): If set to :obj:`False`, the layer will
                not automatically add self-loops to the adjacency matrices.
                (default: :obj:`True`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1

        out = torch.matmul(adj, x)
        # out = out / adj.sum(dim=-1, keepdim=True).clamp(min=1)
        out = torch.matmul(out, self.weight)

        if self.bias is not None:
            out = out + self.bias

        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class MyFilter(object):
    def __call__(self, data):
        return data.num_nodes <= max_nodes


class GNN(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 normalize=False,
                 add_loop=False,
                 lin=True):
        super(GNN, self).__init__()

        self.add_loop = add_loop

        # self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        # self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        # self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        # self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        # self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        # self.bn3 = torch.nn.BatchNorm1d(out_channels)

        self.conv1 = SAGESum(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = SAGESum(hidden_channels, hidden_channels, normalize)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = SAGESum(hidden_channels, out_channels, normalize)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, 'bn{}'.format(i))(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()

        x0 = x
        x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask, self.add_loop)))
        x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask, self.add_loop)))
        x3 = self.bn(3, F.relu(self.conv3(x2, adj, mask, self.add_loop)))

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = F.relu(self.lin(x))

        return x


class Net(torch.nn.Module):
    def __init__(self, feature_size=3, n_classes=6):
        super(Net, self).__init__()

        num_nodes = ceil(0.25 * max_nodes)
        self.gnn1_pool = GNN(feature_size, 64, num_nodes, add_loop=True)
        self.gnn1_embed = GNN(feature_size, 64, 64, add_loop=True, lin=False)

        num_nodes = ceil(0.25 * max_nodes)
        self.gnn2_pool = GNN(3 * 64, 64, num_nodes)
        self.gnn2_embed = GNN(3 * 64, 64, 64, lin=False)

        self.gnn3_embed = GNN(3 * 64, 64, 64, lin=False)

        self.lin1 = torch.nn.Linear(3 * 64, 64)
        self.lin2 = torch.nn.Linear(64, n_classes)
      

    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        x, adj, l1, e1, s1 = dense_diff_pool(x, adj, s, mask)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2, s2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2, s1, s2

import torch
from torch_geometric.data import InMemoryDataset

class CustomDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 use_node_attr=False):
        self.name = root.split("/")[-1]
        super(CustomDataset, self).__init__(root, transform, pre_transform,
                                        pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        if self.data.x is not None and not use_node_attr:
            self.data.x = self.data.x[:, self.num_node_attributes:]

    @property
    def num_node_labels(self):
        if self.data.x is None:
            return 0

        for i in range(self.data.x.size(1)):
            if self.data.x[:, i:].sum().item() == self.data.x.size(0):
                return self.data.x.size(1) - i

        return 0

    @property
    def num_node_attributes(self):
        if self.data.x is None:
            return 0

        return self.data.x.size(1) - self.num_node_labels

    @property
    def raw_file_names(self):
        names = ['A', 'graph_indicator']
        return ['{}_{}.txt'.format(self.name, name) for name in names]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        return

    def process(self):
        self.data, self.slices = read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))

def train(epoch, writer=None):
    model.train()
    loss_all = 0
    loss_linkpred_all = 0
    loss_ent_all = 0

    for idata, data in enumerate(train_loader):
        data = data.to(device)
        if len(data.adj.shape) == 4:
            adj = data.adj.sum(axis=3)
        else:
            adj = data.adj
        optimizer.zero_grad()
        output, link_loss, ent_loss, s1, s2 = model(data.x, adj, data.mask)
        if idata == 1 and writer is not None:
            writer.add_image(f"adj/{idata}", adj[0].unsqueeze(0), 0)
            writer.add_image("assignment_matrix/{}-s1".format(idata), s1[0].unsqueeze(0), epoch)
            writer.add_image("assignment_matrix/{}-s2".format(idata), s2[0].unsqueeze(0), epoch)

        loss = F.nll_loss(output, data.y.view(-1)) + link_loss + ent_loss
        # loss = F.nll_loss(output, data.y.view(-1)) + link_loss
        loss.backward()
        loss_all += data.y.size(0) * loss.item()
        loss_linkpred_all += data.y.size(0) * link_loss.item()
        loss_ent_all += data.y.size(0) * ent_loss.item()
        optimizer.step()
    loss_all = loss_all / len(train_dataset)
    loss_linkpred_all /= len(train_dataset)
    loss_ent_all /= len(train_dataset)
    # summary
    if writer is not None:
        writer.add_scalar("loss/nll", loss_all, epoch)
        writer.add_scalar("loss/linkpred", loss_linkpred_all, epoch)
        writer.add_scalar("loss/ent", loss_ent_all, epoch)
    return loss_all


def test(loader, writer=None):
    model.eval()
    correct = 0

    for idata, data in enumerate(loader):
        data = data.to(device)
        if len(data.adj.shape) == 4:
            adj = data.adj.sum(axis=3)
        else:
            adj = data.adj
        pred, link_loss, ent_loss, s1, s2  = model(data.x, adj, data.mask)
        if writer is not None and idata in [0, 1, 2]:
            writer.add_image(f"adj/test{idata}", adj[0].unsqueeze(0), 0)
            writer.add_image("assignment_matrix/test{}-s1".format(idata), s1[0].unsqueeze(0), epoch)
            writer.add_image("assignment_matrix/test{}-s2".format(idata), s2[0].unsqueeze(0), epoch)

        pred = pred.max(dim=1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


max_nodes = 100

import sys 

dataset_path = sys.argv[1]
dataset_name = dataset_path.split("/")[-1]
writer = tensorboardX.SummaryWriter(log_dir=f"runs/{dataset_name}-{time.time()}")
# writer = None

dataset = CustomDataset(
    dataset_path,
    transform=T.ToDense(max_nodes),
    pre_filter=MyFilter())

dataset = dataset.shuffle()
n = (len(dataset) + 9) // 10
test_dataset = dataset[:n]
val_dataset = dataset[n:2 * n]
train_dataset = dataset[2 * n:]
test_loader = DenseDataLoader(test_dataset, batch_size=20)
val_loader = DenseDataLoader(val_dataset, batch_size=20)
train_loader = DenseDataLoader(train_dataset, batch_size=20)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(feature_size=dataset.data.x.shape[-1], n_classes=dataset.data.y.shape[-1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


best_val_acc = test_acc = 0
for epoch in range(1, 300):
    train_loss = train(epoch, writer)
    val_acc = test(val_loader)
    train_acc = test(train_loader)
    if val_acc > best_val_acc:
        test_acc = test(test_loader, writer=writer)
        best_val_acc = val_acc
    print('Epoch: {:03d}, Train Loss: {:.7f}, '
          'Train Acc: {:.4f}, Val Acc: {:.4f}, Test Acc: {:.4f}'.format(epoch, train_loss,
                                                    train_acc, val_acc, test_acc))
    writer.add_scalar('acc/train', train_acc, epoch)
    writer.add_scalar('acc/val', val_acc, epoch)
    writer.add_scalar('acc/test', test_acc, epoch)

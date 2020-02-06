import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv
from sklearn.metrics import f1_score
import argparse
from tqdm import tqdm
import multiprocessing
from openne.walker import BasicWalker, DegreeCorrelationWalker

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=500, type=int)
parser.add_argument("--walk_length", default=5, type=int)
parser.add_argument("--interval", default=100, type=int)
parser.add_argument("--num_walks", default=40, type=int)
args = parser.parse_args()

def build_neibs_dict(srcs, trgs):
    neibs_dict = {}
    for src, trg in zip(srcs, trgs):
        neibs_dict[src] = neibs_dict.get(src, []) + [trg]
        neibs_dict[trg] = neibs_dict.get(trg, []) + [src]
    for k, v in neibs_dict.items():
        neibs_dict[k] = list(set(neibs_dict[k]))
    return neibs_dict

path = "./data/PPI"
train_dataset = PPI(path, split='train')
val_dataset = PPI(path, split='val')
test_dataset = PPI(path, split='test')
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
train_loader = list(train_dataset)
train_indices = np.arange(len(train_loader))
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

train_neibs_dict = []
val_neibs_dict = []
test_neibs_dict = []
for data in train_loader:
    srcs, trgs = data.edge_index.cpu().numpy()
    train_neibs_dict.append(build_neibs_dict(srcs, trgs))
for data in val_loader:
    srcs, trgs = data.edge_index.cpu().numpy()
    val_neibs_dict.append(build_neibs_dict(srcs, trgs))
for data in test_loader:
    srcs, trgs = data.edge_index.cpu().numpy()
    test_neibs_dict.append(build_neibs_dict(srcs, trgs))



def generate_random_walks(neibs_dict, workers=multiprocessing.cpu_count()//3, number_walks=5, 
    walk_length=5):
    """
    generate random walks with centralized nodes = current node
    """
    assert walk_length%2 == 1, "Walk length must be an odd number!"
    walk = BasicWalker(neibs_dict, workers=workers)
    sentences = walk.simulate_walks(num_walks=args.num_walks, walk_length=walk_length, num_workers=workers)
    # walk = DegreeCorrelationWalker(neibs_dict, workers=workers)
    # sentences = walk.simulate_walks(average_walks=number_walks, min_walks=5,
    #     walk_length=walk_length, num_workers=workers)
    return sentences

def generate_new_edge_index(neibs_dict, number_walks):
    sentences = generate_random_walks(neibs_dict, number_walks=number_walks, walk_length=args.walk_length)
    edge_index = []
    for sentence in sentences:
        middle_node = sentence[args.walk_length//2]
        edges = [[sentence[i], middle_node] for i in range(len(sentence)) if i != args.walk_length//2]
        edge_index += edges
    edge_index = torch.LongTensor(np.array(edge_index, dtype=np.int32)).t()
    return edge_index

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GATConv(train_dataset.num_features, 256, heads=4)
        self.lin1 = torch.nn.Linear(train_dataset.num_features, 4 * 256)
        self.conv2 = GATConv(4 * 256, 256, heads=4)
        self.lin2 = torch.nn.Linear(4 * 256, 4 * 256)
        self.conv3 = GATConv(
            4 * 256, train_dataset.num_classes, heads=6, concat=False)
        self.lin3 = torch.nn.Linear(4 * 256, train_dataset.num_classes)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
        x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
        x = self.conv3(x, edge_index) + self.lin3(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu = torch.device("cpu")
model = Net().to(device)
loss_op = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


def train(edge_inds):
    model.train()

    total_loss = 0
    for data, edge_index in zip(train_loader, edge_inds):
        # num_graphs = data.num_graphs
        num_graphs = 1
        # data.batch = None
        data = data.to(device)
        edge_index = edge_index.to(device)
        optimizer.zero_grad()
        # try:
        loss = loss_op(model(data.x, edge_index), data.y)
        # except:
        #     import pdb; pdb.set_trace()
        total_loss += loss.item() * num_graphs
        loss.backward()
        optimizer.step()
        edge_index = edge_index.to(cpu)
        data = data.to(cpu)
    return total_loss / len(train_loader)


def test(loader, edge_inds):
    model.eval()

    ys, preds = [], []
    for data, edge_index in zip(loader, edge_inds):
        ys.append(data.y)
        with torch.no_grad():
            out = model(data.x.to(device), edge_index.to(device))
        preds.append((out > 0).float().cpu())
        edge_index = edge_index.to(cpu)
        data = data.to(cpu)

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0


for epoch in range(0, args.epochs):
    if epoch%args.interval == 0:
        train_edge_inds = []
        val_edge_inds = []
        test_edge_inds = []
        for i in tqdm(range(len(train_loader))):
            train_edge_inds.append(generate_new_edge_index(train_neibs_dict[i], args.num_walks))
        # for i in tqdm(range(len(val_loader))):
        #     val_edge_inds.append(generate_new_edge_index(val_neibs_dict[i], args.num_walks))
        # for i in tqdm(range(len(test_loader))):
        #     test_edge_inds.append(generate_new_edge_index(test_neibs_dict[i], args.num_walks))
    
    loss = train(train_edge_inds)
    np.random.shuffle(train_indices)
    train_loader = [train_loader[x] for x in train_indices]
    train_edge_inds = [train_edge_inds[x] for x in train_indices]
    train_neibs_dict = [train_neibs_dict[x] for x in train_indices]

    val_f1 = test(val_loader, [data.edge_index for data in val_loader])
    test_f1 = test(test_loader, [data.edge_index for data in test_loader])
    print('Epoch: {:02d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(
        epoch, loss, val_f1, test_f1))
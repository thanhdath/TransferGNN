import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import ChebConv, GATConv, GCNConv
from data import load_graph
import argparse
import os
from sklearn.metrics import f1_score
import numpy as np
#import matplotlib.pyplot as plt
import tensorboardX
from sage import SAGEConv
from openne.walker import BasicWalker, DegreeCorrelationWalker
import multiprocessing
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--data", default="Cora", type=str)
parser.add_argument("--epochs", default=1000, type=int)
parser.add_argument("--hidden", default=16, type=int)
parser.add_argument("--walk_length", default=5, type=int)
parser.add_argument("--interval", default=2, type=int)
parser.add_argument("--num_walks", default=5, type=int)
parser.add_argument("--dropout", default=.4, type=float)
args = parser.parse_args()


# GAT
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GATConv(dataset.num_features, args.hidden, heads=8, 
            dropout=args.dropout, concat=True)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(args.hidden * 8, num_classes, heads=2, 
            concat=True, dropout=args.dropout)
        # self.conv3 = GATConv(16 * 4, num_classes, heads=1, concat=True, dropout=0.2)

    def forward(self, edge_index):
        x = data.x
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv1(x, edge_index) 
        x = F.elu(x)
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        # x = F.elu(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        # x = self.conv3(x, edge_index)
        # return x
        return F.log_softmax(x, dim=1)

def generate_random_walks(workers=multiprocessing.cpu_count()//3, number_walks=5, 
    walk_length=5):
    """
    generate random walks with centralized nodes = current node
    """
    assert walk_length%2 == 1, "Walk length must be an odd number!"
    # walk = BasicWalker(neibs_dict, workers=workers)
    # sentences = walk.simulate_walks(num_walks=args.num_walks, walk_length=walk_length, num_workers=workers)
    walk = DegreeCorrelationWalker(neibs_dict, workers=workers)
    sentences = walk.simulate_walks(average_walks=number_walks, min_walks=5,
        walk_length=walk_length, num_workers=workers)
    return sentences

def generate_new_edge_index(number_walks):
    sentences = generate_random_walks(number_walks=number_walks, walk_length=args.walk_length)
    edge_index = []
    for sentence in sentences:
        middle_node = sentence[args.walk_length//2]
        edges = [[sentence[i], middle_node] for i in range(len(sentence)) if i != args.walk_length//2]
        edge_index += edges
    edge_index = torch.LongTensor(np.array(edge_index, dtype=np.int32)).t()
    return edge_index.to(device)

def train(edge_index):
    model.train()
    optimizer.zero_grad()

    outputs = model(edge_index)[data.train_mask]
    F.nll_loss(outputs, data.y[data.train_mask]).backward()
    optimizer.step()
    return edge_index

def f1(output, labels, multiclass=False):
    if len(output) == 0: return 0, 0
    if not multiclass:
        preds = output.max(1)[1]
        preds = preds.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        micro = f1_score(labels, preds, average='micro')
        macro = f1_score(labels, preds, average='macro')
        return micro, macro
    else:
        probs = torch.sigmoid(output)
        probs[probs>0.5] = 1
        probs[probs<=0.5] = 0
        probs = probs.cpu().detach().numpy().astype(np.int32)
        labels = labels.cpu().detach().numpy().astype(np.int32)
        micro = f1_score(labels, probs, average='micro')
        macro = f1_score(labels, probs, average='macro')
        return micro, macro

def test(edge_index):
    model.eval()
    logits, accs = model(edge_index), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.int().sum().item() * 100
        accs.append(acc)
    return accs

def getfilename(path):
    if path is None:
        return None
    return path.split("/")[-1].split(".")[0]

def edge_index_k_hop(edge_index, k=2):
    adj = torch.FloatTensor(np.zeros((num_nodes, num_nodes), dtype=np.int32)).to(device)
    srcs, trgs = edge_index
    adj[srcs, trgs] = 1
    adjs = [adj]
    for k in range(1, k):
        temp_adj = adjs[-1].mm(adjs[0])
        adjs.append(temp_adj)
    for temp_adj in adjs[1:]:
        adjs[0] += temp_adj
    edge_index = np.argwhere(adjs[0].cpu().numpy() > 0).T
    edge_index = torch.LongTensor(edge_index).to(device)
    return edge_index

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = Planetoid(f'./data/{args.data}', args.data, T.NormalizeFeatures())
data = dataset[0].to(device)
num_nodes, num_features = data.x.shape
num_classes = int(data.y.max() + 1)
# build neibs_dict 
srcs, trgs = dataset[0].edge_index.cpu().numpy()
print("Edges original:", dataset[0].edge_index.shape)
neibs_dict = {}
for src, trg in zip(srcs, trgs):
    neibs_dict[src] = neibs_dict.get(src, []) + [trg]
    neibs_dict[trg] = neibs_dict.get(trg, []) + [src]
for k, v in neibs_dict.items():
    neibs_dict[k] = list(set(neibs_dict[k]))

# edges_k_hop = edge_index_k_hop(dataset[0].edge_index.cpu().numpy(), k=2)
edges_k_hop = generate_new_edge_index(100)
print("Edges k-hop:", edges_k_hop.shape)


print("Num features:", num_features)
print("Num classes:", num_classes)

model = Net().to(device)
lr = 0.01

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
best_val_acc = 0
best_model = None
epochs = args.epochs
if not os.path.isdir("model"):
    os.makedirs("model")

model_name = f"model/cora-gat.pkl"
writer = tensorboardX.SummaryWriter(logdir="runs/"+model_name.split("/")[-1].split(".")[0])

for epoch in range(0, epochs):
    # if args.transfer is not None and epoch < epochs//3:
    #     model.conv1.requires_grad = False
    # else:
    #     model.conv1.requires_grad = True
    if epoch%args.interval == 0:
        edge_index = generate_new_edge_index(args.num_walks)
    train(edge_index)
    if epoch == epochs - 1:
        model.load_state_dict(torch.load(model_name))
        print("Load best model")

    
        # torch.save(model.state_dict(), model_name)
        # test_acc = tmp_test_acc
    if epoch%5 == 0 or epoch == epochs-1:
        train_acc, val_acc, test_acc = test(data.edge_index)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_name)
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, val_acc, test_acc))
        writer.add_scalar("train_acc", train_acc, epoch)
        writer.add_scalar("val_acc", val_acc, epoch)
        writer.add_scalar("test_acc", test_acc, epoch)
# torch.save(model.state_dict(), model_name)
print("Best val acc: {:.3f}".format(best_val_acc))
print("Model has been saved to", model_name)

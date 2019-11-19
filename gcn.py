import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from data import load_graph
import argparse
import os
from sklearn.metrics import f1_score
import numpy as np
#import matplotlib.pyplot as plt
import tensorboardX

parser = argparse.ArgumentParser()
parser.add_argument("--transfer", default=None)
parser.add_argument("--adj", default="twain_tramp/wan_twain_tramp_1.txt")
args = parser.parse_args()

dataset = load_graph(args.adj, "features.npz", "labels.txt")
# num_classes = int(dataset.y.max()) + 1
num_classes = dataset.y.shape[1]

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 16, cached=True)
        self.conv2 = GCNConv(16, num_classes, cached=True)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.relu(self.conv1(x, edge_index,
            edge_weight
        ))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x
        # return F.log_softmax(x, dim=1)

def train():
    model.train()
    optimizer.zero_grad()
    outputs = model()[data.train_mask]
    criterion(outputs, data.y[data.train_mask]).backward()
    # F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()

def f1(output, labels, multiclass=False):
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

def test():
    model.eval()
    logits = model()
    micros = []
    macros = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        micro, macro = f1(logits[mask], data.y[mask], multiclass=True)
        micros.append(micro)
        macros.append(macro)
    return micros, macros
        # pred = logits[mask].max(1)[1]
        # acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        # accs.append(acc)
    # return accs

def getfilename(path):
    if path is None:
        return None
    return path.split("/")[-1].split(".")[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = dataset.to(device)
model = Net().to(device)
if args.transfer is not None:
    print("Load pretrained model", args.transfer)
    model.load_state_dict(torch.load(args.transfer))

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

best_val_acc = 0
best_model = None
epochs = 300
if args.transfer is not None:
    model_name = f"model/{getfilename(args.adj)}-transfer-from-{getfilename(args.transfer)}.pkl"
else:
    model_name = f"model/{getfilename(args.adj)}.pkl"
writer = tensorboardX.SummaryWriter(logdir="runs/"+model_name.split("/")[-1].split(".")[0])
for epoch in range(1, epochs):
    train()
    if epoch == epochs - 1:
        model.load_state_dict(torch.load(model_name))
        print("Load best model")

    micros, macros = test()
    train_acc, val_acc, tmp_test_acc = micros
    train_macro, val_macro, test_macro = macros

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), model_name)
        # test_acc = tmp_test_acc
    if epoch%10 == 0 or epoch == epochs-1:
        log = 'Epoch: {:03d}, micro-macro Train: {:.4f}-{:.4f}, Val: {:.4f}-{:.4f}, Test: {:.4f}-{:.4f}'
        print(log.format(epoch, train_acc, train_macro, val_acc, val_macro, tmp_test_acc, test_macro))
        writer.add_scalar("train_acc", train_acc, epoch)
        writer.add_scalar("val_acc", val_acc, epoch)
        writer.add_scalar("test_acc", tmp_test_acc, epoch)
if not os.path.isdir("model"):
    os.makedirs("model")
# torch.save(model.state_dict(), model_name)
print("Model has been saved to", model_name)

import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import ChebConv, GATConv, GCNConv, SAGEConv
from data import load_graph
import argparse
import os
from sklearn.metrics import f1_score
import numpy as np
#import matplotlib.pyplot as plt
import tensorboardX
from sklearn.metrics import classification_report
# from sage import SAGEConv

parser = argparse.ArgumentParser()
parser.add_argument("--transfer", default=None)
parser.add_argument("--adj", default="twain_tramp/wan_twain_tramp_1.txt")
parser.add_argument("--labels", default="labels.txt")
parser.add_argument("--features", default="features.npz")
parser.add_argument("--multiclass", default=None, type=int)
parser.add_argument("--epochs", default=1000, type=int)
parser.add_argument("--hidden", default=32, type=int)
parser.add_argument("--feature-only", action='store_true')
parser.add_argument("--seed", default=100, type=int)
args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.multiclass == 0:
    args.multiclass = False
elif args.multiclass == 1:
    args.multiclass = True

dataset, multiclass = load_graph(args.adj, args.features, args.labels, args.multiclass)
# num_classes = int(dataset.y.max()) + 1
if multiclass:
    num_classes = dataset.y.shape[1]
else:
    num_classes = int(dataset.y.max() + 1)
print("Num features:", dataset.num_features)
print("Num classes:", num_classes)

# GCN
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = GCNConv(dataset.num_features, 32, cached=True)
        # self.conv2 = GCNConv(32, num_classes, cached=True)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)
        self.conv1 = SAGEConv(dataset.num_features, args.hidden, normalize=False)
        self.conv2 = SAGEConv(args.hidden, num_classes, normalize=False)

    def forward(self):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.conv1(x, edge_index)
        # x = x / data.x.shape[0]
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        # x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.conv2(x, edge_index)
        # x = x / data.x.shape[0]
        if multiclass:
            return x
        return F.log_softmax(x, dim=1)

class SoftmaxRegression(torch.nn.Module):
    def __init__(self):
        super(SoftmaxRegression, self).__init__()
        # self.model = torch.nn.Sequential(
        #     torch.nn.Linear(dataset.num_features, dataset.num_features*2),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(dataset.num_features*2, num_classes)
        # )
        self.model = torch.nn.Linear(dataset.num_features, num_classes)
    def forward(self):
        x = data.x
        x = self.model(x)
        if multiclass:
            return x
        return F.log_softmax(x, dim=1)

def train():
    model.train()
    optimizer.zero_grad()
    outputs = model()[data.train_mask]
    if multiclass:
        criterion(outputs, data.y[data.train_mask]).backward()
    else:
        F.nll_loss(outputs, data.y[data.train_mask]).backward()
    optimizer.step()

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

def test():
    model.eval()
    logits = model()
    micros = []
    macros = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        micro, macro = f1(logits[mask], data.y[mask], multiclass=multiclass)
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
if args.feature_only:
    print("Using softmax regression")
    model = SoftmaxRegression().to(device)
else:
    model = Net().to(device)
lr = 0.001
if args.transfer is not None:
    print("Load pretrained model", args.transfer)
    pretrained_state_dict = torch.load(args.transfer)
    differ_shape_params = []
    model_state_dict = model.state_dict()
    for k in pretrained_state_dict.keys():
        if pretrained_state_dict[k].shape != model_state_dict[k].shape:
            differ_shape_params.append(k)
    pretrained_state_dict.update({k: v for k,v in model.state_dict().items() if k in differ_shape_params})
    model.load_state_dict(pretrained_state_dict)
if multiclass:
    criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

best_val_acc = 0
best_model = None
epochs = args.epochs
if not os.path.isdir("model"):
    os.makedirs("model")
if args.transfer is not None:
    model_name = f"model/{getfilename(args.adj)}-transfer-from-{getfilename(args.transfer)}.pkl"
else:
    model_name = f"model/{getfilename(args.adj)}.pkl"
writer = tensorboardX.SummaryWriter(logdir="runs/"+model_name.split("/")[-1].split(".")[0])

micros, macros = test()
train_acc, val_acc, tmp_test_acc = micros
train_macro, val_macro, test_macro = macros
log = 'Epoch: 0, micro-macro Train: {:.4f}-{:.4f}, Val: {:.4f}-{:.4f}, Test: {:.4f}-{:.4f}'
torch.save(model.state_dict(), model_name)
print(log.format(train_acc, train_macro, val_acc, val_macro, tmp_test_acc, test_macro))

best_val_acc = val_acc
for epoch in range(1, epochs):
    # if args.transfer is not None and epoch < epochs//3:
    #     model.conv1.requires_grad = False
    # else:
    #     model.conv1.requires_grad = True
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
    if epoch%20 == 0 or epoch == epochs-1:
        log = 'Epoch: {:03d}, micro-macro Train: {:.4f}-{:.4f}, Val: {:.4f}-{:.4f}, Test: {:.4f}-{:.4f}'
        print(log.format(epoch, train_acc, train_macro, val_acc, val_macro, tmp_test_acc, test_macro))
        writer.add_scalar("train_acc", train_acc, epoch)
        writer.add_scalar("val_acc", val_acc, epoch)
        writer.add_scalar("test_acc", tmp_test_acc, epoch)
# torch.save(model.state_dict(), model_name)
print("Best val acc: {:.3f}".format(best_val_acc))
print("Model has been saved to", model_name)

model.eval()
logits = model()
with torch.no_grad():
    mask = data['val_mask']
    preds = logits[mask]
    preds = torch.sigmoid(preds).cpu().numpy()
    preds[preds >= 0.5] = 1
    preds[preds < 0.5] = 0
    print(classification_report(data.y[mask].cpu().numpy(), preds))

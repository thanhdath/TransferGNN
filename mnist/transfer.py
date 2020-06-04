from itertools import product

import argparse
from train_eval import eval_acc
import torch

from gin import GIN0, GIN0WithJK, GIN, GINWithJK
from  torch_geometric.datasets.mnist_superpixels import MNISTSuperpixels
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
import torch_geometric.transforms as T

def get_dataset(name, sparse=True, cleaned=False):
    dataset = TUDataset("data/", name, cleaned=cleaned, use_node_attr=True)
    dataset.data.edge_attr = None

    if not sparse:
        num_nodes = max_num_nodes = 0
        for data in dataset:
            num_nodes += data.num_nodes
            max_num_nodes = max(data.num_nodes, max_num_nodes)

        # Filter out a few really large graphs in order to apply DiffPool.
        if name == 'REDDIT-BINARY':
            num_nodes = min(int(num_nodes / len(dataset) * 1.5), max_num_nodes)
        else:
            num_nodes = min(int(num_nodes / len(dataset) * 5), max_num_nodes)

        indices = []
        for i, data in enumerate(dataset):
            if data.num_nodes <= num_nodes:
                indices.append(i)
        dataset = dataset[torch.tensor(indices)]

        if dataset.transform is None:
            dataset.transform = T.ToDense(num_nodes)
        else:
            dataset.transform = T.Compose(
                [dataset.transform, T.ToDense(num_nodes)])

    return dataset
    
def load_mnist_superpixels(train=True, sparse=True):
    dataset = MNISTSuperpixels("data", train=train)
    if not sparse:
        num_nodes = max_num_nodes = 0
        for data in dataset:
            num_nodes += data.num_nodes
            max_num_nodes = max(data.num_nodes, max_num_nodes)

        # Filter out a few really large graphs in order to apply DiffPool.
        num_nodes = min(int(num_nodes / len(dataset) * 5), max_num_nodes)

        indices = []
        for i, data in enumerate(dataset):
            if data.num_nodes <= num_nodes:
                indices.append(i)
        dataset = dataset[torch.tensor(indices)]

        if dataset.transform is None:
            dataset.transform = T.ToDense(num_nodes)
        else:
            dataset.transform = T.Compose(
                [dataset.transform, T.ToDense(num_nodes)])
    return dataset

parser = argparse.ArgumentParser()
parser.add_argument('--from_data', default="grid") # grid or super
parser.add_argument('--to_data', default="super") # grid or super
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--seed', type=int, default=100)
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    val_loss, test_acc = info['val_loss'], info['val_acc']
    print('{:02d}/{:03d}: Train Loss: {:.4f}, Val Loss: {:.4f}, Val Accuracy: {:.3f}'.format(
        fold, epoch, info['train_loss'], val_loss, test_acc))

results = []
Net = GIN
print(f"Net: {Net}")
best_result = (float('inf'), 0, 0)  # (loss, acc, std)
num_layers = 1
hidden = 32
if args.to_data == "grid":
    test_dataset = get_dataset("MNIST_Grid_test", sparse=Net != DiffPool)
elif args.to_data == "super":
    test_dataset = load_mnist_superpixels(train=False, sparse=Net != DiffPool)

if args.from_data == "grid":
    val_dataset = get_dataset("MNIST_Grid_test", sparse=Net != DiffPool)
elif args.from_data == "super":
    val_dataset = load_mnist_superpixels(train=False, sparse=Net != DiffPool)

model = torch.load(f'model/{Net.__name__}-mnist-{args.from_data}-seed{args.seed}.pth')
if 'adj' in val_dataset[0]:
    val_loader = DenseLoader(val_dataset, args.batch_size, shuffle=False)
    test_loader = DenseLoader(test_dataset, args.batch_size, shuffle=False)
else:
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)
acc = eval_acc(model, val_loader)
print(f"Acc on Same graphs: {acc:.3f}")

acc = eval_acc(model, test_loader)
print(f"Acc on transfers: {acc:.3f}")


"""
This code is inherit from Pytorch Geometric
"""
from itertools import product

import argparse
from train_eval import custom_train_test_set
import torch

from gin import GIN0, GIN0WithJK, GIN, GINWithJK
from  torch_geometric.datasets.mnist_superpixels import MNISTSuperpixels
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
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
parser.add_argument('--data', default="grid") # grid or super
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--seed', type=int, default=100)
args = parser.parse_args()

# np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    val_loss, test_acc = info['val_loss'], info['val_acc']
    print('{:02d}/{:03d}: Train Loss: {:.4f}, Val Loss: {:.4f}, Val Accuracy: {:.3f}'.format(
        fold, epoch, info['train_loss'], val_loss, test_acc))

results = []
Net = GIN
num_layers = 1
hidden = 32

print(f"Net: {Net}")
best_result = (float('inf'), 0, 0)  # (loss, acc, std)
if args.data == "grid":
    train_dataset = get_dataset("MNIST_Grid_train", sparse=Net != DiffPool)
    test_dataset = get_dataset("MNIST_Grid_test", sparse=Net != DiffPool)
elif args.data == "super":
    train_dataset = load_mnist_superpixels(train=True, sparse=Net != DiffPool)
    test_dataset = load_mnist_superpixels(train=False, sparse=Net != DiffPool)

model = Net(train_dataset, num_layers, hidden)
loss, acc, std = custom_train_test_set(
    train_dataset,
    test_dataset,
    model,
    folds=1,
    epochs=args.epochs,
    batch_size=args.batch_size,
    lr=args.lr,
    lr_decay_factor=args.lr_decay_factor,
    lr_decay_step_size=args.lr_decay_step_size,
    weight_decay=0,
    logger=logger,
)
if loss < best_result[0]:
    best_result = (loss, acc, std)
torch.save(model, f"model/{Net.__name__}-mnist-{args.data}-seed{args.seed}.pth")

desc = '{:.3f} ± {:.3f}'.format(best_result[1], best_result[2])
print('Best result - {}'.format(desc))

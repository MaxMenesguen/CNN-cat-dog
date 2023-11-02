from __future__ import annotations
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from tqdm import tqdm

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


def prepare(train: bool, batch_size: int = 10_000, path: str = '/tmp/datasets') -> None:
    prefix = 'train' if train else 'test'
    path_xs = os.path.join(path, f'mnist_{prefix}_xs.pt')
    path_ys = os.path.join(path, f'mnist_{prefix}_ys.pt')
    if os.path.exists(path_xs) and os.path.exists(path_ys):
        return

    transform = T.Compose([T.ToTensor(), T.Normalize((0.1307, ), (0.3081, ))])
    set = MNIST(path, train=train, download=True, transform=transform)
    loader = DataLoader(set, batch_size=batch_size, shuffle=train)
    n = len(set)

    xs = torch.empty((n, *set[0][0].shape), dtype=torch.float32)
    ys = torch.empty((n, ), dtype=torch.long)
    for i, (x, y) in enumerate(tqdm(loader, desc=f'Preparing {prefix.capitalize()} Set')):
        xs[i * batch_size:min((i + 1) * batch_size, n)] = x
        ys[i * batch_size:min((i + 1) * batch_size, n)] = y

    torch.save(xs, path_xs)
    torch.save(ys, path_ys)


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin1 = nn.Linear(1 * 28 * 28, 512)
        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)  # .view(x.shape[0], -1)
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))
        return x


if __name__ == "__main__":
    from torch.optim import Adam


    device = torch.device("cpu")
    batch_size = 32
    epochs = 5  #  5 Do not Change

    prepare(train=True)
    prepare(train=False)

    train_xs = torch.load('/tmp/datasets/mnist_train_xs.pt', map_location=device)
    train_ys = torch.load('/tmp/datasets/mnist_train_ys.pt', map_location=device)
    test_xs = torch.load('/tmp/datasets/mnist_test_xs.pt', map_location=device)
    test_ys = torch.load('/tmp/datasets/mnist_test_ys.pt', map_location=device)

    model = MLP()
    optimizer = Adam(model.parameters(), lr=1e-3)

    batches_per_epoch = (len(train_xs) - 1) // batch_size + 1
    # Training loop
    for i in tqdm(range(epochs * batches_per_epoch), desc='Training'):
        i = i % batches_per_epoch
        b_xs = train_xs[i * batch_size:(i + 1) * batch_size]
        b_ys = train_ys[i * batch_size:(i + 1) * batch_size]

        y = model(b_xs)
        loss = F.nll_loss(torch.log_softmax(y, dim=1), b_ys, reduction='sum')  # Cross-entropy
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f'{loss.item():.2e}')

    # Test phase
    model.eval()  # Set the model to evaluation mode
    correct = 0
    test_batch_size = 1000  # Can adjust as needed
    batches_per_test = (len(test_xs) - 1) // test_batch_size + 1
    with torch.no_grad():
        for i in range(batches_per_test):
            b_xs = test_xs[i * test_batch_size:(i + 1) * test_batch_size]
            b_ys = test_ys[i * test_batch_size:(i + 1) * test_batch_size]
            y = model(b_xs)
            pred = y.argmax(dim=1, keepdim=True)
            correct += pred.eq(b_ys.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_xs)
    print(f'\nTest accuracy: {accuracy:.2f}%')
from math import floor
from pathlib import Path
from typing import List
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Network(nn.Module):
    POOL_KERNEL_SIZE = 2
    POOL_STRIDE = 2

    CONV1_IN = 3
    CONV1_OUT = 6
    CONV1_KERNEL_SIZE = 5

    CONV2_OUT = 16
    CONV2_KERNEL_SIZE = 5

    FC1_OUT = 120
    FC2_OUT = 84

    _DICT_CLASSES = 'classes'
    _DICT_MODEL = 'model'
    _DICT_DIMENSIONS = 'dimensions'

    def __init__(self, img_dim: int = 320, out_num: int = 33, classes: List[str] = None):
        super().__init__()

        self.pool = nn.MaxPool2d(self.POOL_KERNEL_SIZE, self.POOL_STRIDE)
        self.activ = nn.Sigmoid()

        self.conv1 = nn.Conv2d(
            self.CONV1_IN, self.CONV1_OUT, self.CONV1_KERNEL_SIZE)
        self.conv2 = nn.Conv2d(
            self.CONV1_OUT, self.CONV2_OUT, self.CONV2_KERNEL_SIZE)

        dim_trans = self._pooled_dim(self._pooled_dim(img_dim))

        self.fc1 = nn.Linear(self.CONV2_OUT * dim_trans *
                             dim_trans, self.FC1_OUT)
        self.fc2 = nn.Linear(self.FC1_OUT, self.FC2_OUT)
        self.fc3 = nn.Linear(self.FC2_OUT, out_num)

        self.classes = tuple(classes) if classes else tuple()
        self.img_dim = img_dim

        self.to(DEVICE)

    def _pooled_dim(self, dimensions: int):
        padding = self.pool.padding
        dilation = self.pool.dilation
        k_size = self.pool.kernel_size
        stride = self.pool.stride

        numerator = dimensions + 2 * padding - dilation * (k_size - 1) - 1
        return floor(numerator / stride + 1) - 2

    def forward(self, x):
        x = self.pool(self.activ(self.conv1(x)))
        x = self.pool(self.activ(self.conv2(x)))

        if self.training:
            x = torch.flatten(x, 1)  # flatten all dimensions except batch
        else:
            x = torch.flatten(x)

        x = self.activ(self.fc1(x))
        x = self.activ(self.fc2(x))
        x = self.activ(self.fc3(x))

        return x

    @classmethod
    def load(cls, path: Path | str) -> 'Network':

        load_dict = torch.load(path, map_location=DEVICE)

        img_dim = load_dict[cls._DICT_DIMENSIONS]
        classes = load_dict[cls._DICT_CLASSES]
        classes_n = len(classes)

        net = Network(img_dim, classes_n, classes)
        net.load_state_dict(load_dict[cls._DICT_MODEL])

        return net

    def save(self, path: Path | str):

        save_dict = {
            self._DICT_MODEL: self.state_dict(),
            self._DICT_CLASSES: self.classes,
            self._DICT_DIMENSIONS: self.img_dim
        }
        torch.save(save_dict, path)


class Trainer:
    verbose = False
    report_freq = 4

    def __init__(self, learning_rate: float = 0.001, optimizer=None, criterion=None):

        self.learning_rate = learning_rate

        self.optimizer = optimizer
        self.criterion = criterion

    def train(self, network: Network, loader: DataLoader) -> float:

        data_len = len(loader)
        report = data_len // self.report_freq

        if not self.optimizer:
            self.optimizer = optim.Adam(
                network.parameters(), lr=self.learning_rate)

        if not self.criterion:
            self.criterion = nn.CrossEntropyLoss()

        loss_sum = 0
        avg_loss = 0

        for i, data in enumerate(loader):
            x, expected = data[0].to(DEVICE), data[1].to(DEVICE)

            network.zero_grad()
            output = network(x)

            loss = self.criterion(output, expected)
            loss.backward()
            self.optimizer.step()

            loss_sum += loss
            avg_loss = loss_sum / (i + 1)

            if self.verbose and i % report == 0:
                print(
                    f'Progress: {100 * (i + 1) / data_len:.2f}% (avg. loss: {avg_loss:.3f})')

        return avg_loss


class Tester:

    def __init__(self):

        self.correct = None
        self.total = None

    def test(self, network: Network, loader: DataLoader) -> tuple[int, int]:
        self.correct = 0
        self.total = 0

        with torch.no_grad():
            for data in loader:
                x, label = data[0].to(DEVICE), data[1].to(DEVICE)
                output = network(x)

                for idx, i in enumerate(output):
                    if torch.argmax(i) == label[idx]:
                        self.correct += 1
                    self.total += 1

        return self.correct, self.total

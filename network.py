from math import floor
from typing import List
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Network(nn.Module):

    _DICT_CLASSES = 'classes'
    _DICT_MODEL = 'model'
    _DICT_DIMENSIONS = 'dimensions'

    def __init__(self, img_dim: int = 320, out_num: int = 33, classes: List[str] = None):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),

            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2)
        )

        pool = nn.MaxPool2d(3, 2)

        self.classify = nn.Sequential(
            nn.Linear(16384, 512),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(512, 512),
            nn.ReLU(True),

            nn.Linear(512, out_num)
        )

        self.classes = tuple(classes) if classes else tuple()
        self.img_dim = img_dim

        self.to(DEVICE)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.classify(x)

        return x

    @classmethod
    def load(cls, path: str) -> 'Network':

        load_dict = torch.load(path, map_location=DEVICE)

        img_dim = load_dict[cls._DICT_DIMENSIONS]
        classes = load_dict[cls._DICT_CLASSES]
        classes_n = len(classes)

        net = Network(img_dim, classes_n, classes)
        net.load_state_dict(load_dict[cls._DICT_MODEL])

        return net

    def save(self, path: str):

        save_dict = {
            self._DICT_MODEL: self.state_dict(),
            self._DICT_CLASSES: self.classes,
            self._DICT_DIMENSIONS: self.img_dim
        }
        torch.save(save_dict, path)


class Trainer:
    verbose = False
    report_freq = 4

    def __init__(self, learning_rate: float = 0.0001, optimizer=None, criterion=None):

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

        network.train()
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
                    f'Progress: {100 * (i + 1) / data_len:.2f}% (avg. loss: {loss:.3f})')

        return avg_loss


class Tester:

    def __init__(self):

        self.correct = None
        self.total = None

    def test(self, network: Network, loader: DataLoader):
        self.correct = 0
        self.total = 0

        network.eval()
        with torch.no_grad():
            for data in loader:
                image, label = data[0].to(DEVICE), data[1].to(DEVICE)
                output = torch.flatten(network(image))

                if torch.argmax(output).item() == label.item():
                    self.correct += 1
                self.total += 1

        return self.correct, self.total

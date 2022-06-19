from typing import List
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMG_SIZE = 224


class Network(models.MobileNetV2):

    _DICT_CLASSES = 'classes'
    _DICT_MODEL = 'model'

    def __init__(self, out_num: int = 33, classes: List[str] = None):
        super().__init__(num_classes=out_num)
        self.classes = tuple(classes) if classes else tuple()
        self.to(DEVICE)

    @classmethod
    def load(cls, path: str) -> 'Network':

        load_dict = torch.load(path, map_location=DEVICE)

        classes = load_dict[cls._DICT_CLASSES]
        classes_n = len(classes)

        net = Network(classes_n, classes)
        net.load_state_dict(load_dict[cls._DICT_MODEL])

        return net

    def save(self, path: str):

        save_dict = {
            self._DICT_MODEL: self.state_dict(),
            self._DICT_CLASSES: self.classes
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

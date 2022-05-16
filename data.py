from pathlib import Path
from time import time
from numpy import append
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


class Loader():

    def __init__(self, data_home: str | Path = 'data', max_rotation: int = 30, max_crop: int = 320, batch_size: int = 64):

        self.data_home = data_home if isinstance(
            data_home, Path) else Path(data_home)
        self.train_path = self.data_home.joinpath('train')
        self.test_path = self.data_home.joinpath('test')

        self.max_rotation = max_rotation
        self.max_crop = max_crop

        self.batch_size = batch_size

    def load_train(self) -> tuple[ImageFolder, DataLoader]:

        transform = self._get_train_transform()

        dataset = ImageFolder(self.train_path, transform=transform)
        loader = DataLoader(dataset, batch_size=self.batch_size)

        return dataset, loader

    def load_test(self) -> tuple[ImageFolder, DataLoader]:

        transform = self._get_test_transform()

        dataset = ImageFolder(self.test_path, transform=transform)
        loader = DataLoader(dataset)

        return dataset, loader

    def _get_train_transform(self):
        return transforms.Compose([
            transforms.RandomRotation(self.max_rotation),
            transforms.RandomResizedCrop(self.max_crop),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    def _get_test_transform(self):
        return transforms.Compose([
            transforms.CenterCrop(self.max_crop),
            transforms.ToTensor()
        ])


class TrainLogger():

    HEADER = 'Epoch,Elapsed,Loss\n'

    def __init__(self, path: str | Path, append: bool = False):

        self.path = path if isinstance(path, Path) else Path(path)
        self.append = append

        self._prev_time = None
        self._finished_epochs = 0

    def start(self):
        if self.append:
            self._prev_time = time()
            return

        with open(self.path, 'w') as f:
            f.write(self.HEADER)

        self._prev_time = time()
        self._finished_epochs = 0

    def log_epoch(self, loss: float):
        curr_time = time()
        elapsed = curr_time - self._prev_time

        line = f'{self._finished_epochs + 1},{elapsed:.3f},{loss}\n'
        with open(self.path, 'a') as f:
            f.write(line)

        self._finished_epochs += 1
        self._prev_time = curr_time

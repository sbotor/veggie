from pathlib import Path
from time import time
from torch import Tensor
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from network import DEVICE, IMG_SIZE
import matplotlib.pyplot as plt
import PIL.Image


PLANT_TYPES = {
    0: 'fruit',
    1: 'fruit',
    2: 'vegetable',
    3: 'vegetable',
    4: 'vegetable',
    5: 'vegetable',
    6: 'vegetable',
    7: 'vegetable',
    8: 'vegetable',
    9: 'vegetable',
    10: 'vegetable',
    11: 'vegetable',
    12: 'vegetable',
    13: 'fruit',
    14: 'vegetable',
    15: 'fruit',
    16: 'fruit',
    17: 'vegetable',
    18: 'fruit',
    19: 'vegetable',
    20: 'fruit',
    21: 'fruit',
    22: 'vegetable',
    23: 'fruit',
    24: 'fruit',
    25: 'vegetable',
    26: 'vegetable',
    27: 'vegetable',
    28: 'vegetable',
    29: 'vegetable',
    30: 'vegetable',
    31: 'vegetable',
    32: 'fruit'
}


class Loader:
    _ROT = 30
    _BATCH_SIZE = 32

    def __init__(self, data_home: str = 'data'):
        self.data_home = data_home if isinstance(
            data_home, Path) else Path(data_home)
        self.train_path = self.data_home.joinpath('train')
        self.test_path = self.data_home.joinpath('test')

        self.batch_size = 32

    def load_train(self):
        transform = self._get_train_transform()

        dataset = ImageFolder(self.train_path, transform=transform)
        loader = DataLoader(dataset, batch_size=self._BATCH_SIZE, shuffle=True)

        return dataset, loader

    def load_test(self):
        transform = self._get_test_transform()

        dataset = ImageFolder(self.test_path, transform=transform)
        loader = DataLoader(dataset)

        return dataset, loader

    @classmethod
    def _get_train_transform(cls) -> transforms.Compose:
        return transforms.Compose([
            transforms.RandomRotation(cls._ROT),
            transforms.Resize(IMG_SIZE),
            transforms.CenterCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225])
        ])

    @classmethod
    def _get_test_transform(cls) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225])
        ])


class TrainLogger:
    HEADER = 'Epoch;Elapsed;Loss\n'

    def __init__(self, path: str, append: bool = False):
        self.path = path if isinstance(path, Path) else Path(path)
        self.append = append

        self._prev_time = None
        self._finished_epochs = 0

    def start(self, epoch: int = None):
        if not self.append:
            with open(self.path, 'w') as f:
                f.write(self.HEADER)

        epoch = epoch if epoch else 1
        self._prev_time = time()
        self._finished_epochs = epoch - 1

    def log_epoch(self, loss: float):
        curr_time = time()
        elapsed = curr_time - self._prev_time

        line = f'{self._finished_epochs + 1};{elapsed:.3f};{loss}\n'
        with open(self.path, 'a') as f:
            f.write(line)

        self._finished_epochs += 1
        self._prev_time = curr_time


class TestLogger:
    HEADER = 'Elapsed;Correct;Total\n'

    def __init__(self, path: str, append: bool = False):
        self.path = path if isinstance(path, Path) else Path(path)
        self.append = append

        self._prev_time = None

    def start(self):
        if not self.append:
            with open(self.path, 'w') as f:
                f.write(self.HEADER)

        self._prev_time = time()

    def log_test_result(self, correct: int, total: int):
        curr_time = time()
        elapsed = curr_time - self._prev_time

        line = f'{elapsed:.3f};{correct};{total}\n'
        with open(self.path, 'a') as f:
            f.write(line)

        self._prev_time = curr_time


def read_img(img_path: str):
    trans = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])
    ])

    img_path = img_path.absolute() if isinstance(
        img_path, Path) else Path(img_path).absolute()

    raw_img = PIL.Image.open(img_path)
    return trans(raw_img).float().to(DEVICE), raw_img


def show_img(img: Tensor, label: str = ''):
    plt.title(label)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

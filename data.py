from pathlib import Path
from time import time
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.io import read_image
from network import DEVICE


def read_img(img_path: str, max_crop: int = 320):

    img_path = img_path if isinstance(
        img_path, Path) else Path(img_path)

    transform = transforms.Compose([
        transforms.CenterCrop(max_crop)
    ])

    raw_img = read_image(str(img_path.absolute())).to(DEVICE)
    return transform(raw_img).float()


class Loader:

    IMG_SIZE = 320

    def __init__(self, data_home: str = 'data', max_rotation: int = 30, max_crop: int = 320, batch_size: int = 32):

        self.data_home = data_home if isinstance(
            data_home, Path) else Path(data_home)
        self.train_path = self.data_home.joinpath('train')
        self.test_path = self.data_home.joinpath('test')

        self.max_rotation = max_rotation
        self.max_crop = max_crop

        self.batch_size = batch_size

    def load_train(self):

        transform = self._get_train_transform()

        dataset = ImageFolder(self.train_path, transform=transform)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        return dataset, loader

    def load_test(self):

        transform = self._get_test_transform()

        dataset = ImageFolder(self.test_path, transform=transform)
        loader = DataLoader(dataset)

        return dataset, loader

    def _get_train_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.RandomRotation(self.max_rotation),
            transforms.RandomResizedCrop(self.max_crop),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    def _get_test_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.CenterCrop(self.max_crop),
            transforms.ToTensor()
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

from pathlib import Path
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import time

from network import AbstractNetwork, Network, SimpleNetwork
import debug

ROTATION = 30
RESIZE = 380
CROP = 320
CROP2 = CROP * CROP
BATCH_SIZE = 32

DATA_HOME = 'data'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 10
LEARNING_RATE = 0.001

PRINT = True
REPORT_FREQ = 20
SAVE = True
SAVE_PATH = 'model.pth'


def load_train_data(home_path: str | Path) -> tuple[ImageFolder, DataLoader]:

    data_path = Path(home_path).joinpath('train')
    transform = transforms.Compose([
        transforms.RandomRotation(ROTATION),
        transforms.RandomResizedCrop(CROP),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    dataset = ImageFolder(data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    return dataset, loader


def load_test_data(home_path: str | Path) -> tuple[ImageFolder, DataLoader]:

    data_path = Path(home_path).joinpath('test')
    transform = transforms.Compose([
        transforms.Resize(RESIZE),
        transforms.CenterCrop(CROP),
        transforms.ToTensor()
    ])

    dataset = ImageFolder(data_path, transform=transform)
    loader = DataLoader(dataset, )

    return dataset, loader


def train(net: AbstractNetwork, train_loader: DataLoader, criterion = None, optimizer = None):
    optimizer = optimizer or optim.Adam(net.parameters(), lr=LEARNING_RATE)
    criterion = criterion or nn.CrossEntropyLoss()
    
    loss = 0.0
    length = len(train_loader)

    for epoch in range(EPOCHS):
        for i, data in enumerate(train_loader):
            x, y = data[0].to(DEVICE), data[1].to(DEVICE)
            net.zero_grad()

            trans_x = net.prepare(x)
            output = net(trans_x)

            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            if PRINT and i % REPORT_FREQ == 0:
                print(
                    f'Epoch: {epoch + 1}/{EPOCHS}, progress {i + 1}/{length}, loss: {loss.item():.3f}')
                # print(output)

        if PRINT:
            print(
                f'---Epoch: {epoch + 1} finished, loss: {loss.item():.3f}---')


def validate(net: AbstractNetwork, test_loader: DataLoader) -> tuple[int, int]:
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            image, label = data[0].to(DEVICE), data[1].to(DEVICE)
            output = net(net.prepare(image))

            for idx, i in enumerate(output):
                if torch.argmax(i) == label[idx]:
                    correct += 1
                total += 1

        if PRINT:
            print(
                f'Accuracy: {correct} / {total} ({(correct * 100.0 / total):.3f}%)')

    return correct, total


def main():
    train_dataset, train_loader = load_train_data(DATA_HOME)
    classes_n = len(train_dataset.classes)

    # net = SimpleNetwork(CROP, classes_n)
    net = Network(CROP, classes_n)

    net.to(DEVICE)
    print(f'Training on device: "{DEVICE.type}".\n')
    t_start = time.process_time()
    train(net, train_loader)
    t_stop = time.process_time()
    print("Training finished.\n")

    _, test_loader = load_test_data(DATA_HOME)
    correct, total = validate(net, test_loader)

    if SAVE:
        torch.save(net.state_dict(), SAVE_PATH)

    debug.dump_info(classes_n, t_stop - t_start, EPOCHS, correct, total)


if __name__ == '__main__':
    main()

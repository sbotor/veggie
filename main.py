from pathlib import Path
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch
import time

from network import Network
import debug

ROTATION = 30
RESIZE = 380
CROP = 320
CROP2 = CROP * CROP
BATCH_SIZE = 32

DATA_HOME = 'data'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 1
LEARNING_RATE = 0.001

PRINT = True
SAVE = True
SAVE_PATH = 'model'


def load_train_data(home_path: str | Path) -> ImageFolder:

    data_path = Path(home_path).joinpath('train')
    transform = transforms.Compose([
        transforms.RandomRotation(ROTATION),
        transforms.RandomResizedCrop(CROP),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    dataset = ImageFolder(data_path, transform=transform)

    return dataset


def load_test_data(home_path: str | Path) -> ImageFolder:

    data_path = Path(home_path).joinpath('test')
    transform = transforms.Compose([
        transforms.Resize(RESIZE),
        transforms.CenterCrop(CROP),
        transforms.ToTensor()
    ])

    dataset = ImageFolder(data_path, transform=transform)

    return dataset


def train(net: Network, train_loader: DataLoader):
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    loss = 0.0
    running_loss = 0.0

    for epoch in range(EPOCHS):
        for data in train_loader:
            x, y = data
            net.zero_grad()

            trans_x = x.view(-1, CROP2 * 3)
            output = net(trans_x)

            loss = F.nll_loss(output, y)
            loss.backward()

            optimizer.step()
            running_loss += float(loss.item())

        if PRINT:
            print(
                f'Epoch: {epoch + 1}, loss: {loss:.3f}, running loss: {running_loss:.3f}')

    if PRINT:
        print("Training finished.")


def validate(net: Network, test_loader: DataLoader) -> tuple[int, int]:
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            input, target = data
            output = net(input.view(-1, CROP2 * 3))

            for idx, i in enumerate(output):
                if torch.argmax(i) == target[idx]:
                    correct += 1
                total += 1

        if PRINT:
            print(
                f'Accuracy: {correct} / {total} ({(correct * 100.0 / total):.3f}%)')

    return correct, total


def main():
    train_dataset = load_train_data(DATA_HOME)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    classes_n = len(train_dataset.classes)

    net = Network(CROP2 * 3, classes_n)
    net.to(DEVICE)

    t_start = time.process_time()
    #train(net, train_loader)
    t_stop = time.process_time()

    test_dataset = load_test_data(DATA_HOME)
    test_loader = DataLoader(test_dataset)
    correct, total = validate(net, test_loader)

    if SAVE:
        torch.save(net.state_dict(), SAVE_PATH)

    debug.dump_info(classes_n, t_stop - t_start, correct, total)


if __name__ == '__main__':
    main()

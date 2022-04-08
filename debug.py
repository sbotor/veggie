import matplotlib.pyplot as plt
from pathlib import Path

def dataset_imshow(dataset):
    img = next(iter(dataset))[0]
    imshow(img)

def imshow(tensor_image):
    plt.imshow(tensor_image.permute(1, 2, 0))
    plt.show()

def dump_info(classes_n: int, train_time: float, correct_n: int, total_n: int, path: str | Path = 'info.txt'):
    lines = [
            f'Number of classes: {classes_n}',
            f'Train time: {train_time} s',
            f'Accuracy: {correct_n}/{total_n} ({(correct_n * 100 / total_n):.3f}%)'
        ]
    buffer = '\n'.join(lines)
    
    with open(path, 'w') as file:
        file.write(buffer)

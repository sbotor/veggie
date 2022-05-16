from argparse import ArgumentParser

from network import DEVICE, Network, Trainer
from data import TrainLogger, Loader, read_img
from pathlib import Path
import torch


def _train(args):

    print(f'*** Training ({DEVICE})... ***')

    data_path = Path(args.data).absolute()
    inp_path = Path(args.input).absolute() if args.input else None
    out_path = inp_path if inp_path else Path(args.output).absolute()

    if args.log:
        if args.log == '-':
            log_path = Path('train.csv')
        else:
            log_path = Path(args.log)

        logger = TrainLogger(log_path)
        starting_epoch = 1
        if args.append_log:
            logger.append = True
            if args.append_log > 1:
                starting_epoch = args.append_log
    else:
        logger = None

    loader = Loader(data_path)
    img_folder, data = loader.load_train()

    classes_n = len(img_folder.classes)
    trainer = Trainer()
    trainer.learning_rate = args.learning_rate or trainer.learning_rate
    trainer.verbose = args.verbose

    network = Network(loader.max_crop, classes_n, img_folder.classes)

    if inp_path:
        network.load(inp_path)

    if logger:
        logger.start(starting_epoch)

    for epoch in range(args.epochs):
        print(f'\n--- Training epoch {epoch + 1}/{args.epochs} ---')

        loss = trainer.train(network, data)

        print(
            f'--- Epoch {epoch + 1}/{args.epochs} finished (avg. loss: {loss:.3f}) ---')

        if logger:
            logger.log_epoch(loss)

    network.save(out_path)


def _classify(args):
    model_path = Path(args.model).absolute()
    image_path = Path(args.image).absolute()

    net = Network()
    net.load(model_path)
    img = read_img(image_path)

    net.eval()
    result = net(img)
    i = torch.argmax(result)

    if (net.classes and i < len(net.classes)):
        print(f'{net.classes[i]} ({100 * result[i] / sum(result):.2f}%)')
    else:
        print(f'{i} ({100 * result[i] / sum(result):.2f}%)')


def _add_train_parser(subparsers):
    train_parser = subparsers.add_parser(
        'train', help='TODO: train help', aliases=['t'])

    train_parser.add_argument('data', help='root data folder path')

    train_parser.add_argument(
        '--verbose', '-v', action='store_true', help='verbose training info')

    train_parser.add_argument(
        '--input', '-i', help='input model (new model will be created if not present)')
    train_parser.add_argument('--output', '-o', default='model.pt',
                              help='model output path (default: model.pt)')
    
    train_parser.add_argument(
        '--epochs', '-e', type=int, default=1, help='number of epochs (default: 1)')
    train_parser.add_argument(
        '--learning-rate', '--lr', type=float, help='learning rate')

    train_parser.add_argument('--log', nargs='?', const='-',
                              help='save results to a csv file if present (default: train.csv)')
    train_parser.add_argument(
        '--append-log', type=int, nargs='?', const=0, help='append to the log file starting at the specified epoch (default: 1)')

    train_parser.set_defaults(func=_train)

def _add_validation_parser(subparsers):
    # TODO

    validation_parser = subparsers.add_parser(
        'validate', help='TODO: validate help', aliases=['v'])


def _add_classification_parser(subparsers):
    class_parser = subparsers.add_parser(
        'classify', help='TODO: classification help', aliases=['class', 'c'])

    class_parser.add_argument('image', help='image path')
    class_parser.add_argument(
        '--model', '-m', default='model.pt', help='trained model path (default: model.pt)')

    class_parser.set_defaults(func=_classify)


def _get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()

    _add_train_parser(subparsers)
    _add_validation_parser(subparsers)
    _add_classification_parser(subparsers)

    return parser


def main():
    parser = _get_parser()
    args = parser.parse_args()

    args.func(args)


if __name__ == '__main__':
    main()

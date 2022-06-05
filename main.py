from argparse import ArgumentParser

from network import DEVICE, Network, Trainer, Tester
from data import TrainLogger, Loader, read_img, TestLogger
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
    classes = img_folder.classes

    trainer = Trainer()
    trainer.learning_rate = args.learning_rate or trainer.learning_rate
    trainer.verbose = args.verbose

    network = None
    if inp_path:
        network = Network.load(inp_path)
    else:
        network = Network(loader.IMG_SIZE, len(classes), classes)

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


def _test(args):
    print(f'*** Testing ({DEVICE})... ***')

    data_path = Path(args.data).absolute()
    model_path = Path(args.model).absolute()

    print(model_path)

    net = Network.load(model_path)
    # inp_path = Path(args.input).absolute() if args.input else None
    # out_path = inp_path if inp_path else Path(args.output).absolute()

    if args.log:
        if args.log == '-':
            log_path = Path('test.csv')
        else:
            log_path = Path(args.log)

        logger = TestLogger(log_path)
        if args.append_log:
            logger.append = True
    else:
        logger = None

    loader = Loader(data_path)
    img_folder, data = loader.load_test()

    tester = Tester()

    if logger:
        logger.start()

    correct, total = tester.test(net, data)

    if logger:
        logger.log_test_result(correct, total)


def _classify(args):
    model_path = Path(args.model).absolute()
    image_path = Path(args.image).absolute()

    net = Network.load(model_path)
    img = read_img(image_path)

    with torch.no_grad():
        net.eval()
        result = net(img)
        i = torch.argmax(result)
        print(f'{result}')
   
    if net.classes and i < len(net.classes):
        print(f'{net.classes[i]} ({result[i]:.3f})')
    else:
        print(f'{i} ({result[i]:.3f})')


def _add_train_parser(subparsers):
    train_parser = subparsers.add_parser(
        'train', help='TODO: train help')

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
        '--append-log', type=int, nargs='?', const=0,
        help='append to the log file starting at the specified epoch (default: 1)')

    train_parser.set_defaults(func=_train)


def _add_test_parser(subparsers):

    test_parser = subparsers.add_parser(
        'test', help='TODO: test help')

    test_parser.add_argument('data', help='root data folder path')

    test_parser.add_argument('--log', nargs='?', const='-',
                                   help='save results to a csv file if present (default: train.csv)')
    test_parser.add_argument(
        '--append-log', type=int, nargs='?', const=0,
        help='append to the log file starting at the specified epoch (default: 1)')

    test_parser.add_argument(
        '--model', '-m', default='model.pt', help='trained model path (default: model.pt)')

    test_parser.set_defaults(func=_test)


def _add_classification_parser(subparsers):
    class_parser = subparsers.add_parser(
        'classify', help='TODO: classification help', aliases=['class'])

    class_parser.add_argument('image', help='image path')
    class_parser.add_argument(
        '--model', '-m', default='model.pt', help='trained model path (default: model.pt)')

    class_parser.set_defaults(func=_classify)


def _get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()

    _add_train_parser(subparsers)
    _add_test_parser(subparsers)
    _add_classification_parser(subparsers)

    return parser


def main():
    parser = _get_parser()
    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

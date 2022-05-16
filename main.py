from argparse import ArgumentParser
from network import DEVICE, Network, Trainer
from data import TrainLogger, Loader
from pathlib import Path


def _train(args):

    print(f'*** Training ({DEVICE})... ***')

    data_path = Path(args.data).absolute()
    inp_path = Path(args.input).absolute() if args.input else None
    out_path = inp_path if inp_path else Path(args.output).absolute()
    
    if args.log == '-':
        log_path = Path('train.csv')
    elif args.log:
        log_path = Path(args.log)
    else:
        log_path = None

    logger = TrainLogger(log_path, args.append_log) if log_path else None
    loader = Loader(data_path)
    _, data = loader.load_train()

    data_len = len(data)
    trainer = Trainer()
    trainer.learning_rate = args.learning_rate or trainer.learning_rate
    trainer.verbose = args.verbose

    network = Network(loader.max_crop, data_len)

    if inp_path:
        network.load(inp_path)

    if logger:
        logger.start()

    for epoch in range(args.epochs):
        print(f'\n--- Training epoch {epoch + 1}/{args.epochs} ---')

        loss = trainer.train(network, data)

        print(
            f'--- Epoch {epoch + 1}/{args.epochs} finished (avg. loss: {loss:.3f}) ---')

        if logger:
            logger.log_epoch(loss)

    network.save(out_path)


def _get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser(
        'train', help='TODO: train help', aliases=['t'])
    train_parser.add_argument('data', help='root data folder path')
    train_parser.add_argument(
        '--input', '-i', type=str, help='input model (new model will be created if not present)')
    train_parser.add_argument(
        '--epochs', '-e', type=int, default=1, help='number of epochs (default: 1)')
    train_parser.add_argument('--output', '-o', type=str, default='model.pt',
                              help='model output path (default: model.pt)')
    train_parser.add_argument(
        '--learning-rate', '--lr', type=float, help='learning rate')
    train_parser.add_argument('--log', type=str, nargs='?', const='-',
                              help='save results to a csv file if present (default: train.csv)')
    train_parser.add_argument(
        '--verbose', '-v', action='store_true', help='verbose training info')
    train_parser.add_argument(
        '--append-log', action='store_true', help='append to the log file')
    train_parser.set_defaults(func=_train)

    validation_parser = subparsers.add_parser(
        'validate', help='TODO: validate help', aliases=['v'])

    classification_parser = subparsers.add_parser(
        'class', help='TODO: classification help', aliases=['c'])

    return parser


def main():
    parser = _get_parser()
    args = parser.parse_args()

    args.func(args)


if __name__ == '__main__':
    main()

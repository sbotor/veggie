from network import DEVICE, Network, Trainer, Tester
from data import TrainLogger, Loader, read_img, show_img, TestLogger, PLANT_TYPES
from pathlib import Path
import torch
from parser import Parser


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
        network = Network(len(classes), classes)

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

    net = Network.load(model_path)

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
    _, data = loader.load_test()

    tester = Tester()

    if logger:
        logger.start()

    correct, total = tester.test(net, data)

    if logger:
        logger.log_test_result(correct, total)

    print(f'Accuracy: {correct}/{total} ({100 * correct / total:.1f}%)')


def _classify(args):
    model_path = Path(args.model).absolute()
    image_path = Path(args.image).absolute()

    net = Network.load(model_path)
    images = read_img(image_path)

    with torch.no_grad():
        net.eval()
        result = net(images[0].unsqueeze_(0)).flatten()
        i = torch.argmax(result).item()

    if net.classes and i < len(net.classes):
        print(
            f'{net.classes[i]} [{PLANT_TYPES.get(i, "plant")}] ({result[i]:.3f})')
        if args.show:
            show_img(
                images[1], f'Prediction: {net.classes[i]} [{PLANT_TYPES.get(i, "plant")}]')
    else:
        print(f'{i} ({result[i]:.3f})')
        if args.show:
            show_img(images[1])


def main():
    parser = Parser(train_f=_train, test_f=_test, class_f=_classify)
    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

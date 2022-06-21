from argparse import ArgumentParser


class Parser(ArgumentParser):
    def __init__(self, train_f: callable, test_f: callable, class_f: callable):
        super().__init__()

        self._subparsers = self.add_subparsers(parser_class=ArgumentParser)

        self._add_train_parser(train_f)
        self._add_test_parser(test_f)
        self._add_classification_parser(class_f)

    def _add_train_parser(self, func: callable):
        self._train_parser = self._subparsers.add_parser(
            'train', help='train the model')

        self._train_parser.add_argument('data', help='root data folder path')

        self._train_parser.add_argument(
            '--verbose', '-v', action='store_true', help='verbose training info')

        self._train_parser.add_argument(
            '--input', '-i', help='input model (new model will be created if not present)')
        self._train_parser.add_argument('--output', '-o', default='model.pt',
                                        help='model output path (default: model.pt)')

        self._train_parser.add_argument(
            '--epochs', '-e', type=int, default=1, help='number of epochs (default: 1)')
        self._train_parser.add_argument(
            '--learning-rate', '--lr', type=float, help='learning rate (default: 0.001)')

        self._train_parser.add_argument('--log', nargs='?', const='-',
                                        help='save results to a csv file if present (default: train.csv)')
        self._train_parser.add_argument(
            '--append-log', type=int, nargs='?', const=0,
            help='append to the log file starting at the specified epoch (default: 1)')

        self._train_parser.set_defaults(func=func)

    def _add_test_parser(self, func: callable):

        self._test_parser = self._subparsers.add_parser(
            'test', help='perform accuracy testing on the model')

        self._test_parser.add_argument('data', help='root data folder path')

        self._test_parser.add_argument('--log', nargs='?', const='-',
                                       help='save results to a csv file if present (default: train.csv)')

        self._test_parser.add_argument(
            '--append-log', action='store_true', help='append to the log file')

        self._test_parser.add_argument(
            '--model', '-m', default='model.pt', help='trained model path (default: model.pt)')

        self._test_parser.set_defaults(func=func)

    def _add_classification_parser(self, func: callable):
        self._class_parser = self._subparsers.add_parser(
            'classify', help='classify a picture', aliases=['class'])

        self._class_parser.add_argument('image', help='image path')
        self._class_parser.add_argument(
            '--model', '-m', default='model.pt', help='trained model path (default: model.pt)')
        self._class_parser.add_argument(
            '--show', '-s', action='store_true', help='show the transformed image')

        self._class_parser.set_defaults(func=func)

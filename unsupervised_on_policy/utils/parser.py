import argparse


def parse_args(args):
    parser = argparse.ArgumentParser(description='Starting...')

    parser.add_argument('--load', dest="load",
                        help='If model is loaded or newly created.',
                        action="store_true")
    parser.set_defaults(load=False)
    parser.add_argument('--skip_pretrain', dest="skip_pretrain",
                        help='If pretraining is skipped or not.',
                        action="store_true")
    parser.set_defaults(load=False)
    parser.add_argument('--prefix', dest="prefix",
                        help='Prefix for the logging.', default='PRETRAIN')
    parser.add_argument('--model_path',
                        help='Path were model is saved and loaded from',
                        default='./saved_models')

    return parser.parse_args(args)

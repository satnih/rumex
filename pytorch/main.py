import argparse
from utils import load_config
from trainer import train

parser = argparse.ArgumentParser()
parser.add_argument('--config_file',
                    default='config.yaml',
                    help="config file path")


if __name__ == '__main__':
    args, unknown = parser.parse_known_args()
    cfg = load_config(args.config_file)
    train(cfg)

import yaml
import argparse
from trainer import train


parser = argparse.ArgumentParser()
parser.add_argument('--config_file',
                    default='config.yaml',
                    help="config file path")


if __name__ == '__main__':
    args, unknown = parser.parse_known_args()

    def load_config(config_file):
        with open(config_file) as file:
            config = yaml.safe_load(file)
        return config

    cfg = load_config(args.config_file)
    train(cfg)

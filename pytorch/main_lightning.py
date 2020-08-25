from lightning import start_training
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name',
                    default='shufflenet_v2',
                    help="pretrained model to be loaded")

if __name__ == '__main__':
    # Load the parameters from json file
    # args = parser.parse_args()  # -> this thros error \
    # ipykernel_launcher.py: error: unrecognized arguments:
    # use  args, unknown = parser.parse_known_args() for solution
    args, unknown = parser.parse_known_args()
    start_training(args.model_name)

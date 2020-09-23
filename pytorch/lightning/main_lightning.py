import argparse
from lightning import lightning_trainer

# resnet: lr [1e-4, 1e-3]
# mobilenet: lr [1e-4, 1e-3]
# shufflenet: lr [1e-3, 1e-1]
# mnasnet: lr [5e-3, 7e-2]


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default="shufflenet")
parser.add_argument('--num_epoch', default=5, type=int)
parser.add_argument('--min_lr', default=1e-4, type=float)
parser.add_argument('--max_lr', default=1e-2, type=float)

if __name__ == '__main__':
    args, unknown = parser.parse_known_args()
    lightning_trainer(args)

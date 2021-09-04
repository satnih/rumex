import torch
import random
import argparse
import numpy as np
import utils as ut
from trainer_10m import Trainer


def set_seed(myseed=0):
    np.random.seed(myseed)
    random.seed(myseed)
    torch.manual_seed(myseed)
    torch.cuda.manual_seed(myseed)
    torch.cuda.manual_seed_all(myseed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', type=str, default="data/10m/256/train_a1/")
parser.add_argument('--valid_dir', type=str, default="data/10m/256/valid/")
parser.add_argument('--test_dir', type=str, default="data/10m/256/test/")
parser.add_argument('--model_name', type=str, default="resnet")
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--bs', type=int, default=16)
parser.add_argument('--gpu', type=bool, default=True)

if __name__ == '__main__':
    args, unknown = parser.parse_known_args()
    print(args)
    exp_name = args.model_name + "10m"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset and data loader
    dstr = ut.RumexDataset(args.train_dir)
    dltr = ut.train_loader(dstr, args.bs)

    dsva = ut.RumexDataset(args.valid_dir)
    dlva = ut.test_loader(dsva, 2*args.bs)

    dste = ut.RumexDataset(args.test_dir)
    dlte = ut.test_loader(dste, 2*args.bs)

    model = ut.RumexNet(args.model_name)

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    lr_dict = {"vgg": 2.92e-04,
               "resnet": 2.48e-04,
               "mobilenet": 4.04e-04,
               "densenet": 1e-3,
               "mnasnet": 2.78e-03,
               "shufflenet": 5.34e-03,
               "efficientnet": 2.42e-03}

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr_dict[args.model_name])

    trainer = Trainer(model=model,
                      max_ep=args.num_epochs,
                      optimizer=optimizer,
                      loss_fn=loss_fn,
                      dltr=dltr,
                      dlva=dlva,
                      dlte=dlte,
                      bs=args.bs,
                      patience=args.patience,
                      device=device,
                      )
    trainer.fit()
    torch.save(trainer, f"{exp_name}_trainer.pt")

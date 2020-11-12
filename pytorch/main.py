import torch
import logging
import argparse
import utils as ut
from torch import optim
from pathlib import Path
from trainer import Trainer


parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', type=str, default="data/10m/256/train_wa/")
parser.add_argument('--valid_dir', type=str, default="data/10m/256/valid/")
parser.add_argument('--test_dir', type=str, default="data/10m/256/test/")
parser.add_argument('--log_dir', type=str, default="logs_temp")
parser.add_argument('--model_name', type=str, default="shufflenet")
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--bs', type=int, default=32)
parser.add_argument('--gpu', type=bool, default=True)

if __name__ == '__main__':
    args, unknown = parser.parse_known_args()
    print(args)
    exp_name = args.model_name
    log_dir = ut.set_logger(Path.cwd() / "logs_temp")
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

    lr_dict = {"resnet": 1e-3,
               "mobilenet": 7e-3,
               "densenet": 1e-3,
               "mnasnet": 1e-2,
               "shufflenet": 5e-2,
               "wide_resnet": 5e-4,
               "resnext": 5e-4}

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
                      device=device,
                      log_dir=log_dir)
    trainer.fit()
    torch.save(trainer, log_dir.joinpath(exp_name+"_trainer.pt"))

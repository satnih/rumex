import torch
import random
import argparse
import numpy as np
import utils as ut
from trainer_15m import (
    Trainer,
)  # train model for a known number of epochs; no early stopping required. It only requires train and test sets no validation set

# cant use trainer_10m because it has train early stopping since we dont know #epochs. It also needs three sets: train, valid and test

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, default="data/10m/")
parser.add_argument("--model_name", type=str, default="mobilenet")
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--num_epochs", type=int, default=13)
parser.add_argument("--bs", type=int, default=64)
parser.add_argument("--seed", type=int, default=0)

if __name__ == "__main__":
    args, unknown = parser.parse_known_args()
    print(args)
    ut.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds0 = ut.RumexDataset(f"{args.data_root}fold0/")
    dl0 = ut.train_loader(ds0, args.bs)

    ds1 = ut.RumexDataset(f"{args.data_root}fold1/")
    dl1 = ut.train_loader(ds1, args.bs)

    ds2 = ut.RumexDataset(f"{args.data_root}fold2/")
    dl2 = ut.train_loader(ds2, args.bs)

    ds3 = ut.RumexDataset(f"{args.data_root}fold3/")
    dl3 = ut.train_loader(ds3, args.bs)

    ds4 = ut.RumexDataset(f"{args.data_root}fold4/")
    dl4 = ut.train_loader(ds4, args.bs)

    ratio = args.ratio
    exp_name = f"{args.model_name}10"
    dstr = torch.utils.data.ConcatDataset([ds0, ds1, ds2, ds3])
    dste = ds4

    dltr = ut.train_loader(dstr, args.bs)
    dlte = ut.test_loader(dste, args.bs)
    model = ut.RumexNet(args.model_name)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    optimizer = torch.optim.Adam(model.parameters(), lr=4.04e-04)

    trainer = Trainer(
        model=model,
        max_ep=args.num_epochs,
        optimizer=optimizer,
        loss_fn=loss_fn,
        dltr=dltr,
        dlte=dlte,
        bs=args.bs,
        patience=args.patience,
        device=device,
    )
    trainer.fit()
    torch.save(trainer, f"{exp_name}.pt")

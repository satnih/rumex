import torch
import random
import argparse
import numpy as np
import utils as ut
from trainer_10m import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default="data/10m/")
parser.add_argument('--model_name', type=str, default="resnet")
parser.add_argument('--test_fold', type=int)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--bs', type=int, default=64)
parser.add_argument('--seed', type=int, default=0)

if __name__ == '__main__':
    args, unknown = parser.parse_known_args()
    print(args)    
    ut.set_seed(args.seed)    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds0 = ut.RumexDataset(f'{args.data_root}fold0/')
    dl0 = ut.train_loader(ds0, args.bs)
    
    ds1 = ut.RumexDataset(f'{args.data_root}fold1/')
    dl1 = ut.train_loader(ds1, args.bs)    

    ds2 = ut.RumexDataset(f'{args.data_root}fold2/')
    dl2 = ut.train_loader(ds2, args.bs)

    ds3 = ut.RumexDataset(f'{args.data_root}fold3/')
    dl3 = ut.train_loader(ds3, args.bs)

    ds4 = ut.RumexDataset(f'{args.data_root}fold4/')
    dl4 = ut.train_loader(ds4, args.bs)

    data_sets = [ds0, ds1, ds2, ds3, ds4]


    i = int(args.test_fold)
    exp_name = f'{args.model_name}_testfold{i}'
    dste = data_sets[i]
    dsva = data_sets[(i + 1)%5]
    dstr = torch.utils.data.ConcatDataset([data_sets[(i + 2)%5], 
                                            data_sets[(i + 3)%5], 
                                            data_sets[(i + 4)%5]])
    dlte = ut.test_loader(dste, args.bs)
    dlva = ut.test_loader(dsva, args.bs)
    dltr = ut.train_loader(dstr, args.bs)

    model = ut.RumexNet(args.model_name)

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    lr_dict = {
        "vgg": 2.92e-04,
        "resnet": 2.48e-04,
        "mobilenet": 4.04e-04,
        "densenet": 1e-3,
        "mnasnet": 2.78e-03,
        "shufflenet": 5.34e-03,
        "efficientnet": 2.42e-03
    }

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_dict[args.model_name])

    trainer = Trainer(
        model=model,
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
    torch.save(trainer, f"{exp_name}.pt")

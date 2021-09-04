import torch
import argparse
import utils as ut
from trainer_15m import Trainer


parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', type=str, default="data/15m/256/train/")
parser.add_argument('--test_dir', type=str, default="data/15m/256/test/")
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--num_epochs', type=int, default=13)
parser.add_argument('--bs', type=int, default=64)
parser.add_argument('--gpu', type=bool, default=True)

if __name__ == '__main__':
    ut.set_seed(0)
    args, unknown = parser.parse_known_args()
    print(args)
    model_name = "mobilenet"
    exp_name = f"15m_{model_name}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset and data loader
    dstr = ut.RumexDataset(args.train_dir)
    dltr = ut.train_loader(dstr, args.bs)

    dste = ut.RumexDataset(args.test_dir)
    dlte = ut.test_loader(dste, 2*args.bs)

    trainer = torch.load(
        f"results_v2/trained_models/a0/{model_name}_seed_0_256_trainer.pt")
    model = ut.RumexNet(model_name)
    model.load_state_dict(trainer.model.state_dict())

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    trainer = Trainer(model=model,
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
    # torch.save(trainer, f"{exp_name}_trainer.pt")

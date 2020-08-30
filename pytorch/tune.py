from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from rumex_dataset import RumexDataset
from rumex_model import RumexNet


def tuner(cfg, model_name=None, checkpoint_dir=None, data_dir=None, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dstr = RumexDataset(data_dir+'train/', train_flag=True)
    dsva = RumexDataset(data_dir+'valid/', train_flag=False)

    dltr = dstr.make_data_loader(cfg['bs'])
    dlva = dsva.make_data_loader(100)

    model = RumexNet(model_name)
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    model.to(device)
    for ep in np.arange(num_epochs):
        #### fit model ##########
        fit(model, dltr, optimizer, loss_fn)

        ##### Model Validation ##########
        loss = validate(model, dlva, loss_fn)

        # with tune.checkpoint_dir(step=ep) as checkpoint_dir:
        #     path = os.path.join(checkpoint_dir, "checkpoint")
        #     torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=loss)
    print("Finished Training")


def fit(model, dl, optimizer, loss_fn):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    for xb, yb, _, _ in dl:
        # Forward pass
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        scoreb = model(xb)
        lossb = loss_fn(scoreb, yb)
        lossb_mean = torch.mean(lossb)
        lossb_mean.backward()
        optimizer.step()


def validate(model, dl, loss_fn):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    loss = []
    with torch.no_grad():
        for xb, yb, _, fnameb in dl:
            xb = xb.to(device)
            yb = yb.to(device)

            # Forward pass
            scoreb = model(xb)  # logits
            lossb = loss_fn(scoreb, yb)

            # book keeping at batch level
            loss.append(lossb)

        loss = torch.cat(loss).mean()
    return loss.item()


search_space = {
    "lr": tune.loguniform(1e-7, 1e-1),
    "bs": tune.choice([16, 32, 64])
}

analysis = tune.run(
    partial(tuner,
            model_name='shufflenet_v2',
            data_dir='/u/21/hiremas1/unix/postdoc/rumex/data_for_fastai_cleaned/',
            num_epochs=10),
    # resources_per_trial={'gpu': 1},
    num_samples=10,
    scheduler=ASHAScheduler(metric="loss", mode="min"),
    local_dir='shufflenet_logs',
    config=search_space,
    verbose=1)


dfs = analysis.trial_dataframes
# Plot by epoch
ax = None  # This plots everything on the same plot
for d in dfs.values():
    print(d.loss)

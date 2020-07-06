# from https://tinyurl.com/yd4o3yf4
# %%
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torch.utils.data import DataLoader
from torchvision import models, datasets
from torchvision import transforms as T
import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
# plt.ion()   # interactive mode
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# writer = SummaryWriter()
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
# %%
# Data


def get_data(bs=16):
    common = T.Compose(
        [T.ToTensor(), T.Normalize(imagenet_mean, imagenet_std)])
    augment = T.Compose([T.RandomHorizontalFlip(), T.RandomVerticalFlip()])
    train_ds = datasets.ImageFolder(
        'data/train/', T.Compose([T.Resize(224), augment, common]))
    val_ds = datasets.ImageFolder(
        'data/val/', T.Compose([T.Resize(224), common]))
    train_dl = DataLoader(train_ds, batch_size=bs,
                          shuffle=True, num_workers=16)
    val_dl = DataLoader(val_ds, batch_size=bs*2, shuffle=True, num_workers=16)
    return (train_dl, val_dl)

# model


def get_model():
    model = models.alexnet(pretrained=True)
    model.classifier[6] = nn.Linear(4096, 2)
    model = model.to(device)
    return model, optim.SGD(model.parameters(), lr=lr)


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xb)


def fit(epochs, model, loss_func, opt, train_dl, val_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb.to(device), yb.to(device)) for xb, yb in val_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print(epoch, val_loss)


# %%
epochs = 3
lr = 0.001
bs = 8  # batch size
epochs = 10

train_dl, val_dl = get_data(bs)
loss_func = nn.CrossEntropyLoss()
model, opt = get_model()
fit(epochs, model, loss_func, opt, train_dl, val_dl)

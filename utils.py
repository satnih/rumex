# This code is based on code from Stanford cs230 class
# https://github.com/cs230-stanford/cs230-code-examples
import os
import torch
import random
import logging
import numpy as np
import torch.nn as nn
from torchvision import models
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from efficientnet_pytorch import EfficientNet
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.metrics import recall_score, f1_score, confusion_matrix


def set_seed(myseed=0):
    np.random.seed(myseed)
    random.seed(myseed)
    torch.manual_seed(myseed)
    torch.cuda.manual_seed(myseed)
    torch.cuda.manual_seed_all(myseed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def compute_metrics(y, yhat, score1):
    y = y.detach().cpu().numpy()
    yhat = yhat.detach().cpu().numpy()
    score1 = score1.detach().cpu().numpy()

    # macro weighs each class equally
    # micro weights classes based on class prior
    # micro=macro if classes balanced
    # in binary classification; micro=macro
    auc = roc_auc_score(y, score1)
    acc = accuracy_score(y, yhat)
    r1 = recall_score(y, yhat)
    p1 = precision_score(y, yhat)
    f1 = f1_score(y, yhat)
    cm = confusion_matrix(y, yhat)
    metrics = {
        "acc": acc,
        "f1": f1,
        "p1": p1,  # precision for class 1
        "r1": r1,  # recall for class 1
        "auc": auc,
        "cm": cm,
    }
    return metrics


class RumexDataset(Dataset):
    def __init__(self, data_dir):
        super(RumexDataset, self).__init__()
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        tfms = T.Compose(
            [T.Resize(224), T.ToTensor(), T.Normalize(imagenet_mean, imagenet_std),]
        )

        self.rumex = ImageFolder(data_dir, tfms)

    def __getitem__(self, idx):
        x, y = self.rumex[idx]
        fname = self.rumex.imgs[idx][0]
        fname = fname.split("/")[-1]
        return x, y, fname

    def __len__(self):
        return len(self.rumex)


def train_loader(ds, bs):
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=12)
    return dl


def test_loader(ds, bs):
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=12)
    return dl


class RumexNet(nn.Module):
    def __init__(self, model_name):
        super(RumexNet, self).__init__()
        num_classes = 2
        if model_name == "resnet":
            model = models.resnet50(pretrained=True)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif model_name == "vgg":
            model = models.vgg19_bn(pretrained=True)
            in_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(in_features, num_classes)
        elif model_name == "mobilenet":
            model = models.mobilenet_v2(pretrained=True)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
        elif model_name == "shufflenet":
            model = models.shufflenet_v2_x0_5(pretrained=True)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif model_name == "densenet":
            model = models.densenet121(pretrained=True)
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
        elif model_name == "mnasnet":
            model = models.mnasnet1_0(pretrained=True)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
        elif model_name == "efficientnet":
            model = EfficientNet.from_pretrained("efficientnet-b0")
            in_features = model._fc.in_features
            model._fc = nn.Linear(in_features, num_classes)
        self.model = model

    def forward(self, x):
        return self.model(x)


def test(model, test_loader, device):
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    metrics = None
    nval = len(test_loader.dataset)
    ##### Model Validation ##########
    y_val = []
    score_val = []
    yhat_val = []
    with torch.no_grad():
        model.eval()
        running_loss_val = 0
        for xb, yb, _ in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            # Forward pass
            scoreb = model(xb)  # logits
            _, yhatb = torch.max(scoreb, 1)
            lossb_val = loss_fn(scoreb, yb)

            # book keeping at batch level
            running_loss_val += torch.sum(lossb_val)

            y_val.append(yb)
            score_val.append(scoreb)
            yhat_val.append(yhatb)

        loss_val = running_loss_val / nval
        # loss_val.append(loss_val)

        # predictions and  metrics
        y_val = torch.cat(y_val)
        score_val = torch.cat(score_val)
        yhat_val = torch.cat(yhat_val)
        metrics = compute_metrics(y_val, yhat_val, score_val[:, 1])
    return loss_val, metrics


def train(model, optimizer, loss_fn, train_loader, device):
    ntrain = len(train_loader.dataset)
    model = model.to(device)
    running_loss_tr = 0.0
    model.train()
    for xb, yb, file_names in train_loader:
        # Forward pass
        print(file_names)
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        scoreb = model(xb)
        # loss of each elem in batch
        lossb = loss_fn(scoreb, yb)
        lossmean_b = torch.mean(lossb)

        # Backward and optimize
        lossmean_b.backward(retain_graph=True)
        optimizer.step()
        running_loss_tr += torch.sum(lossb)
        loss_tr = running_loss_tr / ntrain
    return model, optimizer, loss_tr

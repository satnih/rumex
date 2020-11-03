# This code is based on code from Stanford cs230 class
# https://github.com/cs230-stanford/cs230-code-examples
import os
import yaml
import torch
import logging
import numpy as np
import torch.nn as nn
from torchvision import models
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score, roc_auc_score


def load_config(config_file):
    # credit: https://tinyurl.com/y5tyys6a
    with open(config_file) as file:
        config = yaml.safe_load(file)
    return config


def save_ckpt(ckpt_dict, ckpt_dir):
    """Save  checkpoint and predictions 
    Args:
        ckpt_dict: (dict) contains model's state_dict, 
        ckpt_dir: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(ckpt_dir, 'best.pt')
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    torch.save(ckpt_dict, filepath)


def load_ckpt(ckpt_dir, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is 
    provided, loads state_dict of optimizer assuming it is present in ckpt_dir.

    Args:
        ckpt_dir: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from ckpt_dir
    """
    if not os.path.exists(ckpt_dir):
        raise("File doesn't exist {}".format(ckpt_dir))
    ckpt_dir = torch.load(ckpt_dir)
    model.load_state_dict(ckpt_dir['state_dict'])

    if optimizer:
        optimizer.load_state_dict(ckpt_dir['optim_dict'])

    return ckpt_dir


def set_logger(log_dir):
    """Set the logger to log info in terminal and file `log_path`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """

    # set log dirctory
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "logs.out"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
    return log_dir


def compute_metrics(y, yhat, score):
    tp = torch.sum((y == 1) & (yhat == 1)).float()
    tn = torch.sum((y == 0) & (yhat == 0)).float()
    fp = torch.sum((y == 1) & (yhat == 0)).float()
    fn = torch.sum((y == 0) & (yhat == 1)).float()
    acc = (tp+tn)/len(y)
    recall = tp/(tp + fn)  # predicted pos/condition pos
    precision = tp/(tp + fp)  # predicted neg/condition neg
    f1 = 2*precision*recall/(precision+recall)
    auc = roc_auc_score(y.cpu().numpy(), score.cpu().numpy())
    metrics = {
        'acc': acc,
        'f1': f1,
        'pre': precision,
        'recall': recall,
        'auc': auc
    }
    return metrics


def train(model, dl, optimizer, loss_fn, device):
    model.to(device)
    model.train()
    running_loss = 0.0
    for xb, yb, _, _ in dl:
        # Forward pass
        xb = xb.to(device)
        yb = yb.to(device)
        scoreb = model(xb)
        # loss of each elem in batch
        lossb = loss_fn(scoreb, yb)
        lossb_mean = torch.mean(lossb)

        # Backward and optimize
        lossb_mean.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += torch.sum(lossb)

    loss = running_loss / len(dl.dataset)
    return loss


def validate(model, dl, loss_fn, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        fname = []
        score = []
        loss = []
        y = []
        yhat = []
        for xb, yb, _, fnameb in dl:
            xb = xb.to(device)
            yb = yb.to(device)

            # Forward pass
            scoreb = model(xb)  # logits
            _, yhatb = torch.max(scoreb, 1)
            lossb = loss_fn(scoreb, yb)

            # book keeping at batch level
            score.append(scoreb)
            yhat.append(yhatb)
            loss.append(lossb)
            fname.append(fnameb)
            y.append(yb)

        score = torch.cat(score)
        loss = torch.cat(loss)
        y = torch.cat(y)
        yhat = torch.cat(yhat)

        # flatten list of tuples
        fname = [y for x in fname for y in x]

        # predictions and metrics
        # _, yhat = torch.max(score, 1)
        loss_mean = torch.mean(loss)

        metrics = compute_metrics(y, yhat, score[:, 1])
        metrics['loss'] = loss_mean

        predictions = {
            # 'idx': idx,
            'fname': fname,
            'y': y,
            'yhat': yhat,
            'score': score[:, 1],
            'loss': loss}

    return predictions, metrics


class RumexDataset(Dataset):
    def __init__(self, data_dir, train_flag):
        super(RumexDataset, self).__init__()
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        self.train_flag = train_flag
        sz = 224
        if train_flag:
            tfms = T.Compose([
                T.Resize(sz),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                # T.ColorJitter(),
                T.ToTensor(),
                T.Normalize(imagenet_mean, imagenet_std)
            ])
        else:
            tfms = T.Compose([
                T.Resize(sz),
                T.ToTensor(),
                T.Normalize(imagenet_mean, imagenet_std)
            ])

        self.rumex = ImageFolder(data_dir, tfms)
        class_counts = np.bincount(self.rumex.targets)
        class_weights = np.round((1. / class_counts) * class_counts[0])
        self.sample_weights = class_weights[self.rumex.targets]

    def __getitem__(self, idx):
        x, y = self.rumex[idx]
        fname = self.rumex.imgs[idx][0]
        fname = fname.split('/')[-1]
        return x, y, idx, fname

    def __len__(self):
        return len(self.rumex)


def train_loader(ds, bs):
    # sampler = WeightedRandomSampler(weights=ds.sample_weights,
    #                                 num_samples=len(ds),
    #                                 replacement=True)
    # dl = DataLoader(ds, batch_size=bs, sampler=sampler, num_workers=12)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=12)
    return dl


def test_loader(ds, bs):
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0)
    return dl


class RumexNet(nn.Module):
    def __init__(self, model_name):
        super(RumexNet, self).__init__()
        num_classes = 2
        if model_name == 'alexnet':
            model = models.alexnet(pretrained=True)
            model.classifier[6] = nn.Linear(4096, num_classes)
        elif model_name == 'resnet':
            model = models.resnet50(pretrained=True)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif model_name == 'inception':
            model = models.inception_v3(pretrained=True)
            in_features = model.fc.in_features
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'mobilenet':
            model = models.mobilenet_v2(pretrained=True)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
        elif model_name == 'shufflenet':
            model = models.shufflenet_v2_x0_5(pretrained=True)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif model_name == 'densenet':
            model = models.densenet121(pretrained=True)
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
        elif model_name == 'mnasnet':
            model = models.mnasnet1_0(pretrained=True)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)

        self.model = model

    def forward(self, x):
        return self.model(x)

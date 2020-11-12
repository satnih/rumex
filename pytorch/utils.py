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
    metrics = {
        'acc': acc,
        'f1': f1,
        'p1': p1,  # precision for class 1
        'r1': r1,  # recall for class 1
        'auc': auc
    }
    return metrics

# TODO: move dataset and model these into seperate files


class RumexDataset(Dataset):
    def __init__(self, data_dir):
        super(RumexDataset, self).__init__()
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        sz = 224

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
        return x, y, fname

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
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=12)
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
        elif model_name == 'wide_resnet':
            model = models.wide_resnet50_2(pretrained=True)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif model_name == 'resnext':
            model = models.resnext50_32x4d(pretrained=True)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
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

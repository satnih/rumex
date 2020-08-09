# %%
# credit:
# https://tinyurl.com/y7p7dmpt
# https://tinyurl.com/yd4o3yf4
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import os
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import models
from torchvision import transforms as T
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import stat_scores, f1_score, accuracy
from pytorch_lightning.metrics.functional import auroc
from torchvision.datasets import ImageFolder, DatasetFolder
from pytorch_lightning.callbacks import ModelCheckpoint

# from torch.utils.tensorboard import SummaryWriter
# plt.ion()   # interactive mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
model_name = 'mobilenet_v2'
data_path_tr = '/u/21/hiremas1/unix/postdoc/rumex/data_alexnet/10m/train/'
data_path_val = '/u/21/hiremas1/unix/postdoc/rumex/data_alexnet/10m/valid/'
data_path_te = '/u/21/hiremas1/unix/postdoc/rumex/data_alexnet/test/10m/'
max_epochs = 20
batch_size = 32
test_flag = 1


def load_pretrained(model_name, num_classes):
    if model_name == 'alexnet':
        model = models.alexnet(pretrained=True)
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == 'inception_v3':
        model = models.inception_v3(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == 'shufflenet_v2':
        model = models.shufflenet_v2_x0_5(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == 'densenet201':
        model = models.densenet201(pretrained=True)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif model_name == 'mnasnet0_5':
        model = models.mnasnet0_5(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


class MyModel(pl.LightningModule):
    def __init__(self):
        super(MyModel, self).__init__()

        num_classes = 2

        # mobilenet
        model = models.mobilenet_v2(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

        # # alexnet
        # model = models.alexnet(pretrained=True)
        # model.classifier[6] = nn.Linear(4096, num_classes)

        # # resnet18
        # model = models.resnet18(pretrained=True)
        # in_features = model.fc.in_features
        # model.fc = nn.Linear(in_features, num_classes)

        self.model = model

    def forward(self, x):
        return self.model(x)

    def train_dataloader(self):
        # REQUIRED
        transforms = T.Compose([
            T.Resize(224),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor(),
            T.Normalize(imagenet_mean, imagenet_std)
        ])
        trainset = ImageFolder(data_path_tr, transforms)

        train_loader = DataLoader(trainset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=12)
        return train_loader

    def val_dataloader(self):
        # OPTIONAL
        transforms = T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(imagenet_mean, imagenet_std)
        ])
        valset = ImageFolder(data_path_val, transforms)
        val_loader = DataLoader(valset,
                                batch_size=len(valset),
                                shuffle=False,
                                num_workers=12)
        return val_loader

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        yhat = self(x)
        loss = F.cross_entropy(yhat, y)

        # using TrainResult to enable logging
        result = pl.TrainResult(loss)
        result.log('train_loss', loss, on_epoch=True)

        return result

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        yhat = self(x)
        loss = F.cross_entropy(yhat, y)
        auc = auroc(yhat, y, pos_label=0)
        acc = accuracy(yhat, y)
        # cm = torch.Tensor(stat_scores(yhat, y, class_index=1))

        result = pl.EvalResult(early_stop_on=loss, checkpoint_on=loss)

        result.log('val_loss', loss, prog_bar=True)
        result.log('val_auc', auc, prog_bar=True)
        result.log('val_acc', acc, prog_bar=True)
        return result

    def test_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = F.cross_entropy(yhat, y)
        auc = auroc(yhat, y, pos_label=0)
        acc = accuracy(yhat, y)
        result = pl.EvalResult()
        result.log('test_loss', loss)
        result.log('test_auc', auc)
        result.log('test_acc', acc)
        return result

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.parameters(), lr=1e-3)


model = MyModel()
# most basic trainer, uses good defaults (1 gpu)
trainer = pl.Trainer(max_epochs=max_epochs,
                     gpus=1,
                     checkpoint_callback=ModelCheckpoint())

# automatic lr finder
# lr_finder = trainer.lr_find(model)
# lr_finder.results
# # Plot with
# fig = lr_finder.plot(suggest=True)
# fig.show()
# new_lr = lr_finder.suggestion()
# model.hparams.lr = new_lr
# %% Train
trainer.fit(model)

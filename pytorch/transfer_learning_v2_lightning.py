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

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import models
from torchvision import transforms as T
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import stat_scores, f1_score, accuracy
from pytorch_lightning.metrics.functional import auroc
from torchvision.datasets import ImageFolder, DatasetFolder

# from torch.utils.tensorboard import SummaryWriter
# plt.ion()   # interactive mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
model_name = 'mnasnet0_5'
data_path_tr = '/u/21/hiremas1/unix/postdoc/rumex/data_alexnet/train/'
data_path_val = '/u/21/hiremas1/unix/postdoc/rumex/data_alexnet/val/'
data_path_te = '/u/21/hiremas1/unix/postdoc/rumex/data_alexnet/test/'
max_epochs = 15
batch_size = 32


def load_pretrained(model_name, num_classes):
    if model_name == 'alexnet':
        model = models.alexnet(pretrained=True)
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name == 'resnet152':
        model = models.resnet152(pretrained=True)
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


# %%


class MyModel(pl.LightningModule):
    def __init__(self, model_name):
        super(MyModel, self).__init__()
        self.model_name = model_name
        self.model = load_pretrained(self.model_name, 2)

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


model = MyModel(model_name)
# most basic trainer, uses good defaults (1 gpu)
# trainer = pl.Trainer(gpus=1, max_epochs=5)
trainer = pl.Trainer(max_epochs=max_epochs, gpus=1)

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

# %% Test
transforms = T.Compose(
    [T.Resize(224),
     T.ToTensor(),
     T.Normalize(imagenet_mean, imagenet_std)])
testset = ImageFolder(data_path_te, transforms)
test_loader = DataLoader(testset, shuffle=True, batch_size=batch_size)
result = trainer.test(test_dataloaders=test_loader)
print(result)

# %%

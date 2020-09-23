# %%
# credit:
# https://tinyurl.com/y7p7dmpt
# https://tinyurl.com/yd4o3yf4
import os
import torch
import torchvision
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.metrics.functional import f1_score, accuracy, auroc
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.progress import ProgressBar
import sys
seed_everything(42)

# from torch.utils.tensorboard import SummaryWriter
# plt.ion()   # interactive mode
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
data_dir = "/u/21/hiremas1/unix/postdoc/rumex/data_for_fastai_cleaned/"


class Net(pl.LightningModule):
    def __init__(self, model_name, min_lr, max_lr, bs=32):
        super(Net, self).__init__()
        num_classes = 2
        if model_name == 'alexnet':
            self.model = models.alexnet(pretrained=True)
            self.model.classifier[6] = nn.Linear(4096, num_classes)
        elif model_name == 'resnet':
            self.model = models.resnet18(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif model_name == 'inception':
            self.model = self.models.inception_v3(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif model_name == 'mobilenet':
            self.model = models.mobilenet_v2(pretrained=True)
            self.model.classifier[1] = nn.Linear(
                self.model.classifier[1].in_features, 2)
        elif model_name == 'shufflenet':
            self.model = models.shufflenet_v2_x0_5(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif model_name == 'mnasnet':
            self.model = models.mnasnet0_5(pretrained=True)
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features,
                                                 num_classes)

        self.bs = bs
        self.min_lr = min_lr
        self.max_lr = max_lr

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
        trainset = ImageFolder(data_dir+"train/", transforms)

        class_weights = np.array([1, 3])
        sample_weights = class_weights[trainset.targets]
        sampler = WeightedRandomSampler(weights=sample_weights,
                                        num_samples=len(trainset),
                                        replacement=True)

        train_loader = DataLoader(
            trainset,
            # sampler=sampler,
            batch_size=self.bs,
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
        valset = ImageFolder(data_dir+"valid/", transforms)
        val_loader = DataLoader(valset,
                                batch_size=100,
                                shuffle=True,
                                num_workers=12)
        return val_loader

    # def test_dataloader(self):
    #     # OPTIONAL
    #     transforms = T.Compose([
    #         T.Resize(224),
    #         T.ToTensor(),
    #         T.Normalize(imagenet_mean, imagenet_std)
    #     ])
    #     testset = ImageFolder(data_dir_te, transforms)
    #     test_loader = DataLoader(testset,
    #                              batch_size=32,
    #                              shuffle=True,
    #                              num_workers=12)
    #     return test_loader

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
        yhat_prob = torch.exp(torch.log_softmax(yhat, 1))
        loss = F.cross_entropy(yhat, y)
        # auc = auroc(yhat_prob, y, pos_label=1)
        acc = accuracy(yhat, y)
        # f1 = f1_score(yhat_prob, y, num_classes=2)
        # cm = torch.Tensor(stat_scores(yhat, y, class_index=1))
        result = pl.EvalResult(early_stop_on=loss, checkpoint_on=loss)
        result.log('val_loss', loss, prog_bar=True)
        result.log('val_acc', acc, prog_bar=True)
        # result.log('val_f1', f1, prog_bar=True)
        # result.log('val_auc', auc, prog_bar=True)

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
        # optimizer = torch.optim.SGD(self.parameters(),
        #                             lr=5e-3,
        #                             weight_decay=1e-2,
        #                             momentum=.9)

        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=1e-3,
                                      weight_decay=1e-2)
        scheduler = CyclicLR(optimizer,
                             step_size_up=250,
                             base_lr=self.min_lr,
                             max_lr=self.max_lr,
                             cycle_momentum=False)
        return [optimizer], [scheduler]


def lightning_trainer(args):
    model = Net(args.model_name, args.min_lr, args.max_lr)
    log_dir = args.model_name + "_adamw_logs"
    logger = TensorBoardLogger(save_dir=os.getcwd(), name=log_dir)
    trainer = pl.Trainer(max_epochs=args.num_epoch,
                         gpus=1,
                         logger=logger,
                         deterministic=True,
                         checkpoint_callback=ModelCheckpoint())
    trainer.fit(model)

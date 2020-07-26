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
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# from torch.utils.tensorboard import SummaryWriter
# plt.ion()   # interactive mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
model_name = 'alexnet'


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
        in_features = model.classifier[1].in_features
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
        transforms = T.Compose([T.Resize(224),
                                T.RandomHorizontalFlip(),
                                T.RandomVerticalFlip(),
                                T.ToTensor(),
                                T.Normalize(imagenet_mean, imagenet_std)])
        trainset = datasets.ImageFolder(
            '/u/21/hiremas1/unix/postdoc/rumex/data_alexnet/train/', transforms)

        # num_classes = len(trainset.classes)
        # classcount = trainratio.tolist()
        # class_weights = 1./torch.tensor(classcount, dtype=torch.float)
        # train_weights = class_weights[trainset.labels]
        # train_sampler = WeightedRandomSampler(weights=train_weights,
        #                                       num_samples=len(train_weights))

        train_dl = DataLoader(trainset,
                              #   sampler=train_sampler,
                              batch_size=32,
                              shuffle=True,
                              num_workers=12)
        return train_dl

    def val_dataloader(self):
        # OPTIONAL
        transforms = T.Compose([T.Resize(224),
                                T.ToTensor(),
                                T.Normalize(imagenet_mean, imagenet_std)])
        val_ds = datasets.ImageFolder(
            '/u/21/hiremas1/unix/postdoc/rumex/data_alexnet/val/', transforms)
        val_dl = DataLoader(val_ds, batch_size=32, shuffle=True, num_workers=1)
        return val_dl

    # def test_dataloader(self):
    #     # OPTIONAL
    #     return DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor()), batch_size=32)

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        yhat = self(x)
        loss = F.cross_entropy(yhat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        yhat = self(x)
        val_loss = F.cross_entropy(yhat, y)
        val_auc = auroc(yhat, y, pos_label=0)
        val_cm = torch.Tensor(stat_scores(yhat, y, class_index=1))
        batch_results = {'val_loss': val_loss,
                         'val_cm': val_cm,
                         'val_auc': val_auc}
        return batch_results

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        val_loss_mean = torch.stack([output['val_loss']
                                     for output in outputs]).mean()

        val_auc_mean = torch.stack([output['val_auc']
                                    for output in outputs]).mean()

        val_cm_total = torch.stack([output['val_cm']
                                    for output in outputs]).sum(axis=0)
        tp = val_cm_total[0]
        fp = val_cm_total[1]
        tn = val_cm_total[2]
        fn = val_cm_total[3]
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        val_f1 = 2*precision*recall/(precision + recall)
        tensorboard_logs = {'val_loss': val_loss_mean.item(),
                            'val_f1': val_f1,
                            'val_auc': val_auc_mean}

        results = {'progress_bar': tensorboard_logs,
                   'log': tensorboard_logs}
        return results

    # def test_step(self, batch, batch_nb):
    #     # OPTIONAL
    #     x, y = batch
    #     yhat = self(x)
    #     return {'test_loss': F.cross_entropy(yhat, y)}

    # def test_epoch_end(self, outputs):
    #     # OPTIONAL
    #     avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
    #     logs = {'test_loss': avg_loss}
    #     return {'test_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.SGD(self.parameters(), lr=1e-3)


model = MyModel(model_name)
# most basic trainer, uses good defaults (1 gpu)
# trainer = pl.Trainer(gpus=1, max_epochs=5)
trainer = pl.Trainer(max_epochs=5)

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
# trainer.test()

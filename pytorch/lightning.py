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
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics.functional import f1_score, accuracy, auroc
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import ProgressBar
from tqdm.auto import tqdm
import sys
seed_everything(42)

# from torch.utils.tensorboard import SummaryWriter
# plt.ion()   # interactive mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


class MyProgressBar(ProgressBar):
    def __init__(self,):
        super().__init__()

    def init_validation_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for validation. """
        bar = tqdm(
            desc='Validating',
            position=(2 * self.process_position + 1),
            disable=True,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout
        )
        return bar


class Net(pl.LightningModule):

    def __init__(self, model_name, batch_size=50):
        super(Net, self).__init__()
        num_classes = 2
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

        self.model = model
        self.bs = batch_size
        self.data_path_tr = '/u/21/hiremas1/unix/postdoc/rumex/data/WENR_ortho_Rumex_10m_2_sw/'
        self.data_path_val = '/u/21/hiremas1/unix/postdoc/rumex/data/WENR_ortho_Rumex_10m_3_ne/'
        self.data_path_te = '/u/21/hiremas1/unix/postdoc/rumex/data/WENR_ortho_Rumex_10m_4_se/'

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
        trainset = ImageFolder(self.data_path_tr, transforms)
        class_weights = np.array([1, 3])
        sample_weights = class_weights[trainset.targets]
        sampler = WeightedRandomSampler(weights=sample_weights,
                                        num_samples=len(trainset),
                                        replacement=True)

        train_loader = DataLoader(trainset,
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
        valset = ImageFolder(self.data_path_val, transforms)
        val_loader = DataLoader(valset,
                                batch_size=len(valset),
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
        yhat_prob = torch.exp(torch.log_softmax(yhat, 1))
        loss = F.cross_entropy(yhat, y)
        auc = auroc(yhat_prob, y, pos_label=1)
        acc = accuracy(yhat, y)
        f1 = f1_score(yhat_prob, y, num_classes=2)
        # cm = torch.Tensor(stat_scores(yhat, y, class_index=1))
        result = pl.EvalResult(early_stop_on=loss, checkpoint_on=loss)
        result.log('val_loss', loss, prog_bar=True)
        result.log('val_acc', acc, prog_bar=True)
        result.log('val_f1', f1, prog_bar=True)
        result.log('val_auc', auc, prog_bar=True)

        return result

    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     yhat = self(x)
    #     loss = F.cross_entropy(yhat, y)
    #     auc = auroc(yhat, y, pos_label=0)
    #     acc = accuracy(yhat, y)
    #     result = pl.EvalResult()
    #     result.log('test_loss', loss)
    #     result.log('test_auc', auc)
    #     result.log('test_acc', acc)
    #     return result

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.RMSprop(self.parameters(), lr=1e-3)


# %%
def start_training(model_name):
    max_epochs = 100
    model = Net(model_name)
    logger = TensorBoardLogger(save_dir=os.getcwd(),
                               name=model_name+'_logs')

    bar = MyProgressBar()
    trainer = pl.Trainer(max_epochs=max_epochs,
                         #  gpus=0,
                         logger=logger,
                         deterministic=True,
                         callbacks=[bar],
                         checkpoint_callback=ModelCheckpoint())

    # # Run learning rate finder
    # lr_finder = trainer.lr_find(model)
    # fig = lr_finder.plot(suggest=True)
    # fig.save(model_name+'.png')
    # model.hparams.lr = lr_finder.suggestion()
    trainer.fit(model)

# %% test model


# def test_dataloader():
#     # OPTIONAL
#     transforms = T.Compose([
#         T.Resize(224),
#         T.ToTensor(),
#         T.Normalize(imagenet_mean, imagenet_std)
#     ])
#     testset = ImageFolder(data_path_te, transforms)
#     test_loader = DataLoader(testset, batch_size=len(testset))
#     return test_loader


# test_loader = test_dataloader()


# # best_model_path = 'lightning_logs/version_1/checkpoints/epoch=5.ckpt'
# # best_model = MyModel.load_from_checkpoint(best_model_path)
# for x, y in test_loader:
#     logits = model(x)
#     _, pred = torch.max(logits, 1)

# y = y.detach().numpy()
# pred = pred.detach().numpy()

# print(f'{np.sum(y == pred)/len(y)}')

# %%
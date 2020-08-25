import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler


class RumexDataset(Dataset):
    def __init__(self, data_dir, tfms=None):
        super(RumexDataset, self).__init__()
        self.data_dir = data_dir
        self.rumex = ImageFolder(data_dir, tfms)
        class_counts = np.bincount(self.rumex.targets)
        class_weights = np.round((1. / class_counts) * class_counts[0])
        self.sample_weights = class_weights[self.rumex.targets]

    def __getitem__(self, idx):
        x, y = self.rumex[idx]
        return x, y, idx

    def __len__(self):
        return len(self.rumex)

    def make_data_loader(self, bs=32, num_workers=12):
        # sampler = WeightedRandomSampler(weights=self.sample_weights,
        #                                 num_samples=self.__len__(),
        #                                 replacement=True)
        # dl = DataLoader(self,
        #                 batch_size=bs,
        #                 sampler=sampler,
        #                 shuffle=False,
        #                 drop_last=True)

        dl = DataLoader(self,
                        batch_size=bs,
                        shuffle=True,
                        num_workers=num_workers)
        return (dl)


def make_dataloader(data, bs):
    dl = DataLoader(data,
                    batch_size=bs,
                    shuffle=True)
    return dl


def get_data_loaders(data_dir, bs):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    tfms_tr = T.Compose([
        T.Resize(224),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor(),
        T.Normalize(imagenet_mean, imagenet_std)
    ])

    tfms_te = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(imagenet_mean, imagenet_std)
    ])

    trainset = ImageFolder(data_dir+'train/', tfms_tr)
    valset = ImageFolder(data_dir+'valid/', tfms_te)

    dl_tr = make_dataloader(trainset, bs)
    dl_val = make_dataloader(valset, bs)

    return dl_tr, dl_val

    tfms_tr = T.Compose([
        T.Resize(224),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor(),
        T.Normalize(imagenet_mean, imagenet_std)
    ])

    tfms_te = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(imagenet_mean, imagenet_std)
    ])

    trainset = ImageFolder(data_dir+'train/', tfms_tr)
    valset = ImageFolder(data_dir+'valid/', tfms_te)

    dl_tr = make_dataloader(trainset, bs)
    dl_val = make_dataloader(valset, bs)

    return dl_tr, dl_val

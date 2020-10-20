import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler


class RumexDataset(Dataset):
    def __init__(self, data_dir, train_flag):
        super(RumexDataset, self).__init__()
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        self.train_flag = train_flag
        if train_flag:
            tfms = T.Compose([
                T.Resize(128),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                # T.ColorJitter(),
                T.ToTensor(),
                T.Normalize(imagenet_mean, imagenet_std)
            ])
        else:
            tfms = T.Compose([
                T.Resize(128),
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
    sampler = WeightedRandomSampler(weights=ds.sample_weights,
                                    num_samples=len(ds),
                                    replacement=True)
    dl = DataLoader(ds, batch_size=bs, sampler=sampler, num_workers=12)
    return dl


def test_loader(ds, bs):
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=12)
    return dl

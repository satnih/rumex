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
                T.Resize(224),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ToTensor(),
                T.Normalize(imagenet_mean, imagenet_std)
            ])
        else:
            tfms = T.Compose([
                T.Resize(224),
                T.ToTensor(),
                T.Normalize(imagenet_mean, imagenet_std)
            ])

        self.rumex = ImageFolder(data_dir, tfms)
        class_counts = np.bincount(self.rumex.targets)
        class_weights = np.round((1. / class_counts) * class_counts[0])
        self.sample_weights = class_weights[self.rumex.targets]

    def __getitem__(self, idx):
        x, y = self.rumex[idx]
        return x, y, idx

    def __len__(self):
        return len(self.rumex)

    def make_data_loader(self, bs, num_workers=1):
        if self.train_flag:
            sampler = WeightedRandomSampler(weights=self.sample_weights,
                                            num_samples=self.__len__(),
                                            replacement=True)
            dl = DataLoader(self,
                            batch_size=bs,
                            sampler=sampler,
                            drop_last=True)
        else:
            dl = DataLoader(self,
                            batch_size=self.__len__(),
                            shuffle=False,
                            num_workers=num_workers)
        return (dl)

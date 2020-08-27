
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class RumexNet(nn.Module):
    def __init__(self, model_name):
        super(RumexNet, self).__init__()
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

    def forward(self, x):
        return self.model(x)

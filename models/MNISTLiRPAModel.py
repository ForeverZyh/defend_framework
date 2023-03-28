from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
import torch
from torch.nn import CrossEntropyLoss

from models.LiRPAModel import LiRPAModel


class MNISTModel(LiRPAModel):
    def __init__(self, n_features, n_classes, args, device, lr=1e-3):
        super(MNISTModel, self).__init__(n_features, n_classes, args, device, eval(args.model), lr)

    def data_aug(self, data, **kwargs):
        data = transforms.RandomCrop(28, 3)(data)
        data = transforms.RandomRotation(10)(data)
        return data

    def fit(self, X, y, batch_size, epochs, dummy=None):
        X = np.transpose(X, (0, 3, 1, 2))

        if not self.args.SABR:
            aug_func = self.data_aug
        else:
            aug_func = self.adv_attack

        super(MNISTModel, self).fit(X, y, batch_size, epochs, aug_func)

    def evaluate(self, x_test, y_test):
        x_test = np.transpose(x_test, (0, 3, 1, 2))
        return super(MNISTModel, self).evaluate(x_test, y_test)


class mlp_3layer(nn.Module):
    def __init__(self, in_ch, in_dim, width=1):
        super(mlp_3layer, self).__init__()
        self.fc1 = nn.Linear(in_ch * in_dim * in_dim, 256 * width)
        self.fc2 = nn.Linear(256 * width, 128 * width)
        self.fc3 = nn.Linear(128 * width, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class cnn_4layer(nn.Module):
    def __init__(self, in_ch, in_dim, width=2, linear_size=256):
        super(cnn_4layer, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 4 * width, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(4 * width, 8 * width, 4, stride=2, padding=1)
        self.fc1 = nn.Linear(8 * width * (in_dim // 4) * (in_dim // 4), linear_size)
        self.fc2 = nn.Linear(linear_size, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

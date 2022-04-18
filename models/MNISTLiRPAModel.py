from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.LiRPAModel import LiRPAModel


class MNISTModel(LiRPAModel):
    def __init__(self, n_features, n_classes, args, device, lr=1e-3):
        super(MNISTModel, self).__init__(n_features, n_classes, args, device, mlp_3layer, lr)

    def fit(self, X, y, batch_size, epochs, dummy=None):
        X = np.transpose(X, (0, 3, 1, 2))

        def data_aug(data):
            data = transforms.RandomCrop(28, 3)(data)
            data = transforms.RandomRotation(10)(data)
            return data

        super(MNISTModel, self).fit(X, y, batch_size, epochs, data_aug)

    def evaluate(self, x_test, y_test):
        x_test = np.transpose(x_test, (0, 3, 1, 2))
        super(MNISTModel, self).evaluate(x_test, y_test)


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

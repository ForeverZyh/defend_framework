import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.LiRPAModel import LiRPAModel


class EmberModel(LiRPAModel):
    def __init__(self, n_features, n_classes, args, device, lr=1e-3):
        super(EmberModel, self).__init__([1, 1, n_features], n_classes, args, device, mlp_4layer, lr)

    def data_aug(self, data, **kwargs):
        data += torch.clamp(torch.normal(torch.zeros_like(data), torch.ones_like(data) * 0.1), min=0, max=1)
        return data

    def fit(self, X, y, batch_size, epochs, dummy=None):
        X = np.expand_dims(X, axis=[1, 2])

        if not self.args.SABR:
            aug_func = self.data_aug
        else:
            aug_func = self.adv_attack_l0_one_pixel

        super(EmberModel, self).fit(X, y, batch_size, epochs, aug_func)

    def evaluate(self, x_test, y_test):
        x_test = np.expand_dims(x_test, axis=[1, 2])
        return super(EmberModel, self).evaluate(x_test, y_test)


class mlp_4layer(nn.Module):
    def __init__(self, in_ch, in_dim):
        super(mlp_4layer, self).__init__()
        self.fc1 = nn.Linear(in_ch * in_dim, 2000)
        self.fc2 = nn.Linear(2000, 1000)
        self.fc3 = nn.Linear(1000, 100)
        self.fc4 = nn.Linear(100, 2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

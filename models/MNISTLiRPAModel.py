import torch
import torch.nn as nn
import torch.nn.functional as F

from models.LiRPAModel import LiRPAModel


class MNISTModel(LiRPAModel):
    def __init__(self, n_features, n_classes, args, device, lr=1e-3):
        super(MNISTModel, self).__init__(n_features, n_classes, args, device, mlp_3layer, lr)


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

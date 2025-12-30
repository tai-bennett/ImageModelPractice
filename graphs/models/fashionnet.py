"""
FashionNet Model
name: fashionnet.py
date: Dec 2025
"""
import pdb
import torch
import torch.nn as nn
import json
# from easydict import EasyDict as edict
import numpy as np
import torch.nn.functional as F


class FashionNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(3, 5, 5)
        self.fc1 = nn.Linear(80, 40)
        self.fc2 = nn.Linear(40, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)      # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

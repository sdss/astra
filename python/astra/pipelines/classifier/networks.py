from time import time

import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch import nn
from torch.autograd import Variable


class OpticalCNN(nn.Module):

    class_names = ["cv", "fgkm", "oba", "wd", "sb2", "yso"]

    def __init__(self, in_channels=1, nb_channels=3, nb_classes=None):
        super(OpticalCNN, self).__init__()
        self.nb_channels = nb_channels
        nb_classes = nb_classes or len(self.class_names)
        self.model = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=nb_channels,
                kernel_size=3,
                stride=2,
            ),
            nn.BatchNorm1d(nb_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(
                in_channels=nb_channels,
                out_channels=nb_channels,
                kernel_size=3,
                stride=2,
            ),
            nn.BatchNorm1d(nb_channels),
            nn.ReLU(),
            nn.Dropout(0.1),  # 1024 950
            nn.Conv1d(
                in_channels=nb_channels,
                out_channels=nb_channels * 2,
                kernel_size=3,
                stride=2,
            ),
            nn.BatchNorm1d(nb_channels * 2),
            nn.ReLU(),
            nn.Dropout(0.1),  # 512 475
            nn.Conv1d(
                in_channels=nb_channels * 2,
                out_channels=nb_channels * 4,
                kernel_size=3,
                stride=2,
            ),
            nn.BatchNorm1d(nb_channels * 4),
            nn.ReLU(),
            nn.Dropout(0.1),  # 256 237?
            nn.Conv1d(
                in_channels=nb_channels * 4,
                out_channels=nb_channels * 8,
                kernel_size=3,
                stride=2,
            ),
            nn.BatchNorm1d(nb_channels * 8),
            nn.ReLU(),
            nn.Dropout(0.1),  # 128 118?
            nn.Conv1d(
                in_channels=nb_channels * 8,
                out_channels=nb_channels * 8,
                kernel_size=3,
                stride=2,
                padding=0,
            ),
            nn.BatchNorm1d(nb_channels * 8),
            nn.ReLU(),
            nn.Dropout(0.1),  # 64 58
        )

        self.fc1 = nn.Linear(nb_channels * 8 * 58, 58 * 4)
        self.fc2 = nn.Linear(58 * 4, 58)
        self.fc3 = nn.Linear(58, nb_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 58 * 8 * self.nb_channels)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  ## Activation function
        return x


class NIRCNN(nn.Module):

    class_names = ["fgkm", "oba", "sb2", "yso"]

    def __init__(self, nb_channels=3, nb_classes=4):
        super(NIRCNN, self).__init__()
        self.nb_channels = nb_channels
        self.nb_classes = nb_classes
        self.model = nn.Sequential(
            nn.Conv1d(
                in_channels=nb_channels,
                out_channels=nb_channels,
                kernel_size=3,
                stride=2,
            ),
            nn.BatchNorm1d(nb_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(
                in_channels=nb_channels,
                out_channels=nb_channels,
                kernel_size=3,
                stride=2,
            ),
            nn.BatchNorm1d(nb_channels),
            nn.ReLU(),
            nn.Dropout(0.1),  # 1024
            nn.Conv1d(
                in_channels=nb_channels,
                out_channels=nb_channels * 2,
                kernel_size=3,
                stride=2,
            ),
            nn.BatchNorm1d(nb_channels * 2),
            nn.ReLU(),
            nn.Dropout(0.1),  # 512
            nn.Conv1d(
                in_channels=nb_channels * 2,
                out_channels=nb_channels * 4,
                kernel_size=3,
                stride=2,
            ),
            nn.BatchNorm1d(nb_channels * 4),
            nn.ReLU(),
            nn.Dropout(0.1),  # 256
            nn.Conv1d(
                in_channels=nb_channels * 4,
                out_channels=nb_channels * 8,
                kernel_size=3,
                stride=2,
            ),
            nn.BatchNorm1d(nb_channels * 8),
            nn.ReLU(),
            nn.Dropout(0.1),  # 128
            nn.Conv1d(
                in_channels=nb_channels * 8,
                out_channels=nb_channels * 8,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm1d(nb_channels * 8),
            nn.ReLU(),
            nn.Dropout(0.1),  # 64
        )

        self.fc1 = nn.Linear(nb_channels * 8 * 64, 64 * 4)
        self.fc2 = nn.Linear(64 * 4, 64)
        self.fc3 = nn.Linear(64, nb_classes)
        return None

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 64 * 8 * self.nb_channels)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

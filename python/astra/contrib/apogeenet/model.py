
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Net(nn.Module):

    """
    A convolutional neural network for estimating properties of young stellar objects.
    """

    def __init__(self, num_layers=1, num_targets=3, drop_p=0.0):
        super(Net, self).__init__()
        # 3 input channels, 6 output channels, convolution
        # kernel
        self.conv1 = nn.Conv1d(num_layers, 8, 3, padding=1)
        self.conv2 = nn.Conv1d(8, 8, 3, padding=1)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1)
        self.conv4 = nn.Conv1d(16, 16, 3, padding=1)
        self.conv5 = nn.Conv1d(16, 16, 3, padding=1)
        self.conv6 = nn.Conv1d(16, 16, 3, padding=1)
        self.conv7 = nn.Conv1d(16, 32, 3, padding=1)
        self.conv8 = nn.Conv1d(32, 32, 3, padding=1)
        self.conv9 = nn.Conv1d(32, 32, 3, padding=1)
        self.conv10 = nn.Conv1d(32, 32, 3, padding=1)
        self.conv11 = nn.Conv1d(32, 64, 3, padding=1)
        self.conv12 = nn.Conv1d(64, 64, 3, padding=1)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(64*133*1, 512)
        self.fc1_dropout = nn.Dropout(p=drop_p)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, num_targets)


    def forward(self, x):
        x = F.max_pool1d(F.relu(self.conv2(F.relu(self.conv1(x)))), 2)
        x = F.max_pool1d(F.relu(self.conv4(F.relu(self.conv3(x)))), 2)
        x = F.max_pool1d(F.relu(self.conv6(F.relu(self.conv5(x)))), 2)
        x = F.max_pool1d(F.relu(self.conv8(F.relu(self.conv7(x)))), 2)
        x = F.max_pool1d(F.relu(self.conv10(F.relu(self.conv9(x)))), 2)
        x = F.max_pool1d(F.relu(self.conv12(F.relu(self.conv11(x)))), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1_dropout(self.fc1(x)))
        x = F.relu(self.fc1_dropout(self.fc2(x)))
        x = self.fc3(x)
        return x


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



def predict(model, eval_inputs):
    """
    Predict stellar parameters (teff, logg, [Fe/H]) of young stellar objects, given a spectrum.

    :param model:
        The neural network to use.
    
    :param eval_inputs:
        The spectrum flux.
    """
    
    with torch.no_grad():
        eval_outputs = model.forward(eval_inputs)

    # Calculate mean values.
    # TODO: These should not be hard-coded in! They should be stored with the model.
    means = np.array([
        2.880541250669337,
        4716.915128138449,
        -0.22329606176144642
    ])
    sigmas = np.array([
        1.1648147820369943, # LOGG
        733.0099523547299,  # TEFF
        0.3004270650813916, # FE_H
    ])

    # Scale the outputs.
    outputs = eval_outputs * sigmas + means
    
    param_names = ("logg", "teff", "fe_h")
    result = dict(zip(param_names, torch.mean(outputs, 0).numpy()))
    result.update(zip(
        [f"u_{p}" for p in param_names],
        torch.std(outputs, 0).numpy()
    ))

    return result
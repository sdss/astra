
import torch
import torch.nn as nn
import torch.nn.functional as F

LOGG_STD = 1.1890668869018555
LOGG_MEAN = 2.925783395767212
LOGTEFF_STD = 0.0917675793170929
LOGTEFF_MEAN = 3.6705732345581055
FE_STD = 0.25668784976005554
FE_MEAN = -0.13088105618953705

class Model():

    class MetadataNet(nn.Module):
        """A simple feed-forward network for metadata."""

        def __init__(self):
            """Initializes Metadata Net as a 5-layer deep network."""

            super(Model.MetadataNet, self).__init__()
            self.l1 = nn.Linear(7, 8)
            self.l2 = nn.Linear(8, 16)
            self.l3 = nn.Linear(16, 32)
            self.l4 = nn.Linear(32, 32)
            self.l5 = nn.Linear(32, 64)

            self.activation = F.relu

        def forward(self, x):
            """Feeds some metadata through the network.
            
            Args:
                x: A minibatch of metadata.

            Returns:
                An encoding of the metadata to feed into APOGEE Net.
            """

            x = self.activation(self.l1(x))
            x = self.activation(self.l2(x))
            x = self.activation(self.l3(x))
            x = self.activation(self.l4(x))
            x = self.activation(self.l5(x))
            return x   


    class APOGEENet(nn.Module):
        """ This is the basic VGG network used in APOGEE_Net_I. Weights used are Model 24. """

        def __init__(self, num_layers: int = 1, num_targets: int = 3, drop_p: float = 0.0):
            super(Model.APOGEENet, self).__init__()
            # 3 input channels, 6 output channels,  convolution
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

            self.metadata = Model.MetadataNet()

            # an affine operation: y = Wx + b
            self.fc1 = nn.Linear(64*133*1 + 64, 512)
            self.fc1_dropout = nn.Dropout(p=drop_p)
            self.fc2 = nn.Linear(512, 512)
            self.fc3 = nn.Linear(512, num_targets)

        def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
            """Feeds data through the network.

            Args:
                x (Tensor): A spectra minibatch.
                m (Tensor): A metadata minibatch corresponding to x.

            Returns:
                A prediction from the network.
            """

            # Max pooling over a (2) window
            x = F.max_pool1d(F.relu(self.conv2(F.relu(self.conv1(x)))), 2)
            x = F.max_pool1d(F.relu(self.conv4(F.relu(self.conv3(x)))), 2)
            x = F.max_pool1d(F.relu(self.conv6(F.relu(self.conv5(x)))), 2)
            x = F.max_pool1d(F.relu(self.conv8(F.relu(self.conv7(x)))), 2)
            x = F.max_pool1d(F.relu(self.conv10(F.relu(self.conv9(x)))), 2)
            x = F.max_pool1d(F.relu(self.conv12(F.relu(self.conv11(x)))), 2)
            x = x.view(-1, self.num_flat_features(x))
            m = self.metadata.forward(m)
            x = torch.hstack((x, m))
            x = F.relu(self.fc1_dropout(self.fc1(x)))
            x = F.relu(self.fc1_dropout(self.fc2(x)))
            x = self.fc3(x)
            return x

        def num_flat_features(self, x: torch.Tensor) -> int:
            size = x.size()[1:]  # all dimensions except the batch dimension
            num_features = 1
            for s in size:
                num_features *= s
            return num_features


    def __init__(self, model_path, device):
        """TODO: Docstring this"""

        self.model = self.APOGEENet()
        #model_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "model.pt")
        self.model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        self.model.to(device)
        self.model.eval()

    def predict_spectra(self, spectra, metadata):
        """TODO: Docstring this"""

        pred = self.model.forward(spectra, metadata)
        pred[:, 0] = pred[:, 0] * LOGG_STD + LOGG_MEAN
        pred[:, 1] = pred[:, 1] * LOGTEFF_STD + LOGTEFF_MEAN
        pred[:, 2] = pred[:, 2] * FE_STD + FE_MEAN
        return pred

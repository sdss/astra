from astra.utils import expand_path
from functools import cache

import torch
import numpy as np
from tqdm import tqdm

torch.manual_seed(0)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Network:
    class ApogeeDistBCNN(torch.nn.Module):
        def __init__(self, device="cpu", dtype=torch.float32, **kwargs):
            super().__init__(**kwargs)
            self.factory_kwargs = {"device": device, "dtype": dtype}
            self.activation = "relu"
            self.filter_len = 8
            self.pool_length = 4
            self.dropout_rate = 0.3

            self.targetname = [
                "fakemag"
            ]
            self.input_mean = torch.nn.Parameter(torch.zeros(7514, dtype=torch.float64, device=self.factory_kwargs["device"]), requires_grad=False)
            self.labels_std = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64, device=self.factory_kwargs["device"]), requires_grad=False)

            self.conv1 = torch.nn.Conv1d(
                in_channels=1,
                out_channels=2,
                kernel_size=8,
                padding="same",
                **self.factory_kwargs
            )
            self.conv2 = torch.nn.Conv1d(
                in_channels=2,
                out_channels=4,
                kernel_size=8,
                padding="same",
                **self.factory_kwargs
            )

            self.dense_1 = torch.nn.Linear(7512, 192, **self.factory_kwargs)
            self.dense_2 = torch.nn.Linear(192, 64, **self.factory_kwargs)
            self.dense_output = torch.nn.Linear(64, 1, **self.factory_kwargs)
            self.dense_var = torch.nn.Linear(64, 1, **self.factory_kwargs)

        def forward(self, inputs):
            # Basically the same as ApogeeBCNN structure
            cnn_layer_1 = torch.nn.functional.dropout(
                torch.nn.functional.relu(self.conv1(inputs.permute(0, 2, 1))),
                self.dropout_rate,
            )
            cnn_layer_2 = torch.nn.functional.relu(self.conv2(cnn_layer_1))
            pool_flattened = torch.nn.functional.dropout(
                torch.flatten(
                    torch.nn.functional.max_pool1d(
                        cnn_layer_2, kernel_size=self.pool_length
                    ).permute(0, 2, 1),
                    start_dim=1,
                ),
                self.dropout_rate,
            )
            layer_3 = torch.nn.functional.dropout(
                torch.nn.functional.relu(self.dense_1(pool_flattened)), self.dropout_rate
            )
            layer_4 = torch.nn.functional.relu(self.dense_2(layer_3))
            output = torch.nn.functional.softplus(self.dense_output(layer_4))
            log_variance_output = self.dense_var(layer_4)

            return output, log_variance_output

        def predict(self, inputs, batchsize=256, mc_num=100):
            """
            Parameters
            ----------
            inputs: numpy.ndarray
                input spectra of shape (num_spectra, 7514)

            Returns
            -------
            pd.DataFrame
                predicted labels and uncertainties
            """
            input_mean = self.input_mean.cpu().numpy()
            labels_std = self.labels_std.cpu().numpy()

            data_length = inputs.shape[0]
            num_full_batch = data_length // batchsize
            num_data_remaining = data_length % batchsize

            outputs_holder = np.zeros((data_length, 1))
            outputs_std_holder = np.zeros((data_length, 1))
            outputs_model_std_holder = np.zeros((data_length, 1))
            with tqdm(unit=" stars", total=data_length, disable=num_full_batch<3) as pbar:
                pbar.set_description_str("Spectra Processed: ")
                with torch.inference_mode():
                    if num_full_batch > 0:
                        for i in range(num_full_batch):
                            batch_inputs = torch.tensor(inputs[i * batchsize : (i + 1) * batchsize] - input_mean, **self.factory_kwargs)[:, :, None]
                            temp_outputs = [self(batch_inputs) for i in range(mc_num)]

                            mc_dropout_var = torch.var(torch.stack([i[0] for i in temp_outputs]), dim=0)
                            outputs = torch.mean(torch.stack([i[0] for i in temp_outputs]), dim=0)
                            outputs_log_var = torch.mean(torch.stack([i[1] for i in temp_outputs]), dim=0)
                            outputs_std = torch.sqrt(torch.exp(outputs_log_var) + mc_dropout_var)
                            outputs_model_std = torch.sqrt(mc_dropout_var)
                            outputs_holder[i * batchsize : (i + 1) * batchsize] = outputs.cpu().numpy()
                            outputs_std_holder[i * batchsize : (i + 1) * batchsize] = outputs_std.cpu().numpy()
                            outputs_model_std_holder[i * batchsize : (i + 1) * batchsize] = outputs_model_std.cpu().numpy()
                            pbar.update(batchsize)
                    if num_data_remaining > 0:
                        batch_inputs = torch.tensor(inputs[-num_data_remaining:] - input_mean, **self.factory_kwargs)[:, :, None]
                        temp_outputs = [self(batch_inputs) for i in range(mc_num)]

                        mc_dropout_var = torch.var(torch.stack([i[0] for i in temp_outputs]), dim=0)
                        outputs = torch.mean(torch.stack([i[0] for i in temp_outputs]), dim=0)
                        outputs_log_var = torch.mean(torch.stack([i[1] for i in temp_outputs]), dim=0)
                        outputs_std = torch.sqrt(torch.exp(outputs_log_var) + mc_dropout_var)
                        outputs_model_std = torch.sqrt(mc_dropout_var)
                        outputs_holder[-num_data_remaining:] = outputs.cpu().numpy()
                        outputs_std_holder[-num_data_remaining:] = outputs_std.cpu().numpy()
                        outputs_model_std_holder[-num_data_remaining:] = outputs_model_std.cpu().numpy()
                        pbar.update(num_data_remaining)

            return np.squeeze(outputs_holder * labels_std), np.squeeze(outputs_std_holder * labels_std), np.squeeze(outputs_model_std_holder * labels_std)

    def __init__(self, model_path, device):
        """TODO: Docstring this"""

        self.model = self.ApogeeDistBCNN(device=device)
        self.model.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])

        self.targetname = self.model.targetname
        self.predict = self.model.predict


@cache
def read_model(network_path, device=None):
    #device = 'cpu'
    #print('='*6, 'Running with {}'.format(device or DEVICE))
    return Network(expand_path(network_path), device or DEVICE)

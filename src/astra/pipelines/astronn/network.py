from astra.utils import expand_path
from functools import cache

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Network:
    class ApogeeBCNNCensored(torch.nn.Module):
        def __init__(self, device="cpu", dtype=torch.float32, **kwargs):
            super().__init__(**kwargs)
            self.factory_kwargs = {"device": device, "dtype": dtype}
            self.activation = "relu"
            self.filter_len = 8
            self.pool_length = 4
            self.dropout_rate = 0.3

            self.targetname = [
                "teff",
                "logg",
                "c_h",
                "c1_h",
                "n_h",
                "o_h",
                "na_h",
                "mg_h",
                "al_h",
                "si_h",
                "p_h",
                "s_h",
                "k_h",
                "ca_h",
                "ti_h",
                "ti2_h",
                "v_h",
                "cr_h",
                "mn_h",
                "fe_h",
                "co_h",
                "ni_h",
            ]

            self.aspcap_masks_sum = [3806, 557,3776,505,13,106,51,102,24,8,16,19,31,28,29,46,70,24,206]
            self.aspcap_masks = torch.nn.ParameterList([
                torch.nn.Parameter(
                    torch.zeros(7514, dtype=bool, device=self.factory_kwargs["device"]),
                    requires_grad=False,
                ) for i in range(19)
            ])
            self.input_mean = torch.nn.Parameter(torch.zeros(7514, dtype=torch.float64, device=self.factory_kwargs["device"]), requires_grad=False)
            self.labels_mean = torch.nn.Parameter(torch.zeros(22, dtype=torch.float64, device=self.factory_kwargs["device"]), requires_grad=False)
            self.labels_std = torch.nn.Parameter(torch.zeros(22, dtype=torch.float64, device=self.factory_kwargs["device"]), requires_grad=False)
            dense1_num = np.ones(19, dtype=int) * 32
            dense1_num[0] *= 8
            dense1_num[2] *= 8
            self.elem_dense1 = torch.nn.ModuleList(
                [
                    torch.nn.Linear(i, j, **self.factory_kwargs)
                    for i, j in zip(self.aspcap_masks_sum, dense1_num)
                ]
            )
            dense2_num = np.ones(19, dtype=int) * 16
            dense2_num[0] *= 4
            dense2_num[2] *= 4
            self.elem_dense2 = torch.nn.ModuleList(
                [
                    torch.nn.Linear(i, j, **self.factory_kwargs)
                    for i, j in zip(dense1_num, dense2_num)
                ]
            )

            self.elem_dense_out = torch.nn.ModuleList(
                [torch.nn.Linear(i + 5, 1, **self.factory_kwargs) for i in dense2_num]
            )
            self.elem_dense_var = torch.nn.ModuleList(
                [torch.nn.Linear(i + 5, 1, **self.factory_kwargs) for i in dense2_num]
            )

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
            self.dense_2 = torch.nn.Linear(192, 96, **self.factory_kwargs)
            self.dense_teffloggfeh = torch.nn.Linear(96, 3, **self.factory_kwargs)
            self.dense_teffloggfeh_var = torch.nn.Linear(96, 3, **self.factory_kwargs)
            self.dense_aux_fullspec = torch.nn.Linear(96, 2, **self.factory_kwargs)

        def forward(self, inputs):
            # slice spectra to censor out useless region for elements
            inputs_elem_censored = [
                torch.flatten(inputs, start_dim=1)[:, i] for i in self.aspcap_masks
            ]
            elem_dense_1 = [
                torch.nn.functional.dropout(
                    torch.nn.functional.relu(j(i)), self.dropout_rate
                )
                for i, j in zip(inputs_elem_censored, self.elem_dense1)
            ]
            elem_dense_2 = [
                torch.nn.functional.dropout(
                    torch.nn.functional.relu(j(i)), self.dropout_rate
                )
                for i, j in zip(elem_dense_1, self.elem_dense2)
            ]

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
            teffloggfeh = self.dense_teffloggfeh(layer_4)
            teffloggfeh_var = self.dense_teffloggfeh_var(layer_4)
            aux_fullspec = self.dense_aux_fullspec(layer_4)
            fullspec_hidden = torch.concat([aux_fullspec, teffloggfeh], dim=1)

            elem_dense_out = [
                j(torch.concat([i, fullspec_hidden], dim=1))
                for i, j in zip(elem_dense_2, self.elem_dense_out)
            ]
            elem_dense_var = [
                j(torch.concat([i, fullspec_hidden], dim=1))
                for i, j in zip(elem_dense_2, self.elem_dense_var)
            ]

            # concatenate answer
            output = torch.concat(
                [
                    teffloggfeh[:, :2],
                    *elem_dense_out[:-2],
                    teffloggfeh[:, -1:],
                    *elem_dense_out[-2:],
                ],
                dim=1,
            )

            # concatenate predictive uncertainty
            log_variance_output = torch.concat(
                [
                    teffloggfeh_var[:, :2],
                    *elem_dense_var[:-2],
                    teffloggfeh_var[:, -1:],
                    *elem_dense_var[-2:],
                ],
                dim=1,
            )

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
            labels_mean = self.labels_mean.cpu().numpy()
            labels_std = self.labels_std.cpu().numpy()

            data_length = inputs.shape[0]
            num_full_batch = data_length // batchsize
            num_data_remaining = data_length % batchsize

            outputs_holder = np.zeros((data_length, 22))
            outputs_std_holder = np.zeros((data_length, 22))
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
                            outputs_holder[i * batchsize : (i + 1) * batchsize] = outputs.cpu().numpy()
                            outputs_std_holder[i * batchsize : (i + 1) * batchsize] = outputs_std.cpu().numpy()
                            pbar.update(batchsize)
                    if num_data_remaining > 0:
                        batch_inputs = torch.tensor(inputs[-num_data_remaining:] - input_mean, **self.factory_kwargs)[:, :, None]
                        temp_outputs = [self(batch_inputs) for i in range(mc_num)]

                        mc_dropout_var = torch.var(torch.stack([i[0] for i in temp_outputs]), dim=0)
                        outputs = torch.mean(torch.stack([i[0] for i in temp_outputs]), dim=0)
                        outputs_log_var = torch.mean(torch.stack([i[1] for i in temp_outputs]), dim=0)
                        outputs_std = torch.sqrt(torch.exp(outputs_log_var) + mc_dropout_var)
                        outputs_holder[-num_data_remaining:] = outputs.cpu().numpy()
                        outputs_std_holder[-num_data_remaining:] = outputs_std.cpu().numpy()
                        pbar.update(num_data_remaining)

            #return pd.DataFrame(np.column_stack([outputs_holder * labels_std + labels_mean, 
            #                                    outputs_std_holder * labels_std]), 
            #                                    columns=self.targetname + [self.targetname[i] + "_err" for i in range(22)])
            pred = outputs_holder * labels_std + labels_mean
            pred_err = outputs_std_holder * labels_std
            return pred, pred_err
    
    def __init__(self, model_path, device):
        """TODO: Docstring this"""

        self.model = self.ApogeeBCNNCensored(device=device)
        self.model.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])

        self.targetname = self.model.targetname
        self.predict = self.model.predict


@cache
def read_model(network_path, device=None):
    return Network(expand_path(network_path), device or DEVICE)
# encoding: utf-8


from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
import sys
import os
import pickle
from tqdm import trange

from astropy.table import Table

import torch
from torch.autograd import Variable

from astra.utils import log

# Check for CUDA support.
try:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

except TypeError:
    log.exception("Torch not compiled with CUDA support")
    CUDA_AVAILABLE = False

else:
    CUDA_AVAILABLE = True

default_tensor_type = torch.cuda.FloatTensor if CUDA_AVAILABLE else torch.FloatTensor
torch.set_default_tensor_type(default_tensor_type)



def _prepare_data(training_spectra, training_labels, validation_spectra, validation_labels,
                  label_names):

    # Check label names are in both sets of labels.
    if isinstance(training_labels, Table):
        missing_training_labels = set(label_names).difference(training_labels.dtype.names)
        if missing_training_labels:
            raise ValueError(f"missing labels {missing_training_labels} in training set labels")

        training_labels = np.atleast_2d([training_labels[ln] for ln in label_names]).T

    if isinstance(validation_labels, Table):
        missing_validation_labels = set(label_names).difference(validation_labels.dtype.names)
        if missing_validation_labels:
            raise ValueError(f"missing labels {missing_validation_labels} in validation set labels")

        validation_labels = np.atleast_2d([validation_labels[ln] for ln in label_names]).T

    # Check shapes of data.
    T = len(training_spectra)
    if isinstance(training_spectra, list):
        P = set([spectrum.flux.size for spectrum in training_spectra])
        if len(P) > 1:
            raise ValueError(f"training spectra have various numbers of pixels: {P}")

        P = list(P)[0]

        _P = list(set([spectrum.flux.size for spectrum in validation_spectra]))
        if len(_P) > 1 or P != _P[0]:
            raise ValueError("validation spectra have various numbers of pixels or different pixels to "
                             "the training spectra")

        training_flux = np.zeros((T, P), dtype=float)
        for i, spectrum in enumerate(training_spectra):
            training_flux[i] = spectrum.flux.value

    else:
        training_flux = training_spectra

    if isinstance(validation_spectra, list):
        V = len(validation_spectra)
        validation_flux = np.zeros((V, P), dtype=float)
        for i, spectrum in enumerate(validation_spectra):
            validation_flux[i] = spectrum.flux.value

    else:
        validation_flux = validation_spectra

    return (training_flux, training_labels, validation_flux, validation_labels)


def _whiten_labels(training_labels, validation_labels):

    x_min, x_max = (np.min(training_labels, axis=0), np.max(training_labels, axis=0))

    T = (training_labels - x_min)/(x_max - x_min) - 0.5
    V = (validation_labels - x_min)/(x_max - x_min) - 0.5

    return (T, V, x_min, x_max)



def train(training_spectra, training_labels, validation_spectra, validation_labels, label_names,
          num_neurons=300, num_epochs=1e5, learning_rate=0.001, weight_decay=0, **kwargs):
    r"""
    Train a neural network to emulate spectral models.
    
    :param training_spectra:
        A list of :class:`specutils.Spectrum1D` spectra to use for training.

    :param training_labels:
        A table containing the training labels to use. This must include the list of label names
        given in `label_names`.

    :param validation_spectra:
        A list of :class:`specutils.Spectrum1D` spectra to use for validation.

    :param validation_labels:
        A table containing the validation labels to use. This must include the list of label names
        given in `label_names`.

    :param num_neurons: [optional]
        The number of neurons to use per hidden layer in the network (default: 300).

    :param num_epochs: [optional]
        The maximum nubmer of steps to take until convergence (default: 1e5).

    :param learning_rate: [optional]
        The step size to take for gradient descent (default: 0.001).

    :param weight_decay: [optional]
        The weight decay (regularization) to apply to the model (default: 0).
    
    :returns:
        A three-length tuple containing the state, the model, and the optimizer. All information
        needed to reconstruct the model is saved in the state.
    """

    # Deal with the data.
    training_flux, training_labels, validation_flux, validation_labels = _prepare_data(
        training_spectra, training_labels, validation_spectra, validation_labels, label_names)

    return _train(training_flux, training_labels, validation_flux, validation_labels, label_names,
                  num_neurons, num_epochs, learning_rate, weight_decay, **kwargs)



def _train(training_flux, training_labels, validation_flux, validation_labels, label_names,
           num_neurons, num_epochs, learning_rate, weight_decay, **kwargs):
    
    num_neurons, num_epochs = (int(num_neurons), int(num_epochs))

    # Normalize.
    whitened_training_labels, whitened_validation_labels, *scales = _whiten_labels(training_labels,
                                                                                   validation_labels)

    n_labels = whitened_training_labels.shape[1] # number of labels
    n_pixels = training_flux.shape[1] # number of pixels

    # Define network.
    model = torch.nn.Sequential(
        torch.nn.Linear(n_labels, num_neurons),
        torch.nn.Sigmoid(),
        torch.nn.Linear(num_neurons, num_neurons),
        torch.nn.Sigmoid(),
        torch.nn.Linear(num_neurons, n_pixels)
    )
    if CUDA_AVAILABLE:
        model.cuda()

    # L2 loss
    loss_function = torch.nn.MSELoss(reduction="mean")
    
    x_train = Variable(torch.from_numpy(whitened_training_labels)).type(default_tensor_type)
    y_train = Variable(torch.from_numpy(training_flux), requires_grad=False).type(default_tensor_type)
    x_valid = Variable(torch.from_numpy(whitened_validation_labels)).type(default_tensor_type)
    y_valid = Variable(torch.from_numpy(validation_flux), requires_grad=False).type(default_tensor_type)


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Train the network.
    training_loss = np.zeros(num_epochs, dtype=float)
    validation_loss = np.zeros(num_epochs, dtype=float)

    _BOOSTER = 1e4 # MAGIC HACK

    with trange(num_epochs) as pb:
        for step in range(num_epochs):
            y_train_pred = model(x_train)
            y_valid_pred = model(x_valid)

            loss_train = _BOOSTER * loss_function(y_train_pred, y_train)
            loss_valid = _BOOSTER * loss_function(y_valid_pred, y_valid)

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            training_loss[step] = loss_train.data.item()
            validation_loss[step] = loss_valid.data.item()

            # Update progress bar.
            pb.set_description(f"Step {1 + step}")
            pb.set_postfix(train_loss=training_loss[step],
                           valid_loss=validation_loss[step])
            pb.update()

    state = dict(epoch=1 + step, model_state=model.state_dict(), scales=scales,
                 label_names=label_names, losses=(training_loss, validation_loss),
                 weight_decay=weight_decay, num_neurons=num_neurons, n_pixels=n_pixels,
                 n_labels=n_labels, learning_rate=learning_rate)

    if kwargs.get("full_output", False):
        state["optimizer_state"] = optimizer.state_dict()

    return (state, model, optimizer)



def load_training_data(path):
    '''
    read in the default Kurucz training spectra for APOGEE

    Here we only consider 800 training spectra and 200 validation spectra
    for the tutorial (due to the GitHub upload limit); in practice, more
    training spectra will be better. The default neural networks included were
    trained using 10000 training spectra.
    '''

    '''
    tmp = np.load(path)
    training_labels = (tmp["labels"].T)[:800,:]
    training_spectra = tmp["spectra"][:800,:]
    validation_labels = (tmp["labels"].T)[800:,:]
    validation_spectra = tmp["spectra"][800:,:]
    tmp.close()
    '''

    with open(path, "rb") as fp:
        contents = pickle.load(fp)
    
    N = 800
    training_labels = contents["labels"].T[:N, :]
    training_spectra = contents["spectra"][:N, :]
    validation_labels = contents["labels"].T[N:, :]
    validation_spectra = contents["spectra"][N:, :]
    return (contents["wavelength"], contents["label_names"], training_labels, training_spectra, validation_labels, validation_spectra)


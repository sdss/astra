import os
import pickle
import numpy as np
import torch
from time import time
from tqdm import tqdm
from sdss_access import SDSSPath
from inspect import signature
from collections import OrderedDict

from astra.database import astradb, session
from astra.tools.spectrum import Spectrum1D
from astra.database.utils import create_task_output, deserialize_pks
from astra.contrib.thepayne import training, test
from astra.utils import hashify, log, get_base_output_path
from astra.operators import ApStarOperator, AstraOperator
from astra.operators.utils import prepare_data

def get_model_path(
        training_set_path: str,
        num_epochs=100_000,
        num_neurons=300,
        weight_decay=0.0,
        learning_rate=0.001,
        **kwargs
    ):
    """
    Return the path of where the output model will be stored, given the training set path
    and network hyperparameters.
        
    :param training_set_path:
        the path of the training set

    :param num_epochs: (optional)
        The number of steps to train the network for (default 100000).
    
    :param num_neurons: (optional)
        The number of neurons to use in the hidden layer (default: 300).
    
    :param weight_decay: (optional)
        The weight decay to use during training (default: 0)
    
    :param learning_rate: (optional)
        The learning rate to use during training (default: 0.001).    
    """
    # Expand the training set path for the purposes of hashing.
    training_set_path = os.path.expanduser(os.path.expandvars(training_set_path))

    param_dict = OrderedDict()
    for arg in signature(get_model_path).parameters.keys():
        if arg != "kwargs":
            param_dict[arg] = locals()[arg]
    
    log.debug(f"Hashing {param_dict} for The Payne model path")
    param_hash = hashify(param_dict)
    
    basename = f"thepayne_model_{param_hash}.pkl"
    path = os.path.join(
        get_base_output_path(),
        "thepayne",
        basename
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path



def train_model(
        training_set_path,
        output_model_path=None,
        num_epochs=100_000,
        num_neurons=300,
        weight_decay=0.0,
        learning_rate=0.001,
        **kwargs
    ):
    """
    Train a single-layer neural network given a pre-computed grid of synthetic spectra.

    :param training_set_path:
        The path where the training set spectra and labels are stored.
        This should be a binary pickle file that contains a dictionary with the following keys:

        - wavelength: an array of shape (P, ) where P is the number of pixels
        - spectra: an array of shape (N, P) where N is the number of spectra and P is the number of pixels
        - labels: an array of shape (L, P) where L is the number of labels and P is the number of pixels
        - label_names: a tuple of length L that contains the names of the labels
    
    :param num_epochs: (optional)
        The number of steps to train the network for (default 100000).
    
    :param num_neurons: (optional)
        The number of neurons to use in the hidden layer (default: 300).
    
    :param weight_decay: (optional)
        The weight decay to use during training (default: 0)
    
    :param learning_rate: (optional)
        The learning rate to use during training (default: 0.001).
    """

    if output_model_path is None:
        output_model_path = get_model_path(
            training_set_path=training_set_path,
            num_epochs=num_epochs,
            num_neurons=num_neurons,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            **kwargs
        )

    wavelength, label_names, \
        training_labels, training_spectra, \
        validation_labels, validation_spectra = training.load_training_data(training_set_path)

    state, model, optimizer = training.train(
        training_spectra, 
        training_labels,
        validation_spectra,
        validation_labels,
        label_names,
        num_neurons=int(num_neurons),
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )   

    # Ensure that the output folder exists.
    os.makedirs(
        os.path.dirname(output_model_path),
        exist_ok=True
    )
    log.info(f"Writing model to {output_model_path}")
    with open(output_model_path, "wb") as fp:
        pickle.dump(dict(
                state=state,
                wavelength=wavelength,
                label_names=label_names, 
            ),
            fp
        )

    # Try to send xcom result of the output path.
    try:
        ti = kwargs["ti"]
        ti.xcom_push("model_path", output_model_path)
    except:
        log.exception("Unable to send `model_path` as xcom variable")
    else:
        log.info(f"Passed model_path as {output_model_path}")



def estimate_stellar_labels(pks, **kwargs):
    """
    Estimate stellar labels given a single-layer neural network.

    :param pks:
        The primary keys of task instances to estimate stellar labels for. The
        task instances include information to identify the source SDSS data product.
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    log.info(f"Running ThePayne on device {device} with:")
    log.info(f"CUDA_VISIBLE_DEVICES = '{os.environ.get('CUDA_VISIBLE_DEVICES')}'")
    log.info(f"Using torch version {torch.__version__} in {torch.__path__}")

    states = {}
    masks   = {}

    log.info(f"Estimating stellar labels for task instances")

    results = {}
    for instance, path, spectrum in prepare_data(pks):
        if spectrum is None: continue

        model_path = instance.parameters["model_path"]
        try:
            state = states[model_path]
            mask  = masks[model_path]
        except KeyError:
            log.info(f"Loading model from {model_path}")
            state = states[model_path] = test.load_state(model_path)    
            mask  = masks[model_path] =  test.load_apogee_mask(model_path)
            #mask  = masks[model_path] =  test.load_apogee_mask("/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/sandbox/astra/test/song/ThePayne/other_data/apogee_mask.npz")

            label_names = state["label_names"]
            L = len(label_names)
            log.info(f"Estimating these {L} label names: {label_names}")

        # Run optimization.
        t_init = time()
        p_opt, p_cov, model_flux, meta = test.test(
            spectrum.wavelength.value,
            spectrum.flux.value,
            spectrum.uncertainty.array,
            mask,
            **state
        )
        t_opt = time() - t_init
        
        #log.debug(f"spectrum shape: {spectrum.flux.shape}")
        #log.debug(f"p_opt shape: {p_opt.shape}")
        #log.debug(f"spectrum meta: {spectrum.meta['snr']}")

        # Prepare outputs.
        result = dict(zip(label_names, p_opt.T))
        result.update(snr=spectrum.meta["snr"])
        # Include uncertainties.
        result.update(dict(zip(
            (f"u_{ln}" for ln in label_names),
            np.sqrt(p_cov[:, np.arange(p_opt.shape[1]), np.arange(p_opt.shape[1])].T)
        )))
        
        results[instance.pk] = result
        log.info(f"Result for {instance} took {t_opt} seconds")

    # Write database outputs.
    for pk, result in tqdm(results.items(), desc="Writing database outputs"):
        # Write database outputs.
        create_task_output(pk, astradb.ThePayne, **result)

    return None


import os
import pickle
import numpy as np
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
        output_model_path,
        training_set_path,
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
    
    wavelength, label_names, \
        training_labels, training_spectra, \
        validation_labels, validation_spectra = training.load_training_data(training_set_path)

    state, model, optimizer = training.train(
        training_spectra, 
        training_labels,
        validation_spectra,
        validation_labels,
        label_names,
        num_neurons=num_neurons,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )   

    # Ensure that the output folder exists.
    os.makedirs(
        os.path.dirname(output_model_path),
        exist_ok=True
    )

    with open(output_model_path, "wb") as fp:
        pickle.dump(dict(
                state=state,
                wavelength=wavelength,
                label_names=label_names, 
            ),
            fp
        )


def estimate_stellar_labels(
        pks,
        model_path,
    ):
    """
    Estimate stellar labels given a single-layer neural network.

    :param pks:
        The primary keys of task instances to estimate stellar labels for. The
        task instances include information to identify the source SDSS data product.
    
    :param model_path:
        The path where the pre-trained model is stored.
    
    :param analyze_individual_visits: [optional]
        If `True`, analyze all individual visits in the SDSS data product (default).
        If `False` it will analyze only the first (zero-th index) spectrum in the 
        data product, which is usually the stacked spectrum.
    """
    log.info(f"Loading model from {model_path}")
    
    state = test.load_state(model_path)

    label_names = state["label_names"]
    L = len(label_names)
    log.info(f"Estimating these {L} label names: {label_names}")

    log.info(f"Estimating stellar labels for task instances")

    results = {}
    for instance, path, spectrum in prepare_data(pks):
        if spectrum is None: continue

        # Run optimization.
        p_opt, p_cov, model_flux, meta = test.test(
            spectrum.wavelength.value,
            spectrum.flux.value,
            spectrum.uncertainty.array,
            **state
        )
        
        log.debug(f"spectrum shape: {spectrum.flux.shape}")
        log.debug(f"p_opt shape: {p_opt.shape}")
        log.debug(f"spectrum meta: {spectrum.meta['snr']}")

        # Prepare outputs.
        result = dict(zip(label_names, p_opt.T))
        result.update(snr=spectrum.meta["snr"])
        # Include uncertainties.
        result.update(dict(zip(
            (f"u_{ln}" for ln in label_names),
            np.sqrt(p_cov[:, np.arange(p_opt.shape[1]), np.arange(p_opt.shape[1])].T)
        )))
        
        results[instance.pk] = result
        log.info(f"Result for {instance}: {result}")

    # Write database outputs.
    for pk, result in tqdm(results.items(), desc="Writing database outputs"):
        # Write database outputs.
        create_task_output(pk, astradb.ThePayne, **result)

    return None


class TrainThePayneOperator(AstraOperator):

    def __init__(
        self,
        output_model_path,
        training_set_path,
        num_epochs=100_000,
        num_neurons=300,
        weight_decay=0.0,
        learning_rate=0.001,
        **kwargs,
    ) -> None:
        super(TrainThePayneOperator, self).__init__(**kwargs)
        self.output_model_path = output_model_path
        self.training_set_path = training_set_path
        self.num_epochs = num_epochs
        self.num_neurons = num_neurons
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        
        self.bash_command_prefix = "astra run thepayne train"
        self.python_callable = train_model

        return None

    

class ThePayneApStarOperator(ApStarOperator):

    template_fields = ("model_path", )

    def __init__(
        self,
        model_path: str,
        **kwargs,
    ) -> None:
        super(ThePayneApStarOperator, self).__init__(**kwargs)
        self.model_path = model_path
        self.python_callable = estimate_stellar_labels
        self.bash_command_prefix = "astra run thepayne test"

        return None

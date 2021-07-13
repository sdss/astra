import os
import pickle
from tqdm import tqdm
from sdss_access import SDSSPath

from astra.database import astradb, session
from astra.tools.spectrum import Spectrum1D
from astra.database.utils import create_task_output
from astra.contrib.thepayne import training, test
from astra.utils import hashify, log

def get_model_path(
        training_set_path: str,
        n_steps=100_000,
        n_neurons=300,
        weight_decay=0.0,
        learning_rate=0.001,
        **kwargs
    ):
    """
    Return the path of where the output model will be stored, given the training set path
    and network hyperparameters.
        
    :param training_set_path:
        the path of the training set

    :param n_steps: (optional)
        The number of steps to train the network for (default 100000).
    
    :param n_neurons: (optional)
        The number of neurons to use in the hidden layer (default: 300).
    
    :param weight_decay: (optional)
        The weight decay to use during training (default: 0)
    
    :param learning_rate: (optional)
        The learning rate to use during training (default: 0.001).    
    """

    param_hash = hashify(dict([(arg, locals()[arg]) for arg in getargspec(get_model_path).args]))
    
    # TODO: This should go to some $ASTRA_DATA_DIR or something.
    return f"thepayne_model_{param_hash}.pkl"


def train_model(
        output_model_path,
        training_set_path,
        n_steps=100_000,
        n_neurons=300,
        weight_decay=0.0,
        learning_rate=0.001
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
    
    :param n_steps: (optional)
        The number of steps to train the network for (default 100000).
    
    :param n_neurons: (optional)
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
        n_neurons=n_neurons,
        n_steps=n_steps,
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
        analyze_individual_visits=True,
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

    pks = deserialize_pks(pks)
    N = len(pks)

    log.info(f"Estimating stellar labels for {N} task instances")

    # Get the task instances.
    q = session.query(astradb.TaskInstance)\
               .filter(astradb.TaskInstance.pk.in_(pks))
    
    trees = {}

    sliced = slice(0, 1) if analyze_individual_visits else slice(None)

    for instance in tqdm(q.yield_per(1), total=N):

        # Get the path.
        parameters = instance.parameters
        tree = trees.get(parameters["release"], None)
        if tree is None:
            trees[parameters["release"]] = tree = SDSSPath(release=parameters["release"])
        
        path = tree.full(**parameters)

        # Load spectrum.
        try:
            spectrum = Spectrum1D.read(path)
        except:
            log.exception(f"Unable to load spectrum from {path} for task instance {instance}")
            continue
        
        # Run optimization.
        p_opt, p_cov, model_flux, meta = test.test(
            spectrum.wavelength.value,
            spectrum.flux.value[sliced],
            spectrum.uncertainty.array[sliced],
            **state
        )

        # Prepare outputs.
        results = dict(zip(label_names, p_opt.T))
        results.update(snr=spectrum.meta["snr"][sliced])
        # Include uncertainties.
        results.update(dict(zip(
            (f"u_{ln}" for ln in label_names),
            np.sqrt(p_cov[:, np.arange(L), np.arange(L)].T)
        )))
        
        # Write database outputs.
        create_task_output(
            instance,
            astradb.ThePayne,
            **results
        )

        # Query whether to write AstraSource objects.
        log.warning(f"Not writing AstraSource objects.")

    return None

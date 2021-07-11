
import os
import torch
import numpy as np
from inspect import getargspec
from torch.autograd import Variable
from tqdm import tqdm

from astra.contrib.classifier import networks, model, plot_utils, utils
from astra.database import astradb, session
from astra.database.utils import create_task_output, deserialize_pks
from astra.tools.spectrum import Spectrum1D
from astra.utils import log

from sdss_access import SDSSPath


def get_model_path(
        network_factory: str,
        training_spectra_path: str,
        training_labels_path: str,
        learning_rate: float,
        weight_decay: float,
        n_epochs: int,
        batch_size: int,
        **kwargs
    ):
    """
    Return the path of where the output model will be stored, given the network factory name,
    the training spectra path, training labels path, and training hyperparameters.
    
    :param network_factory: 
        the name of the network factory (e.g., OpticalCNN, NIRCNN)
    
    :param training_spectra_path:
        the path of the training spectra
    
    :param training_labels_path:
        the path where the training labels are stored
    
    :param learning_rate:
        the learning rate to use during training
    
    :param n_epochs:
        the number of epochs to use during training
    
    :param batch_size:
        the batch size to use during training
    """

    param_hash = hashify(dict([(arg, locals()[arg]) for arg in getargspec(get_model_path).args]))
    
    # TODO: This should go to some $ASTRA_DATA_DIR or something.
    return f"classifier_{network_factory}_{param_hash}.pt"


def classify(
        pks,
        model_path,
        network_factory,
    ):
    """
    Classify a source given primary keys that reference some partially created task instances.

    :param pks:
        the primary keys of the task instances in the database that need classification
    
    :param model_path:
        the path where the model is stored on disk
    
    :param network_factory:
        the name of the network factory to use to load the model (e.g., OpticalCNN, NIRCNN)
    """

    print(f"In classify with pks {type(pks)} {pks} {model_path}, {network_factory}")
    
    factory = getattr(networks, network_factory)
    if factory is None:
        raise ValueError(f"unknown network factory '{network_factory}'")
    
    model = utils.read_network(factory, model_path)
    # Disable dropout for inference.
    model.eval()

    # Get the task instances.
    q = session.query(astradb.TaskInstance).filter(astradb.TaskInstance.pk.in_(deserialize_pks(pks)))

    trees = {}

    for instance in tqdm(q.yield_per(1)):
        
        parameters = instance.parameters
        tree = trees.get(parameters["release"], None)
        if tree is None:
            trees[parameters["release"]] = tree = SDSSPath(release=parameters["release"])
        
        path = tree.full(**parameters)

        # Load the spectrum.
        try:
            spectrum = Spectrum1D.read(path)
        except:
            log.exception(f"Unable to load Spectrum1D from path {path} on task instance {instance}")
            continue

        flux = spectrum.flux.value
        if flux.size == 6144:
            # Undithered ApVisit spectra have half as many pixels as dithered spectra (duh). 
            # This is a hack to make them work with the classifier, which expects the test set to be
            # the same shape (in pixels) as the training set.
            # TODO: Consider doing something clever instead.
            flux = np.repeat(flux, 2)

        # TODO: Is this the same for the Optical CNN? Should check.
        flux = flux.reshape((1, 3, -1))
        batch = flux / np.nanmedian(flux, axis=2)[:, :, None]

        with torch.no_grad():
            prediction = model.forward(Variable(torch.Tensor(batch)))
            log_prob = prediction.cpu().numpy().flatten()
                
        # Make sure the log_probs are dtype float so that postgresql does not complain.
        log_prob = np.array(log_prob, dtype=float)

        # Calculate normalized probabilities.
        with np.errstate(under="ignore"):
            relative_log_prob = log_prob - logsumexp(log_prob)
        
        # TODO: This many not be necessary anymore now that we are using SQLAlchemy ORM.
        #       Leaving the logic here in case this fails and we do need to re-cast types.

        # Round for PostgreSQL 'real' type.
        # https://www.postgresql.org/docs/9.1/datatype-numeric.html
        # and
        # https://stackoverflow.com/questions/9556586/floating-point-numbers-of-python-float-and-postgresql-double-precision
        decimals = 36
        prob = np.round(np.exp(relative_log_prob), decimals)
        log_prob = np.round(log_prob, decimals)

        # Write the output to the database.
        create_task_output(
            instance,
            astradb.Classification,
            log_prob=log_prob,
            prob=prob
        )


def classify_sdss5_apstar(pks):
    """
    Classify observations of SDSS5 APOGEE (ApStar) sources, given the existing classifications of the
    individual visits.

    :param pks:
        The primary keys of task instances where visits have been classified. These primary keys will
        be used to work out which stars need classifying, before tasks
    """

    print(pks)
    print(deserialize_pks(pks))
    # Match stars to visits, for the same dag_id.
    q = session.query(astradb.TaskInstance).filter(astradb.TaskInstance.pk.in_(deserialize_pks(pks)))
    
    raise NotImplementedError






def train_model(
        output_model_path,
        training_spectra_path, 
        training_labels_path,
        validation_spectra_path,
        validation_labels_path,
        test_spectra_path,
        test_labels_path,
        class_names,
        network_factory,
        learning_rate=1e-5,
        weight_decay=1e-5,
        n_epochs=200,
        batch_size=100,
        **kwargs
    ):
    """
    Train a classifier.

    :param output_model_path:
        the disk path where to save the model to

    :param training_spectra_path:
        A path that contains the spectra for the training set.
    
    :param training_set_labels:
        A path that contains the labels for the training set.

    :param validation_spectra_path:
        A path that contains the spectra for the validation set.

    :param validation_labels_path:
        A path that contains the labels for the validation set.

    :param test_spectra_path:
        A path that contains the spectra for the test set.
    
    :param test_labels_path:
        A path that contains ths labels for the test set.

    :param class_names:
        A tuple of names for the object classes.
    
    :param n_epochs: (optional)
        The number of epochs to use for training (default: 200).
    
    :param batch_size: (optional)
        The number of objects to use per batch in training (default: 100).
    
    :param weight_decay: (optional)
        The weight decay to use during training (default: 1e-5).
    
    :param learning_rate: (optional)
        The learning rate to use during training (default: 1e-4).
    """

    try:
        network_factory = getattr(networks, network_factory)
    
    except AttributeError:
        raise ValueError(f"No such network factory exists ({network_factory})")
    
    training_spectra, training_labels = utils.load_data(training_spectra_path, training_labels_path)
    validation_spectra, validation_labels = utils.load_data(validation_spectra_path, validation_labels_path)
    test_spectra, test_labels = utils.load_data(test_spectra_path, test_labels_path)

    state, network, optimizer = model.train(
        network_factory,
        training_spectra,
        training_labels,
        validation_spectra,
        validation_labels,
        test_spectra,
        test_labels,
        class_names=class_names,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        n_epochs=n_epochs,
        batch_size=batch_size,
    )

    # Write the model to disk.
    utils.write_network(network, output_model_path)

    '''
    # Disable dropout for inference.
    with torch.no_grad():                
        pred = network.forward(Variable(torch.Tensor(test_spectra)))
        outputs = pred.data.numpy()

    pred_test_labels = np.argmax(outputs, axis=1)

    # Make a confusion matrix plot.
    fig = plot_utils.plot_confusion_matrix(
        test_labels, 
        pred_test_labels, 
        self.class_names,
        normalize=False,
        title=None,
        cmap=plt.cm.Blues
    )
    fig.savefig(
        os.path.join(
            self.output_base_dir,
            f"{self.task_id}.png"
        ),
        dpi=300
    )
    '''



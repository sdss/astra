
import os
import torch
import numpy as np
from inspect import getargspec
from scipy.special import logsumexp
from sqlalchemy import (or_, and_, func, distinct)
from torch.autograd import Variable
from tqdm import tqdm

from astra.contrib.classifier import networks, model, plot_utils, utils
from astra.database import astradb, session
from astra.database.utils import (get_or_create_task_instance, create_task_output, serialize_pks_to_path, deserialize_pks, get_or_create_parameter_pk)
from astra.tools.spectrum import Spectrum1D
from astra.utils import log, get_scratch_dir, hashify, get_base_output_path

from sdss_access import SDSSPath

from astra.operators import (AstraOperator, ApStarOperator, ApVisitOperator, BossSpecOperator)
from astra.operators.utils import prepare_data

def train_model(
        output_model_path,
        training_spectra_path, 
        training_labels_path,
        validation_spectra_path,
        validation_labels_path,
        test_spectra_path,
        test_labels_path,
        network_factory,
        class_names=None,
        learning_rate=1e-5,
        weight_decay=1e-5,
        num_epochs=200,
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

    :param network_factory:
        The name of the network factory to use in `astra.contrib.classifier.model`

    :param class_names: (optional)
        A tuple of names for the object classes.
    
    :param num_epochs: (optional)
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
        num_epochs=num_epochs,
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



def classify(pks):
    """
    Classify sources given the primary keys of task instances.

    :param pks:
        the primary keys of the task instances in the database that need classification
    """

    models = {}    
    results = {}
    for instance, path, spectrum in prepare_data(pks):
        if spectrum is None: continue

        model_path = instance.parameters["model_path"]

        try:
            model, factory = models[model_path]
        except KeyError:
            network_factory = model_path.split("_")[-2]
            factory = getattr(networks, network_factory)

            log.info(f"Loading model from {model_path} using {factory}")
            model = utils.read_network(factory, model_path)
            model.eval()

            models[model_path] = (model, factory)

        flux = torch.from_numpy(spectrum.flux.value.astype(np.float32))

        with torch.no_grad():
            prediction = model.forward(flux)#Variable(torch.Tensor(spectrum.flux.value)))
            log_probs = prediction.cpu().numpy().flatten()
                
        results[instance.pk] = log_probs
    

    for pk, log_probs in tqdm(results.items(), desc="Writing results"):
        
        result = _prepare_log_prob_result(factory.class_names, log_probs)
    
        # Write the output to the database.
        create_task_output(pk, astradb.Classification, **result)



def _prepare_log_prob_result(class_names, log_probs, decimals=3):

    # Make sure the log_probs are dtype float so that postgresql does not complain.
    log_probs = np.array(log_probs, dtype=float)

    # Calculate normalized probabilities.
    with np.errstate(under="ignore"):
        relative_log_probs = log_probs - logsumexp(log_probs)
    
    # Round for PostgreSQL 'real' type.
    # https://www.postgresql.org/docs/9.1/datatype-numeric.html
    # and
    # https://stackoverflow.com/questions/9556586/floating-point-numbers-of-python-float-and-postgresql-double-precision
    probs = np.round(np.exp(relative_log_probs), decimals)
    log_probs = np.round(log_probs, decimals)
    
    probs, log_probs = (probs.tolist(), log_probs.tolist())

    result = {}
    for i, class_name in enumerate(class_names):
        result[f"p_{class_name}"] = [probs[i]]
        result[f"lp_{class_name}"] = [log_probs[i]]

    return result





def classify_apstar(pks, **kwargs):
    """
    Classify observations of APOGEE (ApStar) sources, given the existing classifications of the
    individual visits.

    :param pks:
        The primary keys of task instances where visits have been classified. These primary keys will
        be used to work out which stars need classifying, before tasks
    """

    pks = deserialize_pks(pks, flatten=True)

    # Select distinct objects by `obj` and `apred`.
    distinct_sources = []
    dag_id, task_id = (None, None)
    for pk in pks:
        q = session.query(astradb.TaskInstance).filter(astradb.TaskInstance.pk == pk)
        instance = q.one_or_none()
        distinct_sources.append((instance.parameters["obj"], instance.parameters["apred"]))

        dag_id = dag_id or instance.dag_id
        task_id = task_id or instance.task_id
    
    distinct_sources = set(distinct_sources)
    N = len(distinct_sources)

    try:
        this_task_id = kwargs["task"].task_id
    except:
        this_task_id = "classify_apstar"
    
    try:
        run_id = kwargs["run_id"]
    except:
        print(kwargs)
        raise

    for i, (obj, apred) in enumerate(distinct_sources):
        
        log.info(f"Combining classifications for distinct source {i} / {N}: {obj} ({apred})")

        parameter_pks = [
            get_or_create_parameter_pk("obj", obj)[0],
            get_or_create_parameter_pk("apred", apred)[0]
        ]

        # Get all task instances matching this obj and apred.
        s1 = session.query(astradb.TaskInstanceParameter.ti_pk)\
                    .filter(astradb.TaskInstanceParameter.parameter_pk.in_(parameter_pks))\
                    .group_by(astradb.TaskInstanceParameter.ti_pk)\
                    .having(func.count(distinct(astradb.TaskInstanceParameter.parameter_pk)) == len(parameter_pks))\
                    .subquery()

        s2 = session.query(astradb.TaskInstance)\
                    .filter(astradb.TaskInstance.dag_id == dag_id)\
                    .filter(astradb.TaskInstance.task_id == task_id)\
                    .filter(astradb.TaskInstance.output_pk != None)\
                    .join(s1, astradb.TaskInstance.pk == s1.c.ti_pk)\
                    .subquery()
        
        s3 = session.query(func.max(astradb.TaskInstance.output_pk).label("latest"))\
                    .join(astradb.TaskInstanceParameter)\
                    .join(s2, astradb.TaskInstance.pk == s2.c.pk)\
                    .group_by(astradb.TaskInstance.dag_id, astradb.TaskInstance.run_id, astradb.TaskInstance.task_id, astradb.TaskInstanceParameter)\
                    .subquery()

        q = session.query(astradb.TaskInstance)\
                   .join(s3, astradb.TaskInstance.output_pk == s3.c.latest)\
                   .join(s2, astradb.TaskInstance.pk == s2.c.pk)

        # Get all these results.
        lps = {}
        for instance in q.all():
            if instance.output is None: continue
            for k, v in instance.output.__dict__.items():
                if k.startswith("lp_"):
                    lps.setdefault(k, 0)
                    lps[k] += np.sum(v)
        
        if not lps:
            log.warning(f"No results found for {obj} / {apred}!")
            continue

        class_names = [ea[3:] for ea in lps.keys()]
        log_probs = list(lps.values())
        result = _prepare_log_prob_result(class_names, log_probs)

        # NOTE: This won't be enough parameters to identify the apStar file, because we are combining information here
        #       from visits across multiple telescopes. This might not be what we want to do.
        parameters = dict(
            release=instance.parameters["release"],
            apred=apred,
            obj=obj,
            model_path=instance.parameters["model_path"],
        )

        instance = get_or_create_task_instance(
            dag_id=dag_id,
            task_id=this_task_id,
            run_id=run_id,
            parameters=parameters
        )

        create_task_output(
            instance,
            astradb.Classification,
            **result
        )

        log.debug(f"Created task {instance} with classification {result}")
    


def get_model_path(
        network_factory,
        training_spectra_path,
        training_labels_path,
        learning_rate,
        weight_decay,
        num_epochs,
        batch_size,
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
    
    :param num_epochs:
        the number of epochs to use during training
    
    :param batch_size:
        the batch size to use during training
    """
    kwds = dict()
    for arg in getargspec(get_model_path).args:
        kwds[arg] = locals()[arg]

    param_hash = hashify(kwds)
    
    basename = f"classifier_{network_factory}_{param_hash}.pt"
    path = os.path.join(get_base_output_path(), "classifier", basename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path



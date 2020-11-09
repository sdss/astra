
import pickle
import luigi
import numpy as np
from astropy.table import Table
from astra.tasks.base import BaseTask


class TheCannonMixin(BaseTask):

    """ 
    A mixin class all tasks related to The Cannon. 
    
    :param label_names:
        A list of label names.
    
    :param order: (optional)
        The polynomial order to use for this model (default: 2).
    """

    task_namespace = "TheCannon"

    # These parameters are needed for both training and testing.
    label_names = luigi.ListParameter(
        config_path=dict(section=task_namespace, name="label_names")
    )
    order = luigi.IntParameter(
        default=2,
        config_path=dict(section=task_namespace, name="order")
    )
    


def read_training_set(path, default_inverse_variance=1e6):
    """
    Read a training set from disk.

    The `path` should refer to a disk location that is a `pickle` file storing a dictionary
    that contains the following keys:

    - wavelength: a P-length array containing the wavelengths of P pixels
    - flux: a (N, P) shape array containing pseudo-continuum-normalised fluxes for N stars
    - ivar: a (N, P) shape array containing inverse variances for the pseudo-continuum-normalised fluxes for N stars
    - labels: a (N, L) shape array containing L label values for N stars
    - label_names: a L-length tuple containing the names of the labels

    :param path:
        The location of the training set on disk.
    
    :param default_inverse_variance: (optional)
        The default inverse variance value to set, if no inverse variances values are found.
    """
    # TODO: Betterize integrate this process with the data model specifications.
    with open(path, "rb") as fp:
        training_set = pickle.load(fp)

    dispersion = training_set["wavelength"]
    keys = ("flux", "spectra")
    for key in keys:
        try:
            training_set_flux = training_set[key]
        except KeyError:
            continue
            
        else:
            break
    
    else:
        raise KeyError("no flux specified in training set file")

    try:
        training_set_ivar = training_set["ivar"]
    except KeyError:
        training_set_ivar = np.ones_like(training_set_flux) * self.default_inverse_variance

    label_values = training_set["labels"]
    label_names = training_set["label_names"]

    # Create a table from the labels and their names.
    labels = Table(
        data=label_values.T,
        names=label_names[:label_values.shape[0]] # TODO HACK
    )

    return (labels, dispersion, training_set_flux, training_set_ivar)
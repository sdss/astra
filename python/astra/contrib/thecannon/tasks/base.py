
import pickle
import luigi
import numpy as np
from astropy.table import Table
from astra.tasks.base import BaseTask


class TheCannonMixin(BaseTask):

    # These parameters are needed for both training and testing.
    label_names = luigi.ListParameter()
    order = luigi.IntParameter(default=2)
    


def read_training_set(path, default_inverse_variance=1e6):
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
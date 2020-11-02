import luigi
import numpy as np
import os
import torch
import yaml

from sdss_access import SDSSPath
from tqdm import tqdm
from sqlalchemy import (Column, Float, String)

from luigi.util import (inherits, requires)
from luigi.parameter import ParameterVisibility
from torch.autograd import Variable
from scipy.special import logsumexp
from astra.tasks.io import (ApVisitFile, SDSS4ApVisitFile, LocalTargetTask)
from astra.tools.spectrum import Spectrum1D

from astra.tasks.base import BaseTask
from astra.tasks.targets import DatabaseTarget
from astra.contrib.classifier import networks, utils

from astra.contrib.classifier.tasks.mixin import ClassifierMixin
from astra.contrib.classifier.tasks.train import TrainNIRSpectrumClassifier

class ClassifySource(ClassifierMixin):

    """
    Classify a source.

    This should be sub-classed and mixed with data model class to inherit parameters.
    """

    def read_network(self):
        """
        Read the network from the input path.
        """
        try:
            self.network_factory

        except AttributeError:
            raise RuntimeError("network_factory should be set by the sub-classes")

        classifier, observation = self.input()
        network = utils.read_network(self.network_factory, classifier.path)
        
        # Disable dropout for inference.
        network.eval()
        return network


    def run(self):        
        network = self.read_network()
        raise NotImplementedError("this should be over-written by sub-classes")

    


class ClassifierResult(DatabaseTarget):

    """ A database target (row) indicating the result from the classifier. """

    # Output (unnormalised) log probabilities.
    lp_sb2 = Column("lp_sb2", Float)
    lp_yso = Column("lp_yso", Float)
    lp_fgkm = Column("lp_fgkm", Float)
    lp_hotstar = Column("lp_hotstar", Float)
    
    # Normalised probabilities
    prob_sb2 = Column("prob_sb2", Float)
    prob_yso = Column("prob_yso", Float)
    prob_fgkm = Column("prob_fgkm", Float)
    prob_hotstar = Column("prob_hotstar", Float)

    most_probable = Column("most_probable_class", String(10))



class ClassifySourceGivenApVisitFileBase(ClassifySource):

    """
    Classify the type of stellar source, given an ApVisitFile.

    This task requires the same parameters required by :py:mod:`astra.contrib.classifer.train.TrainNIRSpectrumClassifier`,
    and those required by :py:mod:`astra.tasks.io.ApVisitFile`.
    """

    network_factory = networks.NIRCNN

    def read_observation(self):
        """ Read the input observation from disk. """
        return Spectrum1D.read(self.input()[1].path, format="APOGEE apVisit")


    def prepare_batch(self):
        """ Prepare the input observation for being batched through the network. """
        spectrum = self.read_observation()

        # Normalise the same way the hard-coded training spectra have been.
        # TODO: Consider separating this normalisation if/when the training set for this classifier
        #       is ever updated.
        flux = spectrum.flux.value.reshape((1, 3, -1))
        return flux / np.nanmedian(flux, axis=2)[:, :, None]


    def prepare_result(self, log_probs):
        
        class_names = list(map(str.lower, self.class_names))

        # Unnormalised results.
        result = dict(zip(
            [f"lp_{class_name}" for class_name in class_names],
            log_probs
        ))

        # Normalised probabilities
        with np.errstate(under="ignore"):
            relative_log_probs = log_probs - logsumexp(log_probs)
        
        probs = np.exp(relative_log_probs)
        result.update(dict(zip(
            [f"prob_{class_name}" for class_name in class_names],
            probs
        )))
        
        # Most probable class.
        result["most_probable_class"] = class_names[np.argmax(probs)]
        return result


    def run(self):
        """
        Execute the task.
        """

        network = self.read_network()

        # This can be run in batch mode.
        for task in tqdm(self.get_batch_tasks(), total=self.get_batch_size()):

            batch = task.prepare_batch()

            with torch.no_grad():                
                pred = network.forward(Variable(torch.Tensor(batch)))
                log_probs = pred.data.numpy().flatten()

            result = self.prepare_result(log_probs)
            print(f"Result: {result}")

            task.output().write(result)

            #with open(task.output().path, "w") as fp:
            #    fp.write(yaml.dump(class_probs))

    """
    def output(self):
        if self.is_batch_mode:
            return [task.output() for task in self.get_batch_tasks()]
        
        path = os.path.join(
            self.output_base_dir,
            f"visit/{self.telescope}/{self.field}/{self.plate}/{self.mjd}",
            f"apVisit-{self.apred}-{self.plate}-{self.mjd}-{self.fiber:0>3}-{self.task_id}.yml"
        )
        # Create the directory structure if it does not exist already.
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return luigi.LocalTarget(path)
    """

    def output(self):
        return ClassifierResult(self)


@requires(TrainNIRSpectrumClassifier, ApVisitFile)
class ClassifySourceGivenApVisitFile(ClassifySourceGivenApVisitFileBase):

    """
    Classify the type of stellar source, given an ApVisitFile.

    This task requires the same parameters required by :py:mod:`astra.contrib.classifer.train.TrainNIRSpectrumClassifier`,
    and those required by :py:mod:`astra.tasks.io.ApVisitFile`.
    """
    
    def prepare_batch(self):
        spectrum = self.read_observation()

        # 2020-11-01: For some reason the SDSS5 ApVisit spectra have half as many pixels as SDSS4 ApVisit spectra.
        # TODO: Resolve this with Nidever and fix here.
        flux = np.repeat(spectrum.flux.value, 2).reshape((1, 3, -1))
        batch = flux / np.nanmedian(flux, axis=2)[:, :, None]
        return batch



@requires(TrainNIRSpectrumClassifier, SDSS4ApVisitFile)
class ClassifySourceGivenSDSS4ApVisitFile(ClassifySourceGivenApVisitFileBase):

    """
    Classify the type of stellar source, given an ApVisitFile.

    This task requires the same parameters required by :py:mod:`astra.contrib.classifer.train.TrainNIRSpectrumClassifier`,
    and those required by :py:mod:`astra.tasks.io.sdss4.ApVisitFile`.
    """
    pass





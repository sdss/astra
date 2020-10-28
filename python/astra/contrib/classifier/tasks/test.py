import luigi
import numpy as np
import os
import torch
import yaml

from sdss_access import SDSSPath
from tqdm import tqdm
from sqlalchemy import (Column, Float)

from luigi.util import (inherits, requires)
from luigi.parameter import ParameterVisibility
from torch.autograd import Variable
from scipy.special import logsumexp
from astra.tasks.io import ApVisitFile, LocalTargetTask
from astra.tools.spectrum import Spectrum1D

from astra.tasks.io import BaseTask
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

    

"""
class ClassifierResult(DatabaseTarget):
    results_schema = [
        Column("FGKM", Float()),
        Column("HotStar", Float()),
        Column("SB2", Float()),
        Column("YSO", Float())
    ]
"""

@requires(TrainNIRSpectrumClassifier, ApVisitFile)
class ClassifySourceGivenApVisitFile(ClassifySource):

    """
    Classify the type of stellar source, given an ApVisitFile.

    This task requires the same parameters required by :py:mod:`astra.contrib.classifer.train.TrainNIRSpectrumClassifier`,
    and those required by :py:mod:`astra.tasks.io.ApVisitFile`.
    """

    network_factory = networks.NIRCNN

    def run(self):
        """
        Execute the task.
        """

        network = self.read_network()
        class_names = self.class_names

        # This can be run in batch mode.
        for task in tqdm(self.get_batch_tasks(), total=self.get_batch_size()):
            
            spectrum = Spectrum1D.read(task.input()[1].path, format="APOGEE apVisit")

            # Normalise the same way the hard-coded training spectra have been.
            # TODO: Consider separating this normalisation if/when the training set for this classifier
            #       is ever updated.
            flux = spectrum.flux.value.reshape((1, 3, -1))
            batch = flux / np.nanmedian(flux, axis=2)[:, :, None]

            with torch.no_grad():                
                pred = network.forward(Variable(torch.Tensor(batch)))
                outputs = pred.data.numpy().flatten()

            with np.errstate(under="ignore"):
                log_probs = outputs - logsumexp(outputs)
            
            probs = np.exp(log_probs)
            class_probs = dict(zip(class_names, map(float, probs)))

            #task.output().write(class_probs)
            with open(task.output().path, "w") as fp:
                fp.write(yaml.dump(class_probs))

            most_probable_class = class_names[np.argmax(probs)]
            result = [most_probable_class, class_probs]
            print(f"Result: {result}")


    def output(self):
        return luigi.LocalTarget(f"{self.task_id}.yml")

    #def output(self):
    #    return ClassifierResult(self)
        





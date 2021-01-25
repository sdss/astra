import luigi
import numpy as np
import os
import torch
import yaml

from sdss_access import SDSSPath
from time import time
from tqdm import tqdm

from luigi.util import (inherits, requires)
from luigi.parameter import ParameterVisibility
from luigi.mock import MockTarget
from luigi.task import flatten
from torch.autograd import Variable
from scipy.special import logsumexp
from astra.tasks.io.sdss5 import (ApVisitFile, ApStarFile)
from astra.tasks.io.sdss4 import (ApVisitFile as SDSS4ApVisitFile, ApStarFile as SDSS4ApStarFile)
from astra.tools.spectrum import Spectrum1D
from astra.utils import (batcher, log, timer)

from astra.database import astradb
from astra.tasks import BaseTask
from astra.tasks.targets import DatabaseTarget
from astra.contrib.classifier import networks, utils

from astra.contrib.classifier.tasks.mixin import ClassifierMixin
from astra.contrib.classifier.tasks.train import TrainNIRSpectrumClassifier
from astra.tasks.slurm import slurmify



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

    
    
    def prepare_result(self, log_prob):
        
        # Make sure the log_probs are dtype float so that postgresql does not complain.
        log_prob = np.array(log_prob, dtype=float)

        # Calculate normalized probabilities.
        with np.errstate(under="ignore"):
            relative_log_prob = log_prob - logsumexp(log_prob)
        
        # Round for PostgreSQL 'real' type.
        # https://www.postgresql.org/docs/9.1/datatype-numeric.html
        # and
        # https://stackoverflow.com/questions/9556586/floating-point-numbers-of-python-float-and-postgresql-double-precision
        decimals = 36
        prob = np.round(np.exp(relative_log_prob), decimals)
        log_prob = np.round(log_prob, decimals)
        
        return dict(
            log_prob=log_prob,
            prob=prob,
        )
        

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

        # 2020-11-01: Undithered ApVisit spectra have half as many pixels as dithered spectra.
        #             This is a hack to make them work together. Consider doing something clever.
        flux = np.repeat(spectrum.flux.value, 2) if spectrum.flux.size == 6144 else spectrum.flux
        flux = flux.reshape((1, 3, -1))
        batch = flux / np.nanmedian(flux, axis=2)[:, :, None]
        return batch


    @slurmify
    def run(self):
        """
        Execute the task.
        """

        network = self.read_network()

        # This can be run in batch mode.
        for init, task in timer(tqdm(**self.get_tqdm_kwds("Classifying ApVisitFiles"))):
            if task.complete():
                continue
            
            batch = task.prepare_batch()

            with torch.no_grad():
                pred = network.forward(Variable(torch.Tensor(batch)))
                log_prob = pred.cpu().numpy().flatten()

            task.output()["database"].write(self.prepare_result(log_prob))

            task.trigger_event_processing_time(time() - init, cascade=True)


    def output(self):
        """ The output of the task. """
        if self.is_batch_mode:
            return (task.output() for task in self.get_batch_tasks())
        return dict(database=DatabaseTarget(astradb.Classification, self))


@requires(TrainNIRSpectrumClassifier, ApVisitFile)
class ClassifySourceGivenApVisitFile(ClassifySourceGivenApVisitFileBase):

    """
    Classify the type of stellar source, given an ApVisitFile.

    This task requires the same parameters required by :py:mod:`astra.contrib.classifer.train.TrainNIRSpectrumClassifier`,
    and those required by :py:mod:`astra.tasks.io.ApVisitFile`.
    """
    
    pass


class ClassifySourceGivenApStarFile(ClassifySource, ApStarFile):

    """
    Classify an ApStar source from the joint classifications of individual visits (the ApVisit files that went into that ApStar).

    This task requires the same parameters required by :py:mod:`astra.contrib.classifer.train.TrainNIRSpectrumClassifier`,
    and those required by :py:mod:`astra.tasks.io.ApStarFile`.
    """

    def requires(self):
        """ We require the classifications from individual visits of this source. """        

        # TODO: This should go elsewhere.
        from astra.tasks.daily import get_visits_given_star

        kwds = {}
        for i, task in enumerate(tqdm(**self.get_tqdm_kwds("Matching stars to visits"))):
            for k, v in batcher(get_visits_given_star(task.obj, task.apred)).items():
                if i == 0:
                    # Overwrite common keywords between ClassifySourceGivenApVisitFile and ClassifySourceGivenApStarFile
                    # that are batch keywords here.
                    kwds[k] = []
                kwds[k].extend(v)

        return self.clone(ClassifySourceGivenApVisitFile, **kwds)
        
    
    def run(self):
        """ Execute the task. """
        
        for init, task in timer(tqdm(**self.get_tqdm_kwds("Classifying ApStarFiles"))):
            if task.complete():
                continue

            log_prob = []
            for output in flatten(task.input()):
                try:
                    classification = output.read()
                    
                except TypeError:
                    log.exception(f"Exception occurred:")
                    continue

                else:
                    log_prob.append(classification.log_prob)
                
            log_prob = np.array(log_prob)
            finite_visit = np.all(np.isfinite(log_prob), axis=1)

            log_prob = np.sum(log_prob[finite_visit], axis=0)

            task.output()["database"].write(self.prepare_result(log_prob))
        
            task.trigger_event_processing_time(time() - init, cascade=True)

        return None


    def output(self):
        """ The output for these results. """
        if self.is_batch_mode:
            return (task.output() for task in self.get_batch_tasks())
        return dict(database=DatabaseTarget(astradb.Classification, self))
        

@requires(TrainNIRSpectrumClassifier, SDSS4ApVisitFile)
class ClassifySourceGivenSDSS4ApVisitFile(ClassifySourceGivenApVisitFileBase):

    """
    Classify the type of stellar source, given an ApVisitFile.

    This task requires the same parameters required by :py:mod:`astra.contrib.classifer.train.TrainNIRSpectrumClassifier`,
    and those required by :py:mod:`astra.tasks.io.sdss4.ApVisitFile`.
    """
    pass





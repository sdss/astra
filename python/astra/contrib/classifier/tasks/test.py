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
from luigi.mock import MockTarget
from torch.autograd import Variable
from scipy.special import logsumexp
from astra.tasks.io import (ApVisitFile, ApStarFile, SDSS4ApVisitFile, LocalTargetTask)
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
        if not np.any(np.isfinite(probs)):
            result["most_probable_class"] = "unknown"
        return result




class ClassifyApVisitResult(DatabaseTarget):

    """ A database target (row) indicating the result from the classifier. """

    table_name = "classify_apvisit"

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



class ClassifyApStarResult(DatabaseTarget):

    """ A database target (row) indicating the result from the classifier. """

    table_name = "classify_apstar"

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

        # 2020-11-01: Undithered ApVisit spectra have half as many pixels as dithered spectra.
        #             This is a hack to make them work together. Consider doing something clever.
        flux = np.repeat(spectrum.flux.value, 2) if spectrum.flux.size == 6144 else spectrum.flux
        flux = flux.reshape((1, 3, -1))
        batch = flux / np.nanmedian(flux, axis=2)[:, :, None]
        return batch


    def run(self):
        """
        Execute the task.
        """

        network = self.read_network()

        # This can be run in batch mode.
        for task in tqdm(self.get_batch_tasks(), desc="Classifying", total=self.get_batch_size()):

            batch = task.prepare_batch()

            with torch.no_grad():                
                pred = network.forward(Variable(torch.Tensor(batch)))
                log_probs = pred.data.numpy().flatten()

            task.output().write(self.prepare_result(log_probs))


    def output(self):
        """ The output of the task. """
        if self.is_batch_mode:
            return [task.output() for task in self.get_batch_tasks()]
        return ClassifyApVisitResult(self)


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

        common_kwds = self.get_common_param_kwargs(ClassifySourceGivenApVisitFile)    
        for task in tqdm(self.get_batch_tasks(), desc="Matching stars to visits", total=self.get_batch_size()):
            for visit_kwds in get_visits_given_star(task.obj, task.apred):
                yield ClassifySourceGivenApVisitFile(**{ **common_kwds, **visit_kwds })


    def run(self):
        """ Execute the task. """
        for task in tqdm(self.get_batch_tasks(), total=self.get_batch_size()):

            # We probably don't need to use dictionaries here but it prevents any mis-ordering.
            log_probs = { f"lp_{k}": 0 for k in task.class_names }
            for classification in task.requires():
                output = classification.output().read(as_dict=True)
                for key in log_probs.keys():
                    log_prob = output[key] or np.nan # Sometimes we get nans / Nones.
                    if np.isfinite(log_prob):
                        log_probs[key] += log_prob
        
            result = self.prepare_result([log_probs[f"lp_{k}"] for k in task.class_names])
            task.output().write(result)

        return None
    

    def output(self):
        """ The output for these results. """
        if self.is_batch_mode:
            return [task.output() for task in self.get_batch_tasks()]
        return ClassifyApStarResult(self)
        

@requires(TrainNIRSpectrumClassifier, SDSS4ApVisitFile)
class ClassifySourceGivenSDSS4ApVisitFile(ClassifySourceGivenApVisitFileBase):

    """
    Classify the type of stellar source, given an ApVisitFile.

    This task requires the same parameters required by :py:mod:`astra.contrib.classifer.train.TrainNIRSpectrumClassifier`,
    and those required by :py:mod:`astra.tasks.io.sdss4.ApVisitFile`.
    """
    pass





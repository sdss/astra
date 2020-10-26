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

            task.output().write(class_probs)

            most_probable_class = class_names[np.argmax(probs)]
            result = [most_probable_class, class_probs]
            print(f"Result: {result}")


    #def output(self):
    #    return ClassifierResult(self)
        




if __name__ == "__main__":

    from astropy.table import Table

    class ClassifyAllApVisitSpectra(luigi.WrapperTask, BaseTask):

        release = luigi.Parameter()
        apred = luigi.Parameter()
        use_remote = luigi.BoolParameter(
            default=False, 
            significant=False,
            visibility=ParameterVisibility.HIDDEN
        )
        
        def requires(self):
            all_visit_sum = LocalTargetTask(
                path=SDSSPath(release=self.release).full("allVisitSum", apred=self.apred)
            )
            yield all_visit_sum

            # Load in all the apVisit files.
            table = Table.read(all_visit_sum.path)
            
            # Get the keywords we need.
            kwds = self.get_common_param_kwargs(ClassifySourceGivenApVisitFile)
            for key in ("apred", "telescope", "field", "plate", "mjd", "prefix", "fiber"):
                kwds[key] = []

            for i, row in enumerate(table):
                kwds["apred"].append(self.apred)
                kwds["telescope"].append(row["TELESCOPE"])
                kwds["field"].append(row["FIELD"].lstrip())
                kwds["plate"].append(row["PLATE"].lstrip())
                kwds["mjd"].append(str(row["MJD"]))
                kwds["prefix"].append(row["FILE"][:2])
                kwds["fiber"].append(str(row["FIBERID"]))
            
            yield ClassifySourceGivenApVisitFile(**kwds)
            
        
        def on_success(self):
            # Overwrite the inherited method that will mark this wrapper task as done and never re-run it.
            pass

        def output(self):
            return None

    task = ClassifyAllApVisitSpectra(
        release="dr16",
        apred="r12",
        use_remote=True
    )

    task.run()

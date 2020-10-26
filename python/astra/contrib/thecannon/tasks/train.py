
import os
import luigi
import numpy as np
from astropy.table import Table
from astra.tasks.base import BaseTask
from astra.utils import log
from astra.contrib.thecannon.tasks.base import TheCannonMixin, read_training_set

import astra.contrib.thecannon as tc
from astra.contrib.thecannon import plot


class TrainTheCannonBase(TheCannonMixin):

    regularization = luigi.FloatParameter(default=0.0)
    threads = luigi.IntParameter(default=1, significant=False)
    plot = luigi.BoolParameter(default=True, significant=False)

    def run(self):

        # Load training set labels and spectra.
        labels, dispersion, training_set_flux, training_set_ivar = read_training_set(
            self.input().path, 
        )

        # Set the vectorizer.
        # We sort the label names so that luigi doesn't re-train models if we alter the order.
        vectorizer = tc.vectorizer.PolynomialVectorizer(
            sorted(self.label_names),
            self.order
        )

        # Initiate model.
        model = tc.model.CannonModel(
            labels,
            training_set_flux,
            training_set_ivar,
            vectorizer=vectorizer,
            dispersion=dispersion,
            regularization=self.regularization
        )
    
        log.info(f"Training The Cannon model {model}")
        model.train(threads=self.threads)

        output_path = self.output().path
        log.info(f"Writing The Cannon model {model} to disk {output_path}")
        model.write(output_path)    

        if self.plot:
            # Plot zeroth and first order coefficients.
            fig = plot.theta(
                model,
                indices=np.arange(1 + len(model.vectorizer.label_names)),
                normalize=False
            )
            fig.savefig(f"{self.task_id}-theta.png")

            # Plot scatter.
            fig = plot.scatter(model)
            fig.savefig(f"{self.task_id}-scatter.png")

            # Plot one-to-one.
            test_labels, test_cov, test_meta = model.test(
                training_set_flux, 
                training_set_ivar,
                initial_labels=model.training_set_labels
            )
            fig = plot.one_to_one(model, test_labels, cov=test_cov)
            fig.savefig(f"{self.task_id}-one-to-one.png")

    def output(self):
        return luigi.LocalTarget(f"{self.task_id}.pkl")




class TrainingSetTarget(BaseTask):

    training_set_path = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.training_set_path)



class TrainTheCannon(TrainTheCannonBase):

    """
    A task to train The Cannon, given some file that contains high-quality labels,
    and pseudo-continuum-normalised fluxes and inverse variances.    

    :param training_set_path: The path to a `pickle` file that contains a dictionary
        with the following keys:
            
        - `wavelength`: an array of shape `(P, )` where `P` is the number of pixels
        - `flux`: an array of flux values with shape `(N, P)` where `N` is the number of observed spectra and `P` is the number of pixels
        - `ivar`: an array of inverse variance values with shape `(N, P)` where `N` is the number of observed spectra and `P` is the number of pixels
        - `labels`: an array of shape `(L, N)` where `L` is the number of labels and `N` is the number observed spectra
        - `label_names`: a tuple of length `L` that describes the names of the labels

    :param regularization: (optional)
        The L1 regularization strength to use during training (default: 0.0).
    
    :param threads: (optional)
        The number of threads to use during training (default: 1).
    
    :param plot: (optional)
        Produce quality assurance figures after training (default: True).
    """



    training_set_path = luigi.Parameter()

    def requires(self):
        return TrainingSetTarget(training_set_path=self.training_set_path)

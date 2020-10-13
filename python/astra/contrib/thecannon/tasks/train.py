
import os
import luigi
import numpy as np
from astropy.table import Table
from astra.tasks.base import BaseTask
from astra.utils import log
from astra.contrib.thecannon.tasks.base import TheCannonMixin, read_training_set

import astra.contrib.thecannon as tc



class TrainTheCannonBase(TheCannonMixin):

    regularization = luigi.FloatParameter(default=0.0)
    threads = luigi.IntParameter(default=1, significant=False)
    default_inverse_variance = luigi.FloatParameter(default=1.0e6, significant=False)
    plot = luigi.BoolParameter(default=True, significant=False)

    def run(self):

        # Load training set labels and spectra.
        labels, dispersion, training_set_flux, training_set_ivar = read_training_set(
            self.input().path, 
            self.default_inverse_variance
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
            fig = tc.plot.theta(
                model,
                indices=np.arange(1 + len(model.vectorizer.label_names)),
                normalize=False
            )
            fig.savefig(f"{self.task_id}-theta.png")

            # Plot scatter.
            fig = tc.plot.scatter(model)
            fig.savefig(f"{self.task_id}-scatter.png")

            # Plot one-to-one.
            test_labels, test_cov, test_meta = model.test(
                training_set_flux, 
                training_set_ivar,
                initial_labels=model.training_set_labels
            )
            fig = tc.plot.one_to_one(model, test_labels, cov=test_cov)
            fig.savefig(f"{self.task_id}-one-to-one.png")

    def output(self):
        return luigi.LocalTarget(f"{self.task_id}.pkl")




class TrainingSetTarget(BaseTask):

    training_set_path = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.training_set_path)



class TrainTheCannon(TrainTheCannonBase):
    training_set_path = luigi.Parameter()

    def requires(self):
        return TrainingSetTarget(training_set_path=self.training_set_path)

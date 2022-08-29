import astra
import numpy as np
import os
import pickle
from tqdm import tqdm
from time import time
from astropy.io import fits
from astropy.table import Table
from astra.utils import log, timer
from astra.tasks import BaseTask
from astra.tasks.io import LocalTargetTask
from astra.tasks.io.sdss5 import ApStarFile
from astra.tasks.io.sdss4 import SDSS4ApStarFile
from astra.tasks.targets import DatabaseTarget, LocalTarget, AstraSource
from astra.tasks.continuum import Sinusoidal
from astra.tools.spectrum import Spectrum1D
from astra.tools.spectrum.writers import create_astra_source
from astra.contrib.thepayne import training, test as testing
from astra.tasks.slurm import slurm_mixin_factory, slurmify
from astra.database import astradb
from luigi.parameter import BoolParameter, IntParameter, FloatParameter, Parameter

SlurmMixin = slurm_mixin_factory("ThePayne")


class ThePayneMixin(SlurmMixin, BaseTask):

    task_namespace = "ThePayne"

    n_steps = IntParameter(
        default=100000, config_path=dict(section=task_namespace, name="n_steps")
    )
    n_neurons = IntParameter(
        default=300, config_path=dict(section=task_namespace, name="n_neurons")
    )
    weight_decay = FloatParameter(
        default=0.0, config_path=dict(section=task_namespace, name="weight_decay")
    )
    learning_rate = FloatParameter(
        default=0.001, config_path=dict(section=task_namespace, name="learning_rate")
    )
    training_set_path = Parameter(
        config_path=dict(section=task_namespace, name="training_set_path")
    )


class TrainThePayne(ThePayneMixin):

    """
    Train a single-layer neural network given a pre-computed grid of synthetic spectra.

    :param training_set_path:
        The path where the training set spectra and labels are stored.
        This should be a binary pickle file that contains a dictionary with the following keys:

        - wavelength: an array of shape (P, ) where P is the number of pixels
        - spectra: an array of shape (N, P) where N is the number of spectra and P is the number of pixels
        - labels: an array of shape (L, P) where L is the number of labels and P is the number of pixels
        - label_names: a tuple of length L that contains the names of the labels

    :param n_steps: (optional)
        The number of steps to train the network for (default 100000).

    :param n_neurons: (optional)
        The number of neurons to use in the hidden layer (default: 300).

    :param weight_decay: (optional)
        The weight decay to use during training (default: 0)

    :param learning_rate: (optional)
        The learning rate to use during training (default: 0.001).
    """

    def requires(self):
        """The requirements of this task."""
        return LocalTargetTask(path=self.training_set_path)

    @slurmify
    def run(self):
        """Execute this task."""

        (
            wavelength,
            label_names,
            training_labels,
            training_spectra,
            validation_labels,
            validation_spectra,
        ) = training.load_training_data(self.input().path)

        state, model, optimizer = training.train(
            training_spectra,
            training_labels,
            validation_spectra,
            validation_labels,
            label_names,
            n_neurons=self.n_neurons,
            n_steps=self.n_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        with open(self.output().path, "wb") as fp:
            pickle.dump(
                dict(
                    state=state,
                    wavelength=wavelength,
                    label_names=label_names,
                ),
                fp,
            )

    def output(self):
        """The output of this task."""
        path = os.path.join(self.output_base_dir, f"{self.task_id}.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        return LocalTarget(path)


class EstimateStellarLabels(ThePayneMixin):

    """
    Use a pre-trained neural network to estimate stellar labels. This should be sub-classed to inherit properties from the type of spectra to be analysed.

    :param training_set_path:
        The path where the training set spectra and labels are stored.
        This should be a binary pickle file that contains a dictionary with the following keys:

        - wavelength: an array of shape (P, ) where P is the number of pixels
        - spectra: an array of shape (N, P) where N is the number of spectra and P is the number of pixels
        - labels: an array of shape (L, P) where L is the number of labels and P is the number of pixels
        - label_names: a tuple of length L that contains the names of the labels

    :param n_steps: (optional)
        The number of steps to train the network for (default 100000).

    :param n_neurons: (optional)
        The number of neurons to use in the hidden layer (default: 300).

    :param weight_decay: (optional)
        The weight decay to use during training (default: 0)

    :param learning_rate: (optional)
        The learning rate to use during training (default: 0.001).
    """

    max_batch_size = 10_000
    analyze_individual_visits = BoolParameter(default=False)

    def prepare_observation(self):
        """Prepare the observations for analysis."""

        data_slice = None if self.analyze_individual_visits else [0, 1]
        observation = Spectrum1D.read(
            self.input()["observation"].path, data_slice=slice(*data_slice)
        )

        if "continuum" in self.input():
            continuum_path = self.input()["continuum"]["continuum"].path
            while True:
                with open(continuum_path, "rb") as fp:
                    continuum = pickle.load(fp)

                # If there is a shape mis-match between the observations and the continuum
                # then it likely means that there have been observations taken since the
                # continuum task was run. In this case we need to re-run the continuum
                # normalisation.

                # log.debug(f"Continuum for {self} original shape {continuum.shape}")
                if self.analyze_individual_visits is not None:
                    continuum = continuum[slice(*data_slice)]

                # log.debug(f"New shapes {observation.flux.shape} {continuum.shape}")

                O = observation.flux.shape[0]
                C = continuum.shape[0]

                # TODO: Consider if this is what we want to be doing..
                if O == C:
                    break

                else:
                    if O > C:
                        log.warn(f"Re-doing continuum for task {self} at runtime")
                    else:
                        log.warn(f"More continuum than observations in {self}?!")

                    os.unlink(continuum_path)
                    self.requires()["continuum"].run()
        else:
            continuum = 1

        normalized_flux = observation.flux.value / continuum
        normalized_ivar = continuum * observation.uncertainty.array * continuum

        return (observation, continuum, normalized_flux, normalized_ivar)

    @slurmify
    def run(self):
        """Execute this task."""

        # Load the model.
        log.info(f"Loading model for {self}")
        state = testing.load_state(self.input()["model"].path)

        # We can run this in batch mode.
        label_names = state["label_names"]
        tqdm_kwds = dict(total=self.get_batch_size(), desc="The Payne")
        for init, task in tqdm(timer(self.get_batch_tasks()), **tqdm_kwds):
            if task.complete():
                continue

            # log.debug(f"Running {task}")
            (
                spectrum,
                continuum,
                normalized_flux,
                normalized_ivar,
            ) = task.prepare_observation()

            # log.debug(f"Prepared observations for {task}")

            p_opt, p_cov, model_flux, meta = testing.test(
                spectrum.wavelength.value, normalized_flux, normalized_ivar, **state
            )

            # log.debug(f"Completed inference on {task}. p_opt has shape {p_opt.shape}")

            results = dict(zip(label_names, p_opt.T))
            # Note: we count the number of label names here in case we are sometimes using
            #       radial velocity determination or not, before we add in the SNR.

            L = len(results)
            # Add in uncertainties on parameters.
            results.update(
                dict(
                    zip(
                        (f"u_{ln}" for ln in label_names),
                        np.sqrt(p_cov[:, np.arange(L), np.arange(L)].T),
                    )
                )
            )

            # Add in SNR values for conveninence.
            results.update(snr=spectrum.meta["snr"])

            # Write AstraSource object.
            if "AstraSource" in task.output():
                # log.debug(f"Writing AstraSource object for {task}")
                task.output()["AstraSource"].write(
                    spectrum=spectrum,
                    normalized_flux=normalized_flux,
                    normalized_ivar=normalized_ivar,
                    continuum=continuum,
                    model_flux=model_flux,
                    # TODO: Project uncertainties to flux space.
                    model_ivar=None,
                    results_table=Table(results),
                )

            # Write output to database.
            if "database" in task.output():
                # log.debug(f"Writing database output for {task}")
                task.output()["database"].write(results)

            # Trigger this event as complete, and record task duration.
            task.trigger_event_processing_time(time() - init, cascade=True)

        return None

    def output(self):
        """The output of this task."""
        if self.is_batch_mode:
            return (task.output() for task in self.get_batch_tasks())

        return dict(
            database=DatabaseTarget(astradb.ThePayne, self),
            # AstraSource=AstraSource(self)
        )


class ContinuumNormalizeGivenApStarFile(Sinusoidal, ApStarFile):

    """Pseudo-continuum normalise ApStar spectra using a sum of sines and cosines."""

    def requires(self):
        return self.clone(ApStarFile)

    def output(self):
        if self.is_batch_mode:
            return (task.output() for task in self.get_batch_tasks())

        # TODO: Re-factor to allow for SDSS-IV.
        path = os.path.join(
            self.output_base_dir,
            f"star/{self.telescope}/{int(self.healpix/1000)}/{self.healpix}/",
            f"Continuum-{self.apred}-{self.obj}-{self.task_id}.pkl",
        )
        # Create the directory structure if it does not exist already.
        os.makedirs(os.path.dirname(path), exist_ok=True)

        return dict(continuum=LocalTarget(path))


class ContinuumNormalizeGivenSDSS4ApStarFile(Sinusoidal, SDSS4ApStarFile):

    """Pseudo-continuum normalise SDSS-IV ApStar spectra using a sum of sines and cosines."""

    def requires(self):
        return self.clone(SDSS4ApStarFile)

    def output(self):
        if self.is_batch_mode:
            return (task.output() for task in self.get_batch_tasks())

        # TODO: What is the path system for SDSS-IV products?
        path = os.path.join(
            self.output_base_dir,
            f"sdss4/{self.release}/{self.apred}/{self.telescope}/{self.field}/",
            f"Continuum-{self.release}-{self.apred}-{self.telescope}-{self.obj}-{self.task_id}.pkl",
        )

        # Create directory structure if it does not exist already.
        os.makedirs(os.path.dirname(path), exist_ok=True)

        return dict(continuum=LocalTarget(path))


class EstimateStellarLabelsGivenApStarFile(
    EstimateStellarLabels, Sinusoidal, ApStarFile
):
    """
    Estimate stellar labels given a single-layer neural network and an ApStar file.

    This task also requires all parameters that `astra.tasks.io.sdss5.ApStarFile` requires,
    and that the `astra.tasks.continuum.Sinusoidal` task requires.

    :param training_set_path:
        The path where the training set spectra and labels are stored.
        This should be a binary pickle file that contains a dictionary with the following keys:

        - wavelength: an array of shape (P, ) where P is the number of pixels
        - spectra: an array of shape (N, P) where N is the number of spectra and P is the number of pixels
        - labels: an array of shape (L, P) where L is the number of labels and P is the number of pixels
        - label_names: a tuple of length L that contains the names of the labels

    :param n_steps: (optional)
        The number of steps to train the network for (default: 100000).

    :param n_neurons: (optional)
        The number of neurons to use in the hidden layer (default: 300).

    :param weight_decay: (optional)
        The weight decay to use during training (default: 0)

    :param learning_rate: (optional)
        The learning rate to use during training (default: 0.001).

    :param continuum_regions_path:
        A path containing a list of (start, end) wavelength values that represent the regions to
        fit as continuum.
    """

    max_batch_size = 10_000

    def requires(self):
        return dict(
            model=self.clone(TrainThePayne),
            observation=self.clone(ApStarFile),
            continuum=self.clone(ContinuumNormalizeGivenApStarFile),
        )


class EstimateStellarLabelsGivenSDSS4ApStarFile(
    EstimateStellarLabels, Sinusoidal, SDSS4ApStarFile
):
    """
    Estimate stellar labels given a single-layer neural network and a SDSS-IV ApStar file.

    This task also requires all parameters that `astra.tasks.io.sdss4.ApStarFile` requires,
    and that the `astra.tasks.continuum.Sinusoidal` task requires.

    :param training_set_path:
        The path where the training set spectra and labels are stored.
        This should be a binary pickle file that contains a dictionary with the following keys:

        - wavelength: an array of shape (P, ) where P is the number of pixels
        - spectra: an array of shape (N, P) where N is the number of spectra and P is the number of pixels
        - labels: an array of shape (L, P) where L is the number of labels and P is the number of pixels
        - label_names: a tuple of length L that contains the names of the labels

    :param n_steps: (optional)
        The number of steps to train the network for (default: 100000).

    :param n_neurons: (optional)
        The number of neurons to use in the hidden layer (default: 300).

    :param weight_decay: (optional)
        The weight decay to use during training (default: 0)

    :param learning_rate: (optional)
        The learning rate to use during training (default: 0.001).

    :param continuum_regions_path:
        A path containing a list of (start, end) wavelength values that represent the regions to
        fit as continuum.
    """

    max_batch_size = 10_000

    def requires(self):
        return dict(
            model=self.clone(TrainThePayne),
            observation=self.clone(SDSS4ApStarFile),
            continuum=self.clone(ContinuumNormalizeGivenSDSS4ApStarFile),
        )

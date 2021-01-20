
import astra
import numpy as np
import os
import pickle
from tqdm import tqdm
from astropy.io import fits
from astropy.table import Table
from astra.utils import log
from astra.tasks import BaseTask
from astra.tasks.io import LocalTargetTask
from astra.tasks.io.sdss5 import ApStarFile
from astra.tasks.targets import (DatabaseTarget, LocalTarget, AstraSource)
from astra.tasks.continuum import Sinusoidal
from astra.tools.spectrum import Spectrum1D
from astra.tools.spectrum.writers import create_astra_source
from astra.contrib.thepayne import training, test as testing
from astra.tasks.slurm import (slurm_mixin_factory, slurmify)
from astra import database
from sqlalchemy import (ARRAY as Array, Column, Float, Integer)


SlurmMixin = slurm_mixin_factory("ThePayne")

class ThePayneMixin(SlurmMixin, BaseTask):

    task_namespace = "ThePayne"

    n_steps = astra.IntParameter(
        default=100000,
        config_path=dict(section=task_namespace, name="n_steps")
    )
    n_neurons = astra.IntParameter(
        default=300,
        config_path=dict(section=task_namespace, name="n_neurons")
    )
    weight_decay = astra.FloatParameter(
        default=0.0,
        config_path=dict(section=task_namespace, name="weight_decay")
    )
    learning_rate = astra.FloatParameter(
        default=0.001,
        config_path=dict(section=task_namespace, name="learning_rate")
    )
    training_set_path = astra.Parameter(
        config_path=dict(section=task_namespace, name="training_set_path")
    )


class ThePayneResult(DatabaseTarget):

    table_name = "thepayne_apstar"

    """ A database row to represent an output. """

    snr = Column("snr", Array(Float))

    teff = Column('teff', Array(Float))
    logg = Column('logg', Array(Float))
    v_turb = Column('v_turb', Array(Float))
    c_h = Column('c_h', Array(Float))
    n_h = Column('n_h', Array(Float))
    o_h = Column('o_h', Array(Float))
    na_h = Column('na_h', Array(Float))
    mg_h = Column('mg_h', Array(Float))
    al_h = Column('al_h', Array(Float))
    si_h = Column('si_h', Array(Float))
    p_h = Column('p_h', Array(Float))
    s_h = Column('s_h', Array(Float))
    k_h = Column('k_h', Array(Float))
    ca_h = Column('ca_h', Array(Float))
    ti_h = Column('ti_h', Array(Float))
    v_h = Column('v_h', Array(Float))
    cr_h = Column('cr_h', Array(Float))
    mn_h = Column('mn_h', Array(Float))
    fe_h = Column('fe_h', Array(Float))
    co_h = Column('co_h', Array(Float))
    ni_h = Column('ni_h', Array(Float))
    cu_h = Column('cu_h', Array(Float))
    ge_h = Column('ge_h', Array(Float))
    c12_c13 = Column('c12_c13', Array(Float))
    v_macro = Column('v_macro', Array(Float))

    u_teff = Column('u_teff', Array(Float))
    u_logg = Column('u_logg', Array(Float))
    u_v_turb = Column('u_v_turb', Array(Float))
    u_c_h = Column('u_c_h', Array(Float))
    u_n_h = Column('u_n_h', Array(Float))
    u_o_h = Column('u_o_h', Array(Float))
    u_na_h = Column('u_na_h', Array(Float))
    u_mg_h = Column('u_mg_h', Array(Float))
    u_al_h = Column('u_al_h', Array(Float))
    u_si_h = Column('u_si_h', Array(Float))
    u_p_h = Column('u_p_h', Array(Float))
    u_s_h = Column('u_s_h', Array(Float))
    u_k_h = Column('u_k_h', Array(Float))
    u_ca_h = Column('u_ca_h', Array(Float))
    u_ti_h = Column('u_ti_h', Array(Float))
    u_v_h = Column('u_v_h', Array(Float))
    u_cr_h = Column('u_cr_h', Array(Float))
    u_mn_h = Column('u_mn_h', Array(Float))
    u_fe_h = Column('u_fe_h', Array(Float))
    u_co_h = Column('u_co_h', Array(Float))
    u_ni_h = Column('u_ni_h', Array(Float))
    u_cu_h = Column('u_cu_h', Array(Float))
    u_ge_h = Column('u_ge_h', Array(Float))
    u_c12_c13 = Column('u_c12_c13', Array(Float))
    u_v_macro = Column('u_v_macro', Array(Float))




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
        """ The requirements of this task."""
        return LocalTargetTask(path=self.training_set_path)

    @slurmify
    def run(self):
        """ Execute this task. """

        wavelength, label_names, \
            training_labels, training_spectra, \
            validation_labels, validation_spectra = training.load_training_data(self.input().path)
    
        state, model, optimizer = training.train(
            training_spectra, 
            training_labels,
            validation_spectra,
            validation_labels,
            label_names,
            n_neurons=self.n_neurons,
            n_steps=self.n_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay
        )

        with open(self.output().path, "wb") as fp:
            pickle.dump(dict(
                    state=state,
                    wavelength=wavelength,
                    label_names=label_names, 
                ),
                fp
            )

    def output(self):
        """ The output of this task. """
        path = os.path.join(
            self.output_base_dir,
            f"{self.task_id}.pkl"
        )
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

    def prepare_observation(self):
        """
        Prepare the observations for analysis.
        """
        observation = Spectrum1D.read(self.input()["observation"].path)

        if "continuum" in self.input():

            continuum_path = self.input()["continuum"]["continuum"].path
            while True:
                with open(continuum_path, "rb") as fp:
                    continuum = pickle.load(fp)

                # If there is a shape mis-match between the observations and the continuum
                # then it likely means that there have been observations taken since the
                # continuum task was run. In this case we need to re-run the continuum
                # normalisation.

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
        """ Execute this task. """

        # Load the model.
        state = testing.load_state(self.input()["model"].path)

        # We can run this in batch mode.
        label_names = state["label_names"]
        for task in tqdm(self.get_batch_tasks(), total=self.get_batch_size()):
            if task.complete():
                continue
                
            spectrum, continuum, normalized_flux, normalized_ivar = task.prepare_observation()
            
            p_opt, p_cov, model_flux, meta = testing.test(
                spectrum.wavelength.value,
                normalized_flux,
                normalized_ivar,
                **state
            )

            results = dict(zip(label_names, p_opt.T))
            # Note: we count the number of label names here in case we are sometimes using
            #       radial velocity determination or not, before we add in the SNR.

            L = len(results)
            # Add in uncertainties on parameters.
            results.update(dict(zip(
                (f"u_{ln}" for ln in label_names),
                np.sqrt(p_cov[:, np.arange(L), np.arange(L)].T)
            )))

            # Add in SNR values for conveninence.
            results.update(snr=spectrum.meta["snr"])

            # Write output to database.
            task.output()["database"].write(results)
            
            # Write AstraSource object.
            task.output()["AstraSource"].write(
                spectrum=spectrum,
                normalized_flux=normalized_flux,
                normalized_ivar=normalized_ivar,
                continuum=continuum,
                model_flux=model_flux,
                # TODO: Project uncertainties to flux space.
                model_ivar=None,
                results_table=Table(results)
            )

        return None


    def output(self):
        """ The output of this task. """
        if self.is_batch_mode:
            return [task.output() for task in self.get_batch_tasks()]
        
        return {
            "database": ThePayneResult(self),
            "AstraSource": AstraSource(self)
        }
        


class ContinuumNormalizeApStarResult(DatabaseTarget):

    """ A database row to represent an output. """

    table_name = "continuum_apstar_sinusoidal"

    nvisits = Column("nvisits", Integer)


class ContinuumNormalize(Sinusoidal, ApStarFile):

    """
    Pseudo-continuum normalise ApStar stacked spectra using a sum of sines and cosines. 
    
    :param continuum_regions_path:
        A path containing a list of (start, end) wavelength values that represent the regions to
        fit as continuum.
    """
    
    def requires(self):
        return self.clone(ApStarFile)
    

    def complete(self):
        if self.is_batch_mode:
            return all(task.complete() for task in self.get_batch_tasks())
        
        if self.output()["continuum"].exists():
            # Some continuum has been performed previously
            # Check whether there are new visits that we need to account for.
            result = self.output()["database"].read(as_dict=True)
        
            # Nicely handle things that we have already determined continuum for,
            # but we don't know how many nvisits were there
            if not result:
                return False

            continuum_nvisits = result["nvisits"]

            # TODO: There are mis-matches between the number of visits that exist in the
            #       database, and the number that have gone into ApStar files.
            #       As of 2020/01/11 an example of this is 2M02202229+7135598
            #       where it has ngoodvisits = 3 in the database, but only 2 visits went
            #       into the construction of the ApStar file.

            #       Until this is fixed we can't rely on the database for what information
            #       is correct, so we will revert to checking the actual ApStar files.
            fixed = False
            if fixed:
                            
                # If a visit is considered "bad" in the APOGEE_DRP, then it does not get
                # included in the final ApStar file.
                star_table = database.schema.apogee_drp.star
                visit_column = star_table.ngoodvisits

                # TODO: Ask Nidever to make the data model and SQL tables consistent so
                #       that we don't have to hard code in hacks.
                actual_nvisits, = database.session.query(visit_column).filter(
                    star_table.apogee_id == self.obj,
                    star_table.apred_vers == self.apred,
                    star_table.telescope == self.telescope,
                    star_table.healpix == self.healpix,
                ).order_by(visit_column.desc()).first()


            else:
                with fits.open(self.input().path) as image:
                    N, P = image[1].data.shape
                actual_nvisits = N if N < 2 else N - 2

            if continuum_nvisits != actual_nvisits:
                # TODO: Remove when fixed.
                print(f"Mis-match with {self.obj} / {self.apred} / {self.telescope} / {self.healpix}: {continuum_nvisits} != {actual_nvisits}")

            return continuum_nvisits == actual_nvisits

        else:
            return False        



    def output(self):
        if self.is_batch_mode:
            return [task.output() for task in self.get_batch_tasks()]
        
        # TODO: Re-factor to allow for SDSS-IV.
        path = os.path.join(
            self.output_base_dir,
            f"star/{self.telescope}/{int(self.healpix/1000)}/{self.healpix}/",
            f"Continuum-{self.apred}-{self.obj}-{self.task_id}.pkl"
        )
        # Create the directory structure if it does not exist already.
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        return {
            "database": ContinuumNormalizeApStarResult(self),
            "continuum": LocalTarget(path)
        }




class OldContinuumNormalize(Sinusoidal, ApStarFile):

    """
    Pseudo-continuum normalise ApStar stacked spectra using a sum of sines and cosines. 
    
    :param continuum_regions_path:
        A path containing a list of (start, end) wavelength values that represent the regions to
        fit as continuum.
    """

    def requires(self):
        return self.clone(ApStarFile)


    def output(self):
        if self.is_batch_mode:
            return [task.output() for task in self.get_batch_tasks()]
        
        # TODO: Move this to AstraSource?
        # TODO: Assuming this is only running on SDSS-IV spectra.
        path = os.path.join(
            self.output_base_dir,
            f"star/{self.telescope}/{int(self.healpix/1000)}/{self.healpix}/",
            f"Continuum-{self.apred}-{self.obj}-{self.task_id}.pkl"
        )
        # Create the directory structure if it does not exist already.
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return LocalTarget(path)



class EstimateStellarLabelsGivenApStarFile(EstimateStellarLabels, Sinusoidal, ApStarFile):
    """
    Estimate stellar labels given a single-layer neural network and an ApStar file.

    This task also requires all parameters that `astra.tasks.io.ApStarFile` requires,
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

    max_batch_size = 1_000

    @property
    def task_short_id(self):
        # Since this is the primary task for The Payne, let's give it a short ID.
        return f"{self.task_namespace}-{self.task_id.split('_')[-1]}"
    

    def requires(self):
        return {
            "model": self.clone(TrainThePayne),
            "observation": self.clone(ApStarFile),
            "continuum": self.clone(ContinuumNormalize)
        }

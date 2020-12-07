
import astra
import numpy as np
import os
import pickle
from tqdm import tqdm
from astropy.table import Table
from astra.tasks.base import BaseTask
from astra.tasks.io import (ApStarFile, LocalTargetTask)
from astra.tasks.targets import (DatabaseTarget, LocalTarget, AstraSource)
from astra.tasks.continuum import Sinusoidal
from astra.tools.spectrum import Spectrum1D
from astra.tools.spectrum.writers import create_astra_source
from astra.contrib.thepayne import training, test as testing

from sqlalchemy import (Column, Float)


class ThePayneMixin(BaseTask):

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

    """ A database row to represent an output. """

    teff = Column('teff', Float)
    logg = Column('logg', Float)
    v_turb = Column('v_turb', Float)
    c_h = Column('c_h', Float)
    n_h = Column('n_h', Float)
    o_h = Column('o_h', Float)
    na_h = Column('na_h', Float)
    mg_h = Column('mg_h', Float)
    al_h = Column('al_h', Float)
    si_h = Column('si_h', Float)
    p_h = Column('p_h', Float)
    s_h = Column('s_h', Float)
    k_h = Column('k_h', Float)
    ca_h = Column('ca_h', Float)
    ti_h = Column('ti_h', Float)
    v_h = Column('v_h', Float)
    cr_h = Column('cr_h', Float)
    mn_h = Column('mn_h', Float)
    fe_h = Column('fe_h', Float)
    co_h = Column('co_h', Float)
    ni_h = Column('ni_h', Float)
    cu_h = Column('cu_h', Float)
    ge_h = Column('ge_h', Float)
    c12_c13 = Column('c12_c13', Float)
    v_macro = Column('v_macro', Float)

    u_teff = Column('u_teff', Float)
    u_logg = Column('u_logg', Float)
    u_v_turb = Column('u_v_turb', Float)
    u_c_h = Column('u_c_h', Float)
    u_n_h = Column('u_n_h', Float)
    u_o_h = Column('u_o_h', Float)
    u_na_h = Column('u_na_h', Float)
    u_mg_h = Column('u_mg_h', Float)
    u_al_h = Column('u_al_h', Float)
    u_si_h = Column('u_si_h', Float)
    u_p_h = Column('u_p_h', Float)
    u_s_h = Column('u_s_h', Float)
    u_k_h = Column('u_k_h', Float)
    u_ca_h = Column('u_ca_h', Float)
    u_ti_h = Column('u_ti_h', Float)
    u_v_h = Column('u_v_h', Float)
    u_cr_h = Column('u_cr_h', Float)
    u_mn_h = Column('u_mn_h', Float)
    u_fe_h = Column('u_fe_h', Float)
    u_co_h = Column('u_co_h', Float)
    u_ni_h = Column('u_ni_h', Float)
    u_cu_h = Column('u_cu_h', Float)
    u_ge_h = Column('u_ge_h', Float)
    u_c12_c13 = Column('u_c12_c13', Float)
    u_v_macro = Column('u_v_macro', Float)




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
            with open(self.input()["continuum"].path, "rb") as fp:
                continuum = pickle.load(fp)
        else:
            continuum = 1
        
        normalized_flux = observation.flux.value / continuum
        normalized_ivar = continuum * observation.uncertainty.array * continuum

        return (observation, continuum, normalized_flux, normalized_ivar)


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
                normalized_flux[[0]],
                normalized_ivar[[0]],
                **state
            )

            rows = []
            for p, u in zip(p_opt, p_cov):
                row = dict(zip(label_names, p))
                row.update(dict(zip(
                    (f"u_{ln}" for ln in label_names),
                    np.sqrt(np.diag(u))
                )))
                rows.append(row)

            # Write database row given the first result
            # (Which is either a stacked spectrum, or a single visit)
            task.output()["database"].write(rows[0])
            
            # Write AstraSource object.
            task.output()["AstraSource"].write(
                spectrum=spectrum,
                normalized_flux=normalized_flux[[0]],
                normalized_ivar=normalized_ivar[[0]],
                continuum=continuum[[0]],
                model_flux=model_flux,
                # TODO: Project uncertainties to flux space.
                model_ivar=None,
                results_table=Table(rows=rows)
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
        


class ContinuumNormalize(Sinusoidal, ApStarFile):

    """
    Pseudo-continuum normalise ApStar stacked spectra using a sum of sines and cosines. 
    
    :param continuum_regions_path:
        A path containing a list of (start, end) wavelength values that represent the regions to
        fit as continuum.
    """

    def requires(self):
        return ApStarFile(**self.get_common_param_kwargs(ApStarFile))


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



class EstimateStellarLabelsGivenApStarFile(EstimateStellarLabels, ContinuumNormalize, ApStarFile):
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

    @property
    def task_short_id(self):
        # Since this is the primary task for The Payne, let's give it a short ID.
        return f"{self.task_namespace}-{self.task_id.split('_')[-1]}"
    

    def requires(self):
        return {
            "model": TrainThePayne(**self.get_common_param_kwargs(TrainThePayne)),
            "observation": ApStarFile(**self.get_common_param_kwargs(ApStarFile)),
            "continuum": ContinuumNormalize(**self.get_common_param_kwargs(ContinuumNormalize))
        }


import astra
import numpy as np
import os
import pickle
from tqdm import tqdm
from astropy.table import Table
from astra.tasks.base import BaseTask
from astra.tasks.io import (ApStarFile, LocalTargetTask)
from astra.tasks.targets import LocalTarget, DatabaseTarget
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





class EstimateStellarParameters(ThePayneMixin):
    
    """
    Use a pre-trained neural network to estimate stellar parameters. This should be sub-classed to inherit properties from the type of spectra to be analysed.
    
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

    def run(self):
        """ Execute this task. """

        # Load the model.
        state = testing.load_state(self.input()["model"].path)

        # We can run this in batch mode.
        for task in tqdm(self.get_batch_tasks(), total=self.get_batch_size()):

            spectrum = Spectrum1D.read(task.input()["observation"].path)

            p_opt, p_cov, meta = result = testing.test(spectrum, **state)

            # Write database row.
            row = p_opt.copy()
            row.update(dict(zip(
                [f"u_{k}" for k in p_opt.keys()],
                np.sqrt(np.diag(p_cov))
            )))
            task.output()["database"].write(row)
            
            # Write additional things.
            #with open(task.output()["etc"].path, "wb") as fp:
            #    pickle.dump(result, fp)

            astraSource_path = task.output()["astraSource"].path
            
            data_table = Table(rows=[row])

            image = create_astra_source(
                    # TODO: Check with Nidever on CATID/catalogid.
                    catalog_id=spectrum.meta["header"]["CATID"],
                    obj=task.obj,
                    telescope=task.telescope,
                    healpix=task.healpix,
                    normalized_flux=spectrum.flux.value,
                    normalized_ivar=spectrum.uncertainty.array,
                    model_flux=meta["model_flux"],
                    # TODO: Will this work with BOSS as well?
                    crval=spectrum.meta["header"]["CRVAL1"],
                    cdelt=spectrum.meta["header"]["CDELT1"],
                    crpix=spectrum.meta["header"]["CRPIX1"],
                    ctype=spectrum.meta["header"]["CTYPE1"],
                    header=spectrum.meta["header"],
                    data_table=data_table,
                    reference_task=task
                )
            image.writeto(astraSource_path)

        
        return None


    def output(self):
        """ The output of this task. """
        if self.is_batch_mode:
            return [task.output() for task in self.get_batch_tasks()]
        
        path = os.path.join(
            self.output_base_dir,
            f"star/{self.telescope}/{int(self.healpix/1000)}/{self.healpix}/",
            f"astraSource-{self.apred}-{self.obj}-{self.task_id}.fits"
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)

        return {
            "database": ThePayneResult(self),
            "astraSource": LocalTarget(path)
        }
        





class EstimateStellarParametersGivenApStarFile(EstimateStellarParameters, ApStarFile):
    """
    Estimate stellar parameters given a single-layer neural network and an ApStar file.

    This task also requiresd all parameters that `astra.tasks.io.ApStarFile` requires.

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
        """ The requirements of this task. """
        requirements = {
            "model": TrainThePayne(**self.get_common_param_kwargs(TrainThePayne))
        }
        if not self.is_batch_mode:
            requirements.update(observation=ApStarFile(**self.get_common_param_kwargs(ApStarFile)))
        return requirements



class ContinuumNormalize(Sinusoidal, ApStarFile):

    """
    Pseudo-continuum normalise ApStar stacked spectra using a sum of sines and cosines. 
    
    :param continuum_regions_path:
        A path containing a list of (start, end) wavelength values that represent the regions to
        fit as continuum.
    """

    # Just take the first spectrum, which is stacked by individual pixel weighting.
    # (We will ignore individual visits).
    spectrum_kwds = dict(data_slice=(slice(0, 1), slice(None)))

    def requires(self):
        return ApStarFile(**self.get_common_param_kwargs(ApStarFile))


    def output(self):
        if self.is_batch_mode:
            return [task.output() for task in self.get_batch_tasks()]
        
        path = os.path.join(
            self.output_base_dir,
            f"star/{self.telescope}/{int(self.healpix/1000)}/{self.healpix}/",
            f"apStar-{self.apred}-{self.obj}-{self.task_id}.fits"
        )
        # Create the directory structure if it does not exist already.
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return LocalTarget(path)



class EstimateStellarParametersGivenNormalisedApStarFile(EstimateStellarParameters, ApStarFile):
    """
    Estimate stellar parameters given a single-layer neural network and an ApStar file.

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

    def requires(self):
        return {
            "model": TrainThePayne(**self.get_common_param_kwargs(TrainThePayne)),
            "observation": ContinuumNormalize(**self.get_common_param_kwargs(ContinuumNormalize))
        }

import numpy as np
import os
import torch
import yaml
from tqdm import tqdm
from luigi.parameter import ParameterVisibility
from sqlalchemy import Column, Float

import astra
from astropy.table import Table
from astra.tasks.base import BaseTask
from astra.tasks.targets import (DatabaseTarget, LocalTarget, AstraSource)
from astra.tasks.io import ApStarFile
from astra.tools.spectrum import Spectrum1D
from astra.contrib.apogeenet.model import Net, predict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class APOGEENetMixin(BaseTask):

    """ A mixin class for APOGEENet tasks. """

    task_namespace = "APOGEENet"

    model_path = astra.Parameter(
        description="The path of the trained APOGEENet model.",
        always_in_help=True,
        config_path=dict(section=task_namespace, name="model_path")
    )


class TrainedAPOGEENetModel(APOGEENetMixin):

    """ A trained APOGEENet model file. """

    def output(self):    
        return LocalTarget(self.model_path)


class APOGEENetResult(DatabaseTarget):

    """ A target database row for an APOGEENet result. """

    teff = Column("teff", Float)
    logg = Column("logg", Float)
    fe_h = Column("fe_h", Float)
    u_teff = Column("u_teff", Float)
    u_logg = Column("u_logg", Float)
    u_fe_h = Column("u_fe_h", Float)
    

class EstimateStellarParameters(APOGEENetMixin):

    """
    Estimate stellar parameters of a young stellar object, given a trained APOGEENet model.

    :param model_path:
        The path of the trained APOGEENet model.

    :param uncertainty: (optional)
        The number of draws to use when calculating the uncertainty in the
        network (default: 100).
    """

    uncertainty = astra.IntParameter(
        description="The number of draws to use when calculating the uncertainty in the network.",
        default=100,
        always_in_help=True,
        visibility=ParameterVisibility.HIDDEN,
    )

    @property
    def task_short_id(self):
        return f"{self.task_namespace}-{self.task_id.split('_')[-1]}"


    def output(self):
        """ The output produced by this task. """
        if self.is_batch_mode:
            return [task.output() for task in self.get_batch_tasks()]
        
        return {
            "database": APOGEENetResult(self),
            "astraSource": AstraSource(self)
        }


    def read_model(self):
        """ Read in the trained APOGEENet model. """

        # Load model.
        model = Net()
        model.load_state_dict(
            torch.load(
                self.input()["model"].path,
                map_location=device
            ),
            strict=False
        )
        model.to(device)
        model.eval()
        return model


    def read_observation(self):
        """ Read in the observations. """
        return Spectrum1D.read(self.input()["observation"].path)


    def estimate_stellar_parameters(self, model, spectrum):
        """
        Estimate stellar parameters given a trained APOGEENet model and a spectrum.

        :param model:
            The APOGEENet model as a torch network.
        
        :param spectrum:
            An observed spectrum.
        """

        N, P = spectrum.flux.shape
        
        # Build flux tensor as described.
        dtype = np.float32
        idx = 0

        results = []
        for i in range(N):
                
            flux = spectrum.flux.value[i].astype(dtype)
            error = spectrum.uncertainty.array[i].astype(dtype)**-0.5

            bad = ~np.isfinite(flux) + ~np.isfinite(error)
            flux[bad] = np.nanmedian(flux)
            error[bad] = 1e+10

            flux = torch.from_numpy(flux).to(device)
            error = torch.from_numpy(error).to(device)

            flux_tensor = torch.randn(self.uncertainty, 1, P).to(device)

            median_error = torch.median(error).item()
            
            error = torch.where(error == 1.0000e+10, flux, error)
            error_t = torch.tensor(np.array([5 * median_error], dtype=dtype)).to(device)
            error_t.repeat(P)
            error = torch.where(error >= 5 * median_error, error_t, error)

            flux_tensor = flux_tensor * error + flux
            flux_tensor[0][0] = flux
        
            # Estimate quantities.
            results.append(predict(model, flux_tensor))
        
        return results


    def run(self):
        """ Estimate stellar parameters given an APOGEENet model and ApStarFile(s). """

        model = self.read_model()

        # This task can be run in batch mode.
        failed_tasks = []
        for task in tqdm(self.get_batch_tasks(), total=self.get_batch_size()):
            spectrum = task.read_observation()    
            results = task.estimate_stellar_parameters(model, spectrum)
            # Write the first result to the database.
            task.output()["database"].write(results[0])

            # Write astraSource.
            task.output()["astraSource"].write(
                spectrum,
                normalized_flux=spectrum.flux.value,
                normalized_ivar=spectrum.uncertainty.array,
                continuum=None,
                model_flux=None,
                model_ivar=None,
                results_table=Table(rows=results)
            )

        return None


class EstimateStellarParametersGivenApStarFile(EstimateStellarParameters, ApStarFile):

    """
    Estimate stellar parameters of a young stellar object, given a trained APOGEENet model and an ApStar file.

    This task also requires all parameters that `astra.tasks.io.ApStarFile` requires.


    :param model_path:
        The path of the trained APOGEENet model.

    :param uncertainty: (optional)
        The number of draws to use when calculating the uncertainty in the
        network (default: 100).

    """

    def requires(self):
        requirements = {
            "model": TrainedAPOGEENetModel(**self.get_common_param_kwargs(TrainedAPOGEENetModel))
        }
        if not self.is_batch_mode:
            requirements.update(observation=ApStarFile(**self.get_common_param_kwargs(ApStarFile)))
        return requirements


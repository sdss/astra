
import luigi
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from sdss_access import SDSSPath
from astropy.table import Table
from luigi.parameter import ParameterVisibility

import astra
from astra.tasks.base import BaseTask
from astra.tasks.io import (ApStarFile, AllStarFile)
from astra.tools.spectrum import Spectrum1D

from astra.contrib.apogeenet.model import Net, predict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class APOGEENetMixin(BaseTask):
    """ A mixin class for APOGEENet tasks. """

    task_namespace = "APOGEENet"

    model_path = astra.Parameter(
        description="The path of the trained APOGEENet model.",
        always_in_help=True
    )


class TrainedAPOGEENetModel(APOGEENetMixin):

    """ A trained APOGEENet model file. """

    def output(self):    
        return luigi.LocalTarget(self.model_path)


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

    def output(self):
        """ The output produced by this task. """
        return luigi.LocalTarget(f"{self.task_id}.yaml")
        

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

        n_pixels = spectrum.flux.shape[1]
        
        # Build flux tensor as described.
        idx = 0
        flux = torch.from_numpy(spectrum.flux.value[idx].astype(np.float32)).to(device)
        error = torch.from_numpy(spectrum.uncertainty.array[idx]**-0.5).to(device)

        flux_tensor = torch.randn(self.uncertainty, 1, n_pixels).to(device)

        median_error = torch.median(error).item()
        
        error = torch.where(error == 1.0000e+10, flux, error)
        error_t = torch.tensor([5 * median_error]).to(device)
        error_t.repeat(n_pixels)
        error = torch.where(error >= 5 * median_error, error_t, error)

        flux_tensor = flux_tensor * error + flux
        flux_tensor[0][0] = flux
        
        # Estimate quantities.
        return predict(model, flux_tensor)


    def write_output(self, result):        
        """
        Write stellar parameter estimates to the output location of this task.
        """

        sanitised_result = { k: float(v) for k, v in result.items() }
        with open(self.output().path, "w") as fp:
            yaml.dump(sanitised_result, fp)

        self.trigger_event(luigi.Event.SUCCESS, self)
        return None


    def run(self):
        """
        Estimate stellar parameters given an APOGEENet model and ApStarFile(s).
        """

        model = self.read_model()

        # This task can be run in batch mode.
        failed_tasks = []
        for task in tqdm(self.get_batch_tasks(), total=self.get_batch_size()):
            spectrum = task.read_observation()    
            result = task.estimate_stellar_parameters(model, spectrum)
            task.write_output(result)

        return None


class EstimateStellarParametersGivenApStarFile(EstimateStellarParameters, ApStarFile):

    """
    Estimate stellar parameters of a young stellar object, given a trained APOGEENet model and an ApStar file.

    This task also requires all parameters that `astra.tasks.io.ApStarFile` requires.
    **TODO**: Update this docstring once the parameters for `ApStarFile` in sdss5 are stable.

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


class EstimateStellarParametersForAllStarSpectra(APOGEENetMixin, AllStarFile):

    release = luigi.Parameter()
    apred = luigi.Parameter()
    aspcap = luigi.Parameter()

    use_remote = luigi.BoolParameter(
        default=False, 
        significant=False,
        visibility=ParameterVisibility.HIDDEN
    )
    
    def requires(self):
        return AllStarFile(**self.get_common_param_kwargs(AllStarFile))
    
    def run(self):
        
        # Load in all the apStar files.
        table = Table.read(self.input().path)
        
        def get_task(row):
            return EstimateStellarParametersGivenApStarFile(
                model_path=self.model_path,
                release=self.release,
                apred=self.apred,
                telescope=row["TELESCOPE"],
                field=row["FIELD"],
                prefix=row["FILE"][:2],
                obj=row["APOGEE_ID"],
                apstar="stars",
                use_remote=self.use_remote,
            )
            
        # Load model
        model = get_task(table[0]).read_model()

        N = len(table)
        for i, row in enumerate(table):

            task = get_task(row)
            if task.complete():
                continue

            spectrum = task.read_observation()
            result = task.estimate_stellar_parameters(model, spectrum)

            task.write_output(result)

            try:
                self.set_progress_percentage(100 * i / N)
            except TypeError:
                None

            # Generate the task just so Luigi central scheduler can see that it gets done.
            #yield task            

        return None



if __name__ == "__main__":

    #/home/andy/data/sdss/dr16/apogee/spectro/redux/r12/stars/apo25m/218-04/apStar-r12-2M06440890-0610126.fits

    '''
    task = EstimateStellarParameters(
        model_path="../../../APOGEE_NET.pt",
        release="dr16",
        apred="r12",
        telescope="apo25m", 
        field="218-04",
        prefix="ap",
        obj="2M06440890-0610126",
        apstar="stars"
    )
    '''
    task = EstimateStellarParametersForAllStarSpectra(
        model_path="../../../APOGEE_NET.pt",
        release="dr16",
        apred="r12",
        aspcap="l33",
        use_remote=True
    )

    luigi.build(
        [task],
        detailed_summary=True
    )
    #task.run()


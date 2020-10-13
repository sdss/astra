
import os
import luigi
import pickle
import numpy as np
from astra.tasks.base import BaseTask
from astra.utils import log
from astra.tools.spectrum import Spectrum1D
from astra.tasks.io import ApStarFile
from tqdm import tqdm

import astra.contrib.thecannon as tc

from astra.contrib.thecannon.tasks.base import TheCannonMixin


class TestTheCannon(TheCannonMixin):

    # TODO: Allow for explicit initial values?
    N_initialisations = luigi.IntParameter(default=10)
    use_derivatives = luigi.BoolParameter(default=True)
    
    def requires(self):
        raise NotImplementedError(
            "You must overwrite the `requires()` function in the `astra_thecannon.tasks.Test` class "
            "so that it provides a dictionary with keys 'model' and 'observation', and tasks as values."
        )


    def output(self):
        raise NotImplementedError("This should be provided by the sub-classes")


    def read_observation(self):
        return Spectrum1D.read(self.input()["observation"].path)


    def resample_observation(self, dispersion, spectrum=None):
        if spectrum is None:
            spectrum = self.read_observation()

        o_x = spectrum.wavelength.value
        o_f = np.atleast_2d(spectrum.flux.value)
        o_i = np.atleast_2d(spectrum.uncertainty.quantity.value)

        N, P = shape = (o_f.shape[0], dispersion.size)
        flux = np.empty(shape)
        ivar = np.empty(shape)

        for i in range(N):
            flux[i] = np.interp(dispersion, o_x, o_f[i])
            ivar[i] = np.interp(dispersion, o_x, o_i[i])

        return (flux, ivar)


    def read_model(self):
        return tc.CannonModel.read(self.input()["model"].path)


    def run(self):

        model = self.read_model()

        # This can be run in batch mode.
        for task in tqdm(self.get_batch_tasks()):
            flux, ivar = task.resample_observation(model.dispersion)
            labels, cov, metadata = model.test(
                flux,
                ivar,
                initialisations=task.N_initialisations,
                use_derivatives=task.use_derivatives
            )
            
            # TODO: Write outputs somewhere!
            log.warn("Not writing outputs anywhere!")

            '''
            with open(task.output().path, "wb") as fp:
                pickle.dump(
                    (labels, cov, metadata), 
                    fp
                )

            task.trigger_event(luigi.Event.SUCCESS, task)
            '''


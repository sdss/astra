
import luigi
import numpy as np
import pickle
from tqdm import tqdm

from astra.tasks import BaseTask
from astra.tools.spectrum import Spectrum1D
from astra.utils.continuum.sines_and_cosines import normalize

class Sinusoidal(BaseTask):

    L = luigi.IntParameter(default=1400)
    continuum_order = luigi.IntParameter(default=3)
    continuum_regions_path = luigi.Parameter()
    spectrum_kwds = luigi.DictParameter(default=None)

    def run(self):
        continuum_regions = np.loadtxt(self.continuum_regions_path)

        spectrum_kwds = self.spectrum_kwds or {}

        # This can be run in single star mode or batch mode.
        tqdm_kwds = dict(desc="Continuum normalising", total=self.get_batch_size())
        for task in tqdm(self.get_batch_tasks(), **tqdm_kwds):
            if task.complete(): 
                continue
                
            spectrum = Spectrum1D.read(task.input().path, **spectrum_kwds)
            
            normalized_flux, normalized_ivar, continuum, metadata = normalize(
                spectrum.wavelength.value,
                spectrum.flux.value,
                spectrum.uncertainty.quantity.value,
                continuum_regions=continuum_regions,
                L=task.L,
                order=task.continuum_order,
                full_output=True
            )

            with open(task.output()["continuum"].path, "wb") as fp:
                pickle.dump(continuum, fp)

            N, P = spectrum.flux.shape
            nvisits = N if N < 2 else N - 2
            

    def output(self):
        raise RuntimeError("this should be over-written by the parent classes")
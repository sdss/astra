
import luigi
import numpy as np
import os
import pickle

from astropy.nddata import InverseVariance

from astra.tools.spectrum import Spectrum1D
from astra.utils.continuum.sines_and_cosines import normalize

from astra.tasks.base import BaseTask

class Sinusoidal(BaseTask):

    L = luigi.FloatParameter(default=1400)
    order = luigi.IntParameter(default=3)
    continuum_regions_path = luigi.Parameter()

    def run(self):
        continuum_regions = np.loadtxt(self.continuum_regions_path)
        
        spectrum = Spectrum1D.read(self.input().path)

        # TODO: Return a Spectrum1D object and write that instead.
        normalized_flux, normalized_ivar, continuum, metadata = normalize(
            spectrum.wavelength.value,
            spectrum.flux.value,
            spectrum.uncertainty.quantity.value,
            continuum_regions=continuum_regions,
            L=self.L,
            order=self.order,
        )


        output_path = self.output().path
        with open(output_path, "wb") as fp:
            pickle.dump(
                (
                    spectrum.wavelength.value,
                    normalized_flux[0], # TODO: Don't just dom first spcetrum
                    normalized_ivar[0], # TODO: Don't just dom first spcetrum
                ), 
                fp, 
                -1
            )
    

    def output(self):
        # Put it relative to the input path.
        output_path_prefix, ext = os.path.splitext(self.input().path)
        return luigi.LocalTarget(f"{output_path_prefix}-norm-sinusoidal-{self.L}-{self.order}.pkl")
    
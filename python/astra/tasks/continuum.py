
import luigi
import numpy as np
import os
import pickle

from astropy.nddata import InverseVariance

from astra.tools.spectrum import Spectrum1D
from astra.utils.continuum.sines_and_cosines import normalize

from astra.tasks.base import BaseTask



class Sinusoidal(BaseTask):

    sum_axis = luigi.IntParameter(default=None)
    L = luigi.FloatParameter(default=1400)
    order = luigi.IntParameter(default=3)
    continuum_regions_path = luigi.Parameter()

    def run(self):
        continuum_regions = np.loadtxt(self.continuum_regions_path)
        
        spectrum = Spectrum1D.read(self.input().path)

        normalized_flux, normalized_ivar, continuum, metadata = normalize(
            spectrum.wavelength.value,
            spectrum.flux.value,
            spectrum.uncertainty.quantity.value,
            continuum_regions=continuum_regions,
            L=self.L,
            order=self.order,
            full_output=True
        )

        # Stack if asked.
        if self.sum_axis is not None:
            
            sum_ivar = np.sum(normalized_ivar, axis=self.sum_axis)
            sum_flux = np.sum(normalized_flux * normalized_ivar, axis=self.sum_axis) / sum_ivar

            spectrum_kwds = dict(
                flux=sum_flux * spectrum.flux.unit,
                uncertainty=InverseVariance(sum_ivar * spectrum.uncertainty.unit)
            )

        else:
            spectrum_kwds = dict(
                flux=normalized_flux * spectrum.flux.unit, 
                uncertainty=InverseVariance(normalized_ivar * spectrum.uncertainty.unit)
            )
        
        normalized_spectrum = Spectrum1D(
            wcs=spectrum.wcs,
            meta=spectrum.meta,
            **spectrum_kwds
        )

        self.writer(normalized_spectrum, self.output().path)


    def output(self):
        # Put it relative to the input path.
        output_path_prefix, ext = os.path.splitext(self.input().path)
        return luigi.LocalTarget(f"{output_path_prefix}-{self.task_id}.fits")
    
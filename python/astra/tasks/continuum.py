
import luigi
import numpy as np
import os
import pickle
from tqdm import tqdm

from astropy.nddata import InverseVariance

from astra.tools.spectrum import Spectrum1D
from astra.utils.continuum.sines_and_cosines import normalize

from astra.tasks.base import BaseTask


class Sinusoidal(BaseTask):

    L = luigi.IntParameter(default=1400)
    continuum_order = luigi.IntParameter(default=3)
    continuum_regions_path = luigi.Parameter()
    spectrum_kwds = luigi.DictParameter(default={})

    def run(self):
        continuum_regions = np.loadtxt(self.continuum_regions_path)

        # This can be run in single star mode or batch mode.
        for task in tqdm(self.get_batch_tasks(), desc="Continuum normalising", total=self.get_batch_size()):
            if task.complete(): continue
                
            spectrum = Spectrum1D.read(task.input().path, **self.spectrum_kwds)

            normalized_flux, normalized_ivar, continuum, metadata = normalize(
                spectrum.wavelength.value,
                spectrum.flux.value,
                spectrum.uncertainty.quantity.value,
                continuum_regions=continuum_regions,
                L=task.L,
                order=task.continuum_order,
                full_output=True
            )

            """
            spectrum_kwds = dict(
                flux=normalized_flux * spectrum.flux.unit, 
                uncertainty=InverseVariance(normalized_ivar * spectrum.uncertainty.unit)
            )

            normalized_spectrum = Spectrum1D(
                wcs=spectrum.wcs,
                meta=spectrum.meta,
                **spectrum_kwds
            )

            task.writer(
                normalized_spectrum, 
                task.output().path,
                overwrite=True
            )
            """
            
            with open(task.output().path, "wb") as fp:
                pickle.dump(continuum, fp)


    def output(self):
        raise RuntimeError("this should be over-written by the parent classes")
    
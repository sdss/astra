
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
    spectrum_kwds = luigi.DictParameter(default={})

    def run(self):
        continuum_regions = np.loadtxt(self.continuum_regions_path)

        # This can be run in single star mode or batch mode.
        for task in self.get_batch_tasks():
                
            spectrum = Spectrum1D.read(
                task.input().path,
                **self.spectrum_kwds
            )

            normalized_flux, normalized_ivar, continuum, metadata = normalize(
                spectrum.wavelength.value,
                spectrum.flux.value,
                spectrum.uncertainty.quantity.value,
                continuum_regions=continuum_regions,
                L=task.L,
                order=task.order,
                full_output=True
            )

            # Stack if asked.
            if task.sum_axis is not None:
                sum_ivar = np.sum(normalized_ivar, axis=task.sum_axis)
                sum_flux = np.sum(normalized_flux * normalized_ivar, axis=task.sum_axis) / sum_ivar

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

            task.writer(normalized_spectrum, task.output().path)

            task.trigger_event(luigi.Event.SUCCESS, task)


    def output(self):
        # Put it relative to the input path.
        if self.is_batch_mode:
            return [task.output() for task in self.get_batch_tasks()]
        output_path_prefix, ext = os.path.splitext(self.input().path)
        return luigi.LocalTarget(f"{output_path_prefix}-{self.task_id}.fits")
    
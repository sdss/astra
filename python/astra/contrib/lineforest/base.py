import scipy.optimize as op
import numpy as np
import os
import tensorflow as tf
from functools import cache
from astropy.nddata import StdDevUncertainty
from astropy import units as u
from astra.base import task_decorator
from astra.database.astradb import DataProduct, SDSSOutput
from astra.utils import expand_path
from astra.tools.spectrum import (Spectrum1D, SpectrumList)
from astra.tools.spectrum.utils import spectrum_overlaps
from specutils.manipulation import SplineInterpolatedResampler

from typing import Iterable

from peewee import FloatField, IntegerField, TextField
from playhouse.postgres_ext import ArrayField

tf.autograph.set_verbosity(0)

LINES = [
    ['Halpha',      'hlines.model', 6562.8, 200],
    ['Hbeta',       'hlines.model', 4861.3, 200],
    ['Hgamma',      'hlines.model', 4340.5, 200],
    ['Hdelta',      'hlines.model', 4101.7, 200],
    ['Hepsilon',    'hlines.model', 3970.1, 200],
    ['H8',          'hlines.model', 3889.064, 200],
    ['H9',          'hlines.model', 3835.391, 200],
    ['H10',         'hlines.model', 3797.904, 200],
    ['H11',         'hlines.model', 3770.637, 200],
    ['H12',         'zlines.model', 3750.158, 50],
    ['H13',         'zlines.model', 3734.369, 50],
    ['H14',         'zlines.model', 3721.945, 50],
    ['H15',         'zlines.model', 3711.977, 50],
    ['H16',         'zlines.model', 3703.859, 50],
    ['H17',         'zlines.model', 3697.157, 50],
    ['Pa7',         'hlines.model', 10049.4889, 200],
    ['Pa8',         'hlines.model', 9546.0808, 200],
    ['Pa9',         'hlines.model', 9229.12, 200],
    ['Pa10',        'hlines.model', 9014.909, 200],
    ['Pa11',        'hlines.model', 8862.782, 200],
    ['Pa12',        'hlines.model', 8750.472, 200],
    ['Pa13',        'hlines.model', 8665.019, 200],
    ['Pa14',        'hlines.model', 8598.392, 200],
    ['Pa15',        'hlines.model', 8545.383, 200],
    ['Pa16',        'hlines.model', 8502.483, 200],
    ['Pa17',        'hlines.model', 8467.254, 200],
    ['CaII8662',    'zlines.model', 8662.14, 50],
    ['CaII8542',    'zlines.model', 8542.089, 50],
    ['CaII8498',    'zlines.model', 8498.018, 50],
    ['CaK3933',     'hlines.model', 3933.6614, 200],
    ['CaH3968',     'hlines.model', 3968.4673, 200],
    ['HeI6678',     'zlines.model', 6678.151, 50],
    ['HeI5875',     'zlines.model', 5875.621, 50],
    ['HeI5015',     'zlines.model', 5015.678, 50],
    ['HeI4471',     'zlines.model', 4471.479, 50],
    ['HeII4685',    'zlines.model', 4685.7, 50],
    ['NII6583',     'zlines.model', 6583.45, 50],
    ['NII6548',     'zlines.model', 6548.05, 50],
    ['SII6716',     'zlines.model', 6716.44, 50],
    ['SII6730',     'zlines.model', 6730.816, 50],
    ['FeII5018',    'zlines.model', 5018.434, 50],
    ['FeII5169',    'zlines.model', 5169.03, 50],
    ['FeII5197',    'zlines.model', 5197.577, 50],
    ['FeII6432',    'zlines.model', 6432.68, 50],
    ['OI5577',      'zlines.model', 5577.339, 50],
    ['OI6300',      'zlines.model', 6300.304, 50],
    ['OI6363',      'zlines.model', 6363.777, 50],
    ['OII3727',     'zlines.model', 3727.42, 50],
    ['OIII4959',    'zlines.model', 4958.911, 50],
    ['OIII5006',    'zlines.model', 5006.843, 50],
    ['OIII4363',    'zlines.model', 4363.85, 50],
    ['LiI',         'zlines.model', 6707.76, 50],
]

class LineForestOutput(SDSSOutput):

    name = TextField()
    wavelength_vac = FloatField()
    minmax = FloatField()

    eqw = FloatField()
    abs = FloatField()

    detection_lower = FloatField()
    detection_upper = FloatField()

    eqw_percentile_16 = FloatField(null=True)
    eqw_percentile_50 = FloatField(null=True)
    eqw_percentile_84 = FloatField(null=True)

    abs_percentile_16 = FloatField(null=True)
    abs_percentile_50 = FloatField(null=True)
    abs_percentile_84 = FloatField(null=True)


@task_decorator
def lineforest(data_product: DataProduct, steps: int = 128, reps: int = 100) -> Iterable[LineForestOutput]:

    for spectrum in SpectrumList.read(data_product.path):
        if not spectrum_overlaps(spectrum, 5_000):
            # Exclude non-BOSS things
            continue

        flux = spectrum.flux.flatten()
        flux_error = spectrum.uncertainty.represent_as(StdDevUncertainty).array.flatten()
        median_flux_error = np.nanmedian(flux_error)

        high_error = (flux_error > (5 * median_flux_error))
        bad_pixel = (flux <= 0) | ~np.isfinite(flux)

        flux_error[high_error] = 5 * median_flux_error
        flux.value[bad_pixel] = 1
        # TODO: increase flux error at bad pixels?
        
        specs = Spectrum1D(
            spectral_axis=spectrum.spectral_axis,
            flux=np.log10(flux.value) * flux.unit,
            uncertainty=StdDevUncertainty(flux_error/flux.value/np.log(10))
        )

        spline = SplineInterpolatedResampler()

        for name, model_path, wavelength_air, minmax in LINES:
            
            wavelength_vac = airtovac(wavelength_air)
            
            model = read_model(os.path.join(f"$MWM_ASTRA/component_data/lineforest/{model_path}"))

            spec = spline(specs, (np.linspace(-minmax, minmax, steps) + wavelength_vac) * u.AA)
            window = spec.flux.value.reshape((1,(steps) , 1))
            window = np.tile(window,(reps+1,1,1))
            scatter= spec.uncertainty.array.reshape((1,(steps) , 1))
            scatter= np.tile(scatter,(reps+1,1,1))*np.random.normal(size=(reps+1,steps,1),loc=0,scale=1)
            scatter[0]=0
            window=window+scatter

            predictions = unnormalize(np.array(model(window[0:1,:,:])))

            if np.abs(predictions[0,2])>0.5: 
                eqw, abs = (predictions[0, 0], predictions[0, 1])
                detection_lower = predictions[0, 2]

                predictions = unnormalize(np.array(model(window[1:,:,:])))  

                a = np.where(np.abs(predictions[1:,2])>0.5)[0]
                detection_upper = np.round(len(a)/reps,2)

                kwds = dict(
                    data_product=data_product,
                    spectrum=spectrum,
                    name=name,
                    wavelength_vac=wavelength_vac,
                    minmax=minmax,
                    eqw=eqw,
                    abs=abs,
                    detection_lower=detection_lower,
                    detection_upper=detection_upper,
                )

                if len(a)>2:
                    eqw_percentile_16, eqw_percentile_50, eqw_percentile_84 = np.round(np.percentile(predictions[1:,0][a],[16,50,84]),4)
                    abs_percentile_16, abs_percentile_50, abs_percentile_84 = np.round(np.percentile(predictions[1:,1][a],[16,50,84]),4)
                    kwds.update(
                        eqw_percentile_16=eqw_percentile_16,
                        eqw_percentile_50=eqw_percentile_50,
                        eqw_percentile_84=eqw_percentile_84,
                        abs_percentile_16=abs_percentile_16,
                        abs_percentile_50=abs_percentile_50,
                        abs_percentile_84=abs_percentile_84,
                    )
                
                yield LineForestOutput(**kwds)


def unnormalize(predictions):
    predictions[:,0]=10**predictions[:,0]
    predictions[:,1]=10**predictions[:,1]
    a=np.where(predictions[:,2]<0)[0]
    predictions[a,0]=-predictions[a,0]
    return np.round(predictions.astype(float),4)



@cache
def read_model(path):
    return tf.keras.models.load_model(expand_path(path))


def airtovac(wvl, xc=None):
    # Constants in [(10-6 m)**-2] from Ciddor 1996, Applied Optics, Vol. 35, Issue 9, pp. 1566-1573
    # Appendix A
    k0 = 238.0185
    k1 = 5792105.
    k2 = 57.362
    k3 = 167917.0
    
    s2 = (1e4/wvl)**2
    
    # Eqs. 1 and 2
    nas = (k1/(k0 - s2) + k3/(k2 - s2)) / 1e8 + 1.0
    if xc is not None:
        naxs = (nas - 1.0)*(1.0 + 0.534e-6*(xc - 450.)) + 1.0
        return naxs
    return nas
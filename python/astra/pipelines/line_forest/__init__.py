import scipy.optimize as op
import numpy as np
import os
import copy
import concurrent.futures
from peewee import chunked
from functools import cache
from astropy.nddata import StdDevUncertainty
from astropy import units as u
from astra import task
from astra.utils import log, expand_path
from tqdm import tqdm
from specutils import Spectrum1D
from specutils.manipulation import SplineInterpolatedResampler
from collections import OrderedDict

from typing import Iterable

from astra.models.boss import BossVisitSpectrum
from astra.models.line_forest import LineForest

LINES = [
    ['H_alpha',      'hlines.model', 6562.8, 200],
    ['H_beta',       'hlines.model', 4861.3, 200],
    ['H_gamma',      'hlines.model', 4340.5, 200],
    ['H_delta',      'hlines.model', 4101.7, 200],
    ['H_epsilon',    'hlines.model', 3970.1, 200],
    ['H_8',          'hlines.model', 3889.064, 200],
    ['H_9',          'hlines.model', 3835.391, 200],
    ['H_10',         'hlines.model', 3797.904, 200],
    ['H_11',         'hlines.model', 3770.637, 200],
    ['H_12',         'zlines.model', 3750.158, 50],
    ['H_13',         'zlines.model', 3734.369, 50],
    ['H_14',         'zlines.model', 3721.945, 50],
    ['H_15',         'zlines.model', 3711.977, 50],
    ['H_16',         'zlines.model', 3703.859, 50],
    ['H_17',         'zlines.model', 3697.157, 50],
    ['Pa_7',         'hlines.model', 10049.4889, 200],
    ['Pa_8',         'hlines.model', 9546.0808, 200],
    ['Pa_9',         'hlines.model', 9229.12, 200],
    ['Pa_10',        'hlines.model', 9014.909, 200],
    ['Pa_11',        'hlines.model', 8862.782, 200],
    ['Pa_12',        'hlines.model', 8750.472, 200],
    ['Pa_13',        'hlines.model', 8665.019, 200],
    ['Pa_14',        'hlines.model', 8598.392, 200],
    ['Pa_15',        'hlines.model', 8545.383, 200],
    ['Pa_16',        'hlines.model', 8502.483, 200],
    ['Pa_17',        'hlines.model', 8467.254, 200],
    ['Ca_II_8662',   'zlines.model', 8662.14, 50],
    ['Ca_II_8542',   'zlines.model', 8542.089, 50],
    ['Ca_II_8498',   'zlines.model', 8498.018, 50],
    ['Ca_K_3933',     'hlines.model', 3933.6614, 200],
    ['Ca_H_3968',     'hlines.model', 3968.4673, 200],
    ['He_I_6678',     'zlines.model', 6678.151, 50],
    ['He_I_5875',     'zlines.model', 5875.621, 50],
    ['He_I_5015',     'zlines.model', 5015.678, 50],
    ['He_I_4471',     'zlines.model', 4471.479, 50],
    ['He_II_4685',    'zlines.model', 4685.7, 50],
    ['N_II_6583',     'zlines.model', 6583.45, 50],
    ['N_II_6548',     'zlines.model', 6548.05, 50],
    ['S_II_6716',     'zlines.model', 6716.44, 50],
    ['S_II_6730',     'zlines.model', 6730.816, 50],
    ['Fe_II_5018',    'zlines.model', 5018.434, 50],
    ['Fe_II_5169',    'zlines.model', 5169.03, 50],
    ['Fe_II_5197',    'zlines.model', 5197.577, 50],
    ['Fe_II_6432',    'zlines.model', 6432.68, 50],
    ['O_I_5577',      'zlines.model', 5577.339, 50],
    ['O_I_6300',      'zlines.model', 6300.304, 50],
    ['O_I_6363',      'zlines.model', 6363.777, 50],
    ['O_II_3727',     'zlines.model', 3727.42, 50],
    ['O_III_4959',    'zlines.model', 4958.911, 50],
    ['O_III_5006',    'zlines.model', 5006.843, 50],
    ['O_III_4363',    'zlines.model', 4363.85, 50],
    ['Li_I',         'zlines.model', 6707.76, 50],
]

@task
def line_forest(spectra: Iterable[BossVisitSpectrum], steps: int = 128, reps: int = 100, max_workers: int = 32) -> Iterable[LineForest]:
    """
    Measure spectral line strengths.

    :param spectra:
        Input BOSS spectra.
    
    :param steps:
        Number of steps to use when sampling the line profile.
    
    :param reps:
        Number of times to repeat the measurement.
    
    :param max_workers:
        Maximum number of workers to use.
    """
    
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

    total, futures = (0, [])
    for chunk in chunked(spectra, 1000):
        futures.append(executor.submit(_line_forest, chunk, steps, reps))
        total += len(chunk)

    with tqdm(total=total) as pb:
        for future in concurrent.futures.as_completed(futures):
            results = future.result()
            yield from results
            pb.update(len(results))


def _line_forest(spectra, steps, reps):
        
    models = {
        "zlines.model": read_model(os.path.join(f"$MWM_ASTRA/pipelines/lineforest/zlines2.model")),
        "hlines.model": read_model(os.path.join(f"$MWM_ASTRA/pipelines/lineforest/hlines2.model")),
    }

    results = []
    for spectrum in spectra:

        try:    
            flux = np.copy(spectrum.flux)
            e_flux = np.copy(spectrum.e_flux)
            median_e_flux = np.median(e_flux[np.isfinite(e_flux)])
            if not np.isfinite(median_e_flux):
                median_e_flux = 1e3

            high_error = (e_flux > (5 * median_e_flux)) | (~np.isfinite(e_flux))
            bad_pixel = (flux <= 0) | (~np.isfinite(flux))
            e_flux[high_error] = 5 * median_e_flux
            flux[bad_pixel] = 1
            
            # TODO: increase flux error at bad pixels?
            uncertainty = e_flux/flux/np.log(10)
            uncertainty[~np.isfinite(uncertainty)] = 5 * median_e_flux
            
            specs = Spectrum1D(
                spectral_axis=spectrum.wavelength * u.Angstrom,
                flux=u.Quantity(np.log10(flux)),
                uncertainty=StdDevUncertainty(uncertainty)
            )

            spline = SplineInterpolatedResampler()

            result_kwds = OrderedDict([])
            result_kwds.update(dict(
                source_pk=spectrum.source_pk,
                spectrum_pk=spectrum.spectrum_pk,
            ))

            for name, model_path, wavelength_air, minmax in LINES:
                
                try:
                        
                    wavelength_vac = airtovac(wavelength_air)
                    
                    model = models[os.path.basename(model_path)]

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
                        
                        if detection_upper>0.3:
                            result_kwds.update({
                                f"eqw_{name.lower()}": eqw,
                                f"abs_{name.lower()}": abs,
                                f"detection_lower_{name.lower()}": detection_lower,
                                f"detection_upper_{name.lower()}": detection_upper
                            })
    
                            if len(a)>2:
                                result_kwds.update({
                                    f"eqw_percentiles_{name.lower()}": np.round(np.percentile(predictions[1:,0][a],[16,50,84]),4),
                                    f"abs_percentiles_{name.lower()}": np.round(np.percentile(predictions[1:,1][a],[16,50,84]),4),
                                }) 
                except:
                    log.exception(f"Exception when measuring {name} for spectrum {spectrum}")
                    continue
                                
        except:
            log.warning(f"Exception when running line_forest for spectrum {spectrum}")
        else:
            results.append(LineForest(**result_kwds))
    
    return results



def unnormalize(predictions):
    predictions[:,0]=10**predictions[:,0]
    predictions[:,1]=10**predictions[:,1]
    a=np.where(predictions[:,2]<0)[0]
    predictions[a,0]=-predictions[a,0]
    return np.round(predictions.astype(float),4)



@cache
def read_model(path):
    import tensorflow as tf
    tf.autograph.set_verbosity(0)
    return tf.keras.models.load_model(expand_path(path),compile=False)

# TODO: move this to a utility
def airtovac(wave_air) :
    """ Convert air wavelengths to vacuum wavelengths
        Corrects for the index of refraction of air under standard conditions.  
        Wavelength values below 2000 A will not be altered.  Accurate to about 10 m/s.
        From IDL Astronomy Users Library, which references Ciddor 1996 Applied Optics 35, 1566
    """
    if not isinstance(wave_air, np.ndarray) : 
        air = np.array([wave_air])
    else :
        air = wave_air

    vac = copy.copy(air)
    g = np.where(vac >= 2000)[0]     #Only modify above 2000 A

    for iter in range(2) :
        sigma2 = (1e4/vac[g])**2.     # Convert to wavenumber squared
        # Compute conversion factor
        fact = 1. +  5.792105E-2/(238.0185E0 - sigma2) + 1.67917E-3/( 57.362E0 - sigma2)

        vac[g] = air[g]*fact              #Convert Wavelength
    return vac    
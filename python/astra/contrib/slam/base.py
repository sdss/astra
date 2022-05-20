

import numpy as np
import os        
from tqdm import tqdm
from astra import log
from astra.database import astradb
from astra.tools.spectrum import Spectrum1D
from scipy.interpolate import interp1d

from astra.utils import expand_path
from astra.contrib.slam.slam.slam3 import Slam3 as Slam
from astra.contrib.slam.slam.normalization import normalize_spectra_block


from astra.base import (ExecutableTask, Parameter, TupleParameter, DictParameter)

class EstimateStellarLabels(ExecutableTask):

    model_path = Parameter(bundled=True)
    dwav = Parameter(default=10.0)
    p = TupleParameter(default=(1E-8, 1E-7))
    q = Parameter(default=0.7)
    ivar_block = TupleParameter(default=None)
    eps = Parameter(default=1e-19)
    rsv_frac = Parameter(default=2)
    n_jobs = Parameter(default=1)
    verbose = Parameter(default=5)

    '''
    # TODO: make this nicer    
    :param model_path:
        The disk path of the pre-trained model.
        
    :param dwave_slam: float
        binning width
        
    :param p_slam: tuple of 2 ps [optional]
        smoothing parameter between 0 and 1: (default: 1E-8, 1E-7)
        0 -> LS-straight line
        1 -> cubic spline interpolant
        
    :param q_slam: float in range of [0, 100] [optional]
        percentile, between 0 and 1 (default: 0.7)
        
    :param ivar_block_slam: ndarray (n_pix, ) | None [optional]
        ivar array (default: None)
        
    :param eps_slam: float [optional]
        the ivar threshold (default: 1E-19)
    
    :param rsv_frac_slam: float [optional]
        the fraction of pixels reserved in terms of std. default is 3.
    
    :param n_jobs_slam: int [optional]
        number of processes launched by joblib (default: 1)
        
    :param verbose_slam: int / bool [optional]
        verbose level (default: 5)
    '''

    def execute(self):

        model = Slam.load_dump(expand_path(self.model_path))
        wave_interp = model.wave

        log.info(f"Loaded model from {self.model_path}")

        for i, (task, data_products, parameters) in enumerate(self.iterable()):
            for j, data_product in enumerate(data_products):

                spectrum = Spectrum1D.read(data_product.path)
            
                wave   = spectrum.spectral_axis
                fluxes = spectrum.flux.value
                invars = spectrum.uncertainty.array

                N, P = fluxes.shape
                
                fluxes_resamp, invars_resamp = [], []
                for i in range(N):
                    fluxes_temp, invars_temp = resample(wave[i], fluxes[i], invars[i], wave_interp)
                    fluxes_resamp += [fluxes_temp]
                    invars_resamp += [invars_temp]
                fluxes_resamp, invars_resamp = np.array(fluxes_resamp), np.array(invars_resamp)

                fluxes_norm, fluxes_cont = normalize_spectra_block(
                    wave_interp,
                    fluxes_resamp,
                    (6147., 8910.),
                    dwav=parameters['dwav'],
                    p=parameters['p'],
                    q=parameters['q'],
                    ivar_block=parameters['ivar_block'],
                    eps=parameters['eps'],
                    rsv_frac=parameters['rsv_frac'],
                    n_jobs=parameters['n_jobs'],
                    verbose=parameters['verbose']
                )

                invars_norm = fluxes_cont**2*invars_resamp

                ### Initial estimation: get initial estimate of parameters by chi2 best match
                label_init = model.predict_labels_quick(fluxes_norm, invars_norm, n_jobs=1)

                ### SLAM prediction: optimize parameters
                results_pred = model.predict_labels_multi(label_init, fluxes_norm, invars_norm)
                label_pred = np.array([label['x'] for label in results_pred])
                std_pred   = np.array([label['pstd'] for label in results_pred])

                ### modify the following block for SLAM style
                # Create results array.
                ### log_g, log_teff, fe_h = predictions.T
                ### teff = 10**log_teff
                teff    = label_pred[:,0]
                m_h     = label_pred[:,1]
                log_g   = label_pred[:,2]
                alpha_m = label_pred[:,3]
                u_teff    = std_pred[:,0]
                u_m_h     = std_pred[:,1]
                u_log_g   = std_pred[:,2]
                u_alpha_m = std_pred[:,3]
                result = dict(
                    snr=spectrum.meta["snr"],
                    teff=teff.tolist(),
                    m_h=m_h.tolist(),
                    logg=log_g.tolist(),
                    alpha_m=alpha_m.tolist(),
                    u_teff=u_teff.tolist(),
                    u_m_h=u_m_h.tolist(),
                    u_logg=u_log_g.tolist(),
                    u_alpha_m=u_alpha_m.tolist(),
                )

                raise a


def resample(wave, flux, err, wave_resamp):
    f1 = interp1d(wave, flux, kind='cubic')
    f2 = interp1d(wave, err,  kind='cubic')
    re_flux = f1(wave_resamp)
    re_err = f2(wave_resamp)
    return np.array(re_flux), np.array(re_err)                
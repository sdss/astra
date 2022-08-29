import numpy as np
import os
from tqdm import tqdm

from astra.operators.utils import prepare_data
from astra.database.utils import create_task_output, deserialize_pks
from astra.utils import log
from astra.database import astradb

### from astra.contrib.apogeenet.model import Model
### from astra.contrib.apogeenet.utils import (create_bitmask, get_metadata)
from astra.contrib.slam.slam.slam3 import Slam3 as Slam
from astra.contrib.slam.slam.normalization import (
    normalize_spectrum,
    normalize_spectra_block,
)

from scipy.interpolate import interp1d

### import torch


def resample(wave, flux, err, wave_resamp):
    f1 = interp1d(wave, flux, kind="cubic")
    f2 = interp1d(wave, err, kind="cubic")
    re_flux = f1(wave_resamp)
    re_err = f2(wave_resamp)
    return np.array(re_flux), np.array(re_err)


def estimate_stellar_labels(
    pks,
    model_path,
    dwave_slam=10.0,
    p_slam=(1e-8, 1e-7),
    q_slam=0.7,
    ivar_block_slam=None,
    eps_slam=1e-19,
    rsv_frac_slam=2.0,
    n_jobs_slam=1,
    verbose_slam=5,
):
    """
    Estimate the stellar parameters for APOGEE ApStar observations,
    where task instances have been created with the given primary keys (`pks`).

    :param pks:
        The primary keys of task instances that include information about what
        ApStar observation to load.

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
    """

    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    log.info(f"Running APOGEENet on device {device} with:")
    log.info(f"\tmodel_path: {model_path}")
    log.info(f"\tpks: {pks}")

    log.debug(f"CUDA_VISIBLE_DEVICES = '{os.environ.get('CUDA_VISIBLE_DEVICES')}'")

    log.debug(f"Using torch version {torch.__version__} in {torch.__path__}")
    
    # Load the model.
    ### model = Model(model_path, device)
    """

    # Load the model.
    model = Slam.load_dump(model_path)  ### ("./models/btsettl.dump")
    ### wave_interp = np.load("./models/wave_interp_R1800.npz")['wave'] ### ??? how to load properly
    wave_interp = model.wave

    log.info(f"Loaded model from {model_path}")

    pks = deserialize_pks(pks, flatten=True)
    total = len(pks)

    log.info(f"There are {total} primary keys to process: {pks}")

    for instance, path, spectrum in tqdm(prepare_data(pks), total=total):
        if spectrum is None:
            continue

        N, P = spectrum.flux.shape

        """
        ### original code in apogeenet
        flux = np.nan_to_num(spectrum.flux.value).astype(np.float32).reshape((N, 1, P))
        
        ### original code in MDwarfMachine
        fluxes, invars = [], []
        for i in tqdm(range(len(obs_spec))):
            fluxes += [obs_spec[i]['flux_resamp']]
            invars += [obs_spec[i]['invar_resamp']]
        fluxes, invars = np.array(fluxes), np.array(invars)
        """
        ### wave   = np.nan_to_num(spectrum.spectral_axis.value).astype(np.float32).reshape((N, 1, P))
        ### fluxes = np.nan_to_num(spectrum.flux.value).astype(np.float32).reshape((N, 1, P)) ### ??? reshape to what format
        ### invars = np.nan_to_num(spectrum.uncertainty.array).astype(np.float32).reshape((N, 1, P)) ### ???  spectrum.uncertainity format
        wave = spectrum.spectral_axis
        fluxes = spectrum.flux
        invars = specrrum.uncertainty

        fluxes_resamp, invars_resamp = [], []
        for i in tqdm(range(N)):
            fluxes_temp, invars_temp = resample(
                wave[i], fluxes[i], invars[i], wave_interp
            )
            fluxes_resamp += [fluxes_temp]
            invars_resamp += [invars_temp]
        fluxes_resamp, invars_resamp = np.array(fluxes_resamp), np.array(invars_resamp)

        ### normalization of each spetra
        ### fluxes_norm, fluxes_cont = normalize_spectra_block(wave_interp, fluxes_resamp,
        ###                                           (6147., 8910.), 10., p=(1E-8, 1E-7), q=0.7,
        ###                                           eps=1E-19, rsv_frac=2., n_jobs=1, verbose=5) ### ??? inputs
        fluxes_norm, fluxes_cont = normalize_spectra_block(
            wave_interp,
            fluxes_resamp,
            (6147.0, 8910.0),
            dwave_slam,
            p=p_slam,
            q=q_slam,
            ivar_block=ivar_block_slam,
            eps=eps_slam,
            rsv_frac=rsv_frac_slam,
            n_jobs=n_jobs_slam,
            verbose=verbose_slam,
        )

        invars_norm = fluxes_cont**2 * invars_resamp

        ### Initial estimation: get initial estimate of parameters by chi2 best match
        label_init = model.predict_labels_quick(fluxes_norm, invars_norm, n_jobs=1)

        ### SLAM prediction: optimize parameters
        results_pred = model.predict_labels_multi(label_init, fluxes_norm, invars_norm)
        label_pred = np.array([label["x"] for label in results_pred])
        std_pred = np.array([label["pstd"] for label in results_pred])

        ### modify the following block for SLAM style
        # Create results array.
        ### log_g, log_teff, fe_h = predictions.T
        ### teff = 10**log_teff
        teff = label_pred[:, 0]
        m_h = label_pred[:, 1]
        log_g = label_pred[:, 2]
        alpha_m = label_pred[:, 3]
        u_teff = std_pred[:, 0]
        u_m_h = std_pred[:, 1]
        u_log_g = std_pred[:, 2]
        u_alpha_m = std_pred[:, 3]
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

        # Write the result to database.
        ### create_task_output(instance, astradb.ApogeeNet, **result)
        create_task_output(instance, astradb.SLAM, **result)

    log.info(f"Completed processing of {total} primary keys")

import numpy as np
from astropy.nddata import InverseVariance
from astra import log
from astra.tools.spectrum import Spectrum1D, SpectrumList
from astra.tools.spectrum.utils import spectrum_overlaps
from scipy.interpolate import interp1d
from astropy import units as u

from astra.utils import expand_path, list_to_dict, dict_to_list
from astra.contrib.slam.slam.slam3 import Slam3
from astra.contrib.slam.slam.normalization import normalize_spectra_block
from astra.database.astradb import database, SlamOutput

from astra.base import TaskInstance, Parameter, TupleParameter, DictParameter

from astra.sdss.datamodels.base import get_extname
from astra.sdss.datamodels.pipeline import create_pipeline_product


class Slam(TaskInstance):

    model_path = Parameter(
        default="$MWM_ASTRA/component_data/slam/ASPCAP_DR16_astra.dump", bundled=True
    )
    dwave = Parameter(default=10.0)
    p = TupleParameter(default=(1e-8, 1e-7))
    q = Parameter(default=0.7)
    ivar_block = TupleParameter(default=None)
    eps = Parameter(default=1e-19)
    rsv_frac = Parameter(default=2)
    n_jobs = Parameter(default=1)
    verbose = Parameter(default=5)

    """
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
    """

    def execute(self):

        model = Slam3.load_dump(expand_path(self.model_path))
        wave_interp = model.wave

        log.info(f"Loaded model from {self.model_path}")

        for i, (task, data_products, parameters) in enumerate(self.iterable()):
            # Only use the first data product
            data_product, = data_products

            all_results = {}
            database_results = []
            for spectrum in SpectrumList.read(data_product.path):
                if not spectrum_overlaps(spectrum, model.wave): 
                    continue

                wave = spectrum.spectral_axis.value
                fluxes = np.atleast_2d(spectrum.flux.value)
                invars = np.atleast_2d(spectrum.uncertainty.represent_as(InverseVariance).array)

                N, P = fluxes.shape

                fluxes_resamp, invars_resamp = [], []
                for i in range(N):
                    fluxes_temp, invars_temp = resample(
                        wave, fluxes[i], invars[i], wave_interp
                    )
                    fluxes_resamp += [fluxes_temp]
                    invars_resamp += [invars_temp]
                fluxes_resamp = np.array(fluxes_resamp)
                invars_resamp = np.array(invars_resamp)

                fluxes_norm, fluxes_cont = normalize_spectra_block(
                    wave_interp,
                    fluxes_resamp,
                    (6147.0, 8910.0),
                    dwave=parameters["dwave"],
                    p=parameters["p"],
                    q=parameters["q"],
                    ivar_block=parameters["ivar_block"],
                    eps=parameters["eps"],
                    rsv_frac=parameters["rsv_frac"],
                    n_jobs=parameters["n_jobs"],
                    verbose=parameters["verbose"],
                )

                invars_norm = fluxes_cont**2 * invars_resamp

                ### Initial estimation: get initial estimate of parameters by chi2 best match
                label_init = model.predict_labels_quick(
                    fluxes_norm, invars_norm, n_jobs=1
                )

                ### SLAM prediction: optimize parameters
                results_pred = model.predict_labels_multi(
                    label_init, fluxes_norm, invars_norm
                )
                label_pred = np.array([label["x"] for label in results_pred])
                std_pred = np.array([label["pstd"] for label in results_pred])
                N_, L = label_pred.shape

                # Create results array.
                teff = label_pred[:, 0]
                e_teff = std_pred[:, 0]
                log_g = label_pred[:, 1]
                e_log_g = std_pred[:, 1]
                fe_h = label_pred[:, 2]
                e_fe_h = std_pred[:, 2]
                # alpha_m = label_pred[:,3]
                # u_alpha_m = std_pred[:,3]

                try:
                    snr = spectrum.meta["SNR"]
                except:
                    snr = fluxes * np.sqrt(invars)
                    snr[snr <= 0] = np.nan
                    snr = np.nanmean(snr, axis=1)

                # TODO: use 'pcov' covariance matrix and store correlation coefficients?
                #       Store cost? or any other things about fitting?

                prediction = model.predict_spectra(label_pred)
                chi_sq = (prediction - fluxes_norm) ** 2 * invars_norm
                reduced_chi_sq = np.sum(chi_sq, axis=1) / (P - L - 1)

                # Create AstraStar product.
                model_continuum = fluxes_temp / fluxes_norm

                resampled_continuum = np.empty((N, P))
                resampled_model_flux = np.empty((N, P))
                for i in range(N):
                    f = interp1d(wave_interp, prediction[i], kind="cubic", bounds_error=False)
                    c = interp1d(wave_interp, model_continuum[i], kind="cubic", bounds_error=False)

                    # Re-sample the predicted spectra back to the observed frame.
                    resampled_model_flux[i] = f(wave)
                    resampled_continuum[i] = c(wave)

                results = dict(
                    snr=snr,
                    teff=teff,
                    e_teff=e_teff,
                    logg=log_g,
                    e_logg=e_log_g,
                    fe_h=fe_h,
                    e_fe_h=e_fe_h,
                    chi_sq=chi_sq,
                    reduced_chi_sq=reduced_chi_sq,
                )
                database_results.extend(dict_to_list(results))
                results.update(
                    spectral_axis=spectrum.spectral_axis,  
                    model_flux=resampled_model_flux,
                    continuum=resampled_continuum,
                )

                # Which extname should this go to?
                all_results[get_extname(spectrum, data_product)] = results

            with database.atomic():
                task.create_or_update_outputs(SlamOutput, database_results)

                # Create astraStar/astraVisit data product and link it to this task.
                create_pipeline_product(task, data_product, all_results)


def resample(wave, flux, err, wave_resamp, bounds_error=None):
    f1 = interp1d(wave, flux, kind="cubic", bounds_error=bounds_error)
    f2 = interp1d(wave, err, kind="cubic", bounds_error=bounds_error)
    re_flux = f1(wave_resamp)
    re_err = f2(wave_resamp)
    return np.array(re_flux), np.array(re_err)

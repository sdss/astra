import numpy as np
import os
from astra import log
from astra.tools.spectrum import Spectrum1D
from astra.utils import dict_to_list
from scipy.interpolate import interp1d

from astra.utils import expand_path
from astra.contrib.slam.slam.slam3 import Slam3 as Slam
from astra.contrib.slam.slam.normalization import normalize_spectra_block
from astra.database.astradb import database, Output, TaskOutput, SlamOutput

from astra.base import ExecutableTask, Parameter, TupleParameter, DictParameter
from astra.sdss.datamodels import create_AstraStar_product


class EstimateStellarLabels(ExecutableTask):

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

        model = Slam.load_dump(expand_path(self.model_path))
        wave_interp = model.wave

        log.info(f"Loaded model from {self.model_path}")

        for i, (task, data_products, parameters) in enumerate(self.iterable()):
            for j, data_product in enumerate(data_products):

                spectrum = Spectrum1D.read(data_product.path)

                wave = spectrum.spectral_axis.value
                fluxes = spectrum.flux.value
                invars = spectrum.uncertainty.array

                N, P = fluxes.shape

                fluxes_resamp, invars_resamp = [], []
                for i in range(N):
                    fluxes_temp, invars_temp = resample(
                        wave, fluxes[i], invars[i], wave_interp
                    )
                    fluxes_resamp += [fluxes_temp]
                    invars_resamp += [invars_temp]
                fluxes_resamp, invars_resamp = np.array(fluxes_resamp), np.array(
                    invars_resamp
                )

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
                u_teff = std_pred[:, 0]
                log_g = label_pred[:, 1]
                u_log_g = std_pred[:, 1]
                fe_h = label_pred[:, 2]
                u_fe_h = std_pred[:, 2]
                # alpha_m = label_pred[:,3]
                # u_alpha_m = std_pred[:,3]

                snr = fluxes * np.sqrt(invars)
                snr[snr <= 0] = np.nan
                snr = np.nanmean(snr, axis=1)

                # TODO: use 'pcov' covariance matrix and store correlation coefficients?
                #       Store cost? or any other things about fitting?

                prediction = model.predict_spectra(label_pred)
                chi_sq = (prediction - fluxes_norm) ** 2 * invars_norm
                reduced_chi_sq = np.sum(chi_sq, axis=1) / (P - L - 1)

                results = dict(
                    snr=snr.tolist(),
                    teff=teff.tolist(),
                    u_teff=u_teff.tolist(),
                    fe_h=fe_h.tolist(),
                    u_fe_h=u_fe_h.tolist(),
                    logg=log_g.tolist(),
                    u_logg=u_log_g.tolist(),
                    reduced_chi_sq=reduced_chi_sq.tolist()
                    # alpha_m=alpha_m.tolist(),
                    # u_alpha_m=u_alpha_m.tolist(),
                )

                # Create database records first, because these will be populated into the AstraStar product.
                with database.atomic():
                    for result in dict_to_list(results):
                        output = Output.create()
                        TaskOutput.create(task=task, output=output)
                        table_output = SlamOutput.create(
                            task=task, output=output, **result
                        )

                    log.info(
                        f"Created outputs {output} and {table_output} with {result}"
                    )

                # Create AstraStar product.
                continuum = fluxes_temp / fluxes_norm

                # Re-sample the predicted spectra back to the observed frame.
                f = interp1d(
                    wave_interp, prediction[0], kind="cubic", bounds_error=False
                )
                c = interp1d(
                    wave_interp, continuum[0], kind="cubic", bounds_error=False
                )

                rectified_flux = f(wave)
                model_flux = rectified_flux * c(wave)
                model_ivar = np.zeros_like(model_flux)

                crval = spectrum.meta["header"]["CRVAL1"]
                cdelt = spectrum.meta["header"]["CD1_1"]
                crpix = spectrum.meta["header"]["CRPIX1"]

                output_product = create_AstraStar_product(
                    task,
                    model_flux=model_flux,
                    model_ivar=model_ivar,
                    rectified_flux=rectified_flux,
                    crval=crval,
                    cdelt=cdelt,
                    crpix=crpix,
                )


def resample(wave, flux, err, wave_resamp, bounds_error=None):
    f1 = interp1d(wave, flux, kind="cubic", bounds_error=bounds_error)
    f2 = interp1d(wave, err, kind="cubic", bounds_error=bounds_error)
    re_flux = f1(wave_resamp)
    re_err = f2(wave_resamp)
    return np.array(re_flux), np.array(re_err)

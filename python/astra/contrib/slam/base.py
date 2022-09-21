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
            header_groups = {}
            database_results = []
            for spectrum in SpectrumList.read(data_product.path):
                if not spectrum_overlaps(spectrum, model.wave): 
                    continue

                wave = spectrum.spectral_axis.value
                fluxs = np.atleast_2d(spectrum.flux.value)
                ivars = np.atleast_2d(spectrum.uncertainty.represent_as(InverseVariance).array)

                N, P = fluxs.shape
                R = model.wave.size

                flux_resamp = np.empty((N, R))
                ivar_resamp = np.empty((N, R))
                for i in range(N):
                    # Note: One non-finite value given to scipy.interpolate.interp1d will cause the
                    #       entire interpolated output to be NaN. This is a known issue.
                    non_finite = ~np.isfinite(fluxs[i]) + ~np.isfinite(ivars[i])
                    _flux = np.copy(fluxs[i])
                    _flux[non_finite] = 0.0
                    _ivar = np.copy(ivars[i])
                    _ivar[non_finite] = 0.0

                    f = interp1d(wave, _flux, kind="cubic", bounds_error=None, fill_value=np.nan)
                    g = interp1d(wave, _ivar, kind="cubic", bounds_error=None, fill_value=0)
                    flux_resamp[i] = f(wave_interp)
                    ivar_resamp[i] = g(wave_interp)

                flux_norm, flux_cont = normalize_spectra_block(
                    wave_interp,
                    flux_resamp,
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

                ivar_norm = flux_cont**2 * ivar_resamp

                ### Initial estimation: get initial estimate of parameters by chi2 best match
                label_init = model.predict_labels_quick(
                    flux_norm, ivar_norm, n_jobs=1
                )

                ### SLAM prediction: optimize parameters
                results_pred = model.predict_labels_multi(
                    label_init, flux_norm, ivar_norm
                )
                label_names = ("teff", "logg", "fe_h")
                labels = np.array([label["x"] for label in results_pred])
                results = dict(zip(label_names, labels.T))
                results.update(
                    dict(zip(
                        [f"e_{ln}" for ln in label_names],
                        np.array([label["pstd"] for label in results_pred]).T
                    ))
                )

                # Add initial values.
                results.update(
                    dict(
                        zip(
                            [f"initial_{ln}" for ln in label_names],
                            label_init.T
                        )
                    )
                )

                # Add correlation coefficients.
                L = len(label_names)
                j, k = np.triu_indices(L, 1)
                rho = np.array([np.corrcoef(label["pcov"]) for label in results_pred])
                results.update(
                    dict(
                        zip(
                            [f"rho_{label_names[j]}_{label_names[k]}" for j, k in zip(j, k)],
                            rho[:, j, k].T
                        )
                    )
                )

                # Add optimisation keywords
                opt_keys = ("status", "success", "optimality")
                for result in results_pred:
                    for key in opt_keys:
                        results.setdefault(key, [])
                        results[key].append(result[key])
    
                # Add statistics.
                try:
                    snr = spectrum.meta["SNR"]
                except:
                    snr = fluxs * np.sqrt(ivars)
                    snr[snr <= 0] = np.nan
                    snr = np.nanmean(snr, axis=1)

                axis = 1
                prediction = model.predict_spectra(labels)
                chi_sq = np.sum((prediction - flux_norm) ** 2 * ivar_norm, axis=axis)
                R_finite = np.sum(ivar_norm > 0, axis=axis)
                reduced_chi_sq = chi_sq / (R_finite - L - 1)
                results.update(
                    snr=snr,
                    chi_sq=chi_sq,
                    reduced_chi_sq=reduced_chi_sq,
                )

                # Create AstraStar product.
                model_continuum = flux_resamp / flux_norm

                resampled_continuum = np.empty((N, P))
                resampled_model_flux = np.empty((N, P))
                for i in range(N):
                    assert np.all(np.isfinite(prediction)), "Prediction values not all finite?"
                    assert np.all(np.isfinite(model_continuum[i])), "Model continuum values not all finite?"
                    f = interp1d(wave_interp, prediction[i], kind="cubic", bounds_error=False, fill_value=np.nan)
                    c = interp1d(wave_interp, model_continuum[i], kind="cubic", bounds_error=False, fill_value=np.nan)

                    # Re-sample the predicted spectra back to the observed frame.
                    resampled_model_flux[i] = f(wave)
                    resampled_continuum[i] = c(wave)

                database_results.extend(dict_to_list(results))
                results.update(
                    spectral_axis=spectrum.spectral_axis,  
                    model_flux=resampled_model_flux,
                    continuum=resampled_continuum,
                )

                # Which extname should this go to?
                extname = get_extname(spectrum, data_product)
                all_results[extname] = results
                header_groups[extname] = [
                    ("TEFF", "STELLAR LABELS"),
                    ("INITIAL_TEFF", "INITIAL STELLAR LABELS"),
                    ("RHO_TEFF_LOGG", "CORRELATION COEFFICIENTS"),
                    ("STATUS", "OPTIMISATION INDICATORS"),
                    ("SNR", "SUMMARY STATISTICS"),
                    ("MODEL_FLUX", "MODEL SPECTRA")
                ]

            with database.atomic():
                task.create_or_update_outputs(SlamOutput, database_results)

                # Create astraStar/astraVisit data product and link it to this task.
                create_pipeline_product(
                    task, 
                    data_product, 
                    all_results,
                    header_groups=header_groups
                )
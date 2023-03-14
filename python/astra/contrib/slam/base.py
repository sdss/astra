import numpy as np
from astropy.nddata import InverseVariance
from astra.tools.spectrum import Spectrum1D, SpectrumList
from astra.tools.spectrum.utils import spectrum_overlaps
from scipy.interpolate import interp1d
from functools import cache
from typing import Iterable
from astropy import units as u

from joblib import load

from astra.base import task
from astra.utils import log, expand_path, list_to_dict, dict_to_list
from astra.contrib.slam.slam.slam3 import Slam3
from astra.contrib.slam.slam.normalization import normalize_spectra_block
from astra.database.astradb import DataProduct, SDSSOutput

from peewee import FloatField, IntegerField, BooleanField

@cache
def load_model(model_path):
    return load(expand_path(model_path))

class SlamOutput(SDSSOutput):

    # Initial values.
    initial_teff = FloatField(null=True)
    initial_logg = FloatField(null=True)
    initial_fe_h = FloatField(null=True)

    # Labels.
    teff = FloatField(null=True)
    e_teff = FloatField(null=True)
    logg = FloatField(null=True)
    e_logg = FloatField(null=True)
    fe_h = FloatField(null=True)
    e_fe_h = FloatField(null=True)
    
    # Correlation coefficients.
    rho_teff_logg = FloatField(null=True)
    rho_teff_fe_h = FloatField(null=True)
    rho_logg_fe_h = FloatField(null=True)

    # Optimisation outputs.
    success = BooleanField()
    status = IntegerField()
    optimality = BooleanField()

    # Statistics.
    chi_sq = FloatField()
    reduced_chi_sq = FloatField()


@task
def slam(
    data_product: Iterable[DataProduct],
    model_path: str = "$MWM_ASTRA/component_data/slam/ASPCAP_DR16_astra_wbinaryValid.dump",
    dwave: float = 10.0,
    p_min: float = 1e-8,
    p_max: float = 1e-7,
    q: float = 0.7,
    eps: float = 1e-19,
    rsv_frac: float = 2,
    n_jobs: int = 1,
    verbose: int = 5,
) -> Iterable[SlamOutput]:

    model = load_model(model_path)

    for data_product in data_product:
        for spectrum in SpectrumList.read(data_product.path):
            if not spectrum_overlaps(spectrum, 7_000 * u.Angstrom):
                # Ignore non-BOSS things
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
                flux_resamp[i] = f(model.wave)
                ivar_resamp[i] = g(model.wave)

            flux_norm, flux_cont = normalize_spectra_block(
                model.wave,
                flux_resamp,
                (6147.0, 8910.0),
                dwave=dwave,
                p=(p_min, p_max),
                q=q,
                ivar_block=ivar_resamp,
                eps=eps,
                rsv_frac=rsv_frac,
                n_jobs=n_jobs,
                verbose=verbose,
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
            label_names = ("teff", "fe_h")
            labels = np.array([label["x"] for label in results_pred])

            kwargs = (dict(zip(label_names, labels[0])))
            kwargs.update(
                dict(zip(
                    [f"e_{ln}" for ln in label_names],
                    np.array([label["pstd"] for label in results_pred])[0]
                ))
            )

            # Add initial values.
            kwargs.update(
                dict(
                    zip(
                        [f"initial_{ln}" for ln in label_names],
                        label_init[0]
                    )
                )
            )

            # Add correlation coefficients.
            L = len(label_names)
            j, k = np.triu_indices(L, 1)
            rho = np.array([np.corrcoef(label["pcov"]) for label in results_pred])
            kwargs.update(
                dict(
                    zip(
                        [f"rho_{label_names[j]}_{label_names[k]}" for j, k in zip(j, k)],
                        rho[0, j, k]
                    )
                )
            )

            # Add optimisation keywords
            opt_keys = ("status", "success", "optimality")
            for key in opt_keys:
                kwargs[key] = results_pred[0][key]

            # Add statistics.
            prediction = model.predict_spectra(labels)
            chi_sq = np.sum((prediction - flux_norm) ** 2 * ivar_norm)
            R_finite = np.sum(ivar_norm > 0)
            reduced_chi_sq = chi_sq / (R_finite - L - 1)
            kwargs.update(
                chi_sq=chi_sq,
                reduced_chi_sq=reduced_chi_sq,
            )

            # Prepare model spectrum for final product.
            model_continuum = flux_resamp / flux_norm

            resampled_continuum = np.nan * np.ones((N, P))
            resampled_rectified_model_flux = np.nan * np.ones((N, P))
            if not np.all(np.isfinite(prediction)):
                log.warning(f"Prediction values not all finite!")
            if not np.all(np.isfinite(model_continuum[i])):
                log.warning(f"Not all model continuum values finite!")

            for i in range(N):
                #assert np.all(np.isfinite(prediction)), "Prediction values not all finite?"
                #assert np.all(np.isfinite(model_continuum[i])), "Model continuum values not all finite?"
                finite_prediction = np.isfinite(prediction[i])
                finite_model_continuum = np.isfinite(model_continuum[i])
                if any(finite_prediction):

                    f = interp1d(
                        model.wave[finite_prediction], 
                        prediction[i][finite_prediction], 
                        kind="cubic", 
                        bounds_error=False, 
                        fill_value=np.nan
                    )
                    resampled_rectified_model_flux[i] = f(wave)

                if any(finite_model_continuum): 
                    c = interp1d(
                        model.wave[finite_model_continuum], 
                        model_continuum[i][finite_model_continuum], 
                        kind="cubic", 
                        bounds_error=False, 
                        fill_value=np.nan
                    )

                    # Re-sample the predicted spectra back to the observed frame.
                    resampled_continuum[i] = c(wave)

            # Send back the database result.
            output = SlamOutput(
                data_product=data_product,
                spectrum=spectrum,
                **kwargs
            )
            yield output


    print(f"Create the pipeline product now")
    '''
    create_pipeline_product(
        task, 
        data_product, 
        all_results,
        header_groups=header_groups
    )
    '''

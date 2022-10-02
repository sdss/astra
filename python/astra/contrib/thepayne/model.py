import numpy as np
from scipy.optimize import curve_fit
from astropy.constants import c
from astropy import units as u
from astropy.nddata import StdDevUncertainty
from astra import log
from astra.tools.spectrum import Spectrum1D, SpectrumList
#from astra.utils import
from typing import Union, Tuple, Optional
from collections import OrderedDict

SPEED_OF_LIGHT = c.to("km/s").value


def estimate_labels(
    spectrum: Union[Spectrum1D, SpectrumList],
    weights: Tuple[np.ndarray],
    biases: Tuple[np.ndarray],
    x_min: Tuple[np.ndarray],
    x_max: Tuple[np.ndarray],
    model_wavelength: Tuple[np.ndarray],
    label_names: Tuple[str],
    mask: Optional[np.array] = None,
    initial_labels: Optional[np.array] = None,
    continuum: Optional[np.array] = None,
    v_rad_tolerance: Optional[Union[float, int]] = None,
    opt_tolerance: Optional[float] = 5e-4,
    **kwargs,
):
    """
    Estimate the stellar labels given a spectrum, and the network weights, biases, and scales.

    :param spectrum:
        The input spectrum.
    """

    LARGE = kwargs.get("LARGE", 1e9)

    # number of label names
    K = weights[0].shape[1]
    L = 0 + K
    fit_v_rad = v_rad_tolerance is not None and v_rad_tolerance > 0
    if fit_v_rad:
        L += 1

    if initial_labels is None:
        initial_labels = np.zeros(L)

    bounds = np.zeros((2, L))
    bounds[0, :] = -0.5
    bounds[1, :] = +0.5
    if fit_v_rad:
        bounds[:, -1] = [-abs(v_rad_tolerance), +abs(v_rad_tolerance)]

    N, P = np.atleast_2d(spectrum.flux).shape

    p_opt = np.empty((N, L))
    p_cov = np.empty((N, L, L))
    model_flux = np.empty((N, P))
    meta = []

    def objective_function(x, *labels):
        y_pred = predict_stellar_spectrum(labels[:K], weights, biases)
        if fit_v_rad:
            y_pred = redshift_spectrum(x, y_pred, labels[-1])
        return y_pred

    wavelength = spectrum.wavelength.value
    all_flux = np.atleast_2d(spectrum.flux.value)
    all_e_flux = np.atleast_2d(
        spectrum.uncertainty.represent_as(StdDevUncertainty).array
    )

    if continuum is not None:
        all_flux /= continuum
        all_e_flux /= continuum

    if mask is None:
        mask = np.zeros(model_wavelength.shape, dtype=bool)
    else:
        assert (
            mask.shape == model_wavelength.shape
        ), "Mask and model wavelengths do not have the same shape"

    results = []
    meta_results = []
    kwds = kwargs.copy()
    for i in range(N):

        # Interpolate data onto model wavelengths -- not The Right Thing to do!
        flux = np.interp(model_wavelength, wavelength, all_flux[i], left=1, right=1)
        e_flux = np.interp(
            model_wavelength, wavelength, all_e_flux[i], left=LARGE, right=LARGE
        )
        e_flux[mask] = LARGE

        # Fix non-finite pixels and error values.
        non_finite = ~np.isfinite(flux) + ~np.isfinite(e_flux)
        flux[non_finite] = 1
        e_flux[non_finite] = LARGE

        # "normalize"
        scale = np.median(flux)
        flux /= scale
        e_flux /= scale

        kwds.update(
            xdata=model_wavelength,
            ydata=flux,
            sigma=e_flux,
            p0=initial_labels,
            bounds=bounds,
            absolute_sigma=True,
            method="trf",
            xtol=opt_tolerance,
            ftol=opt_tolerance,
        )

        try:
            p_opt, p_cov = curve_fit(objective_function, **kwds)

        except ValueError:
            log.exception(f"Error occurred fitting spectrum {i}:")
            results.append((None, None, {}))

        else:
            labels = (p_opt + 0.5) * (x_max - x_min) + x_min
            e_labels = np.sqrt(np.diag(p_cov)) * (x_max - x_min)

            result = OrderedDict(zip(label_names, labels))
            result.update(OrderedDict(zip([f"e_{ln}" for ln in label_names], e_labels)))

            rho = np.corrcoef(p_cov)
            for j, k in zip(*np.triu_indices(L, 1)):
                result[f"rho_{label_names[j]}_{label_names[k]}"] = rho[j, k]

            # Interpolate model_flux back onto the observed wavelengths.
            model_flux = objective_function(model_wavelength, *p_opt)
            resampled_model_flux = np.interp(
                wavelength, model_wavelength, model_flux, left=np.nan, right=np.nan
            )
            chi_sq = np.sum(((model_flux - flux) / e_flux) ** 2)
            reduced_chi_sq = chi_sq / (model_flux.size - L - 1)
            result.update(
                snr=spectrum.meta["SNR"][i],
                chi_sq=chi_sq, 
                reduced_chi_sq=reduced_chi_sq
            )
            meta = OrderedDict([("model_flux", resampled_model_flux)])
            if continuum is not None:
                resampled_model_flux *= continuum[i]
                meta["continuum"] = continuum[i]
            
            results.append(result)
            meta_results.append(meta)

    return (results, meta_results)


def leaky_relu(z):
    return z * (z > 0) + 0.01 * z * (z < 0)


def predict_stellar_spectrum(unscaled_labels, weights, biases):
    inside = np.einsum("ij,j->i", weights[0], unscaled_labels) + biases[0]
    outside = np.einsum("ij,j->i", weights[1], leaky_relu(inside)) + biases[1]
    return np.einsum("ij,j->i", weights[2], leaky_relu(outside)) + biases[2]


def redshift_spectrum(dispersion, flux, radial_velocity):
    f = np.sqrt(
        (1 - radial_velocity / SPEED_OF_LIGHT) / (1 + radial_velocity / SPEED_OF_LIGHT)
    )
    return np.interp(f * dispersion, dispersion, flux)

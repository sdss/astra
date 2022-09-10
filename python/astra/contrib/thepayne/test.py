# encoding: utf-8


from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import sys
import os
import torch
from tqdm import trange
import pickle

from astropy.table import Table
from astropy.constants import c
from scipy.optimize import curve_fit
from collections import OrderedDict

from astra import log

# TODO: put this elsewhere
# Check for CUDA support.
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = "cpu"
    log.warning("Torch not compiled with CUDA support")

LARGE = 1e3

c = c.to("km/s").value

sigmoid = lambda z: 1.0 / (1.0 + np.exp(-z))


def _predict_stellar_spectrum(unscaled_labels, weights, biases):

    # This is making extremely strong assumptions about the neural network architecture!
    inside = np.einsum("ij,j->i", weights[0], unscaled_labels) + biases[0]
    outside = np.einsum("ij,j->i", weights[1], sigmoid(inside)) + biases[1]
    spectrum = np.einsum("ij,j->i", weights[2], sigmoid(outside)) + biases[2]
    return spectrum


# TODO: use a specutils.Spectrum1D method.
def _redshift(dispersion, flux, radial_velocity):
    f = np.sqrt((1 - radial_velocity / c) / (1 + radial_velocity / c))
    new_dispersion = f * dispersion
    return np.interp(new_dispersion, dispersion, flux)


def load_state(path):
    with open(path, "rb") as fp:
        contents = torch.load(fp, map_location=device)

    state = contents["state"]

    N = len(state["model_state"])
    biases = [state["model_state"][f"{i}.bias"].data.cpu().numpy() for i in (0, 2, 4)]
    weights = [
        state["model_state"][f"{i}.weight"].data.cpu().numpy() for i in (0, 2, 4)
    ]
    scales = state["scales"]

    return dict(
        neural_network_coefficients=(weights, biases),
        scales=scales,
        model_wavelength=contents["wavelength"],
        label_names=contents["label_names"],
    )


def test(
    wavelength,
    flux,
    ivar,
    neural_network_coefficients,
    scales,
    model_wavelength,
    label_names,
    initial_labels=None,
    radial_velocity_tolerance=None,
    mask=None,
    **kwargs,
):
    r"""
    Use a pre-trained neural network to estimate the stellar labels for the given spectrum.

    :param wavelength:
        The wavelength array of the observed spectrum.

    :param flux:
        The observed (or pseudo-continuum-normalised) fluxes.

    :param ivar:
        The inverse variances of the fluxes.

    :param neural_network_coefficients:
        A two-length tuple containing the weights of the neural network, and the biases.

    :param scales:
        The lower and upper scaling value used for the labels.

    :param initial_labels: [optional]
        The initial labels to optimize from. By default this will be set at the center of the
        training set labels.

    :param radial_velocity_tolerance: [optional]
        Supply a radial velocity tolerance to fit simulatenously with stellar parameters. If `None`
        is given then no radial velocity will be fit. If a float/integer is given then any radial
        velocity +/- that value will be considered. Alternatively, a (lower, upper) bound can be
        given.

    :param mask: [optional]
        An boolean mask to apply when fitting. If `None` is given then no masking will be applied.

    :returns:
        A three-length tuple containing the optimized parameters, the covariance matrix, and a
        metadata dictionary.
    """

    weights, biases = neural_network_coefficients
    K = L = weights[0].shape[1]  # number of label names

    fit_radial_velocity = radial_velocity_tolerance is not None
    if fit_radial_velocity:
        L += 1

    if initial_labels is None:
        initial_labels = np.zeros(L)

    # Set bounds.
    bounds = np.zeros((2, L))
    bounds[0, :] = -0.5
    bounds[1, :] = +0.5
    if fit_radial_velocity:
        if isinstance(radial_velocity_tolerance, (int, float)):
            bounds[:, -1] = [
                -abs(radial_velocity_tolerance),
                +abs(radial_velocity_tolerance),
            ]

        else:
            bounds[:, -1] = radial_velocity_tolerance

    N, P = flux.shape

    p_opts = np.nan * np.ones((N, L))
    p_covs = np.nan * np.ones((N, L, L))
    model_fluxes = np.nan * np.ones((N, P))
    flux_sigma = np.nan * np.ones((N, P))
    meta = []

    x_min, x_max = scales

    # Define objective function.
    def objective_function(x, *labels):
        y_pred = _predict_stellar_spectrum(labels[:K], weights, biases)
        if fit_radial_velocity:
            # Here we are shifting the *observed* spectra. That's not the Right Thing to do, but it
            # probably doesn't matter here.
            y_pred = _redshift(x, y_pred, labels[-1])
        return y_pred

    for i in range(N):
        y_original = flux[i].reshape(wavelength.shape)
        # TODO: Assuming an inverse variance array (likely true).
        y_err_original = ivar[i].reshape(wavelength.shape) ** -0.5
        if mask is not None:
            y_err_original[mask] = 999

        # Interpolate data onto model -- not The Right Thing to do!
        y = np.interp(model_wavelength, wavelength, y_original)
        y_err = np.interp(model_wavelength, wavelength, y_err_original)

        # Fix non-finite pixels and error values.
        non_finite = ~np.isfinite(y * y_err)
        y[non_finite] = 1
        y_err[non_finite] = LARGE

        kwds = kwargs.copy()
        kwds.update(
            xdata=model_wavelength,
            ydata=y,
            sigma=y_err,
            p0=initial_labels,
            bounds=bounds,
            absolute_sigma=True,
            method="trf",
        )

        try:
            p_opt, p_cov = curve_fit(objective_function, **kwds)

        except ValueError:
            log.exception(f"Error occurred fitting spectrum {i}:")
            meta.append(dict(chi_sq=np.nan, reduced_chi_sq=np.nan))

        else:
            y_pred = objective_function(model_wavelength, *p_opt)

            # Calculate summary statistics.
            chi_sq, reduced_chi_sq = get_chi_sq(y_pred, y, y_err, L)

            p_opts[i, :] = (x_max - x_min) * (p_opt + 0.5) + x_min
            p_covs[i, :, :] = p_cov * (x_max - x_min)
            model_fluxes[i, :] = np.interp(
                wavelength,
                model_wavelength,
                y_pred,
            )
            meta.append(
                dict(
                    chi_sq=chi_sq,
                    reduced_chi_sq=reduced_chi_sq,
                )
            )

    return (p_opts, p_covs, model_fluxes, meta)


def get_chi_sq(expectation, y, y_err, L):
    P = np.sum(np.isfinite(y * y_err) * (y_err < LARGE))
    chi_sq = np.nansum(((y - expectation) / y_err) ** 2)
    r_chi_sq = chi_sq / (P - L - 1)
    return (chi_sq, r_chi_sq)

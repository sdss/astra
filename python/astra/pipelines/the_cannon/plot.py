#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plotting utilities for The Cannon.
"""

from __future__ import division, print_function, absolute_import, unicode_literals

__all__ = ["theta", "scatter", "one_to_one"]

import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

except ImportError:
    logger.warn("Could not import matplotlib; plotting functionality disabled")


def plot_sparsity(models):

    mask = model.theta[:,]

    regularization = np.array([model.regularization for model in models])
    sparsity = np.array([np.sum(model.theta == 0)/model.theta.size for model in models])
    
    sort_indices = np.argsort(regularization)
    
    fig, ax = plt.subplots()
    ax.plot(regularization[sort_indices], sparsity[sort_indices], c="k", lw=2, label="all terms")    
    for label, indices in zip(["linear terms", "quadratic terms", "cross terms"], models[0].term_type_indices):
        term_sparsity = np.array([np.sum(model.theta[indices] == 0)/model.theta[indices].size for model in models])        
        ax.plot(regularization[sort_indices], term_sparsity[sort_indices], label=label)
    
    ax.legend(frameon=False)
    ax.semilogx()
    ax.set_xlabel(r"Regularization")
    ax.set_ylabel(r"Sparsity")
    ax.axhline(1, c="#666666", ls=":", lw=0.5)
    fig.tight_layout()

    return fig

from scipy.interpolate import splrep, splev

def plot_validation_chisq(models, validation_flux, validation_ivar, validation_labels):
    
    regularization = np.array([model.regularization for model in models])
    chi2 = np.zeros_like(regularization)    
    chi2_with_model_scatter = np.zeros_like(regularization)
    for i, model in enumerate(tqdm(models)):
        adjusted_ivar = validation_ivar/(1 + validation_ivar*model.s2)
        chi2[i] = np.sum((model.predict(validation_labels) - validation_flux)**2 * validation_ivar)
        chi2_with_model_scatter[i] = np.sum((model.predict(validation_labels) - validation_flux)**2 * adjusted_ivar)    
    sort_indices = np.argsort(regularization)

    x, y = (regularization[sort_indices], chi2[sort_indices]/chi2[sort_indices][0])
    ya = chi2_with_model_scatter[sort_indices]/chi2_with_model_scatter[sort_indices][0]
    
    dy = 1.5 * (1 - np.min(y))
    
    fig, ax = plt.subplots()
    ax.scatter(x, y, facecolor="tab:blue")
    ax.scatter(x, ya, facecolor="tab:orange")
    
    tck = splrep(x, y)
    xi = np.logspace(np.log10(x.min()), np.log10(x.max()), 1000)
    yi = splev(xi, tck)
    
    tck_a = splrep(x, ya)
    yai = splev(xi, tck_a)
    ax.plot(xi, yi, c="tab:blue")
    ax.plot(xi, yai, c="tab:orange")
    
    sj = np.argmin(yi)
    ej = np.where(np.diff(np.sign(yi - 1)))[0][1]
    
    ax.semilogx()
    ax.set_ylim(1 - dy, 1 + dy)
    ax.axhline(1, c="#666666", ls=":", lw=0.5, zorder=-1)
    ax.axvline(xi[sj], c="#666666", ls="--", lw=0.5, zorder=-1)
    ax.axvline(xi[ej], c="#666666", ls="--", lw=0.5, zorder=-1)
    ylim = ax.get_ylim()
    ax.axvspan(
        xi[sj],
        xi[ej],
        ymin=-10,
        ymax=+10,
        facecolor="#cccccc",
        zorder=-10        
    )
    print(xi[sj], xi[ej])
    return fig


def plot_gridsearch_chisq(
    regularization, train_chisq, validation_chisq, ax=None, **kwargs
):
    if ax is None:
        fig, ax = plt.subplots()
        new_figure = True
    else:
        fig = ax.figure
        new_figure = False
    ax.plot(
        regularization,
        validation_chisq / validation_chisq[0],
        label="validation",
        c="tab:blue",
        **kwargs,
    )
    ax.plot(
        regularization,
        train_chisq / train_chisq[0],
        label="train",
        c="tab:green",
        **kwargs,
    )

    if new_figure:
        ax.axhline(1, c="#666666", ls=":", lw=0.5, zorder=-1)

        ax.legend()
        ax.semilogx()
        ax.set_xlabel(r"Regularization")
        ax.set_ylabel(r"$\chi^2$ relative to baseline")
        fig.tight_layout()

    return fig


def plot_theta(
    theta,
    indices,
    dispersion=None,
    label_terms=None,
    show_label_terms=True,
    normalize=True,
    common_axis=False,
    latex_label_names=None,
    xlim=None,
    **kwargs,
):

    K = len(indices)

    fig, axes = plt.subplots(K)
    axes = np.array([axes]).flatten()

    if common_axis:
        raise NotImplementedError

    if dispersion is None:
        x = np.arange(theta.shape[1])
    else:
        x = dispersion

    plot_kwds = dict(c="b", lw=1)
    plot_kwds.update(kwargs.get("plot_kwds", {}))

    for i, (ax, index) in enumerate(zip(axes, indices)):

        y = theta[index]
        # scale = np.max(np.abs(y)) if normalize and label_index != 0 else 1.0
        scale = 1

        ax.plot(x, y / scale, **plot_kwds)

        if ax.is_last_row():
            if dispersion is None:
                xlabel = r"${\rm Pixel}$"
            else:
                xlabel = r"${\rm Wavelength},$ $({\rm AA})$"
            ax.set_xlabel(xlabel)

        else:
            ax.set_xticklabels([])

        # Set RHS label.
        ax.xaxis.set_major_locator(MaxNLocator(6))

        ax.set_xlim(xlim)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.10)

    return fig


def scatter(model, ax=None, **kwargs):
    """
    Plot the noise residuals (:math:`s`) at each pixel.

    :param model:
        A trained CannonModel object.

    :returns:
        A figure showing the noise residuals at every pixel.
    """

    if not model.is_trained:
        raise ValueError("model needs to be trained first")

    fig = None
    if ax is None:
        fig, ax = plt.subplots()

    if model.dispersion is None:
        x = np.arange(model.s2.size)
    else:
        x = model.dispersion

    plot_kwds = dict(lw=1, c="b")
    plot_kwds.update(kwargs.pop("plot_kwds", {}))

    ax.plot(x, model.s2**0.5, **plot_kwds)

    if model.dispersion is None:
        ax.set_xlabel(r"${\rm Pixel}$")
    else:
        ax.set_xlabel(r"${\rm Wavelength}$ $[{\rm \AA}]$")

    ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_ylabel(r"${\rm Scatter},$ $s$")

    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))

    if fig is not None:
        fig.tight_layout()
    else:
        fig = ax.figure

    return fig


def plot_labels(known_labels, test_labels, label_names, show_statistics=True, **kwargs):
    N, K = np.atleast_2d(known_labels).shape

    factor = 2.0
    lbdim = 0.30 * factor
    tdim = 0.25 * factor
    rdim = 0.10 * factor
    wspace = 0.05
    hspace = 0.35
    yspace = factor * K + factor * (K - 1.0) * hspace
    xspace = factor

    xdim = lbdim + xspace + rdim
    ydim = lbdim + yspace + tdim

    fig, axes = plt.subplots(K, figsize=(xdim, ydim))

    l, b = (lbdim / xdim, lbdim / ydim)
    t, r = ((lbdim + yspace) / ydim, ((lbdim + xspace) / xdim))

    fig.subplots_adjust(left=l, bottom=b, right=r, top=t, wspace=wspace, hspace=hspace)

    axes = np.array([axes]).flatten()

    scatter_kwds = dict(s=1, c="k", alpha=0.5, edgecolor=None, lw=0)
    scatter_kwds.update(kwargs.get("scatter_kwds", {}))

    errorbar_kwds = dict(fmt="None", ecolor="k", alpha=0.5, capsize=0)
    errorbar_kwds.update(kwargs.get("errorbar_kwds", {}))

    for i, ax in enumerate(axes):

        x = known_labels[:, i]
        y = test_labels[:, i]

        ax.scatter(x, y, **scatter_kwds)
        # if cov is not None:
        #    yerr = cov[:, i, i]**0.5
        #    ax.errorbar(x, y, yerr=yerr, **errorbar_kwds)

        # Set x-axis limits and y-axis limits the same
        limits = np.array([ax.get_xlim(), ax.get_ylim()])
        limits = (np.min(limits), np.max(limits))

        ax.plot(limits, limits, c="#666666", linestyle=":", zorder=-1)
        ax.set_xlim(limits)
        ax.set_ylim(limits)

        try:
            ax.set_title(label_names[i])
        except:
            None

        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))

        # Show mean and sigma.
        if show_statistics:
            diff = y - x
            mu = np.nanmedian(diff)
            sigma = np.nanstd(diff)
            ax.text(0.05, 0.85, r"$\mu = {0:.2f}$".format(mu), transform=ax.transAxes)
            ax.text(
                0.05, 0.75, r"$\sigma = {0:.2f}$".format(sigma), transform=ax.transAxes
            )
            ax.text(
                0.05,
                0.65,
                r"$N = {:.0f}$".format(np.sum(np.isfinite(diff))),
                transform=ax.transAxes,
            )

        ax.set_aspect(1.0)

    return fig


def one_to_one(
    model, test_labels, cov=None, latex_label_names=None, show_statistics=True, **kwargs
):
    """
    Plot a one-to-one comparison of the training set labels, and the test set
    labels inferred from the training set spectra.

    :param model:
        A trained CannonModel object.

    :param test_labels:
        An array of test labels, inferred from the training set spectra.

    :param cov: [optional]
        The covariance matrix returned for all test labels.

    :param latex_label_names: [optional]
        A list of label names in LaTeX representation.

    :param show_statistics: [optional]
        Show the mean and standard deviation of residuals in each axis.
    """

    if model.training_set_labels.shape != test_labels.shape:
        raise ValueError("test labels must have the same shape as training set labels")

    N, K = test_labels.shape
    if cov is not None and cov.shape != (N, K, K):
        raise ValueError(
            "shape mis-match in covariance matrix ({N}, {K}, {K}) != {shape}".format(
                N=N, K=K, shape=cov.shape
            )
        )

    factor = 2.0
    lbdim = 0.30 * factor
    tdim = 0.25 * factor
    rdim = 0.10 * factor
    wspace = 0.05
    hspace = 0.35
    yspace = factor * K + factor * (K - 1.0) * hspace
    xspace = factor

    xdim = lbdim + xspace + rdim
    ydim = lbdim + yspace + tdim

    fig, axes = plt.subplots(K, figsize=(xdim, ydim))

    l, b = (lbdim / xdim, lbdim / ydim)
    t, r = ((lbdim + yspace) / ydim, ((lbdim + xspace) / xdim))

    fig.subplots_adjust(left=l, bottom=b, right=r, top=t, wspace=wspace, hspace=hspace)

    axes = np.array([axes]).flatten()

    scatter_kwds = dict(s=1, c="k", alpha=0.5)
    scatter_kwds.update(kwargs.get("scatter_kwds", {}))

    errorbar_kwds = dict(fmt="None", ecolor="k", alpha=0.5, capsize=0)
    errorbar_kwds.update(kwargs.get("errorbar_kwds", {}))

    for i, ax in enumerate(axes):

        x = model.training_set_labels[:, i]
        y = test_labels[:, i]

        ax.scatter(x, y, **scatter_kwds)
        if cov is not None:
            yerr = cov[:, i, i] ** 0.5
            ax.errorbar(x, y, yerr=yerr, **errorbar_kwds)

        # Set x-axis limits and y-axis limits the same
        limits = np.array([ax.get_xlim(), ax.get_ylim()])
        limits = (np.min(limits), np.max(limits))

        ax.plot(limits, limits, c="#666666", linestyle=":", zorder=-1)
        ax.set_xlim(limits)
        ax.set_ylim(limits)

        label_name = model.vectorizer.label_names[i]

        if latex_label_names is not None:
            try:
                label_name = r"${}$".format(latex_label_names[i])
            except:
                logger.warn(
                    "Could not access latex label name for index {} ({})".format(
                        i, label_name
                    )
                )

        ax.set_title(label_name)

        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))

        # Show mean and sigma.
        if show_statistics:
            diff = y - x
            mu = np.nanmedian(diff)
            sigma = np.nanstd(diff)
            ax.text(0.05, 0.85, r"$\mu = {0:.2f}$".format(mu), transform=ax.transAxes)
            ax.text(
                0.05, 0.75, r"$\sigma = {0:.2f}$".format(sigma), transform=ax.transAxes
            )
            ax.text(
                0.05,
                0.65,
                r"$N = {:.0f}$".format(np.sum(np.isfinite(diff))),
                transform=ax.transAxes,
            )

        ax.set_aspect(1.0)

    return fig

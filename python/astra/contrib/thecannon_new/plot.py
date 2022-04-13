#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plotting utilities for The Cannon.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["theta", "scatter", "one_to_one"]

import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

except ImportError:
    logger.warn("Could not import matplotlib; plotting functionality disabled")    


def plot_theta(theta, indices, dispersion=None, label_terms=None, show_label_terms=True,
    normalize=True, common_axis=False, latex_label_names=None, xlim=None, 
    **kwargs):

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
        #scale = np.max(np.abs(y)) if normalize and label_index != 0 else 1.0
        scale = 1

        ax.plot(x, y/scale, **plot_kwds)

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


def plot_labels(
    known_labels,
    test_labels,
    label_names,
    show_statistics=True,
    **kwargs
):
    N, K = np.atleast_2d(known_labels).shape

    factor = 2.0
    lbdim = 0.30 * factor
    tdim = 0.25 * factor
    rdim = 0.10 * factor
    wspace = 0.05
    hspace = 0.35
    yspace = factor * K + factor * (K - 1.) * hspace
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
        #if cov is not None:
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
            ax.text(0.05, 0.85, r"$\mu = {0:.2f}$".format(mu),
                transform=ax.transAxes)
            ax.text(0.05, 0.75, r"$\sigma = {0:.2f}$".format(sigma),
                transform=ax.transAxes)
            ax.text(0.05, 0.65, r"$N = {:.0f}$".format(np.sum(np.isfinite(diff))),
                    transform=ax.transAxes)
        
        ax.set_aspect(1.0)

    return fig



def one_to_one(model, test_labels, cov=None, latex_label_names=None,
    show_statistics=True, **kwargs):
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
        raise ValueError(
            "test labels must have the same shape as training set labels")

    N, K = test_labels.shape
    if cov is not None and cov.shape != (N, K, K):
        raise ValueError(
            "shape mis-match in covariance matrix ({N}, {K}, {K}) != {shape}"\
            .format(N=N, K=K, shape=cov.shape))

    factor = 2.0           
    lbdim = 0.30 * factor
    tdim = 0.25 * factor
    rdim = 0.10 * factor
    wspace = 0.05
    hspace = 0.35
    yspace = factor * K + factor * (K - 1.) * hspace
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
            yerr = cov[:, i, i]**0.5
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
                    "Could not access latex label name for index {} ({})"\
                    .format(i, label_name))

        ax.set_title(label_name)

        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))

        # Show mean and sigma.
        if show_statistics:
            diff = y - x
            mu = np.nanmedian(diff)
            sigma = np.nanstd(diff)
            ax.text(0.05, 0.85, r"$\mu = {0:.2f}$".format(mu),
                transform=ax.transAxes)
            ax.text(0.05, 0.75, r"$\sigma = {0:.2f}$".format(sigma),
                transform=ax.transAxes)
            ax.text(0.05, 0.65, r"$N = {:.0f}$".format(np.sum(np.isfinite(diff))),
                    transform=ax.transAxes)
        
        ax.set_aspect(1.0)

    return fig

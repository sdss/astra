from collections import deque
from datetime import date
from typing import Callable, List, Optional

import matplotlib  # noqa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa
import scipy.stats as stats
from matplotlib.axes import Axes
from matplotlib.ticker import FixedLocator, PercentFormatter
from numpy.typing import ArrayLike
from pandas import DataFrame
from scipy.stats import binned_statistic
from tqdm import tqdm

from util import Lookup, pdf_gen, get_cmap

# matplotlib backend render fancy
plt.style.use("science")
params = {
    "text.usetex": True,
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "font.serif": "",
    "legend.frameon": True,
    "figure.figsize": (7 / 2.54, 7 / 2.54),
}
plt.rcParams.update(params)


def plot_pairwise(
    data: pd.DataFrame,
    vars: Optional[ArrayLike] = None,
    figpath: str = "/home/riley/uni/rproj/figs/",
    adjust: Optional[str] = None,
    overlap: bool = False,
    show: bool = False,
):
    """
    Plot Pairwise Distance histogram distribution against standard normal PDF

    :param data:
        DataFrame of data, containing z-scores of respective data.
    :param vars:
        List of variables to plot for. Defaults to all variables.
    :param figpath:
        Where to save figure to. Defaults to Riley's workspace.
    :param adjust:
        Whether to plot with systematic (sys), fractional (frac), or gridsearch (grid)
        adjustments. Defaults to None.
    :param overlap:
        Whether to plot in multiple figures or just as a single figure.
        Defaults to False.
    """
    plt.close("all")
    # Vars cases
    # must be done in two stages since .remove method modifies in place
    # TODO: this is really poorly handled, but I've yet to fix it properly. Need to make
    # it so this isn't just a really poor patch based on the gridsearch data table.
    if vars is None:
        vars = list(data.columns.values)
        adj_vars = deque()
        gs_vars = list()
        param_vars = deque()

        for var in vars:
            if "+" in var:
                adj_vars.append(var)
            elif ("_GS" in var) and ("_E" not in var):
                gs_vars.append(var)
            elif ("_t" in var) or ("_E" in var) or ("_SNR" in var) or ("_TEFF"
                                                                       in var):
                param_vars.append(var)
        for var in adj_vars:
            vars.remove(var)
        for var in gs_vars:
            vars.remove(var)
        for var in param_vars:
            vars.remove(var)
    else:
        adj_vars = deque()
        gs_vars = deque()

        for var in vars:
            if "+" in var:
                adj_vars.append(var)
            elif "_" in var:
                gs_vars.append(var)

    notplot = True
    fig, ax = plt.subplots()
    if adjust == "grid":
        vars = gs_vars
    for var in tqdm(vars):
        # Parse the given dataframe
        subdata = data[var].dropna().values

        # fractional adjustment and label casing
        if adjust == "frac":
            subdata = subdata / np.std(subdata)
            label = r"{varname}({mu},1)".format(mu=round(np.mean(subdata), 2),
                                                varname=var)
        else:
            label = r"{varname}({mu},{std})".format(
                mu=round(np.mean(subdata), 2),
                std=round(np.std(subdata), 2),
                varname=var,
            )

        # set binwidth
        binwidth = 0.05
        bins = np.arange(np.min(subdata), np.max(subdata) + binwidth, binwidth)

        # generate figure
        if not overlap:
            fig, ax = plt.subplots()
            notplot = True
        ax: Axes  # cheap fix for type PyRight bug
        ax.hist(
            subdata,
            bins=bins,
            density=True,
            histtype="step",
            zorder=25,
            alpha=0.6,
            label=label,
        )
        if adjust == "sys":
            # loop for every +- casing
            # index extensions for systematic adjustment
            for adj in adj_vars:
                if var in adj:
                    # Parse the given dataframe for extension
                    subdata = data[adj].dropna().values

                    # fractional adjustment and label casing
                    label = var + "[+" + str(
                        Lookup.convert_extension(adj)) + "]"

                    # set binwidth
                    binwidth = 0.05
                    bins = np.arange(np.min(subdata),
                                     max(subdata) + binwidth, binwidth)
                    ax.hist(
                        subdata,
                        bins=bins,
                        density=True,
                        histtype="step",
                        zorder=25,
                        alpha=0.6,
                        label=label,
                    )

        # labels, titles, text, grid, etc
        ax.set_xlabel("$z$")
        ax.set_ylabel("Normalized Frequency [unitless]")
        # ax.set_title(f"Histogram of pairwise distances for {var}")
        ax.set_ylim(0, 0.8)
        ax.set_xlim(-3, 3)

        # normal PDF
        if notplot:
            points, pdf = pdf_gen(0, 1, stats.norm)
            ax.plot(
                points,
                pdf,
                color="blue",
                linestyle="dashed",
                linewidth=1,
                alpha=0.8,
                zorder=100,
                label=r"$\mathcal{N}(0,1)$",
            )
            ax.grid(linewidth=0.5, alpha=0.5, zorder=0)
            plt.text(
                0.03,
                0.97,
                "$n = $ {no}".format(no=len(subdata)),
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax.transAxes,
            )
            notplot = False

        plt.legend(bbox_to_anchor=(1, 1), fontsize=0.3 * 18)

        # save figure if not in overlap mode
        if not overlap:
            # show if not in overlap mode
            tdy = str(date.today())
            filename = figpath + f"z_{var}_{tdy}"
            if adjust == "frac":
                filename = filename + "_frc"
            if adjust == "sys":
                filename = filename + "_sys"
            filename = filename + ".pdf"
            fig.savefig(filename, bbox_inches="tight")
            if show:
                plt.show(block="False")
            plt.close("all")  # memory saving

    # save complete figure in overlap mode
    if overlap:
        tdy = str(date.today())
        filename = figpath + f"z_{tdy}"
        if adjust == "frac":
            filename = filename + "_frc"
        if adjust == "sys":
            filename = filename + "_sys"
        filename = filename + ".pdf"
        fig.savefig(filename, bbox_inches="tight")
        if show:
            plt.show(block="False")

    return


def plot_visit_frequency(data: pd.DataFrame | ArrayLike):
    """
    Plot visit frequency for a given pairwise distance score.

    :param data: DataFrame of z-scores with the LENGTH, or array of lengths.
    """
    plt.close("all")
    # Get values
    data = data["LENGTH"].dropna().values

    # set bins up
    binwidth = 1
    bins = np.arange(np.min(data), np.max(data) + binwidth, binwidth)

    # plot!
    fig, ax = plt.subplots()
    ax: Axes
    ax.hist(
        data,
        bins=bins,
        color="green",
        log=True,
        edgecolor="black",
        density=True,
        weights=np.ones(len(data)) / len(data),
        alpha=0.7,
        zorder=100,
    )

    # axes logarithmic and correct locator formatting
    ax.xaxis.set_major_locator(FixedLocator(np.arange(1, max(data) + 1, 1)))
    ax.set_xlabel("Number of visits")
    ax.set_xlim(1, max(data))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=2))
    ax.set_ylabel("Percentage of sample")
    ax.grid(linewidth=0.5, alpha=0.5, zorder=0)

    tdy = date.today()
    plt.savefig(f"figs/z_visitfreq_{tdy}.pdf", dpi=300, bbox_inches="tight")


def plot_delta_coadd(
    data: pd.DataFrame,
    vars: Optional[List[str] | str] = None,
    figpath: str = "/home/riley/uni/rproj/figs/",
    type: Callable = np.median,
    plot_scatter: bool = False,
    overlap: bool = False,
    show: bool = False,
):
    """
    Plot a binned scatter plot of the difference between coadd and individual visit
    stellar parameters against SNR.

    :param data:
        DataFrame of coadd data, with SNR and Telescope rows.
    :param vars:
        Names of stellar parameter to plot for. Defaults to all in data.
    :param figpath:
        Directory to save file to. Defaults to Riley's workspace
    :param type:
        Type of function to calculate binned statistic for. Defaults to median.
    :param plot_scatter:
        Boolean of whether to plot scatter points of raw data.
    :param overlap:
        Whether to plot in multiple figures or just as a single figure.
        Defaults to False.
    :param show:
        Whether to show the figure interactively. Defaults to False.
    """
    plt.close("all")
    # Default case for variable list -- all in data.
    if vars is None:
        vars = list(data.columns.values)
        vars.remove("SNR")
        vars.remove("TELESCOPE")

    fig, ax = plt.subplots()
    for var in tqdm(vars):
        # obtain delta
        # drop NaN subtable based on variable
        subdata = data.dropna(subset=[var])

        # obtain SNR
        snr_visit = subdata["SNR"].values
        vardata = subdata[var].values

        # Split our array into rms bins
        binwidth = 10
        bins = np.arange(np.min(snr_visit),
                         max(snr_visit) + binwidth, binwidth)
        delta_rms, rms_bins, indices = binned_statistic(snr_visit,
                                                        vardata,
                                                        bins=bins,
                                                        statistic=type)

        # plot scatter and RMS binned stats for coadd delta
        if not overlap:
            # instantiate new object everytime if overlap
            fig, ax = plt.subplots()
        ax: Axes  # fix angry type hinting PyRight bug
        if plot_scatter:
            ax.scatter(
                snr_visit,
                vardata,
                alpha=0.1,
                s=1,
                marker="s",
                zorder=25,
                label="raw values",
            )
        ax.plot(
            rms_bins[:-1] + 5,
            delta_rms,
            linestyle="dashed",
            marker="o",
            color="black",
            zorder=50,
            alpha=0.8,
            label="binned values",
        )
        ax.grid(linewidth=0.5, alpha=0.5, zorder=0)

        # titles, legends, axis labels
        ax.set_xlabel("$\mathrm{SNR_{visit}}$")
        ax.set_ylabel(r"$\Delta${varname}".format(varname=var))
        ax.set_title(r"$\Delta${varname} vs. SNR".format(varname=var))
        ax.legend(loc="best")
        ax.set_xlim(0, 1.05 * np.max(snr_visit))
        ax.set_ylim(0, 1.5 * np.max(delta_rms))
        plt.text(
            0.97,
            0.85,
            "$\mu = ${mean}\n$\sigma = ${sd}\n$n = ${no}".format(
                mean=round(np.mean(vardata), 4),
                sd=round(np.std(vardata), 4),
                no=len(vardata),
            ),
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax.transAxes,
        )

        # no overlap save/show cases
        tdy = date.today()  # get date
        filename = f"{figpath}coadd_{tdy}_{var}_cmp"
        if plot_scatter:
            filename = filename + "_sct"
        filename = filename + ".pdf"
        plt.savefig(filename, bbox_inches="tight")
        if not overlap:
            if show:
                plt.show(block="False")
            plt.close("all")  # memory saving

    # overlap save/show cases
    if overlap:
        tdy = date.today()  # get date
        filename = f"{figpath}coadd_{tdy}_cmp"
        if plot_scatter:
            filename = filename + "_sct"
        filename = filename + ".pdf"
        plt.savefig(filename, bbox_inches="tight")
        if show:
            plt.show(block="False")

    return


def plot_added_uncertainty_sigma(
    data: pd.DataFrame,
    vars: Optional[List] = None,
    figpath: str = "/home/riley/uni/rproj/figs/",
    show: bool = False,
):
    """
    Plot the relationship between added systematic uncertainty, and resultant
    standard deviation of parameter z-score distribution.

    :param data: main DataFrame of z-scores
    :param vars:
        optional list of variables to compute for. Defaults to all variables in the DataFrame.
    :param figpath:
        Directory to save file to. Defaults to Riley's workspace
    :param show:
        Whether to show the figure interactively. Defaults to False.
    """
    plt.close("all")
    # Vars cases
    # must be done in two stages since .remove method modifies in place
    # TODO: this is really poorly handled, but I've yet to fix it properly. Need to make
    # it so this isn't just a really poor patch based on the gridsearch data table.
    if vars is None:
        vars = list(data.columns.values)
        param_vars = deque()
        for var in vars:
            if ("+" in var or ("_GS" in var) or ("_t" in var) or ("_E" in var)
                    or ("_SNR" in var) or ("_TEFF" in var)):
                param_vars.append(var)
        for var in param_vars:
            vars.remove(var)

    for var in tqdm(vars):
        plt.close("all")
        # variable setup
        adjustments = Lookup.base_adjustments
        if var == "TEFF":
            adjustments = list(np.array(adjustments) * 1e3)
        varlist = np.concatenate(
            ([var],
             [var + Lookup.convert_extension(adj) for adj in adjustments]))
        x = [0] + adjustments
        y = list()
        for i in varlist:
            y.append(np.std(data[i]))

        # plotting
        fig, ax = plt.subplots(figsize=(7 / 2.54, 7 / 2.54))
        ax: plt.Axes
        ax.plot(
            x,
            y,
            linestyle="dashed",
            marker="o",
            color="black",
            zorder=50,
            alpha=0.8,
            label=f"{var}",
        )
        ax.axhline(y=1,
                   zorder=25,
                   linestyle="dashed",
                   color="grey",
                   label=r"$\sigma = 1$")
        ax.grid(linewidth=0.5, alpha=0.5, zorder=0)
        ax.legend(loc="best")
        ax.set_ylim((0, 1.2 * np.max(y)))

        # labels etc
        ax.set_xlabel(r"$\sigma_s$ [unitless]")
        ax.set_ylabel(r"$\sigma_z$ [unitless]")

        # figsave
        tdy = date.today()
        plt.savefig(
            f"{figpath}z_add_sigma_{var}_{tdy}.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        if show:
            plt.show(block=False)


def plot_compare_coadd_snr(
    data: DataFrame,
    type: Callable = np.median,
    figpath: str = "/home/riley/uni/rproj/figs/",
    compare_telescope: bool = False,
    show: bool = False,
):
    """
    Plot a 2x4 or 1x4 binned RMS or median plot of the difference between
    coadd and individual visit stellar parameters against SNR, grouped by
    elements, and compared for each telescope.

    :param data:
        DataFrame of all data, including SNR, and for each telescope
    :param type:
        Function to calculate the binned_statistic with. Defaults to np.median.
    :param figpath:
        Path to save file to. Defaults to Riley's workspace.
    :param compare_telescope:
        Whether to plot an additional row of telescope comparison.
    :param show:
        Whether to show the figure. Defaults to False.
    """
    plt.close("all")
    # Element groupings by nucleosynthetic process
    groups = [
        ["O_H", "MG_H", "SI_H", "S_H", "CA_H", "TI_H", "P_H"],
        ["C_H", "N_H", "NA_H", "AL_H", "K_H"],
        ["FE_H", "V_H", "MN_H", "NI_H", "CR_H", "CO_H"],
        ["CE_H", "ND_H"],
    ]
    groupnames = [
        r"$\alpha$-capture elements",
        "Light elements",
        "Fe-peak elements",
        "Heavy elements",
    ]

    # add tags to filename
    tdy = date.today()  # get date

    # Generate figure, subfigure, and axes list
    # (1x4 if no comparison, 2x4 if comparing)
    if compare_telescope:
        # figsize = (3.25 * 6.75 / 2.54, 2.1 * 6.75 / 2.54)
        figsize = (5 * 7 / 2.54, 3 * 7 / 2.54)
    else:
        # figsize = (3.25 * 6.75 / 2.54, 1 * 6.75 / 2.54)
        figsize = (5 * 7 / 2.54, 1.5 * 7 / 2.54)
    fig = plt.figure(figsize=figsize, layout="constrained")
    if compare_telescope:
        subfigs = fig.subfigures(nrows=2, ncols=1)
        telescope = ("APO", "LCO")
    else:
        subfigs = [fig]
        telescope = ("", "")

    # create subfigures, and then append each axis to the table
    axes = list()
    for row, subfig in enumerate(subfigs):
        row_axes = subfig.subplots(nrows=1, ncols=4, sharey=True)
        axes.append(row_axes)
    axes = np.concatenate(axes)

    # create secondary data array based on telescope groupings
    if compare_telescope:
        data2 = data[data["TELESCOPE"] == b"lco25m"]
        data = data[data["TELESCOPE"] == b"apo25m"]

    i = 0
    for group in groups:
        for element in group:
            # Base case, plot for axes 0->3
            # drop NaN subtable based on element column
            subdata = data.dropna(subset=[element])
            if len(subdata) == 0:
                print(f"ALERT: {element} all NaN for APO")
                snr_visit = np.arange(0, 200, 10)
                vardata = [0] * 20
            else:
                # obtain SNR
                snr_visit = subdata["SNR"].values
                vardata = subdata[element].values

                # Get number of bins for a binwidth of 10
                binwidth = 20
                bins = np.arange(np.min(snr_visit),
                                 max(snr_visit) + binwidth, binwidth)

                # Get statistic values
                delta_rms, rms_bins, indices = binned_statistic(snr_visit,
                                                                vardata,
                                                                bins=bins,
                                                                statistic=type)

                # Plot
                axes[i].plot(
                    rms_bins[:-1] + 5,
                    delta_rms,
                    linestyle="solid",
                    marker="",
                    zorder=50 - i,
                    alpha=0.8,
                    label=element[:-2].capitalize() + f"($n={len(vardata)}$)",
                )

            # compare telescope case, plot for axes 4->7
            if compare_telescope:
                # drop NaN subtable based on element column
                subdata = data2.dropna(subset=[element])
                # exit case in the event of all NaN
                if len(subdata) == 0:
                    print(f"ALERT: {element} all NaN for LCO")
                    snr_visit = np.arange(0, 200, 10)
                    vardata = [0] * 2
                else:
                    # obtain SNR
                    snr_visit = subdata["SNR"].values
                    vardata = subdata[element].values

                    # Get number of bins for a binwidth of 10
                    binwidth = 10
                    bins = np.arange(np.min(snr_visit),
                                     max(snr_visit) + binwidth, binwidth)

                    # Get statistic values
                    delta_rms, rms_bins, indices = binned_statistic(
                        snr_visit, vardata, bins=bins, statistic=type)
                    axes[i + 4].plot(
                        rms_bins[:-1] + 5,
                        delta_rms,
                        linestyle="solid",
                        marker="",
                        zorder=50 - i,
                        alpha=0.8,
                        label=element[:-2].capitalize() +
                        f"($n={len(vardata)}$)",
                    )

        # after completing group, add grid, etc
        # base case
        # axes[i].text(0.505,
        #             0.97,
        #             "$n = $ {no}".format(no=len(data)),
        #             horizontalalignment='left',
        #             verticalalignment='top',
        #             transform=axes[i].transAxes)
        axes[i].grid(linewidth=0.5, color="grey", alpha=0.5, zorder=0)
        axes[i].axhline(y=0.1,
                        color="black",
                        linestyle="dashed",
                        alpha=0.5,
                        zorder=0)  # precision target
        axes[i].axvline(x=40,
                        color="black",
                        linestyle="dashed",
                        alpha=0.5,
                        zorder=0)  # SNR target
        axes[i].legend(loc="upper right")
        axes[i].set_title(groupnames[i])
        axes[i].set_xlim(0, 200)
        axes[i].set_ylim(0, 0.5)
        # compare telescope case
        if compare_telescope:
            # axes[i + 4].text(0.505,
            #                 0.97,
            #                 "$n = $ {no}".format(no=len(data2)),
            #                 horizontalalignment='left',
            #                 verticalalignment='top',
            #                 transform=axes[i + 4].transAxes)
            axes[i + 4].grid(linewidth=0.5, color="grey", alpha=0.5, zorder=0)
            axes[i + 4].axhline(y=0.1,
                                color="black",
                                linestyle="dashed",
                                alpha=1,
                                zorder=0)  # precision target
            axes[i + 4].axvline(x=40,
                                color="black",
                                linestyle="dashed",
                                alpha=1,
                                zorder=0)  # SNR target
            axes[i + 4].legend(loc="upper right")
            axes[i + 4].set_title(groupnames[i])
            axes[i + 4].set_xlim(0, 200)
            axes[i + 4].set_ylim(0, 0.5)

        i += 1

    # super title, super axes labels, subheadings
    for row, subfig in enumerate(subfigs):
        subfig.suptitle(f"{telescope[row]}")
        subfig.supylabel(
            r"$\left| [X/H]_{\mathrm{coadd}} - [X/H]_{\mathrm{visit}} \right|$"
        )
    fig.supxlabel(r"SNR of visit spectrum [$\mathrm{pixel}^{-1}$]")
    # fig.suptitle("Coadds against SNR grouped by nucleosynthetic process")

    # Save file if desired
    filename = f"{figpath}/coadd_{tdy}_cmp"
    if compare_telescope:
        filename = filename + "_tls"
    filename = filename + ".pdf"
    plt.savefig(filename, bbox_inches="tight")
    if show:
        plt.show(block=False)
    return


def plot_binned_statistic(
        x,
        y,
        varname=None,
        fn=None,
        # function='count',
        xlabel=None,
        ylabel=None,
        zlabel=None,
        figpath="/home/riley/uni/rproj/figs/",
        figsize=(8, 8),
        show: bool = False,
        **kwargs,
):
    """
    Generates a plot based on a binned statistic function.

    Adapted from code written by Andy Casey for trex.

    :param x: x axis data
    :param y: y axis data
    :param varname: variable name for filename
    :param fn: filename
    :param label: x y and z labels
    :param figpath: figure path to save to. Defualts to Riley workspace.
    :param figsize: figsize
    """
    plt.close("all")
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # calculate the binned statistic
    # finite = np.isfinite(x * y * z)
    # H, xedges, yedges, binnumber = stats.binned_statistic_2d(
    #    x[finite], y[finite], z[finite], statistic=function, bins=(10, 10))
    # TODO: change this goofy thing to use the binned statistic 2d routine instead
    # i keep getting errors with finiteness

    # mask
    x = np.ma.masked_where(x <= 100, x)
    y = np.ma.masked_where(y <= 0.5, y)

    H, x, y = np.histogram2d(x, y, bins=(50, 50))

    image = ax.imshow(
        H.T,
        norm="log",
        cmap="inferno",
        aspect=np.ptp(x) / np.ptp(y),
        extent=(x[0], x[-1], y[-1], y[0]),
    )
    plt.colorbar(image, ax=ax, label=zlabel)

    # axes labels
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.set_ylim(ax.get_ylim()[::-1])

    tdy = date.today()
    fig.savefig(f"{figpath}binned_hist_{tdy}_{fn}_{varname}.pdf", dpi=300)
    if show:
        plt.show(block=False)

    return fig


def plot_uncertainties_hist(
    data: pd.DataFrame,
    vars: Optional[List] = None,
    figpath: str = "/home/riley/uni/rproj/figs/",
    show: bool = False,
):
    """
    Plot a grid of the uncertainties for variables.

    :param data: DataFrame of pipeline outputs
    :param vars:
        optional list of variables to compute for. Defaults to all variables within the DataFrame.
    :param figpath:
        Directory to save file to. Defaults to Riley's workspace
    :param show:
        Whether to show the figure interactively. Defaults to False.
    """
    plt.close("all")
    # vars case none handling
    if vars is None:
        vars = np.intersect1d(data.columns.values, np.array(Lookup.uvars))

    fig = plt.figure(layout="constrained", figsize=(27 / 2.54, 27 / 2.54))
    subfigs = fig.subfigures(nrows=5, ncols=1)
    axes = []
    for row, subfig in enumerate(subfigs):
        row_axes = subfig.subplots(nrows=1, ncols=5, sharey=True)
        axes.append(row_axes)
    axes = np.concatenate(axes)
    cmap = get_cmap(len(vars))
    for i in range(len(vars)):
        axes[i].hist(data[vars[i]].values,
                     log=True,
                     edgecolor="white",
                     color=cmap(i),
                     alpha=0.8)
        axes[i].set_title(vars[i])
        axes[i].set_ylim(1, 10**5)
    fig.suptitle(
        "Uncertainty logarithmic histograms for ASPCAPVisits-0.4.0\n(post-invalid exclusion)"
    )
    fig.supylabel("Frequency (log-scale)")
    fig.supxlabel("Uncertainty value")

    # save figure routine
    tdy = date.today()
    plt.savefig(f"{figpath}uncertainty_hist_{tdy}_postexc2.pdf",
                bbox_inches="tight")
    # show if requested
    if show:
        plt.show(block=False)

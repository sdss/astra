import numpy as np
from datetime import date
from numpy.typing import ArrayLike
import scienceplots  # noqa
import matplotlib.pyplot as plt
import matplotlib
import mpl_preamble  # noqa
import scipy.stats as stats
from scipy.special import rel_entr
from datetime import date

plt.style.use("science")
params = {
    "text.usetex": True,
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "font.serif": "",
    "legend.frameon": True,
}
plt.rcParams.update(params)


def model(params, snr, teff):
    return params[2] * teff + params[1] / snr + params[0]


def find_params(
    paramdict: dict,
    var: str,
    nparams: int = 20,
    method: str = "std",
    debug_plot: bool = False,
):
    """
    Perform a basic gridsearch to find the best z-score calculation.

    :param dict:
        dictionary of all parameter values
    :param var:
        variable type
    :param nparams:
        number of parameters
    :param method: type of method for gridsearch scoring. Defaults to
        trying to find for minimum distance for std of 1.
    :param debug_plot:
        whether to plot debug plots. Defaults to False.

    :returns: best values of parameters
    """

    tdy = date.today()

    # access data
    A = np.array(paramdict["A"])
    B = np.array(paramdict["B"])
    uAB = np.array(paramdict["uAB"])
    SNR_A = np.array(paramdict["SNR_A"])
    TEFF_A = np.array(paramdict["TEFF_A"])
    SNR_B = np.array(paramdict["SNR_B"])
    TEFF_B = np.array(paramdict["TEFF_B"])

    # parameter grid
    theta_2 = np.linspace(0.1, 1, nparams)
    theta_1 = np.linspace(0.1, 50, nparams)
    theta_0 = np.linspace(0.01, 0.5, nparams)**2

    # If TEFF, scale it all by a factor of 10^2
    if var == "TEFF":
        theta_0 = theta_0 * 1e3
        theta_1 = theta_1 * 1e3
        theta_2 = theta_2 * 1e3

    t0, t1, t2 = np.meshgrid(theta_0, theta_1, theta_2)

    # dummy valuea
    scores = np.zeros((nparams, nparams, nparams))

    # random sample a normal
    snorm = stats.norm()
    samps = snorm.rvs(size=len(A))
    samps /= np.sum(samps)

    # loop for all
    for i in range(nparams):
        for j in range(nparams):
            for k in range(nparams):
                params = (t0[i, j, k], t1[i, j, k], t2[i, j, k])
                additive = (model(params, SNR_A, TEFF_A)**2 +
                            model(params, SNR_B, TEFF_B)**2)
                if method == "std":
                    scores[i, j, k] = np.std((A - B) / np.sqrt(uAB + additive))
                elif method == "kl_div":
                    # get prob dists
                    arr = (A - B) / np.sqrt(uAB + additive)
                    arr /= np.sum(arr)

                    # take relative entropy
                    vec = rel_entr(arr, samps)

                    # mask NaNs
                    vec = np.ma.masked_invalid(vec).compressed()

                    # compute KLdiv
                    scores[i, j, k] = np.sum(vec)
    # find closest to std = 1
    if method == "std":
        loc = np.where(scores == scores.flat[np.abs(scores - 1).argmin()])
    elif method == "kl_div":
        # scores = np.ma.masked_less_equal(scores, 0)
        loc = np.where(scores == scores.flat[scores.argmin()])

    # Create debug plots if requested
    if debug_plot:
        plt.close("all")
        fig, (ax1, ax2) = plt.subplots(nrows=1,
                                       ncols=2,
                                       layout="constrained",
                                       figsize=(9 / 2.54, 7 / 2.54))
        ax1.scatter(
            SNR_A,
            t1[loc] / SNR_A + t1[loc] / SNR_B,
            s=2,
            alpha=0.2,
            color="cornflowerblue",
        )
        ax1.set_xlabel("S/N")
        ax1.set_ylabel(r"$\frac{\theta_1}{S/N}$")
        ax1.set_title(r"$\theta_1$ = {var}".format(var=round(t1[loc][0], 2)))

        ax2.scatter(
            TEFF_A,
            t2[loc] * TEFF_A + t2[loc] * TEFF_B,
            s=2,
            alpha=0.2,
            color="lightcoral",
        )
        ax2.set_title(r"$\theta_2$ = {var}".format(var=round(t2[loc][0], 2)))
        ax2.set_xlabel(r"$T_{\mathrm{eff}}$")
        ax2.set_ylabel(r"$\theta_2 \cdot T_{\mathrm{eff}}$")
        fig.savefig(f"./figs/gridsearch_val_{var}_{tdy}.pdf")

        fig, ax = plt.subplots()
        scat = ax.scatter(np.min(t1, axis=1),
                          np.min(t2, axis=1),
                          c=np.min(scores, axis=1))
        ax.plot(t1[loc], t2[loc], color="green", marker="o", zorder=50)
        ax.set_xlabel(r"$\theta_1$ ($S / N$ parameter)")
        ax.set_ylabel(r"$\theta_2$ ($T_{\mathrm{eff}}$ parameter)")
        plt.colorbar(scat)

        fig.savefig(f"./figs/gridsearch_cmp_{var}_{tdy}.pdf")

    # return best values
    return t0[loc], t1[loc], t2[loc]

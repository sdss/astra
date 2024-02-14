import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
import scipy.stats as stats
import matplotlib.pyplot as plt


class Lookup:
    """
    Class of lookup data, including keys, valid values, and converting extensions.
    """

    infos = [
        "GAIA_DR3_SOURCE_ID", "SDSS4_DR17_APOGEE_ID", "SPECTRUM_ID", "SNR"
    ]
    stellar_params = ["TEFF", "LOGG", "FE_H"]
    other_params = ["V_MICRO", "V_SINI", "M_H_ATM"]
    abundances = [
        "AL_H",
        "CA_H",
        "CE_H",
        "CR_H",
        "CO_H",
        "C_H",
        "K_H",
        "MG_H",
        "MN_H",
        "NA_H",
        "ND_H",
        "NI_H",
        "N_H",
        "O_H",
        "P_H",
        "SI_H",
        "S_H",
        "TI_H",
        "V_H",
    ]
    flags = [abd + "_FLAGS" for abd in abundances]
    u_stellar_params = ["E_" + param for param in stellar_params]
    u_other_params = ["E_" + param for param in other_params]
    u_abundances = ["E_" + param for param in abundances]
    vars = stellar_params + other_params + abundances
    uvars = u_stellar_params + u_other_params + u_abundances
    all_columns = infos + vars + uvars + flags

    base_adjustments = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
    vardict = {
        "TEFF": (3000, 9000),
        "LOGG": (0.3, 4.5),
        "V_MICRO": (0.5, 4.6),
        "V_SINI": (0.05, 90),
        "M_H_ATM": (-2.49, 0.04),
    }
    vardict.update(dict.fromkeys(abundances + ["FE_H"], (-4.0, 4.0)))
    uvardict = dict.fromkeys(u_abundances + ["E_LOGG", "E_M_H_ATM", "E_FE_H"],
                             [0.01, 0.9])
    uvardict.update({
        "E_TEFF": [10, 200],
        "E_V_MICRO": [0.01, 0.9],
        "E_V_SINI": [0.01, 10]
    })

    def convert_extension(ext: str | float):
        """
        Convert the extension name (TEFF+XXX) to a float, or the
        float to the extension tail.

        :param ext: extension as string or float
        :returns: converted extension into opposite format
        """
        if isinstance(ext, str):
            ext = ext.split("+")[-1]
            ext = float(ext)
        else:
            ext = "+" + str(ext)

        return ext


def get_cmap(n, name="hsv"):
    """
    Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color.

    Code sourced from: https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib

    :param n:
        Number of maps.
    :param name:
        An standard mpl colormap name string.
    """
    return plt.cm.get_cmap(name, n)


def rms(x: ArrayLike):
    """
    Obtain RMS value for a given 1D array

    :param x:
        1-D array like of variables to take full RMS for.
    """
    return np.sqrt(np.mean(np.square(x)))


def drop_duplicates(data: pd.DataFrame,
                    key: str = "SPECTRUM_ID") -> pd.DataFrame:
    """
    Drop all duplicates by a column, keeping the ones with the most information.

    :param data: DataFrame to process
    :param key: String of key to column
    :returns: processed DataFrame
    """
    # code is sourced from
    # https://stackoverflow.com/questions/43769693/drop-duplicates-in-a-dataframe-keeping-the-row-with-the-least-nulls
    # data = data.loc[data.notnull().sum(1).groupby(data[key]).idxmax()]
    data = data.sort_values(by=list(data.columns),
                            na_position="last").drop_duplicates(key,
                                                                keep="first")

    return data


def mask_unique_id(
    data: pd.DataFrame,
    main_id: str = "GAIA_DR3_SOURCE_ID",
    backup_id: str = "SDSS4_DR17_APOGEE_ID",
) -> pd.DataFrame:
    """
    Mask unique ID's, replacing any invalid main IDs with a respective backup ID.

    :param data: DataFrame to process
    :param main_id: Main identifier label/key
    :param backup_id: Replacement identifier label/key
    :returns: processed DataFrame
    """
    return data.mask(data[main_id] == -1, other=data[backup_id], axis=1)


def pdf_fit(x: ArrayLike, type: stats.rv_continuous, abs: bool = False):
    """
    Generate PDF for a given variable type.

    :param x: 1D array of data to make PDF for
    :param abs: Whether to generate for a normalized scale. Defaults to False.
    :return x_pdf: Array of corresponding values for PDF of x.
    """
    if abs:
        mu, sigma = type.fit(x)
        x = np.linspace(
            type.ppf(0.0001, loc=mu, scale=sigma),
            type.ppf(0.9999, loc=mu, scale=sigma),
            100,
        )
        x_pdf = type.pdf(x, loc=mu, scale=sigma)
    else:
        x_pdf = type.pdf(x)
    return x_pdf


def pdf_gen(mu, sigma, type: stats.rv_continuous):
    """
    Generate xy plotting coordinates for a given type of variable.

    :param mu: mean
    :param sigma: standard deviation
    :param type: type of distribution to create as a SciPy Continuous RV subclass.
    """
    points = np.linspace(
        type.ppf(0.0001, loc=mu, scale=sigma),
        type.ppf(0.9999, loc=mu, scale=sigma),
        100,
    )
    pdf = type.pdf(points, loc=mu, scale=sigma)
    return points, pdf

from typing import OrderedDict
import numpy as np
from astra.tools.bitmask import BitMask
from astra.tools.spectrum import Spectrum1D


class LabelBitmask(BitMask):

    # TODO: update the old apogee_drp/targeting bitmask class defintions to make it nicer to initiate these

    DEFINITIONS = [
        (1, "TEFF_UNRELIABLE", "log(teff) is outside the range (3.1, 4.7)"),
        (1, "LOGG_UNRELIABLE", "log(g) is outside the range (-1.5, 6)"),
        (
            1,
            "FE_H_UNRELIABLE",
            "Metallicity is outside the range (-2, 0.5), or the log(teff) exceeds 3.82",
        ),
        (
            1,
            "TEFF_ERROR_UNRELIABLE",
            "The median of log(teff) draws is outside the range (3.1, 4.7)",
        ),
        (
            1,
            "LOGG_ERROR_UNRELIABLE",
            "The median of log(g) draws is outside the range (-1.5, 6)",
        ),
        (
            1,
            "FE_H_ERROR_UNRELIABLE",
            "The median of metallicity draws is outside the range (-2, 0.5) or the median of log(teff) draws exceeds 3.82",
        ),
        (1, "TEFF_ERROR_LARGE", "The error on log(teff) is larger than 0.03"),
        (1, "LOGG_ERROR_LARGE", "The error on log(g) is larger than 0.3"),
        (1, "FE_H_ERROR_LARGE", "The error on metallicity is larger than 0.5"),
        (1, "MISSING_PHOTOMETRY", "There is some Gaia/2MASS photometry missing."),
        (
            1,
            "PARAMS_UNRELIABLE",
            "Do not trust these results as there are known issues with the reported stellar parameters in this region.",
        ),
    ]

    name = []
    level = []
    description = []
    for _level, _name, _description in DEFINITIONS:
        name.append(_name)
        level.append(_level)
        description.append(_description)


def create_bitmask(
    label_predictions, meta, median_draw_predictions=None, std_draw_predictions=None
):
    """
    Return a bitmask array given the label predictions.

    :param label_predictions:
        A (N, 3) shape array where N is the number of spectra. The three columns are expected to be
        `log_g`, `teff`, and `fe_h`.

    :param median_draw_predictions: [optional]
        A (N, 3) shape array containing the median of the uncertainty draws. The columns should
        be the same as `label_predictions`.

    :param std_draw_predictions: [optional]
        A (N, 3) shape array containing the standard deviation of the uncertainty draws. The columns
        show be the same as `label_predictions`.
    """

    N, L = label_predictions.shape

    flag_map = LabelBitmask()
    bitmask = np.zeros(N, dtype=int)

    log_g, teff, fe_h = label_predictions.T
    log_teff = np.log10(teff)

    bitmask[(fe_h > 0.5) | (fe_h < -2) | (log_teff > 3.82)] |= flag_map.get_value(
        "FE_H_UNRELIABLE"
    )
    bitmask[(log_g < -1.5) | (log_g > 6)] |= flag_map.get_value("LOGG_UNRELIABLE")
    bitmask[(log_teff < 3.1) | (log_teff > 4.7)] |= flag_map.get_value(
        "TEFF_UNRELIABLE"
    )

    if median_draw_predictions is not None:
        med_log_g, med_teff, med_fe_h = median_draw_predictions.T
        med_log_teff = np.log10(med_teff)

        bitmask[
            (med_fe_h > 0.5) | (med_fe_h < -2) | (med_log_teff > 3.82)
        ] |= flag_map.get_value("FE_H_ERROR_UNRELIABLE")
        bitmask[(med_log_g < -1.5) | (med_log_g > 6)] |= flag_map.get_value(
            "LOGG_ERROR_UNRELIABLE"
        )
        bitmask[(med_log_teff < 3.1) | (med_log_teff > 4.7)] |= flag_map.get_value(
            "TEFF_ERROR_UNRELIABLE"
        )

    if std_draw_predictions is not None:
        std_log_g, std_teff, std_fe_h = std_draw_predictions.T
        std_log_teff = np.log10(std_teff)
        bitmask[std_log_g > 0.3] |= flag_map.get_value("LOGG_ERROR_LARGE")
        bitmask[std_log_teff > 2.7] |= flag_map.get_value("TEFF_ERROR_LARGE")
        bitmask[std_fe_h > 0.5] |= flag_map.get_value("FE_H_ERROR_LARGE")

    if not np.all(
        np.isfinite(np.array([meta[k] for k in ("RP_MAG", "K_MAG", "H_MAG", "PLX")]))
    ):
        bitmask |= flag_map.get_value("MISSING_PHOTOMETRY")

    is_bad = ((meta["RP_MAG"] - meta["K_MAG"]) > 2.3) & (
        (meta["H_MAG"] - 5 * np.log10(1000 / meta["PLX"]) + 5) > 6
    )
    if is_bad:
        bitmask |= flag_map.get_value("PARAMS_UNRELIABLE")
    return bitmask


def get_metadata(spectrum: Spectrum1D):
    """
    Get requisite photometry and astrometry from a given spectrum for APOGEENet.

    :param spectrum:
        An `astra.tools.spectrum.Spectrum1D` spectrum.

    :returns:
        A three-length tuple containing relevant metadata for the given spectrum. The first entry in
        the tuple contains the header keys, the second entry contains the values of the metadata,
        and the last value contains the normalized, clipped values of the metadata, for use with the
        APOGEENet model.
    """

    keys = {
        "PLX": [],
        "G_MAG": ["GMAG"],
        "BP_MAG": ["BPMAG"],
        "RP_MAG": ["RPMAG"],
        "J_MAG": ["JMAG"],
        "H_MAG": ["HMAG"],
        "K_MAG": ["KMAG"],
    }

    de_nanify = lambda x: x if (x != "NaN" and x != -999999) else np.nan
    meta = OrderedDict()
    for preferred_key, alternate_keys in keys.items():
        for key in [preferred_key] + alternate_keys:
            try:
                value = spectrum.meta[key]
            except KeyError:
                try:
                    value = spectrum.meta[key.lower()]
                except KeyError:
                    continue
                else:
                    meta[preferred_key] = de_nanify(value)
                    break
            else:
                meta[preferred_key] = de_nanify(value)
                break

    metadata = np.array([de_nanify(value) for value in meta.values()])
    mdata_replacements = np.array(
        [-84.82700, 21.40844, 24.53892, 20.26276, 18.43900, 24.00000, 17.02500]
    )
    mdata_stddevs = np.array(
        [
            14.572430555504504,
            2.2762944923233883,
            2.8342029214199704,
            2.136884367623457,
            1.6793628207779732,
            1.4888102872755238,
            1.5848713221149886,
        ]
    )
    mdata_means = np.array(
        [
            -0.6959113178296891,
            13.630030428758845,
            14.5224418320574,
            12.832448427460813,
            11.537019017423619,
            10.858717523536697,
            10.702106344460235,
        ]
    )

    metadata = np.where(metadata < 98, metadata, mdata_replacements)
    metadata = np.where(np.isfinite(metadata), metadata, mdata_replacements)
    metadata = np.where(metadata > -1, metadata, mdata_replacements)
    metadata_norm = ((metadata - mdata_means) / mdata_stddevs).astype(np.float32)

    return (meta, metadata_norm)

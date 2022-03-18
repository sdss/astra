import numpy as np
from collections import OrderedDict

from astra.utils.bitmask import BitFlagNameMap


class LabelBitMap(BitFlagNameMap):

    """
    Bitmask class for APOGEENet labels.
    """
    
    TEFF_UNRELIABLE = 0, "log(teff) is outside the range (3.1, 4.7)"
    LOGG_UNRELIABLE = 1, "log(g) is outside the range (-1.5, 6)"
    FE_H_UNRELIABLE = 2, "Metallicity is outside the range (-2, 0.5), or the log(teff) exceeds 3.82"
    TEFF_ERROR_UNRELIABLE = 3, "The median of log(teff) draws is outside the range (3.1, 4.7)"
    LOGG_ERROR_UNRELIABLE = 4, "The median of log(g) draws is outside the range (-1.5, 6)"
    FE_H_ERROR_UNRELIABLE = 5, "The median of metallicity draws is outside the range (-2, 0.5) or the median of log(teff) draws exceeds 3.82"
    TEFF_ERROR_LARGE = 6, "The error on log(teff) is larger than 0.03"
    LOGG_ERROR_LARGE = 7, "The error on log(g) is larger than 0.3"
    FE_H_ERROR_LARGE = 8, "The error on metallicity is larger than 0.5"

    levels = OrderedDict([
        [1, 
            (
                "TEFF_UNRELIABLE", "LOGG_UNRELIABLE", "FE_H_UNRELIABLE", 
                "TEFF_ERROR_UNRELIABLE", "LOGG_ERROR_UNRELIABLE", "FE_H_ERROR_UNRELIABLE",
                "TEFF_ERROR_LARGE", "LOGG_ERROR_LARGE", "FE_H_ERROR_LARGE"
            )
        ],
    ])


def create_bitmask(
        label_predictions, 
        median_draw_predictions=None, 
        std_draw_predictions=None
    ):
    """
    Return a bitmask array given the label predictions.

    :param label_predictions:
        A (N, 3) shape array where N is the number of spectra. The three columns are expected to be
        `log_g`, `log_teff`, and `fe_h`.
    
    :param median_draw_predictions: [optional]
        A (N, 3) shape array containing the median of the uncertainty draws. The columns should 
        be the same as `label_predictions`.
    
    :param std_draw_predictions: [optional]
        A (N, 3) shape array containing the standard deviation of the uncertainty draws. The columns
        show be the same as `label_predictions`.
    """

    N, L = label_predictions.shape

    flag_map = LabelBitMap()
    bitmask = np.zeros(N, dtype=int)

    log_g, log_teff, fe_h = label_predictions.T

    bitmask[(fe_h > 0.5) | (fe_h < -2) | (log_teff > 3.82)] |= flag_map.get_value("FE_H_UNRELIABLE")
    bitmask[(log_g < -1.5) | (log_g > 6)] |= flag_map.get_value("LOGG_UNRELIABLE")
    bitmask[(log_teff < 3.1) | (log_teff > 4.7)] |= flag_map.get_value("TEFF_UNRELIABLE")

    if median_draw_predictions is not None:
        med_log_g, med_log_teff, med_fe_h = median_draw_predictions.T

        bitmask[(med_fe_h > 0.5) | (med_fe_h < -2) | (med_log_teff > 3.82)] |= flag_map.get_value("FE_H_ERROR_UNRELIABLE")
        bitmask[(med_log_g < -1.5) | (med_log_g > 6)] |= flag_map.get_value("LOGG_ERROR_UNRELIABLE")
        bitmask[(med_log_teff < 3.1) | (med_log_teff > 4.7)] |= flag_map.get_value("TEFF_ERROR_UNRELIABLE")

    if std_draw_predictions is not None:
        std_log_g, std_log_teff, std_fe_h = std_draw_predictions.T
        bitmask[std_log_g > 0.3] |= flag_map.get_value("LOGG_ERROR_LARGE")
        bitmask[std_log_teff > 0.03] |= flag_map.get_value("TEFF_ERROR_LARGE")
        bitmask[std_fe_h > 0.5] |= flag_map.get_value("FE_H_ERROR_LARGE")

    return bitmask
    

def get_metadata(spectrum=None, headers=None):
    """
    
    :param spectrum:
        An `astra.tools.spectrum.Spectrum1D` spectrum.
    
    :returns:
        A three-length tuple containing relevant metadata for the given spectrum. The first entry in
        the tuple contains the header keys, the second entry contains the values of the metadata,
        and the last value contains the normalized, clipped values of the metadata, for use with the
        APOGEENet model.
    """
    keys = ("PLX", "GMAG", "BPMAG", "RPMAG", "JMAG", "HMAG", "KMAG")
    if spectrum is not None:
        headers = spectrum.meta["header"]

    metadata = []
    for key in keys:
        try:
            metadata.append(headers[key])
        except KeyError:
            metadata.append(np.nan)
    
    metadata = np.array([(value if value != "NaN" else np.nan) for value in metadata])
    mdata_replacements = np.array([-84.82700,21.40844,24.53892,20.26276,18.43900,24.00000,17.02500])
    mdata_stddevs = np.array([14.572430555504504,2.2762944923233883,2.8342029214199704,2.136884367623457,
                                1.6793628207779732,1.4888102872755238,1.5848713221149886])
    mdata_means = np.array([-0.6959113178296891,13.630030428758845,14.5224418320574,12.832448427460813,
                                11.537019017423619,10.858717523536697,10.702106344460235])

    metadata = np.where(metadata < 98, metadata, mdata_replacements)
    metadata = np.where(np.isfinite(metadata), metadata, mdata_replacements)
    metadata = np.where(metadata > -1, metadata, mdata_replacements)
    metadata_norm = ((metadata - mdata_means) / mdata_stddevs).astype(np.float32)

    return (keys, metadata, metadata_norm)


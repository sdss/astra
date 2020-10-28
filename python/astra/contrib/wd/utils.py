import numpy as np
import pickle

def line_features(
        spectrum, 
        wavelength_regions=(
            [3860, 3900], # Balmer line
            [3950, 4000], # Balmer line
            [4085, 4120], # Balmer line
            [4320, 4360], # Balmer line
            [4840, 4880], # Balmer line
            [6540, 6580], # Balmer line
            [3880, 3905], # He I/II line
            [3955, 3975], # He I/II line
            [3990, 4056], # He I/II line
            [4110, 4140], # He I/II line
            [4370, 4410], # He I/II line
            [4450, 4485], # He I/II line
            [4705, 4725], # He I/II line
            [4900, 4950], # He I/II line
            [5000, 5030], # He I/II line
            [5860, 5890], # He I/II line
            [6670, 6700], # He I/II line
            [7050, 7090], # He I/II line
            [7265, 7300], # He I/II line
            [4600, 4750], # Molecular C absorption band
            [5000, 5160], # Molecular C absorption band
            [3925, 3940], # Ca H/K line
            [3960, 3975], # Ca H/K line
        ),
        polyfit_regions=(
            [3850, 3870],
            [4220, 4245],
            [5250, 5400],
            [6100, 6470],
            [7100, 9000]
        ),
        polyfit_order=5
    ):
    """
    Engineer features based on line ratios for distinguishing different kinds of white dwarfs.

    :param spectrum:
        A `specutils.Spectrum1D` spectrum of a white dwarf.
    
    :param wavelength_regions: [optional]
        A tuple of two-length lists containing the start and end wavelengths to measure a line ratio from.
    
    :param polyfit_regions: [optional]
        A tuple of two-length lists containing the start and end wavelengths to use when fitting the baseline flux.

    :param polyfit_order: [optional]
        The polynomial order to use to fit to the baseline (continuum) flux.

    :returns:
        An array of line ratios for the given wavelength regions.
    """

    # To make it easier for any future data-slicing we have to do.
    wavelength = spectrum.wavelength.value
    flux = spectrum.flux.value[0] 

    mask = np.zeros(wavelength.size, dtype=bool)
    for start, end in polyfit_regions:
        mask += (end > wavelength) * (wavelength > start)

    # Only consider finite values.
    mask *= np.isfinite(flux)

    # NOTE: "sigma" is referred to in the original line_info but is never used when fitting.
    func_poly = np.polyfit(wavelength[mask], flux[mask], polyfit_order)
    
    p = np.poly1d(func_poly)
    
    # Go through the feature list.
    F = len(wavelength_regions)
    features = np.empty(F)
    for i, (start, end) in enumerate(wavelength_regions):
        line_mask = (end > wavelength) * (wavelength > start)
        mean_f_s = flux[line_mask]
        features[i] = np.mean(flux[line_mask]) / np.mean(p(wavelength[line_mask]))
    
    return features



def classify_white_dwarf(model_path, spectrum, **kwargs):
    """
    Classify a white dwarf given a pre-trained model and a spectrum.

    :param model_path:
        The local path to a pickled Random Forest Classifier.
    
    :param spectrum:
        A `specutils.Spectrum1D` spectrum of a white dwarf.

    :returns:
        The most likely white dwarf class.
    """

    # TODO: This is very dangerous to execute things in this way, but this is what I was given.
    with open(model_path, "rb") as fp:
        random_forest_classifier = pickle.load(fp)

    features = line_features(spectrum, **kwargs)

    return random_forest_classifier.predict(features.reshape((1, -1)))[0]
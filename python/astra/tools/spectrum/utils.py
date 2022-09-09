import numpy as np


def overlap(a, b):
    b_min, b_max = (np.min(b), np.max(b))
    return np.any((b_max >= a) & (a >= b_min))


def spectrum_overlaps(spectrum, spectral_axis):
    if spectrum is None:
        return False
    try:
        _spectral_axis = spectral_axis.value
    except:
        _spectral_axis = spectral_axis
    return overlap(spectrum.wavelength.value, _spectral_axis)

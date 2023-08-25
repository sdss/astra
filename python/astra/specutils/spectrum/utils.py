"""General utilities for handling spectra."""
import numpy as np
from specutils import SpectralAxis, Spectrum1D


def _one_side_overlap(a, b):
    return np.any((np.max(b) >= a) & (a >= np.min(b)))


def overlap(a: np.array, b: np.array) -> bool:
    """
    Returns a boolean whether the two arrays share any overlap in their ranges.

    :param a:
        An array of samples.

    :param b:
        An array of samples.

    :returns:
        True if at least some area between ``min(a)`` and ``max(b)`` exists in the
        range of ``min(b)`` and ``max(b)``.
    """
    return _one_side_overlap(a, b) or _one_side_overlap(b, a)


def spectrum_overlaps(spectrum: Spectrum1D, spectral_axis: SpectralAxis) -> bool:
    """
    Returns a boolean whether a spectrum and another spectral axis overlap.

    :param spectrum:
        A spectrum.

    :param spectral_axis:
        A comparison spectral axis.

    :returns:
        True if the spectra overlap, false otherwise.
    """
    if spectrum is None:
        return False
    try:
        _spectral_axis = spectral_axis.to(spectrum.wavelength.unit).value
    except:
        _spectral_axis = spectral_axis
    return overlap(spectrum.wavelength.value, _spectral_axis) or overlap(
        _spectral_axis, spectrum.wavelength.value
    )


import numpy as np
from astra.utils import log

def slice_and_shape(spectrum, slice_args, shape, repeat=None, **kwargs):

    if repeat is not None:
        spectrum._data = np.repeat(spectrum._data, repeat)
        spectrum._uncertainty.array = np.repeat(spectrum._uncertainty.array, repeat)

    if slice_args is not None:
        slices = tuple([slice(*each) for each in slice_args])

        spectrum._data = spectrum._data[slices]
        spectrum._uncertainty.array = spectrum._uncertainty.array[slices]

        try:
            spectrum.meta["snr"] = spectrum.meta["snr"][slices[0]]
        except:
            log.warning(f"Unable to slice 'snr' metadata with {slice_args}")

    spectrum._data = spectrum._data.reshape(shape)
    spectrum._uncertainty.array = spectrum._uncertainty.array.reshape(shape)

    return spectrum



def median(spectrum, axis=None, slice_args=None, shape=None, **kwargs):
    """
    Rectify the spectrum by dividing through the median of (finite) flux values.
    
    :param spectrum:
        The spectrum to rectify.
    
    :param axis: [optional]
        The axis to take the median across (default: 1).
    
    :param slice_args: [optional]
        Slice the spectrum given these arguments.
    
    :param shape: [optional]
        Re-shape the spectrum.
    """

    if slice_args is not None or shape is not None:
        spectrum = slice_and_shape(spectrum, slice_args, shape, **kwargs)

    continuum = np.nanmedian(spectrum.flux.value, axis=axis).reshape((-1, 1))

    spectrum._data /= continuum
    spectrum._uncertainty.array *= continuum**2 # TODO: check this.

    return spectrum


def mean(spectrum, axis=1, slice_args=None, shape=None, **kwargs):
    """
    Rectify a spectrum by dividing through the mean of (finite) flux values.

    :param spectrum:
        The spectrum to rectify.

    :param axis: [optional]
        The axis to take the median across (default: 1).
    
    :param slice_args: [optional]
        Slice the spectrum given these arguments.
    
    :param shape: [optional]
        Re-shape the spectrum.
    """

    continuum = np.nanmean(spectrum.flux.value, axis=axis).reshape((-1, 1))

    spectrum._data /= continuum
    spectrum._uncertainty.array *= continuum**2
    
    if slice_args is not None or shape is not None:
        spectrum = slice_and_shape(spectrum, slice_args, shape, **kwargs)

    return spectrum

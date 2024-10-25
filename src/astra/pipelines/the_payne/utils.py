import pickle
import numpy as np
from functools import cache
from astra.utils import expand_path

@cache
def read_model(model_path):
    """
    Read the network coefficients from disk.

    The `model_path` should be a `pickle` file that contains a `dict` with
    the following keys:
    - `b`: a tuple of arrays containing the biases in each layer
    - `w`: a tuple of arrays containing the weights in each layer
    - `x_min`: an array containing the minimum values of each (unscaled) label
    - `x_max`: an array containing the maximum values of each (unscaled) label
    - `label_names`: a tuple containing the label names
    - `wavelength`: an array of wavelength values for output spectra
    """
    with open(expand_path(model_path), "rb") as fp:
        contents = pickle.load(fp)
    return contents


def overlap(a, b):
    b_min, b_max = (np.min(b), np.max(b))
    return np.any((b_max >= a) & (a >= b_min))


@cache
def read_mask(mask_path):
    return None if mask_path is None else np.load(expand_path(mask_path))

import pickle
import os
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


@cache
def read_elem_mask_all(dir_mask):
    #### get elem list
    elem_list = []
    for file in os.listdir(expand_path(dir_mask)):
        if file.endswith(".mask"):
            elem_list.append(file.split('.')[0])
    
    #### read all elem mask
    elems_mask = {}
    for elem in elem_list:
        elems_mask[elem] = read_elem_mask(expand_path(dir_mask), elem)

    return elems_mask


@cache
def read_elem_mask(dir_mask, elem):
    #### element mask
    path_mask = os.path.join(dir_mask, f'{elem}.mask')
    weight_mask = np.loadtxt(path_mask)

    #### wavelength mask
    path_wave_mask = os.path.join(dir_mask, 'mask_7514_to_7214.txt')
    wave_mask = np.loadtxt(path_wave_mask).astype(bool)

    return weight_mask[wave_mask]


def calc_weighted_chi_square(observed_values, expected_values, errors, weights):
    """
    Calculate the weighted chi-square statistic for a spectrum.

    Args:
        observed_values (numpy.ndarray): Array of observed values for each pixel.
        expected_values (numpy.ndarray): Array of expected values for each pixel.
        errors (numpy.ndarray): Array of errors for each pixel.
        weights (numpy.ndarray): Array of weights (ranging from 0 to 1) for each pixel.

    Returns:
        float: The weighted chi-square statistic.
    """
    return np.sum( weights * (observed_values - expected_values)**2 / errors**2 )


def calc_weighted_reduced_chi_square(observed_values, expected_values, errors, weights, num_params=1):
    """
    Calculate the weighted reduced chi-square statistic for a spectrum.

    Args:
        observed_values (numpy.ndarray): Array of observed values for each pixel.
        expected_values (numpy.ndarray): Array of expected values for each pixel.
        errors (numpy.ndarray): Array of errors for each pixel.
        weights (numpy.ndarray): Array of weights (ranging from 0 to 1) for each pixel.
        num_params (int): Number of model parameters used in fitting, including any scale parameter.

    Returns:
        float: The weighted unbiased reduced chi-square statistic.
    """
    # Calculate the weighted chi-square statistic
    weighted_chi_square = calc_weighted_chi_square(observed_values, expected_values, errors, weights)
    
    # Calculate the degrees of freedom
    mask = weights > 0.0
    num_data_points = np.sum(mask) # len(observed_values)
    dof = num_data_points - num_params # degrees of freedom
    
    # return the unbiased reduced chi-square
    return weighted_chi_square / dof


def calc_error_of_weighted_mean(errors, weights):
    """
    Calculate the standard error of the weighted mean.

    Args:
        errors (numpy.ndarray): Array of errors for each pixel.
        weights (numpy.ndarray): Array of weights (ranging from 0 to 1) for each pixel.

    Returns:
        float: The standard error of the weighted mean.
    """
    return 1. / np.sqrt( np.sum( weights / errors**2 ) )
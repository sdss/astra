import numpy as np
from scipy import sparse, stats
from typing import Tuple

_fwhm_to_sigma = 1/(2 * np.sqrt(2 * np.log(2)))


def lsf_sigma(λ: float, R: Tuple[int, float]):
    """
    Return the Gaussian width for the line spread function at wavelength λ with spectral resolution R.

    :param λ:
        Wavelength to evaluate the LSF.
    
    :param R:
        Spectral resolution.
    
    :returns:
        The width of a Gaussian to represent the width of the line spread function.
    """
    return (λ / R) * _fwhm_to_sigma


def lsf_sigma_and_bounds(λ: float, R: Tuple[int, float], σ_window: Tuple[float, int] = 5):
    """
    Return the Gaussian width for the line spread function, and the lower and upper bounds where the LSF contributes.

    :param λ:
        Wavelength to evaluate the LSF.
    
    :param R:
        Spectral resolution.

    :param σ_window: [optional]
        The number of sigma where the LSF contributes (default: 5).
    
    :returns:
        A two-length tuple containing the LSF sigma, and the (lower, upper) tuple where the LSF contributes.
    """
    σ = lsf_sigma(λ, R)
    return (σ, (λ - σ_window * σ, λ + σ_window * σ))


def instrument_lsf_kernel(λ: np.array, λc: float, R: Tuple[int, float], **kwargs):
    """
    Calculate the convolution kernel for the given instrument line spread function at the wavelengths specified,
    centered on the given central wavelength.

    :param λ:
        The wavelength array.
        
    :param λc:
        Wavelength to evaluate the LSF.

    :param R:
        Spectral resolution.
    
    :returns:
        A two-length tuple containing the mask where the LSF contributes, and the normalised kernel.
    """
    σ, (lower, upper) = lsf_sigma_and_bounds(λc, R, **kwargs)
    mask = (upper >= λ) & (λ >= lower)
    ϕ = stats.norm.pdf(λ[mask], loc=λc, scale=σ)
    ϕ /= np.sum(ϕ)
    return (mask, ϕ)


def instrument_lsf_dense_matrix(λ_input: np.array, λ_output: np.array, R: Tuple[int, float], **kwargs):
    """
    Construct a dense matrix to convolve fluxes at input wavelengths (λ_input) at an instrument spectral
    resolution (R) and resample to the given output wavelengths (λ_output).

    :param λ_input:
        A N-length array of input wavelength values.

    :param λ_output:
        A M-length array of output wavelength values.

    :param R:
        Spectral resolution.
    
    :returns:
        A (N, M) dense array representing a convolution kernel.
    """
    K = np.empty((λ_input.size, λ_output.size), dtype=float)
    for o, λ in enumerate(λ_output):
        mask, ϕ = instrument_lsf_kernel(λ_input, λ, R, **kwargs)
        K[mask, o] += ϕ 
    return K
    

def instrument_lsf_sparse_matrix(λ_input: np.array, λ_output: np.array, R: Tuple[int, float], **kwargs):
    """
    Construct a sparse matrix to convolve fluxes at input wavelengths (λ_input) at an instrument spectral
    resolution (R) and resample to the given output wavelengths (λ_output).

    :param λ_input:
        A N-length array of input wavelength values.

    :param λ_output:
        A M-length array of output wavelength values.

    :param R:
        Spectral resolution.
    
    :returns:
        A (N, M) sparse array representing a convolution kernel.
    """    
    K = instrument_lsf_dense_matrix(λ_input, λ_output, R, **kwargs)
    return sparse.coo_array(K).tocsc()


def rotational_broadening_sparse_matrix(λ: np.array, vsini: Tuple[int, float], epsilon: Tuple[int, float]):
    """
    Construct a sparse matrix to convolve fluxes at input wavelengths (λ) with a rotational broadening kernel
    with a given vsini and epsilon.
    
    :param λ:
        A N-length array of input wavelength values.
        
    :param vsini:
        The projected rotational velocity of the star in km/s.
    
    :param epsilon:
        The limb darkening coefficient.
    
    :returns:
        A (N, N) sparse array representing a convolution kernel.
    """

    # Let's pre-calculate some things that are needed in the hot loop.
    denominator = np.pi * vsini * (1.0 - epsilon / 3.0)
    c1 = 2.0 * (1.0 - epsilon) / denominator
    c2 = 0.5 * np.pi * epsilon / denominator    

    vsini_c = vsini / 299792.458
    scale = vsini_c / (λ[1] - λ[0]) # assume uniform sampling
    N = λ.size

    data, row_index, col_index = ([], [], [])
    for i, λ_i in enumerate(λ):
        n_pix = int(np.ceil(λ_i * scale))
        si, ei = (max(0, i - n_pix), min(i + n_pix + 1, N))
        mask = slice(si, ei) # ignoring edge effects

        λ_delta_max = λ_i * vsini_c
        λ_delta = λ[mask] - λ_i
        λ_ratio_sq = (λ_delta / λ_delta_max)**2.0
        ϕ = c1 * np.sqrt(1.0 - λ_ratio_sq) + c2 * (1.0 - λ_ratio_sq)
        ϕ[λ_ratio_sq >= 1.0] = 0.0 # flew too close to the sun
        ϕ /= np.sum(ϕ)

        data.extend(ϕ)
        row_index.extend(list(range(si, ei)))
        col_index.extend([i] * (ei - si))
    
    return sparse.csr_matrix(
        (data, (row_index, col_index)), 
        shape=(λ.size, λ.size)
    )
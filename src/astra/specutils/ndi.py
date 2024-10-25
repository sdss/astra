import numpy as np
import warnings
from collections import OrderedDict
from typing import List, Tuple, Optional
from itertools import cycle

try:
    import jax.numpy.linalg as linalg
except ImportError:
    from numpy import linalg
else:
    print("Using JAX linear algebra")
#from jax.numpy import linalg as jax_linalg
#from numpy import linalg

def resample_spectrum(
    resample_wavelength: np.array,
    wavelength: np.array,
    flux: np.array,
    ivar: Optional[np.array] = None,
    flags: Optional[np.array] = None,
    mask: Optional[np.array] = None,
    L: Optional[int] = None,
    P: Optional[int] = None,
    min_resampled_flag_value: Optional[float] = 0.1,
    grow: Optional[int] = 1,
    rcond: Optional[float] = None,
) -> Tuple[np.array, np.array]:
    """
    Sample a spectrum on a wavelength array given a set of pixels recorded from one or many visits.
    
    :param resample_wavelength:
        A ``M_star``-length array of wavelengths to sample the spectrum on. In the paper, this is equivalent 
        to the output-spectrum pixel grid $x_\star$.

    :param wavelength:
        A ``M``-length array of wavelengths from individual visits. If you have $N$ spectra where 
        the $i$th spectrum has $m_i$ pixels, then $M = \sum_{i=1}^N m_i$, and this array represents a
        flattened 1D array of all wavelength positions. In the paper, this is equivalent to the 
        input-spectrum pixel grid $x_i$.
    
    :param flux:
        A ``M``-length array of flux values from individual visits. In the paper, this is equivalent to 
        the observations $y_i$.
    
    :param ivar: [optional]
        A ``M``-length array of inverse variance values from individual visits. In the paper, this is 
        equivalent to the individual inverse variance matrices $C_i^{-1}$.

    :param flags: [optional]
        A ``M``-length array of bitmask flags from individual visits.
    
    :param mask: [optional]
        A ``M``-length array of boolean values indicating whether a pixel should be used or not in
        the resampling (`True` means mask the pixel, `False` means use the pixel). If `None` is
        given then all pixels will be used. The `mask` is only relevant for sampling the flux and
        inverse variance values, and not the flags.
        
    :param L: [optional]
        The length scale for the Fourier modes.  If you don't know what you're doing, leave this as `None`.
        If `None` is given, this will default to the peak-to-peak range of `resample_wavelength`,
        after accounting for fully masked regions or pixels with no data.

    :param P: [optional]
        The number of Fourier modes to use when solving for the resampled spectrum. If you don't know
        what you're doing, leave this as `None`. If `None` is given, this will default to the number 
        of pixels to solve in `resample_wavelength`, after accounting for fully masked regions or 
        pixels with no data.
    
    :param min_resampled_flag_value: [optional]
        The minimum value of a flag to be considered "set" in the resampled spectrum. This is
        used to reconstruct the flags in the resampled spectrum. The default is 0.1, but a
        sensible choice could be 1/N, where N is the number of visits.    
        
    :param rcond: [optional]
        Cutoff for small singular values. 
                
        
    :returns:
        A four-length tuple of ``(flux, ivar, flags, meta)`` where ``flux`` is the resampled flux values 
        and ``ivar`` is the variance of the resampled fluxes, ``flags`` are the resampled flags, and
        ``meta`` is a dictionary of metadata.
        
        All three pixel arrays are length $M_\star$ (the same as ``resample_wavelength``).
    """
    
    wavelength, flux, ivar, mask = _check_shapes_and_sort(wavelength, flux, ivar, mask)

    x_star = np.array(resample_wavelength)
    visit_data_mask, star_data_mask = _create_visit_mask_and_star_mask(x_star, wavelength, flux, ivar, mask, grow=grow)
    
    L = L or np.ptp(x_star[~star_data_mask])
    P = P or x_star[~star_data_mask].size
    
    if flags is not None:
        # Construct the full X, as we will need it for flags.
        X_full = construct_design_matrix(wavelength, L, P)
        X_star_full = construct_design_matrix(x_star, L, P)
        X_star = X_star_full[~star_data_mask]
        X = X_full[~visit_data_mask]
    else:
        # Only construct the X we need
        X_star = construct_design_matrix(x_star[~star_data_mask], L, P)        
        X = construct_design_matrix(wavelength[~visit_data_mask], L, P) # M x P
        
    Y = flux[~visit_data_mask]
    Cinv = ivar[~visit_data_mask]
    
    # We need to solve for theta, which is the resampled spectrum at P pixels
    #   theta = X_star @ (X.T @ C^(-1) @ X)^(-1) @ X.T @ C^(-1) @ Y
    # and we want to avoid this big (pseudo-)inverse (X.T @ C^(-1) @ X)^(-1)

    # To avoid overloading nomenclature, we will solve for G in the equation
    #   A @ G = B
    # where
    #   A = (X.T @ C^(-1) @ X)
    #   B = X.T @ C^(-1) @ Y
    # such that
    #   (X.T @ C^(-1) @ X) @ G = X.T @ C^(-1) @ Y
    # and
    #   G = (X.T @ C^(-1) @ X)^(-1) @ X.T @ C^(-1) @ Y
    # and
    #   theta = X_star @ G
    
    XtCinv = X.T * Cinv    
    XtCinvX = XtCinv @ X
    XtCinvY = XtCinv @ Y
    G, G_residuals, G_rank, G_s = linalg.lstsq(XtCinvX, XtCinvY, rcond=rcond)
    condition_number = np.max(G_s)/np.min(G_s)
    
    y_star_masked = X_star @ G
    
    # For the inverse variances we need
    #   C_star = (X_star @ (X.T @ C^(-1) @ X)^(-1) @ X_star.T)^-1
    # The problematic term is 
    #   (X.T @ C^(-1) @ X)^(-1)
    # so we will solve for H in A @ H = B where
    #   A = (X.T @ C^(-1) @ X)
    #   B = X_star.T
    # such that
    #   H = (X.T @ C^(-1) @ X)^(-1) @ X_star.T
    # and
    #   C_star = X_star @ H

    H, H_residuals, H_rank, H_s = linalg.lstsq(XtCinvX, X_star.T, rcond=rcond)
    ivar_star_masked = 1/np.diag(X_star @ H)

    if np.any(ivar_star_masked < 0):
        warnings.warn("Clipping negative inverse variances to zero.")
        ivar_star_masked = np.clip(ivar_star_masked, 0, None)

    separate_flags = OrderedDict()
    flags_star = np.zeros(x_star.size, dtype=np.uint64)
    if flags is not None:        
        # For flags, we take the identity
        #   F_flags = X_star @ (X.T @ X)^(-1) @ X.T @ flag 
        # 
        raise a
        F, F_residuals, F_rank, F_s = linalg.lstsq(X.T @ X, X.T, rcond=rcond)        
        A_star_flags = X_star @ F

        separated_flags = _separate_flags(flags)
        for bit, flag in separated_flags.items():
            separate_flags[bit] = A_star_flags @ flag
            
        # Reconstruct flags
        for k, values in separate_flags.items():
            if np.max(values) > 0:                
                flag = (values > min_resampled_flag_value).astype(int)
                flags_star += (flag * (2**k)).astype(flags_star.dtype)

    y_star = _un_mask(y_star_masked, star_data_mask, np.nan)
    ivar_star = _un_mask(ivar_star_masked, star_data_mask, 0)
    
    meta = dict(
        condition_number=condition_number,
        L=L,
        P=P,
        separate_flags=separate_flags
    )

    return (y_star, ivar_star, flags_star, meta)


def _un_mask(values, mask, default, dtype=float):
    v = default * np.ones(mask.shape, dtype=dtype)
    v[~mask] = values
    return v


def _check_shapes_and_sort(wavelength, flux, ivar, mask):
    wavelength = np.array(wavelength)
    idx = np.argsort(wavelength)

    P = wavelength.size
    flux = _check_shape("flux", flux, P)[idx]
    if ivar is not None:
        ivar = _check_shape("ivar", ivar, P)[idx]
    else:
        ivar = np.ones_like(flux)
    if mask is not None:
        mask = _check_shape("mask", mask, P).astype(bool)[idx]
    else:
        mask = np.zeros(flux.shape, dtype=bool)
    return (wavelength, flux, ivar, mask)
        
        
def _create_visit_mask_and_star_mask(x_star, wavelength, flux, ivar, mask, grow=1):
    wl_min, wl_max = wavelength[[0, -1]]
    
    has_data = (~mask) & (
        (ivar > 0) 
    &   np.isfinite(flux) 
    &   np.isfinite(ivar)
    &   (wavelength >= x_star[0])
    &   (x_star[-1] >= wavelength)
    )
    visit_data_mask = ~has_data

    indices = 1 + np.where(np.diff(has_data))[0]
    if has_data[0]:
        indices = np.hstack([0, indices])
    if (indices.size % 2) > 0:
        indices = np.hstack([indices, has_data.size - 1])

    star_data_mask = np.ones(x_star.size, dtype=bool)
    for wl_min, wl_max in wavelength[indices.reshape((-1, 2))]:
        si, ei = x_star.searchsorted([wl_min, wl_max])
        si, ei = np.clip(np.array([si + grow, ei - grow]), 0, x_star.size)
        star_data_mask[si:ei] = False
    return (visit_data_mask, star_data_mask)


def _separate_flags(flags: np.array):
    """
    Separate flags into a dictionary of flags for each bit.
    
    :param flags:
        An ``M``-length array of flag values.
    
    :returns:
        A dictionary of flags, where each key is a bit and each value is an array of 0s and 1s.
    """
    separated = OrderedDict()
    for q in range(1 + int(np.log2(np.max(flags)))):
        is_set = (flags & np.uint64(2**q)) > 0
        separated[q] = np.clip(is_set, 0, 1)
    return separated    


def construct_design_matrix(wavelength: np.array, L: float, P: int):
    """
    Take in a set of wavelengths and return the Fourier design matrix.

    :param wavelength:
        An ``M``-length array of wavelength values.
        
    :param L:
        The length scale, usually taken as ``max(wavelength) - min(wavelength)``.

    :param P:
        The number of Fourier modes to use.
    
    :returns:
        A design matrix of shape (M, P).
    """
    # TODO: This could be replaced with something that makes use of finufft.
    scale = (np.pi * wavelength) / L
    X = np.ones((wavelength.size, P), dtype=float)
    for j, f in zip(range(1, P), cycle((np.sin, np.cos))):
        X[:, j] = f(scale * (j + (j % 2)))
    return X



def _check_shape(name, a, P):
    a = np.array(a)
    if a.size != P:
        raise ValueError(f"{name} must be the same size as wavelength")
    return a

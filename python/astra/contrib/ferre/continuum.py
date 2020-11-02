
import numpy as np
import scipy.signal

def median_filtered_correction(
        wavelength,
        normalised_observed_flux, 
        normalised_observed_flux_err,
        normalised_model_flux,
        width=151,
        bad_minimum_flux=0.01,
        non_finite_err_value=1e10,
        mode="nearest",
        **kwargs        
    ):
    """
    Perform a median filter correction to the normalised observed spectrum, given some best-fitting normalised model flux.

    :param wavelength:
        A 1-D array of wavelength values, or a N-length tuple containing separate wavelength segments.

    :param normalised_observed_flux:
        The pseudo-continuum normalised observed flux array. This should be the same format as `wavelength`.
    
    :param normalised_observed_flux_err:
        The 1-sigma uncertainty in the pseudo-continuum normalised observed flux array. This should have the same format as `wavelength`.
    
    :param normalised_model_flux:
        The best-fitting pseudo-continuum normalised model flux. This should have the same format as `wavelength`.

    :param width: [optional]
        The width (int) or widths (N-length tuple of ints) for the median filter (default: 151).
    
    :param bad_minimum_flux: [optional]
        The value at which to set pixels as bad and median filter over them. This should be a float, or a N-length tuple of
        floats, or `None` to set no low-flux filtering (default: 0.01).
    
    :param non_finite_err_value: [optional]
        The error value to set for pixels with non-finite fluxes (default: 1e10).

    :param mode: [optional]
        The mode to supply to `scipy.ndimage.filters.median_filter` (default: nearest).

    :returns:
        A two-length tuple of the pseudo-continuum segments, and the corrected pseudo-continuum-normalised observed flux errors.
    """

    if isinstance(wavelength, np.ndarray):
        wavelength = (wavelength, )
    if isinstance(normalised_observed_flux, np.ndarray):
        normalised_observed_flux = (normalised_observed_flux, )
    if isinstance(normalised_observed_flux_err, np.ndarray):
        normalised_observed_flux_err = (normalised_observed_flux_err, )
    if isinstance(normalised_model_flux, np.ndarray):
        normalised_model_flux = (normalised_model_flux, )
        
    N = len(wavelength)
    if isinstance(width, int):
        width = tuple([width] * N)
    if isinstance(bad_minimum_flux, float):
        bad_minimum_flux = tuple([bad_minimum_flux] * N)

    segment_continuum = []
    segment_errs = []
    for i, (wl, flux, err, model) \
    in enumerate(zip(wavelength, normalised_observed_flux, normalised_observed_flux_err, normalised_model_flux)):

        median_filter = scipy.ndimage.filters.median_filter(
            flux, 
            [5 * width[i]],
            mode=mode,
            **kwargs
        )

        bad = np.where(flux < bad_minimum_flux[i])[0]
        flux_copy = flux.copy()
        flux_copy[bad] = median_filter[bad]

        err_copy = err.copy() * flux / flux_copy

        non_finite = ~np.isfinite(err_copy)
        err_copy[non_finite] = non_finite_err_value

        # Get ratio of observed / model and check that edge is reasonable.
        ratio = flux_copy / model
        correction = scipy.ndimage.filters.median_filter(
            ratio,
            [width[i]],
            mode=mode,
            **kwargs
        )
        bad = (~np.isfinite(correction)) + np.isclose(correction, 0)
        correction[bad] = 1.0

        segment_continuum.append(correction)
        segment_errs.append(err_copy)

    return (tuple(segment_continuum), tuple(segment_errs))

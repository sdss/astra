""" 
Continuum-normalization utilities.
"""

import numpy as np
import pickle
from ast import literal_eval

from astra.utils import log
from astra.contrib.ferre import utils
from astra.contrib.aspcap import bitmask
from astra.database import (astradb, session)

from scipy.ndimage.filters import median_filter

def median_and_inflate_errors(
        spectrum,
        slice_args=None,
        axis=1,
        ivar_multiplier_for_sig_skyline=1e-4,
        ivar_min=0,
        ivar_max=40_000,
        flux_bad_pixel=1e-4,
        ivar_bad_pixel=1e-20,
        median_filter_correction_from_task_id_like=None,
        instance=None,
        **kwargs
    ):
    """
    Continuum-normalize the input spectrum by the median value, and inflate the errors
    due to skylines and bad pixels.
    """

    pixel_bit_mask = bitmask.PixelBitMask()

    # Normalize.
    continuum = np.nanmedian(spectrum.flux.value, axis=axis).reshape((-1, 1))
    spectrum._data /= continuum
    spectrum._uncertainty.array *= continuum**2 
    
    # Increase the error around significant skylines.
    skyline_mask = ((spectrum.meta["bitmask"] & pixel_bit_mask.get_value('SIG_SKYLINE')) > 0)
    spectrum._uncertainty.array[skyline_mask] *= ivar_multiplier_for_sig_skyline

    # Set bad pixels to have no useful data.
    bad = ~np.isfinite(spectrum.flux.value) \
        | ~np.isfinite(spectrum.uncertainty.array) \
        | (spectrum.flux.value < 0) \
        | (spectrum.uncertainty.array < 0) \
        | ((spectrum.meta["bitmask"] & pixel_bit_mask.get_level_value(1)) > 0)
    
    spectrum._data[bad] = flux_bad_pixel
    spectrum._uncertainty.array[bad] = ivar_bad_pixel
    
    # Ensure a minimum error.
    # TODO: This seems like a pretty bad idea!
    spectrum._uncertainty.array = np.clip(spectrum._uncertainty.array, ivar_min, ivar_max) # sigma = 5e-3

    if slice_args is not None:
        slices = tuple([slice(*each) for each in slice_args])

        spectrum._data = spectrum._data[slices]
        spectrum._uncertainty.array = spectrum._uncertainty.array[slices]

        try:
            spectrum.meta["snr"] = spectrum.meta["snr"][slices[0]]
        except:
            log.warning(f"Unable to slice 'snr' metadata with {slice_args}")
    

    if median_filter_correction_from_task_id_like is not None:

        upstream_pk = instance.parameters.get("upstream_pk", None)
        if upstream_pk is None:
            raise ValueError(f"cannot do median filter correction because no upstream_pk parameter for {instance}")

        upstream_pk = literal_eval(upstream_pk)

        # There could be many upstream tasks listed, so we should get the matching one.
        q = session.query(astradb.TaskInstance)\
                   .filter(astradb.TaskInstance.pk.in_(upstream_pk))\
                   .filter(astradb.TaskInstance.task_id.like(median_filter_correction_from_task_id_like))

        upstream_instance = q.one_or_none()
        if upstream_instance is None:
            raise RuntimeError(f"cannot find upstream instance in {upstream_pk} matching {median_filter_correction_from_task_id_like}")

        log.info(f"Applying median filtered correction\n\tto {instance}\n\tfrom {upstream_instance}")

        upstream_path = utils.output_data_product_path(upstream_instance.pk)
        with open(upstream_path, "rb") as fp:
            result, data = pickle.load(fp)
        
        # Need number of pixels from header
        n_pixels = [header["NPIX"] for header in utils.read_ferre_headers(utils.expand_path(instance.parameters["header_path"]))][1:]

        # Get the segment indices using the data mask and the known number of pixels.
        indices = 1 + np.cumsum(data["mask"]).searchsorted(np.cumsum(n_pixels))
        segment_indices = np.vstack([indices - n_pixels, indices]).T

        cont = median_filtered_correction(
            wavelength=data["wavelength"],
            # TODO: Check this median filtered correction.
            normalised_observed_flux=data["flux"] / data["continuum"],
            normalised_observed_flux_err=data["sigma"] / data["continuum"],
            normalised_model_flux=data["normalized_model_flux"],
            segment_indices=segment_indices,
            **kwargs
        )

        spectrum._data /= cont
        spectrum._uncertainty.array *= cont * cont
        
    return spectrum




def median_filtered_correction(
        wavelength,
        normalised_observed_flux, 
        normalised_observed_flux_err,
        normalised_model_flux,
        segment_indices=None,
        median_filter_width=151,
        bad_minimum_flux=0.01,
        non_finite_err_value=1e10,
        valid_continuum_correction_range=(0.1, 10.0),
        mode="nearest",
        **kwargs        
    ):
    """
    Perform a median filter correction to the normalised observed spectrum, given some best-fitting normalised model flux.

    :param wavelength:
        A 1-D array of wavelength values.
        
    :param normalised_observed_flux:
        The pseudo-continuum normalised observed flux array. This should be the same format as `wavelength`.
    
    :param normalised_observed_flux_err:
        The 1-sigma uncertainty in the pseudo-continuum normalised observed flux array. This should have the same format as `wavelength`.
    
    :param normalised_model_flux:
        The best-fitting pseudo-continuum normalised model flux. This should have the same format as `wavelength`.

    :param median_filter_width: [optional]
        The width (int) for the median filter (default: 151).
    
    :param bad_minimum_flux: [optional]
        The value at which to set pixels as bad and median filter over them. This should be a float,
        or `None` to set no low-flux filtering (default: 0.01).
    
    :param non_finite_err_value: [optional]
        The error value to set for pixels with non-finite fluxes (default: 1e10).

    :param valid_continuum_correction_range: [optional]
        A (min, max) tuple of the bounds that the final correction can have. Values outside this range will be set
        as 1.

    :param mode: [optional]
        The mode to supply to `scipy.ndimage.filters.median_filter` (default: nearest).

    :returns:
        A two-length tuple of the pseudo-continuum segments, and the corrected pseudo-continuum-normalised observed flux errors.
    """

    '''
    if isinstance(wavelength, np.ndarray):
        wavelength = (wavelength, )
    if isinstance(normalised_observed_flux, np.ndarray):
        normalised_observed_flux = (normalised_observed_flux, )
    if isinstance(normalised_observed_flux_err, np.ndarray):
        normalised_observed_flux_err = (normalised_observed_flux_err, )
    if isinstance(normalised_model_flux, np.ndarray):
        normalised_model_flux = (normalised_model_flux, )
        
    N = len(normalised_observed_flux)
    if isinstance(width, int):
        width = tuple([width] * N)
    if isinstance(bad_minimum_flux, float):
        bad_minimum_flux = tuple([bad_minimum_flux] * N)
    '''
    wavelength = np.atleast_1d(wavelength).flatten()
    normalised_observed_flux = np.atleast_1d(normalised_observed_flux).flatten()
    normalised_observed_flux_err = np.atleast_1d(normalised_observed_flux_err).flatten()
    normalised_model_flux = np.atleast_1d(normalised_model_flux).flatten()

    data = (wavelength, normalised_observed_flux, normalised_observed_flux_err, normalised_model_flux)

    if segment_indices is None:
        segment_indices = np.array([[0, wavelength.size]])
    
    continuum = np.nan * np.ones_like(normalised_observed_flux)

    #E = 9

    for j, (start, end) in enumerate(segment_indices):

        wl = wavelength[start:end]
        flux = normalised_observed_flux[start:end]
        flux_err = normalised_observed_flux_err[start:end]
        model_flux = normalised_model_flux[start:end]
        
        # TODO: It's a little counter-intuitive how this is documented, so we should fix that.
        #       Or allow for a MAGIC number 5 instead.
        median = median_filter(flux, [5 * median_filter_width], mode=mode)

        bad = np.where(flux < bad_minimum_flux)[0]
        
        flux_copy = flux.copy()
        flux_copy[bad] = median[bad]

        ratio = flux_copy / model_flux

        # Clip edges.
        #ratio[0] = np.median(ratio[:E])
        #ratio[-1] = np.median(ratio[-E:])

        continuum[start:end] = median_filter(ratio, [median_filter_width], mode=mode)

        '''
        err_copy = err.copy() * flux / flux_copy

        non_finite = ~np.isfinite(err_copy)
        err_copy[non_finite] = non_finite_err_value

        # Get ratio of observed / model and check that edge is reasonable.
        ratio = flux_copy / model
        correction = scipy.ndimage.filters.median_filter(
            ratio,
            width[i],
            mode=mode,
            **kwargs
        )
        bad = (~np.isfinite(correction)) + np.isclose(correction, 0)
        correction[bad] = 1.0

        segment_continuum.append(correction)
        segment_errs.append(err_copy)
        '''

    bad = ~np.isfinite(continuum)
    if valid_continuum_correction_range is not None:
        l, u = valid_continuum_correction_range
        bad += (continuum < l) + (continuum > u)
    
    continuum[bad] = 1

    return continuum


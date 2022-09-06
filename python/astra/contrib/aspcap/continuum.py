import numpy as np
from scipy.ndimage.filters import median_filter

import pickle
from astra import log
from astra.utils import expand_path
from astra.contrib.ferre import bitmask
from astra.contrib.ferre.utils import read_ferre_headers
from astra.database.astradb import Task
from astra.tools.continuum.base import NormalizationBase


class MedianNormalizationWithErrorInflation(NormalizationBase):

    """
    Continuum-normalize an input spectrum by the median flux value,
    and inflate the errors due to skylines and bad pixels.
    """

    parameter_names = ()

    def __init__(
        self,
        spectrum,
        axis=1,
        ivar_multiplier_for_sig_skyline=1e-4,
        ivar_min=0,
        ivar_max=40_000,
        bad_pixel_flux=1e-4,
        bad_pixel_ivar=1e-20,
    ) -> None:
        super().__init__(spectrum)
        self.axis = axis
        self.ivar_multiplier_for_sig_skyline = ivar_multiplier_for_sig_skyline
        self.ivar_min = ivar_min
        self.ivar_max = ivar_max
        self.bad_pixel_flux = bad_pixel_flux
        self.bad_pixel_ivar = bad_pixel_ivar

    def __call__(self):

        pixel_bit_mask = bitmask.PixelBitMask()

        # Normalize.
        continuum = np.nanmedian(self.spectrum.flux.value, axis=self.axis).reshape(
            (-1, 1)
        )
        self.spectrum._data /= continuum
        self.spectrum._uncertainty.array *= continuum**2

        # Increase the error around significant skylines.
        skyline_mask = (
            self.spectrum.meta["bitmask"] & pixel_bit_mask.get_value("SIG_SKYLINE")
        ) > 0
        self.spectrum._uncertainty.array[
            skyline_mask
        ] *= self.ivar_multiplier_for_sig_skyline

        # Set bad pixels to have no useful data.
        bad = (
            ~np.isfinite(self.spectrum.flux.value)
            | ~np.isfinite(self.spectrum.uncertainty.array)
            | (self.spectrum.flux.value < 0)
            | (self.spectrum.uncertainty.array < 0)
            | ((self.spectrum.meta["bitmask"] & pixel_bit_mask.get_level_value(1)) > 0)
        )

        self.spectrum._data[bad] = self.bad_pixel_flux
        self.spectrum._uncertainty.array[bad] = self.bad_pixel_ivar

        # Ensure a minimum error.
        # TODO: This seems like a pretty bad idea!
        self.spectrum._uncertainty.array = np.clip(
            self.spectrum._uncertainty.array, self.ivar_min, self.ivar_max
        )  # sigma = 5e-3

        return self.spectrum


class MedianFilterNormalizationWithErrorInflation(
    MedianNormalizationWithErrorInflation
):

    parameter_names = ()

    def __init__(
        self,
        spectrum,
        median_filter_from_task,
        segment_indices=None,
        median_filter_width=151,
        bad_minimum_flux=0.01,
        non_finite_err_value=1e10,
        valid_continuum_correction_range=(0.1, 10.0),
        **kwargs,
    ) -> None:
        super().__init__(spectrum, **kwargs)
        self.median_filter_from_task = median_filter_from_task
        self.segment_indices = segment_indices
        self.median_filter_width = median_filter_width
        self.bad_minimum_flux = bad_minimum_flux
        self.non_finite_err_value = non_finite_err_value
        self.valid_continuum_correction_range = valid_continuum_correction_range
        return None

    def __call__(self):

        # Do standard median normalization.
        spectrum = super().__call__()

        if not isinstance(self.median_filter_from_task, Task):
            median_filter_from_task = Task.get_by_id(int(self.median_filter_from_task))
        else:
            median_filter_from_task = self.median_filter_from_task

        # Need number of pixels from header
        n_pixels = np.array(
            [
                header["NPIX"]
                for header in read_ferre_headers(
                    expand_path(median_filter_from_task.parameters["header_path"])
                )
            ][1:]
        )
        _ = n_pixels.cumsum()
        segment_indices = np.sort(np.hstack([[0], _, _]))[:-1].reshape((-1, 2))

        continuum = []
        for output_data_product in median_filter_from_task.output_data_products:
            with open(output_data_product.path, "rb") as fp:
                output = pickle.load(fp)

            kwds = dict(
                wavelength=output["data"]["wavelength"],
                normalised_observed_flux=output["data"]["flux"]
                / output["data"]["continuum"],
                normalised_observed_flux_err=output["data"]["flux_sigma"]
                / output["data"]["continuum"],
                normalised_model_flux=output["data"]["model_flux"],
                segment_indices=segment_indices,
                median_filter_width=self.median_filter_width,
                bad_minimum_flux=self.bad_minimum_flux,
                non_finite_err_value=self.non_finite_err_value,
                valid_continuum_correction_range=self.valid_continuum_correction_range,
            )
            continuum.append(median_filtered_correction(**kwds))

        continuum = np.array(continuum)

        # Construct mask to match FERRE model grid.
        N, P = spectrum.flux.shape
        mask = np.zeros(P, dtype=bool)
        for si, ei in segment_indices:
            # TODO: Building wavelength mask off just the last wavelength array.
            #       We are assuming all have the same wavelength array.
            s_index, e_index = spectrum.wavelength.value.searchsorted(
                output["data"]["wavelength"][si:ei][[0, -1]]
            )
            mask[s_index : e_index + 1] = True

        continuum_unmasked = np.nan * np.ones((N, P))
        continuum_unmasked[:, mask] = np.array(continuum)

        spectrum._data /= continuum_unmasked
        spectrum._uncertainty.array *= continuum_unmasked * continuum_unmasked

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
    **kwargs,
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

    """
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
    """
    wavelength = np.atleast_1d(wavelength).flatten()
    normalised_observed_flux = np.atleast_1d(normalised_observed_flux).flatten()
    normalised_observed_flux_err = np.atleast_1d(normalised_observed_flux_err).flatten()
    normalised_model_flux = np.atleast_1d(normalised_model_flux).flatten()

    data = (
        wavelength,
        normalised_observed_flux,
        normalised_observed_flux_err,
        normalised_model_flux,
    )

    if segment_indices is None:
        segment_indices = np.array([[0, wavelength.size]])

    continuum = np.nan * np.ones_like(normalised_observed_flux)

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
        # ratio[0] = np.median(ratio[:E])
        # ratio[-1] = np.median(ratio[-E:])

        continuum[start:end] = median_filter(ratio, [median_filter_width], mode=mode)

        """
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
        """

    bad = ~np.isfinite(continuum)
    if valid_continuum_correction_range is not None:
        l, u = valid_continuum_correction_range
        bad += (continuum < l) + (continuum > u)

    continuum[bad] = 1

    return continuum

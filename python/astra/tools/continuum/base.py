"""A base to represent the stellar continuum."""

from __future__ import annotations
import numpy as np
from typing import Optional, Union, Tuple, List
from astra.utils import expand_path
from astra.tools.spectrum import SpectralAxis, Spectrum1D


class Continuum:

    """A base class to represent the stellar continuum."""

    def __init__(
        self,
        spectral_axis: Optional[SpectralAxis] = None,
        regions: Optional[List[Tuple[float, float]]] = None,
        mask: Optional[Union[str, np.array]] = None,
        fill_value: Optional[Union[int, float]] = np.nan,
    ):
        """
        :param spectral_axis: [optional]
            If given, the spectral axis of the spectrum that will be fitted. This is useful when using the same
            class on many spectra where the spectral axis is the same.

        :param regions: [optional]
            A list of two-length tuples of the form (lower, upper) in the same units as the spectral axis.

        :param mask: [optional]
            A boolean array of the same length as the spectral axis, where False indicates a continuum pixel,
            and True indicates a pixel to be masked in the continuum fit.

        :param fill_value: [optional]
            The value to use for pixels where the continuum is not defined.
        """
        if isinstance(mask, str):
            self.mask = np.load(expand_path(mask))
        else:
            self.mask = mask
        self.regions = regions
        self.fill_value = fill_value
        self.spectral_axis = spectral_axis
        return None

    def _initialize(self, spectrum):
        try:
            self._initialized_args
        except AttributeError:
            self._initialized_args = _pixel_slice_and_mask(
                spectrum.wavelength, self.regions, self.mask
            )
        finally:
            return self._initialized_args

    @property
    def num_regions(self):
        """
        Return the number of regions used to fit the continuum.
        """
        return 1 if self.regions is None else len(self.regions)

    def _get_shape(self, spectrum: Spectrum1D):
        """
        Get the shape of the spectrum.

        :param spectrum:
            A spectrum, which could be a 1D spectrum or multiple spectra with the same spectral axis.
        """
        try:
            N, P = spectrum.flux.shape
        except:
            N, P = (1, spectrum.flux.size)
        return (N, P)

    def fit(self, spectrum: Spectrum1D) -> Continuum:
        """
        Fit the continuum in the given spectrum.

        :param spectrum:
            A spectrum.
        """
        raise NotImplementedError("This should be implemented by the sub-classes")

    def __call__(
        self, spectrum: Spectrum1D, theta: Optional[Union[List, np.array, Tuple]] = None
    ) -> np.ndarray:
        """
        Return the estimated continuum given a spectrum and parameters.

        :param spectrum:
            A spectrum.

        :param theta: [optional]
            A set of parameters for the continuum. If not provided, this defaults to the parameters
            previous fit to the spectrum.
        """
        raise NotImplementedError("This should be implemented by the sub-classes")


def _pixel_slice_and_mask(
    spectral_axis: SpectralAxis,
    regions: Optional[List[Tuple[float, float]]] = None,
    mask: Optional[np.array] = None,
):
    """
    Return region slices in pixel space, and the continuum masks to use in each region.

    :param spectral_axis:
        The spectral axis of the spectrum.

    :param regions:
        A list of two-length tuples of the form (lower, upper) in the same units as the spectral axis.

    :param mask:
        A boolean array of the same length as the spectral axis, where False indicates a continuum pixel,
        and True indicates a pixel to be masked in the continuum fit.

    :returns:
        A tuple of two lists, the first containing the pixel slices for each region, and the second containing
        the continuum mask for each region.
    """
    if regions is None:
        region_slices = [(0, spectral_axis.size)]
    else:
        region_slices = []
        for lower, upper in regions:
            # TODO: allow for units/quantities in (lower, upper)?
            region_slices.append(spectral_axis.value.searchsorted([lower, upper]))

    region_masks = []
    if mask is None:
        for lower, upper in region_slices:
            # No mask, keep all pixels as continuum.
            region_masks.append(np.arange(lower, upper, dtype=int))
    else:
        if len(mask) != len(spectral_axis):
            raise ValueError(
                f"Pixel mask and spectral axis have different sizes ({len(mask)} != {len(spectral_axis)})"
            )

        for lower, upper in region_slices:
            # Mask given, exclude those masked.
            region_masks.append(np.where(~mask[lower:upper])[0] + lower)

    return (region_slices, region_masks)

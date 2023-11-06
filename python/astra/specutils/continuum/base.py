"""A base to represent the stellar continuum."""

import numpy as np
from typing import Optional, Union, Tuple, List

class Continuum:

    """A base class to represent the stellar continuum."""

    def __init__(
        self,
        wavelength: Optional[np.array] = None,
        mask: Optional[Union[str, np.array]] = None,
        regions: Optional[List[Tuple[float, float]]] = None,
        fill_value: Optional[Union[int, float]] = np.nan,
    ):
        """
        :param regions: [optional]
            A list of two-length tuples of the form (lower, upper) in the same units as the spectral axis.

        :param fill_value: [optional]
            The value to use for pixels where the continuum is not defined.
        """
        
        if isinstance(mask, str):
            self.mask = np.load(expand_path(mask))
        else:
            self.mask = mask        
        self.regions = regions
        self.fill_value = fill_value
        self.wavelength = wavelength
        return None

    def _initialize(self, wavelength):
        try:
            self._initialized_args
        except AttributeError:
            self._initialized_args = _pixel_slice_and_mask(
                wavelength, self.regions, self.mask
            )
        finally:
            return self._initialized_args


    @property
    def num_regions(self):
        """
        Return the number of regions used to fit the continuum.
        """
        return 1 if self.regions is None else len(self.regions)


    def fit(self, spectrum):
        """
        Fit the continuum in the given spectrum.

        :param spectrum:
            A spectrum.
        """
        raise NotImplementedError("This should be implemented by the sub-classes")


    def _get_shape(self, flux):
        """
        Get the shape of the spectrum.

        """
        try:
            N, P = flux.shape
        except:
            N, P = (1, flux.size)
        return (N, P)


    def _get_region_slices(self, spectrum):

        if self.regions is None:
            return [(0, spectrum.wavelength.size)]
    
        slices = []
        for lower, upper in self.regions:
            si, ei = spectrum.wavelength.searchsorted([lower, upper])
            slices.append((si, 1 + ei))
        return slices
    
    

def _pixel_slice_and_mask(
    wavelength: np.array,
    regions: Optional[List[Tuple[float, float]]] = None,
    mask: Optional[np.array] = None,
):
    """
    Return region slices in pixel space, and the continuum masks to use in each region.

    :param wavelength:
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
        region_slices = [(0, wavelength.size)]
    else:
        region_slices = []
        for lower, upper in regions:
            # TODO: allow for units/quantities in (lower, upper)?
            region_slices.append(wavelength.searchsorted([lower, upper]))

    region_masks = []
    if mask is None:
        for lower, upper in region_slices:
            # No mask, keep all pixels as continuum.
            region_masks.append(np.arange(lower, upper, dtype=int))
    else:
        if len(mask) != len(wavelength):
            # Assume mask is a list of regions to mask.
            constructed_mask = np.zeros(len(wavelength), dtype=bool)
            for lower, upper in mask:
                idx_lower, idx_higher = np.clip(
                    wavelength.searchsorted([lower, upper]) - 1,
                    0,
                    wavelength.size - 1
                )
                constructed_mask[idx_lower:idx_higher] = True
            
            for lower, upper in region_slices:
                # Mask given, exclude those masked.
                region_masks.append(np.where(~constructed_mask[lower:upper])[0] + lower)
        else:
            for lower, upper in region_slices:
                # Mask given, exclude those masked.
                region_masks.append(np.where(~mask[lower:upper])[0] + lower)

    return (region_slices, region_masks)
    
"""A base to represent the stellar continuum."""

import numpy as np
from typing import Optional, Union, Tuple, List

class Continuum:

    """A base class to represent the stellar continuum."""

    def __init__(
        self,
        regions: Optional[List[Tuple[float, float]]] = None,
        fill_value: Optional[Union[int, float]] = np.nan,
    ):
        """
        :param regions: [optional]
            A list of two-length tuples of the form (lower, upper) in the same units as the spectral axis.

        :param fill_value: [optional]
            The value to use for pixels where the continuum is not defined.
        """
        self.regions = regions
        self.fill_value = fill_value
        return None


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


    def _get_region_slices(self, spectrum):

        if self.regions is None:
            return [(0, spectrum.wavelength.size)]
    
        slices = []
        for lower, upper in self.regions:
            si, ei = spectrum.wavelength.searchsorted([lower, upper])
            slices.append((si, 1 + ei))
        return slices
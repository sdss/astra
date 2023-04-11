"""Represent the stellar continuum with a scalar."""

import numpy as np
from typing import Optional, Union, Tuple, List

from astra.specutils.continuum.base import Continuum


class Scalar(Continuum):

    """Represent the stellar continuum with a scalar."""

    available_methods = {
        "mean": np.nanmean,
        "median": np.nanmedian,
        "max": np.nanmax,
        "min": np.nanmin,
    }

    def __init__(
        self,
        method="mean",
        regions: Optional[List[Tuple[float, float]]] = None,
        fill_value: Optional[Union[int, float]] = np.nan,
        **kwargs,
    ) -> None:
        f"""
        :param method: [optional]
            The method used to estimate the continuum. Must be one of: {', '.join(Scalar.available_methods)}.

        """    + Continuum.__init__.__doc__
        super(Scalar, self).__init__(regions=regions, fill_value=fill_value, **kwargs)
        try:
            self.callable_method = self.available_methods[f"{method}".lower()]
        except KeyError:
            raise ValueError(
                f"Method must be one of {', '.join(list(self.available_methods.keys()))}"
            )
        else:
            self.method = method
        return None


    def fit(self, spectrum) -> np.ndarray:
        """
        Fit the continuum in the given spectrum.

        :param spectrum:
            The input spectrum.
        
        :returns:
            A continuum array of the same length as the spectrum flux array.
        """
        continuum = self.fill_value * np.ones_like(spectrum.flux)
        for si, ei in self._get_region_slices(spectrum):
            continuum[si:ei] = self.callable_method(spectrum.flux[si:ei])
        return continuum
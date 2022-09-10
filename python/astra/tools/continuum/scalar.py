"""Represent the stellar continuum with a scalar."""

from __future__ import annotations
import numpy as np
from astra.tools.spectrum import SpectralAxis, Spectrum1D
from typing import Optional, Union, Tuple, List

from astra.tools.continuum.base import Continuum


class Scalar(Continuum):

    """Represent the stellar continuum with a scalar."""

    available_methods = {
        "mean": np.mean,
        "median": np.median,
    }

    def __init__(
        self,
        method="mean",
        spectral_axis: Optional[SpectralAxis] = None,
        regions: Optional[List[Tuple[float, float]]] = None,
        mask: Optional[np.array] = None,
        fill_value: Optional[Union[int, float]] = np.nan,
        **kwargs,
    ) -> None:
        (
            """
        :param methods: [optional]
            The method used to estimate the continuum. Must be one of `Scalar.available_methods`.

        """
            + Continuum.__init__.__doc__
        )
        super(Scalar, self).__init__(
            spectral_axis=spectral_axis,
            regions=regions,
            mask=mask,
            fill_value=fill_value,
            **kwargs,
        )
        try:
            self.callable_method = self.available_methods[method]
        except KeyError:
            raise ValueError(
                f"Method must be one of {', '.join(list(self.available_methods.keys()))}"
            )
        else:
            self.method = method
        return None

    def fit(self, spectrum: Spectrum1D) -> Scalar:
        _initialized_args = self._initialize(spectrum)
        N, P = self._get_shape(spectrum)

        all_flux = spectrum.flux.value.reshape((N, P))
        self.theta = np.empty((N, self.num_regions, 1))
        for i, flux in enumerate(all_flux):
            for j, (_, indices) in enumerate(zip(*_initialized_args)):
                self.theta[i, j, 0] = self.callable_method(flux[indices])
        return self

    def __call__(
        self, spectrum: Spectrum1D, theta: Optional[Union[List, np.array, Tuple]] = None
    ) -> np.ndarray:
        if theta is None:
            theta = self.theta

        _initialized_args = self._initialize(spectrum)

        N, P = self._get_shape(spectrum)
        continuum = self.fill_value * np.ones((N, P))
        for i in range(N):
            for j, ((lower, upper), _) in enumerate(zip(*_initialized_args)):
                continuum[i, slice(lower, upper)] = self.theta[i, j, 0]
        return continuum.reshape(spectrum.flux.shape)

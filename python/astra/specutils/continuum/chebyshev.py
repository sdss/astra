"""Represent the stellar continuum with a Chebyshev polynomial."""

from __future__ import annotations
import numpy as np
from astra.specutils.spectrum import SpectralAxis, Spectrum1D
from typing import Optional, Union, Tuple, List
from astropy.nddata import StdDevUncertainty
from astra.specutils.continuum.base import Continuum


class Chebyshev(Continuum):

    """Represent the stellar continuum with a Chebyshev polynomial."""

    def __init__(
        self,
        deg: int,
        spectral_axis: Optional[SpectralAxis] = None,
        regions: Optional[List[Tuple[float, float]]] = None,
        mask: Optional[np.array] = None,
        fill_value: Optional[Union[int, float]] = np.nan,
        **kwargs,
    ) -> None:
        (
            """
        :param deg:
            The deg of the Chebyshev polynomial.
        """
            + Continuum.__init__.__doc__
        )

        super(Chebyshev, self).__init__(
            spectral_axis=spectral_axis,
            regions=regions,
            mask=mask,
            fill_value=fill_value,
            **kwargs,
        )
        self.deg = deg
        return None

    def fit(self, spectrum: Spectrum1D) -> Chebyshev:
        _initialized_args = self._initialize(spectrum)
        N, P = self._get_shape(spectrum)

        flux = np.nan_to_num(spectrum.flux).astype(np.float32).reshape((N, P))
        e_flux = np.nan_to_num(spectrum.ivar**-0.5).astype(np.float32).reshape((N, P))

        self.theta = np.empty((N, self.num_regions, self.deg + 1))
        for i in range(N):
            for j, ((lower, upper), indices) in enumerate(zip(*_initialized_args)):
                x = np.linspace(-1, 1, upper - lower)
                # Restrict to finite values.
                y = flux[i, indices]
                w = 1.0 / e_flux[i, indices]
                finite = np.isfinite(y * w)
                f = np.polynomial.Chebyshev.fit(
                    x[indices - lower][finite],
                    y[finite],
                    self.deg,
                    w=w[finite],
                )
                self.theta[i, j] = f.convert().coef
        return self

    def __call__(
        self, spectrum: Spectrum1D, theta: Optional[Union[List, np.array, Tuple]] = None, **kwargs
    ) -> np.ndarray:
        if theta is None:
            theta = self.theta

        _initialized_args = self._initialize(spectrum)

        N, P = self._get_shape(spectrum)
        continuum = self.fill_value * np.ones((N, P))

        for i in range(N):
            for j, ((lower, upper), _) in enumerate(zip(*_initialized_args)):
                continuum[i, slice(lower, upper)] = np.polynomial.chebyshev.chebval(
                    np.linspace(-1, 1, upper - lower), self.theta[i, j]
                )
        return continuum.reshape(spectrum.flux.shape)

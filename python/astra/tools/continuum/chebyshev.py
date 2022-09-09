import numpy as np
from astra.tools.spectrum import Spectrum1D
from typing import Optional, Union, Tuple, List
from astropy.nddata import StdDevUncertainty
from astra.tools.continuum.base import Continuum


class Chebyshev(Continuum):

    """Represent the stellar continuum with a Chebyshev polynomial."""

    def __init__(self, *args, deg, **kwargs):
        """Represent the stellar continuum with a Chebyshev polynomial.

        :param deg:
            The deg of the Chebyshev polynomial.
        """
        super(Chebyshev, self).__init__(*args, **kwargs)
        self.deg = deg
        return None

    def fit(self, spectrum: Spectrum1D):
        _initialized_args = self._initialize(spectrum)
        N, P = self._get_shape(spectrum)

        flux = spectrum.flux.value.reshape((N, P))
        e_flux = spectrum.uncertainty.represent_as(StdDevUncertainty).array.reshape(
            (N, P)
        )

        self.theta = np.empty((N, self.num_regions, self.deg + 1))
        for i in range(N):
            for j, ((lower, upper), indices) in enumerate(zip(*_initialized_args)):
                x = np.linspace(-1, 1, upper - lower)
                f = np.polynomial.Chebyshev.fit(
                    x[indices - lower],
                    flux[i, indices],
                    self.deg,
                    w=1.0 / e_flux[i, indices],
                )
                self.theta[i, j] = f.convert().coef
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
                continuum[i, slice(lower, upper)] = np.polynomial.chebyshev.chebval(
                    np.linspace(-1, 1, upper - lower), self.theta[i, j]
                )
        return continuum.reshape(spectrum.flux.shape)

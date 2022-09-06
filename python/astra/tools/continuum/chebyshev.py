import numpy as np
from numpy.polynomial.chebyshev import chebval
from astra.tools.continuum.base import NormalizationBase


class ChebyshevPolynomial(NormalizationBase):

    """
    Represent the stellar continuum with a Chebyshev polynomial.

    :param order:
        The order of the Chebyshev polynomial to use.
    """

    parameter_names = ()

    def __init__(self, order) -> None:
        super().__init__()
        self.order = int(order)
        self.parameter_names = tuple([f"c_{i}" for i in range(self.order)])
        return None

    def initial_guess(self, spectrum):
        try:
            N, P = spectrum.flux.shape
        except:
            P = spectrum.flux.size
        self.x = np.linspace(-1, 1, P)
        x0 = np.zeros(self.n_chebyshev)
        x0[0] = 1
        return x0

    def __call__(self, *theta, **kwargs):
        return chebval(self.x, theta)

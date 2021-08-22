
import numpy as np
from tqdm import tqdm

from astra.database import astradb, session
from astra.utils import log

from astra.new_operators import (AstraOperator, ApVisitOperator, BossSpecOperator)

from astra.contrib.thepayne_che.networks import Network
from astra.contrib.thepayne_che.fitting import Fit
from astra.contrib.thepayne_che.uncertfit import UncertFit

class ThePayneCheOperator:

    template_fields = ("network_path", "psf_path")

    def __init__(
        self,
        *,
        network_path: str,
        N_chebyshev: int,
        spectral_resolution: int,
        N_pre_search_iter: int,
        N_pre_search: int,
        wavelength_start=None,
        wavelength_end=None,
        slurm_kwargs=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.network_path = network_path
        self.N_chebyshev = N_chebyshev
        self.spectral_resolution = spectral_resolution
        self.N_pre_search_iter = N_pre_search_iter
        self.N_pre_search = N_pre_search
        self.wavelength_start = wavelength_start
        self.wavelength_end = wavelength_end
        self.slurm_kwargs = slurm_kwargs or dict()


    def execute(self, context):

        network = Network()
        network.read_in(self.network_path)

        fit = Fit(network, self.cheb_order)
        fit.N_presearch_iter = self.N_presearch_iter
        fit.N_pre_search = self.N_presearch

        fitter = UncertFit(fit, self.spectral_resolution)

        for pk, instance, path, spectrum in self.yield_data():
        
        assert self.wavelength_start is None
        assert self.wavelength_end is None
        
        fit_result = fitter.run(
            spectrum.wavelength.value,
            spectrum.flux.array[0],
            spectrum.uncertainty.array**-0.5
        )

        raise a



class ThePayneCheApVisitOperator(ThePayneCheOperator, ApVisitOperator):

    """ Estimate stellar labels for a source based on an ApVisit spectrum. """ 

    pass


class ThePayneCheBossSpecOperator(ThePayneCheOperator, ApStarOperator):

    """ Estimate stellar labels for a source based on an ApStar spectrum. """

    pass


class ThePayneCheBossSpecOperator(ThePayneCheOperator, BossSpecOperator):

    """ Estimate stellar labels for a source based on a BOSS Spec spectrum. """ 

    pass




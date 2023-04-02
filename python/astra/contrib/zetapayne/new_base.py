from astra.base import task_decorator
from astra.utils import expand_path, flatten
from astra.tools.spectrum import SpectrumList
from astra.tools.spectrum.utils import spectrum_overlaps
from astra.database.astradb import DataProduct, SDSSOutput
from typing import Iterable
from functools import cache
from peewee import FloatField
from playhouse.postgres_ext import ArrayField

from astra.contrib.zetapayne.Network import Network
from astra.contrib.zetapayne.run_MWMPayne import fit_spectrum


class ZetaPayneOutput(SDSSOutput):

    teff = FloatField()
    e_teff = FloatField()
    logg = FloatField()
    e_logg = FloatField()
    fe_h = FloatField()
    e_fe_h = FloatField()
    vsini = FloatField()
    e_vsini = FloatField()
    v_micro = FloatField()
    e_v_micro = FloatField()
    v_rel = FloatField()
    e_v_rel = FloatField()

    theta = ArrayField(FloatField)

    chi_sq = FloatField()
    reduced_chi_sq = FloatField()



LOG_DIR = expand_path("$MWM_ASTRA/logs/") # temporary



@cache
def read_network(path):
    nn = Network()
    nn.read_in(expand_path(path))
    return nn


def _zeta_payne(
    data_product: DataProduct,
    network_path: str,
    wave_range: Iterable[float],
    spectral_R: int,
    N_chebyshev: int = 15,
    N_presearch_iter: int = 1,
    N_presearch: int = 4000
) -> Iterable[ZetaPayneOutput]:

    network = read_network(network_path)

    opt_kwds = dict(
        N_chebyshev=N_chebyshev,
        N_presearch_iter=N_presearch_iter,
        N_presearch=N_presearch,
        spectral_R=spectral_R,
        wave_range=wave_range,
    )
    
    for spectrum in SpectrumList.read(data_product.path):
        if not spectrum_overlaps(spectrum, wave_range):
            continue

        result, meta, fit = fit_spectrum(
            spectrum, 
            network, 
            opt_kwds, 
            logger=None
        )

        yield (result, meta, fit)


@task_decorator
def zeta_payne(data_product: DataProduct) -> Iterable[ZetaPayneOutput]:
    
    # BOSS first
    boss_results = _zeta_payne(
        data_product,
        network_path="$MWM_ASTRA/component_data/ZetaPayne/NN_OPTIC_n300_b1000_v0.1_27.npz",
        wave_range=(4_500, 10_170),
        spectral_R=2_000,
        N_chebyshev=15,
        N_presearch_iter=1,
        N_presearch=4000
    )

    # APOGEE
    apogee_results = _zeta_payne(
        data_product,
        network_path="$MWM_ASTRA/component_data/ZetaPayne/OV_5K_n300.npz",
        wave_range=(15_000, 17_000),
        spectral_R=22_500,
        N_chebyshev=15,
        N_presearch_iter=1,
        N_presearch=4000
    )

    results = flatten([boss_results, apogee_results])
    

    # Create AstraStar product
    raise a
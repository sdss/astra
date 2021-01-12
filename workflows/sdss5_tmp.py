
import astra
from astra.utils import batcher, get_default
from astra.tasks.io import ApStarFile
from astra.tasks.daily import (GetDailyApStarFiles, get_visits, get_visits_given_star, get_stars)
import astra.contrib.ferre.tasks.sdss5_refactor as aspcap_sdss5
from astra.contrib.ferre.tasks import (sdss5_refactor as aspcap_sdss5, sdss4_refactor as aspcap_sdss4)



# SDSS5 task
sdss5_kwds = list(get_stars(mjd=59205))#[6:9]
sdss5_kwds = batcher(sdss5_kwds, unique=True)
sdss5_task = aspcap_sdss5.EstimateStellarParametersGivenSDSS5ApStarFile(**sdss5_kwds)

sdss4_kwds = dict(
    # ApStar keywords:
    release="dr16",
    apred="r12",
    apstar="stars",
    telescope="apo25m",
    field="000+14",
    prefix="ap",
    obj="2M16505794-2118004",
)

sdss4_task = aspcap_sdss4.EstimateStellarParametersGivenSDSS4ApStarFile(**sdss4_kwds)


sdss4_abundances = aspcap_sdss4.EstimateChemicalAbundancesGivenSDSS4ApStarFile(**sdss4_kwds)
sdss5_abundances = aspcap_sdss5.EstimateChemicalAbundancesGivenSDSS5ApStarFile(**sdss5_kwds)


tasks = [sdss4_task, sdss5_task]
#tasks = [sdss4_abundances, sdss5_abundances]

astra.build(
    tasks,
    local_scheduler=True
)



import astra
from astra.tasks.continuum import Sinusoidal
from astra.tasks.io import ApStarFile
from astra.contrib.thepayne.tasks import (
    EstimateStellarParametersGivenApStarFile, 
    TrainThePayne
)


# Let's define a continuum normalization task for ApStarFiles using a sum of sines
# and cosines.
class ContinuumNormalize(Sinusoidal, ApStarFile):
    
    # Just take the first spectrum, which is stacked by individual pixel weighting.
    # (We will ignore individual visits).
    spectrum_kwds = dict(data_slice=(slice(0, 1), slice(None)))

    def requires(self):
        return ApStarFile(**self.get_common_param_kwargs(ApStarFile))
    

# The EstimateStellarParametersGivenApStarFile task performs no continuum normalisation.
# Here we will create a new class that requires that the observations are continuum normalised.
class EstimateStellarParameters(EstimateStellarParametersGivenApStarFile, ContinuumNormalize):

    def requires(self):
        requirements = dict(model=TrainThePayne(**self.get_common_param_kwargs(TrainThePayne)))
        if not self.is_batch_mode:
            requirements.update(
                observation=ContinuumNormalize(**self.get_common_param_kwargs(ContinuumNormalize))
            )
        return requirements



# Let's run on one star first.
kwds = dict(
    training_set_path="kurucz_data.pkl",
    continuum_regions_path="continuum-regions.list",

    # ApStar keywords:
    release="dr16",
    apred="r12",
    apstar="stars",
    telescope="apo25m",
    field="000+14",
    prefix="ap",
    obj="2M16505794-2118004",
    use_remote=True # Download the apStar file if we don't have it.

)

task = EstimateStellarParameters(**kwds)

astra.build(
    [task],
    local_scheduler=True
)
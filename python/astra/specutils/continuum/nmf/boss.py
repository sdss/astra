
import numpy as np
from astra.specutils.continuum.nmf.base import BaseNMFSinusoidsContinuum, load_components

class BossNMFContinuum(BaseNMFSinusoidsContinuum):
    
    def __init__(
        self,
        components_path="$MWM_ASTRA/pipelines/nmf/20230217_bosz_nmf.pkl",
        deg=3,
        L=10_000,
        regions=[
            [3750, 6250],
            [6350, 12000]
        ]                
    ):
        dispersion = 10**(3.5523 + 1e-4 * np.arange(4648))
        components = load_components(components_path, dispersion.size)
        super(BossNMFContinuum, self).__init__(
            dispersion,
            components,
            deg=deg,
            L=L,
            regions=regions
        )
        return None    


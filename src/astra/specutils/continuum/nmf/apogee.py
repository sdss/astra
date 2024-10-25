
import numpy as np
from astra.specutils.continuum.nmf.base import BaseNMFSinusoidsContinuum, load_components

class ApogeeNMFContinuum(BaseNMFSinusoidsContinuum):
    
    def __init__(
        self,
        components_path="$MWM_ASTRA/pipelines/nmf/20230621_H_32_10_cd_random.pkl",
        deg=3,
        L=1300,
        regions=[
            [15161.84316643 - 35, 15757.66995776 + 40],
            [15877.64179911 - 25, 16380.9845233 + 45],
            [16494.30420468 - 25, 16898.18264895 + 60]
        ],
        pad=50
    ):
        dispersion = 10**(4.179 + 6e-6 * np.arange(8575))
        components = load_components(components_path, dispersion.size - 2 * pad, pad=pad)        
        super(ApogeeNMFContinuum, self).__init__(
            dispersion,
            components,
            deg=deg,
            L=L,
            regions=regions
        )
        return None    


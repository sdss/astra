from typing import Iterable, Optional
import numpy as np

from astra import task
from astra.utils import executable, flatten 
from astra.models.spectrum import Spectrum
from astra.models.the_payne import ThePayne
from astra.pipelines.the_payne.model import estimate_labels
from astra.pipelines.the_payne.utils import read_mask, read_model, read_elem_mask_all

#from collections import OrderedDict


@task
def the_payne(
    spectra: Iterable[Spectrum],
    model_path: str = "$MWM_ASTRA/pipelines/ThePayne/payne_apogee_nn.pkl", 
    mask_path: str = "$MWM_ASTRA/pipelines/ThePayne/payne_apogee_mask.npy", 
    dir_elem_mask: str="$MWM_ASTRA/pipelines/ThePayne/masks/",
    opt_tolerance: Optional[float] = 5e-4,
    v_rad_tolerance: Optional[float] = 0,
    initial_labels: Optional[float] = None,
    continuum_method: str = "astra.specutils.continuum.Chebyshev",
    continuum_kwargs: dict = dict(
        deg=4,
        regions=[(15_100.0, 15_793.0), (15_880.0, 16_417.0), (16_499.0, 17_000.0)],
        mask="$MWM_ASTRA/pipelines/ThePayne/cannon_apogee_pixels.npy",
    ),
) -> Iterable[ThePayne]:

    model = read_model(model_path)
    mask = read_mask(mask_path)
    elems_mask = read_elem_mask_all(dir_elem_mask)
    
    args = [
        model[k]
        for k in (
            "weights",
            "biases",
            "x_min",
            "x_max",
            "wavelength",
            "label_names",
        )
    ]

    for spectrum in flatten(spectra):
        # the continuum executables will still be available, but i think they need a little updating for ipl-3
        if continuum_method is not None:
            f_continuum = executable(continuum_method)(**continuum_kwargs) 
            f_continuum.fit(spectrum)
            continuum = f_continuum(spectrum)
            continuum = np.atleast_2d(continuum)
        else:
            continuum = None

        #print('*'*6, 'starting estimate_labels', '*'*6)
        result = estimate_labels( 
            spectrum,
            *args,
            mask=mask,
            elems_mask =elems_mask,
            initial_labels=initial_labels,
            v_rad_tolerance=v_rad_tolerance,
            opt_tolerance=opt_tolerance,
            continuum=continuum,
        )
        #print('*'*6, 'finished estimate_labels', '*'*6)
        # if `result` is an instance of `ThePayne(...)` then you can just yield it here

        # create a new instance of `ThePayne` and set the attributes of the new instance
        output = ThePayne()
        for kwrds in result:
            setattr(output, kwrds, result[kwrds])

        yield output
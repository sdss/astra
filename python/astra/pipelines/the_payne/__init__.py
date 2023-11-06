"""The Payne"""

import numpy as np
import os
import pickle
from collections import OrderedDict
from typing import Iterable, Optional

from astra import task
from astra.utils import log, executable, expand_path
from astra.pipelines.the_payne.model import estimate_labels
from astra.pipelines.the_payne.utils import read_mask, read_model

from astra.models.the_payne import ThePayne
from peewee import ModelSelect

@task
def the_payne(
    spectra,
    model_path: str = "$MWM_ASTRA/pipelines/ThePayne/payne_apogee_nn.pkl", 
    mask_path: str = "$MWM_ASTRA/pipelines/ThePayne/payne_apogee_mask.npy", 
    opt_tolerance: Optional[float] = 5e-4,
    v_rad_tolerance: Optional[float] = 0,
    initial_labels: Optional[float] = None,
    continuum_method: str = "astra.specutils.continuum.Chebyshev",
    continuum_kwargs: dict = dict(
        deg=4,
        regions=[(15_100.0, 15_793.0), (15_880.0, 16_417.0), (16_499.0, 17_000.0)],
        mask="$MWM_ASTRA/pipelines/ThePayne/cannon_apogee_pixels.npy",
    ),
    page=None,
    limit=None
) -> Iterable[ThePayne]:

    if isinstance(spectra, ModelSelect):
        if page is not None and limit is not None:
            spectra = spectra.paginate(page, limit)
        elif limit is not None:
            spectra = spectra.limit(limit)        

    model = read_model(model_path)
    mask = read_mask(mask_path)
    
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

    for spectrum in spectra:
        try:
            if continuum_method is not None:
                f_continuum = executable(continuum_method)(**continuum_kwargs)
                continuum = np.atleast_2d(f_continuum.fit(spectrum))
            else:
                continuum = None            

            # With SpectrumList, we should only ever have 1 spectrum
            (result, ), (meta, ) = estimate_labels(
                spectrum,
                *args,
                mask=mask,
                initial_labels=initial_labels,
                v_rad_tolerance=v_rad_tolerance,
                opt_tolerance=opt_tolerance,
                continuum=continuum,
            )

            output = ThePayne(
                spectrum_pk=spectrum.spectrum_pk,
                source_pk=spectrum.source_pk,
                **result
            )

            path = expand_path(output.intermediate_output_path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as fp:
                pickle.dump((meta["continuum"], meta["rectified_model_flux"]), fp)

            yield output
        
        except:
            log.exception(f"Exception when fitting spectrum {spectrum}")
            yield ThePayne(
                spectrum_pk=spectrum.spectrum_pk,
                source_pk=spectrum.source_pk,
                flag_fitting_failure=True
            )

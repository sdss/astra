import os
import matplotlib.pyplot as plt
from typing import Iterable
from astra import task, __version__

from astra.utils import log, expand_path
from astra.models.corv import Corv

__all__ = ["corv"]

@task
def corv(spectra: Iterable) -> Iterable[Corv]:
    """
    Fit the radial velocity and stellar parameters for white dwarfs.

    :param spectra:
        An iterable of DA-type white dwarf spectra.
    """

    from astra.pipelines.corv import models, fit, utils

    corv_model = models.make_koester_model()

    for spectrum in spectra:

        args = (spectrum.wavelength, spectrum.flux, spectrum.ivar, corv_model)
        v_rad, e_v_rad, rchi2, result = fit.fit_corv(*args)
        
        try:
            fig = utils.lineplot(*args, result.params)
            path = expand_path(f"$MWM_ASTRA/{__version__}/pipelines/corv/{spectrum.spectrum_id}-{__version__}.png")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fig.savefig(path)
            plt.close("all")
            del fig
        except:
            log.exception(f"Exception trying to make figure for {spectrum}")

        yield Corv(
            source_id=spectrum.source_id,
            spectrum_id=spectrum.spectrum_id,
            v_rad=v_rad,
            e_v_rad=e_v_rad,
            teff=result.params["teff"].value,
            e_teff=result.params["teff"].stderr,
            logg=result.params["logg"].value,
            e_logg=result.params["logg"].stderr,
            rchi2=rchi2,            
            initial_teff=result.init_values["teff"],
            initial_logg=result.init_values["logg"],
            initial_v_rad=result.init_values["RV"],
        )


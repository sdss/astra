import concurrent.futures
import os
import matplotlib.pyplot as plt
from peewee import JOIN
from typing import Iterable, Optional
from tqdm import tqdm
from astra import task, __version__
from astra.models import BossVisitSpectrum, Corv, SnowWhite
from astra.utils import log, expand_path

from astra.pipelines.corv import models, fit, utils


__all__ = ["corv"]

@task
def corv(
    spectra: Optional[Iterable[BossVisitSpectrum]] = (
        BossVisitSpectrum
        .select()
        .join(SnowWhite, on=(BossVisitSpectrum.spectrum_pk == SnowWhite.spectrum_pk))
        .switch(BossVisitSpectrum)
        .join(Corv, JOIN.LEFT_OUTER, on=(BossVisitSpectrum.spectrum_pk == Corv.spectrum_pk))
        .where(
            Corv.spectrum_pk.is_null()
        &   (SnowWhite.classification == "DA")
        )
    ),
    max_workers: Optional[int] = 4
) -> Iterable[Corv]:
    """
    Fit the radial velocity and stellar parameters for white dwarfs.

    :param spectra:
        An iterable of DA-type white dwarf spectra.
    """

    corv_model = models.make_koester_model()
    
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

    futures = [executor.submit(_corv, s, corv_model) for s in spectra]
    
    with tqdm(total=len(futures)) as pb:
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            yield result
            pb.update()


def _corv(spectrum, corv_model):
    
    args = (spectrum.wavelength, spectrum.flux, spectrum.ivar, corv_model)
    try:
        v_rad, e_v_rad, rchi2, result = fit.fit_corv(*args)
    except:
        log.exception(f"Exception when running corv on {spectrum}")
        return None
    
    try:
        fig = utils.lineplot(*args, result.params)
        path = expand_path(f"$MWM_ASTRA/{__version__}/pipelines/corv/{spectrum.spectrum_pk}-{__version__}.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close("all")
        del fig
    except:
        log.exception(f"Exception trying to make figure for {spectrum}")

    return Corv(
        source_pk=spectrum.source_pk,
        spectrum_pk=spectrum.spectrum_pk,
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
import concurrent.futures
import os
import matplotlib.pyplot as plt
from peewee import JOIN
from typing import Iterable, Optional
from tqdm import tqdm
from astra import task, __version__
from astra.models import BossVisitSpectrum, Corv, SnowWhite
from astra.models.mwm import BossCombinedSpectrum
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
        .join(
            Corv,
            JOIN.LEFT_OUTER, 
            on=(
                (BossVisitSpectrum.spectrum_pk == Corv.spectrum_pk)
            &   (Corv.v_astra == __version__)
            )
        )
        .where(
            Corv.spectrum_pk.is_null()
        &   (SnowWhite.classification == "DA")
        &   (BossVisitSpectrum.run2d == "v6_1_3")
        &   
        )
    ),
    max_workers: Optional[int] = 4
) -> Iterable[Corv]:
    """
    Fit the radial velocity and stellar parameters for white dwarfs.

    :param spectra:
        An iterable of DA-type white dwarf spectra.
    """

    model = models.make_montreal_da_model()

    executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

    futures = [executor.submit(_corv, s, model) for s in spectra]
    
    with tqdm(total=len(futures)) as pb:
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            yield result
            pb.update()


def _corv(spectrum, corv_model):
    
    args = (spectrum.wavelength, spectrum.flux, spectrum.ivar, corv_model)
    try:
        v_rad, e_v_rad, rchi2, result = fit.fit_corv(*args, iter_teff=True)
    except:
        log.exception(f"Exception when running corv on {spectrum}")
        return None
    
    try:
        fig = utils.lineplot(*args, result.params)
        folders = f"{str(spectrum.spectrum_pk)[-4:-2]}/{str(spectrum.spectrum_pk)[-2:]}"
        path = expand_path(f"$MWM_ASTRA/{__version__}/pipelines/corv/{folders}/{spectrum.spectrum_pk}.png")
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



def test_corv_rvs():
    import numpy as np
    from astra.models import Source
    from astropy.table import Table
    df = Table.read(expand_path("$MWM_ASTRA/pipelines/corv/astra_comparison_df.csv"))
    
    model = models.make_montreal_da_model()

    x = df["RV_corv_nb"]
    x_err = df["E_RV_corv_nb"]

    y = []
    y_err = []
    for i, row in enumerate(df):
        #
        try:
            s = BossVisitSpectrum.get(
                run2d="v6_1_1",
                mjd=row["mjd"],
                fieldid=row["fieldid"],
                plateid=row["plateid"],
                cartid=row["cartid"],
                source_pk=Source.get(sdss_id=row["sdss_id"]).pk
            )
            args = (s.wavelength, s.flux, s.ivar, model)
            v_rad, e_v_rad, rchi2, result = fit.fit_corv(*args, iter_teff=True)
        except:
            y.append(np.nan)
            y_err.append(np.nan)
            raise
        
        else:
            y.append(v_rad)
            y_err.append(e_v_rad)
        print(f"{x[i]:.2f} {y[i]:.2f} {x_err[i]:.2f} {y_err[i]:.2f}")

    y = np.array(y)
    y_err = np.array(y_err)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    lim = np.array([ax.get_xlim(), ax.get_ylim()])
    lim = (np.min(lim), np.max(lim))
    ax.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='o', c="#666666")
    ax.plot(lim, lim, c="#666666", ls=":", zorder=-1, lw=0.5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    fig.savefig(expand_path("$MWM_ASTRA/pipelines/corv/compare.png"))

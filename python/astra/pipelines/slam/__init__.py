__slam_version__ = "1.2019.0109.4"
from .slam3 import Slam3 as SlamCode

from .laspec.convolution import conv_spec, fwhm2resolution
from .laspec.qconv import conv_spec_Gaussian
from .laspec.normalization import normalize_spectrum, normalize_spectra_block
from .laspec.binning import rebin

import numpy as np
from tqdm import tqdm
from joblib import load
from peewee import JOIN, fn, ModelSelect
from astra import task, __version__, log
from astra.utils import expand_path
from astra.models import BossCombinedSpectrum
from astra.models.slam import Slam
from astropy.table import Table
from typing import Iterable, Optional
from astra.models import SpectrumMixin, Source
#  
#  According to the Bible, we’re roughly dealing with an absolute magnitude range M_G in [7.57, 13.35] for M dwarfs between 4000 and 3000 K. It might be worth including this cut when training/running the SLAM
@task
def slam(
    spectra: Optional[Iterable[SpectrumMixin]] = (
        BossCombinedSpectrum
        .select()
        .join(Source)
        .switch(BossCombinedSpectrum)
        .join(
            Slam, 
            JOIN.LEFT_OUTER, 
            on=(
                (Slam.spectrum_pk == BossCombinedSpectrum.spectrum_pk)
            &   (Slam.v_astra == __version__)
            )
        )
        .where(
            Slam.spectrum_pk.is_null()
        &   (                
                (
                    # From Zach Way, mwm-astra 413
                    Source.g_mag.is_null(False)
                &   Source.rp_mag.is_null(False)
                &   Source.plx.is_null(False)
                &   (Source.plx > 0)
                &   ((Source.g_mag - Source.rp_mag) > 0.56)
                &   ((Source.g_mag + 5 + 5 * fn.log10(Source.plx/1000)) > 5.553)
                #&   ((Source.g_mag + 5 + 5 * fn.log10(Source.plx/1000)) > 7.57)
                )
            |   (
                Source.assigned_to_program("mwm_yso")
            |   Source.assigned_to_program("mwm_snc")
            )
        )
        &   (BossCombinedSpectrum.v_astra == __version__)
        )
    ), 
    page=None,
    limit=None,
    n_jobs=128
) -> Slam:

    if isinstance(spectra, ModelSelect) and limit is not None:
        if page is not None:
            print("paging")
            spectra = spectra.paginate(page, limit)
        else:
            print("limiting")
            spectra = spectra.limit(limit)

    wave_interp = Table.read(expand_path("$MWM_ASTRA/pipelines/slam/dM_train_wave_standard.csv"))['wave']
    dump_path = expand_path("$MWM_ASTRA/pipelines/slam/Train_FGK_LAMOST_M_BOSS_alpha_from_ASPCAP_teff_logg_from_ApogeeNet_nobinaries.dump")

    import sys
    sys.path.insert(0, "/uufs/chpc.utah.edu/common/home/u6020307/astra/python/astra/pipelines/")
    Pre = load(dump_path)

    flux_boss = []
    ivar_boss = []
    used_spectra = []
    for spectrum in tqdm(spectra, desc="Collecting"):
        # only redshift if it is a visit spectrum (not a stack)
        try:
            if isinstance(spectrum, BossCombinedSpectrum):
                wave = spectrum.wavelength
            else:
                wave = spectrum.wavelength / (1 + spectrum.xcsao_v_rad / 299792.458)
            flux_n, invar_n = rebin(
                wave, 
                flux=spectrum.flux, 
                flux_err=spectrum.e_flux,
                wave_new=wave_interp
            )
            flux_boss.append(flux_n)
            ivar_boss.append(invar_n)
        except Exception:
            log.exception(f"Failed to rebin spectrum {spectrum.spectrum_pk}")
        else:
            used_spectra.append(spectrum)

    flux_boss = np.array(flux_boss)
    ivar_boss = np.array(ivar_boss)

    print("normalizing spectra block")
    flux_norm, flux_cont = normalize_spectra_block(
        wave_interp, flux_boss, 
        (6001.755, 8957.321), 
        10., 
        p=(1E-8, 1E-7), 
        #ivar_block=ivar_boss, # Dan doesn't use this, but there is an option for it.
        q=0.7, eps=1E-19, rsv_frac=2., n_jobs=n_jobs, verbose=5
    )
    flux_norm[flux_norm>2.] = 1
    flux_norm[flux_norm<0] = 0
    ivars_norm = (ivar_boss * flux_cont**2)

    log.info("Predicting labels (first pass)")
    Xinit = Pre.predict_labels_quick(flux_norm, ivars_norm, n_jobs=n_jobs, verbose=5)

    SEP_1_LENGTH = len(flux_norm)

    log.info("Predicting labels")
    Rpred = Pre.predict_labels_multi(Xinit[0:SEP_1_LENGTH], flux_norm[0:SEP_1_LENGTH], ivars_norm[0:SEP_1_LENGTH],n_jobs=n_jobs,verbose=5)
    Xpred = np.array([_["x"] for _ in Rpred])
    Xpred_err = np.array([np.diag(_["pcov"]) for _ in Rpred])

    #feh_niu,feh,alpha_m,teff,logg=(Xpred[:,_] for _ in range(5))
    #feh_niu_err,feh_err,alpha_m_err,teff_err,logg_err=(Xpred_err[:,_] for _ in range(5))

    for i, spectrum in enumerate(used_spectra):
        kwds = dict(source_pk=spectrum.source_pk, spectrum_pk=spectrum.spectrum_pk)
        kwds.update(
            fe_h_niu=Xpred[i, 0],
            e_fe_h_niu=Xpred_err[i, 0],
            fe_h=Xpred[i, 1],
            e_fe_h=Xpred_err[i, 1],
            alpha_fe=Xpred[i, 2],
            e_alpha_fe=Xpred_err[i, 2],
            teff=Xpred[i, 3],
            e_teff=Xpred_err[i, 3],
            logg=Xpred[i, 4],
            e_logg=Xpred_err[i, 4],

            initial_fe_h_niu=Xinit[i, 0],
            initial_fe_h=Xinit[i, 1],
            initial_alpha_fe=Xinit[i, 2],
            initial_teff=Xinit[i, 3],
            initial_logg=Xinit[i, 4],
            status=Rpred[i]["status"],
            success=Rpred[i]["success"],
            optimality=Rpred[i]["optimality"],
            chi2=np.nan,
            rchi2=np.nan
        )
        kwds.update(
            flag_teff_outside_bounds=(kwds["teff"] < 2800) or (kwds["teff"] > 4500),
            flag_fe_h_outside_bounds=(kwds["fe_h"] < -1) or (kwds["fe_h"] > 0.5),
            flag_bad_optimizer_status=(kwds["status"] > 0 and kwds["status"] != 2) | (kwds["status"] < 0),
        )
        yield Slam(**kwds)



if __name__ == "__main__":

    from astra.models import Source, ASPCAP, BossVisitSpectrum, Slam, BossCombinedSpectrum
    '''
    spectra = list(
        BossVisitSpectrum
        .select()
        .join(Source)
        .join(ASPCAP)
        .switch(BossVisitSpectrum)
        .join(Slam, on=(BossVisitSpectrum.source_pk == Slam.source_pk))
        .where(
            (BossVisitSpectrum.run2d == "v6_1_3")
        &   (ASPCAP.flag_as_m_dwarf_for_calibration)
        &   (ASPCAP.v_astra == "0.6.0")
        &   (Slam.v_astra == "0.6.0")
        )

        .limit(10)
    )

    spectra = list(
        BossCombinedSpectrum
        .select()
        .join(Slam, on=(BossCombinedSpectrum.spectrum_pk == Slam.spectrum_pk))
        .where(Slam.v_astra == "0.6.0")
        .limit(100)
    )
    '''
    from astropy.table import Table
    from astra.utils import expand_path
    t = Table.read(expand_path("$MWM_ASTRA/pipelines/slam/SLAM_test_astra_0.6.0.fits"))

    spectra = list(
        BossCombinedSpectrum
        .select()
        .where(BossCombinedSpectrum.spectrum_pk.in_(list(t["spectrum_pk"])))
    )


    from astra.pipelines.slam import slam
    results = list(slam(spectra))
    
    # match up to the rows in the table
    import numpy as np
    spectrum_pks = np.array([r.spectrum_pk for r in results])

    t.sort("spectrum_pk")
    t_spectrum_pks = np.array(t["spectrum_pk"])

    indices = np.array([t_spectrum_pks.searchsorted(s) for s in spectrum_pks])
    t = t[indices]

    assert np.all(np.array(t["spectrum_pk"]) == spectrum_pks)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    label_names = [
        ("teff_pre", "teff", "initial_teff"),
        ("logg_pre", "logg", "initial_logg"),
        ("feh_niu_pre", "fe_h_niu", "initial_fe_h_niu"),
        ("alph_m_pre", "alpha_fe", "initial_alpha_fe")
    ]
    vrad = [getattr(r, "xcsao_v_rad") for r in spectra]

    for i, ax in enumerate(axes.flat):
        xlabel, ylabel, zlabel = label_names[i]
        x = t[xlabel]
        y = [getattr(r, ylabel) for r in results]
        z = [getattr(r, zlabel) for r in results]

        scat = ax.scatter(x, y, c=vrad)
        plt.colorbar(scat, ax=ax)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        limits = np.array([ax.get_xlim(), ax.get_ylim()])
        limits = (np.min(limits), np.max(limits))
        ax.plot(limits, limits, c="k", ls="--")
        ax.set_xlim(limits)
        ax.set_ylim(limits)
    


    raise a
    

    before = list(Slam.select().where(Slam.v_astra == "0.6.0").where(Slam.spectrum_pk.in_([s.spectrum_pk for s in spectra])))
    after = list(slam(spectra))

    index = np.argsort([s.spectrum_pk for s in spectra])
    before_sorted = [before[i] for i in index]




    

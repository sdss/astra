
import numpy as np
import os
from typing import Iterable, Optional
import concurrent.futures
from astra.models import Source, BossVisitSpectrum, ApogeeCoaddedSpectrumInApStar
from astra.models.nmf_rectify import NMFRectify
from astra.specutils.continuum.nmf.apogee import ApogeeNMFContinuum
from astra.specutils.continuum.nmf.boss import BossNMFContinuum
from astra import task, __version__
from astra.utils import log, expand_path
from peewee import chunked, JOIN, fn, ModelSelect
from tqdm import tqdm
from astra.products.utils import dispersion_array

ApogeeNMFRectification = None


@task
def rectify_boss_spectra_by_source(
    sources: Optional[Iterable[Source]] = (
        Source
        .select()
        .distinct(Source.pk)
        .join(NMFRectify, JOIN.LEFT_OUTER, on=(NMFRectify.source_pk == Source.pk))
        .where(
            NMFRectify.source_pk.is_null()
        &   (Source.n_boss_visits > 0)
        )
    ),
    page: Optional[int] = None,
    limit: Optional[int] = None        
) -> Iterable[NMFRectify]:

    # TODO: Should consider this logic to be executed somewhere else, either in the astra CLI call, or in the task wrapper, etc
    if isinstance(sources, ModelSelect):
        if page is not None and limit is not None:
            sources = sources.paginate(page, limit)
        elif limit is not None:
            sources = sources.limit(limit)    
    
    model = BossNMFContinuum()    
    for source in tqdm(sources, total=0, desc="Rectifying"):
        yield from _rectify_boss_spectra_by_source([source], model)
        
    '''
    futures = []
    for chunk in tqdm(chunked(sources, batch_size), total=0, desc="Chunking"):
        futures.append(executor.submit(_rectify_boss_spectra_by_source, chunk, model))
    
    with tqdm(total=len(futures) * batch_size, desc="Rectifying") as pb:
        for future in concurrent.futures.as_completed(futures):
            yield from future.result()
            pb.update(batch_size)
    '''


@task
def rectify_apogee_coadded_spectra_in_apstar(
    spectra: Optional[Iterable[ApogeeCoaddedSpectrumInApStar]] = (
        ApogeeCoaddedSpectrumInApStar
        .select()
        .join(NMFRectify, JOIN.LEFT_OUTER, on=(ApogeeCoaddedSpectrumInApStar.spectrum_pk == NMFRectify.spectrum_pk))
        .where(NMFRectify.spectrum_pk.is_null())
    ),
    page: Optional[int] = None,
    limit: Optional[int] = None    
) -> Iterable[NMFRectify]:
    
    # TODO: Should consider this logic to be executed somewhere else, either in the astra CLI call, or in the task wrapper, etc
    if isinstance(spectra, ModelSelect):
        if page is not None and limit is not None:
            spectra = spectra.paginate(page, limit)
        elif limit is not None:
            spectra = spectra.limit(limit)
            
    model = ApogeeNMFContinuum()
        
    initial_flags = [
        ("flag_initialised_from_small_w", model.get_initial_guess_with_small_W),
        #("flag_initialised_from_llsqb", model.get_initial_guess_by_linear_least_squares_with_bounds),
    ]
    results = []
    for spectrum in tqdm(spectra, total=0):
        
        kwds = dict(
            spectrum_pk=spectrum.spectrum_pk,
            source_pk=spectrum.source_pk,
            L=model.L,
            deg=model.deg,
            log_W=[],
            continuum_theta=[],
        )
        
        try:
            args = list(map(np.atleast_2d, (spectrum.flux, spectrum.ivar)))
        except:
            yield NMFRectify(flag_could_not_read_spectrum=True, **kwds)
        else:
            
            for flag_name, f in initial_flags:        
                try:
                    x0 = f(*args)        
                    continuum, result = model.fit(*args, x0=x0, full_output=True)                
                except:
                    log.exception(f"Exception fitting {flag_name} x0 with spectrum {spectrum}")
                    continue
                else:
                    break
            else:
                yield NMFRectify(flag_runtime_exception=True, **kwds)

            
            dof = result["W"].size + result["theta"].size
            pixel_chi2 = result["pixel_chi2"].reshape((1, -1))
            thetas = result["theta"].reshape((1, -1))
            rchi2s = np.nansum(pixel_chi2, axis=1) / (np.sum(np.isfinite(pixel_chi2), axis=1) - dof - 1)
            
            kwds.update(
                log10_W=np.log10(result["W"]),
                L=model.L,
                deg=model.deg,
                joint_rchi2=result["rchi2"],
                continuum_theta=thetas[0],
                rchi2=rchi2s[0]
            )
            kwds[flag_name] = True
            yield NMFRectify(**kwds)



boss_resample_wavelength = dispersion_array("boss")   
apogee_resample_wavelength = dispersion_array("apogee")
boss_rest_frame_wavelength = lambda s, m: s.wavelength[m] * (1 - s.xcsao_v_rad/2.99792458e5)


def plot_apogee_rectified_spectra():
    q = (
        NMFRectify
        .select(
            NMFRectify,
            ApogeeCoaddedSpectrumInApStar
        )
        .join(ApogeeCoaddedSpectrumInApStar, on=(ApogeeCoaddedSpectrumInApStar.spectrum_pk == NMFRectify.spectrum_pk), attr="spectrum")
        .iterator()
    )
    model = ApogeeNMFContinuum()    
        
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 2))
    data_plot, = ax.plot([], [], c='k')
    model_plot, = ax.plot(apogee_resample_wavelength, np.nan * np.ones_like(apogee_resample_wavelength), c="tab:red")
    ax.axhline(1, c="#666666", ls=":", lw=0.5, zorder=-1)
    ax.set_xlim(apogee_resample_wavelength[[0, -1]])
    ax.set_ylim(0, 1.2)
    ax.set_xlabel(r"Wavelength [A]")
    ax.set_ylabel(r"Rectified flux")
    ax.set_title("spectrum_pk=0 rchi2=0")
    fig.tight_layout()
    
    for result in tqdm(q, total=1):
        
        if result.log10_W is None:
            continue
        
        group_dir = f"{result.spectrum_pk}"[-2:]
        output_path = expand_path(f"$MWM_ASTRA/{__version__}/spectra/rectified-plots/apogee/{group_dir}/{result.spectrum_pk}.png")
        if os.path.exists(output_path):
            continue        
        
        rest_wavelength = result.spectrum.wavelength
        rectified_model_flux = 1 - (10**np.array(result.log10_W)) @ model.components
        continuum = model.continuum(rest_wavelength, result.continuum_theta)[0]
        data_plot.set_data(
            result.spectrum.wavelength,
            result.spectrum.flux / continuum,
        )
        #ax.plot(boss_resample_wavelength, rectified_model_flux, c="tab:red")
        model_plot.set_ydata(rectified_model_flux)
        ax.set_title(f"spectrum_pk={result.spectrum_pk}, rchi2={result.rchi2:.2f}")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300)
    

def plot_boss_rectified_spectra():
    q = (
        NMFRectify
        .select(
            NMFRectify,
            BossVisitSpectrum
        )
        .join(BossVisitSpectrum, on=(BossVisitSpectrum.spectrum_pk == NMFRectify.spectrum_pk), attr="spectrum")
        .iterator()
    )
    model = BossNMFContinuum()    
        
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 2))
    data_plot, = ax.plot([], [], c='k')
    model_plot, = ax.plot(boss_resample_wavelength, np.nan * np.ones_like(boss_resample_wavelength), c="tab:red")
    ax.axhline(1, c="#666666", ls=":", lw=0.5, zorder=-1)
    ax.set_xlim(boss_resample_wavelength[[0, -1]])
    ax.set_ylim(0, 1.2)
    ax.set_xlabel(r"Wavelength [A]")
    ax.set_ylabel(r"Rectified flux")
    ax.set_title("spectrum_pk=0 rchi2=0")
    fig.tight_layout()
    
    for result in tqdm(q, total=1):

        if result.log10_W is None:
            continue
                
        rest_wavelength = boss_rest_frame_wavelength(result.spectrum, ...)
        rectified_model_flux = 1 - (10**np.array(result.log10_W)) @ model.components
        continuum = model.continuum(rest_wavelength, result.continuum_theta)[0]
        data_plot.set_data(
            result.spectrum.wavelength,
            result.spectrum.flux / continuum,
        )
        #ax.plot(boss_resample_wavelength, rectified_model_flux, c="tab:red")
        model_plot.set_ydata(rectified_model_flux)
        ax.set_title(f"spectrum_pk={result.spectrum_pk}, rchi2={result.rchi2:.2f}")
        
        groups = f"{result.spectrum_pk}"[-4:]
        groups = f"{groups[:2]}/{groups[2:]}"
        output_path = expand_path(f"$MWM_ASTRA/{__version__}/spectra/rectified-plots/boss/{groups}/{result.spectrum_pk}.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300)
    


def _rectify_boss_spectra_by_source(sources, model):
    
    P = boss_resample_wavelength.size
    initial_flags = [
        ("flag_initialised_from_small_w", model.get_initial_guess_with_small_W),
        #("flag_initialised_from_llsqb", model.get_initial_guess_by_linear_least_squares_with_bounds),
    ]
    results = []
    for source in sources:
        boss_callable_in_coadd = lambda s: (
                (s.snr is not None)
            &   np.isfinite(s.snr)
            &   (s.snr > 3)
            &   ((s.xcsao_rxc > 6) | ("mwm_wd" in source.sdss5_cartons["program"]))
            &   (s.zwarning_flags <= 0)
        )
        spectra = list(filter(boss_callable_in_coadd, source.boss_visit_spectra))
        
        N = len(spectra)
        if N == 0:
            continue
        visit_flux, visit_ivar = (np.zeros((N, P)), np.zeros((N, P)))
        for i, spectrum in enumerate(spectra):
            rest_wavelength = boss_rest_frame_wavelength(spectrum, ...)
            visit_flux[i] = np.interp(
                boss_resample_wavelength,
                rest_wavelength,
                spectrum.flux, 
                left=np.nan, 
                right=np.nan
            )
            visit_ivar[i] = np.interp(
                boss_resample_wavelength,
                rest_wavelength,
                spectrum.ivar,
                left=0,
                right=0
            )
        
        for flag_name, f in initial_flags:        
            try:                            
                x0 = f(visit_flux, visit_ivar)

                continuum, result = model.fit(
                    visit_flux, 
                    visit_ivar, 
                    x0=x0,
                    full_output=True
                )
                
            except:
                log.exception(f"Exception fitting {flag_name} x0 with source {source}")
                continue
            
            else:
                dof = result["W"].size + result["theta"].size / N
                pixel_chi2 = result["pixel_chi2"].reshape((N, -1))
                thetas = result["theta"].reshape((N, -1))
                rchi2s = np.nansum(pixel_chi2, axis=1) / (np.sum(np.isfinite(pixel_chi2), axis=1) - dof - 1)
                
                common_kwds = dict(
                    log10_W=np.log10(result["W"]),
                    L=model.L,
                    deg=model.deg,
                    joint_rchi2=result["rchi2"]
                )
                common_kwds[flag_name] = True
                
                for i, (spectrum, theta, rchi2) in enumerate(zip(spectra, thetas, rchi2s)):
                    results.append(
                        NMFRectify(
                            source_pk=spectrum.source_pk,
                            spectrum_pk=spectrum.spectrum_pk,
                            continuum_theta=theta,
                            rchi2=rchi2,
                            **common_kwds
                        )
                    )        
                    
    return results
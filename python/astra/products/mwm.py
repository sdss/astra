"""Functions to create mwmVisit and mwmStar products."""

import os
import pickle
import numpy as np
import concurrent.futures
from peewee import JOIN, fn
from tqdm import tqdm
from astra import task
from astra.utils import log, expand_path
from astra.models.source import Source
from astra.models.boss import BossVisitSpectrum, BossCoaddedSpectrum
from astra.migrations.utils import enumerate_new_spectrum_pks
from astra.products.utils import dispersion_array
from astra.specutils.ndi import resample_spectrum, construct_design_matrix, _un_mask
from astra.specutils.continuum.nmf.boss import BossNMFContinuum
from typing import Optional, Iterable

def _prepare_spectra_for_resampling(spectra, callable_in_coadd, callable_rest_frame_wavelength, callable_pixel_in_coadd_mask=None):    
    
    wavelength, flux, ivar, pixel_flags, pixel_in_coadd, spectrum_in_coadd, spectrum_index = ([], [], [], [], [], [], [])
    for i, spectrum in enumerate(spectra):
        
        in_coadd = callable_in_coadd(spectrum)
                
        if callable_pixel_in_coadd_mask is not None:
            spectrum_mask = callable_pixel_in_coadd_mask(spectrum)
        else:
            spectrum_mask = np.ones_like(spectrum.wavelength, dtype=bool)
                    
        P = np.sum(spectrum_mask)
        
        wavelength.extend(callable_rest_frame_wavelength(spectrum, spectrum_mask))
        flux.extend(spectrum.flux[spectrum_mask])
        ivar.extend(spectrum.ivar[spectrum_mask])
        pixel_flags.extend(spectrum.pixel_flags[spectrum_mask])
        pixel_in_coadd.extend([in_coadd] * P)
        spectrum_index.extend([i] * P)
        spectrum_in_coadd.append(in_coadd)   
        
    return tuple(
        map(
            np.array, 
            (wavelength, flux, ivar, pixel_flags, pixel_in_coadd, spectrum_in_coadd, spectrum_index)
        )    
    )

            
@task
def coadd_and_rectify_boss_spectrum(
    sources: Optional[Iterable[Source]] = (
        Source
        .select()
        .join(BossVisitSpectrum)
        .switch(Source)
        .join(BossCoaddedSpectrum, JOIN.LEFT_OUTER, on=(BossCoaddedSpectrum.source_pk == Source.pk))
        .where(
            (Source.n_boss_visits > 1)
        &   BossVisitSpectrum.snr.is_null(False) 
        &   (BossVisitSpectrum.snr > 3) 
        &   ((BossVisitSpectrum.xcsao_rxc > 6) | (Source.assigned_to_program("mwm_wd")))
        &   (BossVisitSpectrum.zwarning_flags <= 0)
        &   (BossCoaddedSpectrum.source_pk.is_null())
        )
        .order_by(fn.Random())
    ),
    rcond=1e-2,
    Lambda=0,
    min_resampled_flag_value=0.1,
    max_workers=1,
) -> Iterable[BossCoaddedSpectrum]:

    nmf_model = BossNMFContinuum()

    resample_wavelength = dispersion_array("boss")        
    region_mask = (10_000 >= resample_wavelength) * (resample_wavelength >= 3700)
    x_star = resample_wavelength[region_mask]
    L = np.ptp(x_star)
    P = x_star.size
    X_star = construct_design_matrix(x_star, L, P)
    
    resample_kwds = dict(
        resample_wavelength=resample_wavelength[region_mask],
        X_star=X_star,
        rcond=rcond,
        Lambda=Lambda,
        min_resampled_flag_value=min_resampled_flag_value,
        L=L,
        P=P,        
    )
    
    
    if max_workers > 1:            
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
        
        futures = []
        #for spectrum_pk, source in enumerate_new_spectrum_pks(tqdm(sources)):
        #    futures.append(executor.submit(_coadd_and_rectify_boss_spectrum, source, spectrum_pk, nmf_model, region_mask, resample_kwds))

        with tqdm(total=len(futures), unit="source") as pb:
            for future in concurrent.futures.as_completed(futures):
                coadd = future.result()
                if coadd is not None:
                    yield coadd
                pb.update(1)
                
    else:
        with tqdm(total=0, unit="source") as pb:
            for source in sources:
                result = _coadd_and_rectify_boss_spectrum(
                    source,
                    nmf_model,
                    region_mask,
                    resample_kwds
                )
                print(f'{source} has {result} ({type(result)})')
                if result is not None:
                    yield result
                    print(result.__data__)
                pb.update(1)
        

def _coadd_and_rectify_apogee_spectrum(source, telescope, spectrum_pk, nmf_model, region_mask, resample_kwds):
    
    resample_wavelength = dispersion_array("apogee")        
    apogee_rest_frame_wavelength = lambda s, m: s.wavelength[m] * (1 - s.v_rad/2.99792458e5) # TODO: include barycentric 
    apogee_callable_pixel_in_coadd_mask = lambda s: np.ones(s.wavelength.shape, dtype=bool)
    
    # Only get spectra that are in the apStar file
    
    (wavelength, flux, ivar, pixel_flags, pixel_in_coadd, spectrum_in_coadd, spectrum_index) = _prepare_spectra_for_resampling(
        spectra=spectra, # TODO: where clause on telescope,
        callable_in_coadd=boss_callable_in_coadd,
        callable_rest_frame_wavelength=boss_rest_frame_wavelength,
        callable_pixel_in_coadd_mask=boss_callable_pixel_in_coadd_mask, 
    )        
    

def _coadd_and_rectify_boss_spectrum(source, nmf_model, region_mask, resample_kwds):
    
    boss_rest_frame_wavelength = lambda s, m: s.wavelength[m] * (1 - s.xcsao_v_rad/2.99792458e5)
    boss_callable_pixel_in_coadd_mask = lambda s: (10_000 >= s.wavelength) * (s.wavelength >= 3_700) # To avoid ringing artefacts
    
    try:
        coadd = BossCoaddedSpectrum.get(sdss_id=source.sdss_id)
    except:
        None
    else:
        log.info(f"Source {source} already has a coadd: {coadd}")
        return None
    
    path = expand_path(BossCoaddedSpectrum(sdss_id=source.sdss_id, telescope="apo25m").path)
    if os.path.exists(path):
        print(f"Source {source} already has path: {path}")
        return None
    
    #os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        boss_callable_in_coadd = lambda s: (
                (s.snr is not None)
            &   np.isfinite(s.snr)
            &   (s.snr > 3)
            &   ((s.xcsao_rxc > 6) | ("mwm_wd" in source.sdss5_cartons["program"]))
            &   (s.zwarning_flags <= 0)
        )
                
        spectra = source.boss_visit_spectra
        
        (wavelength, flux, ivar, pixel_flags, pixel_in_coadd, spectrum_in_coadd, spectrum_index) = _prepare_spectra_for_resampling(
            spectra=spectra, # TODO: where clause on telescope,
            callable_in_coadd=boss_callable_in_coadd,
            callable_rest_frame_wavelength=boss_rest_frame_wavelength,
            callable_pixel_in_coadd_mask=boss_callable_pixel_in_coadd_mask, 
        )
    
        if not np.any(pixel_in_coadd):
            log.warning(f"No suitable BOSS spectra to combine for source {source}")
            return None
            
        coadd_flux, coadd_ivar, coadd_pixel_flags, coadd_separate_pixel_flags, X_star, L, P = resample_spectrum(
            wavelength=wavelength[pixel_in_coadd],
            flux=flux[pixel_in_coadd],
            ivar=ivar[pixel_in_coadd],        
            flags=pixel_flags[pixel_in_coadd],
            full_output=True,
            **resample_kwds
        )
        
        # De-mask it 
        coadd_flux = _un_mask(coadd_flux, region_mask, default=np.nan)
        coadd_ivar = _un_mask(coadd_ivar, region_mask, default=0)
        coadd_pixel_flags = _un_mask(coadd_pixel_flags, region_mask, default=0)
                
        snrs = coadd_flux * np.sqrt(coadd_ivar)
        snr = np.median(snrs[coadd_ivar > 0])
        
        # Continuum normalization of BOSS spectra.t
        try:
            continuum, result = nmf_model.fit(coadd_flux, coadd_ivar, full_output=True)
        except RuntimeError:
            # Continuum fitting failed.
            continuum = np.array([np.nan])
            result = dict(
                W=np.zeros((0, 0)),
                theta=np.zeros(0),
                chi2=np.nan,
                rchi2=np.nan,
                rectified_model_flux=np.nan,
                mask=[],
            )
        
        spectrum_pks_considered = np.array([s.spectrum_pk for s in spectra])
        result = dict(
            L=L,
            P=P,
            snr=snr,
            source_pk=spectra[0].source_pk,
            release=spectra[0].release,
            sdss_id=spectra[0].source.sdss_id,
            run2d=spectra[0].run2d,
            telescope=spectra[0].telescope,            
            rcond=resample_kwds["rcond"],
            Lambda=resample_kwds["Lambda"],
            spectrum_pks_in_coadd=tuple(map(int, spectrum_pks_considered[spectrum_in_coadd])),
            spectrum_pks_considered=tuple(map(int, spectrum_pks_considered)),
            W=np.round(result["W"], 10),
            theta=np.round(result["theta"], 10),
            nmf_chi2=result["chi2"],
            nmf_rchi2=result["rchi2"],                  
            flux=coadd_flux,
            ivar=coadd_ivar,
            pixel_flags=coadd_pixel_flags,
            nmf_model_flux=result["rectified_model_flux"] * continuum[0],
            nmf_continuum=continuum[0],
            nmf_mask=result["mask"],
        )
        coadd = BossCoaddedSpectrum(**result)
        path = expand_path(coadd.path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fp:
            pickle.dump(result, fp)
            
        return None            
    
    except:
        log.exception(f"Exception while coadding BOSS spectra for source {source}")
        raise

"""Functions to create BOSS-related products."""

import numpy as np
from peewee import JOIN, fn
from astra.models.boss import BossVisitSpectrum
from astra.models.mwm import BossCombinedSpectrum, BossRestFrameVisitSpectrum
from astra.specutils.resampling import resample, pixel_weighted_spectrum
from astra.specutils.continuum.nmf.boss import BossNMFContinuum

from astra.migrations.utils import enumerate_new_spectrum_pks

from astra import __version__
from astra.utils import log
from astra.products.utils import (
    get_fields_and_pixel_arrays,
    get_fill_value, 
)

boss_continuum_model = BossNMFContinuum()

def include_boss_spectrum_in_coadd(source, spectrum):
    return (
        np.isfinite(spectrum.snr)
    &   (spectrum.snr > 3)
    &   ((spectrum.xcsao_rxc > 6) | ("mwm_wd" in source.sdss5_cartons["program"]))
    &   (spectrum.zwarning_flags <= 0)
    )
    

def include_boss_spectrum_in_rv_calculation(source, spectrum):
    return (
        np.isfinite(spectrum.xcsao_v_rad)
    &   (spectrum.xcsao_rxc > 6)
    )

def weighted_average(x, e_x):
    ivar = np.array(e_x)**-2
    sum_var = 1/np.sum(ivar)
    return (np.sum(x * ivar) * sum_var, np.sqrt(sum_var))


def prepare_boss_resampled_visit_and_coadd_spectra(source, telescope=None, run2ds=None, n_res=5, fill_values=None):
    q = (
        source
        .boss_visit_spectra
    )
    if telescope is not None:
        q = q.where(BossVisitSpectrum.telescope == telescope)
    if run2ds is not None:
        q = q.where(BossVisitSpectrum.run2d.in_(run2ds))

    q = q.order_by(BossVisitSpectrum.mjd.asc())

    visit_fields = get_fields_and_pixel_arrays((BossVisitSpectrum, ))
    
    visits = []
    v_rads, e_v_rads, in_stack = ([], [], [])
    for spectrum in q.iterator():           
        visit = {}
        for name, field in visit_fields.items():
            if name in ("pk", "v_astra"):
                continue
            
            value = getattr(spectrum, name)
            if value is None:
                value = get_fill_value(field, fill_values)
            visit[name] = value
        
        in_stack.append(include_boss_spectrum_in_coadd(source, spectrum))
        if include_boss_spectrum_in_rv_calculation(source, spectrum):
            v_rads.append(spectrum.xcsao_v_rad)
            e_v_rads.append(spectrum.xcsao_e_v_rad)

        visit["in_stack"] = in_stack[-1]
        visits.append(visit)
    
    if len(visits) == 0:
        return (None, None)
    
    # Let's resample spectra
    coadd_wavelength = BossCombinedSpectrum().wavelength
    N, P = NP = (len(visits), coadd_wavelength.size)
    zwarning_flags, gri_gaia_transform_flags = (0, 0)    
    visit_flux, visit_ivar, visit_pixel_flags = (np.zeros(NP), np.zeros(NP), np.zeros(NP, dtype=np.int64))
    for i, (use, visit) in enumerate(zip(in_stack, visits)):
        if use:            
            v_rad = visit["xcsao_v_rad"]
            zwarning_flags |= visit["zwarning_flags"]
            gri_gaia_transform_flags |= visit["gri_gaia_transform_flags"]
        else:
            v_rad = 0
        
        visit_flux[i], visit_ivar[i], visit_pixel_flags[i] = resample(
            visit["wavelength"] * (1 - v_rad/2.99792458e5),
            coadd_wavelength,
            visit["flux"],
            visit["ivar"],
            n_res=n_res,
            pixel_flags=visit["pixel_flags"]
        )
    
    if any(in_stack):                
        # Co-add the spectra
        coadd_flux, coadd_ivar, coadd_pixel_flags, *_ = pixel_weighted_spectrum(
            visit_flux[in_stack],
            visit_ivar[in_stack],
            visit_pixel_flags[in_stack],
        )
        bad_edge = (coadd_wavelength < 3650)
        coadd_flux[bad_edge] = np.nan
        coadd_ivar[bad_edge] = 0
        
        # Compute star-level statistics:        
        coadd_snr = coadd_flux * np.sqrt(coadd_ivar)
        coadd_snr = np.mean(coadd_snr[coadd_ivar > 0])
            
        v_rad, e_v_rad = weighted_average(v_rads, e_v_rads)
        std_v_rad = np.std(v_rads)
        median_e_v_rad = np.median(e_v_rads)
        n_good_rvs = len(v_rads)
        
        xcsao_params = {}
        for key in ("teff", "logg", "fe_h"):
            y, e_y = weighted_average(
                [v[f"xcsao_{key}"] for u, v in zip(in_stack, visits) if u],
                [v[f"xcsao_e_{key}"] for u, v in zip(in_stack, visits) if u]
            )
            xcsao_params[f"xcsao_{key}"] = y
            xcsao_params[f"xcsao_e_{key}"] = e_y
        xcsao_params["xcsao_mean_rxc"] = np.mean([v["xcsao_rxc"] for u, v in zip(in_stack, visits) if u])
        
        # Run NMF on the coadd spectrum.
        (coadd_continuum, ), meta = boss_continuum_model.fit(coadd_flux, coadd_ivar, full_output=True)
        
        # create the coadd spectrum object
        coadd_spectrum = BossCombinedSpectrum(
            source_pk=source.pk,
            release=spectrum.release,
            telescope=telescope,
            healpix=source.healpix or spectrum.healpix,
            sdss_id=source.sdss_id,
            run2d=spectrum.run2d,
            min_mjd=min([v["mjd"] for v in visits]),
            max_mjd=max([v["mjd"] for v in visits]),
            n_visits=len(visits),
            n_good_visits=sum(in_stack),
            n_good_rvs=n_good_rvs,
            v_rad=v_rad,
            e_v_rad=e_v_rad,
            std_v_rad=std_v_rad,
            median_e_v_rad=median_e_v_rad,    
            snr=coadd_snr,
            zwarning_flags=zwarning_flags,
            gri_gaia_transform_flags=gri_gaia_transform_flags,        
            nmf_rchi2=meta["rchi2"],
            **xcsao_params,
        )

        coadd_spectrum.flux = coadd_flux
        coadd_spectrum.ivar = coadd_ivar
        coadd_spectrum.pixel_flags = coadd_pixel_flags
        coadd_spectrum.continuum = coadd_continuum
        coadd_spectrum.nmf_rectified_model_flux = meta["rectified_model_flux"]
        
        # Now compute continuum coefficients for the visits, conditioned on the model rectified flux    
        theta, visit_continuum = boss_continuum_model._theta_step(
            visit_flux, 
            visit_ivar, 
            coadd_spectrum.nmf_rectified_model_flux
        )
    else:
        coadd_spectrum = None
        visit_continuum = np.nan * np.ones_like(visit_flux)

    # create the visit spectrum objects
    visit_spectra = []
    for i, visit in enumerate(visits):
        visit_spectrum = BossRestFrameVisitSpectrum(
            source_pk=source.pk,
            sdss_id=source.sdss_id,
            drp_spectrum_pk=visit.pop("spectrum_pk"),
            **visit
        )
        visit_spectrum.flux = visit_flux[i]
        visit_spectrum.ivar = visit_ivar[i]
        visit_spectrum.pixel_flags = visit_pixel_flags[i]
        visit_spectrum.continuum = visit_continuum[i]
        visit_spectra.append(visit_spectrum)
    
    save_spectra = []
    if coadd_spectrum is not None:
        save_spectra.append(coadd_spectrum)
    save_spectra.extend(visit_spectra)

    for spectrum_pk, spectrum in enumerate_new_spectrum_pks(save_spectra):
        spectrum.spectrum_pk = spectrum_pk
    
    if coadd_spectrum is not None:
        coadd_spectrum.save()
    if visit_spectra:
        BossRestFrameVisitSpectrum.bulk_create(visit_spectra)
            
    return (coadd_spectrum, visit_spectra)
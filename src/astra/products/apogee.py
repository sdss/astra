"""Functions to create APOGEE-related products."""

import numpy as np
from peewee import JOIN, fn
from astra.models.source import Source
from astra.models.apogee import ApogeeVisitSpectrum, ApogeeVisitSpectrumInApStar, ApogeeCoaddedSpectrumInApStar
from astra.models.boss import BossVisitSpectrum
from astra.specutils.resampling import resample, pixel_weighted_spectrum
from astra.specutils.continuum.nmf.apogee import ApogeeNMFContinuum

from astra.migrations.utils import enumerate_new_spectrum_pks
from astra.models.nmf_rectify import NMFRectify

from astra.models.mwm import ApogeeCombinedSpectrum, ApogeeRestFrameVisitSpectrum
from astra import __version__
from astra.utils import log
from astra.products.utils import (
    get_fields_and_pixel_arrays,
    get_fill_value, 
)

apogee_continuum_model = ApogeeNMFContinuum()
apogee_continuum_model.components[:, np.all(apogee_continuum_model.components == 0, axis=0)] = np.nan # clip the edges that we don't model :-)


def prepare_apogee_resampled_visit_and_coadd_spectra(source, telescope=None, apreds=None, fill_value=None):
    return prepare_apogee_resampled_visit_and_coadd_spectra_from_apstar(source, telescope=telescope, apreds=apreds, fill_values=fill_value)


def prepare_apogee_resampled_visit_and_coadd_spectra_from_apstar(source, telescope=None, apreds=None, fill_values=None):

    # Get a SDSS spectrum first by sorting by release
    sdss_id = source.sdss_id
    q = (
        ApogeeCoaddedSpectrumInApStar
        .select()
        .where(ApogeeCoaddedSpectrumInApStar.source_pk == source.pk)
    )
    if telescope is not None:
        q = q.where(ApogeeCoaddedSpectrumInApStar.telescope.startswith(telescope.lower()))
    if apreds is not None:
        q = q.where(ApogeeCoaddedSpectrumInApStar.apred.in_(apreds))
        
    q = (
        q
        .order_by(ApogeeCoaddedSpectrumInApStar.release.desc()) # prefer `sdss5` over `dr17`
        .objects()
        .limit(1)
    )
    
    coadd_fields = get_fields_and_pixel_arrays((ApogeeCombinedSpectrum, ))

    coadd_data = {"sdss_id": sdss_id}
    
    for result in q.iterator():
        for name, field in coadd_fields.items():
            if name in ("spectrum_pk", "pk", "v_astra"):
                continue
            
            try:
                value = getattr(result, name)
            except FileNotFoundError:
                break
            except:
                if name == "sdss_id":
                    value = sdss_id
                elif name in ("continuum", "nmf_rectified_model_flux", "input_spectrum_pks", "nmf_rchi2", "nmf_flags"):
                    continue
                else:
                    raise RuntimeError(f"Cannot find {name} on {result}")

            if value is None:
                value = get_fill_value(field, fill_values)
            coadd_data[name] = value

    if "flux" not in coadd_data:
        return (None, None)
    
    # Do NMF
    (coadd_continuum, ), meta = apogee_continuum_model.fit(coadd_data["flux"], coadd_data["ivar"], full_output=True)
    
    coadd_data.update(
        nmf_rchi2=meta["rchi2"],
        continuum_theta=meta["theta"],
        log10_W=np.log10(meta["W"]),
        continuum=coadd_continuum,
        nmf_rectified_model_flux=meta["rectified_model_flux"]
    )
    
    coadd_spectrum = ApogeeCombinedSpectrum(**coadd_data)
    for key in ("flux", "ivar", "continuum", "nmf_rectified_model_flux"):
        setattr(coadd_spectrum, key, coadd_data[key])
    
    # Now get the visits. First we will get the visits from the apStar, and then get the ones that aren't.
    q = (
        ApogeeVisitSpectrum
        .select(
            ApogeeVisitSpectrum,
            ApogeeVisitSpectrumInApStar.pk.alias("in_apstar_pk"),
        )
        .join(
            ApogeeVisitSpectrumInApStar, 
            JOIN.LEFT_OUTER,
            on=(ApogeeVisitSpectrum.spectrum_pk == ApogeeVisitSpectrumInApStar.drp_spectrum_pk),
        )
        .where(ApogeeVisitSpectrum.source_pk == source.pk)
    )
    if telescope is not None:
        q = q.where(ApogeeVisitSpectrum.telescope.startswith(telescope.lower()))

    if apreds is not None:
        q = q.where(ApogeeVisitSpectrum.apred.in_(apreds))
        
    q = (
        q
        .order_by(ApogeeVisitSpectrum.mjd.asc())
        .objects()
    )

    visit_data = []
    visit_fields = get_fields_and_pixel_arrays((ApogeeRestFrameVisitSpectrum, ))
    for spectrum in q.iterator():               
        visit = { 
            "in_stack": spectrum.in_apstar_pk is not None,
            "in_apstar_pk": spectrum.in_apstar_pk,
            "sdss_id": sdss_id,
        }
        for name, field in visit_fields.items():
            if name in ("pk", "v_astra", "continuum", "nmf_rchi2", "nmf_rectified_model_flux", "continuum", "nmf_flags") or name in visit: 
                continue
            elif name == "sdss_id":
                value = sdss_id
            elif name == "healpix":
                value = source.healpix
            else:        
                value = getattr(spectrum, name)
                
            if value is None:
                value = get_fill_value(field, fill_values)

            visit[name] = value
        visit_data.append(visit)
    
    q = (
        ApogeeVisitSpectrumInApStar
        .select()
        .where(ApogeeVisitSpectrumInApStar.source_pk == source.pk)        
    )
    if telescope is not None:
        q = q.where(ApogeeVisitSpectrumInApStar.telescope.startswith(telescope.lower()))
    
    if apreds is not None:
        q = q.where(ApogeeVisitSpectrumInApStar.apred.in_(apreds))
        
    apvisit_in_apstar_spectra = { r.pk: r for r in q.iterator() }

    # Create visit flux arrays, either from the apVisit or from the apStar
    N, P = (len(visit_data), ApogeeCombinedSpectrum().wavelength.size)
    visit_flux = np.zeros((N, P))
    visit_ivar = np.zeros((N, P))
    visit_pixel_flags = np.zeros((N, P), dtype=np.uint64)
    for i, visit in enumerate(visit_data):
        if visit["in_stack"]:
            spectrum = apvisit_in_apstar_spectra[visit["in_apstar_pk"]]
            visit_flux[i] = spectrum.flux
            visit_ivar[i] = spectrum.ivar
            visit_pixel_flags[i] = spectrum.pixel_flags
        else:
            # resample with 0 v_rad
            try:                    
                visit_flux[i], visit_ivar[i], visit_pixel_flags[i] = resample(
                    visit["wavelength"],
                    coadd_spectrum.wavelength,
                    visit["flux"],
                    visit["ivar"],
                    n_res=(5, 4.25, 3.5),
                    pixel_flags=visit["pixel_flags"]
                )     
            except:
                log.exception(f"Exception when trying to resample visit {visit}")
                continue       
    
    theta, visit_continuum = apogee_continuum_model._theta_step(
        visit_flux, 
        visit_ivar, 
        coadd_spectrum.nmf_rectified_model_flux
    )
    visit_spectra = []
    for i, visit in enumerate(visit_data):
        visit_spectrum = ApogeeRestFrameVisitSpectrum(
            source_pk=source.pk,
            drp_spectrum_pk=visit.pop("spectrum_pk"),
            **visit
        )        
        visit_spectrum.flux = visit_flux[i]
        visit_spectrum.ivar = visit_ivar[i]
        visit_spectrum.pixel_flags = visit_pixel_flags[i]
        visit_spectrum.continuum = visit_continuum[i]
        visit_spectra.append(visit_spectrum)
    
    for spectrum_pk, spectrum in enumerate_new_spectrum_pks([coadd_spectrum] + visit_spectra):
        spectrum.spectrum_pk = spectrum_pk
    
    coadd_spectrum.save()
    if visit_spectra:    
        ApogeeRestFrameVisitSpectrum.bulk_create(visit_spectra)
    
    return (coadd_spectrum, visit_spectra)

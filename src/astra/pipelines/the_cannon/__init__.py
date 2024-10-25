from typing import Iterable, Optional

import os
import pickle
import numpy as np
import concurrent.futures
from tqdm import tqdm
from peewee import chunked, JOIN, ModelSelect

from astra import task, __version__
from astra.utils import log, expand_path
from astra.models.spectrum import SpectrumMixin
from astra.models import ApogeeCoaddedSpectrumInApStar, ApogeeVisitSpectrumInApStar
from astra.models import ApogeeCombinedSpectrum
from astra.models.nmf_rectify import NMFRectify
from astra.models.the_cannon import TheCannon
from astra.pipelines.the_cannon.model import CannonModel
from astra.specutils.continuum.nmf.apogee import ApogeeNMFContinuum


@task
def the_apogee_cannon_coadd(
    spectra: Optional[Iterable[SpectrumMixin]] = (
        ApogeeCombinedSpectrum
        .select()
        .join(
            TheCannon,
            JOIN.LEFT_OUTER,
            on=(
                (TheCannon.spectrum_pk == ApogeeCombinedSpectrum.spectrum_pk)
            &   (TheCannon.v_astra == __version__)
            )
        )
        .where(TheCannon.spectrum_pk.is_null())        
    ),
    model_path: Optional[str] = "$MWM_ASTRA/pipelines/TheCannon/20231106-beta.model", 
    page=None,
    limit=None,
) -> Iterable[TheCannon]:
    
    total = None
    if isinstance(spectra, ModelSelect):
        if page is not None and limit is not None:
            log.info(f"Restricting to page {page} with {limit} items")
            spectra = spectra.paginate(page, limit)
            total = limit
        elif limit is not None:
            log.info(f"Restricting to {limit} items")
            spectra = spectra.limit(limit)      
            total = limit

        spectra = spectra.iterator()
        
    model = CannonModel.read(expand_path(model_path))
    
    for spectrum in tqdm(spectra, total=total, unit="spectra", desc="Inference"):
        flux = spectrum.flux / spectrum.continuum
        ivar = spectrum.ivar * spectrum.continuum**2
        flux, ivar = (np.atleast_2d(flux), np.atleast_2d(ivar))
        non_finite = (
            ~np.isfinite(flux)
        |   ~np.isfinite(ivar)
        |   (ivar == 0)
        )
        flux[non_finite] = 0
        ivar[non_finite] = 0

        try:
            op_params, op_cov, op_meta = model.fit_spectrum(flux, ivar, tqdm_kwds=dict(disable=True))
        except:
            yield TheCannon(
                spectrum_pk=spectrum.spectrum_pk,
                source_pk=spectrum.source_pk,
                flag_fitting_failure=True
            )
            continue

        i = 0
        result = dict(zip(map(str.lower, model.label_names), op_params[i]))
        result.update(
            dict(
                zip(
                    (f"e_{ln.lower()}" for ln in model.label_names),
                    np.sqrt(np.diag(op_cov[i]))
                )
            )
        )
        # Ignore correlation coeficients
        result.update(
            spectrum_pk=spectrum.spectrum_pk,
            source_pk=spectrum.source_pk,
            chi2=op_meta[i].get("chi2", np.nan),
            rchi2=op_meta[i].get("rchi2", np.nan),
            ier=op_meta[i].get("ier", -1),
            nfev=op_meta[i].get("nfev", -1),
            x0_index=np.argmin(op_meta[i]["trial_chi2"]),
        )
        
        rectified_model_flux = op_meta[i].get("model_flux", np.nan * np.ones_like(flux))
        
        output = TheCannon(**result)

        path = expand_path(output.intermediate_output_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fp:
            pickle.dump((spectrum.continuum, rectified_model_flux), fp)

        yield output


@task
def the_apogee_coadd_cannon(
    spectra: Optional[Iterable[SpectrumMixin]] = (
        ApogeeCoaddedSpectrumInApStar
        .select(
            ApogeeCoaddedSpectrumInApStar,
            NMFRectify.task_pk,
            NMFRectify.continuum_theta
        )
        .join(
            TheCannon, 
            JOIN.LEFT_OUTER, 
            on=(
                (TheCannon.spectrum_pk == ApogeeCoaddedSpectrumInApStar.spectrum_pk)
            &   (TheCannon.v_astra == __version__)
            )
        )
        .switch(ApogeeCoaddedSpectrumInApStar)
        .join(NMFRectify, on=(NMFRectify.spectrum_pk == ApogeeCoaddedSpectrumInApStar.spectrum_pk))
        .where(
            ~NMFRectify.flag_runtime_exception 
        &   ~NMFRectify.flag_could_not_read_spectrum
        &   TheCannon.spectrum_pk.is_null()
        )
        .objects()
    ), 
    model_path: Optional[str] = "$MWM_ASTRA/pipelines/TheCannon/20231106-beta.model", 
    page=None,
    limit=None,
) -> Iterable[TheCannon]:
    """
    Run inference (the test step) on some spectra with The Cannon.    
    """

    yield from _the_cannon(spectra, model_path, page, limit, ApogeeNMFContinuum())


# TODO: it is so dumb to have to split up this as two tasks just because we need the convenience of a default query on `spectra`
@task
def the_apogee_visit_cannon(
    spectra: Optional[Iterable[SpectrumMixin]] = (
        ApogeeVisitSpectrumInApStar
        .select(
            ApogeeVisitSpectrumInApStar,
            NMFRectify.task_pk,
            NMFRectify.continuum_theta
        )
        .join(
            TheCannon, 
            JOIN.LEFT_OUTER, 
            on=(
                (TheCannon.spectrum_pk == ApogeeVisitSpectrumInApStar.spectrum_pk)
            &   (TheCannon.v_astra == __version__)
            )
        )
        .switch(ApogeeVisitSpectrumInApStar)
        .join(NMFRectify, on=(NMFRectify.spectrum_pk == ApogeeVisitSpectrumInApStar.spectrum_pk))
        .where(
            ~NMFRectify.flag_runtime_exception 
        &   ~NMFRectify.flag_could_not_read_spectrum
        &   TheCannon.spectrum_pk.is_null()
        )
        .objects()
    ), 
    model_path: Optional[str] = "$MWM_ASTRA/pipelines/TheCannon/20231106-beta.model", 
    page=None,
    limit=None,
) -> Iterable[TheCannon]:
    """
    Run inference (the test step) on some spectra with The Cannon.    
    """

    yield from _the_cannon(spectra, model_path, page, limit, ApogeeNMFContinuum())




def _the_cannon(spectra, model_path, page, limit, continuum_model):
    
    total = None
    if isinstance(spectra, ModelSelect):
        if page is not None and limit is not None:
            log.info(f"Restricting to page {page} with {limit} items")
            spectra = spectra.paginate(page, limit)
            total = limit
        elif limit is not None:
            log.info(f"Restricting to {limit} items")
            spectra = spectra.limit(limit)      
            total = limit

        spectra = spectra.iterator()
        
    model = CannonModel.read(expand_path(model_path))
    
    for spectrum in tqdm(spectra, total=total, unit="spectra", desc="Inference"):
        continuum = continuum_model.continuum(spectrum.wavelength, spectrum.continuum_theta)[0]
        flux = spectrum.flux / continuum
        ivar = spectrum.ivar * continuum**2
        flux, ivar = (np.atleast_2d(flux), np.atleast_2d(ivar))
        non_finite = (
            ~np.isfinite(flux)
        |   ~np.isfinite(ivar)
        |   (ivar == 0)
        )
        flux[non_finite] = 0
        ivar[non_finite] = 0

        try:
            op_params, op_cov, op_meta = model.fit_spectrum(flux, ivar, tqdm_kwds=dict(disable=True))
        except:
            yield TheCannon(
                spectrum_pk=spectrum.spectrum_pk,
                source_pk=spectrum.source_pk,
                flag_fitting_failure=True
            )
            continue

        i = 0
        result = dict(zip(map(str.lower, model.label_names), op_params[i]))
        result.update(
            dict(
                zip(
                    (f"e_{ln.lower()}" for ln in model.label_names),
                    np.sqrt(np.diag(op_cov[i]))
                )
            )
        )
        # Ignore correlation coeficients
        result.update(
            spectrum_pk=spectrum.spectrum_pk,
            source_pk=spectrum.source_pk,
            chi2=op_meta[i].get("chi2", np.nan),
            rchi2=op_meta[i].get("rchi2", np.nan),
            ier=op_meta[i].get("ier", -1),
            nfev=op_meta[i].get("nfev", -1),
            x0_index=np.argmin(op_meta[i]["trial_chi2"]),
        )
        
        rectified_model_flux = op_meta[i].get("model_flux", np.nan * np.ones_like(flux))
        
        
        output = TheCannon(**result)

        path = expand_path(output.intermediate_output_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fp:
            pickle.dump((continuum, rectified_model_flux), fp)
        
        yield output
        
        
        
    
    
def _the_cannon_parallel(spectra, model_path, page, limit, continuum_model, max_workers=32):
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

    batch_size = 1 + len(spectra) // max_workers

    futures = []    
    for chunk in tqdm(chunked(spectra, batch_size), total=1, desc="Chunking", unit="chunk"):
        futures.append(executor.submit(_the_cannon_worker, chunk, model))
    
    with tqdm(total=len(futures) * batch_size, unit="spectra", desc="Inference") as pb:
        for future in concurrent.futures.as_completed(futures):
            for result_kwd in future.result():
                    
                rectified_model_flux = result_kwd.pop("rectified_model_flux", np.nan)
                continuum = result_kwd.pop("continuum", 0)

                output = TheCannon(**result_kwd)

                path = expand_path(output.intermediate_output_path)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as fp:
                    pickle.dump((continuum, rectified_model_flux), fp)

                yield output
                pb.update()                
    '''
    for result in tqdm(_the_cannon_worker(spectra, model)):
        rectified_model_flux = result.pop("rectified_model_flux", np.nan)
        continuum = result.pop("continuum", 0)

        output = TheCannon(**result)

        path = expand_path(output.intermediate_output_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fp:
            pickle.dump((continuum, rectified_model_flux), fp)

        yield output
    '''
        
def _the_cannon_worker(spectra, model, **kwargs):
    
    continuum_model = ApogeeNMFContinuum()
        
    flux = []
    ivar = []
    continua = []
    fitted_spectra = []
    for spectrum in tqdm(spectra, total=1, desc="Rectifying"):        
        continuum = continuum_model.continuum(spectrum.wavelength, spectrum.continuum_theta)[0]
        continua.append(continuum)
        flux.append(spectrum.flux / continuum)
        ivar.append(spectrum.ivar * continuum**2)
        fitted_spectra.append(spectrum)
            
    flux, ivar = (np.atleast_2d(flux), np.atleast_2d(ivar))
    non_finite = (
        ~np.isfinite(flux)
    |   ~np.isfinite(ivar)
    |   (ivar == 0)
    )
    flux[non_finite] = 0
    ivar[non_finite] = 0
    
    op_params, op_cov, op_meta = model.fit_spectrum(flux, ivar, tqdm_kwds=dict(disable=True))
    
    result_kwds = []
    for i, spectrum in enumerate(fitted_spectra):
            
        result = dict(zip(map(str.lower, model.label_names), op_params[i]))
        result.update(
            dict(
                zip(
                    (f"e_{ln.lower()}" for ln in model.label_names),
                    np.sqrt(np.diag(op_cov[i]))
                )
            )
        )
        # Ignore correlation coeficients
        result.update(
            spectrum_pk=spectrum.spectrum_pk,
            source_pk=spectrum.source_pk,
            rectified_model_flux=op_meta[i]["model_flux"],
            continuum=continua[i],
            chi2=op_meta[i].get("chi2", np.nan),
            rchi2=op_meta[i].get("rchi2", np.nan),
            ier=op_meta[i].get("ier", -1),
            nfev=op_meta[i].get("nfev", -1),
            x0_index=np.argmin(op_meta[i]["trial_chi2"]),
        )
        #yield result
        result_kwds.append(result)
    
    return result_kwds

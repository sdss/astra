
import numpy as np
import os
from typing import Iterable
from astropy.io import fits
import concurrent.futures
from astra.models.spectrum import SpectrumMixin
from astra.models.apogee import ApogeeCoaddedSpectrumInApStar, ApogeeVisitSpectrumInApStar, ApogeeVisitSpectrum
from astra.models.nmf_rectify import ApogeeNMFRectification
from astra.specutils.continuum.nmf import ApogeeContinuum
from astra import task
from astra.utils import log
from peewee import chunked
from tqdm import tqdm
import warnings


@task
def rectify_apogee_spectra(
    spectra: Iterable[SpectrumMixin],
    max_workers: int = 5, 
    batch_size: int = 10
) -> Iterable[ApogeeNMFRectification]:

    executor = concurrent.futures.ProcessPoolExecutor(max_workers)

    model = ApogeeContinuum()

    A = model.full_design_matrix(1)

    futures = []
    for chunk in chunked(spectra, batch_size):
        futures.append(executor.submit(_rectify_spectra, chunk, model, A))
    
    N = len(spectra)
    with tqdm(total=N) as pb:
        for future in concurrent.futures.as_completed(futures):
            yield from future.result()
            pb.update(batch_size)
    


@task
def rectify_spectra_in_apstar_products(
    spectra: Iterable[ApogeeCoaddedSpectrumInApStar], 
    max_workers: int = 5, 
    batch_size: int = 10
) -> Iterable[ApogeeNMFRectification]:

    executor = concurrent.futures.ProcessPoolExecutor(max_workers)

    model = ApogeeContinuum()

    futures = []
    for chunk in chunked(spectra, batch_size):
        futures.append(executor.submit(_rectify_spectra_in_apstar_product, chunk, model))

    N = len(spectra)
    with tqdm(total=N) as pb:
        for future in concurrent.futures.as_completed(futures):
            yield from future.result()
            pb.update(batch_size)


def _rectify_spectra(spectra, model, A):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        results = []
        for spectrum in spectra:
            try:
                continuum, continuum_meta = model.fit(spectrum.flux, spectrum.ivar, A=A)
            except:
                log.exception(f"Exception when trying to rectify spectrum {spectrum}")
            else:                    
                results.append(
                    ApogeeNMFRectification(
                        spectrum_pk=spectrum.spectrum_pk,
                        source_pk=spectrum.source_pk,
                        theta=continuum_meta["p_opt"],
                        rchi2=continuum_meta["rchi2"],
                        L=model.L,
                        deg=model.deg,
                        n_components=model.components.shape[0]
                    )                
                )

    return results


def _rectify_spectra_in_apstar_product(spectra, model):

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        results = []
        for spectrum in spectra:
            try:                
                with fits.open(spectrum.absolute_path) as image:
                    
                    flux = np.atleast_2d(image[1].data)
                    ivar = np.atleast_2d(image[2].data**-2)

                    bad = (flux <= 0) | (ivar == 0) | (~np.isfinite(flux)) | (~np.isfinite(ivar))
                    flux[bad] = 0.0
                    ivar[bad] = 0.0

                    # fit all spectra simultaneously
                    continuum, continuum_meta = model.fit(flux, ivar)
                    
                    results.append(
                        ApogeeNMFRectification(
                            spectrum_pk=spectrum.spectrum_pk,
                            source_pk=spectrum.source_pk,
                            theta=continuum_meta["p_opt"],
                            rchi2=continuum_meta["rchi2"],
                            L=model.L,
                            deg=model.deg,
                            n_components=model.components.shape[0]
                        )    
                    )
            except:
                continue
        
    return results


if __name__ == "__main__":

    from astra.models.apogee import ApogeeCoaddedSpectrumInApStar


    s = ApogeeCoaddedSpectrumInApStar.get()
    rectify_spectra_in_apstar_products([s])
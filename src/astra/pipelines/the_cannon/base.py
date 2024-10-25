import multiprocessing as mp
from typing import Iterable
from itertools import cycle
from astropy.io import fits
from scipy.spatial import Delaunay

import numpy as np

import multiprocessing as mp

from astra.base import task
from astra.utils import log, expand_path, flatten
from astra.models.the_cannon import TheCannon
from astra.models.spectrum import SpectrumMixin
from astra.pipelines.the_cannon.model import CannonModel


@task
def the_cannon(spectra: Iterable[SpectrumMixin], model_path: str) -> Iterable[TheCannon]:

    model = CannonModel.read(expand_path(model_path))
    
    # Build a convex hull

    # Let's just build one for the first 8 dimensions
    np.random.seed(0)
    idx = np.random.choice(model.training_labels.shape[0], 1000, replace=False)
    D_hull = 8
    print(f"Creating a pseudo-covex hull in {D_hull} dimensions")
    hull = Delaunay(model.training_labels[idx, :D_hull])

    for batch in parallel_batch_read(_worker, spectra, 512, N=4):
        flux = np.array([f for dp, kwds, f, i in batch])
        ivar = np.array([i for dp, kwds, f, i in batch])

        op_params, op_cov, op_meta = model.fit_spectrum(
            flux, 
            ivar, 
            n_threads=128,
            prefer="processes"
        )

        for i, (dp, kwds, *_) in enumerate(batch):

            result = dict(zip(map(str.lower, model.label_names), op_params[i]))
            result.update(
                dict(
                    zip(
                        (f"e_{ln.lower()}" for ln in model.label_names),
                        np.sqrt(np.diag(op_cov[i]))
                    )
                )
            )
            # TODO: Ignoring the correlation coefficients for now.
            result.update(
                chi_sq=op_meta[i].get("chi_sq", np.nan),
                reduced_chi_sq=op_meta[i].get("reduced_chi_sq", np.nan),
                ier=op_meta[i].get("ier", -1),
                nfev=op_meta[i].get("nfev", -1),
                x0_index=np.argmin(op_meta[i]["trial_chisq"]),
                in_convex_hull=hull.find_simplex(op_params[i, :D_hull]) >= 0,
                **kwds
            )

            yield TheCannon(data_product=dp, **result)




def _prepare_data(data_product):
    # TODO: assuming mwmStar/mwmVisit!!
    with fits.open(data_product.path) as image:
        for hdu, telescope in enumerate(("apo25m", "lco25m"), start=3):
            if not image[hdu].data:
                continue

            #n_spectra = len(image[hdu].data["FLUX"])

            snr = image[hdu].data["SNR"].flatten()[0]
            flux = image[hdu].data["FLUX"][0]
            ivar = image[hdu].data["E_FLUX"][0]**-2
            continuum = image[hdu].data["CONTINUUM"][0]


            rectified_flux = flux / continuum
            rectified_ivar = continuum * ivar * continuum

            bad_pixel = (
                ~np.isfinite(rectified_flux) 
            |   ~np.isfinite(rectified_ivar) 
            |   (rectified_flux == 0) 
            |   (rectified_ivar == 0)
            )

            rectified_flux[bad_pixel] = 1.0
            rectified_ivar[bad_pixel] = 0.0

            meta = dict(
                snr=snr,
                instrument="APOGEE",
                telescope=telescope,
                plate=None,
                field=None,
                fiber=None,
                apstar_pk=None,
                apvisit_pk=None,
                obj=None,
            )

            yield (data_product, meta, rectified_flux, rectified_ivar)
    
            
    '''
    for spectrum in SpectrumList.read(data_product.path):
        if not spectrum_overlaps(spectrum, 16_500 * u.Angstrom):
            # Skip non-APOGEE spectra.
            continue

        N, P = np.atleast_2d(spectrum.flux).shape
        flux = np.nan_to_num(spectrum.flux.value).astype(np.float32).reshape((N, P))
        ivar = np.nan_to_num(
                spectrum.uncertainty.represent_as(InverseVariance).array
            ).astype(np.float32).reshape((N, P))
        
        kwds = _get_sdss_metadata(data_product, spectrum)
        yield (data_product, kwds, flux, ivar)
    '''


def _worker(q_input, q_output, batch_size):
    while True:
        data_product = q_input.get()
        if data_product is None:
            break
        
        try:
            for result in _prepare_data(data_product):
                # Only put results when the queue is empty, otherwise we might load data faster than we can process it.
                # (which eventually leads to an out-of-memory error)
                while True:
                    #if q_output.qsize() < (2 * batch_size):
                    if q_output.empty():
                        q_output.put(result)
                        break
        except:
            log.exception(f"Exception in worker with data product {data_product}")
            continue
    
    q_output.put(None)
    return None

def parallel_batch_read(target, data_product, batch_size, N=None):
    raise a
    N = N or mp.cpu_count()
    q_output = mp.Queue()
    qps = []
    for i in range(N):
        q = mp.Queue()
        p = mp.Process(target=target, args=(q, q_output, batch_size))
        p.start()
        qps.append((q, p))

    B, batch = (0, [])
    for i, ((q, p), data_product) in enumerate(zip(cycle(qps), flatten(data_product))):
        q.put(data_product)

    for (q, p) in qps:
        q.put(None)

    N_done = 0
    while True:
        result = q_output.get()
        if result is None:
            N_done += 1
            if N_done == N:
                break
        else:                
            batch.append(result)
            B += 1

            if B >= batch_size:
                yield batch
                B, batch = (0, [])

    if batch:
        yield batch

    for q, p in qps:
        p.join()


import numpy as np
import multiprocessing as mp
from itertools import cycle
from time import time, sleep
from tqdm import tqdm

from astra.utils import log
from astra.models.astronn_dist import AstroNNdist

import tensorflow_probability as tfp
from astroNN.apogee import apogee_continuum 
from astroNN.gaia import extinction_correction, fakemag_to_pc


def _prepare_data(spectrum):
    try:
        N, P = np.atleast_2d(spectrum.flux).shape
        flux = np.atleast_2d(spectrum.flux).reshape((N, P))
        e_flux = np.atleast_2d(spectrum.ivar**-0.5).reshape((N, P))
        bitmask = np.atleast_2d(spectrum.pixel_flags).reshape((N, P))
    except:
        return (spectrum.spectrum_pk, spectrum.source_pk, None, None, None, None, None)

    #### continuum normalization
    P_new = 7514
    norm_flux = np.zeros((N, P_new))
    norm_flux_err = np.zeros((N, P_new))
    for i in range(N):
        norm_flux[i, :], norm_flux_err[i, :] = apogee_continuum(flux[i, :], e_flux[i, :], bitmask=bitmask[i, :], dr = 17)

    #### get photometry and extinction
    meta, missing_photometry, missing_extinction = get_metadata(spectrum)
    #meta = np.tile(metadata, N).reshape((N, -1))

    return (spectrum.spectrum_pk, spectrum.source_pk, norm_flux, norm_flux_err, meta, missing_photometry, missing_extinction)
    #return (spectrum.spectrum_id, norm_flux, norm_flux_err, meta)

def _worker(q_input, q_output):
    while True:
        spectrum = q_input.get()
        if spectrum is None:
            break

        try:
            result = _prepare_data(spectrum)
        except:
            log.exception(f"Exception in worker with data product {spectrum}")
            continue
        else:
            # Only put results when the queue is empty, otherwise we might load data faster than we can process it.
            # (which eventually leads to an out-of-memory error)

            # Note from A. Casey: I commented this out while dealing with some weird COMMIT / GPU hanging, and I dn't know what the cause was.
            #       If you get out of memory errors in future, maybe uncomment these.
            while True:
                if q_output.empty():
                    q_output.put(result)
                    #log.info(f"Put result {result}")
                    break

                sleep(1)

    q_output.put(None)
    return None

def _inference(model, batch):
    # Astra does record the time taken by each task, but this is a naive estimate which does not account for cases
    # where inferences are done in batches, which leads to one spectrum in the batch having the estimated time for
    # the whole batch, and all others having some small value (1e-5), so we will calculate the average time here

    t_init = time()
    spectrum_pks, source_pks, all_flux, all_e_flux, all_meta, missing_photometrys, missing_extinctions = ([], [], [], [], [], [], [])
    for spectrum_pk, source_pk, f, e_f, m, mp, me in batch:
        if f is None:
            # OS Error when loading it
            yield AstroNNdist(
                spectrum_pk=spectrum_pk,
                source_pk=source_pk,
                flag_no_result = True
            )
        else:
            spectrum_pks.append(spectrum_pk)
            source_pks.append(source_pk)
            all_flux.append(f)
            all_e_flux.append(e_f)
            all_meta.append(m)
            missing_photometrys.append(mp)
            missing_extinctions.append(me)

    if len(all_flux) > 0:
        N_param = len(model.targetname)
    
        #### astroNN prediction: K-band absolute magnitude and distance
        all_flux = np.atleast_2d(all_flux).reshape((-1, 7514))
        try:
            fakemag, fakemag_err = model.predict(all_flux)
        except ValueError:
            raise
        else:
            k_mag_cor = extinction_correction(all_meta[0][0], all_meta[0][2])
            #pc, pc_err = fakemag_to_pc(fakemag, k_mag_cor, fakemag_err['total']) # for the TensorFlow version
            pc, pc_err = fakemag_to_pc(fakemag, k_mag_cor, fakemag_err) # for the PyTorch version
    
        #### record results
        mean_t_elapsed = (time() - t_init) / len(spectrum_pks)
        for i, (spectrum_pk, source_pk) in enumerate(zip(spectrum_pks, source_pks)):
            dist = np.atleast_1d(pc.value)[i]
            dist_err = np.atleast_1d(pc_err.value)[i]

            if dist > 10**10:
                dist = np.nan
            if dist_err > 10**10:
                dist_err = np.nan


            output = AstroNNdist(
                spectrum_pk=spectrum_pk,
                source_pk=source_pk,
                t_elapsed=mean_t_elapsed,
                k_mag=all_meta[i][0],
                ebv = all_meta[i][1],
                a_k_mag=all_meta[i][2],
                L_fakemag=float(np.atleast_1d(fakemag)[i]),
                #L_fakemag=float(fakemag_err['total']),
                L_fakemag_err=float(np.atleast_1d(fakemag_err)[i]),
                dist=dist,
                dist_err=dist_err,
            )
            output.apply_flags(all_meta[i], missing_photometry=missing_photometrys[i], missing_extinction=missing_extinctions[i])
            #print("+"*6, "(_inference) output:", output.__data__)
            #print("+"*6, "(_inference) flags", np.binary_repr(output.__data__['result_flags']))
            yield output


def parallel_batch_read(target, spectra, batch_size, cpu_count=None):
    """
    Read a batch of spectra in parallel.

    :param target:
        The target function to be executed in parallel. This function must accept a single argument
        containing a list of spectra.

    :param spectra:
        A list of spectra to be processed.

    :param batch_size:
        The number of spectra to be processed in each batch.

    :param cpu_count: [optional]
        The number of CPUs to use. If `None`, the number of CPUs will be determined automatically.

    :returns:
        A generator that yields the results of the target function.
    """

    N = cpu_count or mp.cpu_count()
    q_output = mp.Queue()
    qps = []
    for i in range(N):
        q = mp.Queue()
        p = mp.Process(target=target, args=(q, q_output))
        p.start()
        qps.append((q, p))

    log.info(f"Distributing spectra")
    B, batch = (0, [])
    for i, ((q, p), spectrum) in enumerate(zip(cycle(qps), tqdm(spectra, total=0))):
        q.put(spectrum)

    for (q, p) in qps:
        q.put(None)
    log.info(f"Done")

    N_done = 0
    while True:
        try:
            result = q_output.get(timeout=30)
        except:
            log.exception("Timeout on thing")
            continue

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


def get_metadata(spectrum):
    """
    Get requisite photometry and extinction from a given spectrum for astroNN.

    :param spectrum:
        An `astra.tools.spectrum.Spectrum1D` spectrum.

    :returns:
        A three-length tuple containing relevant metadata for the given spectrum. The first entry in
        the tuple contains the header keys, and the second entry contains the values of the metadata.
    """

    #mdata_replacements = np.array([17.02500, 0.0])
    #mdata_means = np.array([10.702106344460235, 0.0])
    mdata_replacements = np.zeros(3, dtype=float) # k_mag, ebv, a_k_mag

    metadata = np.zeros(3, dtype=float) # k_mag, ebv, a_k_mag 
    try:
        metadata[0] = spectrum.source.k_mag # 2MASS K-band apparent magnitude
    except:
        missing_photometry = True

    try:
        metadata[1] = spectrum.source.ebv # E(B-V) reddening
    except:
        missing_extinction = True

    #### calculate A_K from E(B-V) 
    #### YS note: A_K = 0.918*(H-4.5mu-0.08)= 0.918*E(B-V)/(E(B-V)_0/E(H-4.5mu)_0) = 0.918*E(B-V)/2.61
    metadata[2] = metadata[1]*0.3517  

    #### replace bad values with 0.0
    de_nanify = lambda x: x if (x != "NaN" and x != -999999 and x is not None) else np.nan
    
    metadata = np.array(list(map(de_nanify, metadata)))

    missing_photometry = np.any(
        ~np.isfinite(metadata[0])
    |   (metadata[0] >= 98)
    |   (metadata[0] <= -1)
    )

    missing_extinction = np.any(
        ~np.isfinite(metadata[1])
    |   (metadata[1] >= 98)
    |   (metadata[1] <= 0) # YS note: E(B-V) should be positive and non-zero?
    )

    metadata = np.where(metadata < 98, metadata, mdata_replacements)
    metadata = np.where(np.isfinite(metadata), metadata, mdata_replacements)
    metadata = np.where(metadata > -1, metadata, mdata_replacements)

    #meta = dict(zip(["k_mag", "ebv", "a_k_mag"], metadata))
    return metadata, missing_photometry, missing_extinction
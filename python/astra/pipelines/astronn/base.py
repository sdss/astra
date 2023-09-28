import numpy as np
import multiprocessing as mp
from itertools import cycle
from time import time, sleep

from astra.utils import log, flatten
from astra.models.astronn import AstroNN

import tensorflow_probability as tfp
from astroNN.apogee import apogee_continuum 


def _prepare_data(spectrum):
    try:
        N, P = np.atleast_2d(spectrum.flux).shape
        flux = np.nan_to_num(spectrum.flux).astype(np.float32).reshape((N, P))
        e_flux = np.nan_to_num(spectrum.ivar**-0.5).astype(np.float32).reshape((N, P))
        bitmask = np.nan_to_num(spectrum.pixel_flags).astype(bool).reshape((N, P))
    except:
        return (spectrum.spectrum_id, spectrum.source_id, None, None)

    #### continuum normalization
    P_new = 7514
    norm_flux = np.zeros((N, P_new))
    norm_flux_err = np.zeros((N, P_new))
    for i in range(N):
        norm_flux[i, :], norm_flux_err[i, :] = apogee_continuum(flux[i, :], e_flux[i, :], bitmask=bitmask[i, :], dr = 17)

    return (spectrum.spectrum_id, spectrum.source_id, norm_flux, norm_flux_err)

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
                    log.info(f"Put result {result}")
                    break

                sleep(1)

    q_output.put(None)
    return None

def _inference(model, batch):
    # Astra does record the time taken by each task, but this is a naive estimate which does not account for cases
    # where inferences are done in batches, which leads to one spectrum in the batch having the estimated time for
    # the whole batch, and all others having some small value (1e-5), so we will calculate the average time here

    t_init = time()
    spectrum_ids, source_ids, all_flux, all_e_flux = ([], [], [], [])
    for spectrum_id, source_id, f, e_f in batch:
        if f is None:
            # OS Error when loading it
            yield AstroNN(
                spectrum_id=spectrum_id,
                source_id=source_id,
                flag_no_result = True
            )
        else:
            spectrum_ids.append(spectrum_id)
            source_ids.append(source_id)
            all_flux.append(f)
            all_e_flux.append(e_f)

    if len(all_flux) > 0:
        N_param = len(model.targetname)

        #### astroNN prediction 
        results = []
        for i in range(len(batch)):
            result = []
            try:
                pred, pred_err = model.predict(all_flux[i])
                #print(pred, pred_err)
            except ValueError:
                result = [np.nan]*N_param*2 + [1]
            else:
                for idx in range(N_param):
                    result.append(pred[0, idx])
                    result.append(pred_err['total'][0, idx])
                result.append(0)

            result = np.array(result)
            print("+"*6, 'result:', result)
            results.append(result)
        results = np.array(results)
        print("+"*6, 'shape of results:', results.shape)
    
        #### record results
        mean_t_elapsed = (time() - t_init) / len(spectrum_ids)
        for i, (spectrum_id, source_id) in enumerate(zip(spectrum_ids, source_ids)):
            print("+"*6, 'write to database: Teff =', results[i, 0])
            output = AstroNN(
                spectrum_id=spectrum_id,
                source_id=source_id,
                t_elapsed=mean_t_elapsed,
                teff = results[i, 0],
                e_teff = results[i, 1],
                logg = results[i, 2],
                e_logg = results[i, 3],
                c_h = results[i, 4],
                e_c_h = results[i, 5],
                c_1_h = results[i, 6],
                e_c_1_h = results[i, 7],
                n_h = results[i, 8],
                e_n_h = results[i, 9],
                o_h = results[i, 10],
                e_o_h = results[i, 11],
                na_h = results[i, 12],
                e_na_h = results[i, 13],
                mg_h = results[i, 14],
                e_mg_h = results[i, 15],
                al_h = results[i, 16],
                e_al_h = results[i, 17],
                si_h = results[i, 18],
                e_si_h = results[i, 19],
                p_h = results[i, 20],
                e_p_h = results[i, 21],
                s_h = results[i, 22],
                e_s_h = results[i, 23],
                k_h = results[i, 24],
                e_k_h = results[i, 25],
                ca_h = results[i, 26],
                e_ca_h = results[i, 27],
                ti_h = results[i, 28],
                e_ti_h = results[i, 29],
                ti_2_h = results[i, 30],
                e_ti_2_h = results[i, 31],
                v_h = results[i, 32],
                e_v_h = results[i, 33],
                cr_h = results[i, 34],
                e_cr_h = results[i, 35],
                mn_h = results[i, 36],
                e_mn_h = results[i, 37],
                fe_h = results[i, 38],
                e_fe_h = results[i, 39],
                co_h = results[i, 40],
                e_co_h = results[i, 41],
                ni_h = results[i, 42],
                e_ni_h = results[i, 43],
                result_flags=results[i, 44]
            )
            #output.apply_flags(meta[i])
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
    for i, ((q, p), spectrum) in enumerate(zip(cycle(qps), flatten(spectra))):
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
        log.info(f"Have result {result}, {B}")

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

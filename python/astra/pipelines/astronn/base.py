import numpy as np
import multiprocessing as mp
from itertools import cycle
from time import time, sleep
from tqdm import tqdm

from astra.utils import log
from astra.models.astronn import AstroNN

from astroNN.apogee import apogee_continuum 


def _prepare_data(spectrum):
    try:
        N, P = np.atleast_2d(spectrum.flux).shape
        flux = np.atleast_2d(spectrum.flux).reshape((N, P))
        e_flux = np.atleast_2d(spectrum.ivar**-0.5).reshape((N, P))
        bitmask = np.atleast_2d(spectrum.pixel_flags).reshape((N, P))
    except:
        return (spectrum.spectrum_pk, spectrum.source_pk, None, None)

    #### continuum normalization
    P_new = 7514
    norm_flux = np.zeros((N, P_new))
    norm_flux_err = np.zeros((N, P_new))
    for i in range(N):
        norm_flux[i, :], norm_flux_err[i, :] = apogee_continuum(flux[i, :], e_flux[i, :], bitmask=bitmask[i, :], dr = 17)

    return (spectrum.spectrum_pk, spectrum.source_pk, norm_flux, norm_flux_err)

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
    spectrum_pks, source_pks, all_flux, all_e_flux = ([], [], [], [])
    for spectrum_pk, source_pk, f, e_f in batch:
        if f is None:
            # OS Error when loading it
            yield AstroNN(
                spectrum_pk=spectrum_pk,
                source_pk=source_pk,
                flag_no_result=True
            )
        else:
            spectrum_pks.append(spectrum_pk)
            source_pks.append(source_pk)
            all_flux.append(f)
            all_e_flux.append(e_f)

    if len(all_flux) > 0:
        N_param = len(model.targetname)

        #### astroNN prediction 
        '''
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
        '''
        all_flux = np.atleast_2d(all_flux).reshape((-1, 7514))
        try:
            pred, pred_err = model.predict(all_flux)
        except ValueError: # on TensorFlow some weird thing with fast_mc_inference_v2_internal
            raise

        #results = np.hstack([pred, pred_err['total']]) # for TensorFlow version 
        #results = np.hstack([pred, pred_err]) # for PyTorch version
                    
        #### record results
        mean_t_elapsed = (time() - t_init) / len(spectrum_pks)
        for i, (spectrum_pk, source_pk) in enumerate(zip(spectrum_pks, source_pks)):
            result_kwds = dict(
                spectrum_pk=spectrum_pk,
                source_pk=source_pk,
                t_elapsed=mean_t_elapsed,
            )
            result_kwds.update(dict(zip(model.targetname, pred[i])))
            result_kwds.update(dict(zip([f"e_{ln}" for ln in model.targetname], pred_err[i])))    
            result_kwds["result_flags"] = 0 if result_kwds["e_logg"] < 0.2 else 1 
            
            yield AstroNN(**result_kwds)


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

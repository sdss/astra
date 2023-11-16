
import torch
import numpy as np
import multiprocessing as mp
from itertools import cycle
from time import time, sleep

from astra.utils import log, flatten
from astra.models import ApogeeNetV2

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def _prepare_data(spectrum, large_error):
    try:    
        N, P = np.atleast_2d(spectrum.flux).shape
        flux = np.nan_to_num(spectrum.flux).astype(np.float32).reshape((N, P))
        e_flux = np.nan_to_num(spectrum.ivar**-0.5).astype(np.float32).reshape((N, P))
    except:
        return (spectrum.spectrum_pk, spectrum.source_pk, None, None, None, None, None)

    meta_dict, metadata_norm, missing_photometry = get_metadata(spectrum)
    meta = np.tile(metadata_norm, N).reshape((N, -1))

    N, P = flux.shape
    flux = flux.reshape((N, 1, P))
    e_flux = e_flux.reshape((N, 1, P))
    median_error = 5 * np.median(e_flux, axis=(1, 2))
    for j, value in enumerate(median_error):
        bad_pixel = (e_flux[j] == large_error) | (e_flux[j] >= value)
        e_flux[j][bad_pixel] = value

    return (spectrum.spectrum_pk, spectrum.source_pk, flux, e_flux, meta, meta_dict, missing_photometry)



def _worker(q_input, q_output, large_error):
    
    while True:
        spectrum = q_input.get()
        if spectrum is None:
            break        
        try:
            result = _prepare_data(spectrum, large_error)
        except:
            log.exception(f"Exception in worker with data product {spectrum}")
            continue
        else:
            # Only put results when the queue is empty, otherwise we might load data faster than we can process it.
            # (which eventually leads to an out-of-memory error)

            # Note: I commented this out while dealing with some weird COMMIT / GPU hanging, and I dn't know what the cause was.
            #       If you get out of memory errors in future, maybe uncomment these.
            while True:
                if q_output.empty():
                    q_output.put(result)
                    log.info(f"Put result {result}")
                    break
                
                sleep(1)
            
    q_output.put(None)
    return None


def _inference(network, batch, num_uncertainty_draws):
    # Astra does record the time taken by each task, but this is a naive estimate which does not account for cases
    # where inferences are done in batches, which leads to one spectrum in the batch having the estimated time for
    # the whole batch, and all others having some small value (1e-5), so we will calculate the average time here

    t_init = time()
    spectrum_pks, source_pks, flux, e_flux, meta, meta_dict, missing_photometrys = ([], [], [], [], [], [], [])
    for spectrum_pk, source_pk, f, ef, m, md, mp in batch:
        if f is None:
            # OS Error when loading it.
            yield ApogeeNetV2(
                spectrum_pk=spectrum_pk,
                source_pk=source_pk,
                flag_no_result=True
            )
        else:
            spectrum_pks.append(spectrum_pk)
            source_pks.append(source_pk)
            flux.append(f)
            e_flux.append(ef)
            meta.append(m)
            meta_dict.append(md)
            missing_photometrys.append(mp)

    if len(flux) > 0:
        shape = (-1, 1, 8575)
        flux = torch.from_numpy(np.array(flux).reshape(shape)).to(DEVICE)
        e_flux = torch.from_numpy(np.array(e_flux).reshape(shape)).to(DEVICE)
        N, _, P = flux.shape
        meta = np.array(meta).reshape((N, -1))
        meta_torch = torch.from_numpy(meta).to(DEVICE)

        inputs = (
            torch.randn(
                (num_uncertainty_draws, N, 1, P), device=DEVICE
            )
            * e_flux
            + flux
        )
        inputs = inputs.reshape((num_uncertainty_draws * N, 1, P))
        meta_draws = meta_torch.repeat(num_uncertainty_draws, 1).reshape((num_uncertainty_draws * N, -1))
        with torch.set_grad_enabled(False):
            predictions = network.predict_spectra(flux, meta_torch)
            predictions = predictions.cpu().data.numpy()

            draws = network.predict_spectra(inputs, meta_draws)
            draws = draws.cpu().data.numpy()

        # Replace infinites with non-finite.
        predictions[~np.isfinite(predictions)] = np.nan

        draws = draws.reshape((num_uncertainty_draws, N, -1))

        # un-log10-ify the draws before calculating summary statistics
        predictions[:, 1] = 10 ** predictions[:, 1]
        draws[:, :, 1] = 10 ** draws[:, :, 1]

        predictions = predictions.T
        median_draw_predictions = np.nanmedian(draws, axis=0).T
        std_draw_predictions = np.nanstd(draws, axis=0).T

        logg_median, teff_median, fe_h_median = median_draw_predictions
        logg_std, teff_std, fe_h_std = std_draw_predictions
        logg, teff, fe_h = predictions

        mean_t_elapsed = (time() - t_init) / len(spectrum_pks)
        for i, (spectrum_pk, source_pk, missing_photometry) in enumerate(zip(spectrum_pks, source_pks, missing_photometrys)):
            output = ApogeeNetV2(
                spectrum_pk=spectrum_pk,
                source_pk=source_pk,
                teff=teff[i],
                logg=logg[i],
                fe_h=fe_h[i],
                e_teff=teff_std[i],
                e_logg=logg_std[i],
                e_fe_h=fe_h_std[i],
                teff_sample_median=teff_median[i],
                logg_sample_median=logg_median[i],
                fe_h_sample_median=fe_h_median[i],
                t_elapsed=mean_t_elapsed
            )
            output.apply_flags(meta[i], missing_photometry=missing_photometry)
            yield output


def parallel_batch_read(target, spectra, args, batch_size, cpu_count=None):
    N = cpu_count or mp.cpu_count()
    q_output = mp.Queue()
    qps = []
    for i in range(N):
        q = mp.Queue()
        p = mp.Process(target=target, args=(q, q_output, *args))
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



def get_metadata(spectrum):
    """
    Get requisite photometry and astrometry from a given spectrum for APOGEENet.

    :param spectrum:
        An `astra.tools.spectrum.Spectrum1D` spectrum.

    :returns:
        A a-length tuple containing relevant metadata for the given spectrum. The first entry in
        the tuple contains the header keys, the second entry contains the values of the metadata,
        and the last value contains the normalized, clipped values of the metadata, for use with the
        APOGEENet model.
    """
    mdata_replacements = np.array(
        [-84.82700, 21.40844, 24.53892, 20.26276, 18.43900, 24.00000, 17.02500]
    )
    mdata_stddevs = np.array(
        [
            14.572430555504504,
            2.2762944923233883,
            2.8342029214199704,
            2.136884367623457,
            1.6793628207779732,
            1.4888102872755238,
            1.5848713221149886,
        ]
    )
    mdata_means = np.array(
        [
            -0.6959113178296891,
            13.630030428758845,
            14.5224418320574,
            12.832448427460813,
            11.537019017423619,
            10.858717523536697,
            10.702106344460235,
        ]
    )    
    try:
        metadata = [
            spectrum.source.plx,
            spectrum.source.g_mag,
            spectrum.source.bp_mag,
            spectrum.source.rp_mag,
            spectrum.source.j_mag,
            spectrum.source.h_mag,
            spectrum.source.k_mag
        ]
    except:
        missing_photometry = True
        metadata = mdata_means
        metadata_norm = ((metadata - mdata_means) / mdata_stddevs).astype(np.float32)
    
    else:            
        de_nanify = lambda x: x if (x != "NaN" and x != -999999 and x is not None) else np.nan
        
        metadata = np.array(list(map(de_nanify, metadata)))
        missing_photometry = np.any(
            ~np.isfinite(metadata)
        |   (metadata >= 98)
        |   (metadata <= -1)
        )

        metadata = np.where(metadata < 98, metadata, mdata_replacements)
        metadata = np.where(np.isfinite(metadata), metadata, mdata_replacements)
        metadata = np.where(metadata > -1, metadata, mdata_replacements)
        metadata_norm = ((metadata - mdata_means) / mdata_stddevs).astype(np.float32)

    finally:
        return (metadata, metadata_norm, missing_photometry)
    
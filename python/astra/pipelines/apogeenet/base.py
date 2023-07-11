
import torch
import numpy as np
from itertools import cycle
import multiprocessing as mp

from astra.utils import log, flatten
from astra.models.apogeenet import ApogeeNet 

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def _prepare_data(spectrum, large_error):
    N, P = np.atleast_2d(spectrum.flux).shape
    flux = np.nan_to_num(spectrum.flux).astype(np.float32).reshape((N, P))
    e_flux = np.nan_to_num(spectrum.ivar**-0.5).astype(np.float32).reshape((N, P))
    
    meta_dict, metadata_norm = get_metadata(spectrum)
    meta = np.tile(metadata_norm, N).reshape((N, -1))

    N, P = flux.shape
    flux = flux.reshape((N, 1, P))
    e_flux = e_flux.reshape((N, 1, P))
    median_error = 5 * np.median(e_flux, axis=(1, 2))
    for j, value in enumerate(median_error):
        bad_pixel = (e_flux[j] == large_error) | (e_flux[j] >= value)
        e_flux[j][bad_pixel] = value

    return (spectrum.spectrum_id, flux, e_flux, meta, meta_dict)



def _worker(q_input, q_output, large_error):
    while True:
        spectrum = q_input.get()
        if spectrum is None:
            break
        
        try:
            result = _prepare_data(spectrum, large_error)
            # Only put results when the queue is empty, otherwise we might load data faster than we can process it.
            # (which eventually leads to an out-of-memory error)
            while True:
                if q_output.empty():
                    q_output.put(result)
                    break
        except:
            log.exception(f"Exception in worker with data product {spectrum}")
            continue
    
    q_output.put(None)
    return None


def _inference(network, batch, num_uncertainty_draws):
    spectrum_ids, flux, e_flux, meta, meta_dict = ([], [], [], [], [])
    for spectrum_id, f, ef, m, md in batch:
        spectrum_ids.append(spectrum_id)
        flux.append(f)
        e_flux.append(ef)
        meta.append(m)
        meta_dict.append(md)

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

    for i, spectrum_id in enumerate(spectrum_ids):
        output = ApogeeNet(
            spectrum_id=spectrum_id,
            teff=teff[i],
            logg=logg[i],
            fe_h=fe_h[i],
            e_teff=teff_std[i],
            e_logg=logg_std[i],
            e_fe_h=fe_h_std[i],
            teff_sample_median=teff_median[i],
            logg_sample_median=logg_median[i],
            fe_h_sample_median=fe_h_median[i],
        )
        output.apply_flags(meta[i])
        yield output



def parallel_batch_read(target, data_product, args, batch_size, cpu_count=None):
    N = cpu_count or mp.cpu_count()
    q_output = mp.Queue()
    qps = []
    for i in range(N):
        q = mp.Queue()
        p = mp.Process(target=target, args=(q, q_output, *args))
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



def get_metadata(spectrum):
    """
    Get requisite photometry and astrometry from a given spectrum for APOGEENet.

    :param spectrum:
        An `astra.tools.spectrum.Spectrum1D` spectrum.

    :returns:
        A three-length tuple containing relevant metadata for the given spectrum. The first entry in
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
        metadata = mdata_means
        metadata_norm = ((metadata - mdata_means) / mdata_stddevs).astype(np.float32)
    
    else:            
        de_nanify = lambda x: x if (x != "NaN" and x != -999999 and x is not None) else np.nan
        
        metadata = np.array(list(map(de_nanify, metadata)))

        metadata = np.where(metadata < 98, metadata, mdata_replacements)
        metadata = np.where(np.isfinite(metadata), metadata, mdata_replacements)
        metadata = np.where(metadata > -1, metadata, mdata_replacements)
        metadata_norm = ((metadata - mdata_means) / mdata_stddevs).astype(np.float32)

    finally:
        return (metadata, metadata_norm)

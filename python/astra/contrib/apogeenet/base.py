from scipy import optimize  # if you remove this, everything at Utah breaks. seriously.
import gc
import numpy as np
import torch
import multiprocessing as mp
from typing import Optional, Iterable
from functools import cache
from astropy.nddata import StdDevUncertainty
from astropy import units as u
from peewee import FloatField, IntegerField
from itertools import cycle

from astra.base import task
from astra.contrib.apogeenet.model import Model
from astra.contrib.apogeenet.utils import get_metadata, create_bitmask
from astra.database.astradb import DataProduct, SDSSOutput, _get_sdss_metadata
from astra.tools.spectrum import SpectrumList
from astra.tools.spectrum.utils import spectrum_overlaps
from astra.utils import log, expand_path, flatten, list_to_dict

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ApogeeNetOutput(SDSSOutput):
    teff = FloatField()
    logg = FloatField()
    fe_h = FloatField()
    e_teff = FloatField()
    e_logg = FloatField()
    e_fe_h = FloatField()
    teff_sample_median = FloatField()
    logg_sample_median = FloatField()
    fe_h_sample_median = FloatField()
    bitmask_flag = IntegerField()


@cache
def read_model(model_path, device):
    return Model(expand_path(model_path), device)    


def _prepare_data(data_product, large_error):
    for spectrum in SpectrumList.read(data_product.path):
        if not spectrum_overlaps(spectrum, 16_500 * u.Angstrom):
            # Skip non-APOGEE spectra.
            continue

        N, P = np.atleast_2d(spectrum.flux).shape
        flux = np.nan_to_num(spectrum.flux.value).astype(np.float32).reshape((N, P))
        e_flux = np.nan_to_num(
                spectrum.uncertainty.represent_as(StdDevUncertainty).array
            ).astype(np.float32).reshape((N, P))
        
        meta_dict, metadata_norm = get_metadata(spectrum)
        meta = np.tile(metadata_norm, N).reshape((N, -1))

        N, P = flux.shape
        flux = flux.reshape((N, 1, P))
        e_flux = e_flux.reshape((N, 1, P))
        median_error = 5 * np.median(e_flux, axis=(1, 2))
        for j, value in enumerate(median_error):
            bad_pixel = (e_flux[j] == large_error) | (e_flux[j] >= value)
            e_flux[j][bad_pixel] = value

        kwds = _get_sdss_metadata(data_product, spectrum)
        reference = (data_product, kwds)
        yield (reference, flux, e_flux, meta, meta_dict)


def _worker(q_input, q_output, large_error):
    while True:
        data_product = q_input.get()
        if data_product is None:
            break
        
        try:
            for result in _prepare_data(data_product, large_error):
                # Only put results when the queue is empty, otherwise we might load data faster than we can process it.
                # (which eventually leads to an out-of-memory error)
                while True:
                    if q_output.empty():
                        q_output.put(result)
                        break
        except:
            log.exception(f"Exception in worker with data product {data_product}")
            continue
    
    q_output.put(None)
    return None


def _inference(model, batch, num_uncertainty_draws):
    references, flux, e_flux, meta, meta_dict = ([], [], [], [], [])
    for reference, f, ef, m, md in batch:
        references.append(reference)
        flux.append(f)
        e_flux.append(ef)
        meta.append(m)
        meta_dict.append(md)

    shape = (-1, 1, 8575)
    flux = torch.from_numpy(np.array(flux).reshape(shape)).to(DEVICE)
    e_flux = torch.from_numpy(np.array(e_flux).reshape(shape)).to(DEVICE)
    N, _, P = flux.shape
    meta = torch.from_numpy(np.array(meta).reshape((N, -1))).to(DEVICE)

    inputs = (
        torch.randn(
            (num_uncertainty_draws, N, 1, P), device=DEVICE
        )
        * e_flux
        + flux
    )
    inputs = inputs.reshape((num_uncertainty_draws * N, 1, P))
    meta_draws = meta.repeat(num_uncertainty_draws, 1).reshape((num_uncertainty_draws * N, -1))
    with torch.set_grad_enabled(False):
        predictions = model.predict_spectra(flux, meta)
        predictions = predictions.cpu().data.numpy()

        draws = model.predict_spectra(inputs, meta_draws)
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
    bitmask_meta = { k: np.array(v) for k, v in list_to_dict(meta_dict).items() }

    bitmask_flag = create_bitmask(
        predictions,
        bitmask_meta,
        median_draw_predictions=median_draw_predictions,
        std_draw_predictions=std_draw_predictions,
    )

    del draws, predictions

    for i, (dp, kwds) in enumerate(references):
        result = dict(
            data_product=dp,
            teff=teff[i],
            logg=logg[i],
            fe_h=fe_h[i],
            e_teff=teff_std[i],
            e_logg=logg_std[i],
            e_fe_h=fe_h_std[i],
            teff_sample_median=teff_median[i],
            logg_sample_median=logg_median[i],
            fe_h_sample_median=fe_h_median[i],
            bitmask_flag=bitmask_flag[i]
        )
        result.update(kwds)
        yield result
    

def parallel_batch_read(target, data_product, args, batch_size, N=None):
    N = N or mp.cpu_count()
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


@task
def APOGEENet(
    data_product: Iterable[DataProduct],
    model_path: str = "$MWM_ASTRA/component_data/APOGEENet/model.pt",
    large_error: Optional[float] = 1e10,
    num_uncertainty_draws: Optional[int] = 100,
) -> Iterable[ApogeeNetOutput]:

    model = read_model(model_path, DEVICE)
    
    cpu_count, gpu_batch_size = (4, 100)

    for batch in parallel_batch_read(_worker, data_product, (large_error, ), gpu_batch_size, cpu_count):
        for result in _inference(model, batch, num_uncertainty_draws):
            yield ApogeeNetOutput(**result)

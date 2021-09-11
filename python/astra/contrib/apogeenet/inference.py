
import numpy as np
import os        
from tqdm import tqdm

from astra.operators.utils import prepare_data
from astra.database.utils import (create_task_output, deserialize_pks)
from astra.utils import log
from astra.database import astradb

from astra.contrib.apogeenet.model import Model
from astra.contrib.apogeenet.utils import (create_bitmask, get_metadata)

import torch

def estimate_stellar_labels(
        pks,
        default_num_uncertainty_draws=100,
        default_large_error=1e10
    ):
    """
    Estimate the stellar parameters for APOGEE ApStar observations,
    where task instances have been created with the given primary keys (`pks`).

    :param pks:
        The primary keys of task instances that include information about what
        ApStar observation to load.
         
    :param default_num_uncertainty_draws: [optional]
        The number of random draws to make of the flux uncertainties, which will be
        propagated into the estimate of the stellar parameter uncertainties (default: 100).
    
    :param default_large_error: [optional]
        An arbitrarily large error value to assign to bad pixels (default: 1e10).
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    log.info(f"Running APOGEENet on device {device} with:")
    log.info(f"\tpks: {pks}")

    log.debug(f"CUDA_VISIBLE_DEVICES = '{os.environ.get('CUDA_VISIBLE_DEVICES')}'")

    log.debug(f"Using torch version {torch.__version__} in {torch.__path__}")
    
    models = {}

    pks = deserialize_pks(pks, flatten=True)
    total = len(pks)

    log.info(f"There are {total} primary keys to process: {pks}")

    for instance, path, spectrum in tqdm(prepare_data(pks), total=total):
        if spectrum is None: continue

        model_path = instance.parameters["model_path"]

        # Load the model.
        try:
            model = models[model_path]
        except KeyError:
            log.info(f"Loaded model from {model_path}")

            models[model_path] = model = Model(model_path, device)
                
        N, P = spectrum.flux.shape

        # Build metadata array.
        metadata_keys, metadata, metadata_norm = get_metadata(spectrum)

        flux = np.nan_to_num(spectrum.flux.value).astype(np.float32).reshape((N, 1, P))
        meta = np.tile(metadata_norm, N).reshape((N, -1))

        flux = torch.from_numpy(flux).to(device)
        meta = torch.from_numpy(meta).to(device)

        with torch.set_grad_enabled(False):
            predictions = model.predict_spectra(flux, meta)
            if device != "cpu":
                predictions = predictions.cpu().data.numpy()
                
        # Replace infinites with non-finite.
        predictions[~np.isfinite(predictions)] = np.nan

        # Create results array.
        log_g, log_teff, fe_h = predictions.T
        teff = 10**log_teff
        result = dict(
            snr=spectrum.meta["snr"],
            teff=teff.tolist(),
            logg=log_g.tolist(),
            fe_h=fe_h.tolist(),
        )

        num_uncertainty_draws = int(instance.parameters.get("num_uncertainty_draws", default_num_uncertainty_draws))

        if num_uncertainty_draws > 0:
            large_error = float(instance.parameters.get("large_error", default_large_error))

            flux_error = np.nan_to_num(spectrum.uncertainty.array**-0.5).astype(np.float32).reshape((N, 1, P))
            median_error = 5 * np.median(flux_error, axis=(1, 2))
            
            for j, value in enumerate(median_error):
                bad_pixel = (flux_error[j] == large_error) | (flux_error[j] >= value)
                flux_error[j][bad_pixel] = value
            
            flux_error = torch.from_numpy(flux_error).to(device)

            inputs = torch.randn((num_uncertainty_draws, N, 1, P), device=device) * flux_error + flux
            inputs = inputs.reshape((num_uncertainty_draws * N, 1, P))

            meta_error = meta.repeat(num_uncertainty_draws, 1)
            with torch.set_grad_enabled(False):
                draws = model.predict_spectra(inputs, meta_error)
                if device != "cpu":
                    draws = draws.cpu().data.numpy()
                
            draws = draws.reshape((num_uncertainty_draws, N, -1))
            
            # Need to put the log(teffs) to teffs before calculating std_dev
            draws[:, :, 1] = 10**draws[:, :, 1]

            median_draw_predictions = np.nanmedian(draws, axis=0)
            std_draw_predictions = np.nanstd(draws, axis=0)

            log_g_median, teff_median, fe_h_median = median_draw_predictions.T
            log_g_std, teff_std, fe_h_std = std_draw_predictions.T

            result.update(
                _teff_median=teff_median.tolist(),
                _logg_median=log_g_median.tolist(),
                _fe_h_median=fe_h_median.tolist(),
                u_teff=teff_std.tolist(),
                u_logg=log_g_std.tolist(),
                u_fe_h=fe_h_std.tolist()
            )

        else:
            median_draw_predictions, std_draw_predictions = (None, None)

        # Add the bitmask flag.
        bitmask_flag = create_bitmask(
            predictions,
            median_draw_predictions=median_draw_predictions,
            std_draw_predictions=std_draw_predictions
        )

        result.update(bitmask_flag=bitmask_flag.tolist())

        # Write the result to database.        
        create_task_output(instance, astradb.ApogeeNet, **result)
        
    log.info(f"Completed processing of {total} primary keys")

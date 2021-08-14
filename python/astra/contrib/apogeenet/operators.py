
import numpy as np
import os        
from tqdm import tqdm

from astra.new_operators import (ApStarOperator, _yield_data)
from astra.database.utils import (create_task_output, deserialize_pks, serialize_pks_to_path)
from astra.utils import log
from astra.database import astradb

from astra.contrib.apogeenet.model import Model
from astra.contrib.apogeenet.utils import (create_bitmask, get_metadata)

import torch

class ApogeeNetOperator(ApStarOperator):

    def __init__(
        self,
        *,
        model_path: str,
        analyze_individual_visits=True,
        num_uncertainty_draws=100,
        large_error=1e10,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model_path = model_path
        self.num_uncertainty_draws = num_uncertainty_draws
        self.analyze_individual_visits = analyze_individual_visits
        self.large_error = large_error


    def execute(self, context):

        if self.slurm_kwargs:
            # Write the primary keys to a path that is accessible by all nodes.
            pks_path = serialize_pks_to_path(self.pks)

            all_visits_str = "--all-visits" if self.analyze_individual_visits else ""
            bash_command = f"astra run apogeenet {all_visits_str} {self.model_path} {pks_path}"
            
            self.execute_by_slurm(
                context,
                bash_command,
            )

        else:
            # Just run it in Python.
            estimate_stellar_labels(
                self.pks,
                model_path=self.model_path,
                analyze_individual_visits=self.analyze_individual_visits,
                num_uncertainty_draws=self.num_uncertainty_draws,
                large_error=self.large_error
            )
        return None
    

def estimate_stellar_labels(
        pks,
        model_path,
        analyze_individual_visits=True,
        num_uncertainty_draws=100,
        large_error=1e10
    ):
    """
    Estimate the stellar parameters for APOGEE ApStar observations,
    where task instances have been created with the given primary keys (`pks`).

    :param pks:
        The primary keys of task instances that include information about what
        ApStar observation to load.
    
    :param model_path:
        The disk path of the pre-trained model.
    
    :param analyze_individual_visits: [optional]
        Analyze individual visits stored in the ApStar object. If `False` then it
        will only analyze the stacked (zero-th index) observation (default: `True`).
    
    :param num_uncertainty_draws: [optional]
        The number of random draws to make of the flux uncertainties, which will be
        propagated into the estimate of the stellar parameter uncertainties (default: 100).
    
    :param large_error: [optional]
        An arbitrarily large error value to assign to bad pixels (default: 1e10).
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    log.info(f"Running APOGEENet on device {device} with:")
    log.info(f"\tmodel_path: {model_path}")
    log.info(f"\tpks: {pks}")
    log.info(f"\tanalyze_individual_visits: {analyze_individual_visits}")

    log.debug(f"CUDA_VISIBLE_DEVICES = '{os.environ.get('CUDA_VISIBLE_DEVICES')}'")

    # Load the model.
    model = Model(model_path, device)

    log.info(f"Loaded model from {model_path}")

    pks = deserialize_pks(pks, flatten=True)
    total = len(pks)

    for pk, instance, path, spectrum in tqdm(_yield_data(pks), total=total):
        if spectrum is None: continue
        
        N, P = spectrum.flux.shape

        # Build metadata array.
        metadata_keys, metadata, metadata_norm = get_metadata(spectrum)

        flux = np.nan_to_num(spectrum.flux.value).astype(np.float32).reshape((N, 1, P))
        meta = np.tile(metadata_norm, N).reshape((N, -1))

        flux = torch.from_numpy(flux).to(device)
        meta = torch.from_numpy(meta).to(device)

        with torch.set_grad_enabled(False):
            predictions = model.predict_spectra(flux, meta).detach().numpy()

        # Create results array.
        log_g, log_teff, fe_h = predictions.T

        result = dict(
            snr=spectrum.meta["snr"],
            teff=10**log_teff,
            logg=log_g,
            fe_h=fe_h,
        )
        
        if num_uncertainty_draws > 0:
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
            draws = draws.detach().numpy().reshape((num_uncertainty_draws, N, -1))

            median_draw_predictions = np.median(draws, axis=0)
            std_draw_predictions = np.std(draws, axis=0)

            log_g_median, log_teff_median, fe_h_median = median_draw_predictions.T
            log_g_std, log_teff_std, fe_h_std = std_draw_predictions.T

            result.update(
                _teff_median=10**log_teff_median,
                _logg_median=log_g_median,
                _fe_h_median=fe_h_median,
                u_teff=10**log_teff_std,
                u_logg=log_g_std,
                u_fe_h=fe_h_std
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
        
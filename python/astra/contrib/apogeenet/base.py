import numpy as np
import torch
from tqdm import tqdm

from astra import log
from astra.utils import dict_to_list, logarithmic_tqdm
from astra.base import ExecutableTask, Parameter
from astra.database.astradb import database, Output, TaskOutput, ApogeeNetOutput
from astra.tools.spectrum import Spectrum1D

from astra.contrib.apogeenet.model import Model
from astra.contrib.apogeenet.utils import get_metadata, create_bitmask
from astropy.io import fits


class StellarParameters(ExecutableTask):

    model_path = Parameter(bundled=True)
    num_uncertainty_draws = Parameter(default=100)
    large_error = Parameter(default=1e10)

    def execute(self):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        log.info(f"Executing on {device}")

        model = Model(self.model_path, device)

        all_results = []
        # Since this could be a long process, we update the progress bar only ~100 times on ~pseudo
        # uniform steps in log iterations so that we don't fill the log file with crap.
        with logarithmic_tqdm(total=len(self.context["tasks"]), miniters=100) as pb:
            for i, (task, data_products, parameters) in enumerate(self.iterable()):

                with fits.open(data_products[0].path) as image:
                    N, P = image[1].data.shape
                    keys, metadata, metadata_norm = get_metadata(
                        headers=image[0].header
                    )

                    snr = [image[0].header["SNR"]]
                    n_visits = image[0].header["NVISITS"]
                    if n_visits > 1:
                        snr.append(snr[0])
                        snr.extend(
                            [
                                image[0].header[f"SNRVIS{i}"]
                                for i in range(1, 1 + n_visits)
                            ]
                        )

                    flux = (
                        np.nan_to_num(image[1].data)
                        .astype(np.float32)
                        .reshape((N, 1, P))
                    )
                    error = (
                        np.nan_to_num(image[2].data)
                        .astype(np.float32)
                        .reshape((N, 1, P))
                    )
                    meta = np.tile(metadata_norm, N).reshape((N, -1))

                    median_error = 5 * np.median(error, axis=(1, 2))
                    for j, value in enumerate(median_error):
                        bad_pixel = (error[j] == parameters["large_error"]) | (
                            error[j] >= value
                        )
                        error[j][bad_pixel] = value

                flux = torch.from_numpy(flux).to(device)
                error = torch.from_numpy(error).to(device)
                meta = torch.from_numpy(meta).to(device)

                with torch.set_grad_enabled(False):
                    predictions = model.predict_spectra(flux, meta)
                    if device != "cpu":
                        predictions = predictions.cpu().data.numpy()

                # Replace infinites with non-finite.
                predictions[~np.isfinite(predictions)] = np.nan

                inputs = (
                    torch.randn(
                        (parameters["num_uncertainty_draws"], N, 1, P), device=device
                    )
                    * error
                    + flux
                )
                inputs = inputs.reshape((parameters["num_uncertainty_draws"] * N, 1, P))

                meta_error = meta.repeat(parameters["num_uncertainty_draws"], 1)
                with torch.set_grad_enabled(False):
                    draws = model.predict_spectra(inputs, meta_error)
                    if device != "cpu":
                        draws = draws.cpu().data.numpy()

                draws = draws.reshape((parameters["num_uncertainty_draws"], N, -1))

                raise a

                median_draw_predictions = np.nanmedian(draws, axis=0)
                std_draw_predictions = np.nanstd(draws, axis=0)

                log_g_median, teff_median, fe_h_median = median_draw_predictions.T
                log_g_std, teff_std, fe_h_std = std_draw_predictions.T

                log_g, log_teff, fe_h = predictions.T

                bitmask_flag = create_bitmask(
                    predictions,
                    median_draw_predictions=median_draw_predictions,
                    std_draw_predictions=std_draw_predictions,
                )

                result = {
                    "snr": snr,
                    "teff": 10**log_teff,
                    "logg": log_g,
                    "fe_h": fe_h,
                    "u_teff": teff_std,
                    "u_logg": log_g_std,
                    "u_fe_h": fe_h_std,
                    "teff_sample_median": teff_median,
                    "logg_sample_median": log_g_median,
                    "fe_h_sample_median": fe_h_median,
                    "bitmask_flag": bitmask_flag,
                }

                results = dict_to_list(result)

                # Create rows in the database.
                with database.atomic():
                    for result in results:
                        output = Output.create()
                        TaskOutput.create(task=task, output=output)
                        ApogeeNetOutput.create(task=task, output=output, **result)

                pb.update()
                all_results.append(results)

        return results

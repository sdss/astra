import numpy as np
import torch
from tqdm import tqdm

from astra import log
from astra.utils import dict_to_list
from astra.base import ExecutableTask, Parameter
from astra.database.astradb import Output, TaskOutput
from astra.tools.spectrum import Spectrum1D

from astra.contrib.apogeenet.model import Model
from astra.contrib.apogeenet.database import ApogeeNet
from astra.contrib.apogeenet.utils import get_metadata, create_bitmask


class StellarParameters(ExecutableTask):

    model_path = Parameter(bundled=True)
    num_uncertainty_draws = Parameter(default=100)
    large_error = Parameter(default=1e10)

    def execute(self):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = Model(self.model_path, device)

        results = []
        total = len(self.input_data_products)
        for i, (task, data_products, parameters) in enumerate(
            tqdm(self.iterable(), total=total)
        ):

            assert len(data_products) == 1

            spectrum = Spectrum1D.read(data_products[0].path)
            keys, metadata, metadata_norm = get_metadata(spectrum)

            N, P = spectrum.flux.shape
            flux = (
                np.nan_to_num(spectrum.flux.value).astype(np.float32).reshape((N, 1, P))
            )
            meta = np.tile(metadata_norm, N).reshape((N, -1))

            flux = torch.from_numpy(flux).to(device)
            meta = torch.from_numpy(meta).to(device)

            with torch.set_grad_enabled(False):
                predictions = model.predict_spectra(flux, meta)
                if device != "cpu":
                    predictions = predictions.cpu().data.numpy()

            # Replace infinites with non-finite.
            predictions[~np.isfinite(predictions)] = np.nan

            flux_error = (
                np.nan_to_num(spectrum.uncertainty.array**-0.5)
                .astype(np.float32)
                .reshape((N, 1, P))
            )
            median_error = 5 * np.median(flux_error, axis=(1, 2))

            for j, value in enumerate(median_error):
                bad_pixel = (flux_error[j] == self.large_error) | (
                    flux_error[j] >= value
                )
                flux_error[j][bad_pixel] = value

            flux_error = torch.from_numpy(flux_error).to(device)

            inputs = (
                torch.randn((self.num_uncertainty_draws, N, 1, P), device=device)
                * flux_error
                + flux
            )
            inputs = inputs.reshape((self.num_uncertainty_draws * N, 1, P))

            meta_error = meta.repeat(self.num_uncertainty_draws, 1)
            with torch.set_grad_enabled(False):
                draws = model.predict_spectra(inputs, meta_error)
                if device != "cpu":
                    draws = draws.cpu().data.numpy()

            draws = draws.reshape((self.num_uncertainty_draws, N, -1))

            # Re-scale the temperature draws before calculating statistics.
            raise a

            median_draw_predictions = np.nanmedian(draws, axis=0)
            std_draw_predictions = np.nanstd(draws, axis=0)

            log_g_median, teff_median, fe_h_median = median_draw_predictions.T
            log_g_std, teff_std, fe_h_std = std_draw_predictions.T

            log_g, teff, fe_h = predictions.T

            bitmask_flag = create_bitmask(
                predictions,
                median_draw_predictions=median_draw_predictions,
                std_draw_predictions=std_draw_predictions,
            )

            result = {
                "snr": spectrum.meta["snr"],
                "teff": teff,
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

            results.append(dict_to_list(result))

        return results

    def post_execute(self):

        if not ApogeeNet.table_exists():
            log.info(f"Creating database table for ApogeeNet")
            ApogeeNet.create_table()

        # Create rows in the database.
        total = sum(map(len, self.result))
        with tqdm(total=total) as pb:
            for (task, data_products, parameters), results in zip(
                self.iterable(), self.result
            ):
                for result in results:
                    output = Output.create()
                    TaskOutput.create(task=task, output=output)
                    ApogeeNet.create(task=task, output=output, **result)
                    pb.update()

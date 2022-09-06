import numpy as np
import torch
from astra import log
from astra.utils import dict_to_list, logarithmic_tqdm
from astra.base import ExecutableTask, Parameter
from astra.database.astradb import database, ApogeeNetOutput
from astra.tools.spectrum import SpectrumList
from astra.contrib.apogeenet.model import Model
from astra.contrib.apogeenet.utils import get_metadata, create_bitmask
from astropy.nddata import StdDevUncertainty
from astropy import units as u


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

                # Input data products could be apStar, mwmVisit, or mwmStar files.
                # If they are mwmVisit/mwmStar files then those could include BOSS spectra.
                # Since we don't know yet, just load as SpectrumList.
                flux, e_flux, meta, snrs = ([], [], [], [])
                for spectrum in SpectrumList.read(data_products[0].path):
                    if spectrum.wavelength[0] < (15_000 * u.Angstrom):
                        continue

                    N, P = np.atleast_2d(spectrum.flux).shape
                    flux.append(np.nan_to_num(spectrum.flux.value).astype(np.float32))
                    e_flux.append(
                        np.nan_to_num(
                            spectrum.uncertainty.represent_as(StdDevUncertainty).array
                        ).astype(np.float32)
                    )
                    meta_dict, metadata_norm = get_metadata(spectrum)
                    meta.append(np.tile(metadata_norm, N).reshape((N, -1)))
                    snrs.append(spectrum.meta["SNR"])

                if len(flux) == 0:
                    log.warning(
                        f"No infrared spectra found in {data_products[0]}: {data_products[0].path} -- skipping!"
                    )
                    continue

                flux, e_flux, meta, snrs = [
                    np.vstack(ea) for ea in (flux, e_flux, meta, snrs)
                ]
                median_error = 5 * np.median(e_flux, axis=1)
                for j, value in enumerate(median_error):
                    bad_pixel = (e_flux[j] == parameters["large_error"]) | (
                        e_flux[j] >= value
                    )
                    e_flux[j][bad_pixel] = value

                N, P = flux.shape
                flux = flux.reshape((N, 1, P))
                e_flux = e_flux.reshape((N, 1, P))

                flux = torch.from_numpy(flux).to(device)
                e_flux = torch.from_numpy(e_flux).to(device)
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
                    * e_flux
                    + flux
                )
                inputs = inputs.reshape((parameters["num_uncertainty_draws"] * N, 1, P))

                meta_draws = meta.repeat(parameters["num_uncertainty_draws"], 1)
                with torch.set_grad_enabled(False):
                    draws = model.predict_spectra(inputs, meta_draws)
                    if device != "cpu":
                        draws = draws.cpu().data.numpy()

                draws = draws.reshape((parameters["num_uncertainty_draws"], N, -1))

                # un-log10-ify the draws before calculating summary statistics
                predictions[:, 1] = 10 ** predictions[:, 1]
                draws[:, :, 1] = 10 ** draws[:, :, 1]

                median_draw_predictions = np.nanmedian(draws, axis=0)
                std_draw_predictions = np.nanstd(draws, axis=0)

                logg_median, teff_median, fe_h_median = median_draw_predictions.T
                logg_std, teff_std, fe_h_std = std_draw_predictions.T

                logg, teff, fe_h = predictions.T

                bitmask_flag = create_bitmask(
                    predictions,
                    meta_dict,
                    median_draw_predictions=median_draw_predictions,
                    std_draw_predictions=std_draw_predictions,
                )

                result = {
                    "snr": snrs.flatten(),
                    "teff": teff,
                    "logg": logg,
                    "fe_h": fe_h,
                    "u_teff": teff_std,
                    "u_logg": logg_std,
                    "u_fe_h": fe_h_std,
                    "teff_sample_median": teff_median,
                    "logg_sample_median": logg_median,
                    "fe_h_sample_median": fe_h_median,
                    "bitmask_flag": bitmask_flag,
                }

                results = dict_to_list(result)

                # Create or update rows.
                with database.atomic():
                    task.create_or_update_outputs(ApogeeNetOutput, results)

                pb.update()
                all_results.append(results)

        return results

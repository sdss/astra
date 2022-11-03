from scipy import optimize  # if you remove this, everything at Utah breaks. seriously.
import numpy as np
import torch
from astra import log
from astra.utils import flatten, expand_path, dict_to_list, logarithmic_tqdm
from astra.base import TaskInstance, Parameter
from astra.database.astradb import database, ApogeeNetOutput
from astropy.nddata import StdDevUncertainty
from astropy import units as u

from astra.tools.spectrum import SpectrumList
from astra.tools.spectrum.utils import spectrum_overlaps
from astra.contrib.apogeenet.model import Model
from astra.contrib.apogeenet.utils import get_metadata, create_bitmask


class StellarParameters(TaskInstance):

    """
    Estimate stellar parameters for APOGEE spectra given a pre-trained neural network.

    :param model_path:
        A model path.

    """

    model_path = Parameter(
        default="$MWM_ASTRA/component_data/APOGEENet/model.pt", bundled=True
    )
    num_uncertainty_draws = Parameter(default=100)
    large_error = Parameter(default=1e10)

    data_slice = Parameter(default=[0, 1])  # only affects apStar data products

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def execute(self):
        """Execute the task."""

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        log.info(f"Executing on {device}")

        model = Model(expand_path(self.model_path), device)

        all_results = []
        # Since this could be a long process, we update the progress bar only ~100 times on ~pseudo
        # uniform steps in log iterations so that we don't fill the log file with crap.
        with logarithmic_tqdm(total=len(self.context["tasks"]), miniters=100) as pb:
            for i, (task, data_products, parameters) in enumerate(self.iterable()):

                # Input data products could be apStar, mwmVisit, or mwmStar files.
                # If they are mwmVisit/mwmStar files then those could include BOSS spectra.
                # Since we don't know yet, just load as SpectrumList.
                results = dict(snr=[], source_id=[], parent_data_product_id=[])
                flux, e_flux, meta = ([], [], [])
                for spectrum in SpectrumList.read(
                    data_products[0].path, data_slice=parameters.get("data_slice", None)
                ):
                    # Check for APOGEE spectral range
                    if not spectrum_overlaps(spectrum, 16_500 * u.Angstrom):
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
                    results["snr"].extend(spectrum.meta["SNR"])
                    results["source_id"].extend([spectrum.meta.get("CAT_ID", None)] * N)
                    parent_data_product_ids = spectrum.meta.get("DATA_PRODUCT_ID", None)
                    if parent_data_product_ids is None or len(parent_data_product_ids) == 0:
                        parent_data_product_ids = [data_products[0].id] * N
                    results["parent_data_product_id"].extend(parent_data_product_ids)

                assert len(results["snr"]) == len(results["source_id"])
                assert len(results["snr"]) == len(results["parent_data_product_id"])

                if len(flux) == 0:
                    log.warning(
                        f"No infrared spectra found in {data_products[0]}: {data_products[0].path} -- skipping!"
                    )
                    continue

                flux, e_flux, meta = [
                    np.vstack(ea) for ea in (flux, e_flux, meta)
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

                if N == 1:
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


                else:
                    predictions = []
                    with torch.set_grad_enabled(False):
                        for j in range(N):
                            prediction = model.predict_spectra(
                                torch.from_numpy(flux[[j]]).to(device),
                                torch.from_numpy(meta[[j]]).to(device),
                            )
                            if device != "cpu":
                                prediction = prediction.cpu().data.numpy()
                            predictions.append(prediction)
                    predictions = np.array(predictions).reshape((N, -1))
                    predictions[~np.isfinite(predictions)] = np.nan

                    draws = []
                    with torch.set_grad_enabled(False):
                        for j in range(N):
                            inputs = (
                                torch.randn(
                                    (parameters["num_uncertainty_draws"], 1, P), device=device
                                )
                                * torch.from_numpy(e_flux[[j]]).to(device)
                                + torch.from_numpy(flux[[j]]).to(device)
                            )
                            meta_draws = torch.from_numpy(meta[[j]]).to(device).repeat(
                                parameters["num_uncertainty_draws"], 1
                            )
                            draw = model.predict_spectra(inputs, meta_draws)
                            if device != "cpu":
                                draw = draw.cpu().data.numpy()
                            draws.append(draw)

                    draws = np.array(draws).reshape((parameters["num_uncertainty_draws"], N, -1))

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

                results.update({
                    "teff": teff,
                    "logg": logg,
                    "fe_h": fe_h,
                    "e_teff": teff_std,
                    "e_logg": logg_std,
                    "e_fe_h": fe_h_std,
                    "teff_sample_median": teff_median,
                    "logg_sample_median": logg_median,
                    "fe_h_sample_median": fe_h_median,
                    "bitmask_flag": bitmask_flag,
                })

                results_list = dict_to_list(results)

                # Create or update rows.
                with database.atomic():
                    task.create_or_update_outputs(ApogeeNetOutput, results_list)

                pb.update()
                all_results.append(results_list)

        return all_results
